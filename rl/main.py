import argparse
import json
import numpy as np
import os
import random
import redis
import tensorflow as tf

from bisect import bisect
from collections import defaultdict
from jack import readers
from random import randint

from ir.search_engine import SearchEngine
from playground.datareader import format_paths
from qa.nouns import pre_extract_nouns, SpacyNounParser, NltkNounParser
from qa.question import Question
from rc.utils import get_rc_answers, get_cached_rc_answers
from rl.agent import Agent, RandomAgent, Reinforce
from rl.embeddings import GloveLookup
from shared.utils import trim_index, form_query, get_document_for_query


def format_run_id(args):
    if args.run_id is None:
        return None
    if args.random_agent:
        run_id = ('random-'
                  + '-r' + '-'.join(str(r) for r in [args.default_r, args.found_candidate_r,
                                                     args.penalty, args.success_r])
                  + '-max{}-s{}'.format(args.max_queries, args.seed))
    else:
        run_id = ('-'.join(['l{}'.format(layer_size) for layer_size in args.h_sizes])
                  + '-g{}-lr{}-uf{}'.format(args.gamma, args.lr, args.update_freq)
                  + '-r' + '-'.join(str(r) for r in [args.default_r, args.found_candidate_r,
                                                     args.penalty, args.success_r])
                  + '-max{}-s{}'.format(args.max_queries, args.seed))
    if args.run_id != '':
        run_id += '-' + args.run_id
    return run_id


def format_experiment_paths(query_type, run_id, dirname, dev=False, save_checkpoints=True):
    checkpoint_path = None
    if dirname is not None:
        dirname += '/'
    else:
        dirname = ''
    if save_checkpoints:
        checkpoint_path = 'rl/checkpoints/{}{}/{}/model-'.format(dirname, query_type, run_id)
    if dev:
        summaries_path = 'rl/summaries/{}{}/dev/{}'.format(dirname, query_type, run_id)
        eval_path = 'rl/eval/{}{}/dev/{}.txt'.format(dirname, query_type, run_id)
    else:
        summaries_path = 'rl/summaries/{}{}/train/{}'.format(dirname, query_type, run_id)
        eval_path = 'rl/eval/{}{}/train/{}.txt'.format(dirname, query_type, run_id)
    return summaries_path, checkpoint_path, eval_path


def actions_for_query_type(query_type):
    if query_type == 'located_in_the_administrative_territorial_entity':
        actions = [['where is'],
                   ['municipality of'],
                   ['located'],
                   ['what kind of socks does', 'wear']]
    elif query_type == 'occupation':
        actions = ['employer of',
                   'colleague of',
                   'known for',
                   'work as']
    return actions


def discount_rewards(r, gamma):
    """Discount history of rewards by gamma."""
    discounted_r = np.zeros(len(r))
    running_add = 0
    # Most recent reward is last
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def check_answer(answer, question, incorrect_answers_this_episode, e, corrects, incorrects,
                 is_last_action, default_r, found_candidate_r, penalty, success_r, verbosity_level):
    """Check agent's answer against candidates and true answer and return a reward accordingly."""
    reward = default_r
    if answer.text.lower() in question.candidates_lower:
        if answer.text.lower() == question.answer.lower():
            verbose_print(1, verbosity_level, e, ': Found correct answer', answer.text)
            corrects.append(e)
            reward = success_r
        # If the current answer is not correct and we have not submitted it before
        elif answer.text.lower() not in incorrect_answers_this_episode:
            reward = found_candidate_r
            if is_last_action:
                reward = penalty
            verbose_print(1, verbosity_level, e, ': Found incorrect answer candidate', answer.text)
            incorrect_answers_this_episode.append(answer.text.lower())
            if e not in incorrects:
                incorrects.append(e)
        elif is_last_action:
            reward = penalty

    return reward, incorrect_answers_this_episode, corrects, incorrects


def accuracy_from_history(corrects, ep, horizon=100):
    if horizon is None:
        return float(len(corrects))/(ep + 1)
    first_in_horizon = bisect(corrects, ep - horizon)
    # Number of episodes passed
    ep += 1
    accuracy = float(len(corrects[first_in_horizon:])) / min(horizon, ep)
    if ep <= horizon:
        if not np.isclose(accuracy, float(len(corrects)) / ep):
            print(accuracy, float(len(corrects)) / ep, len(corrects), ep)
        assert np.isclose(accuracy, float(len(corrects)) / ep)
    return accuracy


def scalar_summaries(running_reward, ep_length, corrects, ep):
    accuracy_horizon = 100
    accuracy = accuracy_from_history(corrects, ep, accuracy_horizon)
    return {'reward': running_reward,
            'episode_length': ep_length,
            'accuracy ({})'.format(accuracy_horizon): accuracy}


def write_summary(summary_writer, episode, simple_values, summary_objects=None):
    scalar_summary = tf.Summary()
    for k, v in simple_values.items():
        scalar_summary.value.add(simple_value=v, tag=k)
    summary_writer.add_summary(scalar_summary, episode)
    if summary_objects is not None:
        for s in summary_objects:
            summary_writer.add_summary(s, episode)
    summary_writer.flush()


def write_eval_file(eval_path, corrects, incorrects, num_episodes):
    if not os.path.exists(os.path.dirname(eval_path)):
        os.makedirs(os.path.dirname(eval_path))
    with open(eval_path, 'w') as f:
        f.write('Correct {}/{} = {}'.format(len(corrects), num_episodes,
                                            float(len(corrects))/num_episodes))
        f.write('\nIncorrect {}/{} = {}'.format(len(incorrects), num_episodes,
                                                float(len(incorrects))/num_episodes))


def print_model_weights(sess, args):
    if args.verbose_weights and not args.random_agent:
        all_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
        for var in all_vars:
            print(var, '\n', sess.run(var))


def verbose_print(verbosity, verbose_level, *args):
    if verbose_level >= verbosity:
        for arg in args:
            print(arg, end=' ')
        print()


def new_or_existing_session(existing_sess=None):
    if existing_sess is not None:
        return existing_sess
    return tf.Session()


def run_agent(dataset, search_engine, nouns, reader, redis_server, embs, args,
              agent_from_checkpoint=None, agent=None, dev=False, eval_dataset=None,
              eval_search_engine=None, eval_nouns=None, existing_session=None,
              total_accuracy_only=False, outer_e=None):
    """Shared train/eval routine for REINFORCE agents, or eval for a random agent.

    Use cases
    1) To train an agent:
        Leave all optional arguments in their default settings.

    2) To train an agent and regularly evaluate its performance on another dataset:
        As 1), but also set eval_dataset, eval_search_engine and eval_nouns

    3) To evaluate an existing agent on dev data:
         As 1), but also set agent_from_checkpoint, and dev=True

    Args:
        dataset: Data to iterate over.
        search_engine: Search engine used to interact with dataset.
        nouns: Pre-extracted nouns used to interact with dataset.
        reader: Reading comprehension model used to interact with dataset.
        redis_server: Caching server used for faster reading comprehension answers.

        embs: Word embeddings used to represent states.
        args: Command line arguments.

        agent_from_checkpoint: Trained weights to restore into agent's network.
        agent: Optionally pass in an existing agent to avoid creating duplicate nodes in the graph.

        dev: If True, run once through whole dataset and report dev performance.

        eval_dataset: Data used to report intermediate dev performance every <eval_freq> steps.
        eval_search_engine, eval_nouns: Used to interact with eval_dataset in intermediate 
            performance evaluation.
        existing_session: If None, open a new session. Else, use the existing session.

    Returns:
        embs: Initialised embeddings are returned for reuse.
    """
    actions = actions_for_query_type(args.qtype)
    train = agent_from_checkpoint is None and not args.random_agent and not total_accuracy_only

    # Threshold above which to trust the reading comprehension module's answers
    confidence_threshold = args.conf_threshold
    max_queries = args.max_queries

    if dev or args.num_items_to_eval is None:
        # Always evaluate full set when dev=True
        num_episodes = len(dataset)
    else:
        # Repeat if num_episodes > len(dataset)
        num_episodes = args.num_items_to_eval

    # Only used when train=True
    gamma = args.gamma
    lr = args.lr
    update_frequency = args.update_freq

    # Only used when random_agent=False
    h_sizes = args.h_sizes

    # Only used when eval_dataset is not None, must be > checkpoint_freq
    eval_freq = 4000

    checkpoint_freq = 500
    emb_dim = 50

    if args.run_id is not None:
        run_id = format_run_id(args)
        summaries_path, checkpoint_path, eval_path = format_experiment_paths(
            args.qtype, run_id, args.dirname, dev, save_checkpoints=train)
        for path in [summaries_path, checkpoint_path]:
            if path is not None and not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

    # TODO: should this not be run for existing agent?
    if existing_session is None:
        tf.reset_default_graph()
        tf.set_random_seed(args.seed)

    # State: subj0,  a_t-1,     subj_t-1, d_t-1,   ans_t-1 + history
    #        emb_dim, emb_dim, emb_dim,  emb_dim, emb_dim + TODO
    s_size = 5 * emb_dim
    # TODO: add backtracking action
    a_size = len(actions)

    if agent is None:
        if args.random_agent:
            agent = RandomAgent(state_shape=s_size, action_shape=a_size)
        else:
            agent = Reinforce(lr=lr, state_shape=s_size, action_shape=a_size, hidden_sizes=h_sizes)
    if not args.random_agent:
        saver = tf.train.Saver()

    corrects, incorrects, incorrect_answers_this_episode = [], [], []

    if existing_session is None:
        sess = tf.Session()
    else:
        sess = existing_session

    sess.run(tf.global_variables_initializer())
    if args.run_id is not None:
        summary_writer = tf.summary.FileWriter(summaries_path, sess.graph)
        if not args.random_agent:  # no variables to save/retrieve for random agent
            # saver = tf.train.Saver()
            if agent_from_checkpoint is not None:
                print('Loading saved agent from', agent_from_checkpoint)
                saver.restore(sess, agent_from_checkpoint)
            else:
                print('Initialised agent weights')
    print_model_weights(sess, args)

    reward_history = []
    ep_length_history = []

    if train:
        gradBuffer = sess.run(tf.trainable_variables())
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

    for e in range(num_episodes):
        question = Question(dataset[e % len(dataset)])
        verbose_print(2, args.verbose, e, ':', question.query)
        _, subj0 = question.query.split()[0], ' '.join(question.query.split()[1:]).lower()
        # First action taken with partial information: no a_t-1, subj_t-1, d_t-1, ans_t-1
        query0 = form_query(actions[0], subj0)
        # First query won't have been asked from top doc
        top_idx = search_engine.rank_docs(question.id, query0, topk=len(question.supports))[-1]
        d0 = question.supports[top_idx]

        verbose_print(2, args.verbose, '   ', query0, '\t->', top_idx)
        # Send query to RC module
        if redis_server is None:
            rc_answers = get_rc_answers(reader, query0, d0)
        else:
            rc_answers, _ = get_cached_rc_answers(reader, query0, d0, redis_server)
        ans0 = rc_answers[0]
        # Initial state: pick action 0 automatically
        s0 = [subj0, ' '.join(actions[0]), subj0, d0, ans0.text]
        s0 = np.hstack(embs.embed_state(s0))
        s_prev = s0
        # Should first answer be checked?
        '''
        (r, correct_answers, incorrect_answers, incorrect_answers_this_episode,
         seen_incorrect_answer_this_episode) = (
             check_answer(ans0, question, correct_answers, incorrect_answers,
                          incorrect_answers_this_episode, seen_incorrect_answer_this_episode,
                          e))
        '''
        ep_reward = 0
        subj_t = subj0

        # Store past queries asked -> documents retrieved mappings
        queries_asked = defaultdict(list)
        for t in range(max_queries):
            top_idx = None
            while top_idx is None:
                # Pick action according to policy
                a_distr = sess.run(agent.output, feed_dict={agent.state_in: [s_prev]})
                a_t = np.random.choice(range(len(actions)), p=a_distr[0])
                top_idx, subj_t, query_t = get_document_for_query(
                    actions[a_t], subj_t, search_engine, question, nouns, queries_asked)
            queries_asked[query_t].append(top_idx)
            d_t = question.supports[top_idx]

            verbose_print(2, args.verbose, '   ', query_t, '\t->', top_idx)
            # Send query to RC module
            if redis_server is None:
                rc_answers = get_rc_answers(reader, query_t, d_t)
            else:
                rc_answers, _ = get_cached_rc_answers(reader, query_t, d_t, redis_server)
            ans_t = rc_answers[0]

            (r, incorrect_answers_this_episode, corrects, incorrects) = (
                check_answer(ans_t, question, incorrect_answers_this_episode, e, corrects,
                             incorrects, t == max_queries - 1, args.default_r,
                             args.found_candidate_r, args.penalty, args.success_r, args.verbose))
            # TODO: alternative: form_query(actions[a_t], subj_t)
            s_t = [subj0, ' '.join(actions[a_t]), subj_t, d_t, ans_t.text]
            s_t = np.hstack(embs.embed_state(s_t))

            history_frame = np.expand_dims([s_prev, a_t, r, s_t], axis=0)
            if t == 0:
                ep_history = history_frame
            else:
                ep_history = np.append(ep_history, history_frame, axis=0)
            s_prev = s_t
            if ans_t.score > confidence_threshold:
                subj_t = ans_t.text.lower()
            else:
                # Pick a noun phrase at random from top document
                rand_draw = randint(0, len(nouns[question.id][top_idx])-1)
                subj_t = nouns[question.id][top_idx][rand_draw].lower()
            ep_reward += r
            ep_length = t
            if r == 1:
                # TODO: collect aggregate action history data
                verbose_print(1, args.verbose, '\tAction history:', ep_history[:, 1])
                break

        if train:
            ep_history[:, 2] = discount_rewards(ep_history[:, 2], gamma)
            feed_dict = {agent.reward_holder: ep_history[:, 2],
                         agent.action_holder: ep_history[:, 1],
                         agent.state_in: np.vstack(ep_history[:, 0])}
            grads = sess.run(agent.gradients, feed_dict=feed_dict)
            for idx, grad in enumerate(grads):
                gradBuffer[idx] += grad

            if e % update_frequency == 0 and e != 0:
                feed_dict = dict(zip(agent.gradient_holders, gradBuffer))
                _ = sess.run(agent.update_batch, feed_dict=feed_dict)
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

        if args.run_id is not None and not total_accuracy_only and not args.random_agent:
            summary_objects = None
            scalars = scalar_summaries(ep_reward, ep_length, corrects, e)
            if e % len(dataset) == 0:  # Add action preferences for 1st item in dataset
                summary_objects = sess.run(
                    [tf.summary.histogram('action_preferences_1st_question_item', agent.output,
                                          collections=range(len(actions))),
                     tf.summary.tensor_summary('action_preferences_1st_question_item',
                                               agent.output),
                     tf.summary.histogram('action_history_1st_question_item',
                                          ep_history[:, 1],
                                          collections=range(max_queries))]
                    + [tf.summary.histogram('hidden_{}'.format(l), agent.hidden[l])
                       for l in range(len(h_sizes))],
                    feed_dict={agent.state_in: [s0]})

                scalars['episode_reward (1st question item)'] = ep_reward

            write_summary(summary_writer, e, scalars, summary_objects)
        reward_history.append(ep_reward)
        ep_length_history.append(len(ep_history))

        if e % 10 == 0:
            verbose_print(1, args.verbose, '  Agent output',
                  sess.run(agent.output, feed_dict={agent.state_in: [s_prev]}))
            verbose_print(1, args.verbose, '  Reward history', np.mean(reward_history[-100:]))
            verbose_print(1, args.verbose,
                          '  Correct answers', len(corrects), float(len(corrects))/(e+1))
            verbose_print(1, args.verbose,
                          '  Incorrect answers', len(incorrects), float(len(incorrects))/(e+1))

        # Save to checkpoint and evaluate accuracy on dev set
        if eval_dataset is not None and (e + 1) % eval_freq == 0:
            checkpoint_to_load = None  # Use None as checkpoint when evaluating random agent
            if not args.random_agent:
                checkpoint_to_write = checkpoint_path + '{}.ckpt'.format(e)
                print(e, ': Saving model to', checkpoint_to_write)
                saver.save(sess, checkpoint_to_write)

                checkpoint_to_load = checkpoint_to_write
            print(e, ': Running intermediate evaluation step')
            run_agent(eval_dataset, eval_search_engine, eval_nouns, reader, redis_server, embs,
                      args, agent=agent, agent_from_checkpoint=checkpoint_to_load, dev=True,
                      existing_session=sess, total_accuracy_only=True, outer_e=e)

        # Save to checkpoint without eval
        elif (e + 1) % checkpoint_freq == 0 and train and args.run_id is not None:
            checkpoint_to_write = checkpoint_path + '{}.ckpt'.format(e)
            print(e, ': Saving model to', checkpoint_to_write)
            saver.save(sess, checkpoint_to_write)

    # Calculate total dev accuracy
    if args.run_id is not None and total_accuracy_only:
        write_summary(summary_writer, outer_e,
                      {'total_dev_accuracy':
                           accuracy_from_history(corrects, e, horizon=None)})

    # Save final model
    if train and args.run_id is not None:
        saver.save(sess, checkpoint_path + 'final.ckpt')
        print('Saving final model to', checkpoint_path + 'final.ckpt')
        print_model_weights(sess, args)

    if existing_session is None:
        sess.close()

    # TODO: save activations of hidden layers
    if args.save_embs:
        embs.save_history_to_csv()
    if dev:
        print('\nDev set accuracy:')
    else:
        print('\nTrain set accuracy:')
    print('Correct answers final', len(corrects), '/', num_episodes,
          float(len(corrects))/num_episodes)
    print('Incorrect answers final', len(incorrects), '/', num_episodes,
          float(len(incorrects))/num_episodes, '\n')
    if args.run_id is not None:
        write_eval_file(eval_path, corrects, incorrects, num_episodes)
    return embs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spacy', nargs='?', const=True, default=False, type=bool,
                        help='If True, use Spacy to parse nouns. If False, use NLTK (default).')
    parser.add_argument('--verbose', default=0, type=int,
                        help='If True, print out all mentions of the query subject.')
    parser.add_argument('--verbose_weights', nargs='?', const=True, default=False, type=bool,
                        help='If True, print out model weights when saving or loading.')
    parser.add_argument('--subset_size', default=None, type=int,
                        help='If set, evaluate the baseline on a subset of data.')
    parser.add_argument('--num_items_to_eval', default=None, type=int,
                        help='If set, the number of instances to evaluate. If > subset_size, '
                             'iterate over data more than once.')
    parser.add_argument('--k_most_common_only', type=int, default=None,
                        help='If set, only include the k most commonly occurring relation types.')
    parser.add_argument('--wikihop_version', type=str, default='1.1',
                        help='WikiHop version to use: one of {0, 1.1}.')
    parser.add_argument('--cache', dest='cache', action='store_true')
    parser.add_argument('--nocache', dest='cache', action='store_false')
    parser.add_argument('--redis_host', type=str, default='localhost',
                        help='Host of running redis instance (e.g. localhost, cannon).')
    parser.add_argument('--trim', dest='trim_index', action='store_true')
    parser.add_argument('--notrim', dest='trim_index', action='store_false')
    parser.add_argument('--conf_threshold', default=0.10, type=float,
                        help='Confidence threshold required to use ')
    parser.add_argument('--qtype', type=str,
                        default='located_in_the_administrative_territorial_entity',
                        help='WikiHop question type to include. Defines action space of agent.')

    parser.add_argument('--hidden_sizes', dest='h_sizes', nargs='+', type=int, default=[32],
                        help='List denoting the sizes of hidden layers of the network.')
    parser.add_argument('--gamma', default=0.99, type=float,
                        help='Discount factor for rewards.')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning Rate')
    parser.add_argument('--update_freq', default=50, type=int,
                        help='The number of time steps between training steps.')
    parser.add_argument('--max_queries', default=25, type=int,
                        help='Maximum number of queries to allow per episode.')

    parser.add_argument('--default_r', default=0., type=float,
                        help='Reward for not finding the correct answer or an answer candidate '
                             'at a non-terminal time step.')
    parser.add_argument('--found_candidate_r', default=0., type=float,
                        help='Reward for not finding an (incorrect) answer candidate at a '
                             'non-terminal time step.')
    parser.add_argument('--penalty', default=-1, type=float,
                        help='Reward for ending an episode without finding the correct answer.')
    parser.add_argument('--success_r', default=1, type=float,
                        help='Reward for finding the correct answer.')

    parser.add_argument('--log_dir', type=str, default='./rl/summaries',
                        help='Directory for TensorBoard summaries.')
    parser.add_argument('--save_embs', nargs='?', const=True, default=False, type=bool,
                        help='If True, save state embedding vectors to CSV.')
    parser.add_argument('--dirname', type=str, default=None,
                        help='Directory name identifier for current (series of) experiment(s).')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Identifier for TensorBoard summary files.')
    parser.add_argument('--random_agent', action='store_true', default=False,
                        help='If True, randomly pick out of available actions.')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed (for policy).')

    parser.add_argument('--model_from_checkpoint', type=str, default=None,
                        help='Load model from TF checkpoint instead of training.')
    parser.add_argument('--checkpoint_episode', type=str, default='final',
                        help='Specific checkpoint number to load model from. By default, '
                             'the final checkpoint of an experiment.')

    parser.add_argument('--noeval', dest='eval', action='store_false')

    parser.set_defaults(cache=True, trim_index=True, eval=True)
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def initialise(args, dev=False):

    subset_id, data_path, index_filename, nouns_path = format_paths(args, dev)
    if dev:
        print('\nInitialising dev data...')
    else:
        print('\nInitialising train data...')
    with open(data_path) as dataset_file:
        dataset = json.load(dataset_file)
    if args.subset_size is not None:
        dataset = dataset[:args.subset_size]
    search_engine = SearchEngine(dataset, load_from_path=index_filename)

    if args.spacy:
        print('Extracting Spacy nouns...')
        noun_parser_class = SpacyNounParser
    else:
        print('Extracting NTLK nouns...')
        noun_parser_class = NltkNounParser
    # Load noun phrases from a local file (for speedup) if it exists, or create a new one if not
    nouns = pre_extract_nouns(dataset, nouns_path, noun_parser_class=noun_parser_class)

    # Number of instances to test (if > subset size, repeat items)
    num_items_to_eval = len(dataset)
    if args.num_items_to_eval is not None and not dev:
        num_items_to_eval = min(len(dataset), args.num_items_to_eval)

    one_type_only = True  # Set to True to evaluate templates on a single relation type
    one_type_dataset = []
    if one_type_only:
        for q in dataset:
            if q['query'].split()[0] == args.qtype:
                one_type_dataset.append(q)

        dataset = one_type_dataset

    dataset = dataset[:num_items_to_eval]
    print('Loaded', len(dataset), 'questions')

    if args.trim_index:
        nouns, search_engine = trim_index(dataset, nouns, search_engine)
    if dev:
        return dataset, search_engine, nouns

    print('Initialising reader...')
    reader = readers.reader_from_file('./rc/fastqa_reader')
    redis_server = None
    if args.cache:
        redis_server = redis.StrictRedis(host=args.redis_host, port=6379, db=0)
        try:
            redis_server.client_list()
        except redis.exceptions.ConnectionError:
            # Fall back to computing on-the-fly
            print('No redis instance found at {}:6379, continuing without caching.'.format(
                  args.redis_host))
            args.cache = False
            redis_server = None

    return dataset, search_engine, nouns, reader, redis_server


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)
    dataset, search_engine, nouns, reader, redis_server = initialise(args)

    eval_dataset, eval_search_engine, eval_nouns = None, None, None
    if args.eval:
        # Evaluate on dev data
        eval_dataset, eval_search_engine, eval_nouns = initialise(args, dev=True)

    emb_dim = 50
    # Initialise with train set
    embs = GloveLookup('./data/GloVe/glove.6B.%id.txt' % emb_dim, emb_dim, dataset)

    if args.model_from_checkpoint is None:
        # Train agent
        run_agent(dataset, search_engine, nouns, reader, redis_server, embs, args,
                  eval_dataset=eval_dataset, eval_search_engine=eval_search_engine,
                  eval_nouns=eval_nouns)

    run_id = format_run_id(args)
    _, checkpoint_path, _ = format_experiment_paths(args.qtype, run_id, args.dirname)
    if args.model_from_checkpoint:
        # Replace default formatted run_id with custom argument
        checkpoint_path = checkpoint_path.replace(checkpoint_path.split('/')[-2],
                                                  args.model_from_checkpoint)
    if checkpoint_path is not None:
        # Use custom checkpoint ('final' by default)
        checkpoint_path += args.checkpoint_episode + '.ckpt'

    if args.eval:
        # Run final evaluation
        run_agent(eval_dataset, eval_search_engine, eval_nouns, reader, redis_server,
                  embs, args, agent_from_checkpoint=checkpoint_path, dev=True)

