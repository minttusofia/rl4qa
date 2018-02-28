import argparse
import json
import numpy as np
import os
import random
import redis
import tensorflow as tf

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
                  + '-max%i' % args.max_queries
                  + '-s%i' % args.seed)

    else:
        run_id = ('-'.join(['l%i' % layer_size for layer_size in args.h_sizes])
                  + '-g' + str(args.gamma)
                  + 'lr' + str(args.lr)
                  + '-r' + '-'.join(str(r) for r in [args.default_r, args.found_candidate_r,
                                                     args.penalty, args.success_r])
                  + '-max%i' % args.max_queries
                  + '-s%i' % args.seed)
    if args.run_id != '':
        run_id += '-' + args.run_id
    return run_id


def format_experiment_paths(query_type, random_agent, run_id, dirname, dev=False):
    checkpoint_path = None
    if dirname is not None:
        dirname += '/'
    else:
        dirname = ''
    if not random_agent:
        checkpoint_path = 'rl/checkpoints/{}{}/{}/model-'.format(dirname, query_type, run_id)
    if dev:
        summaries_path = 'rl/summaries/{}{}/dev/{}'.format(dirname, query_type, run_id)
        eval_path = 'rl/eval/{}{}/dev/{}.txt'.format(dirname, query_type, run_id)
    else:
        summaries_path = 'rl/summaries/{}{}/train/{}'.format(dirname, query_type, run_id)
        eval_path = 'rl/eval/{}{}/train/{}.txt'.format(dirname, query_type, run_id)
        #if train:
        #    checkpoint_path = 'rl/checkpoints/{}{}/{}/model-'.format(dirname, query_type, run_id)
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
                 is_last_action, default_r=0., found_candidate_r=0., penalty=-1, success_r=1.):
    """Check agent's answer against candidates and true answer and return a reward accordingly."""
    reward = default_r
    if answer.text.lower() in question.candidates_lower:
        if answer.text.lower() == question.answer.lower():
            print(e, ': Found correct answer', answer.text)
            corrects.append(e)
            reward = success_r
        # If the current answer is not correct and we have not submitted it before
        elif answer.text.lower() not in incorrect_answers_this_episode:
            reward = found_candidate_r
            if is_last_action:
                reward = penalty
            print(e, ': Found incorrect answer candidate', answer.text)
            incorrect_answers_this_episode.append(answer.text.lower())
            if e not in incorrects:
                incorrects.append(e)
        elif is_last_action:
            reward = penalty

    return reward, incorrect_answers_this_episode, corrects, incorrects


def write_summary(summary_writer, running_reward, episode, episode_length):
    ep_summary = tf.Summary()
    # TODO: add intermediate activations, action values (pass in as dict)
    ep_summary.value.add(simple_value=running_reward, tag='reward')
    ep_summary.value.add(simple_value=episode_length, tag='episode_length')
    summary_writer.add_summary(ep_summary, episode)
    summary_writer.flush()


def write_eval_file(eval_path, corrects, incorrects, num_episodes):
    if not os.path.exists(os.path.dirname(eval_path)):
        os.makedirs(os.path.dirname(eval_path))
    with open(eval_path, 'w') as f:
        f.write('Correct {}/{} = {}'.format(len(corrects), num_episodes,
                                            float(len(corrects))/num_episodes))
        f.write('\nIncorrect {}/{} = {}'.format(len(incorrects), num_episodes,
                                                float(len(incorrects))/num_episodes))


def run_agent(query_type, dataset, search_engine, nouns, reader, redis_server, args,
              agent_from_checkpoint=None, dev=False):
    actions = actions_for_query_type(query_type)
    random.seed(args.seed)
    train = agent_from_checkpoint is None and not args.random_agent

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

    emb_dim = 50
    embs = GloveLookup('./data/GloVe/glove.6B.%id.txt' % emb_dim, emb_dim, dataset)

    if args.run_id is not None:
        run_id = format_run_id(args)  # args.run_id + '-lr' + str(lr)
        summaries_path, checkpoint_path, eval_path = format_experiment_paths(
            query_type, args.random_agent, run_id, args.dirname, dev)
        for path in [summaries_path, checkpoint_path]:
            if path is not None and not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

    # TODO: should this not be run for existing agent?
    tf.reset_default_graph()
    # State: subj0,  a_t-1,     subj_t-1, d_t-1,   ans_t-1 + history
    #        emb_dim, emb_dim, emb_dim,  emb_dim, emb_dim + TODO
    s_size = 5 * emb_dim
    # TODO: add backtracking action
    a_size = len(actions)

    if args.random_agent:
        agent = RandomAgent(state_shape=s_size, action_shape=a_size)
    else:
        agent = Reinforce(lr=lr, state_shape=s_size, action_shape=a_size, hidden_sizes=h_sizes)

    corrects = []
    incorrects = []
    incorrect_answers_this_episode = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if args.run_id is not None:
            summary_writer = tf.summary.FileWriter(summaries_path, sess.graph)
            if not args.random_agent:  # no variables to save/retrieve for random agent
                saver = tf.train.Saver()
                if agent_from_checkpoint is not None:
                    saver.restore(sess, agent_from_checkpoint)
        reward_history = []
        ep_length_history = []

        if train:
            gradBuffer = sess.run(tf.trainable_variables())
            for ix, grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0

        for e in range(num_episodes):
            question = Question(dataset[e % len(dataset)])
            _, subj0 = question.query.split()[0], ' '.join(question.query.split()[1:]).lower()
            # First action taken with partial information: no a_t-1, subj_t-1, d_t-1, ans_t-1
            query0 = form_query(actions[0], subj0)
            # First query won't have been asked from top doc
            top_idx = search_engine.rank_docs(question.id, query0, topk=len(question.supports))[-1]
            d0 = question.supports[top_idx]

            # Send query to RC module
            if redis_server is None:
                rc_answers = get_rc_answers(reader, query0, d0)
            else:
                rc_answers, _ = get_cached_rc_answers(reader, query0, d0, redis_server)
            ans0 = rc_answers[0]
            # Initial state: pick action 0 automatically
            s_prev = [subj0, ' '.join(actions[0]), subj0, d0, ans0.text]
            s_prev = np.hstack(embs.embed_state(s_prev))
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

                # Send query to RC module
                if redis_server is None:
                    rc_answers = get_rc_answers(reader, query_t, d_t)
                else:
                    rc_answers, _ = get_cached_rc_answers(reader, query_t, d_t, redis_server)
                ans_t = rc_answers[0]

                (r, incorrect_answers_this_episode, corrects, incorrects) = (
                    check_answer(ans_t, question, incorrect_answers_this_episode, e, corrects,
                                 incorrects, t == max_queries - 1, args.default_r,
                                 args.found_candidate_r, args.penalty, args.success_r))
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
                    print('\tAction history:', ep_history[:, 1])
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

            if args.run_id is not None:
                write_summary(summary_writer, ep_reward, e, ep_length)
            reward_history.append(ep_reward)
            ep_length_history.append(len(ep_history))

            if e % 10 == 0:
                print('  Agent output',
                      sess.run(agent.output, feed_dict={agent.state_in: [s_prev]}))
                print('  Reward history', np.mean(reward_history[-100:]))
                print('  Correct answers', len(corrects), float(len(corrects))/(e+1))
                print('  Incorrect answers', len(incorrects), float(len(incorrects))/(e+1))
                if train and args.run_id is not None:
                    saver.save(sess, checkpoint_path + str(e) + '.ckpt')

        # Save final model
        if train and args.run_id is not None:
            saver.save(sess, checkpoint_path + 'final' + '.ckpt')
            print('Saved weights to', checkpoint_path + 'final' + '.ckpt')
    # TODO: save activations of hidden layers
    if args.save_embs:
        embs.save_history_to_csv()
    # TODO: regularly evaluate on dev set
    if dev:
        print('Dev set accuracy:')
    else:
        print('Train set accuracy:')
    print('Correct answers final', len(corrects), '/', num_episodes,
          float(len(corrects))/num_episodes)
    print('Incorrect answers final', len(incorrects), '/', num_episodes,
          float(len(incorrects))/num_episodes)
    if args.run_id is not None:
        write_eval_file(eval_path, corrects, incorrects, num_episodes)


def initialise(included_type, dev=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--spacy', nargs='?', const=True, default=False, type=bool,
                        help='If True, use Spacy to parse nouns. If False, use NLTK (default).')
    parser.add_argument('--verbose', nargs='?', const=True, default=False, type=bool,
                        help='If True, print out all mentions of the query subject.')
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
    parser.add_argument('--trim', dest='trim_index', action='store_true')
    parser.add_argument('--notrim', dest='trim_index', action='store_false')
    parser.add_argument('--conf_threshold', default=0.10, type=float,
                        help='Confidence threshold required to use ')

    parser.add_argument('--hidden_sizes', dest='h_sizes', nargs='+', default=[32],
                        help='List denoting the sizes of hidden layers of the network.')
    parser.add_argument('--gamma', default=0.99, type=float,
                        help='Discount factor for rewards.')
    parser.add_argument('--lr', default=0.01, type=float,
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
    args = parser.parse_args()

    subset_id, data_path, index_filename, nouns_path = format_paths(args, dev)

    print('Initialising...')
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
    if args.num_items_to_eval is not None:
        num_items_to_eval = min(len(dataset), args.num_items_to_eval)
    print('Evaluating', num_items_to_eval, 'questions')

    one_type_only = True  # Set to True to evaluate templates on a single relation type
    one_type_dataset = []
    if one_type_only:
        for q in dataset:
            if q['query'].split()[0] == included_type:
                one_type_dataset.append(q)

        dataset = one_type_dataset

    dataset = dataset[:num_items_to_eval]
    if args.trim_index:
        nouns, search_engine = trim_index(dataset, nouns, search_engine)

    if dev:
        return dataset, search_engine, nouns

    reader = readers.reader_from_file('./rc/fastqa_reader')
    redis_server = None
    if args.cache:
        redis_server = redis.StrictRedis(host='localhost', port=6379, db=0)

    return dataset, search_engine, nouns, reader, redis_server, args


if __name__ == "__main__":

    query_type = 'located_in_the_administrative_territorial_entity'  # 'occupation'
    dataset, search_engine, nouns, reader, redis_server, args = initialise(query_type)

    if args.model_from_checkpoint is None:
        # Train agent
        run_agent(query_type, dataset, search_engine, nouns, reader, redis_server, args)

    run_id = format_run_id(args)
    _, checkpoint_path, _ = format_experiment_paths(query_type, args.random_agent, run_id,
                                                    args.dirname)
    if args.model_from_checkpoint:
        checkpoint_path = checkpoint_path.replace(checkpoint_path.split('/')[-2],
                                                  args.model_from_checkpoint)
    if checkpoint_path is not None:
        checkpoint_path += args.checkpoint_episode + '.ckpt'

    if args.eval:
        # Evaluate on dev data
        dataset, search_engine, nouns = initialise(query_type, dev=True)
        run_agent(query_type, dataset, search_engine, nouns, reader, redis_server, args,
                  agent_from_checkpoint=checkpoint_path, dev=True)

