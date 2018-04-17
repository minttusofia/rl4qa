import argparse
import json
import numpy as np
import os
import random
import redis
import sys
import tensorflow as tf

from collections import defaultdict
from jack import readers

from ir.search_engine import SearchEngine
from playground.datareader import format_paths
from qa.nouns import pre_extract_nouns, SpacyNounParser, NltkNounParser
from qa.question import Question
from rl.actions import action_space_for_id
from rc.utils import get_rc_answers, get_cached_rc_answers
from rl.agent import RandomAgent, Reinforce
from rl.embeddings import GloveLookup
from rl.utils import accuracy_from_history, format_experiment_paths, format_run_id, scalar_summaries
from rl.utils import write_eval_file, write_summary
from shared.utils import check_answer_confidence, get_document_for_query, form_query, trim_index
from shared.utils import verbose_print


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
            verbose_print(1, verbosity_level, '   Found correct answer', answer.text)
            corrects.append(e)
            reward = success_r
        # If the current answer is not correct and we have not submitted it before
        elif answer.text.lower() not in incorrect_answers_this_episode:
            reward = found_candidate_r
            if is_last_action:
                reward = penalty
            verbose_print(2, verbosity_level, '   Found incorrect answer candidate', answer.text)
            incorrect_answers_this_episode.append(answer.text.lower())
            if e not in incorrects:
                incorrects.append(e)
    elif is_last_action:
        reward = penalty

    return reward, incorrect_answers_this_episode, corrects, incorrects


def embed_state(args, embs, subj0=None, a_t=None, subj_prev=None, d_t=None, subj_t=None, qtype=None,
                initial_state=False, actions=None, qtype_order=None):
    """Construct state according to in_state flags and embed with GloVe."""
    s = []
    if args.one_hot_states:
        # Only actions and query types have scalable one-hot encodings
        if args.a_t_in_state:
            one_hot_a_t = np.zeros(len(actions))
            one_hot_a_t[a_t] = 1.
            s = np.concatenate([s, one_hot_a_t], axis=0)
        if args.qtype_in_state:
            one_hot_qtype = np.zeros(len(qtype_order))
            one_hot_qtype[qtype_order[qtype]] = 1.
            s = np.concatenate([s, one_hot_qtype], axis=0)
        if args.verbose_embs:
            print(actions[a_t], qtype)
            print(s)
        return s

    if args.subj0_in_state:
        s.append(subj0)

    prev_elems = [actions[a_t] if a_t is not None else None, subj_prev, d_t]
    prev_elems_included = [args.a_t_in_state, args.subj_prev_in_state, args.d_t_in_state]
    for elem in range(len(prev_elems)):
        if prev_elems_included[elem]:
            if type(prev_elems[elem]) == list:
                prev_elems[elem] = ' '.join(prev_elems[elem])
            s.append(None if initial_state else prev_elems[elem])

    if args.subj_t_in_state:
        s.append(subj0 if initial_state else subj_t)
    if args.qtype_in_state:
        s.append(qtype.replace('_', ' '))
    if args.verbose_embs:
        print(s)
    return embs.embed_state(s, verbose=args.verbose_embs)


def verbose_print_model_weights(sess, args):
    if args.verbose_weights and not args.random_agent:
        all_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
        for var in all_vars:
            print(var, '\n', sess.run(var))


def run_agent(dataset, search_engine, nouns, reader, redis_server, embs, args,
              agent_from_checkpoint=None, agent=None, dev=False, eval_dataset=None,
              eval_search_engine=None, eval_nouns=None, existing_session=None,
              total_accuracy_only=False, outer_e=None, summary_writer=None):
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

        total_accuracy_only: If True, write summaries only about total accuracy on the dataset.
        outer_e: If set, use as timestamp (episode number) for summaries instead of starting at 0.
        summary_writer: If set, use existing summary_writer (to append to existing summary file)
            instead of creating a new one.
    """
    action_space_id = args.actions
    if args.actions is None:
        action_space_id = args.qtype
    actions = action_space_for_id(action_space_id)
    if args.backtrack:
        actions.append('BACKTRACK')

    train = agent_from_checkpoint is None and not args.random_agent and not total_accuracy_only
    final_eval = dev and not total_accuracy_only

    max_queries = args.max_queries

    if dev or args.num_items_train is None:
        # Always evaluate full set when dev=True
        num_episodes = len(dataset)
    else:
        # Repeat if num_episodes > len(dataset)
        num_episodes = args.num_items_train

    # Only used when eval_dataset is not None
    eval_freq = 4000
    full_eval_freq = 50000
    checkpoint_freq = 500
    emb_dim = 50
    conf_threshold = None
    # Ensure consistent one-hot indexing of WikiHop relation types
    qtype_order = json.load(open('rl/qtype_ordering.json')) if args.one_hot_states else None

    if args.run_id is not None:
        run_id = format_run_id(args)
        summaries_path, checkpoint_path, eval_path = format_experiment_paths(
            args.qtype, args.actions, args.backtrack, run_id, args.dirname, dev,
            save_checkpoints=train)
        for path in [summaries_path, checkpoint_path]:
            if path is not None and not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

    # TODO: should this not be run for existing agent?
    if existing_session is None:
        tf.reset_default_graph()
        tf.set_random_seed(args.seed)

    if train:
        # Only used when baseline is not None
        baseline_r = tf.Variable(0., trainable=False)
        new_reward = tf.placeholder(tf.float32, [])
        current_e = tf.placeholder(tf.float32, [])
        update_baseline = tf.assign(baseline_r,
                                    (new_reward + (current_e - 1) * baseline_r)/current_e)

    # State: subj0,  a_t-1,     subj_t-1, d_t-1,   ans_t-1, q_type,  history
    #        emb_dim, emb_dim, emb_dim,  emb_dim, emb_dim,  emb_dim, TODO
    if args.one_hot_states:
        s_size = len(actions) * args.a_t_in_state + len(qtype_order) * args.qtype_in_state
    else:
        s_size = (sum([args.subj0_in_state, args.a_t_in_state, args.subj_prev_in_state,
                       args.d_t_in_state, args.subj_t_in_state, args.qtype_in_state])
                  * emb_dim)
    print('s_size', s_size)
    a_size = len(actions)

    if agent is None:
        if args.random_agent:
            agent = RandomAgent(s_size, a_size)
        else:
            agent = Reinforce(args.lr, s_size, a_size, args.h_sizes, args.entropy_w)
    if not args.random_agent:
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

    corrects, incorrects, incorrect_answers_this_episode = [], [], []

    sess = tf.Session() if existing_session is None else existing_session
    sess.run(tf.global_variables_initializer())
    if args.run_id is not None:
        summary_writer = (tf.summary.FileWriter(summaries_path, sess.graph)
                          if summary_writer is None else summary_writer)
    if not args.random_agent:  # no variables to save/retrieve for random agent
        if agent_from_checkpoint is not None:
            print('Before loading saved weights:')
            verbose_print_model_weights(sess, args)
            print('Loading saved agent from', agent_from_checkpoint)
            saver.restore(sess, agent_from_checkpoint)
        else:
            print('Initialised agent weights')
    verbose_print_model_weights(sess, args)

    reward_history = []
    ep_length_history = []
    num_actions_taken = 0

    if train:
        gradBuffer = sess.run(tf.trainable_variables())
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

    run_type = 'train'
    if dev:
        if total_accuracy_only:
            run_type = 'eval'
        else:
            run_type = 'final eval'
    for e in range(num_episodes):
        verbose = 2 if e % 500 == 0 else args.verbose  # Log in detail every 500 episodes
        question = Question(dataset[e % len(dataset)])
        verbose_print(1, verbose,
                      '{} : {}     {} - {} ({})'.format(e, question.query, question.id, run_type,
                                                        num_episodes))
        q_type, subj0 = question.query.split()[0], ' '.join(question.query.split()[1:]).lower()
        if subj0 == '':  # WH_dev_1559 and WH_dev_5113 have no question subject
            print(question.id, 'has no question subject')
            continue
        # First action taken with partial information: no a_t-1, subj_t-1, d_t-1
        s0 = embed_state(args, embs, subj0=subj0, qtype=q_type, initial_state=True,
                         actions=actions, qtype_order=qtype_order)
        state_history = [s0]  # used for backtracking
        subj_history = [subj0]

        a_t_prev = None
        subj_prev_prev = None
        d_t = None
        s_prev = s0
        subj_prev = subj0
        ep_reward = 0

        # Store past queries asked -> documents retrieved mappings
        queries_asked = defaultdict(list)
        for t in range(max_queries):
            top_idx = None
            query_t = None
            while top_idx is None:
                random_init = train and num_actions_taken < args.num_init_random_steps
                if random_init:
                    a_t = np.random.randint(len(actions))
                    if len(state_history) <= 1 and args.backtrack:  # Remove backtrack
                        a_t = np.random.randint(len(actions) - 1)
                    while args.backtrack and a_t == len(actions) - 1:  # backtrack
                        subj_history.pop(), state_history.pop()
                        # Should state that we backtrack to be aware of the question that was asked?
                        s_prev = state_history[-1]
                        subj_prev = subj_history[-1]
                        verbose_print(2, verbose, '  Backtracking to subject', subj_prev)
                        if len(state_history) > 1:
                            a_t = np.random.randint(len(actions))
                        else:  # Remove backtrack
                            a_t = np.random.randint(len(actions) - 1)

                elif t == 0 and args.first_action is not None:
                    a_t = args.first_action
                else:
                    # Pick action according to policy
                    a_distr = sess.run(agent.output, feed_dict={agent.state_in: [s_prev]})
                    if len(state_history) > 1 or not args.backtrack:
                        a_t = np.random.choice(range(len(actions)), p=a_distr[0])
                    else:
                        # Remove backtrack action and renormalise
                        a_distr = [a_distr[0][:-1]/np.sum(a_distr[0][:-1])]
                        a_t = np.random.choice(range(len(actions) - 1), p=a_distr[0])

                    while args.backtrack and a_t == len(actions) - 1:  # backtrack
                        subj_history.pop(), state_history.pop()
                        # Should state that we backtrack to be aware of the question that was asked?
                        s_prev = state_history[-1]
                        subj_prev = subj_history[-1]
                        verbose_print(2, verbose, '  Backtracking to subject', subj_prev)
                        a_distr = sess.run(agent.output, feed_dict={agent.state_in: [s_prev]})
                        if len(state_history) > 1:
                            a_t = np.random.choice(range(len(actions)), p=a_distr[0])
                        else:  # Remove backtrack and renormalise
                            a_distr = [a_distr[0][:-1]/np.sum(a_distr[0][:-1])]
                            a_t = np.random.choice(range(len(actions) - 1), p=a_distr[0])

                top_idx, subj_prev, query_t = get_document_for_query(
                    actions[a_t], subj_prev, search_engine, question, nouns, queries_asked,
                    verbose)
                if top_idx is None:  # subject has been reset
                    s_prev = embed_state(args, embs, subj0, a_t_prev, subj_prev_prev, d_t,
                                         subj_prev, q_type, actions=actions,
                                         qtype_order=qtype_order)
            verbose_print(3, args.verbose, '   0.4f' % (a_distr[0, a_t]), end='')
            queries_asked[query_t].append(top_idx)
            d_t = question.supports[top_idx]

            if random_init:  # action was selected at random
                verbose_print(2, verbose, '   init ({:2})'.format(a_t),
                              form_query(actions[a_t], subj_prev, 'red'), '  ->', top_idx)
            else:
                verbose_print(2, verbose, '   ({:2})'.format(a_t),
                              form_query(actions[a_t], subj_prev, 'red'), '  ->', top_idx)
            # Send query to RC module
            if redis_server is None:
                rc_answers = get_rc_answers(reader, query_t, d_t)
            else:
                rc_answers, _ = get_cached_rc_answers(reader, query_t, d_t, redis_server,
                                                      question.id, top_idx)
            ans_t = rc_answers[0]
            subj_t = check_answer_confidence(ans_t, conf_threshold, nouns, question.id,
                                             top_idx, verbose)

            # Option: don't check answers with confidence < threshold
            (r, incorrect_answers_this_episode, corrects, incorrects) = (
                check_answer(ans_t, question, incorrect_answers_this_episode, e, corrects,
                             incorrects, t == max_queries - 1, args.default_r,
                             args.found_candidate_r, args.penalty, args.success_r, verbose))

            s_t = embed_state(args, embs, subj0, a_t, subj_prev, d_t, subj_t, q_type,
                              actions=actions, qtype_order=qtype_order)
            state_history.append(s_t)
            subj_history.append(subj_t)
            subj_prev_prev = subj_prev
            a_t_prev = a_t
            subj_prev = subj_t
            num_actions_taken += 1

            history_frame = np.expand_dims([s_prev, a_t, r, s_t], axis=0)
            if t == 0:
                ep_history = history_frame
            else:
                ep_history = np.append(ep_history, history_frame, axis=0)
            s_prev = s_t

            ep_reward += r
            if r != args.default_r:
                verbose_print(2, verbose, 'Received reward', r)
            ep_length = t
            if r == args.success_r:
                # TODO: collect aggregate action history data
                verbose_print(1, verbose, '\tAction history:', ep_history[:, 1])
                break
            elif r == args.penalty:
                verbose_print(2, verbose, '( Correct answer', question.answer, ')')

        if train:
            # TODO: baseline when non-terminal rewards != 0
            if args.baseline == 'mean':
                raw_reward = ep_history[-1, 2]
                baseline_reward = sess.run(baseline_r)
                ep_history[-1, 2] -= baseline_reward
                sess.run(update_baseline, {current_e: e + 1., new_reward: ep_history[-1, 2]})
            verbose_print(3, verbose, 'ep history before discounting', ep_history[:, 2])
            ep_history[:, 2] = discount_rewards(ep_history[:, 2], args.gamma)
            verbose_print(3, verbose, 'ep history after discounting', ep_history[:, 2])
            feed_dict = {agent.reward_holder: ep_history[:, 2],
                         agent.action_holder: ep_history[:, 1],
                         agent.state_in: np.vstack(ep_history[:, 0])}
            grads, pg_loss, ent_loss, output, indexes, responsible_outputs = sess.run(
                [agent.gradients, agent.pg_loss, agent.ent_loss, agent.output, agent.indexes,
                 agent.responsible_outputs],
                feed_dict)
            verbose_print(2, verbose, 'Probs of actions taken', responsible_outputs)
            verbose_print(2, verbose,
                          'PG Loss {}; Ent Loss {}; total {}'.format(pg_loss, ent_loss,
                                                                     pg_loss + ent_loss))
            if args.baseline is not None:
                verbose_print(2, verbose,
                              'Raw {}; B {}; effective {}'.format(raw_reward, baseline_reward,
                                                                  raw_reward - baseline_reward))
            verbose_print(3, args.verbose, 'Outputs', output, '\nindexes', indexes)
            for idx, grad in enumerate(grads):
                gradBuffer[idx] += grad

            if e % args.update_freq == 0 and e != 0:
                verbose_print(2, verbose, e, 'Updating policy')
                feed_dict = dict(zip(agent.gradient_holders, gradBuffer))
                _ = sess.run(agent.update_batch, feed_dict)
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

        if (args.run_id is not None and not total_accuracy_only and not args.random_agent
                and not random_init):
            summary_objects = None
            scalars = scalar_summaries(ep_reward, ep_length, corrects, e, dev=dev)
            if train and e % len(dataset) == 0:  # Add action preferences for 1st item in dataset
                summary_objects = sess.run(
                    # [tf.summary.histogram('action_preferences_1st_question_item', agent.output,
                    #                       collections=range(len(actions))),
                    #  tf.summary.tensor_summary('action_preferences_1st_question_item',
                    #                            agent.output),
                    #  tf.summary.histogram('action_history_1st_question_item',
                    #                       ep_history[:, 1],
                    #                       collections=range(max_queries))] +
                    [tf.summary.histogram('hidden_{}'.format(l), agent.hidden[l])
                     for l in range(len(args.h_sizes))],
                    feed_dict={agent.state_in: [s0]})

                #scalars['episode_reward (1st question item)'] = ep_reward

            write_summary(summary_writer, e, scalars, summary_objects)
        reward_history.append(ep_reward)
        ep_length_history.append(len(ep_history))

        if e % 10 == 0:
            verbose_print(1, verbose, '  Agent output',
                  sess.run(agent.output, feed_dict={agent.state_in: [s_prev]}))
            verbose_print(1, verbose, '  Reward history', np.mean(reward_history[-100:]))
            verbose_print(1, verbose,
                          '  Correct answers', len(corrects), float(len(corrects))/(e+1))
            verbose_print(1, verbose,
                          '  Incorrect answers', len(incorrects), float(len(incorrects))/(e+1))

        # Save to checkpoint and evaluate accuracy on dev set
        if eval_dataset is not None and e % eval_freq == 0 and args.run_id is not None:
            checkpoint_to_load = None  # Use None as checkpoint when evaluating random agent
            if not args.random_agent:
                checkpoint_to_write = checkpoint_path + '{}.ckpt'.format(e)
                print(e, ': Saving model to', checkpoint_to_write)
                saver.save(sess, checkpoint_to_write)

                checkpoint_to_load = checkpoint_to_write
            print(e, ': Running intermediate evaluation step')
            run_agent(eval_dataset[:args.num_items_eval], eval_search_engine, eval_nouns, reader,
                      redis_server, embs, args, agent=agent,
                      agent_from_checkpoint=checkpoint_to_load, dev=True, existing_session=sess,
                      total_accuracy_only=True, outer_e=e, summary_writer=summary_writer)
        # Save to checkpoint and evaluate accuracy on full dev set
        if eval_dataset is not None and e % full_eval_freq == 0 and args.run_id is not None:
            checkpoint_to_load = None  # Use None as checkpoint when evaluating random agent
            if not args.random_agent:
                checkpoint_to_write = checkpoint_path + '{}.ckpt'.format(e)
                print(e, ': Saving model to', checkpoint_to_write)
                saver.save(sess, checkpoint_to_write)

                checkpoint_to_load = checkpoint_to_write
            print(e, ': Running intermediate full evaluation step')
            run_agent(eval_dataset, eval_search_engine, eval_nouns, reader, redis_server, embs,
                      args, agent=agent, agent_from_checkpoint=checkpoint_to_load, dev=True,
                      existing_session=sess, total_accuracy_only=True, outer_e=e,
                      summary_writer=summary_writer)

        # Save to checkpoint without eval
        elif (e + 1) % checkpoint_freq == 0 and train and args.run_id is not None:
            checkpoint_to_write = checkpoint_path + '{}.ckpt'.format(e)
            print(e, ': Saving model to', checkpoint_to_write)
            saver.save(sess, checkpoint_to_write)

        if e % 100 == 0:  # Log running accuracy
            start_bold = '\033[1m'
            end_bold = '\033[0;0m'
            horizon = None
            print(e, ': accuracy ({}):'.format(horizon if horizon is not None else 'all'),
                  start_bold + '%0.1f' % (accuracy_from_history(corrects, e, horizon) * 100)+'%'
                  + end_bold)
        sys.stdout.flush()
        sys.stderr.flush()

    # Calculate total dev accuracy
    if args.run_id is not None and total_accuracy_only:
        write_summary(summary_writer, outer_e,
                      {'interm_dev_accuracy ({})'.format(e + 1):
                           accuracy_from_history(corrects, e, horizon=None)})
    if args.run_id is not None and final_eval:
        write_summary(summary_writer, e,
                      {'final_dev_accuracy ({})'.format(e + 1):
                           accuracy_from_history(corrects, e, horizon=None)})

    # Save final model
    if train and args.run_id is not None:
        saver.save(sess, checkpoint_path + 'final.ckpt')
        print('Saving final model to', checkpoint_path + 'final.ckpt')
        verbose_print_model_weights(sess, args)

    if existing_session is None:
        sess.close()

    # TODO: save activations of hidden layers
    if args.save_embs:
        embs.save_history_to_csv()
    if num_episodes > 0:
        print('\nDev set accuracy:' if dev else '\nTrain set accuracy:')
        print('Correct answers final', len(corrects), '/', num_episodes,
              float(len(corrects))/num_episodes)
        print('Incorrect answers final', len(incorrects), '/', num_episodes,
              float(len(incorrects))/num_episodes, '\n')
        if args.run_id is not None:
            write_eval_file(eval_path, corrects, incorrects, num_episodes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spacy', nargs='?', const=True, default=False, type=bool,
                        help='If True, use Spacy to parse nouns. If False, use NLTK (default).')
    parser.add_argument('--reader', default='fastqa', type=str,
                        help='Reading comprehension model to use. One of [ fastqa | bidaf ].')
    parser.add_argument('--verbose', default=0, type=int,
                        help='If True, print out all mentions of the query subject.')
    parser.add_argument('--verbose_weights', nargs='?', const=True, default=False, type=bool,
                        help='If True, print out model weights when saving or loading.')
    parser.add_argument('--verbose_embs', nargs='?', const=True, default=False, type=bool,
                        help='If True, print out word and state embeddings.')

    parser.add_argument('--subset_size', default=None, type=int,
                        help='If set, evaluate the baseline on a subset of data.')
    parser.add_argument('--num_items_train', default=None, type=int,
                        help='If set, the number of instances to train on. If > subset_size, '
                             'iterate over data more than once.')
    parser.add_argument('--num_items_eval', default=500, type=int,
                        help='If set, the number of dev instances to evaluate at intermediate '
                             'checkpoints. Capped at dev size.')
    parser.add_argument('--num_items_final_eval', default=None, type=int,
                        help='If set, the number of dev instances to evaluate for final '
                             'accuracy. Capped at dev size.')

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
    parser.add_argument('--conf_threshold', default=None, type=float,
                        help='Confidence threshold required to use reading comprehension answer '
                             'in following query.')

    parser.add_argument('--qtype', type=str, default='all',
                        help='WikiHop question type to include. Defines action space of agent.')
    parser.add_argument('--actions', type=str, default='all-30',
                        help='ID of action space to use. If not set, use default for query type.')
    parser.add_argument('--first_action', type=int, default=None,
                        help='If set, start episode by taking action at this index, else use '
                             'policy to select first action.')
    parser.add_argument('--backtrack', nargs='?', type=bool, default=False, const=True,
                        help='If True, add the action of undoing the previous query.')
    parser.add_argument('--entropy_w', default=0.001, type=float,
                        help='Policy entropy regularisation weight.')
    parser.add_argument('--num_init_random_steps', default=0, type=int,
                        help='Number of actions to take uniformly at random before using learned '
                             'policy.')

    parser.add_argument('--no_qtype_in_s', dest='qtype_in_state', action='store_false',
                        help='If True (default), include WikiHop relation type in RL input state.')
    parser.add_argument('--no_subj0_in_s', dest='subj0_in_state', action='store_false',
                        help='If True (default), include WikiHop query subject in RL input state.')
    parser.add_argument('--no_a_t_in_s', dest='a_t_in_state', action='store_false',
                        help='If True (default), include previous action in RL input state.')
    parser.add_argument('--no_subj_prev_in_s', dest='subj_prev_in_state', action='store_false',
                        help='If True (default), include previous subject in RL input state.')
    parser.add_argument('--no_d_t_in_s', dest='d_t_in_state', action='store_false',
                        help='If True (default), include previous document in RL input state.')
    parser.add_argument('--no_subj_t_in_s', dest='subj_t_in_state', action='store_false',
                        help='If True (default), include next subject in RL input state.')
    parser.add_argument('--one_hot_states', nargs='?', const=True, default=False, type=bool,
                        help='If True, use one-hot encoding of state elements instead of GloVe.')

    parser.add_argument('--hidden_sizes', dest='h_sizes', nargs='+', type=int, default=[32],
                        help='List denoting the sizes of hidden layers of the network.')
    parser.add_argument('--gamma', default=0.8, type=float,
                        help='Discount factor for rewards.')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning Rate')
    parser.add_argument('--update_freq', default=20, type=int,
                        help='The number of time steps between training steps.')
    parser.add_argument('--max_queries', default=25, type=int,
                        help='Maximum number of queries to allow per episode.')

    parser.add_argument('--baseline', default=None, type=str,
                        help='Baseline reward to use for policy updates. One of [None|mean].')
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


def clean_missing_subjects(dataset):
    items_to_remove = ['WH_dev_1559', 'WH_dev_5113']  # These have no question subject
    i = 0
    while i < len(dataset):
        if dataset[i]['id'] in items_to_remove:
            del dataset[i]
        else:
            i += 1


def initialise(args, dev=False):
    # If loading trained model from checkpoint, don't initialise train data
    dataset, search_engine, nouns = None, None, None
    if args.model_from_checkpoint is None or dev:  # don't load seach engine or nouns for train data
        print('\nInitialising dev data...' if dev else '\nInitialising train data...')
        subset_id, data_path, index_filename, nouns_path = format_paths(args, dev)
        with open(data_path) as dataset_file:
            dataset = json.load(dataset_file)
        if args.subset_size is not None:
            dataset = dataset[:args.subset_size]

        if dev:
            # WH_dev_1559 and WH_dev_5113 have no question subject
            clean_missing_subjects(dataset)
        # Number of instances to include (if train and > subset size, repeat items)
        num_items = len(dataset)
        if args.num_items_train is not None and not dev:
            num_items = args.num_items_train
        elif dev:
            if args.num_items_eval is None:
                args.num_items_eval = len(dataset)
            if args.num_items_final_eval is None:
                args.num_items_final_eval = len(dataset)
            num_items = max(args.num_items_eval, args.num_items_final_eval)

        filter_by_type = args.qtype != 'all'
        if filter_by_type:
            included_types = args.qtype.split(',')
            filtered_dataset = []
            for q in dataset:
                if q['query'].split()[0] in included_types:
                    filtered_dataset.append(q)
            dataset = filtered_dataset

        dataset = dataset[:num_items]
        print('Loaded', len(dataset), 'questions')

        search_engine = SearchEngine(dataset, load_from_path=index_filename)
        if args.spacy:

            print('Extracting Spacy nouns...')
            noun_parser_class = SpacyNounParser
        else:
            print('Extracting NTLK nouns...')
            noun_parser_class = NltkNounParser
        # Load noun phrases from a local file (for speedup) if it exists, or create a new one if not
        nouns = pre_extract_nouns(dataset, nouns_path, noun_parser_class=noun_parser_class)

        if args.trim_index:
            nouns, search_engine = trim_index(dataset, nouns, search_engine)
        if dev:
            return dataset, search_engine, nouns

    print('Initialising {}...'.format(args.reader))
    sys.stdout.flush()
    sys.stderr.flush()
    reader = readers.reader_from_file('./rc/{}_reader'.format(args.reader))

    redis_server = None
    if args.cache:
        if args.reader == 'fastqa':
            redis_server = redis.StrictRedis(host=args.redis_host, port=6379, db=0)
        elif args.reader == 'bidaf':
            redis_server = redis.StrictRedis(host=args.redis_host, port=6379, db=1)

        try:
            redis_server.client_list()
        except redis.exceptions.ConnectionError:
            # Fall back to computing on-the-fly
            print('No redis instance found at {}:6379, continuing without caching.'.format(
                  args.redis_host))
            args.cache = False
            redis_server = None

    return dataset, search_engine, nouns, reader, redis_server


def main():
    args = parse_args()
    set_random_seed(args.seed)
    dataset, search_engine, nouns, reader, redis_server = initialise(args)

    sys.stdout.flush()
    sys.stderr.flush()

    eval_dataset, eval_search_engine, eval_nouns = None, None, None
    if args.eval:
        # Evaluate on dev data
        eval_dataset, eval_search_engine, eval_nouns = initialise(args, dev=True)

    emb_dim = 50
    # Initialise with train set
    embs = GloveLookup('./data/GloVe/glove.6B.%id.txt' % emb_dim, emb_dim)

    if args.model_from_checkpoint is None:
        # Train agent
        run_agent(dataset, search_engine, nouns, reader, redis_server, embs, args,
                  eval_dataset=eval_dataset, eval_search_engine=eval_search_engine,
                  eval_nouns=eval_nouns)
    sys.stdout.flush()
    sys.stderr.flush()

    if args.eval:
        run_id = format_run_id(args)
        _, checkpoint_path, _ = format_experiment_paths(args.qtype, args.actions, args.backtrack,
                                                        run_id, args.dirname)
        if args.model_from_checkpoint:
            # Replace default formatted run_id with custom argument
            checkpoint_path = checkpoint_path.replace(checkpoint_path.split('/')[-2],
                                                      args.model_from_checkpoint)
        if checkpoint_path is not None:
            # Use custom checkpoint ('final' by default)
            checkpoint_path += args.checkpoint_episode + '.ckpt'
            print('Loading model from checkpoint', checkpoint_path)

        # Make final evaluation repeatable
        set_random_seed(args.seed)
        run_agent(eval_dataset[:args.num_items_final_eval], eval_search_engine, eval_nouns, reader,
                  redis_server, embs, args, agent_from_checkpoint=checkpoint_path, dev=True)
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
