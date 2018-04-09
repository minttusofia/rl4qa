"""Utities used by rl/main for file I/O."""

import os
import numpy as np
import tensorflow as tf

from bisect import bisect


def format_run_id(args):
    if args.run_id is None:
        return None
    dev_sizes = '' if args.num_items_eval is None else '-dev{}'.format(args.num_items_eval)
    dev_sizes += ('' if args.num_items_final_eval is None
                  else '-final{}'.format(args.num_items_final_eval))
    dataset_sizes = ('' if args.num_items_train is None else '-train{}'.format(args.num_items_train)
                                                             + dev_sizes)
    baseline_str = '' if args.baseline is None else '-{}bl'.format(args.baseline)
    if args.random_agent:
        run_id = ('random-'
                  + '-r' + '-'.join(str(r) for r in [args.default_r, args.found_candidate_r,
                                                     args.penalty, args.success_r])
                  + '-{}'.format(args.reader)
                  + '{}'.format(dev_sizes)
                  + '-conf{}'.format(args.conf_threshold)
                  + '-max{}-s{}'.format(args.max_queries, args.seed))
    else:
        run_id = ('-'.join(['l{}'.format(layer_size) for layer_size in args.h_sizes])
                  + '-ew{}-init{}'.format(args.entropy_w, args.num_init_random_steps)
                  + '-g{}-lr{}-uf{}'.format(args.gamma, args.lr, args.update_freq)
                  + '-r' + '-'.join(str(r) for r in [args.default_r, args.found_candidate_r,
                                                     args.penalty, args.success_r])
                  + '{}'.format(baseline_str)
                  + '-{}'.format(args.reader)
                  + '{}'.format(dataset_sizes)
                  + '-conf{}'.format(args.conf_threshold)
                  + '-max{}-s{}'.format(args.max_queries, args.seed))
    if args.run_id != '':
        run_id += '-' + args.run_id
    return run_id


def format_experiment_paths(query_type, actions, backtracking, run_id, dirname, dev=False,
                            save_checkpoints=True):
    checkpoint_path = None
    if dirname is not None:
        dirname += '/'
    else:
        dirname = ''

    if actions is not None:
        actions = '/' + actions
        if backtracking:
            actions += '_bt'
    else:
        actions = 'bt' if backtracking else ''

    if save_checkpoints:
        checkpoint_path = 'rl/checkpoints/{}{}{}/{}/model-'.format(dirname, query_type,
                                                                   actions, run_id)
    # Use same path for train and dev as names of plots written to specify train/dev
    summaries_path = 'rl/summaries/{}{}{}/{}'.format(dirname, query_type, actions, run_id)
    if dev:
        eval_path = 'rl/eval/{}{}{}/dev/{}.txt'.format(dirname, query_type, actions, run_id)
    else:
        eval_path = 'rl/eval/{}{}{}/train/{}.txt'.format(dirname, query_type, actions, run_id)
    return summaries_path, checkpoint_path, eval_path


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


def scalar_summaries(running_reward, ep_length, corrects, ep, dev=False):
    accuracy_horizon = 100
    accuracy = accuracy_from_history(corrects, ep, accuracy_horizon)
    if dev:
        return {'reward, dev': running_reward,
                'episode_length, dev': ep_length,
                'accuracy ({}), dev'.format(accuracy_horizon): accuracy}
    else:
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
        f.write('{}\n'.format(eval_path))
        f.write('Correct {}/{} = {}\n'.format(len(corrects), num_episodes,
                                              float(len(corrects))/num_episodes))
        f.write('Incorrect {}/{} = {}\n'.format(len(incorrects), num_episodes,
                                                float(len(incorrects))/num_episodes))

