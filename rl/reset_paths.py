"""Delete TensorBoard and checkpoint outputs from a past experiment to reuse its dirname & ID."""

import argparse
import shutil
import os
import numpy as np


def clear_paths(summaries_top_dir, checkpoints_top_dir, args):
    # Train summaries and checkpoints
    summaries_dir = summaries_top_dir
    checkpoints_dir = checkpoints_top_dir
    dirnames = [args.dirname, args.qtype, 'train', args.run_id]
    for dir in dirnames:
        summaries_dir = os.path.join(summaries_dir, dir)
        checkpoints_dir = os.path.join(checkpoints_dir, dir)
    try:
        if np.all(['events.out.tfevents.' in f for f in os.listdir(summaries_dir)]):
            shutil.rmtree(summaries_dir)
            print('Deleted train summaries\n\t', summaries_dir)
        else:
            print('Train summary dir contains non-summary files; Not deleted.\n\t', summaries_dir)
            print(os.listdir(summaries_dir))
    except OSError:
        print('Train summary dir doesn\'t exist; No action needed.\n\t', summaries_dir)

    try:
        if np.all([f == 'checkpoint' or '.ckpt' in f for f in os.listdir(checkpoints_dir)]):
            shutil.rmtree(checkpoints_dir)
            print('Deleted checkpoints\n\t', checkpoints_dir)
        else:
            print('Checkpoint dir contains non-checkpoint files; Not deleted.\n\t', checkpoints_dir)
            print(os.listdir(checkpoints_dir))
    except OSError:
        print('Checkpoint dir doesn\'t exist; No action needed.\n\t', checkpoints_dir)

    # Dev summaries
    summaries_dir = summaries_top_dir
    dirnames = [args.dirname, args.qtype, 'dev', args.run_id]
    for dir in dirnames:
        summaries_dir = os.path.join(summaries_dir, dir)
    try:
        if np.all(['events.out.tfevents.' in f for f in os.listdir(summaries_dir)]):
            shutil.rmtree(summaries_dir)
            print('Deleted dev summaries\n\t', summaries_dir)
        else:
            print('Dev summary dir contains non-summary files; Not deleted.\n\t', summaries_dir)
            print(os.listdir(summaries_dir))
    except OSError:
        print('Dev summary dir doesn\'t exist; No action needed.\n\t', summaries_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, default=None, help='Run ID of paths to clear.')
    parser.add_argument('--qtype', type=str,
                        default='located_in_the_administrative_territorial_entity',
                        help='Query type used in experiment to clear.')
    parser.add_argument('--dirname', type=str, default=None,
                        help='Name of directory (or chain of directories under checkpoints/ and '
                             'summaries/ in which to look for files marked by run ID (the dirname'
                             'parameter used in experiment to clear).')
    args = parser.parse_args()

    summaries_top_dir = 'rl/summaries'
    checkpoints_top_dir = 'rl/checkpoints'
    clear_paths(summaries_top_dir, checkpoints_top_dir, args)

