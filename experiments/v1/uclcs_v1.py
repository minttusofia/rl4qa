#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import logging
import os
import sys

from rl.utils import format_run_id


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])


def to_cmd(c, dirname, run_id):
    command = 'sh ../python3.c -m rl.main --dirname {} --lr {} --gamma {} --update_freq {} ' \
              '{} {} --entropy_w {} --num_init_random_steps {} ' \
              '--default_r {} --found_candidate_r {} --penalty {} --success_r {} ' \
              '--qtype all --actions all-30 --verbose 0 --seed {} ' \
              '--hidden_sizes {} --reader {} ' \
              '--run_id {} --num_items_train {} --num_items_eval 500 --redis_host ' \
              'cannon.cs.ucl.ac.uk' \
              ''.format(dirname, c['lr'], c['gamma'], c['update_freq'],
                        c['baseline'], c['backtrack'],
                        c['entropy_w'], c['num_init_random_steps'],
                        c['default_r'], c['found_candidate_r'], c['penalty'], c['success_r'],
                        c['seed'], ' '.join(c['hidden_sizes']), c['reader'],
                        run_id, c['num_items_train'], c['num_items_eval'])
    return command


def to_logfile(c, path, dirname, run_id, qtype='all', actions='all-30'):
    args = argparse.Namespace()
    args.random_agent = False
    args.num_items_train = c['num_items_train']
    args.num_items_eval = c['num_items_eval']
    args.num_items_final_eval = None

    args.lr = c['lr']
    args.gamma = c['gamma']
    args.update_freq = c['update_freq']
    args.baseline = 'mean' if c['baseline'] == '--baseline=mean' else None
    args.entropy_w = c['entropy_w']
    args.num_init_random_steps = c['num_init_random_steps']

    args.default_r = c['default_r']
    args.found_candidate_r = c['found_candidate_r']
    args.penalty = c['penalty']
    args.success_r = c['success_r']

    args.seed = c['seed']
    args.h_sizes = c['hidden_sizes']
    args.reader = c['reader']
    args.max_queries = 25
    args.run_id = run_id

    if dirname is not None:
        dirname += '/'
    else:
        dirname = ''

    if actions is not None:
        actions = '/' + actions
        if c['backtrack']:
            actions += '_bt'
    else:
        actions = 'bt' if c['backtrack'] else ''

    run_id = format_run_id(args)
    outfile = os.path.join(path, '{}{}{}/{}.log'.format(dirname, qtype, actions, run_id))
    return outfile


def main(argv):

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dirname', default='cl0', type=str,
                           help='Directory name identifier for current set of experiments.')
    argparser.add_argument('--run_id_base', default='0', type=str,
                           help='Experiment identifier for logs, TensorBoard summary files, '
                                'checkpoints and evaluation files.')

    args = argparser.parse_args(argv)

    default_hyperparameters = dict(
        lr=[1e-3],
        gamma=[0.8],
        update_freq=[20],
        default_r=[0.0],
        found_candidate_r=[0.0],
        penalty=[-1.0],
        success_r=[1.0],
        num_items_train=[300000],
        num_items_eval=[500],
        hidden_sizes=[['32']],  # [['32', '32']] for multiple layers
        num_init_random_steps=[0]
    )

    hyperparameters_space_1 = dict(
        lr=[1e-4, 1e-3, 1e-2],
        gamma=[0.8, 0.9, 0.95, 0.99],
        update_freq=[10, 20, 50, 100],
    )

    hyperparameters_space_2 = dict(
        default_r=[-0.1, 0.0],
        found_candidate_r=[0.0, 0.1],
        penalty=[-2, -1],
        success_r=[1, 2],
        gamma=[0.8, 0.9]
    )

    hyperparameters_space_3 = dict(
        baseline=['--baseline=mean', ''],
        gamma=[0.8, 0.9, 0.95, 0.99],
        lr=[1e-4, 1e-3],
        update_freq=[10, 20]
    )

    hyperparameters_space_4 = dict(
        baseline=['--baseline=mean', ''],
        backtrack=['--backtrack', ''],
        num_init_random_steps=[0, 1000],
        entropy_w=[0., 0.001, 0.01],
        gamma=[0.8],
        lr=[1e-4, 1e-3],
        update_freq=[20]
    )

    hyperparameters_space_5 = dict(
        baseline=['--baseline=mean', ''],
        backtrack=[''],
        num_init_random_steps=[0, 1000],
        entropy_w=[0., 0.001, 0.01],
        gamma=[0.8],
    )

    hyperparameters_space_6 = dict(
        baseline=['--baseline=mean'],
        backtrack=['--backtrack'],
        num_init_random_steps=[1000],
        entropy_w=[0.001],
        gamma=[0.5, 0.8],
        hidden_sizes=['32 32 32', '64 64']
    )

    hyperparameters_space_7 = dict(
        baseline=['--baseline=mean', ''],
        backtrack=['--backtrack', ''],
        num_init_random_steps=[0, 10000],
        entropy_w=[0., 0.001],
        reader=['bidaf'],
        gamma=[0.8]
    )

    hyperparameters_space_7 = dict(
        baseline=['--baseline=mean', ''],
        backtrack=['--backtrack', ''],
        num_init_random_steps=[0, 10000],
        entropy_w=[0., 0.001],
        reader=['bidaf', 'fastqa'],
        gamma=[0.5, 0.8],
    )

    hyperparameters_space_8 = dict(
        baseline=['--baseline=mean', ''],
        backtrack=['--backtrack', ''],
        entropy_w=[0., 0.001],
        reader=['bidaf', 'fastqa'],
    )
    

    dirname = args.dirname
    run_id_base = args.run_id_base

    current_experiment = dict()
    current_experiment.update(default_hyperparameters)
    current_experiment.update(hyperparameters_space_8)
    configurations = list(cartesian_product(current_experiment))

    path = './rl/logs/'

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/malakuij/'):
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

    configurations = list(configurations)

    command_lines = []
    job_id = 1
    for c in range(len(configurations)):
        cfg = configurations[c]
        for seed in range(0, 1):
            run_id = '{}-{}-{}-{}'.format(run_id_base, job_id, c + 1, seed)
            cfg['seed'] = seed
            logfile = to_logfile(cfg, path, dirname, run_id)
            if not os.path.exists(os.path.dirname(logfile)):
                os.makedirs(os.path.dirname(logfile))
    
            completed = False
            if os.path.isfile(logfile):
                with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    completed = 'Relation specific results' in content
    
            if not completed:
                command_line = '{} > {} 2>&1'.format(to_cmd(cfg, dirname, run_id), logfile)
                command_lines.append(command_line)
            job_id += 1

    nb_jobs = len(command_lines)

    header = """#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -o /dev/null
#$ -e /dev/null
#$ -t 1-{}
#$ -pe smp 2
#$ -R y
#$ -l h_vmem=10G,tmem=10G
#$ -l h_rt=32:00:00

cd /home/malakuij/rl4qa

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
