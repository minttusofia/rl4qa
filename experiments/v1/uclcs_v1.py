#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import argparse
import logging


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])


def to_cmd(c, dirname, run_id):
    command = 'sh ../python3.c -m rl.main --dirname {} --lr {} --gamma {} --update_freq {} ' \
              '{} {} --entropy_w {} --num_init_random_steps {} ' \
              '--default_r {} --found_candidate_r {} --penalty {} --success_r {} ' \
              '--qtype all --actions all-30 --verbose 2 --seed {} ' \
              '--hidden_sizes {} ' \
              '--run_id {} --num_items_train 300000 --num_items_eval 500 --redis_host ' \
              'cannon.cs.ucl.ac.uk' \
              ''.format(dirname, c['lr'], c['gamma'], c['update_freq'],
                        c['baseline'], c['backtrack'],
                        c['entropy_w'], c['num_init_random_steps'],
                        c['default_r'], c['found_candidate_r'], c['penalty'], c['success_r'],
                        c['seed'], c['hidden_sizes'],
                        run_id)
    return command


def to_logfile(c, path, dirname, run_id):
    path = os.path.join(path, dirname)
    outfile = os.path.join(path, "uclcs_v1.{}{}.log".format(summary(c).replace("/", "_"), run_id))
    return outfile


def main(argv):
    def formatter(prog):
        return argparse.HelpFormatter(prog, max_help_position=100, width=200)

    argparser = argparse.ArgumentParser('Generating experiments for the UCL cluster', formatter_class=formatter)
    argparser.add_argument('--debug', '-D', action='store_true', help='Debug flag')
    argparser.add_argument('--path', '-p', action='store', type=str, default=None, help='Path')

    args = argparser.parse_args(argv)

    default_hyperparameters = dict(
        lr=[1e-3],
        gamma=[0.99],
        update_freq=[20],
        default_r=[0.0],
        found_candidate_r=[0.0],
        penalty=[-1],
        success_r=[1]
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

    dirname = '04_05/cl1'
    run_id_base = '1'

    current_experiment = dict()
    current_experiment.update(default_hyperparameters)
    current_experiment.update(hyperparameters_space_6)
    configurations = list(cartesian_product(current_experiment))

    path = './logs/v1/uclcs_v1/'

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
        for seed in range(0,3):
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

    # Sort command lines and remove duplicates
    #sorted_command_lines = sorted(command_lines)
    sorted_command_lines = command_lines
    nb_jobs = len(sorted_command_lines)

    header = """#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -o /dev/null
#$ -e /dev/null
#$ -t 1-{}
#$ -l h_vmem=16G,tmem=16G
#$ -l h_rt=32:00:00

cd /home/malakuij/rl4qa

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
