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


def to_cmd(c, path=None):
    command = 'python3 -m rl.agent --dirname cl1 --lr {} --gamma {} --update_freq {}' \
              '--default_r {} --found_candidate_r {} --penalty {} --success_r {}' \
              '--run_id 1 --num_items_to_eval 50000 ' \
              ''.format(c['lr'],
                        c['gamma'],
                        c['update_freq'],
                        c['default_r'],
                        c['found_candidate_r'],
                        c['penalty'],
                        c['success_r'])
    return command


def to_logfile(c, path):
    outfile = "%s/uclcs_v1.%s.log" % (path, summary(c).replace("/", "_"))
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
        update_freq=[50],
        default_r=[0.0],
        found_candidate_r=[0.0],
        penalty=[-1],
        success_r=[1],
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
    )

    current_experiment = dict()
    current_experiment.update(default_hyperparameters)
    current_experiment.update(hyperparameters_space_1)
    configurations = list(cartesian_product(current_experiment))

    path = './logs/v1/uclcs_v1/'

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/malakuij/'):
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

    configurations = list(configurations)

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'Relation specific results' in content

        if not completed:
            command_line = '{} > {} 2>&1'.format(to_cmd(cfg, _path=args.path), logfile)
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)
    nb_jobs = len(sorted_command_lines)

    header = """#!/bin/bash

#$ -cwd
#$ -S /bin/bash
#$ -o /dev/null
#$ -e /dev/null
#$ -t 1-{}
#$ -l h_vmem=8G,tmem=8G
#$ -l h_rt=10:00:00

cd /home/malakuij/rl4qa

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
