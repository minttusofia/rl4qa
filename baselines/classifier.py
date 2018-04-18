import csv
import json
import numpy as np
import operator

from collections import defaultdict


def majority_qtype(data, actions):
    qtype_to_id = {}
    action_to_id = dict([(str(a), idx) for idx, a in enumerate(actions)])
    for row in data:
        qtype = row[1]
        if qtype not in qtype_to_id:
            qtype_to_id[qtype] = len(qtype_to_id)

    num_qtypes = len(qtype_to_id)
    num_actions = len(actions)
    corrects = np.zeros([num_qtypes, num_actions])
    not_corrects = np.zeros([num_qtypes, num_actions])
    str_actions = [str(a) for a in actions]

    for row in data:
        qtype = row[1]
        action = row[3]
        if action not in str_actions:  # only use current template list
            continue
        found_correct = row[4]
        if int(found_correct) == 1:
            corrects[qtype_to_id[qtype], action_to_id[action]] += 1
        elif int(found_correct) == 0:
            not_corrects[qtype_to_id[qtype], action_to_id[action]] += 1

    no_correct_answers = []
    not_enough_support = []
    top_action = {}
    for qtype in qtype_to_id.keys():
        qtype_id = qtype_to_id[qtype]
        ratios = list(enumerate(
            corrects[qtype_id]/(corrects[qtype_id] + not_corrects[qtype_id])
        ))
        print(qtype)
        sorted_ratios = sorted(ratios, key=operator.itemgetter(1), reverse=True)
        sorted_ratios = [list(ratio) for ratio in sorted_ratios]
        for ratio in sorted_ratios:
            ratio.append(corrects[qtype_id, ratio[0]])
        num_tied = 0
        for ratio in sorted_ratios:
            if ratio[1] == sorted_ratios[0][1]:
                num_tied += 1
            else:
                break
        # Sort ties based on the most support
        sorted_ratios[:num_tied] = sorted(sorted_ratios[:num_tied], key=operator.itemgetter(2),
                                          reverse=True)
        printed = []
        for ratio in sorted_ratios:
            if np.isnan(ratio[1]) or ratio[1] == 0:
                continue
            print(actions[ratio[0]], ratio[1],
                  corrects[qtype_id, ratio[0]],
                  not_corrects[qtype_id, ratio[0]])
            if len(printed) == 0:
                top_action[qtype] = actions[ratio[0]]
            printed.append(ratio)
            if len(printed) == 3:
                break
        if len(printed) == 0:
            no_correct_answers.append(qtype)
        elif sum([corrects[qtype_id, printed[0][0]], not_corrects[qtype_id, printed[0][0]]]) < 3:
            not_enough_support.append(qtype)

    print('No correct answers for', no_correct_answers)
    print('Less than three data points for', no_correct_answers + not_enough_support)

    counts = defaultdict(int)
    for k, v in top_action.items():
        print('{}, {}'.format(k, v))
        counts[str(v)] += 1
    best_actions = sorted(list(counts.items()), key=operator.itemgetter(1), reverse=True)
    print('Best actions:', best_actions)
    best_action = eval(best_actions[0][0])
    for qtype in qtype_to_id:
        if qtype not in top_action:
            top_action[qtype] = best_action

    top_action['default'] = best_action

    return top_action


def main():
    reader = 'fastqa'
    data_size = 300000
    actions_file = 'template_list_30'
    paths = ['baselines/data/v1.1/train_ids__{}_25_None_{}_nltk_{}.csv'.format(data_size, reader,
                                                                               actions_file)]
    rows = []
    for path in paths:
        with open(path, 'r') as f:
            f_reader = csv.reader(f)
            next(f_reader, None)  # header
            rows.extend(list(f_reader))

    print(len(rows), 'rows')
    actions = json.load(open('baselines/{}.json'.format(actions_file)))

    top_action = majority_qtype(rows, actions)
    path = 'baselines/single_templates_{}_{}.json'.format(reader, len(top_action))
    print('Writing to', path)
    json.dump(top_action, open(path, 'w'))


if __name__ == "__main__":
    main()