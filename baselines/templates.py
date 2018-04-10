"""Establish baselines on how well structured question templates perform on WikiHop."""

import argparse
import collections
import csv
import json
import math
import os
import random
import redis
import time

from jack import readers
from random import randint

from ir.search_engine import SearchEngine
from playground.datareader import format_paths
from qa.nouns import pre_extract_nouns, SpacyNounParser, NltkNounParser
from qa.question import Question
from qa.utils import print_time_taken, phrase_appears_in_doc
from rc.utils import get_rc_answers, get_cached_rc_answers
from shared.utils import trim_index, form_query


def discount_by_answer_length(answer):
    if len(answer.text.split()) > 3:
        # TODO: try more sophisticated discounts
        return 3.0/len(answer.text.split())
    return 1.0


def eval_nouns(dataset, nouns):
    """Calculate the fraction of correct answers and of answer candidates that are parsed as nouns.
    """
    answer_found = 0
    answer_mentioned_and_not_found = 0
    candidate_found = 0
    candidate_mentioned_and_not_found = 0
    total_candidates = 0
    total_answers_mentioned = 0
    total_candidates_mentioned = 0
    for q_i in range(len(dataset)):
        question = Question(dataset[q_i])
        nouns_flat = [noun.lower() for doc in nouns[question.id] for noun in doc]
        for candidate in question.candidates_lower:
            total_candidates += 1
            candidate_mentioned = False
            for doc in question.supports:
                if phrase_appears_in_doc(candidate, doc):
                    candidate_mentioned = True
                    break
            total_candidates_mentioned += candidate_mentioned
            if candidate in nouns_flat:
                candidate_found += 1
                '''if not candidate_mentioned:
                    s = input(str(q_i) + ' ' + candidate + ' found as a noun but not mentioned')
                    print(question.query)
                    for doc in question.supports:
                        print(doc)
                        s = input('next?')'''
            else:
                candidate_mentioned_and_not_found += candidate_mentioned
        answer_mentioned = False
        for doc in question.supports:
            if phrase_appears_in_doc(question.answer.lower(), doc):
                answer_mentioned = True
                break
        total_answers_mentioned += answer_mentioned
        if question.answer.lower() in nouns_flat:
            answer_found += 1
        else:
            answer_mentioned_and_not_found += answer_mentioned

        #    print(question.answer)
    print(len(dataset) - answer_found, 'answers found /', len(dataset))
    print('', float(answer_found)/len(dataset))
    print(answer_mentioned_and_not_found, 'answers mentioned /',
          len(dataset) - answer_found, 'answers not found')
    print('', float(answer_mentioned_and_not_found)/(len(dataset) - answer_found))
    print(candidate_found, 'candidates found /', total_candidates)
    print('', float(candidate_found)/total_candidates)
    print(candidate_mentioned_and_not_found, 'candidates mentioned /',
          total_candidates - candidate_found, 'candidates not found')
    print('', float(candidate_mentioned_and_not_found)/(total_candidates - candidate_found))

    print('Total answers mentioned', total_answers_mentioned, '/', len(dataset))
    print('', float(total_answers_mentioned)/len(dataset))
    print('Total candidates mentioned', total_candidates_mentioned, '/', total_candidates)
    print('', float(total_candidates_mentioned)/total_candidates)


def eval_single_templates(templates, search_engine, dataset, nouns, reader,
                          redis_server, num_items_to_eval, max_num_queries, confidence_threshold,
                          question_marks, csv_path=None, penalize_long_answers=False,
                          verbose=False):
    """Evaluates fixed-structure templates which vary in time by the noun passed in as the object.
    """
    correct_answers = collections.defaultdict(float)
    incorrect_answers = collections.defaultdict(float)
    counts_by_type = collections.defaultdict(float)

    store_results = False
    if csv_path is not None:
        store_results = True
        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path))
        collected_data_header = ['question_id', 'question_type', 'question_subj', 'template',
                                 'found_correct', 'found_incorrect']
        if not os.path.exists(csv_path):
            collected_data = [collected_data_header]
        else:
            collected_data = []

    t = time.time()
    for i in range(num_items_to_eval):
        found_correct = False
        if not verbose and i % 10 == 0:
            if i != 0:
                t = print_time_taken(t)
            print('Evaluating question', i, '/', num_items_to_eval, end='  -  ')
            print_accuracy(correct_answers, i)
        question = Question(dataset[i % len(dataset)])
        query_type, subject = question.query.split()[0], ' '.join(question.query.split()[1:])
        counts_by_type[query_type] += 1.0
        if type(templates) == dict:  # if templates are assigned to query types
            if query_type not in templates:
                print('\nNo template found for', query_type, '\n')
                template = templates['default']
            else:
                template = templates[query_type]
        else:
            rand_draw = randint(0, len(templates)-1)
            template = templates[rand_draw]
        if store_results:
            collected_data_row = [question.id, query_type, subject, template]
        if verbose:
            print('\n' + str(i), ':', question.query, '(', question.answer, ')')

        prev_subj = subject.lower()
        # Store past queries asked -> documents retrieved mappings
        queries_asked = collections.defaultdict(list)
        incorrect_answers_this_episode = []
        seen_incorrect_answer_this_episode = False
        for _ in range(max_num_queries):
            top_doc_found = False
            while not top_doc_found:
                query = form_query(template, prev_subj, question_marks)
                top_idxs = search_engine.rank_docs(question.id, query, topk=len(question.supports))
                # Iterate over ranking backwards (last document is best match)
                for d in range(len(top_idxs)-1, -1, -1):
                    if top_idxs[d] not in queries_asked[query]:
                        top_idx = top_idxs[d]
                        top_doc_found = True
                        break

                # If question has been asked from all documents, pick a new subject from top doc at
                # random
                if not top_doc_found:
                    top_idx = top_idxs[-1]
                    prev_subj = (nouns[question.id]
                                      [top_idx]
                                      [randint(0, len(nouns[question.id][top_idx])-1)])
            queries_asked[query].append(top_idx)
            if verbose:
                print(query, '\n\t->', top_idx)

            # Send query to RC module
            if redis_server is None:
                rc_answers = get_rc_answers(reader, query, question.supports[top_idx])
            else:
                rc_answers, _ = get_cached_rc_answers(reader, query, question.supports[top_idx],
                                                      redis_server, question.id, top_idx)
            answer = rc_answers[0]
            score = answer.score
            if penalize_long_answers:
                score *= discount_by_answer_length(answer)
            if confidence_threshold is None or score > confidence_threshold:
                prev_subj = answer.text.lower()
                if verbose:
                    print('\t->', prev_subj, '(', score, ')')
                # If current answer is an answer candidate, submit it
                if answer.text.lower() in question.candidates_lower:
                    if answer.text.lower() == question.answer.lower():
                        print(i, ': Found correct answer', answer.text)
                        correct_answers[query_type] += 1
                        found_correct = True
                        break
                    # If the current answer is not correct and we have not submitted it before
                    elif answer.text.lower() not in incorrect_answers_this_episode:
                        print(i, ': Found incorrect answer candidate', answer.text)
                        incorrect_answers_this_episode.append(answer.text.lower())
                        if not seen_incorrect_answer_this_episode:
                            incorrect_answers[query_type] += 1
                            seen_incorrect_answer_this_episode = True
            else:
                if verbose:
                    print('\t->', prev_subj, '(', score, ') -> pick at random')
                # Pick a noun phrase at random from top document
                rand_draw = randint(0, len(nouns[question.id][top_idx])-1)
                prev_subj = nouns[question.id][top_idx][rand_draw].lower()

        if store_results:
            collected_data_row.extend([int(found_correct), len(incorrect_answers_this_episode)])
            collected_data.append(collected_data_row)

            if (i + 1) % 10 == 0:
                with open(csv_path, 'a') as f:
                    writer = csv.writer(f)
                    for line in collected_data:
                        writer.writerow(line)
                print('Written to', csv_path)
                collected_data = []

    if store_results:
        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            for line in collected_data:
                writer.writerow(line)
        print('Written to', csv_path)
    return correct_answers, incorrect_answers, counts_by_type


def parallel_eval_single_templates(templates, search_engine, dataset, nouns, reader, redis_server,
                                   num_items_to_eval, max_num_queries, confidence_threshold,
                                   penalize_long_answers, verbose=False):
    """Accelerate evaluation by querying RC model with multiple questions in parallel."""
    correct_answers = collections.defaultdict(float)
    incorrect_answers = collections.defaultdict(float)
    counts_by_type = collections.defaultdict(float)
    batch_size = 5

    for batch in range(int(math.ceil(num_items_to_eval/batch_size))):  # split data into batches
        print('Evaluating batch', batch)
        num_items_in_batch = batch_size
        if batch_size * (batch + 1) > len(dataset):
            num_items_in_batch -= batch_size * (batch + 1) - len(dataset)
        found_candidate = [False for _ in range(num_items_in_batch)]
        found_correct_answer = [False for _ in range(num_items_in_batch)]
        seen_incorrect_answer_this_episode = [False for _ in range(num_items_in_batch)]
        incorrect_answers_this_episode = [[] for _ in range(num_items_in_batch)]
        # Store past queries asked -> documents retrieved mappings
        queries_asked = collections.defaultdict(set)

        query_types = []
        prev_subj = []
        # initialise
        for item in range(num_items_in_batch):
            print('Initialising item', item, '/', num_items_in_batch)
            question = Question(dataset[batch_size * batch + item])
            query_type, subject = (question.query.split()[0],
                                   ' '.join(question.query.split()[1:]))
            counts_by_type[query_type] += 1.0
            if verbose:
                print(item, ':', question.query)
            query_types.append(query_type)
            prev_subj.append(subject.lower())
        for q in range(max_num_queries):  # for each time step
            print('\nTime step', q)
            batch_queries = []
            batch_supports = []
            batch_ids = []
            batch_top_idxs = []
            for item in range(num_items_in_batch):  # Prepare query for each question item in batch
                if found_correct_answer[item]:
                    continue
                question = Question(dataset[batch_size * batch + item])
                top_doc_found = False
                while not top_doc_found:
                    query = templates[query_types[item]] + ' ' + prev_subj[item]
                    top_idxs = search_engine.rank_docs(question.id, query,
                                                       topk=len(question.supports))
                    # Iterate over ranking backwards (last document is best match)
                    for d in range(len(top_idxs)-1, -1, -1):
                        if top_idxs[d] not in queries_asked[(item, query)]:
                            top_idx = top_idxs[d]
                            top_doc_found = True
                            break

                    # If question has been asked from all documents, pick a new subject from top
                    # doc at random to avoid loops
                    if not top_doc_found:
                        top_idx = top_idxs[-1]
                        prev_subj[item] = (nouns[question.id][top_idx]
                                                [randint(0, len(nouns[question.id][top_idx])-1)]
                                           .lower())
                        print('picked', prev_subj[item], 'at random')
                queries_asked[(item, query)].add(top_idx)
                batch_queries.append(query)
                batch_supports.append(question.supports[top_idx])
                batch_ids.append(question.id)
                batch_top_idxs.append(top_idx)
                if verbose:
                    print('   ', query, '\t->', top_idx, '\n\t', top_idxs)

            # Send queries to RC module
            if redis_server is None:
                rc_answers = get_rc_answers(reader, batch_queries, batch_supports)
            else:
                rc_answers, _ = get_cached_rc_answers(reader, batch_queries, batch_supports,
                                                      redis_server, batch_ids, top_idxs)

            for item in range(num_items_in_batch):  # generate next query
                # TODO: remove found_candidate to match eval_single_templates
                found_candidate[item] = False
                answer = rc_answers[item]
                q_i = batch * batch_size + item
                question = Question(dataset[q_i])
                score = answer.score
                if penalize_long_answers:
                    score *= discount_by_answer_length(answer)
                print(item + 1, '- answer (score %0.4f):\t' % score, answer.text)
                if score > confidence_threshold:
                    prev_subj[item] = answer.text.lower()
                    if verbose:
                        print('\t->', prev_subj[item], '(', score, ')')
                    found_candidate[item] = True
                    # If current answer is an answer candidate, submit it
                    if answer.text.lower() in question.candidates_lower:
                        if answer.text.lower() == question.answer.lower():
                            print(q_i, ': Found correct answer', answer.text)
                            correct_answers[query_type] += 1
                            break
                        # If the current answer is not correct and we have not submitted it before
                        elif answer.text.lower() not in incorrect_answers_this_episode[item]:
                            print(q_i, ': Found incorrect answer candidate', answer.text)
                            incorrect_answers_this_episode.append(answer.text.lower())
                            if not seen_incorrect_answer_this_episode[item]:
                                incorrect_answers[query_type] += 1
                                seen_incorrect_answer_this_episode[item] = True
                        else:  # We have already seen this incorrect candidate
                            found_candidate[item] = False
                if not found_candidate[item]:
                    # Pick a noun phrase at random from top document
                    rand_draw = randint(0, len(nouns[question.id][batch_top_idxs[item]])-1)
                    prev_subj[item] = nouns[question.id][batch_top_idxs[item]][rand_draw]
                    if type(prev_subj[item]) == list:
                        prev_subj[item] = prev_subj[item][0].lower()
                    else:
                        prev_subj[item] = prev_subj[item].lower()
                    print('     random draw:', prev_subj[item])

    return correct_answers, incorrect_answers, counts_by_type


def format_csv_path(data_path, len_dataset, max_num_queries, confidence_threshold, reader,
                    str_noun_parser_class, templates_from_file, qtypes_from_file):
    csv_path = 'baselines/data'

    # trim 'baselines/' and '.json'
    qtypes = '' if qtypes_from_file is None else '_' + qtypes_from_file[:-5].split('baselines/')[-1]
    csv_filename = (data_path[:-5].split('wikihop/')[-1]  # trim './data/wikihop/' and '.json',
                                                          # to keep version directory and filename
                                                          # e.g. 'v1.1/train_ids'
                    + '__' + str(len_dataset)
                    + '_' + str(max_num_queries)
                    + '_' + str(confidence_threshold)
                    + '_' + reader
                    + '_' + str_noun_parser_class
                    + '_' + templates_from_file[:-5].split('baselines/')[-1]  # keep templates
                                                                              # filename
                    + qtypes
                    + '.csv')
    return os.path.join(csv_path, csv_filename)


def print_accuracy(correct_answers, total_count):
    start_bold = '\033[1m'
    end_bold = '\033[0;0m'
    total_correct = sum(correct_answers.values())
    if total_count != 0:
        print(str(int(total_correct)) + '/' + str(total_count), '=',
              start_bold + '%0.1f' % (float(total_correct)/total_count * 100)+'%' + end_bold)
    else:
        print()


def print_eval_as_table(correct_answers, incorrect_answers, counts_by_type):
    start_bold = '\033[1m'
    end_bold = '\033[0;0m'
    total_correct = sum(correct_answers.values())
    total_incorrect = sum(incorrect_answers.values())
    total_count = sum(counts_by_type.values())  # = subset_size
    for t in sorted(counts_by_type.keys()):
        if counts_by_type[t] == 0:
            print('No instances of', t, '\n')
        else:
            print(t, ':\n', str(int(correct_answers[t])) + '/' + str(int(counts_by_type[t])), '=',
                  start_bold + '%0.1f' % (float(correct_answers[t])/counts_by_type[t] * 100)+'%'
                  + end_bold, ' ,',
                  str(int(incorrect_answers[t])) + '/' + str(int(counts_by_type[t])), '=',
                  start_bold + '%0.1f' % (float(incorrect_answers[t])/counts_by_type[t] * 100)+'%'
                  + end_bold, '\n')
    print(str(int(total_correct)) + '/' + str(total_count), '=',
          start_bold + '%0.1f' % (float(total_correct)/total_count * 100)+'%' + end_bold)
    print(str(int(total_incorrect)) + '/' + str(total_count), '=',
          start_bold + '%0.1f' % (float(total_incorrect)/total_count * 100)+'%' + end_bold)


def eval_templates():
    """Shared setup for parallel and sequential evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--spacy', nargs='?', const=True, default=False, type=bool,
                        help='If True, use Spacy to parse nouns. If False, use NLTK (default).')
    parser.add_argument('--reader', default='fastqa', type=str,
                        help='Reading comprehension model to use. One of [ fastqa | bidaf ].')
    parser.add_argument('--verbose', nargs='?', const=True, default=False, type=bool,
                        help='If True, print out all mentions of the query subject.')
    parser.add_argument('--subset_size', default=None, type=int,
                        help='If set, evaluate the baseline on a subset of data.')
    parser.add_argument('--num_items_to_eval', default=None, type=int,
                        help='If set, the number of instances to evaluate. If not set, evaluate '
                             'full dataset (or current subset).')
    parser.add_argument('--k_most_common_only', type=int, default=None,
                        help='If set, only include the k most commonly occurring relation types.')
    parser.add_argument('--wikihop_version', type=str, default='1.1',
                        help='WikiHop version to use: one of {0, 1.1}.')
    parser.add_argument('--dev', nargs='?', const=True, default=False,
                        help='If True, evaluate templates on dev data instead of train.')
    parser.add_argument('--parallel', nargs='?', const=True, default=False, type=bool,
                        help='If True, evaluate multiple questions in parallel.')
    parser.add_argument('--cache', dest='cache', action='store_true')
    parser.add_argument('--nocache', dest='cache', action='store_false')
    parser.add_argument('--trim', dest='trim_index', action='store_true')
    parser.add_argument('--notrim', dest='trim_index', action='store_false')
    parser.add_argument('--conf_threshold', default=None, type=float,
                        help='Confidence threshold required to use current answer in next query.')
    parser.add_argument('--no_question_marks', dest='question_marks', action='store_false',
                        help='If True, append question marks to questions asked to reader.')
    parser.add_argument('--store_results', nargs='?', const=True, default=False, type=bool,
                        help='If True, save the QA results in a CSV file.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed.')
    parser.add_argument('--qtypes_from_file', type=str, default=None,
                        help='If set, use only qtypes from the specified json file.')
    parser.add_argument('--templates_from_file', default='baselines/template_list_70.json',
                        type=str, help='File from which to read question templates.')
    parser.set_defaults(cache=True, trim_index=True, question_marks=True)
    args = parser.parse_args()

    subset_id, data_path, index_filename, nouns_path = format_paths(args, args.dev)

    redis_server = None
    if args.cache:
        redis_server = redis.StrictRedis(host='localhost', port=6379, db=0)
    parallel_eval = args.parallel

    print('Initialising...')
    with open(data_path) as dataset_file:
        dataset = json.load(dataset_file)
    if args.subset_size is not None:
        dataset = dataset[:args.subset_size]
    search_engine = SearchEngine(dataset, load_from_path=index_filename)

    if args.spacy:
        print('Extracting Spacy nouns...')
        noun_parser_class = SpacyNounParser
        str_noun_parser_class = 'spacy'
    else:
        print('Extracting NTLK nouns...')
        noun_parser_class = NltkNounParser
        str_noun_parser_class = 'nltk'
    # Load noun phrases from a local file (for speedup) if it exists, or create a new one if not
    nouns = pre_extract_nouns(dataset, nouns_path, noun_parser_class=noun_parser_class)
    print('Initialising {}...'.format(args.reader))
    reader = readers.reader_from_file('./rc/{}_reader'.format(args.reader))

    # Maximum number of queries allowed per instance
    max_num_queries = 25
    # Threshold above which to trust the reading comprehension module's answers
    confidence_threshold = args.conf_threshold
    # Whether to penalise answer length
    penalize_long_answers = False
    if args.seed is not None:
        # Make experiments repeatable
        random.seed(args.seed)

    verbose = args.verbose
    evaluate_nouns = False

    if args.qtypes_from_file is not None:
        included_types = json.load(open(args.qtypes_from_file))
        qtype_dataset = []
        for q in dataset:
            if q['query'].split()[0] in included_types:
                qtype_dataset.append(q)

        dataset = qtype_dataset

    # Number of instances to test (<= subset size)
    if args.num_items_to_eval is None:
        num_items_to_eval = len(dataset)
    else:
        num_items_to_eval = args.num_items_to_eval
    dataset = dataset[:num_items_to_eval]
    print('Evaluating', len(dataset), 'questions')
    if args.trim_index:
        nouns, search_engine = trim_index(dataset, nouns, search_engine)

    single_pass_counts_by_type = collections.defaultdict(float)
    for item in dataset:
        single_pass_counts_by_type[item['query'].split()[0]] += 1.0
    print('Question instances by type:', dict(single_pass_counts_by_type))

    templates = json.load(open(args.templates_from_file))
    csv_path = None
    if args.store_results:
        csv_path = format_csv_path(data_path, len(dataset), max_num_queries, args.conf_threshold,
                                   args.reader, str_noun_parser_class, args.templates_from_file,
                                   args.qtypes_from_file)
        print('Collecting data to', csv_path)
        if os.path.exists(csv_path):
            csv_reader = csv.reader(open(csv_path))
            last_row = None
            while True:
                try:
                    last_row = next(csv_reader)
                except StopIteration:
                    break
            if last_row is not None:
                last_id = int(last_row[0].split('_')[2])
                print('Continuing data collection at last ID', last_row[0])
                # continue appending to file in order
                dataset = dataset[last_id+1:] + dataset[:last_id]

    if evaluate_nouns:
        eval_nouns(dataset, nouns)
    if parallel_eval:
        correct_answers, incorrect_answers, counts_by_type = (
            parallel_eval_single_templates(templates, search_engine, dataset, nouns, reader,
                                           redis_server, num_items_to_eval, max_num_queries,
                                           confidence_threshold, penalize_long_answers, verbose))
    else:
        correct_answers, incorrect_answers, counts_by_type = (
            eval_single_templates(templates, search_engine, dataset, nouns, reader, redis_server,
                                  num_items_to_eval, max_num_queries, confidence_threshold,
                                  args.question_marks, csv_path, penalize_long_answers, verbose))

    print('Correct guesses', dict(correct_answers))
    print('Incorrect guesses', dict(incorrect_answers))

    # Print evaluation in readable format
    print_eval_as_table(correct_answers, incorrect_answers, counts_by_type)
    print('Freeing memory...')


if __name__ == "__main__":
    eval_templates()
