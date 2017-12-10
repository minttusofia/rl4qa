"""Establish baselines on how well structured question templates perform on WikiHop."""

import argparse
import collections
import json
import math
import os
import random
import time

from jack import readers
from random import randint

from ir.search_engine import SearchEngine
from rc.utils import get_rc_answers
from qa.nouns import pre_extract_nouns, SpacyNounParser, NltkNounParser
from qa.question import Question
from qa.utils import print_time_taken


def discount_by_answer_length(answer):
    if len(answer.text.split()) > 3:
        # TODO: try more sophisticated discounts
        return 3.0/len(answer.text.split())
    return 1.0


def eval_single_templates(templates, search_engine, dataset, nouns, reader, num_items_to_eval,
                          max_num_queries, confidence_threshold, penalize_long_answers,
                          verbose=False):
    """Evaluates fixed-structure templates which vary in time by the noun passed in as the object.
    """
    correct_answers = collections.defaultdict(float)
    incorrect_answers = collections.defaultdict(float)

    t = time.time()
    for i in range(num_items_to_eval):
        if not verbose and i % 10 == 0:
            if i != 0:
                t = print_time_taken(t)
            print('Evaluating question', i, '/', num_items_to_eval)
        question = Question(dataset[i])
        query_type, subject = question.query.split()[0], ' '.join(question.query.split()[1:])
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
                query = templates[query_type] + ' ' + prev_subj
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
                    prev_subj = ' '.join((nouns[question.id]
                                               [top_idx]
                                               [randint(0, len(nouns[question.id][top_idx])-1)]))
            queries_asked[query].append(top_idx)
            if verbose:
                print(query, '\n\t->', top_idx)

            # Send query to RC module
            rc_answers = get_rc_answers(reader, query, question.supports[top_idx])
            found_candidate = False
            answer = rc_answers[0]
            score = answer.score
            if penalize_long_answers:
                score *= discount_by_answer_length(answer)
            if score > confidence_threshold:
                prev_subj = answer.text
                if verbose:
                    print('\t->', prev_subj, '(', answer.score, ')')
                found_candidate = True
                # If current answer is an answer candidate, submit it
                if answer.text.lower() in question.candidates_lower:
                    if answer.text.lower() == question.answer.lower():
                        print(i, ': Found correct answer', answer.text)
                        correct_answers[query_type] += 1
                        break
                    # If the current answer is not correct and we have not submitted it before
                    elif answer.text not in incorrect_answers_this_episode:
                        print(i, ': Found incorrect answer candidate', answer.text)
                        incorrect_answers_this_episode.append(answer.text)
                        if not seen_incorrect_answer_this_episode:
                            incorrect_answers[query_type] += 1
                            seen_incorrect_answer_this_episode = True
                    else:  # We have already seen this incorrect candidate
                        found_candidate = False
            if not found_candidate:
                # Pick a noun phrase at random from top document
                rand_draw = randint(0, len(nouns[question.id][top_idx])-1)
                prev_subj = ' '.join(nouns[question.id][top_idx][rand_draw]).lower()

    return correct_answers, incorrect_answers


def parallel_eval_single_templates(templates, search_engine, dataset, nouns, reader,
                                   num_items_to_eval, max_num_queries, confidence_threshold,
                                   penalize_long_answers, verbose=False):
    """Accelerate evaluation by querying RC model with multiple questions in parallel."""
    correct_answers = collections.defaultdict(float)
    incorrect_answers = collections.defaultdict(float)
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
            if verbose:
                print(item, ':', question.query)
            query_types.append(query_type)
            prev_subj.append(subject.lower())
        for q in range(max_num_queries):  # for each time step
            print('\nTime step', q)
            batch_queries = []
            batch_supports = []
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
                batch_top_idxs.append(top_idx)
                if verbose:
                    print('   ', query, '\t->', top_idx, '\n\t', top_idxs)

            # Send queries to RC module
            rc_answers = get_rc_answers(reader, batch_queries, batch_supports)

            for item in range(num_items_in_batch):  # generate next query
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
                        print('\t->', prev_subj[item], '(', answer.score, ')')
                    found_candidate[item] = True
                    # If current answer is an answer candidate, submit it
                    if answer.text.lower() in question.candidates_lower:
                        if answer.text.lower() == question.answer.lower():
                            print(q_i, ': Found correct answer', answer.text)
                            correct_answers[query_type] += 1
                            break
                        # If the current answer is not correct and we have not submitted it before
                        elif answer.text not in incorrect_answers_this_episode[item]:
                            print(q_i, ': Found incorrect answer candidate', answer.text)
                            incorrect_answers_this_episode.append(answer.text)
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

    return correct_answers, incorrect_answers


def eval_templates():
    """Shared setup for parallel and sequential evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--nltk', nargs='?', const=True, default=False, type=bool,
                        help='If True, use NLTK to parse nouns. If False, use Spacy.')
    parser.add_argument('--parallel', nargs='?', const=True, default=False, type=bool,
                        help='If True, use NLTK to parse nouns. If False, use Spacy.')
    parser.add_argument('--subset_size', nargs=1, default=None, type=int,
                        help='If set, evaluate the baseline on a subset of data.')
    args = parser.parse_args()

    use_ntlk = args.nltk
    parallel_eval = args.parallel
    # Subset ID and subset size to use as identifiers in index, data, and noun filenames
    subset_size = int(args.subset_size[0])
    subset_id = '-6mc'
    file_path = './data/wikihop/train_ids' + subset_id + '.json'
    index_dir = './se_index'
    index_filename = os.path.join(index_dir, 'se_index' + subset_id)
    use_subset = subset_size is not None

    templates = {'instance_of': 'what is',
                 'located_in_the_administrative_territorial_entity': 'where is',
                 'occupation': 'what work as',
                 'place_of_birth': 'where was born',
                 'record_label': 'record label of',
                 'genre': 'what type is'
                 }
    print('Initialising...')
    with open(file_path) as dataset_file:
        dataset = json.load(dataset_file)
    if use_subset:
        dataset = dataset[:subset_size]
        index_filename += '_' + str(subset_size)
    search_engine = SearchEngine(dataset, load_from_path=index_filename)

    if use_ntlk:
        print('Extracting NTLK nouns...')
        noun_parser_class = NltkNounParser
    else:
        print('Extracting Spacy nouns...')
        noun_parser_class = SpacyNounParser
    # Load noun phrases from a local file (for speedup) if it exists, or create a new one if not
    stored_nouns_path = 'nouns/nouns' + subset_id + '_' + str(subset_size) + '.pkl'
    if noun_parser_class == NltkNounParser:
        stored_nouns_path = 'nouns/nouns' + subset_id + '-nltk_' + str(subset_size) + '.pkl'
    nouns = pre_extract_nouns(dataset, stored_nouns_path, noun_parser_class=noun_parser_class)#,
    # reuse_from_file='./nouns/nouns-6mc.pkl')

    reader = readers.reader_from_file('./rc/fastqa_reader')

    # Number of instances to test (<= subset size)
    num_items_to_eval = 1
    # Maximum number of queries allowed per instance
    max_num_queries = 25
    # Threshold above which to trust the reading comprehension module's answers
    confidence_threshold = 0.25
    # Whether to penalise answer length
    penalize_long_answers = True
    # Make experiments repeatable
    random.seed(0)

    verbose = True

    dataset = dataset[:num_items_to_eval]
    counts_by_type = collections.defaultdict(float)
    for item in dataset:
        counts_by_type[item['query'].split()[0]] += 1.0
    print('Question instances by type:', dict(counts_by_type))

    if parallel_eval:
        correct_answers, incorrect_answers = (
            parallel_eval_single_templates(templates, search_engine, dataset, nouns, reader,
                                           num_items_to_eval, max_num_queries, confidence_threshold,
                                           penalize_long_answers, verbose))
    else:
        correct_answers, incorrect_answers = (
            eval_single_templates(templates, search_engine, dataset, nouns, reader,
                                  num_items_to_eval, max_num_queries, confidence_threshold,
                                  penalize_long_answers, verbose))

    print('Correct guesses', dict(correct_answers))
    print('Incorrect guesses', dict(incorrect_answers))
    print('Freeing memory...')


if __name__ == "__main__":
    eval_templates()
