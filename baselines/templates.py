"""Establish baselines on how well structured question templates perform on WikiHop."""

import collections
import json
import os
import random
import time

from jack import readers
from random import randint

from ir.search_engine import SearchEngine
from rc.utils import get_rc_answer
from qa.nouns import pre_extract_nouns
from qa.question import Question
from qa.utils import print_time_taken


def eval_single_templates(automatic_first_query=False):
    """Evaluates fixed-structure templates which vary in time by the noun passed in as the object.
    """
    # Subset ID and subset size to use as identifiers in index, data, and noun filenames
    subset_id = '-6mc'
    file_path = './data/wikihop/train_ids' + subset_id + '.json'
    index_dir = './se_index'
    index_filename = os.path.join(index_dir, 'se_index' + subset_id)
    use_subset = True
    subset_size = 100

    # included_question_types = ['instance_of', 'located_in_the_administrative_territorial_entity',
    #                            'occupation', 'place_of_birth', 'record_label', 'genre']
    templates = {'instance_of': 'what is',
                 'located_in_the_administrative_territorial_entity': 'where is',
                 'occupation': 'what work as',
                 'place_of_birth': 'where was born',
                 'record_label': 'record label of',
                 'genre': 'what type is'
                 }
    correct_answers = collections.defaultdict(float)
    incorrect_answers = collections.defaultdict(float)

    print('Initialising...')
    with open(file_path) as dataset_file:
        dataset = json.load(dataset_file)
    if use_subset:
        dataset = dataset[:subset_size]
        index_filename += '_' + str(subset_size)
    se = SearchEngine(dataset, load_from_path=index_filename)

    counts_by_type = collections.defaultdict(float)
    for item in dataset:
        counts_by_type[item['query'].split()[0]] += 1.0
    print('Question instances by type:', dict(counts_by_type))
    print('Extracting nouns...')
    # Load noun phrases from a local file (for speedup) if it exists, or create a new one if not
    stored_nouns_path = 'nouns/nouns' + subset_id + '_' + str(subset_size) + '.pkl'
    nouns = pre_extract_nouns(dataset, stored_nouns_path)

    reader = readers.reader_from_file('./rc/fastqa_reader')

    # Number of instances to test
    batch_size = 100
    # Maximum number of queries allowed per instance
    max_episode_len = 25
    # Threshold above which to trust the reading comprehension module's answers
    confidence_threshold = 0.25
    # Whether to penalise answer length
    penalize_long_answers = True
    # Make experiments repeatable
    random.seed(0)

    verbose = False

    t = time.time()
    for i in range(batch_size):
        if not verbose and i % 10 == 0:
            if i != 0:
                t = print_time_taken(t)
            print('Evaluating question', i, '/', batch_size)
        question = Question(dataset[i])
        read_this_episode = [False for _ in range(len(question.supports))]
        query_type, subject = question.query.split()[0], ' '.join(question.query.split()[1:])
        if verbose:
            print('\n' + str(i), ':', question.query, '(', question.answer, ')')

        if automatic_first_query:
            first_query = question.replace('_', ' ')
            top_idx = se.rank_docs(question.id, first_query)
            read_this_episode[top_idx[0]] = True

        prev_subj = subject
        # Store past queries asked -> documents retrieved mappings
        queries_asked = collections.defaultdict(list)
        incorrect_answers_this_episode = []
        for _ in range(max_episode_len):
            question_found = False
            while not question_found:
                query = templates[query_type] + ' ' + prev_subj
                top_idxs = se.rank_docs(question.id, query, topk=len(question.supports))#[0]
                # Iterate over ranking backwards (last document is best match)
                for d in range(len(top_idxs)-1, -1, -1):
                    if top_idxs[d] not in queries_asked[query]:
                        top_idx = top_idxs[d]
                        question_found = True
                        break

                # If question has been asked from all documents, pick a new subject from top doc at
                # random
                if not question_found:
                    top_idx = top_idxs[-1]
                    prev_subj = ' '.join((nouns[question.id]
                                               [top_idx]
                                               [randint(0, len(nouns[question.id][top_idx])-1)]))
            queries_asked[query].append(top_idx)
            if verbose:
                print(query, '\n\t->', top_idx)

            read_this_episode[top_idx] = True
            rc_answers = get_rc_answer(reader, query, question.supports[top_idx])
            found_candidate = False
            answer = rc_answers[0]
            score = answer.score
            if penalize_long_answers and len(answer.text.split()) > 3:
                # TODO: try more sophisticated discounts
                score = 3.0/len(answer.text.split()) * score
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
                        incorrect_answers[query_type] += 1
                        incorrect_answers_this_episode.append(answer.text)
                    else:  # We have already seen this incorrect candidate
                        found_candidate = False
            if not found_candidate:
                prev_subj = ' '.join(nouns[question.id]
                                          [top_idx]
                                          [randint(0, len(nouns[question.id][top_idx])-1)])

    print('Correct guesses', dict(correct_answers))
    print('Incorrect guesses', dict(incorrect_answers))
    print('Freeing memory...')


if __name__ == "__main__":
    eval_single_templates()
