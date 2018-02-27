"""Allows experimental interaction between data, search engine and reading comprehension module."""

import argparse
import collections
import json
import os
import random
import sys

from colors import color
from jack import readers

from ir.search_engine import SearchEngine
from rc.utils import show_rc_answers
from qa.nouns import SpacyNounParser, NltkNounParser
from qa.nouns import pre_extract_nouns
from qa.question import Question
from qa.utils import print_colored_doc, phrase_appears_in_doc


def show_mentions_of_noun(noun_phrase, supports, already_shown=None, noun_parser=None,
                          num_indents=0, extra_words_to_highlight=None):
    show_extracted_nouns = False
    if noun_parser is None:
        noun_parser = SpacyNounParser()
    if already_shown is None:
        already_shown = set()
    elif noun_phrase.lower() in already_shown:
        return already_shown
    if extra_words_to_highlight is None:
        extra_words_to_highlight = ''
    already_shown.add(noun_phrase.lower())
    nouns = []
    found_exact_phrase = False
    for d in range(len(supports)):
        sentences = supports[d].split('. ')
        for s in sentences:
            if phrase_appears_in_doc(noun_phrase, s):
                found_exact_phrase = True
                print('\t' * num_indents, end='')
                # Extract nouns for a single sentence only
                ns = noun_parser.extract_nouns(s)
                if show_extracted_nouns:
                    print(d, '---', color(noun_phrase, fg='green'), '---', s, '::', ns, end=' ')
                else:
                    print(d, '---', color(noun_phrase, fg='green'), '---', end=' ')

                # Print a single sentence instead of a whole doc
                print_colored_doc(s, query=noun_phrase + ' ' + extra_words_to_highlight, nouns=ns)
                #print_colored_doc(supports[d], query=noun_phrase + ' ' + extra_words_to_highlight)
                nouns.extend(ns)

    if not found_exact_phrase:
        for word in noun_phrase:
            already_shown = show_mentions_of_noun(word, supports, already_shown,
                                                  noun_parser=noun_parser,
                                                  extra_words_to_highlight=noun_phrase)
    for n in nouns:
        already_shown = show_mentions_of_noun(n, supports, already_shown, noun_parser,
                                              num_indents+1)
    return already_shown


def populate_outgoing_connections(all_sentences, nouns, noun, connections, noun_set, already_added):
    for d in range(len(all_sentences)):
        for s in range(len(all_sentences[d])):
            sent = all_sentences[d][s]
            if phrase_appears_in_doc(noun, sent):
                for n in nouns[d][s]:
                    if n not in already_added and noun != n:
                        connections[(noun, n)].add((d, s))
                        noun_set.add(n)


def show_connections_to_correct_answer(question, noun_parser):
    """Print out chains from question subject to correct answer.

    e.g. subject : d0 -> n1 : d1 -> n2 : d2 -> n3 : d3 -> answer
    """
    subject = ' '.join(question.query.split()[1:])
    answer = question.answer
    # TODO: consider non-contiguous mentions of subject (e.g. firstname middlename lastname)
    # Assume nouns have to appear in same sentence
    all_sentences = []
    print('Showing connections from', subject, 'to', answer)
    nouns = [[] for _ in range(len(question.supports))]
    for d in range(len(question.supports)):
        doc = question.supports[d]
        sentences = doc.split('. ')
        all_sentences.append(sentences)
        for sent in sentences:
            # Extract nouns for a single sentence only
            nouns[d].append([w.lower() for w in noun_parser.extract_nouns(sent)])
            if answer in nouns[d][-1]:
                print('\nfound anwer', answer, '\n')
        if answer in doc:
            print('\n', doc, '\ncontains', answer)
    connections = collections.defaultdict(set)
    remaining_set = set()
    already_added = set()
    populate_outgoing_connections(all_sentences, nouns, subject, connections, remaining_set,
                                  already_added)
    while not len(remaining_set) == 0:
        noun = remaining_set.pop()
        if noun != subject:
            populate_outgoing_connections(all_sentences, nouns, noun, connections, remaining_set,
                                          already_added)
        already_added.add(noun)
    print(connections)


def playground_main(dataset, search_engine, reader, nouns, noun_parser, verbose,
                    debug_noun_extraction=False, sp_noun_parser=None, nltk_noun_parser=None,
                    automatic_first_query=False, allow_multiple_reads=False):
    while True:  # Loop forever and pick questions at random
        q_i = random.randint(0, len(dataset)-1)
        question = Question(dataset[q_i])
        #show_connections_to_correct_answer(question, noun_parser)
        print(q_i, end=' - ')
        question.show_question()
        read_this_episode = [False for _ in range(len(question.supports))]
        if verbose:
            show_mentions_of_noun(' '.join(question.query.split()[1:]), question.supports,
                                  noun_parser=noun_parser)
        if automatic_first_query:
            first_query = question.replace('_', ' ')
            top_idx = search_engine.rank_docs(question.id, first_query)
            read_this_episode[top_idx[0]] = True
            print('\n', question.supports[top_idx[0]], '\n')
        while True:
            action = input("Type 'q' for next query, 'a' to submit an answer: ")
            if action.lower() == 'a':
                submission = input("Submit your answer: ")
                if submission.lower() == question.answer.lower():
                    print("That's correct!")
                else:
                    show_true = input("Incorrect. Would you like to see the correct answer? (Y/n) ")
                    if show_true.lower() != 'n':
                        print("The correct answer is", question.answer)
                break
            elif action.lower() == 'q':
                query = input("Query: ")
                top_idx = search_engine.rank_docs(question.id, query, topk=len(question.supports))
                print('Document ranking:', top_idx)
                # Iterate over ranking backwards (last document is best match)
                if allow_multiple_reads:
                    top_idx = top_idx[-1]
                    doc = top_idx
                    print('\n', 1, '- (doc ' + str(top_idx) + ') :')
                else:
                    for d in range(len(top_idx)-1, -1, -1):
                        doc = top_idx[d]
                        if read_this_episode[doc]:
                            # Indicate document has been read
                            print(len(top_idx) - d, '- (doc ' + str(doc) + ') : [READ] ',
                                  question.supports[doc])
                        else:
                            # Select Best-ranked new document
                            top_idx = doc
                            break
                    print('\n', len(question.supports) - d, '- (doc ' + str(doc) + ') :')
                print_colored_doc(question.supports[top_idx], query, nouns[question.id][top_idx])
                print()
                read_this_episode[top_idx] = True
                print(nouns[question.id][top_idx])
                if debug_noun_extraction:
                    print('sp', sp_noun_parser.extract_nouns(question.supports[top_idx]))
                    print('ntlk', nltk_noun_parser.extract_nouns(question.supports[top_idx]))
                show_rc_answers(reader, query, question.supports[top_idx])

        cont = input("Continue? (Y/n) ")
        if cont.lower() == 'n':
            break


def format_paths(args, dev=False):
    base_filename = './data/wikihop/v' + args.wikihop_version + '/'
    if args.dev:
        base_filename += 'dev'
    else:
        base_filename += 'train'
    data_path = base_filename + '_ids'

    subset_id = ''
    if args.k_most_common_only:
        subset_id = '-' + str(args.k_most_common_only) + 'mc'
    data_path += subset_id + '.json'
    index_dir = './se_index/v' + args.wikihop_version
    if dev:
        index_dir = os.path.join(index_dir, 'dev')
    index_filename = os.path.join(index_dir, 'se_index' + subset_id)

    nouns_path = 'nouns/v' + args.wikihop_version
    if dev:
        nouns_path = os.path.join(nouns_path, 'dev')
    nouns_path = os.path.join(nouns_path, 'nouns' + subset_id)
    if args.spacy:
        nouns_path += '-spacy'
    else:
        nouns_path += '-nltk'
    if args.subset_size is not None:
        nouns_path += '_' + str(args.subset_size)
    nouns_path += '.pkl'

    return subset_id, data_path, index_filename, nouns_path


def playground_setup():
    # Set random seed to system time
    random.seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--spacy', nargs='?', const=True, default=False, type=bool,
                        help='If True, use Spacy to parse nouns. If False, use NLTK (default).')
    parser.add_argument('--verbose', nargs='?', const=True, default=False, type=bool,
                        help='If True, print out all mentions of the query subject.')
    parser.add_argument('--debug_noun_extraction', nargs='?', const=True, default=False, type=bool,
                        help='If set, evaluate the baseline on a subset of data.')
    parser.add_argument('--subset_size', default=None, type=int,
                        help='If set, evaluate the baseline on a subset of data.')
    parser.add_argument('--k_most_common_only', type=int, default=None,
                        help='If set, only include the k most commonly occurring relation types.')
    parser.add_argument('--wikihop_version', type=str, default='1.1',
                        help='WikiHop version to use: one of {0, 1.1}.')
    parser.add_argument('--dev', nargs='?', const=True, default=False,
                        help='If True, build an index on dev data instead of train.')
    args = parser.parse_args()

    subset_id, data_path, index_filename, nouns_path = format_paths(args, args.dev)
    debug_noun_extraction = args.debug_noun_extraction
    allow_multiple_reads = True

    print('Initialising...')
    with open(data_path) as dataset_file:
        dataset = json.load(dataset_file)
    if args.subset_size is not None:
        dataset = dataset[:args.subset_size]
        index_filename += '_' + str(args.subset_size)
    search_engine = SearchEngine(dataset, load_from_path=index_filename)

    if args.spacy:
        print('Extracting Spacy nouns...')
        noun_parser_class = SpacyNounParser
    else:
        print('Extracting NTLK nouns...')
        noun_parser_class = NltkNounParser
    # To find mentions of subject noun at runtime (if verbose)
    noun_parser = noun_parser_class()

    # Load noun phrases from a local file (for speedup) if it exists, or create a new one if not
    nouns = pre_extract_nouns(dataset, nouns_path, noun_parser_class=noun_parser_class)
    print('Loaded extracted nouns for', len(nouns.keys()), 'WikiHop items')

    sp_noun_parser = None
    nltk_noun_parser = None
    if debug_noun_extraction:
        sp_noun_parser = SpacyNounParser()
        nltk_noun_parser = NltkNounParser()

    reader = readers.reader_from_file('./rc/fastqa_reader')
    # Playground main loop
    playground_main(dataset, search_engine, reader, nouns, noun_parser, args.verbose,
                    debug_noun_extraction, sp_noun_parser, nltk_noun_parser,
                    allow_multiple_reads=allow_multiple_reads)
    print('Freeing memory...')


if __name__ == "__main__":
    playground_setup()
