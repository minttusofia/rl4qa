"""Allows experimental interaction between data, search engine and reading comprehension module."""

import argparse
import json
import os
import random

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
                ns = noun_parser.extract_nouns(s)
                if show_extracted_nouns:
                    print(d, '---', color(noun_phrase, fg='green'), '---', s, '::', ns, end=' ')
                else:
                    print(d, '---', color(noun_phrase, fg='green'), '---', end=' ')
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


def playground_main(dataset, search_engine, reader, nouns, noun_parser, verbose,
                    debug_noun_extraction=False, sp_noun_parser=None, nltk_noun_parser=None,
                    automatic_first_query=False):
    while True:  # Loop forever and pick questions at random
        q_i = 0#random.randint(0, len(dataset)-1)
        question = Question(dataset[q_i])
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
                print_colored_doc(question.supports[top_idx], query, nouns[question.id][top_idxq])
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


def playground_setup():
    # Set random seed to system time
    random.seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--nltk', nargs='?', const=True, default=False, type=bool,
                        help='If True, use NLTK to parse nouns. If False, use Spacy.')
    parser.add_argument('--verbose', nargs='?', const=True, default=False, type=bool,
                        help='If True, print out all mentions of the query subject.')
    parser.add_argument('--debug_noun_extraction', nargs='?', const=True, default=False, type=bool,
                        help='If set, evaluate the baseline on a subset of data.')
    parser.add_argument('--subset_size', nargs=1, default=100, type=int,
                        help='If set, evaluate the baseline on a subset of data.')
    args = parser.parse_args()

    use_ntlk = args.nltk
    # Subset ID and subset size to use as identifiers in index, data, and noun filenames
    subset_size = int(args.subset_size[0])
    subset_id = '-6mc'
    file_path = './data/wikihop/train_ids' + subset_id + '.json'
    index_dir = './se_index'
    index_filename = os.path.join(index_dir, 'se_index' + subset_id)
    use_subset = subset_size is not None
    verbose = args.verbose
    debug_noun_extraction = args.debug_noun_extraction

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
    # To find mentions of subject noun at runtime (if verbose)
    noun_parser = noun_parser_class()

    # Load noun phrases from a local file (for speedup) if it exists, or create a new one if not
    stored_nouns_path = 'nouns/nouns' + subset_id + '_' + str(subset_size) + '.pkl'
    if noun_parser_class == NltkNounParser:
        stored_nouns_path = 'nouns/nouns' + subset_id + '-nltk_' + str(subset_size) + '.pkl'
    nouns = pre_extract_nouns(dataset, stored_nouns_path, noun_parser_class=noun_parser_class)

    sp_noun_parser = None
    nltk_noun_parser = None
    if debug_noun_extraction:
        sp_noun_parser = SpacyNounParser()
        nltk_noun_parser = NltkNounParser()

    reader = readers.reader_from_file("./rc/fastqa_reader")
    # Playground main loop
    playground_main(dataset, search_engine, reader, nouns, noun_parser, verbose,
                    debug_noun_extraction, sp_noun_parser, nltk_noun_parser)
    print('Freeing memory...')


if __name__ == "__main__":
    playground_setup()
