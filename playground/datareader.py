"""Allows experimental interaction between data, search engine and reading comprehension module."""

import json
import os
import random

from jack import readers

from ir.search_engine import SearchEngine
from rc.utils import show_rc_answer
from qa.nouns import SpacyNounParser, NltkNounParser
from qa.nouns import pre_extract_nouns
from qa.question import Question


def phrase_appears_in_sentence(phrase, sent):
    phrase_words = phrase.lower().split()
    sent_words = sent.lower().split()
    for s in range(len(sent_words)):
        if phrase_words[0] == sent_words[s]:
            for p in range(1, len(phrase_words)):
                if s + p >= len(sent_words) or phrase_words[p] != sent_words[s + p]:
                    break
                elif p == len(phrase_words)-1:
                    return True
    return False


def show_mentions_of_noun(noun_phrase, supports, already_shown=None, noun_parser=None,
                          num_indents=0):
    show_extracted_nouns = False
    if noun_parser is None:
        noun_parser = SpacyNounParser()
    if already_shown is None:
        already_shown = set()
    elif noun_phrase.lower() in already_shown:
        return already_shown
    already_shown.add(noun_phrase.lower())
    nouns = []
    for d in range(len(supports)):
        sentences = supports[d].split('. ')
        for s in sentences:
            if phrase_appears_in_sentence(noun_phrase, s):
                print('\t' * num_indents, end='')
                ns = noun_parser.extract_nouns(s)
                if show_extracted_nouns:
                    print(d, '---', noun_phrase, '---', s, '::', ns)
                else:
                    print(d, '---', noun_phrase, '---')
                nouns.extend(ns)
    for n in nouns:
        already_shown = show_mentions_of_noun(' '.join(n), supports, already_shown, noun_parser,
                                              num_indents+1)
    return already_shown


def playground(automatic_first_query=False):
    # Set random seed to system time
    random.seed()
    subset_id = '-6mc'
    file_path = './data/wikihop/train_ids' + subset_id + '.json'
    index_dir = './se_index'
    index_filename = os.path.join(index_dir, 'se_index' + subset_id)
    use_subset = True
    subset_size = 100
    verbose = False

    print('Initialising...')
    with open(file_path) as dataset_file:
         dataset = json.load(dataset_file)
    if use_subset:
        dataset = dataset[:subset_size]
        index_filename += '_' + str(subset_size)
    se = SearchEngine(dataset, load_from_path=index_filename)

    print('Extracting nouns...')
    # Load noun phrases from a local file (for speedup) if it exists, or create a new one if not
    stored_nouns_path = 'nouns/nouns' + subset_id + '_' + str(subset_size) + '.pkl'
    nouns = pre_extract_nouns(dataset, stored_nouns_path)

    debug_noun_extraction = False
    if debug_noun_extraction:
        sp_noun_parser = SpacyNounParser()
        nltk_noun_parser = NltkNounParser()

    reader = readers.reader_from_file("./rc/fastqa_reader")
    if verbose:
        #noun_parser = NltkNounParser()
        noun_parser = SpacyNounParser()

    while True:
        q_i = random.randint(0, len(dataset)-1)
        question = Question(dataset[q_i])
        question.show_question()
        read_this_episode = [False for _ in range(len(question.supports))]
        if verbose:
            show_mentions_of_noun(' '.join(question.query.split()[1:]), question.supports,
                                  noun_parser=noun_parser)
        if automatic_first_query:
            first_query = question.replace('_', ' ')
            top_idx = se.rank_docs(question.id, first_query)
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
                top_idx = se.rank_docs(question.id, query, topk=len(question.supports))#[0]
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
                print('\n', len(question.supports) - d, '- (doc ' + str(doc) + ') :',
                      question.supports[top_idx], '\n')
                read_this_episode[top_idx] = True
                print(nouns[question.id][top_idx])
                if debug_noun_extraction:
                    print('sp', sp_noun_parser.extract_nouns(question.supports[top_idx]))
                    print('ntlk', nltk_noun_parser.extract_nouns(question.supports[top_idx]))
                show_rc_answer(reader, query, question.supports[top_idx])

        cont = input("Continue? (Y/n) ")
        if cont.lower() == 'n':
            break
    print('Freeing memory...')


if __name__ == "__main__":
    playground()
