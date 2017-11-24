"""Allows experimental interaction between data, search engine and reading comprehension module."""

import json

from random import randint
from ir.search_engine import SearchEngine
from playground.nouns import nounPhrases

from jack import readers
from jack.core import QASetting


class Question:
    def __init__(self, json):
        self.id = json['id']
        self.query = json['query']
        self.answer = json['answer']
        self.candidates = json['candidates']
        self.supports = json['supports']


def showQuestion(question):
    print(question.query)


def showNouns(doc):
    print(nounPhrases(doc))


def showRCAnswer(reader, query, doc):
    print('Reading Comprehension answers:')
    answers = reader([QASetting(question=query, support=[doc])])
    for a in answers:
        print(a.text, '\tscore:', a.score)


def playground(automatic_first_query=False):
    index_filename = "se_index"
    use_subset = True
    subset_size = 100
    print('Initialising...')
    file_path = '../data/wikihop/train_ids.json'
    with open(file_path) as dataset_file:
         dataset = json.load(dataset_file)
    if use_subset:
        dataset = dataset[:subset_size]
        index_filename += '_' + str(subset_size)
    se = SearchEngine(dataset, load_from_path=index_filename)
    reader = readers.reader_from_file("./fastqa_reader")

    while True:
        q_i = randint(0, len(dataset))
        question = Question(dataset[q_i])
        showQuestion(question)
        read_this_episode = [False for _ in range(len(question.supports))]
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
                for d in range(len(top_idx)-1, -1, -1):
                    doc = top_idx[d]
                    if read_this_episode[doc]:
                       print(len(top_idx) - d, ': [READ] ', question.supports[doc])
                    else:
                        top_idx = doc
                        break
                print('\n', len(question.supports) - d, ':', question.supports[top_idx], '\n')
                read_this_episode[top_idx] = True
                showNouns(question.supports[top_idx])
                showRCAnswer(reader, query, question.supports[top_idx])

        cont = input("Continue? (Y/n) ")
        if cont.lower() == 'n':
            break
    print('Freeing memory...')


if __name__ == "__main__":
    playground()
