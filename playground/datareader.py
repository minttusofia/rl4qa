import json

from random import randint
from ir.search_engine import SearchEngine
from playground.nouns import nounPhrases

#from jack import readers
#from jack.core import QASetting

class Question():
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

def playground(automatic_first_query=False):
    index_filename = "se_index"
    use_subset = True
    subset_size = 1000
    print('Initialising...')
    file_path = '../data/wikihop/train_ids.json'
    with open(file_path) as dataset_file:
         dataset = json.load(dataset_file)
    if use_subset:
        dataset = dataset[:subset_size]
        index_filename += '_' + str(subset_size)
    se = SearchEngine(dataset, load_from_path=index_filename)

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
                submission = input("Submit your answer:")
                if submission == question.answer:
                    print("That's correct!")
                else:
                    print("False, the correct answer is ", question.answer)
                break
            elif action.lower() == 'q':
                query = input("Query: ")
                top_idx = se.rank_docs(question.id, query, topk=len(question.supports))#[0]
                for doc in range(len(top_idx)-1, -1, -1):
                    if read_this_episode[doc]:
                       print('[READ]', question.supports[doc])
                    else:
                        top_idx = doc
                print('\n', question.supports[top_idx], '\n')
                showNouns(question.supports[top_idx])
        cont = input("Continue? (Y/n) ")
        if cont.lower() == 'n':
            break
    print('Freeing memory...')


if __name__ == "__main__":
    playground()
