"""Defines SearchEngine class, which builds an index for each WikiHop question and ranks its docs.

usage:
    python -m ir.search_engine
        Builds a new index consisting of the whole dataset. Also creates train_ids.json with 
        question->id mappings, and stores the index under './se_index'. If either of these exists,
        they will be reused and simply the test cases will be executed.

    python -m ir.search_engine --subset_size=100
        Builds a new index consisting of the first 100 questions.

    python -m ir.search_engine --new_index
        Override an existing index (of the same size).

    python -m ir.search_engine --new_ids
        Override an existing question->id allocation.
"""

import argparse
import collections
import json
import numpy as np
import os
import pickle
import time
import uuid

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from qa.utils import print_time_taken


class SearchEngine:
    def __init__(self, queries=None, save_index_to_path=None, load_from_path=None):
        if load_from_path:
            self.vec_for_q = pickle.load(open(load_from_path + '_vec.pkl', 'rb'))
            self.sparse_for_q = pickle.load(open(load_from_path + '_sparse.pkl', 'rb'))
            self.shape_for_q = pickle.load(open(load_from_path + '_shape.pkl', 'rb'))
            assert self.shape_for_q.keys() == self.sparse_for_q.keys()
            return
        self.data = [{} for _ in range(len(queries))]
        self.vec_for_q = {}
        self.shape_for_q = {}
        self.sparse_for_q = {}
        for i in range(len(queries)):
            self.data[i]['id'] = queries[i]['id']
            self.data[i]['docs'] = queries[i]['supports']
        del queries
        self.build_index()
        if save_index_to_path:
            print('Writing indices to file...')
            pickle.dump(self.vec_for_q, open(save_index_to_path + '_vec.pkl', 'wb'),
                        pickle.HIGHEST_PROTOCOL)
            print('Written ' + save_index_to_path + '_vec.pkl')
            pickle.dump(self.sparse_for_q, open(save_index_to_path + '_sparse.pkl', 'wb'),
                        pickle.HIGHEST_PROTOCOL)
            print('Written ' + save_index_to_path + '_sparse.pkl')
            pickle.dump(self.shape_for_q, open(save_index_to_path + '_shape.pkl', 'wb'),
                        pickle.HIGHEST_PROTOCOL)
            print('Written ' + save_index_to_path + '_shape.pkl')

    def build_index(self):
        print('Constructing indices')
        for i in range(len(self.data)):
            vectorizer = TfidfVectorizer(dtype=np.float32)
            q = self.data[i]
            if (i+1) % 1000 == 0:
                print(i+1, '/', len(self.data))
            documents = q['docs']
            vectorizer.fit(documents)
            data = vectorizer.transform(documents)
            self.vec_for_q[q['id']] = vectorizer
            self.shape_for_q[q['id']] = data.shape
            sparse = csr_matrix(data)
            self.sparse_for_q[q['id']] = sparse
        if len(self.data) % 1000 != 0:
            print(len(self.data), '/', len(self.data))

    def rank_docs(self, q_id, search_query, topk=1):
        sparse = self.sparse_for_q[q_id]
        if type(search_query) == str:
            search_query = [search_query]

        # transforming the queries to a sparse tfidf matrix
        Y = self.vec_for_q[q_id].transform(search_query).toarray()
        results = np.empty((Y.shape[0], sparse.shape[0]))

        # spare matrix - dense vector multiplication
        # this gets a score for each document for a single query
        for i, vec in enumerate(Y):
            results[i] = sparse.dot(vec)

        # sorting the output from smallest to largest per score
        # highest scoring documents are the last documents in each row
        ranked = np.argsort(results, 1)
        top_ranked = ranked[:,-topk:]

        if len(top_ranked) == 1:
            return top_ranked[0]
        # return the top scoring documents by selecting the last elements in each row
        return top_ranked


def load_dataset_with_ids(assign_new_ids, id_filename, data_filename, t):
    if assign_new_ids or not os.path.exists(id_filename):
        with open(data_filename) as f:
            dataset = json.load(f)
        t = print_time_taken(t)
        print('Assigning new ids...')
        for q in dataset:
            q['id'] = str(uuid.uuid4())
        with open(id_filename, 'w') as f:
            json.dump(dataset, f)
    else:
        dataset = json.load(open(id_filename))
    return dataset, t


def run_test_queries(se, t):
    print('Executing test queries...')

    test_queries = ['games pan american', 'Christian charismatic megachurch Houston Texas',
                    'between 231 and 143 million']
    test_instances = [0, 1, 2]
    expected_doc = [1, 1, 14]
    for i in range(len(test_instances)):
        ranked = se.rank_docs(dataset[test_instances[i]]['id'], test_queries[i])
        print(i, ranked, dataset[test_instances[i]]['supports'][ranked[0]][:60] + "...")
        assert ranked[0] == expected_doc[i]
    t = print_time_taken(t)
    print('Cleaning up...')
    del se
    _ = print_time_taken(t)


def answer_lengths(data):
    answer_lengths = collections.defaultdict(int)
    for j in range(len(data)):
        length = len(data[j]['answer'].split())
        answer_lengths[length] += 1
    return answer_lengths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_index', nargs='?', const=True, default=False,
                        help='If True, create new index .pkl files (rather than using existing '
                             'ones).')
    parser.add_argument('--new_ids', nargs='?', const=True, default=False,
                        help='If True, assign each question with a new ID (needed to map '
                             'questions to indices).')
    parser.add_argument('--subset_size', nargs=1, default=None,
                        help='If set, use a subset of data for development.')
    args = parser.parse_args()

    assign_new_ids = args.new_ids
    create_new_index = args.new_index
    if assign_new_ids:  # Cannot use previous index if ID's are reassigned
        create_new_index = True
    use_subset = False
    if args.subset_size:
        if type(args.subset_size) == list:
            args.subset_size = args.subset_size[0]
        subset_size = int(args.subset_size)
        use_subset = True
    # Set to an integer k to only include k most frequently occurring question types. Set to None
    # to include all.
    k_most_common_relations_only = 6

    # Calculate how many correct answers consist of 1, 2, 3, ... words
    show_answer_lengths = True
    if show_answer_lengths:
        print('Answer length - occurrence map:')
        print(answer_lengths(json.load(open('./data/wikihop/dev.json'))))

    t = time.time()
    print('Loading data...')
    subset_id = ''
    if k_most_common_relations_only:
        subset_id = '-' + str(k_most_common_relations_only) + 'mc'
    data_filename = './data/wikihop/train' + subset_id + '.json'
    id_filename = './data/wikihop/train_ids' + subset_id + '.json'
    dataset, t = load_dataset_with_ids(assign_new_ids, id_filename, data_filename, t)

    t = print_time_taken(t)
    print('Initialising search engine...')

    index_dir = './se_index'
    index_filename = os.path.join(index_dir, 'se_index' + subset_id)
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    if use_subset:
        dataset = dataset[:subset_size]
        index_filename += '_' + str(subset_size)
    if create_new_index or not os.path.exists(index_filename + '_vec.pkl'):
        se = SearchEngine(dataset, save_index_to_path=index_filename)
    else:
        se = SearchEngine(load_from_path=index_filename)
    t = print_time_taken(t)

    if not k_most_common_relations_only:
        # Test initialised search engine
        run_test_queries(se, t)


