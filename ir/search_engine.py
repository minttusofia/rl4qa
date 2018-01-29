"""Defines SearchEngine class, which builds an index for each WikiHop question and ranks its docs.

usage:
    python -m ir.search_engine
        Builds a new index consisting of the whole dataset. Also creates train_ids.json with 
        question->id mappings, and stores the index under './se_index'. If either of these exists,
        they will be reused and simply the test cases will be executed.

    python -m ir.search_engine --subset_size=100
        Builds a new index consisting of the first 100 questions.
        
    python -m ir.search_engine --k_most_common_only=5
        Builds a new index consisting of the 5 most commonly occurring relation types only.

"""

import argparse
import collections
import json
import nltk
import numpy as np
import os
import pickle
import regex
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


def create_full_id_map(base_filename):
    t = time.time()
    data_filename = base_filename + '.json'
    id_filename = base_filename + '_ids.json'
    with open(data_filename) as f:
        dataset = json.load(f)
    t = print_time_taken(t)
    print('Assigning new ids...')
    for q in dataset:
        if 'id' not in q:  # v1.1 has preassigned IDs
            q['id'] = str(uuid.uuid4())
    t = print_time_taken(t)
    print('Tokenizing documents...')
    # Tokenize support documents properly before building TF-IDF
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    r = regex.compile(r'\w+\. ')
    for i in range(len(dataset)):
        q = dataset[i]
        for d in range(len(q['supports'])):
            q['supports'][d] = ' '.join(tokenizer.tokenize(q['supports'][d]))
            periods_not_tokenized = regex.findall(r, q['supports'][d])
            for match in periods_not_tokenized:
                q['supports'][d] = q['supports'][d].replace(match, match[:-2] + ' . ')
        if (i+1) % 1000 == 0:
            print(i+1, '/', len(dataset), end='\t')
            t = print_time_taken(t)

    json.dump(dataset, open(id_filename, 'w'))


def run_test_queries(search_engine, t):
    print('Executing test queries...')

    test_queries = ['games pan american', 'Christian charismatic megachurch Houston Texas',
                    'between 231 and 143 million']
    test_instances = [0, 1, 2]
    expected_doc = [1, 1, 14]
    for i in range(len(test_instances)):
        ranked = search_engine.rank_docs(dataset[test_instances[i]]['id'], test_queries[i])
        print(i, ranked, dataset[test_instances[i]]['supports'][ranked[0]][:60] + "...")
        assert ranked[0] == expected_doc[i]
    t = print_time_taken(t)
    print('Cleaning up...')
    del search_engine
    _ = print_time_taken(t)


def filter_by_most_common(k_most_common_only, data_path):
    most_common_relations = ['instance_of',
                             'located_in_the_administrative_territorial_entity',
                             'occupation',
                             'place_of_birth',
                             'record_label',
                             'genre',
                             'country_of_citizenship',
                             'parent_taxon',
                             'place_of_death',
                             'inception',
                             'date_of_birth',
                             'country',
                             'headquarters_location']
    included_relations = most_common_relations[:k_most_common_only]

    dataset[:] = [x for x in dataset if x['query'].split()[0] in included_relations]
    if not os.path.exists(data_path):
        json.dump(dataset, open(data_path, 'w'))

    return dataset


def answer_lengths(data):
    answer_lengths = collections.defaultdict(int)
    for j in range(len(data)):
        length = len(data[j]['answer'].split())
        answer_lengths[length] += 1
    return answer_lengths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset_size', type=int, default=None,
                        help='If set, use a subset of data for development.')
    parser.add_argument('--k_most_common_only', type=int, default=None,
                        help='If set, only include the k most commonly occurring relation types.')
    parser.add_argument('--wikihop_version', type=str, default='1.1',
                        help='WikiHop version to use: one of {0, 1.1}.')
    parser.add_argument('--dev', nargs='?', const=True, default=False,
                        help='If True, build an index on dev data instead of train.')
    args = parser.parse_args()

    base_filename = './data/wikihop/v' + args.wikihop_version + '/'
    if args.dev:
        base_filename += 'dev'
    else:
        base_filename += 'train'

    # Calculate how many correct answers consist of 1, 2, 3, ... words
    show_answer_lengths = False
    if show_answer_lengths:
        print('Answer length - occurrence map:')
        print(answer_lengths(json.load(open(base_filename + '.json'))))

    t = time.time()
    print('Loading data...')
    # Ensure consistent IDs are used between subsets
    id_filename = base_filename + '_ids.json'
    if not os.path.exists(id_filename):
        create_full_id_map(base_filename)

    dataset = json.load(open(id_filename))

    t = print_time_taken(t)
    print('Initialising search engine...')

    index_dir = './se_index/v' + args.wikihop_version
    index_filename = os.path.join(index_dir, 'se_index')
    if args.k_most_common_only:
        split_id = '-' + str(args.k_most_common_only) + 'mc'
        index_filename += split_id
        split_filename = base_filename + '_ids' + split_id + '.json'
        dataset = filter_by_most_common(args.k_most_common_only, split_filename)
    if args.subset_size:
        dataset = dataset[:args.subset_size]
        index_filename += '_' + str(args.subset_size)

    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    if not os.path.exists(index_filename + '_vec.pkl'):
        # Create an index
        se = SearchEngine(dataset, save_index_to_path=index_filename)
    else:
        se = SearchEngine(load_from_path=index_filename)
    t = print_time_taken(t)

    if not args.k_most_common_only:
        # Test initialised search engine
        run_test_queries(se, t)


