import uuid
import json
import numpy as np
import pickle
import time

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


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


def print_time_taken(prev_t):
    new_t = time.time()
    print(' ' + str(new_t - prev_t) + ' s')
    return new_t


if __name__ == '__main__':
    assign_new_ids = False
    create_new_index = True
    use_subset = True
    subset_size = 100
    t = time.time()
    print('Loading data...')
    if assign_new_ids:
        with open('../data/wikihop/train.json') as f:
            dataset = json.load(f)
        t = print_time_taken(t)
        print('Assigning new ids...')
        for q in dataset:
            q['id'] = str(uuid.uuid4())
        with open('../data/wikihop/train_ids.json', 'w') as f:
            json.dump(dataset, f)
    else:
        dataset = json.load(open('../data/wikihop/train_ids.json'))
    t = print_time_taken(t)
    print('Initialising search engine...')
    index_filename = "se_index"
    if use_subset:
        dataset = dataset[:subset_size]
        index_filename += '_' + str(subset_size)
    if create_new_index:
        se = SearchEngine(dataset[:subset_size], save_index_to_path=index_filename)
    else:
        se = SearchEngine(load_from_path=index_filename)
    t = print_time_taken(t)
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
    t = print_time_taken(t)


