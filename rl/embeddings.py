import io
import json
import numpy as np
import os
import unicodecsv

from collections import defaultdict
from jack.io import embeddings


def vocab_for_dataset(dataset):
    vocab = set()
    counts = defaultdict(int)
    for question in dataset:
        for doc in question['supports']:
            for w in doc.split():
                vocab.add(w)
                counts[w] += 1
    return vocab, counts


def idf_for_dataset(dataset, use_lowercase=True):
    df = defaultdict(int)
    for question in dataset:
        for doc in question['supports'] + [question['query']]:
            for word in doc.split():
                if use_lowercase:
                    word = word.lower()
                df[word] += 1
    '''for template in templates:
        for part in template:
            for word in part.split():
                if use_lowercase:
                    word = word.lower()
                df[word] += 1'''
    total_num_words = sum(df.values())
    for word in df.keys():
        # idf = log(N/n_t)
        df[word] = np.log(total_num_words/df[word])
    return defaultdict(lambda: total_num_words/1., df), total_num_words


def unit_sphere(var_matrix, norm=1.0, axis=1):
    # Norm for each word embedding vector
    row_norms = np.sqrt(np.sum(np.square(var_matrix), axis=axis))
    # Divide vectors by their norms to obtain unit vectors
    scaled = var_matrix * np.expand_dims(norm / row_norms, axis=axis)
    return scaled


class GloveLookup:
    def __init__(self, path, dim, dataset, idf_from_file='rl/idf_lower.json'):
        self.emb_dim = dim
        print('Loading GloVe...')
        vocab, _ = vocab_for_dataset(dataset)
        print('Train vocab length', len(vocab))
        if idf_from_file is not None and os.path.exists(idf_from_file):
            self.idf = defaultdict(float)
            print('Loaded IDF weights from', idf_from_file)
            stored_idf = json.load(io.open(idf_from_file, 'r', encoding='utf-8'))
            num_total_words, idf_weights = stored_idf[0], stored_idf[1]
            self.idf = defaultdict(lambda: np.log(num_total_words/1))
            self.idf.update(idf_weights)
        else:
            print('Computing IDF for %i question items...' % len(dataset))
            self.idf, num_total_words = idf_for_dataset(dataset)
            idf_with_num_total = [num_total_words, self.idf]
            with io.open(idf_from_file, 'w', encoding='utf8') as f:
                json.dump(idf_with_num_total, f, ensure_ascii=False)

        self.word2idx, self.lookup = embeddings.glove.load_glove(open(path, 'rb'))
        num_OOV_words = 0
        for word in vocab:
            if word.lower() not in self.word2idx:
                idx = len(self.word2idx)
                num_OOV_words += 1
                self.word2idx[word.lower()] = idx
                if idx > len(self.lookup) - 1:
                    self.lookup.resize([self.lookup.shape[0] + 50000, self.lookup.shape[1]])
                self.lookup[idx] = np.random.normal(0.0, 1.0, size=[1, dim])
        # TODO: learn linear transformation for task
        print('Initialised', num_OOV_words, 'new words')
        self.OOV = np.zeros(dim)
        self.OOV[0] = 1
        print(self.lookup[:len(self.word2idx), :].shape)
        self.lookup = unit_sphere(self.lookup[:len(self.word2idx), :])
        self.state_str_history = []
        self.state_history = []
        self.avg_state_history = []
        self.dataset_len = len(dataset)

    def save_history_to_csv(self):
        print('History length', len(self.state_history), len(self.state_str_history),
              len(self.avg_state_history))

        with open('rl/tf_idf_states-%i.csv' % self.dataset_len, 'wb') as f:
            writer = unicodecsv.writer(f)
            for s in self.state_history:
                writer.writerow(list(s))
        with open('rl/avg_states-%i.csv' % self.dataset_len, 'wb') as f:
            writer = unicodecsv.writer(f)
            for s in self.avg_state_history:
                writer.writerow(list(s))
        with open('rl/state-strs-%i.txt' % self.dataset_len, 'w') as f:
            for s in self.state_str_history:
                f.write(str(s) + '\n')

    def lookup_word(self, word):
        if word.lower() in self.word2idx:
            return self.lookup[self.word2idx[word.lower()]]
        return self.OOV

    def lookup_doc_tf_idf(self, doc, tf):
        words = doc.split()
        '''for w in words:
            if w.lower() not in tf:
                print(w.lower(), 'not in tf')
            if w.lower() not in self.idf:
                print(doc)
                print(w.lower(), 'not in idf')'''
        return np.mean([tf[w.lower()] * self.idf[w.lower()] * self.lookup_word(w) for w in words],
                       axis=0)

    def lookup_doc_avg(self, doc):
        words = doc.split()
        return np.mean([self.lookup_word(w) for w in words], axis=0)

    def embed_state(self, s):
        tfs = []
        for elem in s:
            tfs.append(defaultdict(int))
            for word in elem.split():
                tfs[-1][word.lower()] += 1
        tf_idf_state = np.stack([self.lookup_doc_tf_idf(s[e], tfs[e]) for e in range(len(s))],
                                axis=0).flatten()
        naive_state = np.stack([self.lookup_doc_avg(elem) for elem in s], axis=0).flatten()
        self.state_history.append(list(tf_idf_state))
        self.avg_state_history.append(list(naive_state))
        self.state_str_history.append(s)
        return tf_idf_state
