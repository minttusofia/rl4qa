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
    num_total_words = sum(df.values())
    for word in df.keys():
        # idf = log(N/n_t)
        df[word] = np.log(num_total_words/df[word])
    return defaultdict(lambda: num_total_words/1., df), num_total_words


def unit_sphere(var_matrix, norm=1.0, axis=1):
    # Norm for each word embedding vector
    row_norms = np.sqrt(np.sum(np.square(var_matrix), axis=axis))
    # Divide vectors by their norms to obtain unit vectors
    scaled = var_matrix * np.expand_dims(norm / row_norms, axis=axis)
    return scaled


class GloveLookup:
    def __init__(self, path, dim, dataset, idf_from_file='rl/idf_lower_logN-n.json',
                 oov_from_file='rl/oov_embs.json'):
        self.emb_dim = dim
        # IDF weight for randomly initialised embeddings for train time OOV words (low even if rare)
        self.rand_init_idf = 1.
        # IDF weight for test time OOV words (low even if rare)
        self.oov_idf = 1.
        print('\nLoading GloVe...')
        vocab, _ = vocab_for_dataset(dataset)
        print('Train vocab length', len(vocab))
        if idf_from_file is not None and os.path.exists(idf_from_file):
            self.idf = defaultdict(float)
            print('Loaded IDF weights from', idf_from_file)
            stored_idf = json.load(io.open(idf_from_file, 'r', encoding='utf-8'))
            num_total_words, idf_weights = stored_idf[0], stored_idf[1]
            self.idf = defaultdict(lambda: num_total_words/1.)
            self.idf.update(idf_weights)
        else:
            print('Computing IDF for %i question items...' % len(dataset))
            self.idf, num_total_words = idf_for_dataset(dataset)
            idf_with_num_total = [num_total_words, self.idf]
            with io.open(idf_from_file, 'w', encoding='utf8') as f:
                json.dump(idf_with_num_total, f, ensure_ascii=False)

        self.word2idx, self.lookup = embeddings.glove.load_glove(open(path, 'rb'))
        use_existing_oov_embs = oov_from_file is not None and os.path.exists(oov_from_file)
        num_glove_words = len(self.word2idx)
        self.initialised_oov_words = {}
        if use_existing_oov_embs:
            initialised_oov = json.load(io.open(oov_from_file, 'r', encoding='utf8'))
            self.oov_words = set(initialised_oov.keys())
        else:
            oov_embs = {}
        for word in vocab:
            if word.lower() not in self.word2idx:
                self.oov_words.add(word.lower())
                idx = len(self.word2idx)
                self.word2idx[word.lower()] = idx
                if idx > len(self.lookup) - 1:
                    self.lookup.resize([self.lookup.shape[0] + 50000, self.lookup.shape[1]])
                if use_existing_oov_embs:
                    self.lookup[idx] = np.array(initialised_oov[word.lower()])
                else:
                    new_random_normal = np.random.normal(0.0, 1.0, size=[1, dim])
                    self.lookup[idx] = new_random_normal
                    oov_embs[word.lower()] = new_random_normal.tolist()
        if not use_existing_oov_embs:
            with io.open(oov_from_file, 'w', encoding='utf8') as f:
                json.dump(oov_embs, f, ensure_ascii=False)
        for word in self.initialised_oov_words:
            # Override idf dictionary default log(N/n_t)
            self.idf[word] = self.rand_init_idf

        # TODO: learn linear transformation for task
        print('Initialised', len(self.word2idx) - num_glove_words, 'new words')
        self.oov = np.zeros(dim)
        self.oov[0] = 1
        print(self.lookup[:len(self.word2idx), :].shape)
        self.lookup = unit_sphere(self.lookup[:len(self.word2idx), :])
        self.history_len = 1000
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

    def lookup_word_idf(self, word):
        if word.lower() in self.word2idx:
            return self.lookup[self.word2idx[word.lower()]] * self.idf[word.lower()]
        # OOV words should have low IDF despite being rare
        return self.oov * self.oov_idf

    def lookup_word(self, word):
        if word.lower() in self.word2idx:
            return self.lookup[self.word2idx[word.lower()]]
        return self.oov

    def lookup_doc_tf_idf(self, doc, tf):
        words = doc.split()
        return np.mean([tf[w.lower()] * self.lookup_word_idf(w) for w in words], axis=0)

    def lookup_doc_avg(self, doc):
        words = doc.split()
        return np.mean([self.lookup_word(w) for w in words], axis=0)

    def embed_state(self, s, store_naive=False):
        tfs = []
        for elem in s:
            tfs.append(defaultdict(int))
            for word in elem.split():
                tfs[-1][word.lower()] += 1
        emb_elements = [self.lookup_doc_tf_idf(s[e], tfs[e]) for e in range(len(s))]
        if np.any([type(component) == np.float64 or len(component) != self.emb_dim
                   for component in emb_elements]):
            print('State element has incorrect format:\n', s, '\n', emb_elements)
        tf_idf_state = np.stack(emb_elements, axis=0).flatten()
        if store_naive:
            naive_state = np.stack([self.lookup_doc_avg(elem) for elem in s], axis=0).flatten()
            self.avg_state_history.append(list(naive_state))
            self.avg_state_history = self.avg_state_history[-self.history_len:]
        self.state_history.append(list(tf_idf_state))
        self.state_history = self.state_history[-self.history_len:]
        self.state_str_history.append(s)
        self.state_str_history = self.state_str_history[-self.history_len:]
        return tf_idf_state
