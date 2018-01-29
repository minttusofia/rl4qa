import collections
import glob
import nltk
import os
import pickle
import time
import spacy

from qa.utils import print_time_taken


class NounParser:
    def extract_nouns(self, sentences):
        raise NotImplementedError()


class NltkNounParser(NounParser):
    """Regex matching noun phrase parser. Takes 2-20s per WikiHop item."""

    def __init__(self):
        grammar = r"""
        NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
            {<NNP>+}                # chunk sequences of proper nouns
        """
        self.cp = nltk.RegexpParser(grammar)

    def extract_nouns(self, sentences):
        # Part of Speech tagging
        sentences = nltk.sent_tokenize(sentences)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        sentences = [nltk.pos_tag(sent) for sent in sentences]

        nouns = []
        for s in sentences:
            p = self.cp.parse(s)
            for part in p.subtrees(filter=lambda x: x.label() == "NP"):
                nouns.append(' '.join(x[0] for x in part.leaves()))
        return nouns


class SpacyNounParser(NounParser):
    """Noun phrase parser from Spacy. Takes 0.1-1s per WikiHop item."""

    def __init__(self):
        self.nlp = spacy.load('en')

    def extract_nouns(self, sentences):
        doc = self.nlp(sentences)
        return [np.text for np in doc.noun_chunks]


def load_as_one_dict(pickle_path):
    """Load all pickled dictionary objects from a .pkl file."""
    d = collections.defaultdict(list)
    with open(pickle_path, 'rb') as f:
        # .pkl file may consist of multiple objects
        while True:
            try:
                d.update(pickle.load(f))
            except EOFError:
                return d


def load_existing_nouns(nouns_path):
    """Reuse noun files created with the same noun parser and WikiHop version."""
    nouns_file = nouns_path.split('/')[-1]
    nouns_path_glob = '/'.join(nouns_path.split('/')[:-1]) + '/'  # match wikihop version exactly
    nouns_path_glob += (nouns_file.split('-')[0]  # "nouns"
                        + '-*'  # match any k_most_common
                        # match noun parser type exactly
                        + nouns_file.split('-')[-1].split('_')[0].replace('.pkl', '')
                        + '*')  # match any subset size
    print('Matching existing nouns files:', nouns_path_glob)
    similar_files = glob.glob(nouns_path_glob)
    if len(similar_files) > 0:
        print('Reusing nouns from', ', '.join(similar_files))
    existing_nouns = collections.defaultdict(list)
    for file in similar_files:
        existing_nouns.update(load_as_one_dict(file))
    return existing_nouns


def pre_extract_nouns(dataset, nouns_path, noun_parser_class=SpacyNounParser):
    """Extract noun phrases from dataset using noun_parser_class and write to nouns_path."""
    if os.path.exists(nouns_path):
        # Load existing file for this dataset
        return load_as_one_dict(nouns_path)
    else:  # If file doesn't exist
        print('\nWriting nouns to', nouns_path)
        parser = noun_parser_class()
        nouns = collections.defaultdict(list)
        existing_nouns = load_existing_nouns(nouns_path)

        print('Noun extraction format:', parser.extract_nouns(dataset[0]['supports'][0]))

        if not os.path.exists(os.path.dirname(nouns_path)):
            os.makedirs(os.path.dirname(nouns_path))

        t = time.time()
        for i in range(len(dataset)):
            item = dataset[i]
            # If entry for item exists in subset file
            if item['id'] in existing_nouns:
                nouns[item['id']] = existing_nouns[item['id']]
            else:
                for doc in item['supports']:
                    nouns[item['id']].append(parser.extract_nouns(doc))
            if (i + 1) % 10 == 0:
                print(i + 1, '/', len(dataset), end='\t')
                t = print_time_taken(t)
            if (i + 1) % 100 == 0:
                # Append to file regularly for improved fault tolerance
                pickle.dump(nouns, open(nouns_path, 'ab', pickle.HIGHEST_PROTOCOL))
                nouns = collections.defaultdict(list)
                print('Saved to', nouns_path)

        # Append remaining nouns to file
        pickle.dump(nouns, open(nouns_path, 'ab', pickle.HIGHEST_PROTOCOL))
        return load_as_one_dict(nouns_path)  # Load whole dataset

