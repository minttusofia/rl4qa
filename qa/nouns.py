import collections
import json
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
                nouns.append([x[0] for x in part.leaves()])
        return nouns


class SpacyNounParser(NounParser):
    """Noun phrase parser from Spacy. Takes 0.1-1s per WikiHop item."""

    def __init__(self):
        self.nlp = spacy.load('en')

    def extract_nouns(self, sentences):
        doc = self.nlp(sentences)
        return [str(np.text) for np in doc.noun_chunks]


def pre_extract_nouns(dataset, stored_nouns_path=None, noun_parser_class=None):
    if noun_parser_class is None:
        noun_parser_class = SpacyNounParser
    parser = noun_parser_class()
    if stored_nouns_path is not None and os.path.exists(stored_nouns_path):
        return pickle.load(open(stored_nouns_path, 'rb'))
    else:
        nouns = collections.defaultdict(list)
        t = time.time()
        for i in range(len(dataset)):
            item = dataset[i]
            for doc in item['supports']:
                nouns[item['id']].append(parser.extract_nouns(doc))
            if (i + 1) % 10 == 0:
                print(i + 1, '/', len(dataset), end='\t')
                t = print_time_taken(t)
            # TODO: regularly append results to file

        if stored_nouns_path is not None:
            if not os.path.exists(os.path.dirname(stored_nouns_path)):
                os.makedirs(os.path.dirname(stored_nouns_path))

            pickle.dump(nouns, open(stored_nouns_path, 'wb', pickle.HIGHEST_PROTOCOL))
        return nouns


if __name__ == '__main__':
    subset_size = None
    subset_id = '-6mc'
    stored_nouns_path = 'nouns/nouns' + subset_id
    use_nltk = False
    if use_nltk:
        np_class = NltkNounParser
        stored_nouns_path += '-nltk'
    else:
        np_class = SpacyNounParser

    if subset_size is not None:
        stored_nouns_path += '_' + str(subset_size)
        data = json.load(open('./data/wikihop/train_ids.json'))[:subset_size]
    else:
        data = json.load(open('./data/wikihop/train_ids.json'))

    nouns = pre_extract_nouns(data, noun_parser_class=np_class,
                              stored_nouns_path=stored_nouns_path + '.pkl')
    if subset_size <= 500:
        print(nouns)

