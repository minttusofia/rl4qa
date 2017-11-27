import collections
import nltk
import os
import pickle
import time

from qa.utils import print_time_taken


class NounParser:

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


def pre_extract_nouns(dataset, stored_nouns_path):
    parser = NounParser()
    if os.path.exists(stored_nouns_path):
        return pickle.load(open(stored_nouns_path, 'rb'))
    else:
        nouns = collections.defaultdict(list)
        t = time.time()
        for i in range(len(dataset)):
            item = dataset[i]
            for doc in item['supports']:
                nouns[item['id']].append(parser.extract_nouns(doc))
            t = print_time_taken(t)
            if (i + 1) % 10 == 0:
                print(i + 1, '/', len(dataset))

        if not os.path.exists(os.path.dirname(stored_nouns_path)):
            os.makedirs(os.path.dirname(stored_nouns_path))
        pickle.dump(nouns, open(stored_nouns_path, 'wb', pickle.HIGHEST_PROTOCOL))
        return nouns



