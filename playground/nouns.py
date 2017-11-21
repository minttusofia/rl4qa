import nltk

def nounPhrases(sentences):
    # Part of Speech tagging
    sentences = nltk.sent_tokenize(sentences)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    #print(sentences)

    grammar = r"""
      NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
          {<NNP>+}                # chunk sequences of proper nouns
    """
    cp = nltk.RegexpParser(grammar)
    nouns = []
    for s in sentences:
        p = cp.parse(s)
        for part in p.subtrees(filter=lambda x: x.label() == "NP"):
            nouns.append([x[0] for x in part.leaves()])
    return nouns
