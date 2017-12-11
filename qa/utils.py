import time

from colors import color

STOP_WORDS = ['a', 'an', 'is', 'of', 'the']


def print_time_taken(prev_t):
    new_t = time.time()
    print(' ' + str(new_t - prev_t) + ' s')
    return new_t


def print_colored_doc(doc, query=None, nouns=None):
    if query is None and nouns is None:
        print(doc)
    else:
        noun_appearances = []
        if nouns is not None:
            for noun in nouns:
                appearances = phrase_appearances_in_doc(noun, doc)
                for a in appearances:
                    noun_appearances.extend(range(a[0], a[1]))
        for w in range(len(doc.split())):
            word = doc.split()[w]
            if word.lower() in query.lower().split() and word.lower() not in STOP_WORDS:
                print(color(word, fg='blue'), end=' ')
            elif w in noun_appearances and word.lower() not in STOP_WORDS:
                print(color(word, fg='red'), end=' ')
            else:
                print(word, end=' ')
        print()


def phrase_appearances_in_doc(phrase, doc):
    """Find phrase in document and return the start (incl.) and end (excl.) word indices, or None.
    """
    appearances = []
    phrase_words = phrase.lower().split()
    sent_words = doc.lower().split()
    if len(phrase_words) < 1 or len(sent_words) < 1:
        return appearances
    for s in range(len(sent_words)):
        if phrase_words[0] == sent_words[s]:
            for p in range(1, len(phrase_words)):
                if s + p >= len(sent_words) or phrase_words[p] != sent_words[s + p]:
                    break
                elif p == len(phrase_words)-1:
                    appearances.append((s, s + p + 1))
    return appearances


def phrase_appears_in_doc(phrase, doc):
    """Return True if a single or multi-word phrase appears in the document."""
    if type(phrase) == list:
        print('----', phrase)
    phrase_words = phrase.lower().split()
    sent_words = doc.lower().split()
    if len(phrase_words) < 1 or len(sent_words) < 1:
        return False
    for s in range(len(sent_words)):
        if phrase_words[0] == sent_words[s]:
            for p in range(1, len(phrase_words)):
                if s + p >= len(sent_words) or phrase_words[p] != sent_words[s + p]:
                    break
                elif p == len(phrase_words)-1:
                    return True
    return False


