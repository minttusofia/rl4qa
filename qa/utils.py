import time

from colors import color

STOP_WORDS = ['a', 'an', 'is', 'of', 'the']


def print_time_taken(prev_t):
    new_t = time.time()
    print(' ' + str(new_t - prev_t) + ' s')
    return new_t


def print_colored_doc(doc, query=None, nouns=None, query_color='blue', noun_color='red'):
    """Print document with colored query terms and noun appearances."""
    if query is None and nouns is None:
        print(doc)
    else:
        # TODO: Not all docs have separated punctuation
        # doc = doc.replace('. ', ' . ') -> doesn't work on already separated punctuation
        # doc = doc.replace(', ', ' , ')
        if query is None:
            query = ''
        noun_appearances = []
        if nouns is not None:
            for noun in nouns:
                appearances = phrase_appearances_in_doc(noun, doc)
                for a in appearances:
                    noun_appearances.extend(range(a[0], a[1]))

        contiguous_sequence = False
        for w in range(len(doc.split())):
            missing_dot = False
            missing_comma = False
            word = doc.split()[w]
            if len(word) > 1 and word[-1] == '.':
                word = word[:-1]
                missing_dot = True
            if len(word) > 1 and word[-1] == ',':
                word = word[:-1]
                missing_comma = True
            if word.lower() in query.lower().split() and (
                   contiguous_sequence or word.lower() not in STOP_WORDS):
                print(color(word, fg=query_color), end=' ')
                contiguous_sequence = True
            elif w in noun_appearances and (contiguous_sequence or word.lower() not in STOP_WORDS):
                print(color(word, fg=noun_color), end=' ')
                contiguous_sequence = True
            else:
                print(word, end=' ')
                contiguous_sequence = False
            if missing_dot:
                print('.', end=' ')
            if missing_comma:
                print(',', end=' ')
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
            for p in range(len(phrase_words)):
                if p == len(phrase_words)-1:
                    appearances.append((s, s + p + 1))
                elif s + p >= len(sent_words) or phrase_words[p] != sent_words[s + p]:
                    break
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
        for p in range(len(phrase_words)):
            if p == len(phrase_words)-1:
                return True
            if s + p >= len(sent_words) or phrase_words[p] != sent_words[s + p]:
                break
    return False


