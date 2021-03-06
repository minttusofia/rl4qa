import gc
import random
import time

from colors import color
from random import randint


def verbose_print(verbosity, verbose_level, *args, end='\n'):
    """Print *args if current verbosity level surpasses required verbosity of print statement."""
    if verbose_level >= verbosity:
        for arg in args:
            print(arg, end=' ')
        print(end, end='')


def trim_index(dataset, nouns, search_engine, keep_most=None):
    """Trim index and pre-extracted nouns to only include instances included in dataset."""
    print('Trimming index...', end=' ')
    prev_time = time.time()
    if ((search_engine is not None and len(dataset) != len(search_engine.shape_for_q))
        or (nouns is not None and len(dataset) != len(nouns))):
        if keep_most is None:
            if search_engine is not None:
                keep_most = len(dataset) > 0.5 * len(search_engine.shape_for_q)
            else:
                keep_most = len(dataset) > 0.5 * len(nouns)
        old_len = len(search_engine.shape_for_q) if search_engine is not None else len(nouns)
        # If most keys are kept, it's faster to delete unused than to copy shared
        if keep_most:
            dataset_keys = set([q['id'] for q in dataset])
            unused_index_keys = set(search_engine.shape_for_q.keys()) - dataset_keys
            if search_engine is not None:
                for k in unused_index_keys:
                    del search_engine.shape_for_q[k]
                    del search_engine.sparse_for_q[k]
                    del search_engine.vec_for_q[k]
            if nouns is not None:
                for k in unused_index_keys:
                    del nouns[k]
        else:
            if nouns is not None:
                trimmed_nouns = {}
                for q in dataset:
                    trimmed_nouns[q['id']] = nouns[q['id']]
                nouns = trimmed_nouns
            if search_engine is not None:
                trimmed_shape_for_q = {}
                trimmed_sparse_for_q = {}
                trimmed_vec_for_q = {}
                for q in dataset:
                    trimmed_shape_for_q[q['id']] = search_engine.shape_for_q[q['id']]
                    trimmed_sparse_for_q[q['id']] = search_engine.sparse_for_q[q['id']]
                    trimmed_vec_for_q[q['id']] = search_engine.vec_for_q[q['id']]

                search_engine.shape_for_q = trimmed_shape_for_q
                search_engine.sparse_for_q = trimmed_sparse_for_q
                search_engine.vec_for_q = trimmed_vec_for_q
        new_len = len(search_engine.shape_for_q) if search_engine is not None else len(nouns)
        print(old_len, '->', new_len)
    print('({} s)'.format(time.time() - prev_time))
    return nouns, search_engine


def form_query(template, subject, fg_color=None, add_question_mark=True):
    """Turn query template - subject pairs into strings."""
    if type(template) == str:
        query = template + ' ' + subject
        if fg_color is not None:
           query = color(template, fg=fg_color) + ' ' + subject
    elif len(template) > 1:  # Two-part templates expect subject to appear in the middle
        if template[1][0] == '\'':  # template contains [x]'s
            query = template[0] + ' ' + subject + template[1]
            if fg_color is not None:
                query = (color(template[0], fg=fg_color) + ' ' + subject
                         + color(template[1], fg=fg_color))
        else:
            query = template[0] + ' ' + subject + ' ' + template[1]
            if fg_color is not None:
                query = (color(template[0], fg=fg_color) + ' ' + subject + ' '
                         + color(template[1], fg=fg_color))
    else:
        query = template[0] + ' ' + subject
        if fg_color is not None:
            query = color(template[0], fg=fg_color) + ' ' + subject
    if add_question_mark:
        query += '?'
    return query


def get_document_for_query(action, subj, search_engine, question, nouns, queries_asked,
                           verbose_level=0):
    query = form_query(action, subj)
    top_idxs = search_engine.rank_docs(question.id, query, topk=len(question.supports))
    # Iterate over ranking backwards (last document is best match)
    for d in range(len(top_idxs)-1, -1, -1):
        if top_idxs[d] not in queries_asked[query]:
            top_idx = top_idxs[d]
            return top_idx, subj, query

    # If question has been asked from all documents, pick a new subject from top doc at random
    top_idx = top_idxs[-1]
    subj_t = (nouns[question.id]
              [top_idx]
              [random.randint(0, len(nouns[question.id][top_idx])-1)])
    verbose_print(2, verbose_level,
                  '  Question has been asked from all docs, continuing at', subj_t)
    return None, subj_t, None


def check_answer_confidence(answer, confidence_threshold, nouns, question_id, top_idx,
                            verbose_level=0):
    if confidence_threshold is None or answer.score > confidence_threshold:
        return answer.text.lower()
    else:
        # Pick a noun phrase at random from top document
        rand_draw = randint(0, len(nouns[question_id][top_idx])-1)
        subject = nouns[question_id][top_idx][rand_draw].lower()
        verbose_print(2, verbose_level, '  Conf', answer.score, '<', confidence_threshold,
                      ': picked', subject, '(', rand_draw, ') at random')
        return subject
