import gc
import random
import time


def trim_index(dataset, nouns, search_engine, keep_most=False):
    """Trim index and pre-extracted nouns to only include instances included in dataset."""
    print('Trimming index...')
    prev_time = time.time()
    # If most keys are kept, it's faster to delete unused than to copy shared
    if keep_most:
        dataset_keys = set([q['id'] for q in dataset])
        unused_index_keys = set(search_engine.shape_for_q.keys()) - dataset_keys
        for k in unused_index_keys:
            del search_engine.shape_for_q[k]
            del search_engine.sparse_for_q[k]
            del search_engine.vec_for_q[k]
            del nouns[k]
    else:
        one_type_nouns = {}
        one_type_shape_for_q = {}
        one_type_sparse_for_q = {}
        one_type_vec_for_q = {}
        for q in dataset:
            one_type_nouns[q['id']] = nouns[q['id']]
            one_type_shape_for_q[q['id']] = search_engine.shape_for_q[q['id']]
            one_type_sparse_for_q[q['id']] = search_engine.sparse_for_q[q['id']]
            one_type_vec_for_q[q['id']] = search_engine.vec_for_q[q['id']]

        nouns = one_type_nouns
        search_engine.shape_for_q = one_type_shape_for_q
        search_engine.sparse_for_q = one_type_sparse_for_q
        search_engine.vec_for_q = one_type_vec_for_q
    gc.collect()
    print('Finished trimming ({} s)'.format(time.time() - prev_time))
    return nouns, search_engine


def form_query(template, subject):
    """Turn query template - subject pairs into strings."""
    if type(template) == str:
        query = template + ' ' + subject + '?'
    elif len(template) > 1:  # Two-part templates expect subject to appear in the middle
        if template[1][0] == '\'':  # template contains [x]'s
            query = template[0] + ' ' + subject + template[1] + '?'
        else:
            query = template[0] + ' ' + subject + ' ' + template[1] + '?'
    else:
        query = template[0] + ' ' + subject + '?'
    return query


def get_document_for_query(action, subj, search_engine, question, nouns, queries_asked):
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
    return None, subj_t, None