import gc


def trim_index(dataset, nouns, search_engine):
    """Trim index and pre-extracted nouns to only include instances included in dataset."""
    one_type_nouns = {}
    one_type_shape_for_q = {}
    one_type_sparse_for_q = {}
    one_type_vec_for_q = {}
    print('Trimming index...')
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
    print('Finished trimming')
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
