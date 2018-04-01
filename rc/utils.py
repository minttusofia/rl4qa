import pickle
import time

from jack.core import QASetting
from jack.core.data_structures import Answer


def get_rc_answers(reader, queries, documents):
    if type(queries) != list:
        answer = reader([QASetting(question=queries, support=[documents])])
        if type(answer[0]) == list:
            answer = answer[0]
        return answer
    questions = []
    for q in range(len(queries)):
        questions.append(QASetting(question=queries[q], support=[documents[q]]))
    answers = reader(questions)
    return answers


def get_cached_rc_answers(reader, queries, documents, redis_server, q_ids=None, doc_idxs=None):
    t = time.time()
    used_cache = False
    if type(queries) != list:
        if q_ids is not None:
            answer = redis_server.get(pickle.dumps((queries, q_ids, doc_idxs)))
        else:  # use full document as index
            answer = redis_server.get(pickle.dumps((queries, documents)))
        if answer is None or type(pickle.loads(answer)) is not tuple:
            answer = get_rc_answers(reader, queries, documents)
            if type(answer[0]) == list:
                answer = answer[0]
            if q_ids is not None:
                redis_server.set(pickle.dumps((queries, q_ids, doc_idxs)),
                                 pickle.dumps((answer[0].text, answer[0].score)))
            else:
                redis_server.set(pickle.dumps((queries, documents)),
                                 pickle.dumps((answer[0].text, answer[0].score)))

        else:
            answer = pickle.loads(answer)
            if type(answer[0]) == list:
                answer = answer[0]
            answer = [Answer(text=answer[0], score=answer[1])]
            used_cache = True

        return list(answer), used_cache

    unanswered_queries = queries
    unanswered_documents = doc_idxs
    answers = [None for _ in range(len(queries))]
    for q in range(len(queries)):
        if q_ids is not None:
            answer = redis_server.get(pickle.dumps((queries[q], q_ids[q], doc_idxs[q])))
        else:  # use full document as index
            answer = redis_server.get(pickle.dumps((queries[q], documents[q])))
        if answer is not None:
            print('answer found in cache')
            answers[q] = answer
            unanswered_queries[q] = None
            unanswered_documents[q] = None
    answers_to_compute = get_rc_answers(reader,
                                        [q for q in unanswered_queries if q is not None],
                                        [d for d in unanswered_documents if d is not None])
    a_i = 0
    for a in range(len(answers)):
        if answers[a] is None:
            print('answer not found in cache')
            answers[a] = answers_to_compute[a_i]
            redis_server.set(pickle.dumps((queries[a], documents[a])),
                             pickle.dumps(answers[a].text, answers[a].score))
            a_i += 1
    print(time.time() - t)
    return answers


def show_rc_answers(reader, queries, documents):
    print('Reading Comprehension answers:')
    if type(queries) != list:
        answers = reader([QASetting(question=queries, support=[documents])])
        for a in answers:
            if type(a) == list:
                a = a[0]
            print(a.text, '\tscore:', a.score)
    else:
        questions = []
        for q in range(len(queries)):
            questions.append(QASetting(question=queries[q], support=[documents]))
        answers = reader(questions)
        for q in answers:
            for a in q:
                if type(a) == list:
                    a = a[0]
                print(a.text, '\tscore:', a.score)
            print()
