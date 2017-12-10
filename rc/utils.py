from jack.core import QASetting


def get_rc_answers(reader, queries, documents):
    if type(queries) != list:
        return reader([QASetting(question=queries, support=[documents])])
    questions = []
    for q in range(len(queries)):
        questions.append(QASetting(question=queries[q], support=[documents[q]]))
    answers = reader(questions)
    return answers


def show_rc_answers(reader, queries, documents):
    print('Reading Comprehension answers:')
    if type(queries) != list:
        answers = reader([QASetting(question=queries, support=[documents])])
        for a in answers:
            print(a.text, '\tscore:', a.score)
    else:
        questions = []
        for q in range(len(queries)):
            questions.append(QASetting(question=queries[q], support=[documents]))
        answers = reader(questions)
        for q in answers:
            for a in q:
                print(a.text, '\tscore:', a.score)
            print()
