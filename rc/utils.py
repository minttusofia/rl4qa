from jack.core import QASetting


def get_rc_answer(reader, query, doc):
    answers = reader([QASetting(question=query, support=[doc])])
    return answers


def show_rc_answer(reader, query, doc):
    print('Reading Comprehension answers:')
    answers = reader([QASetting(question=query, support=[doc])])
    for a in answers:
        print(a.text, '\tscore:', a.score)


