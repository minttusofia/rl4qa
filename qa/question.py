"""Defines utility functions and classes for interacting with QA data."""


class Question:
    def __init__(self, json_item):
        self.id = json_item['id']
        self.query = json_item['query']
        self.answer = json_item['answer']
        self.candidates = json_item['candidates']
        self.candidates_lower = [c.lower() for c in self.candidates]
        self.supports = json_item['supports']

    def show_question(self):
        print(self.query)



