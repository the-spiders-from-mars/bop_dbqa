# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Question and answer pair structure
"""

__author__ = "freemso"


class Doc(object):
    def __init__(self, question):
        self.pos_answers = []
        self.question = question
        self.neg_answers = []

    def add_ans(self, answer, is_pos):
        if is_pos:
            self.pos_answers.append(answer)
        else:
            self.neg_answers.append(answer)
