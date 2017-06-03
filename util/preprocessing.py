#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" pre-processing util for chinese nlp task """

__author__ = 'freemso'

import re

punc_str = """“”＋＝＆:|,，.。][}{：;；'\"、()（）`·《》
            <>?？/+-=—_\\~～!#$%^&*！＠@￥……×\n\t\r\b 　．＊【 】"""


def is_punctuation_mark(c):
    """ check whether a character is a punctuation mark """
    return c in punc_str


def rm_paren_content(s):
    """ remove the parentheses and the contents within """
    pattern = re.compile('[(|（].*?[)|）]')
    return pattern.sub('', s)


def is_chinese_char(c):
    """ check whether a character is a chinese character """
    if u'\u4e00' <= c <= u'\u9fff':
        return True
    else:
        return False


def has_non_chinese_char(s):
    """ check whether a string contains non chinese characters """
    for c in s:
        if not is_chinese_char(c):
            return True
    return False


def rm_non_chinese_char(s):
    """ remove all non chinese characters """
    return "".join([c for c in s if is_chinese_char(c)])


def to_sentences(doc):
    """ cut the chinese doc into list of sentences """
    sents = []
    buf = ""
    for char in doc:
        if char in "，。！？；;,.?!":
            if len(buf) > 1:
                sents.append(buf.strip())
            buf = ""
        else:
            buf += char
    if len(buf) > 1:
        sents.append(buf.strip())
    return sents
