# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    
"""
import re

__author__ = "freemso"

ask_when = [re.compile(s) for s in ["什么时[候间]", "哪一?[年月日天]", "何时", "几[年月日天]", "多少[年月日天]", "多久", "时间？"]]
ask_where = [re.compile(s) for s in ["多[远近]", "什么地[方点]", "在哪", "何地", "位于哪", "哪里"]]
ask_why = [re.compile(s) for s in ["为什么", "为何"]]
ask_what = [re.compile(s) for s in ["[是指]什么", "哪[个所本]", "是？", "叫做什么", "什么是", "以什么为", "干什么", "解释.+"]]
ask_list = [re.compile(s) for s in ["[哪那][些几两三四五]个?", "有？", "有$", "有些?什么"]]
ask_who = [re.compile(s) for s in ["谁", "哪人"]]
ask_how = [re.compile(s) for s in ["怎么?样", "怎么", "如何"]]
ask_if = [re.compile(s) for s in ["是否", "是不是", "有没有", "[有是].*吗？?", "会不会"]]
ask_how_many = [re.compile(s) for s in ["有多.+", "几[个层家种人]", "多[少大宽长粗]", "什么程度", "什么数量"]]
other = [re.compile(s) for s in [
    "[哪那]",
    "什么",
    "何",
    "为？",
    "[？\\?]",
    # Below is wrong
    "中华人民共和国教育部",
    "中国人民解放军军事经济学院",
    "安乐公主",
    "浊度仪",
]]


def main():
    total = 0
    with open("data/BoP2017_DBAQ_dev_train_data/BoP2017-DBQA.train.txt") as in_file:
        for _ in in_file:
            total += 1
    with open("data/BoP2017_DBAQ_dev_train_data/BoP2017-DBQA.train.txt") as in_file:
        when_count = 0
        where_count = 0
        why_count = 0
        what_count = 0
        who_count = 0
        how_count = 0
        list_count = 0
        how_many_count = 0
        if_count = 0
        other_count = 0
        i = 0
        for line in in_file:
            i += 1
            label, question, answer = line.strip().split("\t")
            # Match the question to certain class using regular expression
            if any(re.search(p, question) for p in ask_when):
                when_count += 1
            elif any(re.search(p, question) for p in ask_where):
                where_count += 1
            elif any(re.search(p, question) for p in ask_list):
                list_count += 1
            elif any(re.search(p, question) for p in ask_what):
                what_count += 1
            elif any(re.search(p, question) for p in ask_who):
                who_count += 1
            elif any(re.search(p, question) for p in ask_why):
                why_count += 1
            elif any(re.search(p, question) for p in ask_how_many):
                how_many_count += 1
            elif any(re.search(p, question) for p in ask_how):
                how_count += 1
            elif any(re.search(p, question) for p in ask_if):
                if_count += 1
            elif any(re.search(p, question) for p in other):
                other_count += 1
            else:
                if int(label) == 1:
                    print(str(i) + "/" + str(total) + "\t" + question + "\t" + answer)
                    exit(-1)
        print("when: {}\n"
              "where: {}\n"
              "why: {}\n"
              "what: {}\n"
              "who: {}\n"
              "how: {}\n"
              "list: {}\n"
              "how_many: {}\n"
              "if: {}\n"
              "other: {}\n".format(when_count,
                                   where_count,
                                   why_count,
                                   what_count,
                                   who_count,
                                   how_count,
                                   list_count,
                                   how_many_count,
                                   if_count,
                                   other_count))


if __name__ == '__main__':
    main()
