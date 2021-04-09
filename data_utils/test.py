# -*- coding: utf-8 -*-
# @Time    : 2021/4/8 下午8:25


from pypinyin import pinyin, lazy_pinyin, Style

# 不考虑多音字() 5音表示中性音
# print(pinyin('中心', style=Style.TONE3, heteronym=False, neutral_tone_with_five=True))
# print(pinyin('女性', style=Style.TONE3, heteronym=False, neutral_tone_with_five=True))

# 不考虑多音字() 5音表示中性音
print(lazy_pinyin('中心', style=Style.TONE3, neutral_tone_with_five=True))
print(lazy_pinyin('女性', style=Style.TONE3, neutral_tone_with_five=True))
print(lazy_pinyin('啦啦', style=Style.TONE3, neutral_tone_with_five=True))
