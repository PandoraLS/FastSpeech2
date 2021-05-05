# -*- coding: utf-8 -*-
# @Time    : 2021/5/5 下午2:10


import re, string
from zhon.hanzi import punctuation
line = "测试。。去除！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠.,.,｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏标点。。，，！"
res = re.sub(r"[%s]+" %punctuation, "", line)
print (res) # 去除掉所有中文标点后

# 去除英文标点
# out = s.translate(None, string.punctuation)


exclude = set(string.punctuation)
out = ''.join(ch for ch in res if ch not in exclude)
print(out)