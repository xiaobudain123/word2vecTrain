# -*- coding: utf-8 -*-
import jieba.posseg as pseg
import codecs#自然语言的编码转换
from jieba import analyse

'***********************文本的关键字的抽取**********************************'

#提取关键字，使用TF-IDF来实现的
def keyword_extract(data, file_name):
   tfidf = analyse.extract_tags
   keywords = tfidf(data)
   return keywords


#提取文档的关键字
def getKeywords(docpath, savepath):
   '''
   :param docpath: 需要提取关键字的文档在哪个路径
   :param savepath: 保存关键字的文档路径
   :return:
   '''
   with codecs.open(docpath, 'r','utf-8') as docf, codecs.open(savepath, 'w','utf-8') as outf:#这里也可以直接使用open，前面不加codecs
      for data in docf:#每次读取一行
         data = data[:len(data)-1]
         keywords = keyword_extract(data, savepath)
         for word in keywords:
            outf.write(word + ' ')#每一行里面的字用空格'  '拼接
         outf.write('\n')#每一行的末尾加上\n
