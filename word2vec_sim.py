# -*- coding: utf-8 -*-
import codecs
import numpy
import gensim
import numpy as np
from GensimWikiWordVector.word2vec训练与相似度计算.keyword_extract import *

wordvec_size=400 #此处的需要与模型的训练时长度一致

#获取字符串string中char的位置列表
def get_char_pos(string,char):#string表示字符串， char这里表示的空格
    chPos=[]
    try:
        chPos=list(((pos) for pos,val in enumerate(string) if(val == char)))#列表推导式
    except:
        pass
    return chPos


#从word2vec模型中获取关键词向量
def word2vec(file_name,model):#model是训练好的模型，file_name存储关键字的文件
    with codecs.open(file_name, 'r','utf-8') as f:
        word_vec_all = numpy.zeros(wordvec_size)#零向量，维度需要和词汇的维度一样
        for data in f:#一行行的读取
            #获取空格位置，因为一行的关键词，包含了好多空格
            space_pos = get_char_pos(data, ' ')
            #获取第一个词语
            first_word=data[0:space_pos[0]]
            #我们使用的语料库不是很大，无法包含所有的汉语词典
            #因此需要判断是否包含该词语，否则可能报错
            if model.__contains__(first_word):
                #求词向量的总和，一次计算文档/网页的相似度
                word_vec_all= word_vec_all+model[first_word]

            for i in range(len(space_pos) - 1):
                word = data[space_pos[i]:space_pos[i + 1]]
                if model.__contains__(word):
                    word_vec_all = word_vec_all+model[word]
        return word_vec_all

#计算向量的相似度
def simlarityCalu(vector1,vector2):
    vector1Mod=np.sqrt(vector1.dot(vector1))#dot表示点积 sqrt开方
    vector2Mod=np.sqrt(vector2.dot(vector2))
    if vector2Mod!=0 and vector1Mod!=0:
        #求余弦相识度方法（就是向量夹角的余弦）
        simlarity=(vector1.dot(vector2))/(vector1Mod*vector2Mod)
    else:
        simlarity=0
    return simlarity

if __name__ == '__main__':
    model = gensim.models.Word2Vec.load('G:/alldatas/维基百科2019语料库/wiki.model')
    p1 = './data/J2EE框架.txt'
    p2 = './data/J2EE框架_mooc.txt'
    # p1 = './data/P1.txt'
    # p2 = './data/P2.txt'
    p1_keywords = './data/P1_keywordsJ2EE框架.txt'
    # p1_keywords = './data/P1_keywords.txt'
    # p2_keywords = './data/P2_keywords.txt'
    p2_keywords = './data/P2_keywordsJ2EE框架_mooc.txt'

    #1.关键字的提取
    getKeywords(p1, p1_keywords)
    getKeywords(p2, p2_keywords)

    #2.关键词的向量化
    p1_vec=word2vec(p1_keywords,model)
    p2_vec=word2vec(p2_keywords,model)

    #计算向量的余弦相似度
    print(simlarityCalu(p1_vec,p2_vec))#0.6588375110058241
