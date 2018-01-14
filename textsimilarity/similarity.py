#-*- coding:utf-8
'''
Created on 2018年1月13日

@author: Shaw Joe
'''

'''
similarity.py
这个文件的作用是训练后的的lda模型，对语料进行相似度的计算

'''
from gensim import corpora, models, similarities
import traceback

'''
这个函数没有用到
'''
# 基于lda模型的相似度计算
def lda_similarity( query_token_list, dictionary, corpus_tf, lda_model ):
    try:
        # 建立索引
        index = similarities.MatrixSimilarity( lda_model[corpus_tf] )
        # 在dictionary建立query的vsm_tf表示
        query_bow = dictionary.doc2bow( query_token_list )
        # 查询在K维空间的表示
        query_lda = lda_model[query_bow]
        # 计算相似度
        # simi保存的是 query_lda和corpus_lda的相似度
        simi = index[query_lda]
        query_simi_list = [ item for _, item in enumerate(simi) ]
        return query_simi_list
    except:
        print(traceback.print_exc())

'''
功能：语聊基于lda模型的相似度计算
@ corpus_tf:语聊的vsm_tf表示
@ lda_model:训练好的lda模型
'''
def lda_similarity_corpus( corpus_tf, lda_model ):
    try:
        # 语料库相似度矩阵
        lda_similarity_matrix = []
        # 建立索引
        index = similarities.MatrixSimilarity( lda_model[corpus_tf] )
        # 计算相似度
        for query_bow in corpus_tf:
            # K维空间表示
            query_lda = lda_model[query_bow]
            # 计算相似度
            simi = index[query_lda]
            # 保存
            query_simi_list = [item for _, item in enumerate(simi)]
            lda_similarity_matrix.append(query_simi_list)
        print("[INFO]:lda_similarity_corpus is finished!")
        return lda_similarity_matrix
    except:
        print(traceback.print_exc())


def main():
    pass

if __name__ == '__main__':
    main()