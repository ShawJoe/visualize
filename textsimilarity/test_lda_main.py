#-*- coding:utf-8
'''
Created on 2018年1月13日

@author: Shaw Joe
'''
'''
test_lda_main.py
这个文件的作用是汇总前面各部分代码，对文档进行基于lda的相似度计算

'''

from textsimilarity.lda_model import get_lda_model
from textsimilarity.similarity import lda_similarity_corpus
from textsimilarity.save_result import save_similarity_matrix
import traceback

INPUT_PATH = ""
OUTPUT_PATH = "../file/test/lda_simi_matrix.txt"

def main():
    try:
        # 语料
        documents = ["Shipment of gold damaged in a fire",
                     "Delivery of silver arrived in a silver truck",
                     "Shipment of gold arrived in a truck"]
        # 训练lda模型
        K = 2 # number of topics
        lda_model, _, _,corpus_tf, _ = get_lda_model(documents, K)
        # 计算语聊相似度
        lda_similarity_matrix = lda_similarity_corpus( corpus_tf, lda_model )
        # 保存结果
        save_similarity_matrix( lda_similarity_matrix, OUTPUT_PATH )
    except:
        print(traceback.print_exc())

if __name__ == '__main__':
    main()