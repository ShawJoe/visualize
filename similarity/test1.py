'''
Created on 2018年1月13日

@author: Shaw Joe
'''
import gensim
from gensim import corpora, models, similarities
import traceback


documents = [ "Shipment of gold damaged in a fire",
              "Delivery of silver arrived in a silver truck",
              "Shipment of gold arrived in a truck"]

'''
@:return:texts是token_list,只要我生成了token_list，给它就行了
'''
def pre_process( documents ):
    try:
        documents_token_list = [ [word for word in document.lower().split() ] for document in documents ]
        print("[INFO]: pre_process is finished!")
        return documents_token_list
    except:
        print(traceback.print_exc())

'''
这个函数是比较通用的，可以跟我自己写的结合。
这个是根据document[ token_list ]来训练tf_idf模型的
@texts: documents = [ document1, document2, ... ] document1 = token_list1
@return: dictionary 根据texts建立的vsm空间，并且记录了每个词的位置，和我的实现一样，对于vsm空间每个词，你要记录他的位置。否则，文档生成vsm空间的时候，每个词无法找到自己的位置
@return: corpus_idf 每篇document在vsm上的tf-idf表示.但是他的输出和我的不太一样，我的输出就是单纯的vsm空间中tf-idf的值，但是它的空间里面不是。还有位置信息在。并且输出的时候，看到的好像没有值为0的向量，但是vsm向量的空间是一样的。所以，我觉得应该是只输出了非0的。

这两个返回值和我的都不一样，因为字典(vsm)以及corpus_idf(vsm)都输出了位置信息。
但是这两个信息，可以快速生成lda和lsi模型

'''
def tf_idf_trainning(documents_token_list):
    try:
        # 将所有文章的token_list映射为 vsm空间
        dictionary = corpora.Dictionary(documents_token_list)
        # 每篇document在vsm上的tf表示
        corpus_tf = [ dictionary.doc2bow(token_list) for token_list in documents_token_list ]
        # 用corpus_tf作为特征，训练tf_idf_model
        tf_idf_model = models.TfidfModel(corpus_tf)
        # 每篇document在vsm上的tf-idf表示
        corpus_tfidf = tf_idf_model[corpus_tf]
        print("[INFO]: tf_idf_trainning is finished!")
        return dictionary, corpus_tf, corpus_tfidf
    except:
        print(traceback.print_exc())

def lsi_trainning( dictionary, corpus_tfidf, K ):
    try:
        # 用tf_idf作为特征，训练lsi模型
        lsi_model = models.LsiModel( corpus_tfidf, id2word=dictionary, num_topics = K )
        # 每篇document在K维空间上表示
        corpus_lsi = lsi_model[corpus_tfidf]
        print("[INFO]: lsi_trainning is finished!")
        return lsi_model, corpus_lsi
    except:
        print(traceback.print_exc())

def lda_trainning( dictionary, corpus_tfidf, K ):
    try:
        # 用corpus_tf作为特征，训练lda_model
        lda_model = models.LdaModel( corpus_tfidf, id2word=dictionary, num_topics = K )
        # 每篇document在K维空间上表示
        corpus_lda = lda_model[corpus_tfidf]
        for aa in corpus_lda:
            print(aa)
        print("[INFO]: lda_trainning is finished!")
        return lda_model, corpus_lda
    except:
        print(traceback.print_exc())

def similarity( query, dictionary, corpus_tf, lda_model ):
    try:
        # 建立索引
        index = similarities.MatrixSimilarity( lda_model[corpus_tf] )
        # 在dictionary建立query的vsm_tf表示
        query_bow = dictionary.doc2bow( query.lower().split() )
        # 查询在K维空间的表示
        query_lda = lda_model[query_bow]
        # 计算相似度
        simi = index[query_lda]
        query_simi_list = [ item for _, item in enumerate(simi) ]
        print(query_simi_list)
    except:
        print(traceback.print_exc())

documents_token_list = pre_process(documents)
dict, corpus_tf, corpus_tfidf = tf_idf_trainning(documents_token_list)
#lsi_trainning(corpus_tfidf, dict, 2)
lda_model, corpus_lda = lda_trainning(dict, corpus_tfidf, 2)

similarity( "Shipment of gold arrived in a truck", dict, corpus_tf, lda_model )

def main():
    pass

if __name__ == '__main__':
    main()