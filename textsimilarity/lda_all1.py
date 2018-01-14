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
def pre_process( documents ):#实现分词，按空格进行分割
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
        dictionary = corpora.Dictionary(documents_token_list)#将所有的文本先变成集合，然后再每个单词分配一个id,dictionary[0]对应第一个词
        for d in dictionary:#输出预料集中所有的单词
            print(d,dictionary[d])
        # 每篇document在vsm上的tf表示
        corpus_tf = [ dictionary.doc2bow(token_list) for token_list in documents_token_list ]#每个句子中每个词的词频
        print(corpus_tf)#输出：[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],...]
        # 用corpus_tf作为特征，训练tf_idf_model
        tf_idf_model = models.TfidfModel(corpus_tf)#TF-IDF模型
        # 每篇document在vsm上的tf-idf表示
        corpus_tfidf = tf_idf_model[corpus_tf]
        for c in corpus_tfidf:#逐句输出每个词的tfidf值
            print(c)#输出[(0, 0.5), (2, 0.5), (9, 0.5), (10, 0.5)]等，每句一个数组
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
        lda_model = models.LdaModel( corpus_tfidf, id2word=dictionary, num_topics = K )#根据TF-IDF进行计算
        # 每篇document在K维空间上表示
        corpus_lda = lda_model[corpus_tfidf]#corpus_tfidf是每篇document在vsm上的tf-idf表示
        for aa in corpus_lda:#在二维空间上的表示
            print(aa)#输出两个元组，在每一维空间上对应一个值，输出：[(0, 0.36092998872839893), (1, 0.63907001127160101)]
        print("[INFO]: lda_trainning is finished!")
        return lda_model, corpus_lda
    except:
        print(traceback.print_exc())

def similarity( query, dictionary, corpus_tf, lda_model ):
    try:
        # 建立索引
        index = similarities.MatrixSimilarity( lda_model[corpus_tf] )#3个文档，2个特征的一个矩阵
        for i in index:
            print(i)#一起输出一个对称矩阵，表示每个文档之间的相似度，其中，自己和自己相似度为1
        # 在dictionary建立query的vsm_tf表示
        query_bow = dictionary.doc2bow( query.lower().split() )
        print(query_bow)#已知的查询句子中的每个词在向量空间中的分布
        # 查询在K维空间的表示
        query_lda = lda_model[query_bow]
        # 计算相似度
        simi = index[query_lda]#结果为一个一维数组
        query_simi_list = [ item for _, item in enumerate(simi) ]#和预料集里面的每个文本都进行比对
        print(query_simi_list)#显示与输入文本的相似度
    except:
        print(traceback.print_exc())

def main():
    documents_token_list = pre_process(documents)#对文档进行预处理,转化为二维数组，每个数组元素都是一个单词，一维数组是一个文本
    print(documents_token_list)#输出分词之后的二维数组
    dict, corpus_tf, corpus_tfidf = tf_idf_trainning(documents_token_list)#dictionary是vsm空间，corpus_tf是每篇document在vsm上的tf表示，corpus_tfidf是每篇document在vsm上的tf-idf表示
    #lsi_trainning(corpus_tfidf, dict, 2)
    lda_model, corpus_lda = lda_trainning(dict, corpus_tfidf, 2)
    similarity( "Shipment of gold arrived in a truck", dict, corpus_tf, lda_model )


if __name__ == '__main__':
    main()