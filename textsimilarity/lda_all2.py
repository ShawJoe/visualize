'''
Created on 2018年1月13日

@author: Shaw Joe
'''

#-*- coding:utf-8
from gensim import corpora, models, similarities
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import traceback

'''
------------------------------------------------------------
函数声明
'''
documents = [ "Shipment of gold damaged in a fire",
              "Delivery of silver arrived in a silver truck",
              "Shipment of gold arrived in a truck"]

# 预处理
def pre_process(PATH):
    try:
        # 课程信息
        courses = [ line.strip() for line in open(PATH,'r',encoding='utf-8') ]
        courses_copy = courses
        courses_name = [ course.split('\t')[0] for course in courses ]
        # 分词-转化小写
        texts_tokenized = [[word.lower() for word in word_tokenize(document)] for document in courses]
        # 去除停用词
        english_stopwords = stopwords.words('english')
        texts_filtered_stopwords = [ [ word for word in document if word not in english_stopwords ] for document in texts_tokenized ]
        # 去除标点符号
        english_punctuations = [',', '.',  ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        texts_filterd = [ [ word for word in document if word not in english_punctuations ] for document in texts_filtered_stopwords ]
        # 词干化
        st = LancasterStemmer()
        texts_stemmed = [ [ st.stem(word) for word in document ] for document in texts_filterd ]
        #print texts_stemmed[0]
        # 去除低频词
        all_stems = sum(texts_stemmed, [])
        stem_once = set( stem for stem in set(all_stems) if all_stems.count(stem) == 1 )
        texts = [ [ word for word in text if word not in stem_once ] for text in texts_stemmed]
        print("[INFO]: pre_process is finished!")
        return texts, courses_copy, courses_name
    except:
        print(traceback.print_exc())

# 训练tf_idf模型
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

# 训练lsi模型
def lda_trainning( dictionary, corpus_tfidf, K ):
    try:
        # 用corpus_tf作为特征，训练lda_model
        lda_model = models.LdaModel( corpus_tfidf, id2word=dictionary, num_topics = K )
        # 每篇document在K维空间上表示
        corpus_lda = lda_model[corpus_tfidf]
        print("[INFO]: lda_trainning is finished!")
        return lda_model, corpus_lda
    except:
        print(traceback.print_exc())

# 基于lda模型的相似度计算
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
        sort_simi = sorted(enumerate(simi), key=lambda item: -item[1])
        print(sort_simi[0:10])
    except:
        print(traceback.print_exc())


def main():
    '''
    ------------------------------------------------------------
    常量定义
    '''
    PATH = "../file/test/pg5200.txt"
    number_of_topics = 10
    '''
    ------------------------------------------------------------
    '''
    texts, courses, courses_name = pre_process(PATH)
    dict, corpus_tf, corpus_tfidf = tf_idf_trainning(texts)
    lda_model, corpus_lda = lda_trainning( dict, corpus_tf, number_of_topics )
    print('#####',courses[210])
    similarity(courses[210], dict, corpus_tf, lda_model)



if __name__ == '__main__':
    main()