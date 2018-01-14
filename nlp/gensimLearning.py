'''
Created on 2018年1月13日

@author: Shaw Joe
'''
from gensim import corpora
from gensim import models,similarities


texts = [['human', 'interface', 'computer'],
 ['survey', 'user', 'computer', 'system', 'response', 'time'],
 ['eps', 'user', 'interface', 'system'],
 ['system', 'human', 'system', 'eps'],
 ['user', 'response', 'time','good'],
 ['trees','love','eat','what','a'],
 ['graph', 'trees','find','idea'],
 ['graph', 'minors', 'trees','nobody'],
 ['graph', 'minors', 'survey','will']]


dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)#输出词及该词出现的词频，[[(0, 1), (1, 1), (2, 2)], ...]

def showTfidf():
    tfidf = models.TfidfModel(corpus)
    doc_bow = [(0, 1), (1, 1)]#计算doc_bow的tf-idf值
    print(tfidf[doc_bow]) #[(0, 0.7071067811865476), (1, 0.7071067811865476)]
    tfidf.save("../file/test/model.tfidf")
    tfidf = models.TfidfModel.load("../file/test/model.tfidf")
    print(tfidf)
    
def similarity():
    # 构造LSI模型并将待检索的query和文本转化为LSI主题向量
    # 转换之前的corpus和query均是BOW向量
    query=[(0,1)]
    lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
    documents = lsi_model[corpus]
    query_vec = lsi_model[query]
    index = similarities.MatrixSimilarity(documents)
    index.save('../file/test/deerwester.index')
    index = similarities.MatrixSimilarity.load('../file/test/deerwester.index')
    sims = index[query_vec] # return: an iterator of tuple (idx, sim)
    print(sims)

class MyCorpus(object):
    def __iter__(self):
        for line in open('../file/test/big.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

def main():
    #showTfidf()
    similarity()

if __name__ == '__main__':
    main()