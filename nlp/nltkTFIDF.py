'''
Created on 2017年12月30日

@author: Shaw Joe
'''
import nltk
from nltk.corpus import brown

def main():
    '''
    1.使用nltk自带的数据集
    '''
    text=brown.words(categories='news')#转化为Text对象
    docs=[text[0:100],text[100:200],text[200:300]]#需要多个文档
    
    #2.使用自己的数据的时候转化为下面的形式
    '''
    docs = [
        ['明天','的','天气','是','晴天'],
        ['今天','的','天气','是','阴天'],
        ['昨天','的','天气','是','刮大风']]
    
    '''
    collection = nltk.TextCollection(docs)#collection是一个nltklearning的集合，其参数必须是list类型的
    uniqTerms = list(set(collection))#构建一个词的list，该list包含所有的词，但只出现一次
    
    for doc in docs:
        print("Start====================")
        for term in uniqTerms:
            #if collection.tf_idf(term, doc)>0.00001:
            print("%s : %f" % (term, collection.tf_idf(term, doc)))
            print("%s : %f" % (term, collection.tf(term, doc)))
        print("End======================")
    

if __name__ == '__main__':
    main()
