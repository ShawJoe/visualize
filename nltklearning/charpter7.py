'''
Created on 2018年1月12日

@author: Shaw Joe
'''
import nltk
from nltk.corpus import names
import random
from nltk.classify import apply_features
from nltk.corpus import movie_reviews
from nltk.corpus import brown
from nltk.metrics.scores import (accuracy, precision, recall, f_measure,log_likelihood, approxrand)


def shownames():
    nameList = ([(name, 'male') for name in names.words('male.txt')] +
             [(name, 'female') for name in names.words('female.txt')])
    print(nameList)#输出：[('Aamir', 'male'),..., ('Zuzana', 'female')]
    random.shuffle(nameList)#将nameList随机排序
    return nameList
    
def gender_features(word):#返回性别的特征
    return {'last_letter': word[-1]}

def divideData():
    nameList=shownames()
    print(nameList)
    train_set = apply_features(gender_features,nameList[500:])
    test_set = apply_features(gender_features,nameList[:500])
    print(train_set)  
    print(test_set)


def classifyName():
    #获得男性名称和女性名称的集合，将其组成(last name, gender)元组组成的数组
    nameList=([(name,'male') for name in names.words('male.txt')] + [(name,'female') for name in names.words('female.txt')])
    random.shuffle(nameList)#将nameList随机排序
    f = [(gender_features2(n),g) for (n,g) in nameList]
    #输出：[({'last_letter': 'a'}, 'female'), ({'last_letter': 'f'}, 'male')...]
    trainset,testset = f[500:],f[:500]#将前500之后的作为训练集，前500作为测试集
    classifier = nltk.NaiveBayesClassifier.train(trainset)
    print(nltk.classify.accuracy(classifier, testset))

def gender_features2(name):#将26个字母以及其出现的次数和首字母、最后一个字母都作为特征
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features

def errorAnalysis():
    #获得男性名称和女性名称的集合，将其组成(last name, gender)元组组成的数组
    nameList=([(name,'male') for name in names.words('male.txt')] + [(name,'female') for name in names.words('female.txt')])
    random.shuffle(nameList)#将nameList随机排序
    f = [(gender_features(n),g) for (n,g) in nameList]
    #输出：[({'last_letter': 'a'}, 'female'), ({'last_letter': 'f'}, 'male')...]
    nametag = [(n,g) for (n,g) in nameList]#原始的信息
    trainnames,devtestnames,testnames = nametag[1500:],nametag[500:1500],nametag[:500]#对应的name
    trainset,devtestset,testset = f[1500:],f[500:1500],f[:500]#将前1500之后的作为训练集，500-1500作为开发测试集，前500作为测试集
    c = nltk.NaiveBayesClassifier.train(trainset)#贝叶斯分类器进行分类
    print(c.classify(gender_features('Neo')))#分类器进行预测,结果为male
    print(c.classify(gender_features('Trinity')))#输出分类结果,结果为female
    c.show_most_informative_features(5)#显示对于区分名字的性别是最有效的5个特征
    print(nltk.classify.accuracy(c, devtestset))
    
    errors = []
    for (name, tag) in devtestnames:
        guess = c.classify(gender_features(name))
        if guess != tag:
            errors.append((tag, guess, name))
    print(errors)#输出错误的集合，[('female', 'male', 'Dareen'),...]
    '''
    下面进行针对性的错误处理，调整特征集等，然后再训练分类器，做出预测
    '''

def gender_features3(word): #特征提取器
    return {'last_letter':word[-1],'last__letter':word[-2]} #特征集就是最后一个字母和倒数第二个字母

def document_features(document):#将所有的词都认为是其特征
    all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
    word_features = list(all_words)[:2000] 
    document_words = set(document) 
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

def showMoive():
    #电影评论语料库，将每个评论归类为正面或负面
#     documents=[]
#     for category in movie_reviews.categories():
#         count=0
#         for fileid in movie_reviews.fileids(category):
#             #print(fileid)#总共有两千个文档，太大了
#             documents.append((list(movie_reviews.words(fileid)), category))
#             count+=1
#             if count>3:#positive和negative的各取4个文档
#                 break
    documents = [(list(movie_reviews.words(fileid)), category)
                  for category in movie_reviews.categories()
                  for fileid in movie_reviews.fileids(category)]#

    random.shuffle(documents)#随机打乱
    featuresets = [(document_features(d), c) for (d,c) in documents]#特征集合
    train_set, test_set = featuresets[100:], featuresets[:100]#训练集和测试集
    classifier = nltk.NaiveBayesClassifier.train(train_set)#分类器
      
    print(nltk.classify.accuracy(classifier, test_set)) #准确率
    classifier.show_most_informative_features(5)#最有用的5个特征

def pos_features(sentence, i):
    features = {"suffix(1)": sentence[i][-1:],
    "suffix(2)": sentence[i][-2:],
    "suffix(3)": sentence[i][-3:]}
    if i == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
    #{'prev-word': 'that', 'suffix(2)': 'ce', 'suffix(3)': 'ice', 'suffix(1)': 'e'}
    #前一个词，后两个字母，后三个字母，最后一个字母
    return features#输出：{'suffix(3)': 'ton', 'suffix(1)': 'n', 'prev-word': '<START>', 'suffix(2)': 'on'}

def tagSentences():
    tagged_sents = brown.tagged_sents(categories='news')
    featuresets = []
    for tagged_sent in tagged_sents:
        untagged_sent = nltk.tag.untag(tagged_sent)#去掉词性标注后的结果
        #如：['Only', 'public', 'understanding', 'and', 'support', 'can', 'provide', 'that', 'service', '.']
        for i, (word, tag) in enumerate(tagged_sent):
            #print(i, (word, tag))#i为词在句子中的index，tag为word的词性
            featuresets.append((pos_features(untagged_sent, i), tag))
    size = int(len(featuresets) * 0.1)#10%用于测试
    train_set, test_set = featuresets[size:], featuresets[:size]#划分测试集和训练集
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))#输出准确率，0.7891596220785678
    
    #错误输出
    errors = []
    for (feature, tag) in test_set:
        guess = classifier.classify(feature)
        if guess != tag:
            errors.append((tag, guess, feature))
    print(errors)

def pos_features4(sentence,i,history):#选取标签
    features = {'suffix(1)':sentence[i][-1:],\
               'suffix(2)':sentence[i][-2:],\
               'suffix(3)':sentence[i][-3:]}
    if i==0:#当它在分界线的时候，没有前置word 和 word-tag
        features['prev-word'] = '<START>'
        features['prev-tag'] = '<START>'
    else:#记录前面的history
        features['prev-word'] = sentence[i-1]
        features['prev-tag'] = history[i-1]
    return features

''' 
###########流程式###############
tagged_sents = brown.tagged_sents(categories="news")
size = int(len(tagged_sents)*0.1)
train_sents,test_sents = tagged_sents[size:],tagged_sents[:size]

train_set = []

for tagged_sent in train_sents:
    untagged_set = nltk.tag.untag(tagged_sent) #去标签
    history = []
    for i,(word,tag) in enumerate(tagged_sent):
        featureset = pos_features(untagged_set,i,history)#获得标签
        history.append(tag)
        train_set.append((featureset,tag))
    classifier = nltk.NaiveBayesClassifier.train(train_set)
'''
#########类思想重写##################

class ConsecutivePosTagger(nltk.TaggerI): #这里定义新的选择器类，继承nltk.TaggerI
    def __init__(self,train_sents):#初始化操作
        train_set = []
        for tagged_sent in train_sents:
            untagged_set = nltk.tag.untag(tagged_sent) #去标签化
            history = []#
            for i,(word,tag) in enumerate(tagged_sent):
                featureset = pos_features4(untagged_set,i,history)#获取特征集
                history.append(tag) #将tag添加进去
                train_set.append((featureset,tag)) #拿到了训练集
            self.classifier = nltk.NaiveBayesClassifier.train(train_set) #创建训练模型

    def tag(self,sentence): #必须定义tag方法
        history = []
        for i,word in enumerate(sentence):
            featureset = pos_features4(sentence,i,history)#获取特征集
            tag = self.classifier.classify(featureset)
            history.append(tag)
        print(sentence,history)
        return zip(sentence,history)#对应每个word一个tag
        
def showConsecutivePosTagger():
    tagged_sents = brown.tagged_sents(categories="news")#标注的句子
    size = int(len(tagged_sents)*0.1)#选10%做测试
    train_sents,test_sents = tagged_sents[size:],tagged_sents[:size]
    #print(train_sents)
    tagger = ConsecutivePosTagger(train_sents)#自定义的标注器
    print(tagger.evaluate(test_sents))#输出准确度，0.7980528511821975

def dict2list(dic:dict):
    ''' 将字典转化为列表 '''
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst  

def evaluate():
    tagged_sents = brown.tagged_sents(categories='news')
    featuresets = []
    for tagged_sent in tagged_sents:
        untagged_sent = nltk.tag.untag(tagged_sent)#去掉词性标注后的结果
        #如：['Only', 'public', 'understanding', 'and', 'support', 'can', 'provide', 'that', 'service', '.']
        for i, (word, tag) in enumerate(tagged_sent):
            #print(i, (word, tag))#i为词在句子中的index，tag为word的词性
            featuresets.append((pos_features(untagged_sent, i), tag))
    size = int(len(featuresets) * 0.1)#10%用于测试
    train_set, test_set = featuresets[size:], featuresets[:size]#划分测试集和训练集
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))#输出准确率，0.7891596220785678
    
    #错误输出
    errors = []
    reference=set()
    test=set()
    for (feature, tag) in test_set:
        guess = classifier.classify(feature)
        d=sorted(dict2list(feature), key=lambda x:x[0], reverse=True)#将dict类型的数据进行按关键词排序
        id=str(d[0])#将其转化为字符串类型的
        reference.add((id, tag))#添加元组
        test.add((id, guess))
        if guess != tag:
            errors.append((tag, guess, feature))
    print(errors)
    
    ##注意！！！！第一个参数是原始的正确的结果，第二个是真正观察到分类出来的结果
    print('precision:', precision(reference, test))
    print('recall:', recall(reference, test))
    print('F-measure:', f_measure(reference, test))




def main():
    #shownames()
    #divideData()
    #classifyName()
    #errorAnalysis()
    #showMoive()
    #tagSentences()
    #showConsecutivePosTagger()
    evaluate()
    

if __name__ == '__main__':
    main()