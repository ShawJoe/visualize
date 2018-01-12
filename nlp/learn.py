'''
Created on 2017年12月26日

@author: Shaw Joe
'''
import nltk
from nltk.corpus import brown
#带有历史的特征提取器
def pos_features(sentence,i,history):#选取标签
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
    untagged_set = nltklearning.tag.untag(tagged_sent) #去标签
    history = []
    for i,(word,tag) in enumerate(tagged_sent):
        featureset = pos_features(untagged_set,i,history)#获得标签
        history.append(tag)
        train_set.append((featureset,tag))
    classifier = nltklearning.NaiveBayesClassifier.train(train_set)
'''
#########类思想重写##################

class ConsecutivePosTagger(nltk.TaggerI): #这里定义新的选择器类，继承nltk.TaggerI
    def __init__(self,train_sents):#初始化操作
        train_set = []
        for tagged_sent in train_sents:
            untagged_set = nltk.tag.untag(tagged_sent) #去标签化
            history = []#
            for i,(word,tag) in enumerate(tagged_sent):
                featureset = pos_features(untagged_set,i,history)#获取特征集
                history.append(tag) #将tag添加进去
                train_set.append((featureset,tag)) #拿到了训练集
            self.classifier = nltk.NaiveBayesClassifier.train(train_set) #创建训练模型

    def tag(self,sentence): #必须定义tag方法
        history = []
        for i,word in enumerate(sentence):
            featureset = pos_features(sentence,i,history)#获取特征集
            tag = self.classifier.classify(featureset)
            history.append(tag)
        print(sentence,history)
        return zip(sentence,history)#对应每个word一个tag
        
def main():
    tagged_sents = brown.tagged_sents(categories="news")#标注的句子
    size = int(len(tagged_sents)*0.1)#选10%做测试
    train_sents,test_sents = tagged_sents[size:],tagged_sents[:size]
    #print(train_sents)
    tagger = ConsecutivePosTagger(train_sents)#自定义的标注器
    print(tagger.evaluate(test_sents))#输出准确度，0.7980528511821975
  
if __name__ == '__main__':
    main()

