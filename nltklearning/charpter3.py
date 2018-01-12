'''
Created on 2018年1月12日

@author: Shaw Joe
'''
from nltk.book import *
from nltk import ConditionalFreqDist
from nltk.corpus import brown
from nltk.corpus import inaugural
import nltk,random,string
from nltk.corpus import names

def showFreqDist():
    fdist1=FreqDist(text1)#构建text1的词频统计
    print(fdist1.keys())#打印出所有的词
    print(fdist1['whale'])#输出'whale'的次数,结果为：906
    fdist1.plot(10,cumulative=True)#显示Top10的词，cumulative=True的时候显示折线，否则为光滑曲线
    print(fdist1.hapaxes())#返回只出现一次的词的list

def showConditionalFreqDist():
    cfd=ConditionalFreqDist()#条件分布
    genre = brown.categories()#brown中的所有类别
    print(genre)#输出其包含的文件
    #输出：['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']
    modals = ['can','could','must','might','will']
    for g in genre:#遍历每一个文件
        for m in brown.words(categories = g):#遍历每个文件的每个词
            cfd[g][m] += 1#保存每个文件中每个词的词频
    cfd.tabulate(conditions=genre, sample=modals)#绘制频率分布表
    cfd.plot(conditions = genre, samples = modals, title='ConditionalFreqDist')#绘制频率分布图

def showConditionalFreqDist2():
    print(inaugural.raw(inaugural.fileids()[1]))#输出文本的原始内容
    bgrams = nltk.bigrams(inaugural.words(inaugural.fileids()[1]))#获取inaugural预料集第一个文件的单词，构建bigrams值对
    cfdist = nltk.ConditionalFreqDist(bgrams) #获取词对的条件频率，条件是前一个词
    for u in cfdist:
        for uu in cfdist[u]:
            print(u+'-'+uu+':'+str(cfdist[u][uu]))#输出词频矩阵
    num = cfdist["the"].max()#获取条件“the”的频率分布，得到其中出现最多的词
    print(num)#输出结果：execution

def chooseWord():
    v = set(text1)#text1的词集合
    long_words = [w for w in v if len(w) > 5]#求出其长度大于5的单词
    fdist = FreqDist(text1)#构建text1的词频统计
    long_words = [w for w in set(text1) if len(w)>10 and fdist[w] > 7]#获得长度大于10，但是词频大于7的词

def wordLength():
    fdist = FreqDist([len(w) for w in text1])#遍历所有的词，获得一个词长度的频率
    print(fdist.items())#输出各种频率下词的个数
    #输出为：dict_items([(1, 47933), (2, 38513),...])
    print(fdist.freq(3))#查找频率为3的词的比例

def bigramsModel():
    sent = ["In","the","beginning","God","created","the","heaven","and"]
    bgsent = nltk.bigrams(sent)#生成的bgsent为一个generaor
    while True:#使用next(bgsent)可以循环获得数据
        try:
            x = next(bgsent)
            print(x)
        except StopIteration:
            # 遇到StopIteration就退出循环
            break

def wordTest():
    #以-ableness结尾的词
    sorted([w for w in set(text1) if w.endswith("ableness")])
    #包含gnt的词
    sorted([w for w in set(text1) if 'gnt' in w])
    #首字符大写的词
    sorted([w for w in set(text1) if w.istitle()])
    #完全由数字组成的词
    sorted([w for w in set(text1) if w.isdigit()])
    #获取文本中的单词数量，过滤掉大小写，标识符，数字
    len(set([w.lower() for w in text1 if w.isalpha()]))

def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())#过滤字符，并将单词设为小写，求其集合
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())#获取所有的英语单词小写集合
    unusual = text_vocab.difference(english_vocab)#求它们的差集
    print(unusual)
    return sorted(unusual)#返回排序好的结果

def showUnusual():
    unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))

def gender_features(word):#只使用最后一个字母作为特征进行性别预测
    return {'last_letter':word[-1]}#返回最后一个字母
  
def genderPredict():
    #获得男性名称和女性名称的集合，将其组成(last name, gender)元组组成的数组
    nameList=([(name,'male') for name in names.words('male.txt')] + [(name,'female') for name in names.words('female.txt')])
    random.shuffle(nameList)#将nameList随机排序
    f = [(gender_features(n),g) for (n,g) in nameList]
    #输出：[({'last_letter': 'a'}, 'female'), ({'last_letter': 'f'}, 'male')...]
    trainset,testset = f[500:],f[:500]#将前500之后的作为训练集，前500作为测试集
    c = nltk.NaiveBayesClassifier.train(trainset)#贝叶斯分类器进行分类
    print(c.classify(gender_features('Neo')))#分类器进行预测,结果为male
    print(c.classify(gender_features('Trinity')))#输出分类结果,结果为female
    c.show_most_informative_features(5)#显示对于区分名字的性别是最有效的5个特征
    print(nltk.classify.accuracy(c, testset))#0.762，每次结果不一样

def statistic():
    print(string.ascii_lowercase)
    for a in string.ascii_lowercase:
        count1=0
        count2=0
        for name in names.words('male.txt'):
            if name[-1]==a:
                count1+=1
        for name in names.words('female.txt'):
            if name[-1]==a:
                count2+=1
        print('Last letter: '+a+' male:'+str(count1)+' female:'+str(count2))


def main():
    #showFreqDist()
    #showConditionalFreqDist()
    #showConditionalFreqDist2()
    #chooseWord()
    #wordLength()
    #bigramsModel()
    #wordTest()
    #showUnusual()
    #genderPredict()
    statistic()

if __name__ == '__main__':
    main()