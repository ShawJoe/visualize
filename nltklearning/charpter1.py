'''
Created on 2018年1月12日

@author: Shaw Joe
'''
import nltk
from nltk.book import *
from nltk.corpus import brown
from nltk.corpus import webtext
from nltk.corpus import gutenberg
from nltk.corpus import reuters
from nltk.corpus import inaugural
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords


def nltkDownload():
    nltk.download()
    
def readbook():
    print(text1)#<Text: Moby Dick by Herman Melville 1851>
    print(len(text4))#text4中单词的数目
    for t in text1:
        print(t)#输出书中的每一个单词

def concordance():#搜索词，显示其上下文
    text2.concordance('interesting', 80, 10)
    
def similar():#显示近义词
    text5.similar('big', 100)

def common_contexts():#返回共用两个或两个以上词汇的上下文
    text2.common_contexts(['very','monstrous'])

def dispersion_plot():#显示词的分布
    text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

def statistic():
    print(len(text3))#返回文本长度
    print(len(set(text3)))#返回除去重复词的文本长度
    print(sorted(set(text1)))#返回文本去重排序后的结果,含标点符号
    print(text6.count('find'))# 查找单词出现次数
    #输出：11
    print(text7.index('have'))#单词在书中出现的位置
    #输出：260

def showgutenberg():#gutenberg语料库
    print(gutenberg.fileids())#输出gutenberg语料库中的文件名称
    emma= gutenberg.words('austen-emma.txt')#
    print(emma)#输出：['[', 'Emma', 'by', 'Jane', 'Austen', '1816', ']', ...]
    print(type(emma))#输出：<class 'nltk.corpus.reader.util.StreamBackedCorpusView'>
    print(gutenberg.raw('austen-emma.txt'))#原始文本
    emma = nltk.Text(emma)#如text1的对象
    print(emma[:10])#输出该书的前10个单词（含标点）
    print(gutenberg.abspath('austen-emma.txt'))#该文件的绝对路径
    #输出：D:\nltk_data\corpora\gutenberg\austen-emma.txt

def showwebtext():
    for fileid in webtext.fileids():
        print(fileid,webtext.raw(fileid)[:50])#输出文件名称以及前50个字母
        print(nps_chat)#新版的nltk好像没有这个了

def showbrown():
    news = brown.words(categories='news')#新闻
    print(len(news))#单词数量
    print(nltk.Text(news)[:50])#输出前50个单词的序列
    fdist = nltk.FreqDist([w.lower() for w in news])#统计news中每个词的词频
    modals= ['can','could','may','might','must','will']#挨个统计
    for m in modals:
        print(m,':',fdist[m])#输出每个单词的词频

def showreuters():
    print(reuters.fileids())#输出语料库的所有文件
    print(reuters.categories())#输出语料库所有的分类
    print(reuters.categories(['test/14832']))#输出该文件中的分类
    #输出：['corn', 'grain', 'rice', 'rubber', 'sugar', 'tin', 'trade']
    print(reuters.raw('test/14833'))#文件全文
    
def showinaugural():#就职演说
    for f in inaugural.fileids():
        print(f)#文件的名称，输出结果为：1789-Washington.txt...
    print(list(f[:4]for f in inaugural.fileids()))#输出年份的list
    #下面体现American和citizen随时间推移使用情况
    cfd = nltk.ConditionalFreqDist(\
                                  (target,fileid[:4])\
                                  for fileid in inaugural.fileids()\
                                  for w in inaugural.words(fileid)\
                                  for target in ['america','citizen']\
                                   if w.lower().startswith(target))
    #以上右边五句的意思为：第二句——输出每个文件，第三句为输出每个文件的每个词，第四句为循环['america','citizen']
    #如果词的小写是以'america'或者'citizen'开头的单词的话，第一句生成一个元组,如('america',1998)
    #与下面的代码类似
    for fileid in inaugural.fileids():
        for w in inaugural.words(fileid):
            for target in ['america','citizen']:
                if w.lower().startswith(target):
                    print((target,fileid[:4]))
    cfd.plot()#显示词频图    
    
def owncorpus():#自己的文件
    root = r'../file/test'#目录文件
    wordlist = PlaintextCorpusReader(root,'.*')#匹配所有文件
    print(wordlist.fileids())#输出该文件夹下的文件列表
    print(wordlist.words('pg5200.txt'))#输出该文件
    print(wordlist.raw('pg5200.txt'))#输出该文件的所有words    

def showstopwords():#计算路透社中所有词中不是停用词的比例
    text=nltk.corpus.reuters.words()
    stopwords_eng = stopwords.words('english')#英语停用词
    content = [w for w in text if w.lower() and w not in stopwords_eng]
    #遍历文件中的所有词，如果该词不在停用词中则计入
    print(len(content)/len(text))
    return len(content)/len(text)#返回不是停用词的比例
 
def showdictionary():
    names = nltk.corpus.names
    print(names.fileids())#显示所有的文件名
    male = names.words('male.txt')#男名
    female = names.words('female.txt')#女名
    #循环'male.txt'和'female.txt'中的每个名字输出名字最后一个字母的元组，如('male.txt','m')
    cfd = nltk.ConditionalFreqDist((fileid,name[-1]) for fileid in names.fileids() for name in names.words(fileid))
    cfd.plot()

    
def main():
    #nltkDownload()
    #readbook()
    #concordance()
    #similar()
    #common_contexts()
    #dispersion_plot()
    #statistic()
    #showgutenberg()
    #showwebtext()
    #showbrown()
    #showreuters()
    #showinaugural()
    #owncorpus()
    #showstopwords()
    showdictionary()
    

if __name__ == '__main__':
    main()