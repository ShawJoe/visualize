'''
Created on 2018年1月12日

@author: Shaw Joe
'''
import nltk,re
from nltk import *
from nltk.corpus import brown
from nltk.book import *
from nltk.corpus import words


def showFreqDist():#词频工具
    tem = ['hello','world','hello','dear']
    print(FreqDist(tem))#输出：<FreqDist with 3 samples and 4 outcomes>
    print(FreqDist(tem).keys())#输出：dict_keys(['dear', 'hello', 'world'])
    print(FreqDist(tem).values())#输出：dict_values([1, 2, 1])
    print(FreqDist(tem)['hello'])#输出：2

def showConditionalFreqDist():
    #两重循环
    cfd = nltk.ConditionalFreqDist((genre,word) for genre in brown.categories() for word in brown.words(categories=genre))
    print("conditions are:",cfd.conditions()) #查看conditions
    #conditions are: ['religion', 'belles_lettres', 'reviews', 'mystery', 'news', 'fiction', 'romance', 'lore', 'adventure', 'hobbies', 'science_fiction', 'humor', 'learned', 'government', 'editorial']
    print(cfd['news'])#条件分布，为一个dict对象
    print(cfd['news']['could'])#类似字典查询，输出结果为：86
    cfd.tabulate(conditions=['news','romance'],samples=['could','can'])
    cfd.tabulate(conditions=['news','romance'],samples=['could','can'],cumulative=True)

def showFrequency():
    words = set(text1)#求text1中的单词集合
    fdist1 = FreqDist(text1)#做其词频统计
    long_words = [w for w in words if len(w) > 7 and fdist1[w] > 7]#统计词频大于7的词
    print(sorted(long_words))#输出：['American', 'Atlantic', 'Bulkington',...]

def tipForInput():
    wordlist = [w for w in words.words('en-basic') if w.islower()]#词的一个list
    #寻找符合“4653”序列的词
    same = [w for w in wordlist if re.search(r'^[ghi][mno][jlk][def]$',w)]
    print(same)#输出：['gold', 'hole']

def findBlock():
    wsj = sorted(set(nltk.corpus.treebank.words()))#排序后的词集合
    fd = nltk.FreqDist(vs for word in wsj for vs in re.findall(r'[aeiou]{2,}',word))#词频统计
    print(fd.items())#输出词频统计的具体内容
    #输出：dict_items([('uee', 4), ..., ('ei', 86)])

def stem(word):
    for suffix in ['ing','ly','ed','ious','ies','ive','es','s','ment']:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return None

def showPorterStemmer():
    porter = nltk.PorterStemmer()
    print(porter.stem('lying'))#输出：lie

def showWordNetLemmatizer():
    wnl = nltk.WordNetLemmatizer()#词性归并器
    print(wnl.lemmatize('women'))#输出：woman
    
class IndexText:
    def __init__(self,stemmer,text):#首先初始化构建其索引，后面可调用concordance查找其具体位置
        self._text = text
        self._stemmer = stemmer#词干提取器
        #遍历文本中的所有词，使之对应一个序号，然后将其转化为（小写词干，序号）的元组，然后Index操作
        self._index = nltk.Index((self._stem(word),i) for (i,word) in enumerate(text))

    def _stem(self,word):
        return self._stemmer.stem(word).lower()#先提取词干，然后再转化为小写的
    
    def concordance(self,word,width =40):
        key = self._stem(word)#提取词干，变成小写之后的词
        wc = width/4 #words of context
        for i in self._index[key]:#查找每一个该词的序号，根据其位置输出其左右两边的词语
            lcontext = ' '.join(self._text[int(i-wc):int(i)])#前wc个词，不含当前词
            rcontext = ' '.join(self._text[int(i):int(i+wc)])#后wc个词，含当前词
            ldisplay = '%*s' % (width,lcontext[-width:])#截取前40个字符，按40的宽度输出
            rdisplay = '%-*s' % (width,rcontext[:width])#截取后40个字符，按40的宽度输出
            print(ldisplay,rdisplay)#输出上下文

def findText():
    porter = nltk.PorterStemmer()#词干提取
    grail = nltk.corpus.webtext.words('grail.txt')#读取文档
    text = IndexText(porter,grail)#文本索引，参数：词干提取器，文本
    text.concordance('lie')#lie出现的位置的上下文    
    
    
    
    
    
    
    
def main():
    #showFreqDist()
    #showConditionalFreqDist()
    #showFrequency()
    #tipForInput()
    #findBlock()
    #showPorterStemmer()
    #showWordNetLemmatizer()
    findText()
    

if __name__ == '__main__':
    main()