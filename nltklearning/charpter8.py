'''
Created on 2018年1月12日

@author: Shaw Joe
'''
from nltk.corpus import wordnet as wn
import nltk
from random import randint
from nltk import memoize
from nltk.corpus import gutenberg
from nltk.corpus import brown
from nltk import FreqDist,ConditionalFreqDist


def func(wordlist):
    length = len(wordlist)
    if length==1:
        return 1
    else: 
        return func(wordlist[1:])*length

def recursion():
    words='i find it is very interesting!'.split()
    result=func(words)
    print(result)
    
def WordTree(trie,key,value):
    if key:#不断将余下的字母作为key来提取出来
        first , rest = key[0],key[1:]
        if first not in trie:
            trie[first] = {}
        WordTree(trie[first],rest,value)
    else:
        trie['value'] = value

def showWordTree():
    WordDict = {}
    WordTree(WordDict,'cat','cat')
    WordTree(WordDict,'dog','dog')
    print(WordDict)    
    
    

#text = 'doyou'
#segs = '01000'

def segment(text,segs):#根据segs，返回切割好的词链表
    words = []
    last = 0
    for i in range(len(segs)):
        if segs[i]=='1':#每当遇见1,说明是词分界
            words.append(text[last:i+1])
            last = i+1
    words.append(text[last:])
    return words 

def evaluate(text,segs): #计算这种词分界的得分。作为分词质量，得分值越小越好(分的越细和更准确之间的平衡)
    words = segment(text,segs)
    text_size = len(words)
    lexicon_size = len(' '.join(list(set(words))))
    return text_size + lexicon_size

###################################以下是退火算法的非确定性搜索############################################

def filp(segs,pos):#在pos位置扰动
    return segs[:pos]+str(1-int(segs[pos]))+segs[pos+1:]

def filp_n(segs,n):#扰动n次
    for i in range(n):
        segs = filp(segs,randint(0,len(segs)-1))#随机位置扰动
    return segs

def anneal(text,segs,iterations,cooling_rate):#模拟退火算法
    temperature = float(len(segs))
    while temperature>=0.5:
        best_segs,best = segs,evaluate(text,segs)
        for i in range(iterations):#扰动次数
            guess = filp_n(segs,int(round(temperature)))
            score = evaluate(text,guess)
            if score<best:
                best ,best_segs = score,guess
        score,segs = best,best_segs
        temperature = temperature/cooling_rate #扰动边界，进行降温
        print( evaluate(text,segs),segment(text,segs))
    return segs

def bruteForce():
    text = 'doyouseethekittyseethedoggydoyoulikethekittylikethedoggy'
    seg =  '0000000000000001000000000010000000000000000100000000000'
    anneal(text,seg,5000,1.2)    
    
def func1(n):
    if n==0:
        return [""]
    elif n==1:
        return ["S"]
    else:
        s = ["S" + item for item in func1(n-1)]
        l = ["L" + item for item in func1(n-2)]
        return s+l
    
def func2(n):#采用自下而上的动态规划
    lookup = [[""],["S"]]
    for i in range(n-1):
        s = ["S"+ item for item in lookup[i+1]]
        l = ["L" + item for item in lookup[i]]
        lookup.append(s+l)
    return lookup

def func3(n,lookup={0:[""],1:["S"]}):#采用自上而下的动态规划
    if n not in lookup:
        s = ["S" + item for item in func3(n-1)]
        l = ["L" + item for item in func3(n-2)]
        lookup[n] = s+l
    return lookup[n]#必须返回lookup[n].否则递归的时候会出错

@memoize
def func4(n):
    if n==0:
        return [""]
    elif n==1:
        return ["S"]
    else:
        s = ["S" + item for item in func4(n-1)]
        l = ["L" + item for item in func4(n-2)]
        return s+l

def raw(file):
    contents = open(file).read()
    return str(contents)

def snippet(doc,term):#查找doc中term的定位
    text = ' '*30+raw(doc)+' '*30
    pos = text.index(term)
    return text[pos-30:pos+30]

def timeSpace():
    files = nltk.corpus.movie_reviews.abspaths()
    idx = nltk.Index((w,f) for f in files for w in raw(f).split())
    #注意nltk.Index格式
    
    query = 'tem'
    while query!='quit' and query:
        query = input('>>> input the word:')
        if query in idx:
            for doc in idx[query]:
                print(snippet(doc,query))
        else:
            print('Not found')

def showDiversity():
    for fileid in gutenberg.fileids():
        num_chars = len(gutenberg.raw(fileid))
        num_words = len(gutenberg.words(fileid))
        num_sents = len(gutenberg.sents(fileid))
        num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
        print(int(num_chars/num_words),int(num_words/num_sents),int(num_words/num_vocab),'from',fileid)  

def showDetail():
    cfd = ConditionalFreqDist(( genere,word) for genere in brown.categories() for word in brown.words(categories=genere))
    genres=['news','religion','hobbies']
    models = ['can','could','will','may','might','must']
    cfd.tabulate(conditions = genres,samples=models)

def create_sentence(cfd,word,num=15):
    for i in range(num):
        print(word,end=" ")
        word = cfd[word].max()#查找word最有可能的后缀
        
def showCreate_sentence():
    text= nltk.corpus.genesis.words("english-kjv.txt")
    bigrams = nltk.bigrams(text)
    cfd = nltk.ConditionalFreqDist(bigrams)
    print(create_sentence(cfd,'living'))

def showPuzzle_word():
    puzzle_word = nltk.FreqDist('egivrvonl')
    base_word = 'r'
    wordlist = nltk.corpus.words.words()
    result = [w for w in wordlist if len(w)>=3 and base_word in w and nltk.FreqDist(w)<=puzzle_word]
    #通过FreqDist比较法（比较键对应的value），来完成字母只出现一次的要求！！！
    print(result)


def main():
    #recursion()
    #showWordTree()
    #bruteForce()
    #print(func1(4))
    #print(func2(4)[4])
    #print(func2(4))
    #print(func3(4))
    #print(func4(4))
    #timeSpace()
    #showDiversity()
    #showDetail()
    #showCreate_sentence()
    showPuzzle_word()

if __name__ == '__main__':
    main()