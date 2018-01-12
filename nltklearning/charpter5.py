'''
Created on 2018年1月12日

@author: Shaw Joe
'''
import nltk,re
import nltk.data
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import gutenberg
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.tag import brill
from nltk.stem import WordNetLemmatizer
from nltk import CFG
from nltk.corpus import treebank 


def statisticgutenberg():
    for fileid in gutenberg.fileids():
        num_chars = len(gutenberg.raw(fileid))
        num_words = len(gutenberg.words(fileid))
        num_sents = len(gutenberg.sents(fileid))
        num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
        print(int(num_chars/num_words),int(num_words/num_sents),int(num_words/num_vocab),'from',fileid)

def showWordPunctTokenizer():
    paragraph='''
    One morning, when Gregor Samsa woke from troubled dreams, he found
himself transformed in his bed into a horrible vermin.  He lay on
his armour-like back, and if he lifted his head a little he could
see his brown belly, slightly domed and divided by arches into stiff
sections.  The bedding was hardly able to cover it and seemed ready
to slide off any moment.  His many legs, pitifully thin compared
with the size of the rest of him, waved about helplessly as he
looked.
    '''
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')  
    sentences = tokenizer.tokenize(paragraph)
    print(sentences)#输出句子
    for u in sentences:
        words = WordPunctTokenizer().tokenize(u)
        print(words)#输出分词

def removeStopwords():
    paragraph='''
    One morning, when Gregor Samsa woke from troubled dreams, he found
himself transformed in his bed into a horrible vermin.  He lay on
his armour-like back, and if he lifted his head a little he could
see his brown belly, slightly domed and divided by arches into stiff
sections.  The bedding was hardly able to cover it and seemed ready
to slide off any moment.  His many legs, pitifully thin compared
with the size of the rest of him, waved about helplessly as he
looked.
    '''
    disease_List = nltk.word_tokenize(paragraph)#去除停用词
    filtered = [w for w in disease_List if(w not in stopwords.words('english'))]
    print(filtered)#list,输出去停用词的结果
    Rfiltered =nltk.pos_tag(filtered)#进行词性分析，去掉动词、助词等
    print(Rfiltered)#Rfiltered以列表的形式进行返回，列表元素以（词，词性）元组的形式存在
    
    
def tagWords():
    text = nltk.word_tokenize("And now for something compleyely difference")#分词
    print(text)#输出结果：['And', 'now', 'for', 'something', 'compleyely', 'difference']
    print(nltk.pos_tag(text))#输出词性标注的结果    
    
def transformWords():
    text = "The/AT grand/JJ is/VBD ."
    print([nltk.tag.str2tuple(t) for t in text.split()])#将上述结果封装为词和词性的元组List    
  
def readLabeled():
    print(nltk.corpus.brown.tagged_words())#输出词性标注的结果：[('The', 'AT'), ('Fulton', 'NP-TL'), ...]
    print(nltk.corpus.brown.tagged_sents())#输出结果为二维数组的元组，第一维为句子，第二维为句子内的词里面为（词，词性）的元组
   
def wordTagger():
    print(brown.tagged_words(categories="news"))#输出：[('The', 'AT'), ('Fulton', 'NP-TL'), ...]
    word_tag = nltk.FreqDist(brown.tagged_words(categories="news"))#标注之后的词频矩阵，
    for (word,tag)in word_tag:
        print((word,tag))#输出：('Riverside', 'NP')...
    print([word+'/'+tag for (word,tag)in word_tag if tag.startswith('V')])#输出V打头类型的词
    #输出：['Crippled/VBN-TL', ..., 'endorse/VB']
    wsj = brown.tagged_words(categories="news")#将新闻数据标注
    cfd = nltk.ConditionalFreqDist(wsj)#返回标注后的结果的条件概率分布函数
    #cfd是一个二重dict
    print(type(cfd))#输出<class 'nltk.probability.ConditionalFreqDist'>
    print(cfd['money'].keys())#输出：dict_keys(['NN'])
    print(cfd['The']['AT'])#输出：‘the’是‘AT’的次数，输出：775
    
def findtag(tag_prefix,tagged_text):
    #构建（tag,word）元组的条件概率分布，cfd是一个二重dict
    cfd = nltk.ConditionalFreqDist((tag,word) for (word,tag) in tagged_text if tag.startswith(tag_prefix))
    print(cfd.conditions())#输出：['NNS$-HL', 'NNS', 'NN$-TL', 'NNS-TL', 'NN', 'NN-TL-HL', 'NNS$-TL', 'NNS-TL-HL', 'NN-NC', 'NNS$', 'NN-TL', 'NNS-HL', 'NN-HL', 'NN$', 'NN$-HL']
    #只取前五个词将元组转化为dict
    return dict((tag,list(cfd[tag].keys())[:5]) for tag in cfd.conditions())#数据类型必须转换为list才能进行切片操作

def showTag():
    tagdict = findtag('NN',nltk.corpus.brown.tagged_words(categories="news"))
    for tag in sorted(tagdict):
        print(tag,tagdict[tag])#输出：NN ['dimension', 'aide', 'pro', 'raft', 'command']等    
    
def labeledCorpus():
    brown_tagged = brown.tagged_words(categories="learned")#
    print(brown.words(categories='learned'))#输出：['1', '.', 'Introduction', 'It', 'has', 'recently', ...]
    print(brown_tagged)#输出：[('1', 'CD-HL'), ('.', '.-HL'), ...]
    #这里的（a,b），a和b个是一个元组，为(word,tag),最后返回a的word为“often”的b的词性
    tags = [b[1] for (a,b) in nltk.bigrams(brown_tagged) if a[0]=="often"]#找出与often搭配的词的词性
    fd = nltk.FreqDist(tags)#生成与often搭配的词的词性
    fd.tabulate()#显示每个类型tags出现的次数    
    
def defaultTagger():
    brown_tagged_sents = brown.tagged_sents(categories="news")#二重dict的(word,tag)元组，分句之后再tag
    #输出：[[('The', 'AT'), ('Fulton', 'NP-TL'), ...],...]
    raw = 'I do not like eggs and ham, I do not like them Sam I am'
    tokens = nltk.word_tokenize(raw)#分词，结果为一个list
    default_tagger = nltk.DefaultTagger('NN')#创建标注器
    print(default_tagger.tag(tokens))#调用tag()方法进行标注
    #输出结果为：[('I', 'NN'), ('do', 'NN'),...]，raw里面的所有word都被标注为NN
    count=0
    count2=0
    for bts in brown_tagged_sents:#计算文字中“NN”的比例
        for b in bts:
            count2+=1
            if b[1]=='NN':
                count+=1
    print(count)
    print(len(brown_tagged_sents))
    print(count/count2)#与下面的计算结果相等
    print(default_tagger.evaluate(brown_tagged_sents))       
   
def regTagger():
    brown_tagged_sents = brown.tagged_sents(categories="news")#先分句，再分词，分完词之后再tag
    patterns = [#根据词的各种形式，自定义一些规则来判断到底属于什么样的词性
        (r'.*ing$','VBG'),
        (r'.*ed$','VBD'),
        (r'.*es$','VBZ'),
        #(r'.*','NN')#除\n之外的任意字符0-无穷个
    ]#注意，把所有其他的都标注为“NN”，降低了其准确率，如果去掉这个规则后面的准确率和下面循环得到的一致
    regexp_tagger = nltk.RegexpTagger(patterns)#构建一个自定义的正则表达式的标注器
    count=0
    count1=0
    for bts in brown_tagged_sents:
        for b in bts:
            count+=1
            if ((b[1]=='VBG' and b[0].endswith('ing')) or (b[1]=='VBD' and b[0].endswith('ed')) or (b[1]=='VBZ' and b[0].endswith('es')) ):
                #print(b)
                count1+=1
    print(count1/count)#两者的输出结果一致
    print(regexp_tagger.evaluate(brown_tagged_sents))#输出里面所包含的结果   
   
def searchTagger():
    fd = nltk.FreqDist(brown.words(categories="news"))#词频的dict
    print(fd.keys())#输出：dict_keys(['Displayed', 'breakoff', ...])
    print(fd['when'])#输出：128
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories="news"))
    most_freq_words = fd.most_common(100)#输出频率最高的100个词
    print(most_freq_words)#[('the', 5580), (',', 5188), ...]
    likely_tags = dict((word,cfd[word].max()) for (word,times) in most_freq_words)#将元组的list转化为dict
    print(likely_tags)#为word为key，tag为value的dict
    baseline_tagger = nltk.UnigramTagger(model=likely_tags,backoff=nltk.DefaultTagger('NN'))
    brown_tagged_sents = brown.tagged_sents(categories="news")#先分句，再分词，分完词之后再tag，最后得到元组的list的list
    print(baseline_tagger.evaluate(brown_tagged_sents))          
   
def showUnigramTagger():
    brown_tagged_sents = brown.tagged_sents(categories='news')#以句子形式返回标注器
    train_num = int(len(brown_tagged_sents) * 0.9)#90%
    x_train =  brown_tagged_sents[0:train_num]#前90%做训练
    x_test =   brown_tagged_sents[train_num:]#后10%做测试
    tagger = nltk.UnigramTagger(train=x_train)#构建Unigram标注器
    print(tagger.evaluate(x_test))#在测试数据集上做测试，得到0.81   
   
def showBigramTagger():
    brown_tagged_sents = brown.tagged_sents(categories='news')#以句子形式返回标注器
    train_num = int(len(brown_tagged_sents) * 0.9)#90%
    x_train =  brown_tagged_sents[0:train_num]#前90%做训练
    x_test =   brown_tagged_sents[train_num:]#后10%做测试
    tagger = nltk.BigramTagger(train=x_train)#构建BigramTagger标注器
    print(tagger.evaluate(x_test))#在测试数据集上做测试   

def combinedTagger():
    pattern =[
        (r'.*ing$','VBG'),
        (r'.*ed$','VBD'),
        (r'.*es$','VBZ'),
        (r'.*\'s$','NN$'),
        (r'.*s$','NNS'),
        (r'.*', 'NN')  #未匹配的仍标注为NN
    ]
    brown_tagged_sents = brown.tagged_sents(categories='news')
    train_num = int(len(brown_tagged_sents) * 0.9)
    x_train =  brown_tagged_sents[0:train_num]
    x_test =   brown_tagged_sents[train_num:]
    
    t0 = nltk.RegexpTagger(pattern)
    t1 = nltk.UnigramTagger(x_train, backoff=t0)#t1的默认标注器为t0
    t2 = nltk.BigramTagger(x_train, backoff=t1)#t2的默认标注器的t1
    print(t2.evaluate(x_test))#0.863

def brillTagger():
    #templates = nltk.tag.brill.nltkdemo18()
    templates = nltk.tag.brill.nltkdemo18plus()
    trainer = nltk.brill_trainer.BrillTaggerTrainer(initial_tagger=nltk.DefaultTagger('NN'),#默认标注器“NN”
    templates=templates, trace=3,
    deterministic=True)
    
    brown_tagged_sents = brown.tagged_sents(categories='news')
    size = int(len(brown_tagged_sents) * 0.9)#90%
    train_sents = brown_tagged_sents[:size]#前90%用于训练
    test_sents = brown_tagged_sents[size:]#后10%用于测试
    
    tagger = trainer.train(train_sents, max_rules=20, min_score=2, min_acc=None)
    print(tagger.evaluate(test_sents))
    
def showPorterStemmer():
    sentence = "Listen, strange women lying in ponds \
    distributing swords. Supreme executive power derives \
    from a mandate from the masses, not from some farcical aquatic ceremony"
    sentences = nltk.sent_tokenize(sentence)#分句
    print(sentences)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]#分词
    print(sentences)
    for sent in sentences:#循环每一个句子
        # Porter
        porter = nltk.PorterStemmer()
        words = [porter.stem(word) for word in sent]
        print(words)

        # Lancaster
        lancaster = nltk.LancasterStemmer()
        lwords = [lancaster.stem(t) for t in sent]
        print(lwords)

        # WordNet
        wnl = nltk.WordNetLemmatizer()
        wwrods = [wnl.lemmatize(t) for t in sent]
        print(wwrods)    
    
def showWordNetLemmatizer():
    lemmatizer=WordNetLemmatizer()
    print(lemmatizer.lemmatize("cats"))#第二个参数，默认为NOUN
    print(lemmatizer.lemmatize("cacti"))#还原为单数名词：cactus
    print(lemmatizer.lemmatize("geese"))#还原为单数名词：goose
    print(lemmatizer.lemmatize("rocks"))#还原为单数名词：rock
    print(lemmatizer.lemmatize("pythonly"))#结果还是pythonly
    print(lemmatizer.lemmatize("better", pos="a"))#指定还原为形容词
    print(lemmatizer.lemmatize("best", pos="a"))
    print(lemmatizer.lemmatize("run"))
    print(lemmatizer.lemmatize("run",'v'))#指定还原为动词    
    
def showNER():
    sent = nltk.corpus.treebank.tagged_sents()[22]# 取出语料库中的一个句子
    print(sent)#[('The', 'DT'), ..., ('.', '.')]
    # 使用NE分块器进行命名实体识别，返回Tree对象，Tree对象的label()方法可以查看命名实体的标签
    for tree in nltk.ne_chunk(sent, binary=True).subtrees():
        # 过滤根树，否则会把整个句子输出
        if tree.label() == "S":
            continue
        print(tree)    
    

def showRelation():
    re_in = re.compile(r'.*\bin\b(?!\b.+ing)')#命名实体的关系是 X in Y
    for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):#从ieer预料集中读取NYT_19980315文件
        #print(doc.text)#这个是一个Tree类型的对象
        print(' '.join(doc.text.leaves()))#输出文本对象
        for rel in nltk.sem.extract_rels('ORGANIZATION', 'LOCATION', doc, corpus='ieer', pattern=re_in):
            print(rel)#输出：defaultdict(<class 'str'>, {'subjtext': 'WHYY', 'objtext': 'Philadelphia', 'untagged_filler': 'in', 'rcon': '. Ms.', 'objsym': 'philadelphia', 'subjclass': 'ORGANIZATION', 'lcon': 'station', 'subjsym': 'whyy', 'objclass': 'LOCATION', 'filler': 'in'})
            print(rel['subjtext'], 'IN', rel['objtext'])    
    
def showSentence():
    grammar = nltk.CFG.fromstring("""
      S -> NP VP
      VP -> V NP | V NP PP
      PP -> P NP
      V -> "saw" | "ate" | "walked"
      NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
      Det -> "a" | "an" | "the" | "my"
      N -> "man" | "dog" | "cat" | "telescope" | "park"
      P -> "in" | "on" | "by" | "with"
      """)
    sent = 'Mary saw Bob'.split()#按照空格分割
    rd_parser = nltk.RecursiveDescentParser(grammar)#递归下降分析
    for i in rd_parser.parse(sent):#按照上面的规则，对其中的每个成分进行分析
        print(i)#输出：(S (NP Mary) (VP (V saw) (NP Bob)))    

def showTreeBank():
    #t = treebank.parsed_sents('wsj_0001.mrg')[0]#读取文件
    #print(t) #查看封装好的文法
    
    def filter(tree):
        child_nodes = [child.label() for child in tree if isinstance(child,nltk.Tree)]
        return (tree.label() == 'VP') and ('S' in child_nodes)#找出带句子补语的动词
     
    #[subtree for tree in treebank.parsed_sents() for subtree in tree.subtrees(filter)]
        
def main():
    #statisticgutenberg()
    #showWordPunctTokenizer()
    #removeStopwords()
    #tagWords()
    #transformWords()
    #readLabeled()
    #wordTagger()
    #showTag()
    #labeledCorpus()
    #defaultTagger()
    #regTagger()
    #searchTagger
    #showUnigramTagger()
    #showBigramTagger()
    #combinedTagger()
    #brillTagger()
    #showPorterStemmer()
    #showWordNetLemmatizer()
    #showNER()
    #showRelation()
    #showSentence()
    showTreeBank()
    

if __name__ == '__main__':
    main()