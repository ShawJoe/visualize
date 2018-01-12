'''
Created on 2018年1月12日

@author: Shaw Joe
'''
import nltk,re
from nltk.corpus import conll2000

def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document) #分句
    sentences = [nltk.word_tokenize(sent) for sent in sentences] #分词
    sentences = [nltk.pos_tag(sent) for sent in sentences] #词性标注

def showchunk():
    # 分词
    text = "the little yellow dog barked at the cat"
    sentence = nltk.word_tokenize(text)
    
    # 词性标注
    sentence_tag = nltk.pos_tag(sentence)
    print(sentence_tag)
    
    # 定义分块语法
    # 这个规则是说一个NP块由一个可选的限定词后面跟着任何数目的形容词然后是一个名词组成
    # NP(名词短语块) DT(限定词) JJ(形容词) NN(名词)
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    
    # 进行分块
    cp = nltk.RegexpParser(grammar)#正则表达式分块器
    tree = cp.parse(sentence_tag)#分块
    tree.draw()#画图
    
def showchunk2():
    # 分词
    text = "Lucy let down her long golden hair"
    sentence = nltk.word_tokenize(text)
    
    # 词性标注
    sentence_tag = nltk.pos_tag(sentence)
    print(sentence_tag)#显示标注过的词
    
    # 定义分块语法
    # NNP(专有名词) PRP$(格代名词)
    # 第一条规则匹配可选的词（限定词或格代名词），零个或多个形容词，然后跟一个名词
    # 第二条规则匹配一个或多个专有名词
    # $符号是正则表达式中的一个特殊字符，必须使用转义符号\来匹配PP$
    grammar = r"""
        NP: {<DT|PRP\$>?<JJ>*<NN>}
            {<NNP>+}
    """
    
    # 进行分块
    cp = nltk.RegexpParser(grammar)#正则表达式分块器
    tree = cp.parse(sentence_tag)#分块
    tree.draw()#画图显示    

def showchunk3():
    # 分词
    text = "the little yellow dog barked at the cat"
    sentence = nltk.word_tokenize(text)
    
    # 词性标注
    sentence_tag = nltk.pos_tag(sentence)
    print(sentence_tag)
    
    # 定义缝隙语法
    # 第一个规则匹配整个句子
    # 第二个规则匹配一个或多个动词或介词
    # 一对}{就代表把其中语法匹配到的词作为缝隙
    grammar = r"""
    NP: {<.*>+}
        }<VBD|IN>+{
    """
    cp = nltk.RegexpParser(grammar)#分块器
    
    # 分块
    tree = cp.parse(sentence_tag)#分块
    tree.draw()#画图显示

def showchunk4():
    tree1 = nltk.Tree('NP',['Alick'])
    print(tree1)#输出：(NP Alick)
    tree2 = nltk.Tree('N',['Alick','Rabbit'])
    print(tree2)#输出：(N Alick Rabbit)
    tree3 = nltk.Tree('S',[tree1,tree2])
    print(tree3.label()) #查看树的结点S
    tree3.draw()#画图显示

def showchunk5():
    print(conll2000.chunked_sents('train.txt')[99])#查看已经分块的一个句子
    text = """
he /PRP/ B-NP
accepted /VBD/ B-VP
the DT B-NP
position NN I-NP
of IN B-PP
vice NN B-NP
chairman NN I-NP
of IN B-PP
Carlyle NNP B-NP
Group NNP I-NP
, , O
a DT B-NP
merchant NN I-NP
banking NN I-NP
concern NN I-NP
. . O
    """#这里的字符串必须顶格，否则正则表达式会匹配错误
    ##下面几段是conllstr2tree中的代码，是为了找出其中的问题
    _LINE_RE = re.compile('(\S+)\s+(\S+)\s+([IOB])-?(\S+)?')
    for lineno, line in enumerate(text.split('\n')):
        if not line.strip(): continue
        match = _LINE_RE.match(line.strip())
        #print(match)
        #print(lineno,'#'+line.replace('\t','').strip()+'#')
    result = nltk.chunk.conllstr2tree(text,chunk_types=['NP'], root_label="S")#分块
    print(result)#输出分块结果
    result.draw()


class UnigramChunker(nltk.ChunkParserI):
    """
        一元分块器，
        该分块器可以从训练句子集中找出每个词性标注最有可能的分块标记，
        然后使用这些信息进行分块
    """
    def __init__(self, train_sents):
        """
            构造函数
            :param train_sents: Tree对象列表
        """
        train_data = []
        for sent in train_sents:
            # 将Tree对象转换为IOB标记列表[(word, tag, IOB-tag), ...]
            conlltags = nltk.chunk.tree2conlltags(sent)
            # 找出每个词性标注对应的IOB标记
            ti_list = [(t, i) for w, t, i in conlltags]
            train_data.append(ti_list)
        # 使用一元标注器进行训练
        self.__tagger = nltk.UnigramTagger(train_data)

    def parse(self, tokens):
        """
            对句子进行分块
            :param tokens: 标注词性的单词列表
            :return: Tree对象
        """
        # 取出词性标注
        tags = [tag for (word, tag) in tokens]
        # 对词性标注进行分块标记
        ti_list = self.__tagger.tag(tags)
        # 取出IOB标记
        iob_tags = [iob_tag for (tag, iob_tag) in ti_list]
        # 组合成conll标记
        conlltags = [(word, pos, iob_tag) for ((word, pos), iob_tag) in zip(tokens, iob_tags)]
        return nltk.chunk.conlltags2tree(conlltags)

def showchunk6():
    test_sents = conll2000.chunked_sents("test.txt", chunk_types=["NP"])#读取测试集,为tree对象
    train_sents = conll2000.chunked_sents("train.txt", chunk_types=["NP"])#读取训练集,为tree对象
    
    unigram_chunker = UnigramChunker(train_sents)#分块器
    print(unigram_chunker.evaluate(test_sents))#输出对分块器的衡量结果
    
        

def main():
    #showchunk()
    #showchunk2()
    #showchunk3()
    #showchunk4()
    #showchunk5()
    showchunk6()
    

if __name__ == '__main__':
    main()