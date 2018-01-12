'''
Created on 2018年1月12日

@author: Shaw Joe
'''
import spacy
from collections import Counter
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import string
from spacy.en import English

noisy_pos_tags = ["PROP"]
min_token_length = 2
nlp = spacy.load('en')
filename='../file/test/big.txt'
document = open(filename,encoding='utf-8').read()
document = nlp(document)


#检查 token 是不是噪音的函数
def isNoise(token):
    is_noise = False
    if token.pos_ in noisy_pos_tags:
        is_noise = True
    elif token.is_stop == True:
        is_noise = True
    elif len(token.string) <= min_token_length:
        is_noise = True
    return is_noise

def cleanup(token, lower = True):
    if lower:
        token = token.lower()
    return token.strip()

def showCommon():
    print(document)
    #一些参数定义
    # 评论中最常用的单词
    cleaned_list = [cleanup(word.string) for word in document if not isNoise(word)]
    result=Counter(cleaned_list).most_common(5)
    print(result)
    
def NRE():
    labels = set([w.label_ for w in document.ents])
    for label in labels:
        entities = [cleanup(e.string, lower=False) for e in document.ents if label==e.label_]
        entities = list(set(entities))
        print(label,entities)

def showSentences():
    # 取出所有句中包含“hotel”单词的评论
    hotel = [sent for sent in document.sents if 'hotel' in sent.string.lower()]
    
    # 创建依存树
    sentence = hotel[2]
    for word in sentence:
        print(word, ': ', str(list(word.children)))

# 检查修饰某个单词的所有形容词
def pos_words (sentence, token, ptag):
    sentences = [sent for sent in sentence.sents if token in sent.string]     
    pwrds = []
    for sent in sentences:
        for word in sent:
            if token in word.string:
                pwrds.extend([child.string.strip() for child in word.children if child.pos_ == ptag] )
    return Counter(pwrds).most_common(10)

def showHotel():
    result=pos_words(document, 'hotel', "ADJ")
    print(result)

def outputNP():
    nlp = spacy.load('en')
    # 生成名词短语
    doc = nlp(u'I love data science on analytics vidhya')
    for np in doc.noun_chunks:
        print(np.text, np.root.dep_, np.root.head.text)

def showSimilarWords():
    parser = English()
    # 生成“apple”的词向量 
    apple = parser.vocab[u'apple']
    # 余弦相似性计算函数
    cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
    others = list({w for w in parser.vocab if w.has_vector and w.orth_.islower() and w.lower_ != "apple"})
    # 根据相似性值进行排序
    others.sort(key=lambda w: cosine(w.vector, apple.vector))
    others.reverse()
    print("top most similar words to apple:")
    for word in others[:10]:
        print(word.orth_)


# 使用 spaCy 自定义 transformer
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

# 进行文本清洗的实用的基本函数
def clean_text(text):     
    return text.strip().lower()

punctuations = string.punctuation
parser = English()
    
#创建 spaCy tokenizer，解析句子并生成 token
#也可以用词向量函数来代替它
def spacy_tokenizer(sentence):
    tokens = parser(sentence)
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
    return tokens

#创建 vectorizer 对象，生成特征向量，以此可以自定义 spaCy 的 tokenizer
vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
classifier = LinearSVC()

def machinelearning():
    # 创建管道，进行文本清洗、tokenize、向量化、分类操作
    pipe = Pipeline([("cleaner", predictors()),
                     ('vectorizer', vectorizer),
                     ('classifier', classifier)])

    # Load sample data
    train = [('I love this sandwich.', 'pos'),          
             ('this is an amazing place!', 'pos'),
             ('I feel very good about these beers.', 'pos'),
             ('this is my best work.', 'pos'),
             ("what an awesome view", 'pos'),
             ('I do not like this restaurant', 'neg'),
             ('I am tired of this stuff.', 'neg'),
             ("I can't deal with this", 'neg'),
             ('he is my sworn enemy!', 'neg'),          
             ('my boss is horrible.', 'neg')]
    test =   [('the beer was good.', 'pos'),     
             ('I do not enjoy my job', 'neg'),
             ("I ain't feelin dandy today.", 'neg'),
             ("I feel amazing!", 'pos'),
             ('Gary is a good friend of mine.', 'pos'),
             ("I can't believe I'm doing this.", 'neg')]
    
    # 创建模型并计算准确率
    pipe.fit([x[0] for x in train], [x[1] for x in train])
    pred_data = pipe.predict([x[0] for x in test])
    for (sample, pred) in zip(test, pred_data):
        print(sample, pred)
    print("Accuracy:", accuracy_score([x[1] for x in test], pred_data))
    



def main():
    #showCommon()
    #NRE()
    #showSentences()
    #showHotel()#输出[('our', 1), ('clever', 1), ('select', 1), ('expensive', 1)]
    #outputNP()
    #showSimilarWords()
    machinelearning()

if __name__ == '__main__':
    main()