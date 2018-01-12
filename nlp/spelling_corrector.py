import re, collections

def words(text):#返回文件中的所有词小写的List
    return re.findall('[a-z]+', text.lower()) #只包含了字母的

def train(features):
    model = collections.defaultdict(lambda: 0)#初始化一个空的字典,不管这个字典的key是什么，其值都是0
    for f in features:#统计文件中每个词的词频
        model[f] += 1
    return model#词频字典

file1=open('../file/test/big.txt','r')
NWORDS = train(words(file1.read()))#最后得到文件中每个词的词频

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]#逐步拆分，将这个词从第一个单词起一个单词一个单词的去拆分成两个词
    deletes    = [a + b[1:] for a, b in splits if b]#挨个删除其中的一个单词，返回List，即该词缺失一个单词的情况
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]#任意两个字母位置错误的情况
    replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]#长度为len(word)*26,将其中任意一个字母替换为a-z中的一个
    inserts    = [a + c + b     for a, b in splits for c in alphabet]#长度为（len(word)+1）*26，在任意一个地方插入一个a-z的字母
    return set(deletes + transposes + replaces + inserts)#数组之间可以直接加，返回不同数组的并集

def known_edits2(word):#判断该词经过上述的两次变换之后是否还在这个里面
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)#判断该词是否在词频字典中

def correct(word):
#     print(known([word]))#如果这个词在我们的词频字典里面则返回，（词频字典NWORDS只包含了）
#     print(known(edits1(word)))#
#     print(known_edits2(word))#
#     print([word])#当前这个词，这里是个List，和其他的不一样
    '''
    下面这句话是从左边到右边进行编译的，直到遇到第一个不为空的为止，并返回该值
    '''
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)#返回给定参数的最大值,get()函数返回指定键的值，如果值不在字典中返回默认值

def main():
    #print(correct("copyright"))#在原来的文本里面
    #print(correct("sssssssssssssss"))#返回该word自己
    print(correct('korrecter'))
    #print(correct('speling'))

if __name__ == '__main__':
    main()