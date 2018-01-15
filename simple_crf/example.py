from simple_crf.crf import *
from collections import defaultdict
import re
import sys


def get_feature_functions(word_sets, labels, observes):
    """生成各种特征函数"""
    print("get feature functions ...")
    transition_functions = [
        lambda yp, y, x_v, i, _yp=_yp, _y=_y: 1 if yp == _yp and y == _y else 0
        for _yp in labels[:-1] for _y in labels[1:]
        ]
    
    def set_membership(tag, word_sets):
        print('##')
        def fun(yp, y, x_v, i):
            print('$$')
            if i < len(x_v) and x_v[i].lower() in word_sets[tag]:
                return 1
            else:
                return 0
        return fun

    observation_functions = [set_membership(t, word_sets) for t in word_sets]

    misc_functions = [
        lambda yp, y, x_v, i: 1 if i < len(x_v) and re.match('^[^0-9a-zA-Z]+$', x_v[i]) else 0,
        lambda yp, y, x_v, i: 1 if i < len(x_v) and re.match('^[A-Z\.]+$', x_v[i]) else 0,
        lambda yp, y, x_v, i: 1 if i < len(x_v) and re.match('^[0-9\.]+$', x_v[i]) else 0
    ]

    tagval_functions = [
        lambda yp, y, x_v, i, _y=_y, _x=_x: 1 if i < len(x_v) and y == _y and x_v[i].lower() == _x else 0
        for _y in labels
        for _x in observes]
    
    print(transition_functions)

    return transition_functions + tagval_functions + observation_functions + misc_functions


if __name__ == '__main__':
    word_data = []#词
    label_data = []#标签
    all_labels = set()#所有的标签
    word_sets = defaultdict(set)#默认字典类型的数据
    observes = set()
    for line in open("sample.txt"):#将数据拆分
        words, labels = [], []
        for token in line.strip().split():#空格进行分割
            word, label = token.split('/')#将词和标签
            all_labels.add(label)#所有标签的集合
            word_sets[label].add(word.lower())#标签-词的字典，
            observes.add(word.lower())#保存单词小写的集合
            words.append(word)#保存所有的单词
            labels.append(label)#保存所有的标签

        word_data.append(words)#二维数组，按行存储单词
        label_data.append(labels)#二维数组，按行存储标签


    labels = [START, END] + list(all_labels)#将所有标签的集合变成List，并加上开始和结束标签
    feature_functions = get_feature_functions(word_sets, labels, observes)
#     print(feature_functions)
#     for f in feature_functions:
#         print(f)
 
#     crf = CRF(labels=labels, feature_functions=feature_functions)
#     crf.train(word_data, label_data)#训练
#     for x_vec, y_vec in zip(word_data[-5:], label_data[-5:]):
#         print(x_vec, y_vec)
#         print("raw data: ", x_vec)
#         print("prediction: ", crf.predict(x_vec))
#         print("ground truth: ", y_vec)


