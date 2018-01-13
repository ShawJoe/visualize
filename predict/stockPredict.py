'''
Created on 2018年1月12日

@author: Shaw Joe
'''
import tushare as ts#财经数据接口包
import nltk,random,os
from nltk.classify import apply_features
from idlelib.IOBinding import encoding

###---------------------------------------1、数据处理部分

def stock_industry_download():
    """
    下载股票行业分类数据
    """
    df = ts.get_industry_classified()
    df.to_csv('../file/Data/stock_industry.csv',encoding='utf-8')#加上一个编码的

def industry_stock_download(name):
    """
    下载某一行业的所有股票数据
    :param name: 行业名称
    """
    industry_data = stock_industry_load('../file/Data/stock_industry.csv')#一个文件的总表
    for stock in industry_data[name]:
        stock_code = stock[0]
        stock_name = stock[1]
        print(stock_code)
        print(stock_name)
        df = ts.get_hist_data(stock_code)#通过股票代码获取股票数据
        if df is None:
            continue
        df.to_csv('../file/Data/%s-%s-%s.csv' % (name, stock_code, stock_name.replace('*','')))#将获取得到的股票数据存为csv文件

def stock_industry_load(file_name):
    """
    加载股票行业分类信息
    :param file_name: 文件名
    :return: {'行业名':[('股票代码', '股票名'), ...]}
    """
    stock_industry = {}
    file = open(file_name.replace('*',''),encoding='utf-8')

    is_first_line = True
    for line in file:
        if is_first_line:
            is_first_line = False
            continue
        prop_list = line.split(',')
        industry_name = prop_list[3].strip()
        if industry_name not in stock_industry:
            stock_industry[industry_name] = []
        stock_industry[industry_name].append((prop_list[1], prop_list[2]))
    file.close()
    return stock_industry

def stock_load(file_name):
    """
    从指定数据文件中加载股票数据
    :param file_name: 文件名
    :return: 数据列表[(开盘价, 最高价, 收盘价, 最低价, 价格变动, 涨跌幅)]
    """
    stock_list = []
    if not os.path.exists(file_name.replace('*','')):
        return None#返回一个空值
    file = open(file_name.replace('*',''))

    is_first_line = True
    for line in file:
        if is_first_line:
            is_first_line = False
            continue
        prop_list = line.split(',')
        open_price = float(prop_list[1])
        high_price = float(prop_list[2])
        close_price = float(prop_list[3])
        low_price = float(prop_list[4])
        price_change = float(prop_list[6])
        p_change = int(float(prop_list[7]))
        stock_list.append((open_price, high_price, close_price, low_price, price_change, p_change))#一条数据存为一个元组
    file.close()
    return stock_list

##-----------------------------------------2、算法处理部分
 
def stock_split(data_list, days=5):#默认为前5天
    """
    股票数据分割，将某天涨跌情况和前几天数据关联在一起
    :param data_list: 股票数据列表
    :param days: 关联的天数
    :return: [([day1, day2, ...], label), ...]
    """
    stock_days = []
    #最后的五天的数据不要了
    for n in range(0, len(data_list)-days):#最近的数据在最前面，因此是将5天前的数据和当前的数据关联
        before_days = []
        for i in range(1, days+1):#将前面的五天的数据关联起来
            before_days.append(data_list[n + i])#五天的数据不包含当前天
        #n是当前天
        if data_list[n][4] > 0.0:#第四个数据项是价格变动
            label = '+'#标记为涨
        else:
            label = '-'#标记为跌
        stock_days.append((before_days, label))#组装为一个元组
    return stock_days    
    
def stock_feature(before_days):
    """
    股票特征提取
    :param before_days: 前几日股票数据
    :return: 股票特征
    """
    features = {}
    for n in range(0, len(before_days)):#将原来的[([day1, day2, ...], label), ...]数据转化为了字典类型
        stock = before_days[n]
        open_price = stock[0]
        high_price = stock[1]
        close_price = stock[2]
        low_price = stock[3]
        price_change = stock[4]
        p_change = stock[5]
        features['Day(%d)PriceIncrease' % (n + 1)] = (price_change > 0.0)
        features['Day(%d)High==Open' % (n + 1)] = (high_price == open_price)
        features['Day(%d)High==Close' % (n + 1)] = (high_price == close_price)
        features['Day(%d)Low==Open' % (n + 1)] = (low_price == open_price)
        features['Day(%d)Low==Close' % (n + 1)] = (low_price == close_price)
        features['Day(%d)Close>Open' % (n + 1)] = (close_price > open_price)
        features['Day(%d)PChange' % (n + 1)] = p_change
    return features    
    
def train_model(industry_name):
    """
    训练模型
    :param industry_name: 行业名称
    """
    stock_industry = stock_industry_load('../file/Data/stock_industry.csv')#返回一个字典类型的数据
    electronic_stocks = stock_industry[industry_name]#选取一个行业的数据

    stock_days = []
    totalLength=0
    for stock in electronic_stocks:
        file_name = '../file/Data/%s-%s-%s.csv' % (industry_name, stock[0], stock[1])
        stock_data = stock_load(file_name)#加载股票数据，返回：数据列表[(开盘价, 最高价, 收盘价, 最低价, 价格变动, 涨跌幅)]
        if stock_data is None:#当是空值的时候不管
            continue
        totalLength+=len(stock_data)
        stock_days += stock_split(stock_data)#返回[([day1, day2, ...], label), ...]，这里day1，day2...为前几天的数据，label为当前天的价格涨跌标签
    print(len(electronic_stocks))#共247只股票（两只没有信息，无法存储为文件），时间不一样这里的输出也不一样
    print(len(stock_days))#155280条信息
    print(totalLength)#247只股票共有156505条信息，156505=155280+（247-2）*5，这里每只股票的最后五天的数据都舍弃了（无法和其前五天进行关联）
    random.shuffle(stock_days)#随机将数据打乱
    train_set_size = int(len(stock_days) * 0.6)#获取前60%的长度
    train_stock = stock_days[:train_set_size]#前60%作为训练数据
    test_stock = stock_days[train_set_size:]#后40%作为测试数据
    #stock_feature是函数，train_stock作为参数传进了函数里面
    train_set = apply_features(stock_feature, train_stock, True)#nltk自带功能，自动生成特征集
    test_set = apply_features(stock_feature, test_stock, True)#其封装过程是在函数stock_feature中完成的
    print(test_set)#其基本形式为：字典类型
    #输出结果如下：[({'Day(3)High==Open': False, 'Day(5)PriceIncrease': True, 'Day(5)Low==Close': False, 'Day(1)PriceIncrease': True, 'Day(4)PChange': -1, 'Day(2)High==Close': False, 'Day(2)PriceIncrease': False, 'Day(3)High==Close': False, 'Day(5)High==Open': False, 'Day(1)High==Open': False, 'Day(4)Close>Open': False, 'Day(5)Close>Open': True, 'Day(1)Low==Open': False, 'Day(2)High==Open': True, 'Day(3)Low==Close': False, 'Day(3)Close>Open': False, 'Day(5)PChange': 1, 'Day(1)High==Close': False, 'Day(4)High==Open': False, 'Day(1)Close>Open': True, 'Day(3)PriceIncrease': False, 'Day(4)Low==Close': False, 'Day(3)PChange': 0, 'Day(2)Low==Open': False, 'Day(5)High==Close': False, 'Day(2)Close>Open': False, 'Day(2)Low==Close': False, 'Day(4)PriceIncrease': False, 'Day(1)Low==Close': False, 'Day(2)PChange': 0, 'Day(3)Low==Open': False, 'Day(5)Low==Open': False, 'Day(4)Low==Open': False, 'Day(1)PChange': 0, 'Day(4)High==Close': False}, '-'), ...]
    classifier = nltk.NaiveBayesClassifier.train(train_set)#使用朴素贝叶斯分类器
    print(nltk.classify.accuracy(classifier, train_set))#训练集的准确度
    print(nltk.classify.accuracy(classifier, test_set))#测试集的准确度
    classifier.show_most_informative_features(20)#输出20个最有用的特征
        
        
def main():
    #stock_industry_download()#先下载分类数据
    #industry_stock_download- ('电子信息')#再下载各个股票的数据
    train_model('电子信息')

if __name__ == '__main__':
    main()