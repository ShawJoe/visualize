'''
Created on 2018年1月12日

@author: Shaw Joe
'''
import nltk,os
from urllib import request

def remoteData():
    url = "http://www.gutenberg.org/files/12345/12345.txt"
    response = request.urlopen(url)
    raw = response.read().decode('utf8')
    print(type(raw))#输出：<class 'str'>
    print(raw)#输出文本信息
    
def readLocalData():
    #这里的路径好像必须是绝对路径
    rootdir=os.getcwd()
    pdir=os.path.dirname(rootdir)
    cdir=os.path.join(pdir,'file\\test\\pg5200.txt')
    filepath=os.path.abspath(cdir)
    print(filepath)
    path = nltk.data.find(filepath)  
    raw = open(path, 'rU').read()  
    print(raw)    
    
    

def main():
    #remoteData()
    readLocalData()

if __name__ == '__main__':
    main()