'''
Created on 2017年12月26日

@author: Shaw Joe
'''
import nltk

def main():
    filename = '../file/test/pg5200.txt'
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    words = text.split()
    nltk.download() 

if __name__ == '__main__':
    main()