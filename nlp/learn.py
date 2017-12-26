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
    words = [word.upper() for word in words]
    print(words[:100])
    
    nltk.download() 

if __name__ == '__main__':
    main()