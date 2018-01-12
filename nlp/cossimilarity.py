'''
Created on 2017年12月27日

@author: Shaw Joe
'''
def main():
    v1=(1,1,1)
    v2=(-1,-1,-1)
    print(distance(v1,v2))
    
    v3=(1,1)
    v4=(-1,-1)
    print(distance(v3,v4))
    
    v1=(1,1)
    v2=(-1,-1)
    print(cos(v1,v2))
    
    v3=(0,1,1)
    v4=(0,-1,1)
    print(cos(v3,v4))

#-*-coding:utf-8-*-
def distance(vector1,vector2):
    d=0
    for a,b in zip(vector1,vector2):
        d+=(a-b)**2
    return d**0.5    

#-*-coding:utf-8-*-
def cos(vector1,vector2):
    dot_product = 0.0;
    normA = 0.0;
    normB = 0.0;
    for a,b in zip(vector1,vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB==0.0:
        return None
    else:
        return dot_product / ((normA*normB)**0.5)
    
if __name__ == '__main__':
    main()