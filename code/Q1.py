'''
===================================================
### 特征抽取时的问题
### 一个小例子
===================================================
'''

from gensim import corpora
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
# from scipy import sparse
a = ['爱','苹果','我们','他们',',',' ','吃','苹果']
b = ['我们', '他', '他们', ',', '吃']
c = ['我','他们', '香蕉',',','苹果表']
d = ['我', '香蕉',',','苹果', '爱']
e = ['L', '4.4']
ab = pd.Series([a,b,c,d,e])
dictionary = corpora.Dictionary(ab)
dictionary.filter_extremes(no_below=2, no_above=1)
vocab=dictionary.token2id
print(vocab)
vectorizer = CountVectorizer(min_df=1, vocabulary=vocab)

p1 = '我 爱 吃 香蕉'
p2 = '他们 爱 吃 苹果 苹果'
p3 = '吃 爱 , 吃 , 苹果表'
p4 = '爱 我们 吃 香蕉'
p5 = '4.4 L'
paper = pd.Series([p1, p2, p3, p4, p5])
known_vec = vectorizer.fit_transform(paper)
known_vec1 = vectorizer.transform(paper)
print(vocab)
print(vectorizer.get_feature_names())
print(known_vec.toarray())   ## why there is not the 吃 爱 , ?
print(known_vec1.toarray())
