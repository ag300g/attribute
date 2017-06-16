# -*- coding: utf-8 -*-
# import copy
# import os
# import sys
#
import jieba
# import numpy as np
import pandas as pd
# from gensim import corpora
# from scipy import sparse
# from sklearn import preprocessing
# from sklearn.cross_validation import KFold
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.multiclass import OneVsOneClassifier
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import LinearSVC
# from tabulate import tabulate

# CONSTANTS
MIN_COMPLETION = .05
MIN_TOKEN_FREQ = 3
MIN_DISTINCT_CAT = 2
MIN_CONF = .3
PREF_CUT = 'cut_search'
SKU_NAME = 'ProductDesc'
N_FOLDS = 5
# main process
df = pd.read_table('input.txt', quotechar='\0', dtype={'item_sku_id': str, 'ext_attr_cd': str, 'ext_attr_value_cd': str})
df.columns = ['ProductKey', 'ProductDesc', 'AttributeKey', 'AttributeDesc', 'AttributeValueKey', 'AttributeValueDesc']

## 第一个模块
from pivotAttributes import *
df, maps = pivotAttributes(df, VERBOSE=True, outputExcel=True)

## 第二个模块
from tokenise import *
df = tokenise(df)

## 第三个模块
from runModels import *
df = runModels(df)

## 第四个模块
from writeAll import *
writeAll(df)

## 第五个模块
from unpivotAttributes import *
df = unpivotAttributes(df, maps)

df.to_csv('unpivot.csv')
