# -*- coding: utf-8 -*-

# CONSTANTS
MIN_COMPLETION = .05
MIN_TOKEN_FREQ = 3
MIN_DISTINCT_CAT = 2
MIN_CONF = .3
PREF_CUT = 'cut_search'
SKU_NAME = 'ProductDesc'
N_FOLDS = 5

# columns to be omitted - common to every file, already filled in
OMIT_COLS = set([
    'ITEM_SKU_ID',
    'ProductDesc',
    'AFTER_PREFR_AMOUNT',
    'ITEM_FIRST_CATE_CD',
    'ITEM_FIRST_CATE_NAME',
    'ITEM_SECOND_CATE_CD',
    'ITEM_SECOND_CATE_NAME',
    'ITEM_THIRD_CATE_CD',
    'ITEM_THIRD_CATE_NAME',
    'cut_search',
    'cut_full',
    'cut_part'
])

# classifiers to use for learning
CLASSIFIERS = [
    OneVsRestClassifier(LinearSVC()),
    MultinomialNB(),
    MLPClassifier(early_stopping=False, hidden_layer_sizes=(4, 6), max_iter=500, alpha=1e-4,
                  solver='sgd', verbose=False, tol=1e-4, random_state=1,
                  learning_rate_init=.1),
    RandomForestClassifier(),
    KNeighborsClassifier(4)
]

# headers to show on screen
CLASSIFIER_NAMES = [
    '1VR_LSVC',
    'NaiveBayes',
    'MLPercep',
    'R_Forest',
    'KNN'
]

'''
==================
main process
==================
'''

import pandas as pd
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
