# -*- coding: utf-8 -*-

# CONSTANTS
MIN_COMPLETION = .05
MIN_TOKEN_FREQ = 3
MIN_DISTINCT_CAT = 2
MIN_CONF = .3
PREF_CUT = 'cut_search'
SKU_NAME = 'ProductDesc'
N_FOLDS = 5

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
