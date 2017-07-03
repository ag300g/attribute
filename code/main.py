# -*- coding: utf-8 -*-
## 所用到的包
import copy
import jieba
import numpy as np
import pandas as pd
from gensim import corpora
from scipy import sparse
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from tabulate import tabulate

# from sys import path
# import os
# pth=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath("")))))
# path.append(pth)
from auxiliary import fileManagement as fm

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
======================================================
main process
======================================================
'''

def main(df, Scenario):

    ## 第一个模块
    writer = pd.ExcelWriter('outputfor2676.xlsx')
    df, maps = pivotAttributes(df, writer,VERBOSE=True, outputExcel=True)

    ## 第二个模块
    df = tokenise(df)

    ## 第三个模块
    df = runModels(df)

    df.to_excel(writer, 'afterfill')
    writer.save()





def pivotAttributes(raw_df, writer, VERBOSE=False, outputExcel=False):
    #specify columns to create mappings for
    mapNames = {'ProductKey': 'ProductDesc', 'AttributeKey': 'AttributeDesc', 'AttributeValueKey': 'AttributeValueDesc'}
    #save and return dropped key columns in dict object
    maps = {}

    #construct skuKey: skuName, attKey: attName, attValKey: attValName maps
    for key, desc in mapNames.items():
        #groupby and take max to produce mappings
        tmpMap = raw_df[[key, desc]].groupby([key])
        tmpMap = tmpMap[desc].max() #requires uniqueness of keys
        tmpMap_df = pd.DataFrame(tmpMap)
        tmpMap_df.reset_index(inplace=True)
        tmpMap_df.set_index([key], drop=False, inplace=True)
        maps[key] = tmpMap_df

        #test if id and names are bijective (generally not true)
        if VERBOSE and tmpMap_df.shape[0] != raw_df[desc].nunique():
            print('WARNING: {} & {} not bijective'.format(key, desc))

    #pivot, taking the max for multi-value attribute values
    raw_df_grp = raw_df[['ProductDesc', 'AttributeDesc', 'AttributeValueDesc']].groupby(['ProductDesc', 'AttributeDesc'])
    raw_df_grp = raw_df_grp['AttributeValueDesc'].max()
    df = pd.DataFrame(raw_df_grp)
    df.reset_index(inplace=True)
    df = df.pivot(index='ProductDesc', columns='AttributeDesc', values='AttributeValueDesc')
    df.reset_index(inplace=True)

    if outputExcel:
        df.to_excel(writer, '去重后的原始数据')
    #exclude columns with too much missing information
    skuCount = len(df)
    for col in df:
        if (float)(sum(df[col].notnull())) / (float)(skuCount) < MIN_COMPLETION:
            del df[col]
            if VERBOSE:
                print('col %s is skipped' % (col,))

    #optional excel spreadsheet as intermediate output
    if outputExcel:
        df.to_excel(writer, '删掉率填充率低的列')
    return df, maps


def tokenise(df, colname=SKU_NAME):
    cuts = {
        'cut_search': lambda s: jieba.lcut_for_search(s),
        'cut_full': lambda s: jieba.lcut(s, cut_all=True),
        'cut_part': lambda s: jieba.lcut(s, cut_all=False)
    }
    for cut_name, cut in cuts.items():
        df[cut_name]=df[colname].map(cut)
    return df



def runModels(df):
    vocab = makeDict(df)
    vectorizer = CountVectorizer(min_df=1, vocabulary=vocab)  ## min_df和max_df在有参考的字典时没有用
    for colname in list(df):
        # fails for (nearly) single-valued columns

        # 列中除了None之外值域小于2的列不进行任何操作
        if colname in OMIT_COLS or df[colname].nunique() < MIN_DISTINCT_CAT:
            continue  ## these cols do not apply the following operation

        known_df = df[df[colname].notnull()]
        unknown_df = df[df[colname].isnull()]
        # transform X to vector representation
        known_corpus = known_df[PREF_CUT].map(lambda s: ' '.join(s))
        known_vec = vectorizer.fit_transform(known_corpus)
        known_X = sparse.csr_matrix(known_vec.toarray())

        unknown_corpus = unknown_df[PREF_CUT].map(lambda s: ' '.join(s))
        unknown_vec = vectorizer.transform(unknown_corpus)  # don't re-fit
        unknown_X = sparse.csr_matrix(unknown_vec.toarray())

        # transform y to vector representation
        known_y_raw = known_df[colname].values
        le = preprocessing.LabelEncoder()
        le = le.fit(known_y_raw)
        known_y = le.transform(known_y_raw)

        # validate all models
        wts = validateForCol(colname, known_X, known_y)

        # make predictions & update dataframe
        df = makePred(df, unknown_df, colname, known_X, known_y, unknown_X, le, wts)

    return df


def makeDict(df, cut=PREF_CUT):
    dictionary = corpora.Dictionary(df[cut])
    dictionary.filter_extremes(no_below=MIN_TOKEN_FREQ,no_above=1)
    return dictionary.token2id


def validateForCol(colname, known_X, known_y):
    kf = KFold(known_X.shape[0], n_folds=N_FOLDS, shuffle=True)
    score_raw = np.zeros((N_FOLDS, len(CLASSIFIERS)))
    fold = 1
    score_mat = []

    # randomly generate all training & test sets
    for train_index, test_index in kf:
        train_X = known_X[train_index]
        train_y = known_y[train_index]
        test_X = known_X[test_index]
        test_y = known_y[test_index]

        # train & score all training & test sets
        classifiers = copy.deepcopy(CLASSIFIERS)
        for i in range(len(classifiers)):
            classifiers[i].fit(train_X, train_y)
            score_raw[fold - 1, i] = classifiers[i].score(test_X, test_y)
        score_mat.append(['Fold ' + str(fold) + ':'] + \
                         ['{:.1%}'.format(x) for x in list(score_raw[fold - 1, :])])
        fold += 1

    # tabulate results
    headers = [colname] + CLASSIFIER_NAMES
    results = tabulate(score_mat, headers=headers)

    # weights normalised by product of N_FOLDS & no. of CLASSIFIERS
    wts = sum(np.multiply(score_raw, score_raw)) / (N_FOLDS * len(CLASSIFIERS))
    print('\n{}'.format(results))
    return wts



def makePred(df, unknown_df, colname, known_X, known_y, unknown_X, le, wts):
    classifiers = copy.deepcopy(CLASSIFIERS)

    # more confident for more complete attributes
    completeness = (len(df.index) - sum(df[colname].isnull())) / len(df.index)
    wts = [wt * completeness for wt in wts]

    # set up empty list of dictionaries for voting
    predictionsForAll = []

    for j in range(len(unknown_df)):
        predictionsforOne = {}
        predictionsForAll.append(predictionsforOne)
        # predictionsForAll is list whose length = unknown_df.shape[0]
        # the element of predictionsForAll is a dictionary of the prediction result with all possibilities

    # fit each classifier independently
    for i in range(len(classifiers)):
        classifiers[i].fit(known_X, known_y)
        y = le.inverse_transform(classifiers[i].predict(unknown_X))  # 通过le.inverse_transform可以把y的标签重新变为y的值

        # count up votes from different classifiers
        # if 2 classifier vote for one result then sum the wts of the 2 classifier as the result's weight
        # if 2 classifier vote for different result then list a new possibility
        for j in range(len(y)):
            if y[j] in predictionsForAll[j]:
                predictionsForAll[j][y[j]] += wts[i]
            else:
                predictionsForAll[j][y[j]] = wts[i]

    # take the most popular option & record score
    yMerged = [max(y_j, key=y_j.get) for y_j in predictionsForAll]
    yMergedConf = [predictionsForAll[j][yMerged[j]] for j in range(len(yMerged))]

    for j in range(len(yMerged)):
        if yMergedConf[j] < MIN_CONF:
            yMerged[j] = ''

    unknown_rows = list(unknown_df.index.values)

    print('before fill {} nulls in col {}'.format(sum(df[colname].isnull()), colname))

    for j in range(len(yMerged)):
        df.set_value(unknown_rows[j], colname, yMerged[j])

    print('after fill {} left blank in col {}'.format(sum(df[colname] == ''), colname))
    return df





if __name__ == '__main__':
    # get scenario if running this module outside of main.py
    # scenario = settings.getCurrentScenario()

    # 数据来源：
    # select sku.item_sku_id, sku_name, ext_attr_cd, ext_attr_name, ext_attr_value_cd, ext_attr_value_name from gdm_m03_item_sku_da sku
    # join
    # gdm_m03_sku_ext_attr_da att
    # on sku.dt='2017-04-01'
    # and att.dt='2017-04-01'
    # and sku.item_third_cate_cd = 2676
    # and sku.item_sku_id=att.item_sku_id;

    # scenarioSettingsPath = '../settings/settings_fillAttributes.yaml'

    # settingsScenario = fm.loadSettingsFromYamlFile(scenarioSettingsPath)

    df = pd.read_table('input.txt', quotechar='\0', dtype={'item_sku_id': str, 'ext_attr_cd': str, 'ext_attr_value_cd': str})
    df.columns = ['ProductKey', 'ProductDesc', 'AttributeKey', 'AttributeDesc', 'AttributeValueKey', 'AttributeValueDesc']

    main(df)