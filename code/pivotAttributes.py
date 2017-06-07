# -*- coding: UTF-8 -*-
import pandas as pd
def pivotAttributes(raw_df, VERBOSE=False, outputExcel=False):
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

    df.to_csv('pivoted.txt', sep='\t', index=False)
    #exclude columns with too much missing information
    skuCount = len(df)
    for col in df:
        if (float)(sum(df[col].notnull())) / (float)(skuCount) < 0.05:
            del df[col]
            if VERBOSE:
                print('col %s is skipped' % (col,))

    #optional excel spreadsheet as intermediate output
    if outputExcel:
        writer = pd.ExcelWriter('default.xlsx')
        df.to_excel(writer, 'Sheet1')
        writer.save()
    return df, maps
