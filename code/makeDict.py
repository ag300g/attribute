# -*- coding: utf-8 -*-
from gensim import corpora
PREF_CUT = 'cut_search'
MIN_TOKEN_FREQ = 3  ## user defined

def makeDict(df, cut=PREF_CUT):
    dictionary = corpora.Dictionary(df[cut])
    dictionary.filter_extremes(no_below=MIN_TOKEN_FREQ,no_above=1)
    return dictionary.token2id
