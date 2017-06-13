# -*- coding: utf-8 -*-
from gensim import corpora
PREF_CUT = 'cut_search'
MIN_TOKEN_FREQ = 3
def makeDict(df, cut=PREF_CUT):
    dictionary = corpora.Dictionary(df[cut])
    dictionary.filter_extremes(no_below=MIN_TOKEN_FREQ,no_above=1)
    # corpus = [dictionary.doc2bow(s) for s in df[cut]]
    return dictionary.token2id
