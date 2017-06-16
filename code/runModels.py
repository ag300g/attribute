# -*- coding: utf-8 -*-
"""
====================
core method
====================
need method:
            > makeDict
            > CountVectorizer
            > validateForCol
            > makePred

need costant:
            > MIN_DISTINCT_CAT
            > OMIT_COLS
            > PREF_CUT
"""

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
