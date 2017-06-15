# -*- coding: utf-8 -*-
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

    # fit each classifier independently
    for i in range(len(classifiers)):
        classifiers[i].fit(known_X, known_y)
        y = le.inverse_transform(classifiers[i].predict(unknown_X))

        # count up votes from different classifiers
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
