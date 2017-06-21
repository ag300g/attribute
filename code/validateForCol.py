# -*- coding: utf-8 -*-
# run all classifiers & evaluate each for specified column

# CONSTANTS
N_FOLDS = 5

CLASSIFIERS = [
    # OneVsOneClassifier(LinearSVC()),
    OneVsRestClassifier(LinearSVC()),
    MultinomialNB(),
    MLPClassifier(early_stopping=False, hidden_layer_sizes=(4, 6), max_iter=500, alpha=1e-4,
                  solver='sgd', verbose=False, tol=1e-4, random_state=1,
                  learning_rate_init=.1),
    RandomForestClassifier(),
    KNeighborsClassifier(4)
]

CLASSIFIER_NAMES = [
    # '1V1_LSVC',
    '1VR_LSVC',
    'NaiveBayes',
    'MLPercep',
    'R_Forest',
    'KNN'
]
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
