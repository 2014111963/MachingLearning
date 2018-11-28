# -*- coding: utf-8 -*-
"""
Classifiers: dict
    功能 : 存储种群的所有解集合即sklearn的所有算法模型
author : zzc
date: 2018-5
"""
import numpy as np

from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AffinityPropagation

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

"""
设置所有搜索库的算法以及算法超参数选择，超参数取值，
共同组成所有解空间,整个搜索空间为一个二级大字典-AllModels，
里面的每一个算法又是一个字典，包括算法名字，超参数以及超参数的取值范围
"""
AllModels = {

    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
        },
    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
        },
    'sklearn.naive_bayes.GaussianNB': {
               },
    'sklearn.tree.DecisionTreeClassifier': {
        'criterion': ["gini", "entropy"],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
                 },
    'sklearn.ensemble.ExtraTreesClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21, 2),
        'min_samples_leaf': range(1, 21, 2),
        'bootstrap': [True, False]
    },
    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.1),
        'min_samples_split': range(2, 21, 3),
        'min_samples_leaf':  range(1, 21, 3),
        'bootstrap': [True, False]
    },
    'sklearn.ensemble.GradientBoostingClassifier': {
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21, 4),
        'min_samples_leaf': range(1, 21, 4),
        'subsample': np.arange(0.05, 1.01, 0.1),
        'max_features': np.arange(0.05, 1.01, 0.1)
    },
    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },
    'sklearn.svm.LinearSVC': {
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    },
    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    },

    'sklearn.linear_model.ElasticNetCV': {
        'l1_ratio': np.arange(0.0, 1.01, 0.05),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    },
    'sklearn.ensemble.ExtraTreesRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.1),
        'min_samples_split': range(2, 21, 2),
        'min_samples_leaf': range(1, 21, 2),
        'bootstrap': [True, False]
    },
    'sklearn.ensemble.GradientBoostingRegressor': {
        'loss': ["ls", "lad", "huber", "quantile"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5],
        'max_depth': range(1, 11, 4),
        'min_samples_split': range(2, 21, 4),
        'min_samples_leaf': range(1, 21, 4),
        'subsample': np.arange(0.5, 1.01, 0.12),
        'max_features': np.arange(0.5, 1.01, 0.12),
        'alpha': [0.7, 0.8, 0.9, 0.99]
    },
    'sklearn.ensemble.AdaBoostRegressor': {
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ["linear", "square", "exponential"]
    },
    'sklearn.tree.DecisionTreeRegressor': {
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },
    'sklearn.neighbors.KNeighborsRegressor': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },
    'sklearn.linear_model.LassoLarsCV': {
        'normalize': [True, False]
    },
    'sklearn.svm.LinearSVR': {
        'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },
    'sklearn.ensemble.RandomForestRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.1),
        'min_samples_split': range(2, 21, 2),
        'min_samples_leaf': range(1, 21, 2),
        'bootstrap': [True, False]
    },
    'sklearn.linear_model.RidgeCV': {
    },
    'sklearn.sklearn.cluster.KMeans': {
        'n_clusters': range(2, 16, 2),
        'init': ['k-means++', 'random'],
        'n_init': range(5, 15, 1)
    },
    'sklearn.sklearn.cluster.MiniBatchKMeans': {
        'n_clusters': range(2, 16, 2),
        'init': ['k-means++', 'random']
    },
    'sklearn.sklearn.cluster.AffinityPropagation': {
        'damping': np.arange(0.5, 1, 0.1),
        'max_iter': range(100, 200, 5)
    },
    'class sklearn.neural_network.MLPClassifier': {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
        'activation': ['identity', 'logistic', 'tanh', 'relu']
    },
    'class sklearn.neural_network.MLPRegressor': {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
        'activation': ['identity', 'logistic', 'tanh', 'relu']
    },
}

"""
根据上面设置的Python字典类型取到每一个算法的名字以及关联的超参数
及取值
"""
_BernoulliNB = AllModels.get('sklearn.naive_bayes.BernoulliNB')
_MultinomialNB = AllModels.get('sklearn.naive_bayes.MultinomialNB')
_GaussianNB = AllModels.get('sklearn.naive_bayes.GaussianNB')
_DecisionTreeClassifier = AllModels.get('sklearn.tree.DecisionTreeClassifier')
_ExtraTreesClassifier = AllModels.get('sklearn.ensemble.ExtraTreesClassifier')
_RandomForestClassifier = AllModels.get('sklearn.ensemble.RandomForestClassifier')
_GradientBoostingClassifier = AllModels.get('sklearn.ensemble.GradientBoostingClassifier')
_KNeighborsClassifier = AllModels.get('sklearn.neighbors.KNeighborsClassifier')
_LinearSVC = AllModels.get('sklearn.svm.LinearSVC')
_LogisticRegression = AllModels.get('sklearn.linear_model.LogisticRegression')

_ElasticNetCV = AllModels.get('sklearn.linear_model.ElasticNetCV')
_ExtraTreesRegressor = AllModels.get('sklearn.ensemble.ExtraTreesRegressor')
_GradientBoostingRegressor = AllModels.get('sklearn.ensemble.GradientBoostingRegressor')
_AdaBoostRegressor = AllModels.get('sklearn.ensemble.AdaBoostRegressor')
_DecisionTreeRegressor = AllModels.get('sklearn.tree.DecisionTreeRegressor')
_KNeighborsRegressor = AllModels.get('sklearn.neighbors.KNeighborsRegressor')
_LassoLarsCV = AllModels.get('sklearn.linear_model.LassoLarsCV')
_LinearSVR = AllModels.get('sklearn.svm.LinearSVR')
_RandomForestRegressor = AllModels.get('sklearn.ensemble.RandomForestRegressor')
_RidgeCV = AllModels.get('sklearn.linear_model.RidgeCV')

_MLPClassifier = AllModels.get('class sklearn.neural_network.MLPClassifier')
_MLPRegressor = AllModels.get('class sklearn.neural_network.MLPRegressor')
_KMeans = AllModels.get('sklearn.sklearn.cluster.KMeans')
_MiniBatchKMeans = AllModels.get('sklearn.sklearn.cluster.MiniBatchKMeans')
_AffinityPropagation = AllModels.get('sklearn.sklearn.cluster.AffinityPropagation')


"""
导入sklearn里面集成的数据测试系统性能
"""
iris = load_iris()
x_train1, x_test1, y_train1, y_test1 = train_test_split(iris.data, iris.target,
                                                        train_size=0.75, test_size=0.25)


def getmodel(x=0, isbest=False, x_train=x_train1, y_train=y_train1, x_test=x_test1, y_test=y_test1):
    """
    param x: int
           模型编码对应的整数
    param isbest : bool
           是否找到最优解
    param x_train, y_train : numpy
           训练每个解的得分的原始数据集
    param x_test, y_test : numpy
          测试每个解的得分的原始数据集
    return clf.score(X_test, y_test) : float
          模型的对于给定数据集适应度，预测准确率
    """

    populationcount = -1

    """
    循环遍历所有解空间，遍历字典
    """

    for _criterion in _DecisionTreeClassifier.get('criterion'):
        for _max_depth in _DecisionTreeClassifier.get('max_depth'):
            for _min_samples_split in _DecisionTreeClassifier.get('min_samples_split'):
                for _min_samples_leaf in _DecisionTreeClassifier.get('min_samples_leaf'):
                    populationcount += 1
                    if(x == populationcount and isbest is False):
                        clf = DecisionTreeClassifier(criterion=_criterion, max_depth=_max_depth,
                                                     min_samples_split=_min_samples_split,
                                                     min_samples_leaf=_min_samples_leaf)
                        """训练完成后计算得分并且返回"""
                        clf.fit(x_train, y_train)
                        return clf.score(x_test, y_test)
                    if(x == populationcount and isbest):
                        return (DecisionTreeClassifier(criterion=_criterion, max_depth=_max_depth,
                                                     min_samples_split=_min_samples_split,
                                                     min_samples_leaf=_min_samples_leaf))
                        return 1

    for _alpha in _BernoulliNB.get('alpha'):
        for _fit_prior in _BernoulliNB.get('fit_prior'):
            populationcount += 1
            if(x == populationcount and isbest is False):
                clf = BernoulliNB(alpha=_alpha, fit_prior=_fit_prior)
                clf.fit(x_train, y_train)
                return clf.score(x_test, y_test)
            if (x == populationcount and isbest):
                return (BernoulliNB(alpha=_alpha, fit_prior=_fit_prior))
                return 1

    for _alpha in _MultinomialNB.get('alpha'):
        for _fit_prior in _MultinomialNB.get('fit_prior'):
            populationcount += 1
            if(x == populationcount and isbest is False):
                clf = MultinomialNB(alpha=_alpha, fit_prior=_fit_prior)
                clf.fit(x_train, y_train)
                return clf.score(x_test, y_test)
            if (x == populationcount and isbest):
                return (MultinomialNB(alpha=_alpha, fit_prior=_fit_prior))
                return 1

    populationcount += 1
    if (x == populationcount and isbest is False):
        clf = GaussianNB()
        clf.fit(x_train, y_train)
        return clf.score(x_test, y_test)
    if (x == populationcount and isbest):
        return (GaussianNB())
        return 1

    for _n_estimators in _ExtraTreesClassifier.get('n_estimators'):
        for _criterion in _ExtraTreesClassifier.get('criterion'):
            for _max_features in _ExtraTreesClassifier.get('max_features'):
                for _min_samples_split in _ExtraTreesClassifier.get('min_samples_split'):
                    for _min_samples_leaf in _ExtraTreesClassifier.get('min_samples_leaf'):
                        for _bootstrap in _ExtraTreesClassifier.get('bootstrap'):
                            populationcount += 1
                            if (x == populationcount and isbest is False):
                                clf = ExtraTreesClassifier(n_estimators=_n_estimators, criterion=_criterion,
                                                           max_features=_max_features,
                                                           min_samples_split=_min_samples_split,
                                                           min_samples_leaf=_min_samples_leaf, bootstrap=_bootstrap)
                                clf.fit(x_train, y_train)
                                return clf.score(x_test, y_test)
                            if (x == populationcount and isbest):
                                return (ExtraTreesClassifier(n_estimators=_n_estimators, criterion=_criterion,
                                                           max_features=_max_features,
                                                           min_samples_split=_min_samples_split,
                                                           min_samples_leaf=_min_samples_leaf, bootstrap=_bootstrap))
                                return 1
    for _n_estimators in _RandomForestClassifier.get('n_estimators'):
        for _criterion in _RandomForestClassifier.get('criterion'):
            for _max_features in _RandomForestClassifier.get('max_features'):
                for _min_samples_split in _RandomForestClassifier.get('min_samples_split'):
                    for _min_samples_leaf in _RandomForestClassifier.get('min_samples_leaf'):
                        for _bootstrap in _RandomForestClassifier.get('bootstrap'):
                            populationcount += 1
                            if (x == populationcount and isbest is False):
                                clf = RandomForestClassifier(n_estimators=_n_estimators, criterion=_criterion,
                                                             max_features=_max_features,bootstrap=_bootstrap,
                                                             min_samples_split=_min_samples_split,
                                                             min_samples_leaf=_min_samples_leaf)
                                clf.fit(x_train, y_train)
                                return clf.score(x_test, y_test)
                            if (x == populationcount and isbest):
                                return (RandomForestClassifier(n_estimators=_n_estimators, criterion=_criterion,
                                                             max_features=_max_features,bootstrap=_bootstrap,
                                                             min_samples_split=_min_samples_split,
                                                             min_samples_leaf=_min_samples_leaf))
                                return 1

    for _learning_rate in _GradientBoostingClassifier.get('learning_rate'):
        for _max_depth in _GradientBoostingClassifier.get('max_depth'):
            for _min_samples_split in _GradientBoostingClassifier.get('min_samples_split'):
                for _min_samples_leaf in _GradientBoostingClassifier.get('min_samples_leaf'):
                    for _subsample in _GradientBoostingClassifier.get('subsample'):
                        for _max_features in _GradientBoostingClassifier.get('max_features'):
                            populationcount += 1
                            if (x == populationcount and isbest is False):
                                clf = GradientBoostingClassifier(learning_rate=_learning_rate, max_depth=_max_depth,
                                                                 min_samples_split=_min_samples_split,
                                                                 min_samples_leaf=_min_samples_leaf,
                                                                 subsample=_subsample, max_features=_max_features)
                                clf.fit(x_train, y_train)
                                return clf.score(x_test, y_test)
                            if (x == populationcount and isbest):
                                return (GradientBoostingClassifier(learning_rate=_learning_rate, max_depth=_max_depth,
                                                                 min_samples_split=_min_samples_split,
                                                                 min_samples_leaf=_min_samples_leaf,
                                                                 subsample=_subsample, max_features=_max_features))
                                return 1

    for _n_neighbors in _KNeighborsClassifier.get('n_neighbors'):
        for _weights in _KNeighborsClassifier.get('weights'):
            for _p in _KNeighborsClassifier.get('p'):
                populationcount += 1
                if (x == populationcount and isbest is False):
                    clf = KNeighborsClassifier(n_neighbors=_n_neighbors, weights=_weights, p=_p)
                    clf.fit(x_train, y_train)
                    return clf.score(x_test, y_test)
                if (x == populationcount and isbest):
                    return (KNeighborsClassifier(n_neighbors=_n_neighbors, weights=_weights, p=_p))
                    return 1

    for _loss in _LinearSVC.get('loss'):
        for _dual in _LinearSVC.get('dual'):
            for _tol in _LinearSVC.get('tol'):
                for _C in _LinearSVC.get('C'):
                    populationcount += 1
                    if(x == populationcount and isbest is False):
                        clf = LinearSVC(loss=_loss, dual=_dual, tol=_tol, C=_C)
                        clf.fit(x_train, y_train)
                        return clf.score(x_test, y_test)
                    if (x == populationcount and isbest):
                        return (LinearSVC(loss=_loss, dual=_dual, tol=_tol, C=_C))
                        return 1

    for _penalty in _LogisticRegression.get('penalty'):
        for _C in _LogisticRegression.get('C'):
            for _dual in _LogisticRegression.get('dual'):
                populationcount += 1
                if (x == populationcount and isbest is False):
                    clf = LogisticRegression(penalty=_penalty, C=_C, dual=_dual)
                    clf.fit(x_train, y_train)
                    return clf.score(x_test, y_test)
                if (x == populationcount and isbest):
                    return (LogisticRegression(penalty=_penalty, C=_C, dual=_dual))
                    return 1

    for _l1_ratio in _ElasticNetCV.get('l1_ratio'):
        for _tol in _ElasticNetCV.get('tol'):
            populationcount += 1
            if (x == populationcount and isbest is False):
                clf = ElasticNetCV(l1_ratio=_l1_ratio, tol=_tol)
                clf.fit(x_train, y_train)
                return clf.score(x_test, y_test)
            if (x == populationcount and isbest):
                return (ElasticNetCV(l1_ratio=_l1_ratio, tol=_tol))
                return 1

    for _n_estimators in _ExtraTreesRegressor.get('n_estimators'):
        for _max_features in _ExtraTreesRegressor.get('max_features'):
            for _min_samples_split in _ExtraTreesRegressor.get('min_samples_split'):
                for _min_samples_leaf in _ExtraTreesRegressor.get('min_samples_leaf'):
                    for _bootstrap in _ExtraTreesRegressor.get('bootstrap'):
                        populationcount += 1
                        if (x == populationcount and isbest is False):
                            clf = ExtraTreesRegressor(n_estimators=_n_estimators, max_features=_max_features,
                                                      min_samples_split=_min_samples_split,
                                                      min_samples_leaf=_min_samples_leaf,
                                                      bootstrap=_bootstrap)
                            clf.fit(x_train, y_train)
                            return clf.score(x_test, y_test)
                        if (x == populationcount and isbest):
                            return (ExtraTreesRegressor(n_estimators=_n_estimators, max_features=_max_features,
                                                      min_samples_split=_min_samples_split,
                                                      min_samples_leaf=_min_samples_leaf,
                                                      bootstrap=_bootstrap))
                            return 1

    for _learning_rate in _AdaBoostRegressor.get('learning_rate'):
        for _loss in _AdaBoostRegressor.get('loss'):
            populationcount += 1
            if (x == populationcount and isbest is False):
                clf = AdaBoostRegressor(learning_rate=_learning_rate, loss=_loss)
                clf.fit(x_train, y_train)
                return clf.score(x_test, y_test)
            if (x == populationcount and isbest):
                return (AdaBoostRegressor(learning_rate=_learning_rate, loss=_loss))
                return 1

    for _max_depth in _DecisionTreeRegressor.get('max_depth'):
        for _min_samples_split in _DecisionTreeRegressor.get('min_samples_split'):
            for _min_samples_leaf in _DecisionTreeRegressor.get('min_samples_leaf'):
                populationcount += 1
                if (x == populationcount and isbest is False):
                    clf = DecisionTreeRegressor(max_depth=_max_depth, min_samples_split=_min_samples_split,
                                                min_samples_leaf=_min_samples_leaf)
                    clf.fit(x_train, y_train)
                    return clf.score(x_test, y_test)
                if (x == populationcount and isbest):
                    return (DecisionTreeRegressor(max_depth=_max_depth, min_samples_split=_min_samples_split,
                                                min_samples_leaf=_min_samples_leaf))
                    return 1

    for _n_neighbors in _KNeighborsRegressor.get('n_neighbors'):
        for _weights in _KNeighborsRegressor.get('weights'):
            for _p in _KNeighborsRegressor.get('p'):
                populationcount += 1
                if (x == populationcount and isbest is False):
                    clf = KNeighborsRegressor(n_neighbors=_n_neighbors, weights=_weights, p=_p)
                    clf.fit(x_train, y_train)
                    return clf.score(x_test, y_test)
                if (x == populationcount and isbest):
                    return (KNeighborsRegressor(n_neighbors=_n_neighbors, weights=_weights, p=_p))
                    return 1

    for _normalize in _LassoLarsCV.get('normalize'):
        populationcount += 1
        if (x == populationcount and isbest is False):
            clf = LassoLarsCV(normalize=_normalize)
            clf.fit(x_train, y_train)
            return clf.score(x_test, y_test)
        if (x == populationcount and isbest):
            return (LassoLarsCV(normalize=_normalize))
            return 1

    for _loss in _LinearSVR.get('loss'):
        for _dual in _LinearSVR.get('dual'):
            for _tol in _LinearSVR.get('tol'):
                for _C in _LinearSVR.get('C'):
                    for _epsilon in _LinearSVR.get('epsilon'):
                        populationcount += 1
                        if(x == populationcount and isbest is False):
                            clf = LinearSVR(loss=_loss, dual=_dual, tol=_tol, C=_C, epsilon=_epsilon)
                            clf.fit(x_train, y_train)
                            return clf.score(x_test, y_test)
                        if (x == populationcount and isbest):
                            return (LinearSVR(loss=_loss, dual=_dual, tol=_tol, C=_C, epsilon=_epsilon))
                            return 1

    for _n_estimators in _RandomForestRegressor.get('n_estimators'):
        for _max_features in _RandomForestRegressor.get('max_features'):
            for _min_samples_split in _RandomForestRegressor.get('min_samples_split'):
                for _min_samples_leaf in _RandomForestRegressor.get('min_samples_leaf'):
                    for _bootstrap in _RandomForestRegressor.get('bootstrap'):
                        populationcount += 1
                        if (x == populationcount and isbest is False):
                            clf = RandomForestRegressor(n_estimators=_n_estimators,
                                                        max_features=_max_features, bootstrap=_bootstrap,
                                                        min_samples_split=_min_samples_split,
                                                        min_samples_leaf=_min_samples_leaf)
                            clf.fit(x_train, y_train)
                            return clf.score(x_test, y_test)
                        if (x == populationcount and isbest):
                            return (RandomForestRegressor(n_estimators=_n_estimators,
                                                        max_features=_max_features, bootstrap=_bootstrap,
                                                        min_samples_split=_min_samples_split,
                                                        min_samples_leaf=_min_samples_leaf))
                            return 1
    populationcount += 1
    if (x == populationcount and isbest is False):
        clf = RidgeCV()
        clf.fit(x_train, y_train)
        return clf.score(x_test, y_test)
    if (x == populationcount and isbest):
        return (RidgeCV())
        return 1

    for _alpha in _MLPClassifier.get('alpha'):
        for _activation in _MLPClassifier.get('activation'):
            populationcount += 1
            if (x == populationcount and isbest is False):
                clf = MLPClassifier(alpha=_alpha, activation=_activation)
                clf.fit(x_train, y_train)
                return clf.score(x_test, y_test)
            if (x == populationcount and isbest):
                return (MLPClassifier(alpha=_alpha, activation=_activation))
                return 1

    for _alpha in _MLPRegressor.get('alpha'):
        for _activation in _MLPRegressor.get('activation'):
            populationcount += 1
            if (x == populationcount and isbest is False):
                clf = MLPRegressor(alpha=_alpha, activation=_activation)
                clf.fit(x_train, y_train)
                return clf.score(x_test, y_test)
            if (x == populationcount and isbest):
                return (MLPRegressor(alpha=_alpha, activation=_activation))
                return 1

    for _n_clusters in _KMeans.get('n_clusters'):
        for _init in _KMeans.get('init'):
            for _n_init in _KMeans.get('n_init'):
                populationcount += 1
                if (x == populationcount and isbest is False):
                    clf = KMeans(n_clusters=_n_clusters, init=_init, n_init=_n_init)
                    clf.fit(x_train, y_train)
                    return clf.score(x_test, y_test)
                if (x == populationcount and isbest):
                    return (KMeans(n_clusters=_n_clusters, init=_init, n_init=_n_init))
                    return 1

    for _n_clusters in _MiniBatchKMeans.get('n_clusters'):
        for _init in _MiniBatchKMeans.get('init'):
            populationcount += 1
            if (x == populationcount and isbest is False):
                clf = MiniBatchKMeans(n_clusters=_n_clusters, init=_init)
                clf.fit(x_train, y_train)
                return clf.score(x_test, y_test)
            if (x == populationcount and isbest):
                return (MiniBatchKMeans(n_clusters=_n_clusters, init=_init))
                return 1

    for _damping in _AffinityPropagation.get('damping'):
        for _max_iter in _AffinityPropagation.get('max_iter'):
            populationcount += 1
            if (x == populationcount and isbest is False):
                clf = AffinityPropagation(damping=_damping, max_iter=_max_iter)
                clf.fit(x_train, y_train)
                return clf.score(x_test, y_test)
            if (x == populationcount and isbest):
                return (AffinityPropagation(damping=_damping, max_iter=_max_iter))
                return 1

    for _learning_rate in _GradientBoostingRegressor.get('learning_rate'):
        for _max_depth in _GradientBoostingRegressor.get('max_depth'):
            for _min_samples_split in _GradientBoostingRegressor.get('min_samples_split'):
                for _min_samples_leaf in _GradientBoostingRegressor.get('min_samples_leaf'):
                    for _subsample in _GradientBoostingRegressor.get('subsample'):
                        for _max_features in _GradientBoostingRegressor.get('max_features'):
                            for _loss in _GradientBoostingRegressor.get('loss'):
                                for _alpha in _GradientBoostingRegressor.get('alpha'):
                                    populationcount += 1
                                    if (x == populationcount and isbest is False):
                                        clf = GradientBoostingRegressor(learning_rate=_learning_rate,
                                                                        max_depth=_max_depth,
                                                                        min_samples_split=_min_samples_split,
                                                                        min_samples_leaf=_min_samples_leaf,
                                                                        subsample=_subsample,
                                                                        max_features=_max_features,
                                                                        loss=_loss, alpha=_alpha)

                                        clf.fit(x_train, y_train)
                                        return clf.score(x_test, y_test)
                                    if (x == populationcount and isbest):
                                        return (GradientBoostingRegressor(learning_rate=_learning_rate,
                                                                        max_depth=_max_depth,
                                                                        min_samples_split=_min_samples_split,
                                                                        min_samples_leaf=_min_samples_leaf,
                                                                        subsample=_subsample,
                                                                        max_features=_max_features,
                                                                        loss=_loss, alpha=_alpha))
                                        return 1


if __name__ == '__main__':
    X = []
    Y = []
    print(getmodel(x=0, isbest=True))

"""
    for i in range(1):
        X.append(i)
        print(getmodel(i))
        Y.append(getmodel(i))
    plt.plot(X, Y)
    plt.show()
"""