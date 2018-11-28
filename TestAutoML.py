# -*- coding: utf-8 -*-
"""
filename: TestAutoML
    功能 : 测试GaAutoML功能
author : zzc
date: 2018-4

"""
import datetime
from AtuoML.ModelLibraries import getmodel
from AtuoML import GaAutoML
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from AtuoML.LoadDataSet import readlocaldataset

"""
train_DataSet, trainLabelsList = readlocaldataset('D:/PycharmProjects/AtuoML/traindata')
X_train, X_test, y_train, y_test = train_test_split(train_DataSet, trainLabelsList,
                                                    train_size=0.75, test_size=0.25)

"""
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    train_size=0.75, test_size=0.25)

t1 = datetime.datetime.now()
clf = GaAutoML.AutoML(generation_size=4, population_size=30, x_train=X_train, y_train=y_train,
                      x_test=X_test, y_test=y_test)

results = clf.fit()
t2 = datetime.datetime.now()
print(results[-1])
print("耗时为：%s 秒" % (t2-t1))
print("算法搜索到的最优模型以及模型参数为：")
getmodel(results[-1][1], True)
