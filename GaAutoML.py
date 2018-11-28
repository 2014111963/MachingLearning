# -*- coding: utf-8 -*-
"""
classname: AutoML
    功能 : 使用遗传算法的思想实现机器学习
          自动选取学习模型，以及超参数的调整
author : zzc
date: 2018-4

"""

import math
import random
import datetime
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from AtuoML.LoadDataSet import readlocaldataset
from sklearn.model_selection import train_test_split
from AtuoML.ModelLibraries import getmodel


class AutoML:
    """使用遗传算法的思想实现机器学习自动选取学习模型，以及超参数的调整"""
    iris = load_iris()
    x_train1, x_test1, y_train1, y_test1 = train_test_split(iris.data, iris.target,
                                                            train_size=0.75, test_size=0.25)

    def __init__(self, generation_size=20, population_size=50, mutation=0.01,
                 x_train=x_train1, y_train=y_train1, x_test=x_test1, y_test=y_test1,
                 crossover=0.6, min_value=0, max_value=262143, genecoding_length=18):

        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._generation_size = generation_size
        self._population_size = population_size
        self._mutation = mutation
        self._crossover = crossover
        self._max_value = max_value
        self._min_value = min_value
        self._best_solutions = [[]]
        self._fit_value = []
        self._genecoding_length = genecoding_length
        """调用genecoding函数生成种群初始个体基因"""
        self._populations = self.genecoding(population_size, genecoding_length)
        print(self._populations)
        """遗传算法使用的相关参数以及初始设置 

        参数:
        ---------------------
        _generation_size: int
            种群遗传迭代的次数
        _population_size: int
            每一代的种群数量
        _variation: float
            种群变异概率
        _mating: float
            种群父母交叉概率
        _best_solutions: 双重列表[[]]
            存储最优基因以及对应的最优解
        _genecoding_length: int
            基因编码长度
        _min_value :int 
            解的最小值
        _max_value :int 
            解的最大值
        """

    def genecoding(self, population_size, genecodingsize):
        """
        parameters:
        ---------------------
          population_size: int
               种群数量大小
          genecodingsize: int
               染色体长度
        return populations: [](list)
        随机编码的第一代种群基因
        """
        populations = []
        """ 创建列表，保存初始的种群"""
        for i in range(population_size):
            population = []
            """每一个个体的初始化"""
            for j in range(genecodingsize):
                population.append(random.randint(0, 1))
            populations.append(population)
        return populations

    def fit(self):
        """
        找到最优解的函数
        return _best_solutions: [[]](双重列表)
           返回每一代的最优值与最优解
        """
        for i in range(self._generation_size):
            """individualfitness存储当代种群里面每个个体的适应度值大小"""

            individualfitness = self.fitnessfunction(self._populations, self._genecoding_length,
                                                     self._min_value, self._max_value)

            best_individual, best_fitvalue = self.bestindividual(self._populations, individualfitness)
            """存储每一代的最优值"""
            self._best_solutions.append([best_fitvalue, self.getbestsolution(best_individual, self._min_value,
                                                                             self._max_value, self._genecoding_length)])


            """选择"""

            self._populations = self.selection(self._populations, individualfitness)
            """交叉"""
            self._populations = self.crossovering(self._populations, self._crossover)
            """变异"""
            self._populations = self.mutating(self._populations, self._mutation)

        self._best_solutions = self._best_solutions[1:]
        self._best_solutions.sort()
        return self._best_solutions

    def fitnessfunction(self, populations, genecoding_length, min_value, max_value):
        """
        parameters:
        ---------------------
          populations: int
               当前迭代的种群
          genecoding_length: int
               个体染色体长度
          min_value:  int
               搜索解空间的最小值
          max_value: int
               搜索解空间的最大值
        return individualfitvalue:  [](list)
             当前种群个体适应度
        """
        individualfitvalue = []
        """调用decodinggenes函数解码基因"""
        tempindividual = self.decodinggenes(populations, genecoding_length)
        for i in range(len(tempindividual)):
            x = min_value + tempindividual[i] * (max_value - min_value) \
                / (math.pow(2, genecoding_length) - 1)
            """调用score函数计算个体适应度大小"""
            individualfitvalue.append(getmodel(x=x, x_train=self._x_train, y_train=self._y_train,
                                               x_test=self._x_test, y_test=self._y_test))
        return individualfitvalue

    def decodinggenes(self, populations, genecoding_length):
        """
        parameters:
        ---------------------
        populations: [](list)
             当前种群的基因编码列表
        genecoding_length: int
              个体染色体长度
        return individualfitvalue: [](列表)
              解码后对应的自变量值
        """
        individualfitvalue = []
        for i in range(len(populations)):
            decodinggenes_x = 0
            for j in range(genecoding_length):
                decodinggenes_x += populations[i][j] * (math.pow(2, genecoding_length - 1 - j))
            """添加解码后的自变量值"""
            individualfitvalue.append(decodinggenes_x)
        return individualfitvalue

    def getbestsolution(self, bestindividual, min_value, max_value, genecoding_length):
        """
        parameters:
        ---------------------
        param bestindividual: [](列表)
              当代种群每个个体适应度值大小
        param min_value: int
              搜索解空间的最小值
        param max_value: int
              搜索解空间的最大值
        param genecoding_length: int
              个体染色体长度
        return value: float
              当代种群适应度值最大的个体
        """
        value = 0
        for j in range(len(bestindividual)):
            value += bestindividual[j] * (math.pow(2, genecoding_length - 1 - j))
        value = min_value + value * (max_value - min_value) / (math.pow(2, genecoding_length) - 1)
        return value

    def bestindividual(self, populations, individualfitness):
        """
        parameters:
        ---------------------
        param populations:
        param individualfitness: [](列表)
              当代种群个体适应度值列表
        return [best_individual, best_fit]: [(,)]列表二元组
             包含最优解与最优解基因
        """
        pcount = len(populations)
        best_individual = []
        best_fit = individualfitness[0]
        for i in range(1, pcount):
            if (individualfitness[i] > best_fit):
                best_fit = individualfitness[i]
                best_individual = populations[i]
        return [best_individual, best_fit]

    def sumfitvalue(self, individualfitness):
        """
        param individualfitness: [](列表)
              当代种群个体适应度值列表
        return total : float
              返回种群适应度总和
        """
        total = 0
        """积累叠加概率"""
        for i in range(len(individualfitness)):
            total += individualfitness[i]
        return total

    def selection(self, population, individualfitness):
        """
        parameters:
        ---------------------
        param populations:
        param individualfitness: [](列表)
              当代种群个体适应度值列表
        return newpopulations [](列表)
              选择当代最优一部分个体
        """
        newfit_value = []
        """种群的适应度总和"""
        total_fitvalue = self.sumfitvalue(individualfitness)
        for i in range(len(individualfitness)):
            newfit_value.append(individualfitness[i] / total_fitvalue)
        """计算累计概率"""
        newfit_value = self.cumsum(newfit_value)
        ms = []
        pop_len = len(population)
        for i in range(pop_len):
            """产生随机数，用来确定选择范围"""
            ms.append(random.random())
        ms.sort()
        fitin = 0
        newin = 0
        newpopulations = population
        """转轮盘选择法"""
        while newin < pop_len:
            if (ms[newin] < newfit_value[fitin]):
                newpopulations[newin] = population[fitin]
                newin = newin + 1
            else:
                fitin = fitin + 1

        return newpopulations

    def cumsum(self, individualfitness):
        """
        parameters:
        ---------------------
        param individualfitness: [](列表)
              排序完成后的当代种群个体适应度值列表
        return b: [](列表)
               计算累加概率，比如[0.1, 0.2, 0.3, 0.4] 累加之后为 [0.1, 0.3, 0.6, 1.0]
        """
        b = []
        for i in range(1, len(individualfitness)):
            b.append(sum(individualfitness[0:i]))
        b.append(1)

        return b

    def crossovering(self, population, crossover):
        """
        parameters:
        ---------------------
        param population:
        param crossover: float
              父母交叉的概率
        return population: [](列表)
           根据交配概率，随机选择父代交配产生子代
        """
        pop_length = len(population)
        for i in range(pop_length - 2):
            if (random.random() < crossover):
                """随机生成染色体的交配节点"""
                matingpoint = random.randint(0, len(population[0]))
                """子代定义"""
                child1 = []
                child2 = []
                """分别获得父母双方的一段基因序列"""
                child1.extend(population[i][0:matingpoint])
                child1.extend(population[i + 2][matingpoint:self._genecoding_length])
                child2.extend(population[i + 2][0:matingpoint])
                child2.extend(population[i][matingpoint:self._genecoding_length])
                """代替父母作为子代种群"""
                if(getmodel(int(self.getindividualvalue(population[i]))) <=
                   getmodel(int(self.getindividualvalue(child1)))):
                    population[i] = child1
                if (getmodel(int(self.getindividualvalue(population[i+2]))) <=
                        getmodel(int(self.getindividualvalue(child2)))):
                    population[i + 2] = child2
        return population

    def getindividualvalue(self, individual):
        """
        param individual:
              需要计算的个体0、1串即二进制值
        return: individualvalue
              个体的实际实数值
        """
        individualvalue = 0
        for j in range(len(individual)):
            individualvalue += individual[j] * (math.pow(2, len(individual) - 1 - j))
        return individualvalue

    def mutating(self, population, mutation, ):
        """
        parameters:
        ---------------------
        param population:
        param mutation: float
             种群突变概率大小
        return population: []
              随机突变后的种群
        """
        population_size = len(population)

        for i in range(population_size):
            if (random.random() < mutation):
                """随机选取突变节点"""
                mpoint = random.randint(0, self._genecoding_length - 1)
                if (population[i][mpoint] == 1):
                    population[i][mpoint] = 0
                else:
                    population[i][mpoint] = 1

        return population


if __name__ == '__main__':
    t1 = datetime.datetime.now()
    iris1 = load_digits()


    data = pd.read_csv('C:/Users/24465/Desktop/test.csv', header=None)
    train_x = data[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    train_y = data[[16]]

    iris = load_iris()
    #train_x, train_y = readlocaldataset('D:/PycharmProjects/AtuoML/traindata/')
    x_train1, x_test1, y_train1, y_test1 = train_test_split(iris1.data, iris1.target,
                                                            train_size=0.75, test_size=0.25)
    automl = AutoML(generation_size=20, population_size=10, x_train=x_train1, y_train=y_train1, x_test=x_test1,
                    y_test=y_test1)
    results = automl.fit()
    t2 = datetime.datetime.now()
    print(results[-1])
    print("耗时为：%s 秒" % (t2 - t1))
    print("算法搜索到的最优模型以及模型参数为：")
    algorithm = getmodel(int(results[-1][1]), True)
    print(algorithm)









