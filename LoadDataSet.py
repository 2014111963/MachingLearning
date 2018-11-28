# -*- coding: utf-8 -*-

"""
#类名: ReadLocalDataSet
#函数作用: 根据文件路径加载本地数据集
Author：zzc
Time：2018-4-10
"""

import numpy as np
from os import listdir


def converttovector(filename):
	"""
	param filename:
	文件路径名
	return:
	所有数据的1*1024的向量
	"""
	datavector = np.zeros([1024], int)
	try:
		datavector = np.zeros([1024], int)          # 定义返回的矩阵向量，大小为1*1024
		openfile = open(filename)                  # 打开包含32*32大小的数字文件
		datalines = openfile.readlines()          # 读取文件的所有行
		for i in range(32):                      # 遍历文件所有行
			for j in range(32):                 # 并将01数字存放在矩阵向量DataVector中
				datavector[i * 32 + j] = datalines[i][j]
	except IOError:
		print("Error: 没有找到文件或读取文件失败")
		
	return datavector


def readlocaldataset(filepathname):
	"""
	param filepathname:
		文件名
	return: traindataset, numberlabels
		训练的每个数据以及对应的分类标签
	"""
	# 存储label数字标签
	numberlabels = [] #数据集的分类标签
	traindataset = np.zeros([1, 1024], int)
	try:
		filelist = listdir(filepathname)                     # 获取文件夹下的所有文件名
		allnumberfiles = len(filelist)                        # 统计需要读取的文件的数目
		traindataset = np.zeros([allnumberfiles, 1024], int)  # 用于存放所有的数字文件
		for i in range(allnumberfiles):                                # 遍历所有的文件
			numberfilepath = filelist[i]                               # 获取文件名称/路径
			numberlabels.append(int(numberfilepath.split('_')[0]))     # 通过文件名获取标签
			traindataset[i] = converttovector(filepathname + '/' + numberfilepath)  # 读取文件内容
				
	except IOError:
		print("Error: 没有找到文件或读取文件失败")
	return traindataset, numberlabels

