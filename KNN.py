# -*- code: utf-8 -*-
"""
KNN算法核心思想：
找出距离待分类的点最近的K个点
根据K个点中每个点在K个点中出现的概率，将待分类的点归为概率最大的点所在的分类
此处计算的距离用的是常用的欧氏距离
"""
from numpy import *
from os import listdir
import operator


def classify(in_x, data_set, labels, k):
    data_set_size = data_set.shape[0]
    # 列数不变，行数复制data_set_size倍
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    # 计算距离，先乘方，后开方
    sq_diff_mat = diff_mat**2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances**0.5
    # 获得distances 中按降序排列的索引
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
        # operator.itemgetter(1) 按字典值排序，reverse是否按降序排列字典
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    # 返回概率最大的分类
    return sorted_class_count[0][0]


def img2vector(filename):
    # 将文件中的32x32的矩阵转换为1x1024的矩阵
    return_vector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line_str = fr.readline()
        for j in range(32):
            return_vector[0, 32*i+j] = int(line_str[j])
    return return_vector


def handwriting_recoginze():
    hw_labels = []
    train_file_list = listdir('trainingDigits')
    m = len(train_file_list)
    train_mat = zeros((m, 1024))
    for i in range(m):
        # 按照trainDigits目录下的文件初始化训练数据，及分类标签
        filename_str = train_file_list[i]
        file_str = filename_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        # train_mat 训练矩阵
        train_mat[i, :] = img2vector('trainingDigits/%s' % filename_str)
    test_file_list = listdir('testDigits')
    # 错误计数
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        # 初始化测试集
        filename_str = test_file_list[i]
        file_str = filename_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        vector_under_test = img2vector('testDigits/%s' % filename_str)
        # 预测结果
        classifier_result = classify(vector_under_test, train_mat, hw_labels, 3)
        if classifier_result != class_num_str:
            error_count += 1.0
        print("分类结果: %d, 实际值: %d" % (classifier_result, class_num_str))
    print("\n 总错误数: %d" % error_count)
    print("\n 错误率: %f " % (error_count/m_test))
