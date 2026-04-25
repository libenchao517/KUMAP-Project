################################################################################
# 本文件用于实现Kernel UMAP算法的核函数
################################################################################
# 导入模块
import numpy as np
import torch as tc
from DR import Neighborhood_Preserving_Embedding
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
################################################################################
class core:
    """
    KUMAP的核函数
    """
    def __init__(self, Alpha=0.1):
        """
        初始化函数
        :param Alpha: 核函数的超参数
        """
        self.replace_inf = float('inf')
        self.replace_0 = 0.0
        self.Alpha = Alpha

    def Gauss(self, D):
        """
        高斯核函数
        :param D: 成对距离矩阵
        :return:
        """
        temp = tc.where(D == 0.0, self.replace_inf, D)
        # 计算到最近邻点的距离
        D_min = tc.unsqueeze(tc.min(temp, dim=0).values, dim=0).expand(D.shape[0], -1)
        D = tc.where(tc.isinf(temp), self.replace_0, temp)
        # 计算核矩阵
        K = tc.exp(-0.5 * tc.square(D) / tc.square(D_min) / self.Alpha ** 2)
        # 归一化
        K_sum = tc.sum(K, dim=0)
        K_result = K / K_sum
        return K_result

    def Exponential(self, D):
        """
        指数核函数
        :param D: 成对距离矩阵
        :return:
        """
        temp = tc.where(D == 0.0, self.replace_inf, D)
        # 计算到最近邻点的距离
        D_min = tc.unsqueeze(tc.min(temp, dim=0).values, dim=0).expand(D.shape[0], -1)
        D = tc.where(tc.isinf(temp), self.replace_0, temp)
        # 计算核矩阵
        K = tc.exp(-0.5 * D / D_min / self.Alpha ** 2)
        # 归一化
        K_sum = tc.sum(K, dim=0)
        K_result = K / K_sum
        return K_result
################################################################################
class KUMAP_Supervised_Labels:
    """
    定义SKUMAP的新标签
    """
    def __init__(self, model = 'PCA'):
        """
        初始化函数
        :param model: 计算标签的方法
        """
        self.model = model

    def Make_label(self, data, target, label_n, target_n):
        """
        生产监督学习的标签
        :param data:     数据集
        :param target:   旧标签
        :param label_n:  新标签的维度
        :param target_n: 数据集中的类别数
        :return:
        """
        label = self.choice_model(data=data, target=target, label_n=label_n)
        for i in range(target_n):
            index = target == i
            temp = label[index]
            avg = np.mean(temp, axis=0)
            label[index] = avg
        return label

    def choice_model(self, data, target, label_n=10):
        """
        选择监督算法
        :param data:     数据集
        :param target:   旧标签
        :param label_n:  新标签的维度
        :return:
        """
        if self.model.upper() == 'PCA':
            return self.PCA_label(data, label_n=label_n)
        elif self.model.upper() == 'LDA':
            label_n = min(label_n,len(set(target))-1)
            return self.LDA_label(data, target, label_n=label_n)
        elif self.model.upper() == 'NPE':
            return self.NPE_label(data, label_n=label_n)

    def PCA_label(self, data, label_n=10):
        """
        使用PCA算法进行监督
        :param data:
        :param label_n:
        :return:
        """
        return PCA(n_components=label_n).fit_transform(data)

    def LDA_label(self, data, target, label_n=10):
        """
        使用LDA算法进行监督
        :param data:
        :param target:
        :param label_n:
        :return:
        """
        return LinearDiscriminantAnalysis(n_components=label_n).fit_transform(data, target)

    def NPE_label(self, data, label_n=10):
        """
        使用NPE算法进行监督
        :param data:
        :param label_n:
        :return:
        """
        return Neighborhood_Preserving_Embedding(n_components=label_n).fit_transform(data)
