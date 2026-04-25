################################################################################
# 本文件用于对Kernel UMAP算法的评价进行标准化
################################################################################
# 导入模块
import os
import numpy as np
import pandas as pd
from .Assessment import K_Nearest_Neighbors
from .Assessment import Normalized_Mutual_Info_Score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")
################################################################################
class Analysis_Kernel_Manifold:
    """
    核方法流形学习标准分析过程
    """
    def __init__(self, object):
        """
        初始化函数
        :param object: 核方法的实例化对象
        """
        self.object = object
        # 初始化指标
        self.knn = K_Nearest_Neighbors()
        self.nmi = Normalized_Mutual_Info_Score()
        # 初始化结果表格
        self.result = pd.DataFrame(
            columns=['Method', 'Datasets', 'KNN', "PRE", "REC", "F1", 'NMI', 'time', 'CPU', 'GPU'],
            index=['Total', 'Train', 'Test'])
        self.xlsx_path = "-".join(self.object.para[0:4]) + '.xlsx'
        self.xn = 80

    def Analysis(self):
        """
        结果分析的标准化过程
        :return:
        """
        os.makedirs(os.path.join("Analysis", self.object.data_name), exist_ok=True)
        print("*" * self.xn)
        print(self.object.para[2] + "算法在" + self.object.para[3] + "数据集上的降维效果定量评价报告")
        print("*" * self.xn)
        # 分析总体的分类性能
        self.knn.KNN_predict_odds_splited(
            self.object.Y_train, self.object.Y_test, self.object.T_train,
            self.object.T_test, name=None)
        self.result['Method'].Total = self.object.func_name
        self.result['Datasets'].Total = self.object.data_name
        classification_label = self.knn.t_pred
        self.result['KNN'].Total = accuracy_score(self.object.T_test, classification_label)
        self.result['PRE'].Total = precision_score(self.object.T_test, classification_label, average="macro")
        self.result['REC'].Total = recall_score(self.object.T_test, classification_label, average="macro")
        self.result['F1'].Total = f1_score(self.object.T_test, classification_label, average="macro")
        self.result['time'].Total = self.object.time
        self.result['CPU'].Total = self.object.CPU
        self.result['GPU'].Total = self.object.GPU

        print(self.result)
        print("*" * self.xn)
        self.result.to_excel(os.path.join("Analysis", self.object.data_name, self.xlsx_path))
################################################################################
class Analysis_Kernel_Paras:
    """
    分析方法的超参数
    """
    def __init__(self, object, para, paraname):
        """
        初始化函数
        :param object: 算法的实例化对象
        :param para: 可用的参数值列表
        :param paraname: 参数名称
        """
        self.object = object
        self.para = para
        self.paraname = paraname
        self.knn = K_Nearest_Neighbors()
        self.nmi = Normalized_Mutual_Info_Score()
        self.lts = len(self.para)
        self.KNN = np.zeros(self.lts)
        self.PRE = np.zeros(self.lts)
        self.REC = np.zeros(self.lts)
        self.F1 = np.zeros(self.lts)
        self.NMI = np.zeros(self.lts)
        self.time = np.zeros(self.lts)
        self.CPU = np.zeros(self.lts)
        self.GPU = np.zeros(self.lts)
        self.reaults = pd.DataFrame(
            columns=['KNN', "PRE", "REC", "F1", 'NMI', 'time', 'CPU', 'GPU'],
            index=list(self.para))

    def Analysis(self):
        """
        分析的标准化过程
        :return:
        """
        for i in range(self.lts):
            setattr(self.object, self.paraname, self.para[i])
            self.object.fit_transform()
            self.knn.KNN_predict_odds_splited(
                self.object.Y_train, self.object.Y_test, self.object.T_train,
                self.object.T_test, name=None)
            classification_label = self.knn.t_pred
            self.KNN[i] = accuracy_score(self.object.T_test, classification_label)
            self.PRE[i] = precision_score(self.object.T_test, classification_label, average="macro")
            self.REC[i] = recall_score(self.object.T_test, classification_label, average="macro")
            self.F1[i] = f1_score(self.object.T_test, classification_label, average="macro")
            self.time[i] = self.object.time
            self.CPU[i] = self.object.CPU
            self.GPU[i] = self.object.GPU

            self.reaults.loc[self.para[i], 'KNN'] = self.KNN[i]
            self.reaults.loc[self.para[i], 'PRE'] = self.PRE[i]
            self.reaults.loc[self.para[i], 'REC'] = self.REC[i]
            self.reaults.loc[self.para[i], 'F1'] = self.F1[i]
            self.reaults.loc[self.para[i], 'time'] = self.time[i]
            self.reaults.loc[self.para[i], 'CPU'] = self.CPU[i]
            self.reaults.loc[self.para[i], 'GPU'] = self.GPU[i]
