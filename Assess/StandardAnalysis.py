################################################################################
# 本文件用于对Kernel UMAP算法的对比算法的评价进行标准化
################################################################################
# 导入模块
import os
import pandas as pd
from .Assessment import K_Nearest_Neighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
################################################################################
class Analysis_Standard_Contrast_Manifold:
    """
    流形学习对比算法标准分析过程
    """
    def __init__(self, object):
        """
        初始化函数
        :param object: 流形学习方法的实例化对象
        """
        self.object = object
        self.knn = K_Nearest_Neighbors()
        self.result = pd.DataFrame(
            columns=['Method', 'Datasets', 'KNN', "PRE", "REC", "F1", 'time', 'CPU', 'GPU'],
            index=['Total'])
        self.xlsx_path = "-".join(self.object.para[0:4]) + '.xlsx'
        self.xn = 80

    def Analysis(self):
        """
        标准化分析过程
        :return:
        """
        os.makedirs(os.path.join("Analysis", self.object.data_name), exist_ok=True)
        print("*" * self.xn)
        print(self.object.para[2] + "算法在" + self.object.para[3] + "数据集上的降维效果定量评价报告")
        print("*" * self.xn)
        if getattr(self.object, "Y_train") is None:
            y_tr, y_te, t_tr, t_te = train_test_split(self.object.embedding, self.object.target, train_size=0.2)
        else:
            y_tr, y_te, t_tr, t_te =  self.object.Y_train, self.object.Y_test, self.object.T_train, self.object.T_test
        self.knn.KNN_predict_odds_splited(y_tr, y_te, t_tr, t_te, name=None)
        classification_label = self.knn.t_pred
        self.result['Method'].Total = self.object.func_name
        self.result['Datasets'].Total = self.object.data_name
        self.result['KNN'].Total = accuracy_score(t_te, classification_label)
        self.result['PRE'].Total = precision_score(t_te, classification_label, average="macro")
        self.result['REC'].Total = recall_score(t_te, classification_label, average="macro")
        self.result['F1'].Total = f1_score(t_te, classification_label, average="macro")
        self.result['time'].Total = self.object.time
        self.result['CPU'].Total = self.object.CPU
        self.result['GPU'].Total = self.object.GPU

        print(self.result)
        print("*" * self.xn)
        self.result.to_excel(os.path.join("Analysis", self.object.data_name, self.xlsx_path))
################################################################################
class Analysis_OOS_Methods:
    """
    局外样本点嵌入方法分析的标准化过程
    """
    def __init__(self, object):
        """
        初始化函数
        :param object: 算法的实例化对象
        """
        self.object = object
        self.knn = K_Nearest_Neighbors()
        self.result = pd.DataFrame(columns=['Method', 'Datasets', 'KNN', 'PRE', 'REC', 'F1', 'time'], index=['Total'])
        self.xlsx_path = "-".join(self.object.para[0:4]) + '.xlsx'
        self.xn = 80

    def Analysis(self, classification=True):
        """
        标准化分析过程
        :param classification:
        :return:
        """
        os.makedirs(os.path.join("Analysis", self.object.data_name), exist_ok=True)
        print("*" * self.xn)
        print(self.object.para[2] + "算法在" + self.object.para[3] + "数据集上的降维效果定量评价报告")
        print("*" * self.xn)
        self.knn.KNN_predict_odds_splited(
            self.object.Y_train, self.object.Y_test,
            self.object.T_train, self.object.T_test, name=None)
        classification_label = self.knn.t_pred
        self.result['Method'].Total = self.object.func_name
        self.result['Datasets'].Total = self.object.data_name
        if classification:
            self.result['KNN'].Total = accuracy_score(self.object.T_test, classification_label)
            self.result['PRE'].Total = precision_score(self.object.T_test, classification_label, average="macro")
            self.result['REC'].Total = recall_score(self.object.T_test, classification_label, average="macro")
            self.result['F1'].Total = f1_score(self.object.T_test, classification_label, average="macro")
        self.result['time'].Total = self.object.time
        print(self.result)
        print("*" * self.xn)
        self.result.to_excel(os.path.join("Analysis", self.object.data_name, self.xlsx_path))
