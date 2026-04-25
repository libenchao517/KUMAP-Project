################################################################################
# 本文件用于对KUMAP的对比算法进行封装和标准化
################################################################################
# 导入模块
import io
import sys
import warnings
warnings.filterwarnings("ignore")
import torch
from time import perf_counter
import numpy as np
import pacmap
import pymde
import trimap
import umap
import umato
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from metric_learn import NCA
from metric_learn import LMNN
from metric_learn import ITML_Supervised
from kfda import Kfda
import tracemalloc
################################################################################
class Contrast_Method_KUMAP:
    """
    对KUMAP项目中的对比方法进行统一封装
    """
    def __init__(
        self,
        data,
        target,
        func_name='UMAP',
        func_type='',
        data_name='USPS',
        train_size=2000,
        random_state=None,
        return_time=True,
        sec_part='Comparatation',
        sec_num=0
    ):
        """
        初始化函数
        :param data:         全体数据
        :param target:       全体标签
        :param func_name:    算法名称
        :param func_type:    PUMAP的嵌入类型
        :param data_name:    数据名称
        :param train_size:   训练比例
        :param random_state: 随机种子
        :param return_time:  是否返回时间
        :param sec_part:     项目名称
        :param sec_num:      实验编号
        """
        self.data = data
        self.target = target
        self.func_name = func_name
        self.func_type = func_type
        self.data_name = data_name
        self.para = [sec_part, str(sec_num), func_name, data_name, 'total']
        self.train = [sec_part, str(sec_num), func_name, data_name, 'train']
        self.test = [sec_part, str(sec_num), func_name, data_name, 'test']
        self.return_time = return_time
        self.time = None
        self.CPU = None
        self.GPU = None
        self.start_text = "当前正在" + data_name + "数据集上运行" + func_name + "算法......"
        self.embedding = None
        self.train_size = train_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.T_train = None
        self.T_test = None

    def Print_time(self):
        """
        格式输出时间
        :return:
        """
        print("\r", "{:8s}".format(self.para[2]),
              "{:8s}".format(self.para[3]),
              "{:8s}".format("time"),
              "{:.6F}".format(self.time_end - self.time_start) + " " * 20)
        if self.return_time:
            self.time = self.time_end - self.time_start

    def Embedded(self):
        """
        统一的调用函数
        :return:
        """
        print(self.start_text, end="")
        bytes_to_MB = 1024 ** 2

        tracemalloc.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.time_start = perf_counter()

        if "-" in self.func_name:
            func = self.func_name.split("-")[0]
        else:
            func = self.func_name
        eval("self." + func.upper() + self.func_type + "_embed()")
        self.time_end = perf_counter()

        if torch.cuda.is_available():
            gpu_peak = torch.cuda.max_memory_allocated()
            self.GPU = gpu_peak / bytes_to_MB

        _, cpu_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.CPU = cpu_peak / bytes_to_MB  # 转换为 MB
        self.Print_time()

    def UMATO_embed(self):
        """
        UMATO
        :return:
        """
        self.embedding = umato.UMATO().fit_transform(self.data)

    def TRIMAP_embed(self):
        """
        TRIMAP
        :return:
        """
        self.embedding = trimap.TRIMAP().fit_transform(np.array(self.data))

    def PACMAP_embed(self):
        """
        PACMAP
        :return:
        """
        self.embedding = pacmap.PaCMAP().fit_transform(self.data)

    def MDE_embed(self):
        """
        MDE
        :return:
        """
        self.embedding = pymde.preserve_neighbors(self.data, verbose=True, device="cuda").embed()
        self.embedding = self.embedding.cpu().numpy()

    def PUMAP_embed(self):
        """
        PUMAP分别对训练集和测试集进行降维
        :return:
        """
        self.X_train, self.X_test, self.T_train, self.T_test = train_test_split(self.data, self.target, train_size=self.train_size, random_state=self.random_state)
        original = sys.stdout
        output = io.StringIO()
        sys.stdout = output
        PUMAP = umap.ParametricUMAP(verbose=True).fit(self.X_train)
        self.Y_train = PUMAP.transform(self.X_train)
        self.Y_test = PUMAP.transform(self.X_test)
        self.X = np.concatenate((self.X_train, self.X_test), axis=0)
        self.Y = np.concatenate((self.Y_train, self.Y_test), axis=0)
        self.T = np.concatenate((self.T_train, self.T_test), axis=0)
        sys.stdout = original
        del output

    def UMAP_embed(self):
        """
        UMAP
        :return:
        """
        self.embedding = umap.UMAP().fit_transform(self.data)

    def TSNE_embed(self):
        """
        t-SNE
        :return:
        """
        self.embedding = TSNE().fit_transform(self.data)

    def SUMAP_embed(self):
        """
        Supervised UMAP
        :return:
        """
        self.embedding = umap.UMAP().fit_transform(self.data, self.target)

    def NCA_embed(self):
        """
        NCA
        :return:
        """
        self.X_train, self.X_test, self.T_train, self.T_test = train_test_split(self.data, self.target, train_size=self.train_size, random_state=self.random_state)
        nca = NCA(n_components=2)
        nca.fit(self.X_train, self.T_train)
        self.Y_train = nca.transform(self.X_train)
        self.Y_test = nca.transform(self.X_test)

    def LMNN_embed(self):
        """
        LMNN
        :return:
        """
        self.X_train, self.X_test, self.T_train, self.T_test = train_test_split(self.data, self.target, train_size=self.train_size, random_state=self.random_state)
        lmnn = LMNN(n_components=2, max_iter=100, learn_rate=1e-5, verbose=True)
        lmnn.fit(self.X_train, self.T_train)
        self.Y_train = lmnn.transform(self.X_train)
        self.Y_test = lmnn.transform(self.X_test)

    def LDA_embed(self):
        """
        LDA
        :return:
        """
        self.X_train, self.X_test, self.T_train, self.T_test = train_test_split(self.data, self.target, train_size=self.train_size, random_state=self.random_state)
        lda = LinearDiscriminantAnalysis(n_components=min(2, len(np.unique(self.target))-1))
        lda.fit(self.X_train, self.T_train)
        self.Y_train = lda.transform(self.X_train)
        self.Y_test = lda.transform(self.X_test)

    def KLDA_embed(self):
        """
        Kernel LDA
        :return:
        """
        self.X_train, self.X_test, self.T_train, self.T_test = train_test_split(self.data, self.target, train_size=self.train_size, random_state=self.random_state)
        klda = Kfda(n_components=2, kernel='rbf')
        klda.fit(self.X_train, self.T_train)
        self.Y_train = klda.transform(self.X_train)
        self.Y_test = klda.transform(self.X_test)

    def ITML_embed(self):
        """
        Supervised ITML
        :return:
        """
        self.X_train, self.X_test, self.T_train, self.T_test = train_test_split(self.data, self.target, train_size=self.train_size, random_state=self.random_state)
        itml = ITML_Supervised(n_constraints=200)
        itml.fit(self.X_train, self.T_train)
        self.Y_train = itml.transform(self.X_train)
        self.Y_test = itml.transform(self.X_test)
