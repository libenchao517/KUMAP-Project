################################################################################
# 本文件用于实现Kernel UMAP算法
################################################################################
# 导入模块
from time import perf_counter
import numpy as np
import torch as tc
import umap
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import Subset
from tqdm import tqdm
from .CoreFunction import core
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
import tracemalloc
################################################################################
class KUMAP:
    """
    Kernel Uniform Manifold Approximation and Projection
    """
    def __init__(
            self,
            data,
            target,
            func_name=None,
            data_name=None,
            train_func=None,
            Kernel_func=None,
            is_Supervised=None,
            label=None,
            label_num=None,
            ita=None,
            train_size=None,
            random_state=None,
            train_index = None,
            test_index = None,
            ALpha=None,
            device="cpu",
            return_time=True,
            sec_part=None,
            sec_num=None
    ):
        """
        初始化函数
        :param data:          # 数据集
        :param target:        # 标签
        :param func_name:     # 算法名称
        :param data_name:     # 数据名称
        :param train_func:    # 基本算法名称 [UMAP, TSNE, LocallyLinearEmbedding, Isomap]
        :param Kernel_func:   # 核函数
        :param is_Supervised: # 是否有监督
        :param label:         # SKUMAP中的标签向量
        :param label_num:     # 标签向量的维度
        :param ita:           # SKUMAP的超参数
        :param train_size:    # 训练比例
        :param random_state:  # 随机种子
        :param train_index:   # 训练数据的索引
        :param test_index:    # 测试数据的索引
        :param ALpha:         # 核函数的参数
        :param return_time:   # 是否返回时间
        :param sec_part:      # 项目名称
        :param sec_num:       # 实验编号
        """
        self.train_func = train_func
        self.Kernel_func = Kernel_func
        self.train_size = train_size
        self.func_name = func_name
        self.data_name = data_name
        self.random_seed = random_state
        self.train_index = train_index
        self.test_index = test_index
        self.device = device
        # self.device = "cuda" if tc.cuda.is_available() else "cpu"
        self.data = tc.tensor(data).float().to(self.device)
        self.target = tc.tensor(target).float().to(self.device)
        self.is_Supervised = is_Supervised
        if self.is_Supervised:
            self.label = tc.tensor(label).float().to(self.device)
            self.label_num = label_num
            self.ita = ita
        self.return_time = return_time
        self.time = None
        self.CPU = None
        self.GPU = None
        self.para = [sec_part, str(sec_num), func_name, data_name, 'total']
        self.train = [sec_part, str(sec_num), func_name, data_name, 'train']
        self.test = [sec_part, str(sec_num), func_name, data_name, 'test']
        self.batch_size = 100
        self.datasets = None
        self.Alpha=ALpha

    def Make_Datasets(self):
        """
        构建数据集
        :return:
        """
        if self.is_Supervised:
            self.datasets = TensorDataset(self.data, self.target, self.label)
            self.label = None
        else:
            self.datasets = TensorDataset(self.data, self.target)
        self.data = None
        self.target = None

    def Split_Data(self):
        """
        通过比例随机划分训练集和测试集
        :return:
        """
        if self.train_size < 1.0:
            train_num = int(self.train_size * len(self.datasets))
            test_num = int(len(self.datasets)) - train_num
        else:
            train_num = self.train_size
            test_num = len(self.datasets) - train_num
        self.train_data, self.test_data = random_split(self.datasets, [train_num, test_num], tc.manual_seed(self.random_seed))

    def Index_Split(self):
        """
        通过索引划分训练集和测试集
        :return:
        """
        if self.train_index is not None:
            self.train_data = Subset(self.datasets, self.train_index)
            self.test_data = Subset(self.datasets, self.test_index)

    def Calculate_distance(self, x_a, x_b, l_a=None, l_b=None):
        """
        计算距离矩阵
        :param x_a:
        :param x_b:
        :param l_a:
        :param l_b:
        :return:
        """
        dist = tc.cdist(x_a, x_b)
        return dist

    def Calculate_A(self):
        """
        计算系数矩阵
        :return:
        """
        self.A = tc.tensor([]).to(self.device)
        dataloader = DataLoader(self.train_data, batch_size=self.batch_size)
        for loader in tqdm(dataloader):
            K = self.Calculate_K_train(loader)
            self.A = tc.cat((self.A, tc.pinverse(K) @ self.Y_train), dim=0)

    def Calculate_K_train(self, loader):
        """
        计算训练数据的核矩阵
        :param loader:
        :return:
        """
        dist = self.Calculate_distance(self.datasets[self.train_data.indices][0], loader[0])
        K = eval('self.Core.' + self.Kernel_func + "(dist)")
        return K

    def Calculate_K_test(self, loader):
        """
        计算局外样本点的核矩阵
        :param loader:
        :return:
        """
        dist = self.Calculate_distance(self.datasets[self.train_data.indices][0], loader[0])
        K = eval('self.Core.' + self.Kernel_func + "(dist)")
        return K.t()

    def Calculate_Y_train(self):
        """
        计算训练数据的投影
        :return:
        """
        print("正在生成训练数据的投影......", end="")
        temp_x = self.datasets[self.train_data.indices][0]
        x = temp_x.cpu().numpy()
        if self.is_Supervised:
            temp_l = self.datasets[self.train_data.indices][2]
            l = temp_l.cpu().numpy()
            self.Y_train = eval(self.train_func + "(target_metric='l2').fit_transform(x,l)")
        else:
            if self.train_func == 'UMAP' or self.train_func == 'TSNE':
                self.Y_train = eval(self.train_func + "().fit_transform(x)")
            else:
                self.Y_train = eval(self.train_func + "().fit_transform(x).astype(np.float32)")
        print("\r训练数据投影生成完毕！" + " " * 10)
        print("*" * 80)
        self.Y_train = tc.tensor(self.Y_train).to(self.device)

    def Calculate_Y_test(self):
        """
        计算局外样本点的投影
        :return:
        """
        self.Y_test = tc.tensor([]).to(self.device)
        dataloader = DataLoader(self.test_data, batch_size=self.batch_size)
        for loader in tqdm(dataloader):
            K = self.Calculate_K_test(loader)
            self.Y_test = tc.cat((self.Y_test, K @ self.A), dim=0)

    def Closing_tasks(self):
        """
        收尾工作，数据下GPU和整合
        :return:
        """
        self.X_train = self.datasets[self.train_data.indices][0]
        self.X_train = self.X_train.cpu().numpy()
        self.X_test = self.datasets[self.test_data.indices][0]
        self.X_test = self.X_test.cpu().numpy()
        self.X = np.concatenate((self.X_train, self.X_test), axis=0)
        self.T_train = self.datasets[self.train_data.indices][1]
        self.T_train = self.T_train.cpu().numpy()
        self.T_test = self.datasets[self.test_data.indices][1]
        self.T_test = self.T_test.cpu().numpy()
        self.T = np.concatenate((self.T_train, self.T_test), axis=0)
        if self.is_Supervised:
            self.L_train = self.datasets[self.train_data.indices][2]
            self.L_train = self.L_train.cpu().numpy()
            self.L_test = self.datasets[self.test_data.indices][2]
            self.L_test = self.L_test.cpu().numpy()
            self.L = np.concatenate((self.L_train, self.L_test), axis=0)
        self.Y_train = self.Y_train.cpu().numpy()
        self.Y_test = self.Y_test.cpu().numpy()
        self.Y = np.concatenate((self.Y_train, self.Y_test), axis=0)

    def Print_time(self):
        """
        计算和格式输出算法运行时间
        :return:
        """
        print("\r", "{:8s}".format(self.para[2]),
              "{:8s}".format(self.para[3]),
              "{:8s}".format("time"),
              "{:.6F}".format(self.time_end - self.time_start) + " " * 20)
        if self.return_time:
            self.time = self.time_end - self.time_start

    def fit_transform(self):
        """
        KUMAP和SKUMAP的主函数
        :return:
        """
        if self.data is not None:
            self.Make_Datasets()
        if self.train_index is None:
            self.Split_Data()
        else:
            self.Index_Split()

        bytes_to_MB = 1024 ** 2
        tracemalloc.start()
        if tc.cuda.is_available():
            tc.cuda.reset_peak_memory_stats()
        self.time_start = perf_counter()

        self.Core = core(Alpha=self.Alpha)
        self.Calculate_Y_train()
        self.Calculate_A()
        self.Calculate_Y_test()

        self.time_end = perf_counter()
        if tc.cuda.is_available():
            gpu_peak = tc.cuda.max_memory_allocated()
            self.GPU = gpu_peak / bytes_to_MB
        _, cpu_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.CPU = cpu_peak / bytes_to_MB  # 转换为 MB

        self.Print_time()
        self.Closing_tasks()
