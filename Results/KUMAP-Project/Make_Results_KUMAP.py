################################################################################
# 本代码用于整理实验结果
################################################################################
# 添加路径和消除警告
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.append("/".join(Path(__file__).parts[0:Path(__file__).parts.index('REUMAP') + 1]))
################################################################################
# 导入必要模块
import os
import shutil
import datetime as dt
import numpy as np
import pandas as pd
from Utils import Make_Table
from KUMAP import KUMAP_config
config = KUMAP_config()
################################################################################
# 定义必要变量
Root = os.getcwd()
Analysis = 'Analysis'
Figure = "Figure"
Temp_File= 'Temp-Files'
Result_File= 'Result-Files'
os.makedirs(Temp_File, exist_ok=True)
os.makedirs(Result_File, exist_ok=True)
methods = ['UMAP', 'UMATO','PACMAP', 'TRIMAP', 'MDE', 'LDA','KLDA', 'NCA', 'ITML', 'PUMAP', 'KUMAP','SKUMAP']
MT = Make_Table()
################################################################################
# 整理到临时汇总文件
for d in config.KUMAP_data:
    temp = pd.DataFrame()
    path = os.path.join(Root, Analysis, d)
    xlsx_list = list(map(str, list(Path(path).rglob("*.xlsx"))))
    for xlsx in xlsx_list:
        df = pd.read_excel(xlsx, header=0, index_col=0)
        temp = pd.concat([temp, df])
    temp.to_excel(os.path.join(Root, Temp_File, d + '.xlsx'))
################################################################################
# 汇总相同维度分析的所有结果
Total_Results = pd.DataFrame()
path = os.path.join(Root, Temp_File)
xlsx_list = list(map(str, list(Path(path).rglob("*.xlsx"))))
for xlsx in xlsx_list:
    df = pd.read_excel(xlsx, header=0, index_col=0)
    Total_Results = pd.concat([Total_Results, df])
KUMAP_index = ['KNN', 'PRE', 'REC', 'F1', "NMI", 'time', 'CPU', 'GPU']
################################################################################
# 整理相同维度分析的各个指标
for idx in KUMAP_index:
    Results = Total_Results[['Method', 'Datasets', idx]].copy()
    Result_total = Results[Total_Results.index == 'Total']
    Result_total.set_index(['Method', 'Datasets'], inplace=True)
    Result_3 = pd.DataFrame(index=methods, columns=config.KUMAP_data)
    for m in methods:
        for d in config.KUMAP_data:
            try:
                Result_3.loc[m, d] = Result_total.loc[m, d][0]
            except:
                Result_3.loc[m, d] = None
    with pd.ExcelWriter(os.path.join(Root, Result_File, idx + '.xlsx')) as writer:
        Result_3.to_excel(writer, sheet_name='total')
################################################################################
# 整理文件和文件夹
todir = "KUMAP-"+str(dt.date.today()) + "-" + dt.datetime.now().time().strftime("%H-%M-%S")
os.makedirs(todir, exist_ok=True)
shutil.move(os.path.join(Root, Analysis), todir)
shutil.move(os.path.join(Root, Result_File), todir)
shutil.move(os.path.join(Root, Temp_File), todir)
if os.path.exists(os.path.join(Root, Figure)):
    shutil.move(os.path.join(Root, Figure), todir)
################################################################################
# 整理结果文件格式
xlsx_list = list(map(str, list(Path(todir).rglob("*.xlsx"))))
for xlsx in xlsx_list:
    MT.Make(xlsx)
