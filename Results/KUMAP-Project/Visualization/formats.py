################################################################################
# 本代码用于将实验结果转换为 mean ± std 的形式
################################################################################
# 导入必要的库
import pandas as pd
################################################################################
def generate_latex_table_(mean_df, std_df, precision=2):
    """
    将均值和标准差合并，并生成 LaTeX 表格代码
    :param mean_df: 均值的表格
    :param std_df:  标准差的表格
    :param precision: 精度
    :return:
    """
    # 合并为 mean ± std 字符串
    formatted_df = mean_df.copy()
    for col in mean_df.columns:
        formatted_df[col] = [
            f"{m:.{precision}f} $\\pm$ {s:.{precision}f}"
            for m, s in zip(mean_df[col], std_df[col])
        ]
    # 转换为 LaTeX 代码
    latex_code = formatted_df.to_latex(
        index=True,
        escape=False,  # 允许渲染 \pm 符号
        column_format='l' + 'c' * len(mean_df.columns),
        caption="Classification Accuracy (Mean $\\pm$ Std)",
        label="tab:results",
        header=True
    )
    return latex_code

def reshape_table(df, split_index=4):
    """
    将表格按列切分并上下拼接（论文中的形式）
    split_index: 从第几列开始切分（这里是第4列）
    """
    part1 = df.iloc[:, :split_index]
    part2 = df.iloc[:, split_index:]
    return part1, part2

if __name__ == "__main__":
    path = "../KUMAP-Results/KUMAP-time.xlsx"

    # 加载数据
    df_mean = pd.read_excel(path, sheet_name="Mean", index_col=0, header=0)
    df_std = pd.read_excel(path, sheet_name="Std", index_col=0, header=0)

    # 执行转换
    m1, m2 = reshape_table(df_mean, split_index=4)
    s1, s2 = reshape_table(df_std, split_index=4)

    # 生成并打印 LaTeX
    latex_output = generate_latex_table_(m1, s1, precision=4)
    print(latex_output)
    latex_output = generate_latex_table_(m2, s2, precision=4)
    print(latex_output)