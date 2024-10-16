import os
import re
import pandas as pd
import numpy as np


def read_csv(path):
    """
    对csv格式的近红外光谱进行读取,输出dataframe

    Parameters
    ----------
    path : str
        光谱数据文件路径

    Returns
    -------
    dataframe
        光谱数据

    """

    df = pd.read_csv(path)
    # 判断csv格式
    df_return = pd.DataFrame()
    if isinstance(df.iloc[0, 0], str):
        # df.iloc[0, 0]为str，代表未将名称读入index
        index = df.iloc[:, 0].values[:]
        df_return = df.iloc[:, 1:]
        coloumns = df_return.columns.values
        df_return.columns = [int(x) for x in coloumns]
        df_return.index = index
    elif isinstance(df.iloc[0, 0], float):
        # df.iloc[0, 0]为float，代表df.values为数据
        df_return = df
        coloumns = df_return.columns.values[1:]
        df_return.columns = [int(x) for x in coloumns]

    return df_return


def read_xlsx(path):
    """
    对xlsx格式的近红外光谱进行读取,输出dataframe

    Parameters
    ----------
    path : str
        光谱数据文件路径

    Returns
    -------
    dataframe
        光谱数据

    """

    df = pd.read_excel(path)
    # 判断excel格式
    df_return = pd.DataFrame()
    if isinstance(df.iloc[0, 0], str):
        # df.iloc[0, 0]为str，代表未将名称读入index
        index = df.iloc[:, 0].values[:]
        df_return = df.iloc[:, 1:]
        coloumns = df_return.columns.values
        df_return.columns = [float(x) for x in coloumns]
        df_return.index = index
    elif isinstance(df.iloc[0, 0], float):
        # df.iloc[0, 0]为float，代表df.values为数据
        df_return = df
        coloumns = df_return.columns.values[1:]
        df_return.columns = [float(x) for x in coloumns]

    return df_return


def read_txt(path):
    """
    对ViewSpecPro导出的txt格式的近红外光谱进行读取,输出dataframe

    Parameters
    ----------
    path : str, list
        光谱数据文件夹路径, 或者光谱数据文件路径

    Returns
    -------
    dataframe
        光谱数据

    """

    def my_sort(x):
        array = re.findall(r'\d+', x)
        if array:
            return int(array[1])
        else:
            return 999

    if isinstance(path, str):
        items = []
        for i in os.listdir(path):
            if i[-3:] == "txt":
                items.append(i)
    elif isinstance(path, list):
        items = [i.split('\\')[-1] for i in path]
        path = '\\'.join(path[0].split('\\')[:-1])

    items.sort(key=my_sort)
    raw_data = []
    raw_name = [i[:-4] for i in items]

    for i in items:
        data = []
        with open(os.path.join(path, i), 'r', encoding='gbk') as f:
            for line in f.readlines():
                line = line.strip('\n')
                data.append(line.split()[-1])
        data = np.array(data)
        raw_data.append(data[1:].astype(np.float64))

    raw_data = np.array(raw_data)

    df_return = pd.DataFrame(index=raw_name, data=raw_data, columns=range(350, 2501))
    if '/' in df_return.index[0]:
        df_return.index = [i.split('/')[-1] for i in df_return.index]

    return df_return

