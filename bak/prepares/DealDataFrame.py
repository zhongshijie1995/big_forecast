import pandas as pd
import numpy as np
from bak.settings import Run_Val
from bak.utils import Log


def replace_val_from_df_dict(data: dict, replace_dict: dict) -> dict:
    """
    根据指定的替换字典，对整个数据集字典中的所有数据表的需替换值进行替换

    :param data: 待替换的数据集字典
    :param replace_dict: 替换字典
    :return: 替换后的数据集字典
    """
    for x in data:
        data[x] = pd.DataFrame(data[x]).replace(replace_dict)
    return data


def replace_types_in_df_dict(data: dict, replace_dict: dict) -> dict:
    """
    根据指定的替换字典，对整个数据集字典中的所有数据表的需替换数据类型进行替换

    :param data: 待替换的数据集字典
    :param replace_dict: 替换字典
    :return: 替换后的数据集字典
    """
    for index, index_type in replace_dict.items():
        for data_name in data:
            if index in list(data[data_name].columns):
                data[data_name][index] = data[data_name][index].fillna(0).astype(index_type)
    return data


def set_index_in_df_dict(data: dict, index_col_name: str) -> dict:
    """
    根据指定的列名，将整个数据集字典中的素有数据表的需指定索引列指定为索引

    :param data: 待指定的数据集字典
    :param index_col_name: 索引列名
    :return: 索引被指定后的字典
    """
    for df_name in data:
        data[df_name].set_index(index_col_name, inplace=True)
    return data


def mix_dataset(data: dict, d_name: set = None) -> dict:
    """
    指定数据集字典、字典中数据集名称，对数据集进行混合操作

    :param data: 数据集字典
    :param d_name: 数据集名称集合，存在以下两种情况：
        1. 当所有数据表在同一层目录中以X_A、X_B为数据集名读入时，传入{A, B}。
        2. 当不同数据表按照数据集文件夹夹存放被读入时，不需要传入该参数
    :return: 混合后的数据集最颠
    """
    need_mix = {}
    if d_name is None:
        d_name = Run_Val.dataset_names
    for k in data:
        for x in d_name:
            if str(k).endswith(x):
                tmp_name = str(k).replace('_' + x, '')
                if tmp_name in need_mix:
                    need_mix[tmp_name].add(k)
                else:
                    need_mix[tmp_name] = {k}
    for k, v in need_mix.items():
        if len(v) > 1:
            Log.info('合并数据表 【{}】 为 【{}】'.format(','.join(v), k))
            data[k] = pd.concat([data.get(x) for x in v], sort=True, ignore_index=True)
            for x in v:
                del data[x]
    return data


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    将数据表转换为内存占用更低的数据类型

    :param df: 转换前的数据表
    :return: 转换后的数据表
    """
    for c in df:
        # 将 objects 转换为 category
        if (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')
        # 将 booleans 转为 integers
        elif set(df[c].unique()) == {0, 1}:
            df[c] = df[c].astype(bool)
        # 将 float64 转为 float32
        elif df[c].dtype == 'float64':
            df[c] = df[c].astype(np.float32)
        # 将 int64 转为 int32
        elif df[c].dtype == 'int64':
            df[c] = df[c].astype(np.int32)
    return df


def zip_dataset(df_dict: dict) -> dict:
    """
    用普适方法（内存占用更低的类型转换）压缩数据表字典

    :param df_dict: 待压缩数据表字典
    :return: 压缩后数据表字典
    """
    Log.memory_used()
    for df_name in df_dict:
        Log.debug('压缩数据表 {}'.format(df_name))
        df_dict[df_name] = convert_types(df_dict[df_name])
    Log.memory_used()
    return df_dict
