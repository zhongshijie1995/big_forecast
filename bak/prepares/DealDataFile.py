import os
import collections
import pandas as pd

from bak.utils import Log
from bak.settings import Run_Val


def get_data_dict_by_path(base_path: str, skips=None) -> dict:
    """
    读取指定目录下的所有数据文件

    :param base_path: 指定目录
    :param skips: 跳过文件列表，不携带目录
    :return: 数据表字典：key-文件名，value-pandas.DataFrame
    """
    base_path = os.path.abspath(base_path)
    result = {}
    for k, v in get_base_name_dict(base_path, get_all_suffix_files(base_path, '.csv', skips)).items():
        Log.debug('读取数据表 {} ，将该数据表命名为 【{}】'.format(v, k))
        result[k] = pd.read_csv(v)
    return result


def get_all_suffix_files(base_path: str, suffix: str, skips: list = None) -> list:
    """
    获取目录下全部指定后缀的文件路径名

    :param base_path: 根目录
    :param suffix: 文件后缀名（如：.csv）
    :param skips: 跳过文件列表，不携带目录
    :return: 文件路径名列表
    """
    if skips is None:
        skips = []
    result = []
    for i, j, k in os.walk(os.path.abspath(base_path)):
        for file_name in k:
            if file_name in skips:
                Log.debug('跳过文件 {}'.format(os.path.join(i, file_name)))
                continue
            if file_name.endswith(suffix):
                result.append(os.path.join(i, file_name))
    return result


def get_base_name_dict(base_path: str, path_name_list: list) -> dict:
    """
    获取路径名字典

    :param base_path: 根目录
    :param path_name_list: 文件路径名列表
    :return: 路径名字典：key-基础名，value-路径名
    """
    result = {}
    tmp_base_name_list = [os.path.basename(x) for x in path_name_list]
    repeats = [item for item, count in collections.Counter(tmp_base_name_list).items() if count > 1]
    for idx, base_name in enumerate(tmp_base_name_list):
        if base_name in repeats:
            add_name = os.path.dirname(path_name_list[idx]).replace(base_path, '').strip('\\').strip('/')
            if len(add_name) != 0:
                Run_Val.dataset_names.add(add_name)
            tmp_name = ''.join(os.path.splitext(base_name)[:-1])
            tmp_name = '_'.join([tmp_name, add_name]) if len(add_name) != 0 else tmp_name
            result[tmp_name] = path_name_list[idx]
        else:
            tmp_name = ''.join(os.path.splitext(base_name)[:-1])
            result[tmp_name] = path_name_list[idx]
    return result


def merge_p_by_path(base_path: str, p_name: str, output_path: str) -> None:
    if os.path.exists(output_path):
        Log.info('特征矩阵已经存在，跳过特征矩阵合成！')
        return
    Log.info('开始特征矩阵合并 {} ~ {} -> {}'.format(base_path, p_name, output_path))
    suffix_files = get_all_suffix_files(base_path, p_name)
    len_suffix = len(suffix_files)
    for i, suffix_file in enumerate(suffix_files):
        Log.debug('开始合并矩阵 {} / {}'.format(i + 1, len_suffix))
        if i == 0:
            tmp = pd.read_csv(suffix_file, encoding='utf-8', low_memory=True)
            tmp.to_csv(output_path, mode='a', index=False)
        else:
            tmp = pd.read_csv(suffix_file, encoding='utf-8', low_memory=True)
            tmp.to_csv(output_path, mode='a', index=False, header=None)
    Log.info('完成特征矩阵合并！')


if __name__ == '__main__':
    pass
