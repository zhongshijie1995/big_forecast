"""
-*- coding: utf-8 -*-
@File  : datasets.py
@Author: 钟世杰
@Date  : 2023/1/28
@Desc  : 数据集的定义、读取等操作
@Contact : zhongshijie1995@outlook.com
"""

from typing import Dict, List, Any

import pandas as pd
from loguru import logger

from _Tool.io import FileIO
from _Tool.str import StrList


class CsvDataset:
    """
    CSV数据集
    """

    @staticmethod
    def __get_df(
            _file: str,
            _dtype: Dict[str, Dict[str, Any]] = None,
            _parse_dates: Dict[str, List[str]] = None,
    ) -> pd.DataFrame:
        """
        从文件获取DataFrame

        :param _file: 文件路径
        :param _dtype: 指定数据类型列的字典
        :param _parse_dates: 指定日期解析列的字典

        :return: 读入的Pandas
        """
        # 读入数据
        result = pd.read_csv(
            _file,  # 文件路径
            dtype=_dtype,  # 指定数据类型列
            parse_dates=_parse_dates, infer_datetime_format=_parse_dates is None  # 指定日期解析列
        )
        # 返回数据
        return result

    @staticmethod
    def __merge_dataset(
            _dataset: Dict[str, Dict[str, pd.DataFrame]],
            _keys: List[str],
            _key: str,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        合并数据集字典中的多个DataFrame实体到

        :param _dataset: 数据集字典
        :param _keys: 待合并的数据集名
        :param _key: 合并后数据集名

        :return: 数据集字典
        """
        # 参数检查
        if len(_keys) < 2:
            logger.error('无需合并')
            return _dataset
        # 合并前检查
        if _key in _dataset:
            logger.error('合并后的字典键已存在于合并前字典中')
            return _dataset
        # 开始合并数据集
        _dataset[_key] = {}
        for _tmp_key in _keys:  # 遍历数据集
            for k, v in _dataset[_tmp_key].items():  # 遍历实体
                if k not in _dataset[_key]:
                    _dataset[_key][k] = _dataset[_tmp_key][k]  # 如果该实体还未有记录，则纳入合并
                    continue
                _dataset[_key][k] = pd.concat([_dataset[_key][k], _dataset[_tmp_key][k]])  # 如果实体已经有记录，则合并
        return _dataset

    @staticmethod
    def __set_col_upper(
            _path: str
    ) -> None:
        """
        给定csv文件，将其表头统一为大写

        :param _path: 文件路径

        :return: 无
        """
        tmp_content = ''
        with open(_path, 'r', encoding='UTF-8') as f:
            i = 0
            for line in f:
                if i == 0:
                    if line == line.upper():
                        return None
                    tmp_content += line.upper()
                else:
                    tmp_content += line
                i += 1
        with open(_path, 'w', encoding='UTF-8') as f:
            f.write(tmp_content)

    @staticmethod
    def get_dataset(
            _paths: Dict[str, str],
            _types: Dict[str, Dict[str, Any]],
            _dates: Dict[str, List[str]],
            _skips: List[str],
            _bad_cols: Dict[str, List[str]],
            _target_id_suffix: bool = False,
            _target_id: str = None,
            _shadow_ids: Dict[str, List[str]] = None,
            _set_col_upper: bool = False,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        给定路径字典，获取数据集字典

        :param _paths: 数据集
        :param _types: 指定类型的列及类型
        :param _dates: 指定为日期类型的列
        :param _skips: 跳过的文件关键字
        :param _bad_cols: 无用列
        :param _target_id_suffix: 添加后缀ID
        :param _target_id: 目标列
        :param _shadow_ids: 同名ID列
        :param _set_col_upper: 是否更新文件列名为大写

        :return: 数据集字典
        """
        # 定义数据集字典
        result = {}
        # 开始读取数据集字典
        for dataset_nam, dataset_path in _paths.items():
            # 定义数据集
            result[dataset_nam] = {}
            # 获取数据集的路径
            csv_paths = FileIO.get_file_list(_path=dataset_path)
            # 过滤跳过的路径
            csv_paths = StrList.filter_keys(_strs=csv_paths, _keys=_skips, _contain=False)
            # 遍历路径
            for csv_path in csv_paths:
                # 提取实体名
                csv_nam = FileIO.get_base_name(csv_path)
                # 开始获取
                logger.info('{}-{}', dataset_nam, csv_nam)
                # 若要求更新文件列名为大写则更新文件
                if _set_col_upper:
                    CsvDataset.__set_col_upper(csv_path)
                # 获取实体
                result[dataset_nam][csv_nam] = CsvDataset.__get_df(csv_path, _types.get(csv_nam), _dates.get(csv_nam))
                # 若存在需要删除的列，则删除上述列
                if csv_nam in _bad_cols:
                    for j in _bad_cols.get(csv_nam):
                        del result[dataset_nam][csv_nam][j]
                # 若存在影子ID则将其替换为统一ID
                if _shadow_ids is not None and csv_nam in _shadow_ids:
                    for shadow_target_id in _shadow_ids[csv_nam]:
                        result[dataset_nam][csv_nam].rename(columns={shadow_target_id: _target_id}, inplace=True)
                # 若要求添加ID后缀中则将其更新为[ID.数据集]
                if _target_id_suffix:
                    if _target_id is not None:
                        suffix_nam = '.%s' % dataset_nam
                        result[dataset_nam][csv_nam][_target_id] = result[dataset_nam][csv_nam][_target_id] + suffix_nam
        # 返回数据集字典
        return result

    @staticmethod
    def get_merge_data(
            _dataset: Dict[str, Dict[str, pd.DataFrame]],
            _keys: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """
        给定数据集字典、数据集名，获取合并后数据集

        :param _dataset: 数据集字典
        :param _keys: 数据集名

        :return: 合并后数据集
        """
        merge_name = '_'.join(_keys)
        return CsvDataset.__merge_dataset(_dataset, _keys, merge_name)[merge_name]

    @staticmethod
    def set_remove_ids_suffix(
            _df: pd.DataFrame,
            _target_id: str,
            _keys: List[str],
    ) -> pd.DataFrame:
        """
        移除ID后缀

        :param _df: 实体
        :param _target_id: ID列名
        :param _keys: 数据集名
        :return: 
        """

        def suffix_remove(x: str):
            if x.endswith(suffix_name):
                x = x[: -len(suffix_name)]
            return x

        for _key in _keys:
            suffix_name = '.%s' % _key
            _df[_target_id] = _df[_target_id].apply(suffix_remove)
        return _df
