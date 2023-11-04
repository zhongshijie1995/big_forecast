import os
import warnings
from typing import List, Dict

import featuretools as ft
import numpy as np
import pandas as pd
from dataprep.eda import plot, plot_correlation, create_report
from dataprep.eda.container import Container
from dataprep.eda.create_report.report import Report
from loguru import logger


class CsvAnalysis:
    """
    CSV预处理
    """

    @staticmethod
    def plot_sample(
            _df: pd.DataFrame,
            _cols: List[str] = None,
    ) -> Container:
        """
        绘制简单分析图

        :param _df: 数据实体
        :param _cols: 指定需分析的列

        :return: 绘制图
        """
        if _cols is None:
            _cols = []
        return plot(_df, *_cols)

    @staticmethod
    def plot_corr(
            _df: pd.DataFrame,
            _cols: List[str] = None,
    ) -> Container:
        """
        绘制相关性分析图

        :param _df: 数据实体
        :param _cols: 指定需分析的列

        :return: 绘制图
        """
        if _cols is None:
            _cols = []
        return plot_correlation(_df, *_cols)

    @staticmethod
    def report(
            _df: pd.DataFrame,
            _save_path: str = None,
            _save_name: str = 'report.html',
    ) -> Report:
        """
        生成数据报告

        :param _df: 数据实体
        :param _save_path: 保存报告路径
        :param _save_name: 保存的文件名

        :return: 数据报告
        """
        with warnings.catch_warnings():
            # 过滤警告
            warnings.simplefilter('ignore', category=RuntimeWarning)
            warnings.simplefilter('ignore', category=FutureWarning)
            # 生成报告
            report = create_report(_df)
            # 若指定了保存路径，则保存数据报告
            if _save_path is not None:
                os.makedirs(_save_path, exist_ok=True)
                report.save(os.path.join(_save_path, _save_name))
            # 返回数据报告
            return report

    @staticmethod
    def multi_report(
            _data: Dict[str, pd.DataFrame],
            _save_path: str,
    ) -> None:
        """
        为实体集生成数据报告

        :param _data: 实体集
        :param _save_path: 保存报告路径
        :return:
        """
        for k, v in _data.items():
            CsvAnalysis.report(v, _save_path, k)
        return None


class DfProcessing:

    @staticmethod
    def category_replace_with_value_count(_df_all: pd.DataFrame, _col_name_all: pd.DataFrame, _df_target: pd.DataFrame,
                                          _col_name_target: pd.DataFrame):
        replace_dict = _df_all[_col_name_all].value_counts().to_dict()
        replace_dict['nan'] = '0'
        _df_target[_col_name_target] = _df_target[_col_name_target].astype('str')
        _df_target[_col_name_target] = _df_target[_col_name_target].replace(replace_dict)
        _df_target[_col_name_target] = _df_target[_col_name_target].astype('int64')
        return _df_target

    @staticmethod
    def change_int64_to_float64(_df: pd.DataFrame) -> pd.DataFrame:
        """
        将给定的Pandas中的int64、Int64列转为float64类型

        :param _df: 给定的Pandas

        :return: 处理后的pandas
        """
        feature_list = _df.select_dtypes(include=['Int64', 'int64']).columns.to_list()
        for i in feature_list:
            _df[i] = _df[i].astype('float64')
        return _df

    @staticmethod
    def change_object_to_category(_df: pd.DataFrame) -> pd.DataFrame:
        """
        将给定的Pandas中的object类型列转为category类型

        :param _df: 给定的Pandas
        :return: 处理后的pandas
        """
        feature_list = _df.select_dtypes(include=['object']).columns.to_list()
        for i in feature_list:
            _df[i] = _df[i].astype('category')
        return _df

    @staticmethod
    def change_object_to_int(_df: pd.DataFrame, skips=None) -> (pd.DataFrame, List[str]):
        if skips is None:
            skips = []
        feature_list = _df.select_dtypes(include=['object', 'category']).columns.to_list()
        feature_list = [i for i in feature_list if i not in skips]
        for i in feature_list:
            _df[i] = pd.factorize(_df[i])[0]
        return _df, feature_list

    @staticmethod
    def change_category_to_object(_df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
        """
        将给定的Pandas中的object类型列转为category类型

        :param _df: 给定的Pandas
        :param feature_list: 待处理的列名列表

        :return: 处理后的pandas
        """
        if feature_list is None:
            feature_list = _df.select_dtypes(include=['category']).columns.to_list()
        for i in feature_list:
            _df[i] = _df[i].astype('str')
            _df[i] = _df[i].fillna('Nan')
        for i in feature_list:
            _df[i] = _df[i].astype('category')
        return _df

    @staticmethod
    def convert_date_to_int(_df: pd.DataFrame) -> pd.DataFrame:
        """
        将给定的Pandas中的日期列转为距离1900-01-01的天数

        :param _df: 给定的Pandas

        :return: 处理后的pandas
        """
        feature_list = _df.select_dtypes(include=['datetime64[ns]']).columns.to_list()
        for i in feature_list:
            _df[i] = _df[i].astype(np.int64) / (1000000000 * 3600 * 24) + 2556
        return _df

    @staticmethod
    def convert_cols_to_int(_df: pd.DataFrame, _cols: List[str]) -> pd.DataFrame:
        for i in _cols:
            _df[i] = _df[i].astype(int)
        return _df

    @staticmethod
    def bad_datetime_repair(_df: pd.DataFrame, _col: str, errors: str = 'ignore') -> pd.DataFrame:
        """
        对日期列中存在无法正常转换为日期的数据进行修复

        :param _df: 给定的Pandas
        :param _col: 待修复列
        :param errors: 错误时处理 【ignore、coerce、raise】

        :return: 处理后的pandas
        """
        _df[_col] = pd.to_datetime(_df[_col], errors=errors)
        return _df

    @staticmethod
    def drop_low_info_cols(_df: pd.DataFrame, na_rate: float, _target: str = None) -> pd.DataFrame:
        """
        删除低信息的列

        :param _df: 给定的Pandas
        :param na_rate: Nan过高判定的阈值
        :param _target: [可选]标签列，若存在则按照此列划分训练集和测试集，且仅删除训练集和测试集同时Nan过高

        :return: 处理后的pandas
        """
        logger.info('处置前形状[{}]', _df.shape)
        parts = []
        low_info_cols = []
        if _target is not None:
            parts.append(_df[_df[_target].notnull()])
            parts.append(_df[_df[_target].isnull()])
        else:
            parts.append(_df)
        for col in _df.columns:
            if col is _target:
                continue
            is_low_info = True
            for part in parts:
                if (part[col].isna().sum() / part.shape[0]) <= na_rate:
                    is_low_info = False
            if is_low_info:
                low_info_cols.append(col)
        result = _df.drop(low_info_cols, axis=1, inplace=False)
        logger.info('处置后形状[{}]', result.shape)
        return result

    @staticmethod
    def drop_high_corr_cols(_df: pd.DataFrame, high_rate: float = 0.95, _target: str = None) -> pd.DataFrame:
        """
        删除高相关性的列

        :param _df: 给定的Pandas
        :param high_rate: 高相关性判定的阈值
        :param _target: [可选]标签列，若存在则按照此列划分训练集和测试集，且仅删除训练集和测试集同时Nan过高

        :return: 处理后的pandas
        """
        logger.info('处置前形状[{}]', _df.shape)
        features_to_keep = [_target, ] if _target is not None else None
        result = ft.selection.remove_highly_correlated_features(
            _df, pct_corr_threshold=high_rate, features_to_keep=features_to_keep
        )
        logger.info('处置后形状[{}]', result.shape)
        return result

    @staticmethod
    def drop_cols_by_name(_df: pd.DataFrame, drop_list: List[str]) -> pd.DataFrame:
        """
        根据列名删除

        :param _df: 给定的Pandas
        :param drop_list: 待删除的列名列表

        :return: 处理后的pandas
        """
        logger.info('处置前形状[{}]', _df.shape)
        result = _df.drop(drop_list, axis=1, inplace=False)
        logger.info('处置后形状[{}]', result.shape)
        return result

    @staticmethod
    def update_vals_by_vals(_df: pd.DataFrame, _id: str, id_vals: List, target: str, target_vals: List) -> pd.DataFrame:
        """
        根据id更新指定列的值

        :param _df:
        :param _id:
        :param id_vals:
        :param target:
        :param target_vals:

        :return: 处理后的pandas
        """
        if len(id_vals) != len(target_vals):
            return _df
        for i in range(len(id_vals)):
            _df.loc[_df[_id] == id_vals[i], target] = target_vals[i]
        return _df

    @staticmethod
    def alignment_data_dict():
        # for k, v in dataset.mix_data.items():
        #     num_features = v.select_dtypes(include=['float64']).columns.to_list()
        #     print(k, num_features)
        #     tmp = v
        #     found_id = target_id
        #     if k == 'DZ_TARGET':
        #         continue
        #     if k == 'DZ_TR_APS':
        #         found_id = 'APSDCUSNO'
        #     if k == 'DZ_MBNK_BEHAVIOR':
        #         found_id = 'PID'
        #     print('train', tmp.shape, 'test', tmp[tmp[found_id].str.endswith('.B')].shape)
        #     for i in num_features:
        #         ft_max = np.ceil(v[v[found_id].str.endswith('.B')][i].max())
        #         ft_min = np.floor(v[v[found_id].str.endswith('.B')][i].min())
        #         dataset.mix_data[k] = tmp[((tmp[i]>=ft_min)&(tmp[i]<=ft_max))|(tmp[i].isnull())]
        #     print('train', dataset.mix_data[k].shape, 'test', dataset.mix_data[k][dataset.mix_data[k][found_id].str.endswith('.B')].shape)
        pass
