"""
-*- coding: utf-8 -*-
@File  : features.py
@Author: 钟世杰
@Date  : 2023/1/31
@Desc  : 
@Contact : zhongshijie1995@outlook.com
"""
import os
from typing import List, Dict, Any

import featuretools as ft
import pandas as pd
from loguru import logger

from _ML.models import TabBinary
from _Tool.io import FileIO


class TabFeatures:
    """
    表格类特征工程
    """
    class Define:
        @staticmethod
        def gen_entity_set(_name: str) -> ft.EntitySet:
            """
            生成实体集

            :param _name: 实体集名称

            :return: 实体集
            """
            logger.info('开始创建实体集[{}]', _name)
            return ft.EntitySet(id=_name)

        @staticmethod
        def add_entity(_es: ft.EntitySet, _data: dict, _name: str, _index: str = None, _time_index: str = None,
                       _logical_types: dict = None) -> ft.EntitySet:
            """
            从数据表字典中为实体集添加一个实体

            :param _es: 实体集
            :param _data: 数据表字典
            :param _name: 数据表名
            :param _index: 索引名
            :param _time_index: 时间索引名
            :param _logical_types: 逻辑类型字典

            :return: 实体集
            """
            _df_cp = _data[_name]
            _make_index = True if _index is None else False
            _index = _index if _index is not None else _name + '.idx'
            logger.info(
                '为实体集[{}]添加实体[{}]，索引名[{}]，时间索引[{}]，逻辑类型[{}]',
                _es.id, _name, _index, _time_index, _logical_types
            )
            _es.add_dataframe(
                dataframe_name=_name,
                dataframe=_df_cp,
                make_index=_make_index,
                index=_index,
                time_index=_time_index,
            )
            return _es

        @staticmethod
        def add_relationship(_es: ft.EntitySet, a: str, a_idx: str, b: str, b_idx: str) -> ft.EntitySet:
            """
            对实体集中的表增加一个表间关系

            :param _es: 实体集
            :param a: 表A
            :param a_idx: 表A的ID
            :param b: 表B
            :param b_idx: 表B的ID

            :return: 实体集
            """
            logger.info('为实体集[{}]添加关系[{}.{} -> {}.{}]', _es.id, a, a_idx, b, b_idx)
            _es.add_relationship(a, a_idx, b, b_idx)
            return _es

    class Gen:
        @staticmethod
        def gen_feature_defs(
                _es: ft.EntitySet,
                _target_entity: str,
                _agg_primitives: list = None,
                _trans_primitives: list = None,
                _groupby_trans_primitives: list = None,
                _ignore_variables: dict = None,
                _primitive_options: dict = None,
                _interesting_values_agg_list: list = None,
                _max_depth: int = 2
        ) -> List[ft.FeatureBase]:
            """
            生成衍生特征定义

            :param _es: 实体集
            :param _target_entity: 目标实体名
            :param _agg_primitives: 聚合基元
            :param _trans_primitives: 转换基元
            :param _groupby_trans_primitives: 分组的转换基元
            :param _ignore_variables: 无视的列
            :param _primitive_options: 基元选项
            :param _interesting_values_agg_list: 关注的值
            :param _max_depth: 最大深度

            :return:
            """
            feature_defs = ft.dfs(
                entityset=_es,
                target_dataframe_name=_target_entity,
                agg_primitives=_agg_primitives,
                trans_primitives=_trans_primitives,
                groupby_trans_primitives=_groupby_trans_primitives,
                ignore_columns=_ignore_variables,
                max_depth=_max_depth,
                primitive_options=_primitive_options,
                where_primitives=_interesting_values_agg_list,
                features_only=True
            )
            logger.info('实体集[{}]可衍生特征共[{}]', _es.id, len(feature_defs))
            result = []
            exs = ['.idx', '.time_idx']
            for i in feature_defs:
                if '.idx' in i.get_name() or '.time_idx' in i.get_name():
                    continue
                else:
                    result.append(i)
            logger.info('实体集[{}]可衍生特征排除{}类型的特征后共[{}]', _es.id, exs, len(result))
            return result

        @staticmethod
        def gen_feature_matrix(_feature_defs: List, _es: ft.EntitySet, _n_jobs: int) -> pd.DataFrame:
            """
            按照定义生成特征

            :param _feature_defs: 特征定义
            :param _es: 实体集
            :param _n_jobs: 线程数

            :return:
            """
            logger.info('以[{}]线程为实体集[{}]计算特征', _n_jobs, _es.id)
            return ft.calculate_feature_matrix(_feature_defs, entityset=_es, verbose=0, n_jobs=_n_jobs)

        @staticmethod
        def create_features_dict(_fdp: str, _feature_list: List[ft.FeatureBase], _unknown_flag: int = -9999) -> None:
            """
            保存特征字典

            :param _fdp: 特征字典路径
            :param _feature_list: 特征列表
            :param _unknown_flag: 未知标记

            :return: 无
            """
            if not os.path.exists(_fdp):
                feature_dict = {}
                for i in _feature_list:
                    feature_dict[i.get_name()] = _unknown_flag
                FileIO.save_pickle(_fdp, feature_dict)
            return None

        @staticmethod
        def pick_features_dict(_fdp: str, _max_col_nums) -> list:
            """
            收集特征字典

            :param _fdp: 特征字典路径
            :param _max_col_nums: 保留的特征个数

            :return: 好特征名[列表]
            """
            features_dict = FileIO.get_pickle(_fdp)
            order_features_dict = sorted(features_dict.items(), key=lambda x: x[1], reverse=True)
            result = [x for x in order_features_dict[: _max_col_nums]]
            logger.info('提取选择后特征[{}]个', len(result))
            return result

    @staticmethod
    def select_features(
            _es: ft.EntitySet,
            _fdp: str,
            _feature_list: list,
            _target_tab: str,
            _target_id: str,
            _target: str,
            _select_params: Dict[str, Any] = None,
            _select_params_metric: List[float] = None,
            _keep_features_num: int = 800,
            _unknown_flag: int = -9999,
            _try_step: int = 400,
            _n_jobs: int = 10,
    ) -> None:
        try_time = 0
        while True:
            # 计数
            try_time += 1
            logger.info('第[{}]轮探索特征', try_time)
            # 读出待探索特征
            unknown_list = []
            _feature_dict = FileIO.get_pickle(_fdp)
            for k, v in _feature_dict.items():
                if v == _unknown_flag:
                    unknown_list.append(k)
            logger.info('读入总特征[{}]，发现待探索[{}]特征', len(_feature_dict), len(unknown_list))
            if len(unknown_list) == 0:
                logger.info('特征探索结束')
                break
            # 开始探索特征
            try_feature_names = unknown_list[: _try_step]
            try_features = []
            for i in _feature_list:
                if i.get_name() in try_feature_names:
                    try_features.append(i)
            logger.info('开始探索特征共[{}]个', len(try_feature_names))
            tmp_feature_matrix = TabFeatures.Gen.gen_feature_matrix(try_features, _es, _n_jobs)
            tmp_feature_matrix = pd.merge(tmp_feature_matrix, _es[_target_tab][[_target_id, _target]], on=_target_id)
            feature_df = TabBinary.get_importance(
                tmp_feature_matrix,
                _target,
                _target_id,
                _select_params,
                _select_params_metric,
            )
            for idx, row in feature_df.iterrows():
                now_update_key = row['feature_name']
                if now_update_key not in _feature_dict:
                    raise
                else:
                    _feature_dict[now_update_key] = row['importance']
            # 保存特征
            FileIO.save_pickle(_fdp, _feature_dict)

    @staticmethod
    def auto(
            _name: str,
            _data: Dict[str, pd.DataFrame],
            _target_tab: str,
            _target_id: str,
            _target_col: str,
            _entities: List[Dict[str, str]],
            _relationships: List[List[str]],
            _trans_list: List[str],
            _agg_list: List[str],
            _groupby_trans_primitives: List[str],
            _max_depth: int,
            _max_col_nums: int,
            _primitive_options: Dict[str, Any],
            _n_jobs: int,
            _try_step: int,
            _select_params: Dict[str, Any],
            _select_params_metric: List[float],
            _interesting_values: Dict[str, Dict[str, List[str]]] = None,
            _interesting_values_agg_list: List[str] = None,
            _reappear_col_list: List[str] = None,
    ) -> pd.DataFrame:
        # 生成一个实体集
        _es = TabFeatures.Define.gen_entity_set(_name)
        # 添加实体
        for entity in _entities:
            _es = TabFeatures.Define.add_entity(_es, _data, **entity)
        # 添加关系
        for a, a_idx, b, b_idx in _relationships:
            _es = TabFeatures.Define.add_relationship(_es, a, a_idx, b, b_idx)
        # 添加关注变量值
        if _interesting_values is not None:
            for k, v in _interesting_values.items():
                _es.add_interesting_values(dataframe_name=k, values=v)
        # 预览实体集
        logger.info('实体集如下\n{}', _es)
        # 生成特征定义
        feature_list = TabFeatures.Gen.gen_feature_defs(
            _es=_es,
            _target_entity=_target_tab,
            _trans_primitives=_trans_list,
            _agg_primitives=_agg_list,
            _groupby_trans_primitives=_groupby_trans_primitives,
            _ignore_variables={_target_tab: [_target_col]},
            _interesting_values_agg_list=_interesting_values_agg_list,
            _max_depth=_max_depth,
            _primitive_options=_primitive_options,
        )
        result_feature_list = []
        if _reappear_col_list is not None:
            logger.info('重现特征工程，直接提取指定特征{}个', len(feature_list))
            for i in feature_list:
                if i.get_name() in [x for x in _reappear_col_list]:
                    result_feature_list.append(i)
        elif _max_col_nums >= len(feature_list):
            logger.info('自动特征工程，直接提取指定特征{}个', len(feature_list))
            for i in feature_list:
                result_feature_list.append(i)
        else:
            logger.info('自动特征工程，进行特征探索')
            # 保存特征定义
            os.makedirs('ft', exist_ok=True)
            _fdp = 'ft/feature_dict-%s.pickle' % _name
            TabFeatures.Gen.create_features_dict(_fdp, feature_list)
            # 进行特征选择
            TabFeatures.select_features(
                _es=_es,
                _fdp=_fdp,
                _feature_list=feature_list,
                _target_tab=_target_tab,
                _target_id=_target_id,
                _target=_target_col,
                _try_step=_try_step,
                _select_params=_select_params,
                _select_params_metric=_select_params_metric,
                _n_jobs=_n_jobs,
            )
            # 提取选择后的特征
            picked_features = TabFeatures.Gen.pick_features_dict(_fdp, _max_col_nums)
            for i in feature_list:
                if i.get_name() in [x[0] for x in picked_features]:
                    result_feature_list.append(i)
        # 计算特征矩阵
        main = TabFeatures.Gen.gen_feature_matrix(result_feature_list, _es, _n_jobs)
        # 为特征矩阵拼接目标列
        main = pd.merge(
            left=_es[_target_tab][[_target_id, _target_col]],
            right=main,
            left_on=_target_id,
            right_on=_target_id,
            how='left'
        )
        return main

    @staticmethod
    def join(
            _data_dict: Dict[str, pd.DataFrame],
            _target_id: str,
            _target: str,
    ):
        result = None
        for k, v in _data_dict.items():
            if result is None:
                result = v
            else:
                tmp = v.loc[:, ~v.columns.isin([_target])]
                result = pd.merge(result, tmp, on=_target_id, how='left')
        return result


class DatetimeEngineer:
    @staticmethod
    def calc_date_away(
            _df: pd.DataFrame,
            _cols: List[str],
            _date: str = '1900-01-01',
            _p: List[int] = None,
            _abs: bool = False,
            _round: int = None,
    ) -> pd.DataFrame:
        if _p is None:
            _p = [1, ]
        result = _df.copy()
        for _col in _cols:
            tmp = result[_col].apply(lambda x: int((x - pd.to_datetime(_date)).days))
            for i in _p:
                tmp_col = '{}_{}_days_away'.format(_col, i)
                result[tmp_col] = tmp / i
                if _abs:
                    result[tmp_col] = result[tmp_col].abs()
                if _round is not None:
                    result[tmp_col] = result[tmp_col].round(_round)
        return result
