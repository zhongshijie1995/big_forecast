"""
-*- coding: utf-8 -*-
@File  : metrics.py
@Author: 钟世杰
@Date  : 2023/3/6
@Desc  : 
@Contact : zhongshijie1995@outlook.com
"""
from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe
from loguru import logger
from sklearn.metrics import confusion_matrix


class Metrics:

    @staticmethod
    def trans_pred(_y_pred_prob: Any, threshold: float = 0.5) -> pd.Series:
        """
        根据阈值划分0-1

        :param _y_pred_prob: 预测概率
        :param threshold: 阈值

        :return: 预测分类
        """
        _y_pred = np.minimum(np.round(_y_pred_prob + (0.5 - threshold)), 1)
        return _y_pred

    @staticmethod
    def f1_score_self(_y_true, _y_pred) -> float:
        tn, fp, fn, tp = confusion_matrix(y_true=_y_true, y_pred=_y_pred).ravel()
        # tp_rate = tp / (tp + fn)
        # fp_rate = fp / (fp + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def f1_metrics(_y_pred_prob, _y_true, threshold: float = 0.5) -> (str, float, bool):
        """
        [0-1分类]F1分数验证

        :param _y_pred_prob: 预测概率
        :param _y_true: 真实结果
        :param threshold: 阈值

        :return: F1, F1分数, True
        """
        if type(_y_true) == lgb.Dataset:
            _y_true = _y_true.get_label()
        _y_pred_prob = Metrics.trans_pred(_y_pred_prob, threshold=threshold)
        # return 'F1', f1_score(_y_true, _y_pred_prob), True
        return 'F1', Metrics.f1_score_self(_y_true, _y_pred_prob), True

    @staticmethod
    def search_f1_best_threshold(_y_pred_prob, _y_true) -> float:
        """
        [0-1分类]F1分数搜索阈值

        :param _y_pred_prob: 预测概率
        :param _y_true: 真实结果

        :return: 最佳阈值
        """

        def convert_f1_metrics(args):
            """
            转化F1分数为最优化

            :param args: 预测概率，真实结果， 阈值

            :return: F1最大化的值
            """
            pred_prob, act_true, threshold = args
            return - Metrics.f1_metrics(pred_prob, act_true, threshold)[1]

        space = hp.choice(
            'min(-f1)',
            [
                (_y_pred_prob, _y_true, hp.uniform('threshold', 0, 1)),
            ]
        )
        best = fmin(convert_f1_metrics, space, algo=tpe.suggest, verbose=False, show_progressbar=False, max_evals=100)
        logger.info('最佳验证集为{}，最佳阈值为{}', best['min(1-f1)'], best['threshold'])
        return best['threshold']

    @staticmethod
    def guess_attr(attr: Dict[str, Any]) -> Dict:
        all_nums = attr.get('总数')
        target_one_nums = attr.get('目标数')
        pred_one_nums = attr.get('预测数')
        same_threshold = attr.get('精度')
        if same_threshold is None:
            same_threshold = 0.000001
        if attr.get('分数') is None:
            right_nums = attr.get('正确数')
            y_true = np.concatenate([
                np.ones(target_one_nums),
                np.zeros(all_nums - target_one_nums),
            ])
            y_pred = np.concatenate([
                np.ones(right_nums),
                np.zeros(all_nums - pred_one_nums),
                np.ones(pred_one_nums - right_nums)
            ])
            attr['分数'] = Metrics.f1_metrics(y_pred, y_true)[1]
        if attr.get('正确数') is None:
            score = attr.get('分数')
            maybe_list = []
            y_true = np.concatenate([
                np.ones(target_one_nums),
                np.zeros(all_nums - target_one_nums),
            ])
            for right_num in range(pred_one_nums + 1):
                if right_num > target_one_nums:
                    continue
                y_pred = np.concatenate([
                    np.ones(right_num),
                    np.zeros(all_nums - pred_one_nums),
                    np.ones(pred_one_nums - right_num),
                ])
                if abs(Metrics.f1_metrics(y_true, y_pred)[1] - score) <= same_threshold:
                    maybe_list.append(right_num)
            attr['正确数'] = maybe_list
            if len(maybe_list) > 0:
                logger.info('发现一组：{}', attr)
        return attr
