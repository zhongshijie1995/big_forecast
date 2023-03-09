"""
-*- coding: utf-8 -*-
@File  : model.py
@Author: 钟世杰
@Date  : 2023/1/31
@Desc  : 
@Contact : zhongshijie1995@outlook.com
"""
import os
import warnings
from typing import List, Callable, Dict

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold

from _ML.metrics import Metrics
from _ML.preprocessing import DfProcessing

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class TabBinary:
    @staticmethod
    def get_importance(
            _main: pd.DataFrame,
            _target: str,
            _id: str,
            _params: dict,
            _metric_directions: List[float],
    ) -> pd.DataFrame:
        """
        获取特征重要值

        :param _main: 主矩阵
        :param _target: 目标列名
        :param _id: 目标ID
        :param _params: 训练参数
        :param _metric_directions: 训练参数

        :return: 重要性Pandas
        """

        def metric_avg(s: pd.Series, best_dict: dict, ms: List[str], mds: List[float]):
            """
            :param s:
            :param best_dict:
            :param ms:
            :param mds:
            :return:
            """
            if len(ms) != len(mds):
                raise ValueError("验证指标和证券指标权重个数不匹配")
            result = pd.Series(np.zeros(s.shape[0]))
            for i in range(len(ms)):
                result += (s * (best_dict['training'][ms[i]] * mds[i]) / len(ms))
            return result

        # 准备数据
        main = _main[_main[_target].notnull()]
        main = DfProcessing.change_int64_to_float64(main)
        main = DfProcessing.change_object_to_category(main)
        # 临时为置换列名中的空格
        main.columns = main.columns.str.replace(' ', '!@!')
        trn_data = lgb.Dataset(main.loc[:, ~main.columns.isin([_target, _id])], label=main[_target])
        # 进行训练
        bst = lgb.train(_params, trn_data, valid_sets=[trn_data], verbose_eval=False)
        # 统计特征重要性
        feature_df = pd.DataFrame()
        feature_df['feature_name'] = bst.feature_name()
        feature_df['importance'] = metric_avg(
            bst.feature_importance(), bst.best_score, _params['metric'], _metric_directions
        )
        # 还原临时为置换列名中的空格
        feature_df['feature_name'] = feature_df['feature_name'].str.replace('!@!', ' ')
        return feature_df

    @staticmethod
    def train_lgb(
            _data: pd.DataFrame,
            _target: str,
            _target_id: str,
            _params: dict,
            _check_score_func: Callable,
            _n_fold: int = 5,
    ) -> (List[lgb.Booster], pd.DataFrame, Dict[str, pd.DataFrame]):
        """
        模型训练

        :param _data:
        :param _target:
        :param _target_id:
        :param _params:
        :param _check_score_func:
        :param _n_fold:

        :return:
        """
        def log_scores(log_score_tag: str, log_score_body: Dict[str, float]):
            logger.info('{}: {}', log_score_tag, log_score_body)

        def log_all_scores(all_scores: Dict[str, List[Dict[str, float]]]):
            for log_score_tag, log_score_body_list in all_scores.items():
                show_tag = []
                score_df = pd.DataFrame(log_score_body_list)
                for score_typ in score_df:
                    show_tag.append('{}: {}'.format(score_typ, pd.DataFrame(log_score_body_list)[score_typ].mean()))
                logger.info('{}: [{}]', log_score_tag, ', '.join(show_tag))

        # 定义随机种子
        seed = 2023
        # 指定HYPEROPT的随机种子
        os.environ['HYPEROPT_FMIN_SEED'] = str(seed)
        # 将不利于模型训练的数据类型进行转换
        _data = DfProcessing.change_int64_to_float64(_data)
        _data, cat_list = DfProcessing.change_object_to_int(_data, skips=[_target_id])
        # 切分数据集-训练数据
        train = _data[_data[_target].notnull()]
        train_x = train.loc[:, ~train.columns.isin([_target, _target_id])]
        train_y = train.loc[:, train.columns.isin([_target])]
        # 切分数据集-测试集
        test = _data[_data[_target].isnull()]
        test_x = test.loc[:, ~test.columns.isin([_target, _target_id])]
        # 评价分数
        scores = {'train': [], 'val': []}
        # 预测集
        predicts = {'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}
        predicts['train'][_target_id] = train[_target_id]
        predicts['val'][_target_id] = train[_target_id]
        predicts['val'][_target] = train[_target]
        predicts['test'][_target_id] = test[_target_id]
        # 预定义模型列表
        clf_list = []
        # 开始训练
        importance_df = pd.DataFrame()
        importance_df['Feature'] = train_x.columns
        # 进行K折划分
        stratified_k_fold = StratifiedKFold(_n_fold, shuffle=True, random_state=seed)
        for k, (trn_idx, val_idx) in enumerate(stratified_k_fold.split(train, train_y)):
            # 获得K折轮次
            fold_num = k + 1
            # 划分本K折的训练集和验证集
            trn_data = lgb.Dataset(train_x.iloc[trn_idx], label=train_y.iloc[trn_idx])
            val_data = lgb.Dataset(train_x.iloc[val_idx], label=train_y.iloc[val_idx])
            # 打印当前拆分结构
            logger.info('------ 第{}次K折拆分，训练形状{} ------', fold_num, train_x.iloc[trn_idx].shape)
            # 将训练的seed指定为折数，保证10折内各模型的初始参数是不同的，但是重复运行则10折是相同的
            if 'seed' not in _params:
                _params['seed'] = fold_num
            # 开始训练
            clf = lgb.train(
                params=_params,
                train_set=trn_data,
                categorical_feature=cat_list,
                valid_sets=[trn_data, val_data],
                verbose_eval=False,
                feval=_check_score_func,
            )
            scores['train'].append(dict(clf.best_score['training']))
            log_scores('train', dict(clf.best_score['training']))
            scores['val'].append(dict(clf.best_score['valid_1']))
            log_scores('val', dict(clf.best_score['valid_1']))
            # 获得特征重要性
            importance_df[fold_num] = clf.feature_importance()
            # 验证集-进行预测
            val_pred = pd.DataFrame()
            val_pred[_target_id] = train.iloc[val_idx][_target_id]
            val_pred[fold_num] = clf.predict(data=train_x.iloc[val_idx])
            predicts['val'] = pd.merge(predicts['val'], val_pred, on=_target_id, how='left')
            # 训练集-进行预测
            train_pred = clf.predict(data=train_x[train_x.columns])
            predicts['train'][fold_num] = train_pred
            # 测试集-进行预测
            test_pred = clf.predict(data=test_x[train_x.columns])
            predicts['test'][fold_num] = test_pred
            clf_list.append(clf)
        # 总览
        logger.info('---------------- 总览 ----------------')
        log_all_scores(scores)
        # 返回
        return clf_list, importance_df, predicts

    @staticmethod
    def train_step_by_about_importance(
            _data: pd.DataFrame,
            _target: str,
            _target_id: str,
            _params: dict,
            _check_score_func: Callable,
            _n_fold_list=None,
            _n_top_importance_list=None,
    ):
        if _n_top_importance_list is None:
            _n_top_importance_list = [-1, ]
        if _n_fold_list is None:
            _n_fold_list = [10, ]
        for train_time in range(len(_n_top_importance_list)):
            logger.info(
                '********************** 第{}次训练, {}, {}] **********************',
                train_time + 1, _n_fold_list[train_time], _n_top_importance_list[train_time],
            )
            # 如果不是全部，则需要截取数据
            if _n_top_importance_list[train_time] != -1:
                select_feature_after_train = TabBinary.view_importance(
                    importance_df,
                    _k=_n_fold_list[train_time],
                    _top=_n_top_importance_list[train_time],
                    _without_zero=True
                )['Feature'].to_list()
                select_feature_after_train.append(_target_id)
                select_feature_after_train.append(_target)
                _data = _data[select_feature_after_train]
            # 开始训练
            clf_list, importance_df, predicts = TabBinary.train_lgb(
                _data=_data,
                _target=_target,
                _target_id=_target_id,
                _params=_params,
                _check_score_func=_check_score_func,
                _n_fold=_n_fold_list[train_time],
            )
        return clf_list, importance_df, predicts

    @staticmethod
    def pred(
            _predicts: Dict[str, pd.DataFrame],
            _target_id: str,
            _target: str,
            _k: int,
            r: int,
            pred_tag_list: List[str] = ['概率均值_裸分,']
    ) -> Dict[str, pd.DataFrame]:
        result = {}
        for pred_tag in pred_tag_list:
            result[pred_tag] = pd.DataFrame()
            result[pred_tag][_target_id] = _predicts['test'][_target_id]
            result[pred_tag][_target] = 0
            if pred_tag == '概率均值_裸分':
                logger.info('{}', pred_tag)
                for k_fold in range(1, _k + 1):
                    result[pred_tag][_target] += (_predicts['test'][k_fold] / _k)
                result[pred_tag][_target] = result[pred_tag][_target].round().astype(int)
            if pred_tag == '各折搜阈_统划':
                logger.info('{}', pred_tag)
                m_threshold = 0
                for k_fold in range(1, _k + 1):
                    # 为测试集计算均值
                    result[pred_tag][_target] += (_predicts['test'][k_fold] / _k)
                    # 为验证集搜索阈值
                    tmp_val = _predicts['val'].loc[:, [_target_id, k_fold, _target]]
                    tmp_val = tmp_val[tmp_val[k_fold].notnull()]
                    m_threshold += Metrics.search_f1_best_threshold(tmp_val[k_fold], tmp_val[_target])[0] / _k
                result[pred_tag][_target] = Metrics.trans_pred(result[pred_tag][_target], m_threshold)
            if pred_tag == '概率均值_排名':
                logger.info('{}', pred_tag)
                for k_fold in range(1, _k + 1):
                    result[pred_tag][_target] += (_predicts['test'][k_fold] / _k)
                tmp = np.zeros(len(result[pred_tag]))
                for i in np.argsort(-np.array(result[pred_tag][_target]))[: r]:
                    tmp[i] = 1
                result[pred_tag][_target] = tmp
        return result

    @staticmethod
    def view_importance(
            _importance_df: pd.DataFrame,
            _k: int,
            _top: int = 10,
            _without_zero: bool = False
    ) -> pd.DataFrame:
        tmp = pd.DataFrame()
        tmp['Feature'] = _importance_df['Feature']
        tmp['Importance'] = _importance_df[[i for i in range(1, _k + 1)]].mean(axis=1)
        if _without_zero:
            tmp = tmp[tmp['Importance'] > 0]
        return tmp.sort_values(by='Importance', ascending=False).head(_top)

    @staticmethod
    def adv_verify(
            data: pd.DataFrame,
            target_id: str,
            target: str,
            del_col=None
    ) -> pd.DataFrame:
        # 复制一份数据
        if del_col is None:
            del_col = []
        _data = data.copy()
        # 删除列
        logger.info('删除列{}', del_col)
        _data.drop(columns=del_col, inplace=True)
        # 类型转换
        _data = DfProcessing.change_int64_to_float64(_data)
        _data = DfProcessing.change_object_to_category(_data)
        # 标注训练集和验证集
        train = _data[_data[target].notnull()]
        train_x = train.loc[:, ~train.columns.isin([target, target_id])]
        test_x = _data[_data[target].isnull()][train_x.columns]
        train_x['Is_Test'] = 0
        test_x['Is_Test'] = 1
        # 合并数据集
        df_adv = pd.concat([train_x, test_x])
        # 进行交叉验证训练
        adv_data = lgb.Dataset(data=df_adv.drop('Is_Test', axis=1), label=df_adv.loc[:, 'Is_Test'])
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'seed': 2022,
            'verbose': -1,
        }
        adv_cv_result = lgb.cv(params, adv_data, nfold=5, verbose_eval=-1, seed=2022)
        max_auc_mean = max(adv_cv_result['auc-mean'])
        n_estimators = adv_cv_result['auc-mean'].index(max_auc_mean) + 1
        logger.info('交叉验证中最优AUC为{}，迭代轮次为{}', max_auc_mean, n_estimators)
        params['n_estimators'] = n_estimators
        model_adv = lgb.LGBMClassifier(**params)
        model_adv.fit(df_adv.drop('Is_Test', axis=1), df_adv.loc[:, 'Is_Test'])
        lgb.plot_importance(model_adv, max_num_features=20)
        plt.show()
        df_importance = pd.DataFrame()
        df_importance['feature'] = list(df_adv.drop('Is_Test', axis=1).columns)
        df_importance['importance'] = model_adv.feature_importances_
        return df_importance.sort_values(by='importance', ascending=False)
