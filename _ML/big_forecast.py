import base64
import os
import pickle
import sys
from typing import Dict, List, Callable, Any

# import dataprep.eda
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn.metrics
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import featuretools as ft


# ------------------------------------------- 公共配置 -------------------------------------------
class LogShorter:
    """
    简短日志快捷配置
    """

    def __init__(self):
        """
        初始化后，会将的输出日志改为精简版
        """
        logger.configure(
            handlers=[
                {
                    'sink': sys.stdout,
                    'format': '<cyan>{time:YYYY-MM-DD HH:mm:ss.SSS}</>[<lvl>{level:5}</>]-{message}',
                    'colorize': True
                }
            ]
        )


class PandasDisplayFully:
    """
    Pandas完整显示快捷配置
    """

    def __init__(self):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)


# ------------------------------------------- 文件操作 -------------------------------------------
class DataReptile:
    @staticmethod
    def plain_text_files_merge(plain_text_files_dir: str, target_file: str):
        """
        将纯文本文件合并为一个文件
        :param plain_text_files_dir: 待合并文件夹
        :param target_file: 目标文件
        :return: 无
        """
        plain_text_files = [os.path.join(plain_text_files_dir, i) for i in os.listdir(plain_text_files_dir)]
        result = ''
        for plain_text_file in plain_text_files:
            with open(plain_text_file, encoding='utf-8') as f:
                print(plain_text_file)
                tmp = ''.join(f.readlines())
                logger.info(f'读入[{plain_text_file}]长度[{len(tmp)}]')
                result += tmp
        with open(target_file, 'w', encoding='utf-8') as f:
            logger.info(f'写入[{target_file}]长度[{len(result)}]')
            f.write(result)
        logger.info('合并完成')
        return None

    @staticmethod
    def plain_text_file_split(plain_text_file: str, target_dir: str, max_len: int = 20 * 1000 * 1000):
        """
        将纯文本文件拆分为多个文件
        :param plain_text_file: 待拆分文件
        :param target_dir: 目标文件夹
        :param max_len: 单文件最大字符数
        :return: 无
        """
        with open(plain_text_file, encoding='utf-8') as f:
            result = ''.join(f.readlines())
            logger.info(f'读入[{plain_text_file}]长度[{len(result)}]')
        part_num = int(len(result) / max_len) + 1 if len(result) % max_len != 0 else 0
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        for i in range(part_num):
            this_file = os.path.join(target_dir, f'{str(i).rjust(len(str(part_num)) + 1, "0")}.txt')
            with open(this_file, 'w', encoding='utf-8') as f:
                x = i * max_len
                tmp = result[x: x + max_len]
                logger.info(f'写入[{this_file}]长度[{len(tmp)}]')
                f.write(result[x: x + max_len])
        return None

    @staticmethod
    def base64_to_file(b64_txt_file: str, bin_file: str):
        with open(b64_txt_file, 'r', encoding='utf-8') as f:
            b64_str = f.read()
        with open(bin_file, 'wb') as f:
            f.write(base64.b64decode(b64_str))
        logger.info(f'[{bin_file}]写入完成')

    @staticmethod
    def file_to_base64(bin_file: str, b64_txt_file: str):
        with open(bin_file, 'rb') as f:
            bin_data = f.read()
        with open(b64_txt_file, 'w', encoding='utf-8') as f:
            f.write(base64.b64encode(bin_data).decode('utf-8'))
        logger.info(f'[{b64_txt_file}]写入完成')


class PickleIO:
    @staticmethod
    def get_pickle(_file_name: str) -> Any:
        """
        读取文件到变量
        :param _file_name: 读取的文件
        :return: Python任意类型的变量
        """
        with open(_file_name, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save_pickle(_file_name: str, _bin: Any) -> None:
        """
        保存变量和文件
        :param _file_name: 写入的文件
        :param _bin: 变量
        :return: 无
        """
        with open(_file_name, 'wb') as f:
            pickle.dump(_bin, f)
        return None


# ------------------------------------------- 数据集合 -------------------------------------------
class CsvDataset:

    @staticmethod
    def load_csv(
            file_path: str,
            data_type_dict=None,
            datetime_type_dict=None,
            del_col_dict=None,
            header=None,
            header_names=None
    ):
        if header_names is None:
            header_names = []
        if del_col_dict is None:
            del_col_dict = {}
        if datetime_type_dict is None:
            datetime_type_dict = []
        if data_type_dict is None:
            data_type_dict = {}

        result = pd.read_csv(file_path, parse_dates=datetime_type_dict,
                             dtype=data_type_dict, low_memory=False)

        if del_col_dict is not None:
            result.drop(columns=del_col_dict, inplace=True)
        return result

    @staticmethod
    def set_col_name(df: pd.DataFrame, names: List[str], inplace: bool = True):
        result = df.copy() if not inplace else df
        result.columns = names
        return result

    @staticmethod
    def set_col_data_type(df: pd.DataFrame, type_dict: Dict[str, str], inplace: bool = True):
        result = df.copy() if not inplace else df
        result = result.astype(type_dict)
        return result

    @staticmethod
    def set_col_datetime(df: pd.DataFrame, col: str, col_format: str = None, inplace: bool = True):
        result = df.copy() if not inplace else df
        result[col] = pd.to_datetime(df[col], format=col_format)
        return result

    @staticmethod
    def make_df_dict(
            data_dir_list: List[str],
            mask_name_dict: Dict[str, str] = None,
            data_type_dict: Dict[str, Dict[str, str]] = None,
            datetime_type_dict: Dict[str, List[str]] = None,
            del_col_dict: Dict[str, List[str]] = None,
            header=0,
            header_names: Dict[str, List[str]] = None,
    ):
        """

        :param data_dir_list:
        :param mask_name_dict:
        :param data_type_dict:
        :param datetime_type_dict:
        :param del_col_dict:
        :param header:
        :param header_names:
        :return:
        """
        if mask_name_dict is None:
            mask_name_dict = {}
        if data_type_dict is None:
            data_type_dict = {}
        if datetime_type_dict is None:
            datetime_type_dict = {}
        if del_col_dict is None:
            del_col_dict = {}
        result = {}
        data_key_list = [os.path.basename(data_path) for data_path in data_dir_list]
        for data_key, data_dir in zip(data_key_list, data_dir_list):
            result[data_key] = {}
            data_path_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
            for data_path in data_path_list:
                if '.ipynb_checkpoints' in data_path:
                    continue
                file_name = os.path.basename(data_path)
                logger.debug(f'读入[{data_path}]')
                tab_name = file_name.replace('.csv', '').replace('.txt', '')
                if tab_name in mask_name_dict:
                    tab_name = mask_name_dict[tab_name]
                result[data_key][tab_name] = CsvDataset.load_csv(
                    data_path,
                    data_type_dict=data_type_dict.get(tab_name),
                    datetime_type_dict=datetime_type_dict.get(tab_name),
                    del_col_dict=del_col_dict.get(tab_name),
                    header=header,
                    header_names=header_names.get(tab_name),
                )
        for k, v in result.items():
            logger.info(f'--------------- {k} --------------- ')
            for vk, vv in v.items():
                logger.info(f'[{k}]-{vk}-{vv.shape}-{", ".join(list(vv.columns))}')
        return result

    @staticmethod
    def do_things_with_df_dict(
            df_dict: Dict[str, pd.DataFrame],
            tab_name: str,
            func: Callable,
            inplace: bool = True
    ):
        result = df_dict.copy() if not inplace else df_dict
        for k, v in result.items():
            result[k][tab_name] = func(v[tab_name])
        return result


# ------------------------------------------- 数据处理 -------------------------------------------
# class DataAnalyse:
#
#     @staticmethod
#     def report(df: pd.DataFrame):
#         """
#         绘制数据分析报告
#
#         :param df: 待分析表
#         :return: 分析图
#         """
#         return dataprep.eda.create_report(df)
#
#     @staticmethod
#     def plot(df: pd.DataFrame, cols: str = None):
#         """
#         数据可视化
#
#         :param df: 待分析表
#         :param cols: 专注列，可提供【无：概览】【1列：该列分析】【2列：2列间关系】
#         :return: 分析图
#         """
#         if cols is None:
#             cols = []
#         if len(cols) > 2:
#             cols = cols[0:2]
#         return dataprep.eda.plot(df, *cols)
#
#     @staticmethod
#     def plot_corr(df: pd.DataFrame, cols: List[str] = None):
#         """
#         相关性可视化
#
#         :param df: 待分析表
#         :param cols: 专注列，可提供【无：所有列间相关性】【1列：返回最相关列】【2列：2列间相关性分析】
#         :return: 分析图
#         """
#         if cols is None:
#             cols = []
#         if len(cols) > 2:
#             cols = cols[0:2]
#         return dataprep.eda.plot_correlation(df, *cols)
#
#     @staticmethod
#     def plot_missing(df: pd.DataFrame, cols: List[str] = None):
#         """
#         缺失值可视化
#
#         :param df: 待分析表
#         :param cols: 专注列，可提供【无：所有列缺失值分布】【1列：该列缺失值对其他列的影响】【2列：A列缺失值对B列的影响】
#         :return: 分析图
#         """
#         if cols is None:
#             cols = []
#         if len(cols) > 2:
#             cols = cols[0:2]
#         return dataprep.eda.plot_missing(df, *cols)
#
#     @staticmethod
#     def plot_diff(df1: pd.DataFrame, df2: pd.DataFrame, labels: List[str] = None, density: bool = False):
#         """
#         差异可视化
#
#         :param df1: 表1
#         :param df2: 表2
#         :param labels: 列表【表1名称,表2名称】
#         :param density: 是否输出密度图
#         :return: 分析图
#         """
#         config_dict = {}
#         if labels is not None:
#             config_dict['diff.label'] = labels
#         config_dict['diff.density'] = density
#         return dataprep.eda.plot_diff([df1, df2], config=config_dict)

class SimplePlot:
    @staticmethod
    def cat_diff_train_and_test(train_df, test_df, col):
        train_vc = pd.DataFrame(train_df[col].value_counts() / train_df.shape[0]).reset_index()
        train_vc.columns = ['cat', 'TRAIN']
        test_vc = pd.DataFrame(test_df[col].value_counts() / test_df.shape[0]).reset_index()
        test_vc.columns = ['cat', 'TEST']
        result = pd.merge(train_vc, test_vc, on='cat', how='left')
        result.plot()
        plt.show()


# ------------------------------------------- 特征工程 -------------------------------------------
class FeatureEngineer:
    @staticmethod
    def make_one_hot_cols(df: pd.DataFrame, target_cols: List[str] = None, inplace: bool = True):
        result = df.copy() if not inplace else df
        if target_cols is None:
            target_cols = df.select_dtypes(include=['O']).columns.tolist()
        for target_col in target_cols:
            if result[target_col].nunique() > 20:
                logger.warning(f'[{target_col}]列中数据类别超过20，放弃独热转换')
                continue
            result = result.join(pd.get_dummies(result[target_col], prefix=target_col))
            result.drop(target_col, axis=1, inplace=True)
        return result

    @staticmethod
    def ft_auto(
            id_col: str,
            label_col: str,
            target_df_name: str,
            df_map: Dict[str, pd.DataFrame],
            ft_dfs_params: Dict[str, Any] = None,
    ):
        # -------- 创建实体 --------
        es = ft.EntitySet(id=target_df_name)
        # 指定目标表
        es = es.add_dataframe(dataframe_name=target_df_name, dataframe=df_map[target_df_name], index=id_col)
        # 遍历其他表
        for k, v in df_map.items():
            if k == target_df_name:
                continue
            if id_col not in v.columns:
                continue
            logger.info(f'添加{k}表和关系')
            es = es.add_dataframe(dataframe_name=k, dataframe=v, index=f'{k}.id')
            es = es.add_relationship(target_df_name, id_col, k, id_col)
        # -------- 获得衍生特征列表 --------
        if ft_dfs_params is None:
            ft_dfs_params = {
                'agg_primitives': ['sum', 'std', 'max', 'skew', 'min', 'mean', 'count', 'num_unique'],
                'trans_primitives': ['day', 'month', 'divide_numeric'],
                'ignore_columns': {target_df_name: [label_col,],},
                'max_depth': 2,
                'cutoff_time': None,
            }
        feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name=target_df_name,
            features_only=True,
            agg_primitives=ft_dfs_params['agg_primitives'],
            trans_primitives=ft_dfs_params['trans_primitives'],
            ignore_columns=ft_dfs_params['ignore_columns'],
            max_depth=ft_dfs_params['max_depth'],
            cutoff_time=ft_dfs_params['cutoff_time'],
        )
        logger.info(f'获得衍生特征定义{len(feature_defs)}个')
        # -------- 对特征进行淘金 --------
        ft.calculate_feature_matrix()
        return es


# ------------------------------------------- 机器学习 -------------------------------------------
class MachineLearn:
    @staticmethod
    def train(
            x: pd.DataFrame,
            y: pd.Series,
            n_folds: int = 5,
            params: dict = None,
            feval: Callable = None,
            num_boost_round: int = 100,
            seed: int = 2024,
    ):
        if params is None:
            params = {}
        result = {}
        stratified_k_fold = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        for k, (train_idx, val_idx) in enumerate(stratified_k_fold.split(x, y)):
            # 获取本折数据
            print(f'------------ {k} ------------')
            all_data = lgb.Dataset(x, y)
            train_data = lgb.Dataset(x.iloc[train_idx], y.iloc[train_idx])
            val_data = lgb.Dataset(x.iloc[val_idx], y.iloc[val_idx])
            # 开始训练并记录训练数据
            eval_result = {}
            gbm = lgb.train(
                params=params,
                train_set=train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                feval=feval if feval is not None else None,
                callbacks=[lgb.log_evaluation(int(num_boost_round / 5)), lgb.record_evaluation(eval_result)],
                num_boost_round=num_boost_round,
            )
            best_score = dict(gbm.best_score['val'])
            print(f'best-best_iteration:[{gbm.best_iteration}], best-score[{best_score}]]')
            result[k] = {
                'gbm': gbm,
                'eval': eval_result,
            }
        return result

    @staticmethod
    def adv_verify(train_data_X: '训练集', test_data_X: '测试集', del_col: '删除的列' = [],
                   cats: '类别特征' = None) -> '对抗性验证':
        train_data = train_data_X.copy()
        test_data = test_data_X.copy()

        # 删除列
        train_data.drop(columns=del_col, inplace=True)
        test_data.drop(columns=del_col, inplace=True)

        train_data['Is_Test'] = 0
        test_data['Is_Test'] = 1
        assert (train_data.shape[1] == test_data.shape[1])
        df_adv = pd.concat([train_data, test_data])
        adv_data = lgb.Dataset(data=df_adv.drop('Is_Test', axis=1), label=df_adv.loc[:, 'Is_Test'])

        params = {
            'boosting_type': 'gbdt',
            'colsample_bytree': 0.9,
            'learning_rate': 0.1,
            'max_depth': 6,
            'num_leaves': 64,
            'objective': 'binary',
            'subsample': 0.9,
            'subsample_freq': 0,
            'metric': 'auc',
            'verbose': -1,
            'seed': 2021,
            'n_jobs': 10,
            'early_stopping_rounds': 20,
        }

        adv_cv_result = lgb.cv(params, adv_data, num_boost_round=1000, nfold=5, seed=2021, categorical_feature=cats)
        print('交叉验证中最优的AUC为 {:.5f}, 对应的标准差为{:.5f}.'.format(adv_cv_result['valid auc-mean'][-1],
                                                                           adv_cv_result['valid auc-stdv'][-1]))
        print('模型最优迭代次数为{}'.format(len(adv_cv_result['valid auc-mean'])))

        params['n_estimators'] = len(adv_cv_result['valid auc-mean'])
        del params['early_stopping_rounds']
        model_adv = lgb.LGBMClassifier(**params)
        model_adv.fit(df_adv.drop('Is_Test', axis=1), df_adv.loc[:, 'Is_Test'])

        preds_adv = model_adv.predict_proba(df_adv.drop('Is_Test', axis=1))[:, 1]

        # plt.rc("font", family="FZFangSong")
        # lgb.plot_importance(model_adv, max_num_features=25)
        # plt.title('Featuretances')
        # plt.show()

        df_importance = pd.DataFrame()
        df_importance['feature'] = list(df_adv.drop('Is_Test', axis=1).columns)
        df_importance['importance'] = model_adv.feature_importances_

        return preds_adv, df_importance.sort_values(by='importance', ascending=False), adv_cv_result['valid auc-mean'][
            -1]

    @staticmethod
    def better_adv_verify(train_data_X, test_data_X, del_col=[], cats=None):
        train_data = train_data_X.copy()
        test_data = test_data_X.copy()
        # 删除列
        train_data.drop(columns=del_col, inplace=True)
        test_data.drop(columns=del_col, inplace=True)
        # 区分数据集
        train_data['Is_Test'] = 0
        test_data['Is_Test'] = 1
        # 检查形状是否一致
        assert (train_data.shape[1] == test_data.shape[1])
        # 进行机器学习
        df_adv = pd.concat([train_data, test_data])
        adv_data = lgb.Dataset(data=df_adv.drop('Is_Test', axis=1), label=df_adv.loc[:, 'Is_Test'])
        params = {
            'learning_rate': 0.1,
            'metric': 'auc',
            'verbose': -1,
            'seed': 2024,
            'n_jobs': 10,
        }
        adv_cv_result = lgb.cv(params, adv_data, num_boost_round=100, nfold=5, seed=2024, categorical_feature=cats)
        # 取得CV中最高的AUC值
        adv_score = max(adv_cv_result['valid auc-mean'])
        # 返回对抗性分数
        return adv_score


# ------------------------------------------- 深度学习 -------------------------------------------
class DeepLearn:
    pass


# ------------------------------------------- 评价指标 -------------------------------------------
class Metrics:
    @staticmethod
    def f1(y_pred, y_true):
        func_name = 'F1'
        score = f1_score(
            y_true=y_true.get_label() if type(y_true) == lgb.Dataset else y_true,
            y_pred=y_pred.round()
        )
        is_higher_better = True
        return func_name, score, is_higher_better

    @staticmethod
    def roc_auc(y_pred, y_true):
        func_name = 'ROC'
        score = roc_auc_score(
            y_true=y_true.get_label() if type(y_true) == lgb.Dataset else y_true,
            y_score=y_pred
        )
        is_higher_better = True
        return func_name, score, is_higher_better

    @staticmethod
    def ks(y_pred, y_true):
        func_name = 'KS'
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true.get_label() if type(y_true) == lgb.Dataset else y_true, y_pred)
        score = max(tpr - fpr)
        is_higher_better = True
        return func_name, score, is_higher_better
