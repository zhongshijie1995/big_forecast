import numpy as np

from settings import Data_Fun

# --------------------1. 参数配置 ---------------------
"""
dp: 
    - 描述：待读入数据目录
    - 取值：绝对目录，例如：'/data/home-credit-default-risk'
sp: 
    - 描述：跳过读取的文件列表
    - 取值：文件名列表，例如： [sample_submission.csv, description.csv]、可填[]
rs: 
    - 描述：替换值字典
    - 取值：需要被总体替换的字典，例如：{6365243: np.nan}、可填None
ds: 
    - 描述：数据集集合
    - 取值：数据目录中包含的数据集， 例如：{'train', 'test'}
    - 备注：上述示例，反应为数据文件的数据集分类以_train、_test结尾
dd: 
    - 描述：数据处理函数，用于读取数据集后操作
    - 取值：参考Data_Fun的任意一个自行开发，例如：Data_Fun.merge_for_sk_id、可填None
ic:
    - 描述：指定索引列名
    - 取值：用于标识学习和预测的行索引，例如：'SK_ID_CURR'
mt: 
    - 描述：分块主表
    - 取值：用于确定分块数据的主表，例如：'application'（该名称为去除_train、_test、.csv的文件名）
op: 
    - 描述：存放分块数据文件目录
    - 取值：绝对目录，例如：'/data/home-credit-default-risk-partitions'
oi: 
    - 描述：分块数据文件时保留原索引的数据表列表
    - 取值：表名列表，例如：['application', 'bureau', 'previous_application']、可填[]
od: 
    - 描述：读取分块数据后的处理函数
    - 取值：参考Data_Fun的任意一个自行开发，例如：Data_Fun.set_idx、可填None
tg:
    - 描述：标签列名
sc:
    - 描述：预测结果样例csv文件
rc:
    - 描述：预测结果csv文件
"""
dp = 'D:\\99_Data\\02_home-credit-default-risk'
sp = ['feature_matrix_article.csv', 'HomeCredit_columns_description.csv', 'sample_submission.csv', 'p.csv']
rs = {6365243: np.nan}
ds = {'train', 'test'}
dd = Data_Fun.merge_for_sk_id
ic = 'SK_ID_CURR'
mt = 'application'
op = 'D:\\99_Data\\02_home-credit-default-risk-partitions'
oi = ['application', 'bureau', 'previous_application']
od = Data_Fun.set_idx
tg = 'TARGET'
sc = 'D:\\99_Data\\02_home-credit-default-risk\\sample_submission.csv'
rc = 'D:\\99_Data\\02_home-credit-default-risk-partitions\\result.csv'

"""
esc:
    - 描述: 定制实体列表
    - 取值: 列表嵌套元组，例如：[(表1名, 表1索引名), (表2名, 表2索引名)]
rls:
    - 描述: 定制关系列表
    - 取值: 列表嵌套元组，例如：[(表1名, 表1索引名, 表2名, 表2索引名)]
"""
esc = [
    ('application', 'SK_ID_CURR'),
    ('bureau', 'SK_ID_BUREAU'),
    ('bureau_balance', 'bureau_balance_index'),
    ('credit_card_balance', 'credit_card_balance_index'),
    ('installments_payments', 'installments_payments_index'),
    ('POS_CASH_balance', 'POS_CASH_balance_index'),
    ('previous_application', 'SK_ID_PREV')
]
rls = [
    ('application', 'SK_ID_CURR', 'bureau', 'SK_ID_CURR'),
    ('bureau', 'SK_ID_BUREAU', 'bureau_balance', 'SK_ID_BUREAU'),
    ('application', 'SK_ID_CURR', 'previous_application', 'SK_ID_CURR'),
    ('previous_application', 'SK_ID_PREV', 'POS_CASH_balance', 'SK_ID_PREV'),
    ('previous_application', 'SK_ID_PREV', 'installments_payments', 'SK_ID_PREV'),
    ('previous_application', 'SK_ID_PREV', 'credit_card_balance', 'SK_ID_PREV')
]

"""
feature_defs:
    - 描述: 自动特征工程生成的特征名文件路径
feature_matrix:
    - 描述: 自动特征工程生成的矩阵文件路径
feature_matrix_part_file:
    - 描述: 自动特征工程生成的矩阵文件的分块文件名
"""
feature_defs = 'D:\\99_Data\\02_home-credit-default-risk-partitions\\features_defs'
feature_matrix = 'D:\\99_Data\\02_home-credit-default-risk-partitions\\features.csv'
feature_matrix_part_file = 'features_part.csv'

# --------------------2. 调教配置 ----------------------
"""
agg_primitives：
    - 描述： 聚合元选项列表
    - 可选列表：['sum', 'max', 'min', 'mean', 'count', 'percent_true', 'num_unique', 'mode']
trans_primitives：
    - 描述：交易元选项列表
    - 可选列表：['percentile', 'and']
"""
agg_primitives = ['sum', 'max', 'min', 'mean', 'count', 'percent_true', 'num_unique', 'mode']
trans_primitives = ['percentile', 'and']

"""
space：
    - 描述：自动学习的搜索空间
    - 可参考下述文档中的kaggle示例
    - https://mlbox.readthedocs.io/en/latest/
scroing：
    - 描述：评价指标
    - 可参考下述文档中的kaggle示例
    - https://mlbox.readthedocs.io/en/latest/
n_folds：
    - 描述：交叉验证折数
    - 可参考下述文档中的kaggle示例
    - https://mlbox.readthedocs.io/en/latest/
"""
space = {'est__strategy': {'search': 'choice', 'space': ['LightGBM']}}
scoring = 'roc_auc'
n_folds = 3
