import os
import featuretools as ft
import dask.bag as db
from dask.distributed import Client
from typing import Any

from utils import Log
from settings import Data_Val
from prepares import DealDataFile


def create_entity_set(dp: str, sp: list, esc: list, rls: list, od: Any, mt: str, oge: bool = False) -> Any:
    """
    创建实体集

    :param dp: 数据文件所在目录
    :param sp:  跳过文件列表
    :param esc: 定制实体列表
    :param rls: 定制关系列表
    :param od: 读取分块文件后的处理
    :param mt: 主表
    :param oge: 是否仅返回实体
    :return: 返回的实体集
    """

    if os.path.exists(os.path.join(dp, Data_Val.feature_matrix_part_file)):
        Log.debug('跳过创建实体集 {}'.format(dp))
        return None

    data = DealDataFile.get_data_dict_by_path(dp, sp)
    es = ft.EntitySet(id='clients')
    data = od(data)
    # 定制实体
    for x in esc:
        if len(x) != 2:
            return None
        if x[1] in data[x[0]]:
            es = es.entity_from_dataframe(entity_id=x[0], dataframe=data[x[0]], index=x[1])
        else:
            es = es.entity_from_dataframe(entity_id=x[0], dataframe=data[x[0]], make_index=True, index=x[1])
    # 定制关系
    r = []
    for x in rls:
        if len(x) != 4:
            return None
        r.append(ft.Relationship(es[x[0]][x[1]], es[x[2]][x[3]]))
    es = es.add_relationships(r)
    if oge:
        return es
    else:
        feature_matrix_from_entity_set(es, dp, mt)
        return None


def feature_matrix_from_entity_set(es: ft.EntitySet, dp: str, mt: str) -> None:
    """
    计算特征矩阵，并保存到指定目录

    :param es: 实体集
    :param dp: 路径
    :param mt: 主表
    :return: 无
    """
    feature_defs = Data_Val.feature_defs
    feature_defs = ft.load_features(open(feature_defs, 'rb'))
    feature_matrix = ft.calculate_feature_matrix(feature_defs, entityset=es, n_jobs=1, verbose=0)
    feature_matrix.to_csv(os.path.join(dp, 'p.csv'), index=True)


def compute_feature_defs(dp: str, sp: list, esc: list, rls: list, od: Any, mt: str, oge: bool = True) -> None:
    es = create_entity_set(dp=os.path.join(dp, str(1)), sp=sp, esc=esc, rls=rls, od=od, mt=mt, oge=oge)
    feature_defs = Data_Val.feature_defs
    if not os.path.exists(feature_defs):
        feature_names = ft.dfs(
            entityset=es, target_entity=mt, agg_primitives=Data_Val.agg_primitives,
            trans_primitives=Data_Val.trans_primitives,
            n_jobs=-1, verbose=1, features_only=True, max_depth=2
        )
        ft.save_features(feature_names, Data_Val.feature_defs)


def get_feature_matrix_dask(op: str, sp: list, esc: list, rls: list, od: Any, mt: str, ck: int) -> None:
    """
    通过并行运算方式获取特征矩阵

    :param op:
    :param sp:
    :param esc:
    :param rls:
    :param od:
    :param mt:
    :param ck:
    :return:
    """

    if len(DealDataFile.get_all_suffix_files(op, 'p.csv')) == ck:
        Log.info('分块特征矩阵已存在，跳过计算特征矩阵！')
        return

    Log.info('开始计算特征定义...')
    compute_feature_defs(dp=op, sp=sp, esc=esc, rls=rls, od=od, mt=mt, oge=True)
    Log.info('特征定义完成！')

    Log.info('开始并行计算特征矩阵... {}'.format(op))
    client = Client(processes=True)
    path_sequence = [os.path.join(op, str(i + 1)) for i in range(ck)]
    b = db.from_sequence(path_sequence)
    b = b.map(create_entity_set, sp=sp, esc=esc, rls=rls, od=od, mt=mt)
    b.compute()
    client.close()
    Log.info('并行计算特征矩阵完成！')


if __name__ == '__main__':
    pass
