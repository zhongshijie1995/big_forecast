import os
import pandas as pd
from typing import Any
from utils import Log
from prepares import DealDataFile, DealDataFrame


def read_data_and_make_partitions(dp: str, sp: str, rs: dict, ds: set, dd: Any, ic: str, mt: str, op: str, oi: list) -> int:
    """
    读取数据文件，并制作分块数据文件

    :param dp: 数据目录
    :param sp: 跳过列表
    :param rs: 替换值字典
    :param ds: 数据集集合
    :param dd: 数据处理函数
    :param ic: 指定索引列名
    :param mt: 分块主表
    :param op: 分块数据文件目录
    :param oi: 分块数据文件时保留原索引的数据表列表
    :return: 分块数
    """
    def check_run_necessity(_op):
        """
        检查运行必要性

        :param _op: 分块数据文件目录
        :return: 分块数据文件目录下文件数
        """
        bs = 0
        if os.path.exists(op):
            # bs = len(os.listdir(op))
            bs = len([lists for lists in os.listdir(op) if os.path.isdir(os.path.join(op, lists))])
        if bs != 0:
            Log.info('数据文件分区检查：已经存在分区数据')
            return bs
        Log.info('数据文件分区检查：尚未存在分区数据')
        return bs

    # 运行必要性检查
    block_size = check_run_necessity(op)
    if block_size > 0:
        Log.info('跳过读取数据和创建分区数据文件')
        return block_size

    Log.info('开始读取数据文件...')
    if dp is not None:
        data = DealDataFile.get_data_dict_by_path(dp, sp)
    else:
        Log.critical('未配置数据文件路径')
        return 0
    Log.info('读取数据文件完成！')

    Log.info('开始执行替换异常数据命令...')
    if rs is not None:
        data = DealDataFrame.replace_val_from_df_dict(data, rs)
    Log.info('执行替换异常数据命令完成！')

    Log.info('开始混合训练集与预测集...')
    if ds is not None:
        data = DealDataFrame.mix_dataset(data, ds)
    Log.info('混合训练集与预测集完成！')

    Log.info('开始读执行额外操作数据命令...')
    if dd is not None:
        dd(data)
    Log.info('执行额外操作数据命令完成！')

    Log.info('开始压缩数据表...')
    data = DealDataFrame.zip_dataset(data)
    Log.info('压缩数据表完成！')

    Log.info('开始指定索引...')
    if ic is not None:
        data = DealDataFrame.set_index_in_df_dict(data, ic)
    Log.info('指定索引完成！')

    Log.info('开始创建数据分块...')
    if mt is not None and op is not None:
        block_size = intelligent_partition(data, mt, op, oi)
    Log.info('创建数据分块完成！')
    return block_size


def create_partition(data: dict, row_list: list, part_num: int, output_path: str, oi: list) -> None:
    """
    根据指定的行列表、分块数、输出路径，生成数据集字典的分块文件

    :param data: 数据集字典
    :param row_list: 行列表（由index组成的列表）
    :param part_num: 分块数
    :param output_path: 输出路径
    :param oi: 分块数据文件时保留原索引的数据表列表
    :return: 无返回值
    """
    if output_path is None:
        Log.critical('未配置分块数据文件路径')
        return
    directory = os.path.join(output_path, str(part_num + 1))
    if os.path.exists(directory):
        return
    else:
        os.makedirs(directory)
        for table_name in data:
            tmp_df = pd.DataFrame(data[table_name])
            if table_name in oi:
                tmp_df_subset = tmp_df[tmp_df.index.isin(row_list)].copy().reset_index()
            else:
                tmp_df_subset = tmp_df[tmp_df.index.isin(row_list)].copy().reset_index(drop=True)
            tmp_df_subset.to_csv('{}/{}.csv'.format(directory, table_name), index=False)
    return


def intelligent_partition(data: dict, table_name: str, output_path: str = None, oi: list = None) -> int:
    """
    智能数据分块数

    :param data: 数据集字典
    :param table_name: 数据表名
    :param output_path: 输出路径
    :param oi: 分块数据文件时保留原索引的数据表列表
    :return:
    """
    df = pd.DataFrame(data[table_name])
    len_target = df.shape[0]
    chunk_size = int(len_target/(len_target/2000))
    id_list = [list(df.iloc[i:i + chunk_size].index) for i in range(0, df.shape[0], chunk_size)]
    Log.info('开始进行分块操作，批次共有 {} ... （开启DEBUG级别日志时刻查看详细进度）'.format(len(id_list)))
    for i, ids in enumerate(id_list):
        Log.debug('开始处理分块操作 {}/{}'.format(i, len(id_list)))
        create_partition(data, ids, i, output_path, oi)
    Log.info('结束分块操作，共完成批次 {}'.format(len(id_list)))
    return chunk_size


if __name__ == '__main__':
    pass
