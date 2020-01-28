from settings import Run_Val
from settings import Data_Val
from prepares import PartitionsDataFile, FeatureEngineer, DealDataFile
from fits import MLBoxFits


if __name__ == '__main__':
    # 初始化运行时配置
    Run_Val.init()

    # 进行数据分块
    chunk_size = PartitionsDataFile.read_data_and_make_partitions(
        dp=Data_Val.dp, sp=Data_Val.sp, rs=Data_Val.rs, ds=Data_Val.ds, dd=Data_Val.dd,
        ic=Data_Val.ic, mt=Data_Val.mt, op=Data_Val.op, oi=Data_Val.oi
    )

    # 深度特征工程
    FeatureEngineer.get_feature_matrix_dask(
        op=Data_Val.op, sp=Data_Val.sp, esc=Data_Val.esc, rls=Data_Val.rls, od=Data_Val.od,
        mt=Data_Val.mt, ck=chunk_size
    )
    DealDataFile.merge_p_by_path(Data_Val.op, Data_Val.feature_matrix_part_file, Data_Val.feature_matrix)

    # MLBox自动学习
    MLBoxFits.go()
