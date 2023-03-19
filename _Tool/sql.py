from typing import Dict

import pandas as pd
import sqlalchemy
from loguru import logger


class DataframeSQL:
    @staticmethod
    def __get_sqlite_engine(db_path_name) -> sqlalchemy.engine.Engine:
        return sqlalchemy.create_engine('sqlite:///{}'.format(db_path_name))

    @staticmethod
    def df_into_sqlite(db_path_name: str, table_name: str, df: pd.DataFrame) -> None:
        engine = DataframeSQL.__get_sqlite_engine(db_path_name)
        with engine.connect() as conn:
            logger.info('数据库连接为[]，写入数据表[{}], 形状为[{}]', engine.url, table_name, df.shape)
            df.to_sql(table_name, conn, index=False)
        return None

    @staticmethod
    def df_dict_into_sqlite(db_path_name: str, df_dict: Dict[str, pd.DataFrame]) -> None:
        for k, v in df_dict.items():
            logger.info('提取[{}]，形状[{}]', k, v.shape)
            DataframeSQL.df_into_sqlite(db_path_name, k, v)
        return None

    @staticmethod
    def sqlite_query_into_df(db_path_name: str, query: str) -> pd.DataFrame:
        engine = DataframeSQL.__get_sqlite_engine(db_path_name)
        with engine.connect() as conn:
            logger.info('数据库连接为[]，查询语句[{}]', engine.url, query)
            return pd.read_sql_query(query, conn)
