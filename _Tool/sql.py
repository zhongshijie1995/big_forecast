from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger
from pandasql import sqldf

from _Tool.io import VarIO


class DataframeSQL:
    @staticmethod
    def exec_sql(context: Dict, q: str) -> pd.DataFrame:
        for k, v in context.items():
            locals()[k] = v
        logger.debug('执行SQL语句[{}]', q)
        result = sqldf(q)
        logger.debug('执行结果形状为[{}]', result.shape)
        return result

    @staticmethod
    def select(context: Dict, cols: List[str] = None, tabs: Tuple[str] = None, cons: List[str] = None) -> pd.DataFrame:
        if cols is None:
            cols = ['*']
        if cons is None:
            cons = ['1=1']
        df_dict = VarIO.filter_by_type(context, pd.DataFrame) if tabs is None else VarIO.filter_by_name(context, tabs)
        q = 'select {} from {} where {}'.format(
            ','.join(cols), ','.join(df_dict.keys()), ' and '.join(cons)
        )
        result = DataframeSQL.exec_sql(context, q)
        return result
