import pandas as pd
import sqlalchemy


class DataframeSQL:
    @staticmethod
    def df_into_sqlite(db_path_name: str, table_name: str, df: pd.DataFrame) -> None:
        engine = sqlalchemy.create_engine('sqlite:///'.format(db_path_name))
        df.to_sql(table_name, engine)

    @staticmethod
    def sqlite_into_df(db_path_name: str, table_name: str) -> pd.DataFrame:
        engine = sqlalchemy.create_engine('sqlite:///'.format(db_path_name))
        return pd.read_sql(table_name, engine)
