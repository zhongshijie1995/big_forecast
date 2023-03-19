import pandas as pd
import sqlalchemy


class DataframeSQL:
    @staticmethod
    def __get_sqlite_engine(db_path_name) -> sqlalchemy.engine.Engine:
        return sqlalchemy.create_engine('sqlite:///{}'.format(db_path_name))

    @staticmethod
    def df_into_sqlite(db_path_name: str, table_name: str, df: pd.DataFrame) -> None:
        engine = DataframeSQL.__get_sqlite_engine(db_path_name)
        with engine.connect() as conn:
            df.to_sql(table_name, conn, index=False)

    @staticmethod
    def sqlite_query_into_df(db_path_name: str, query: str) -> pd.DataFrame:
        engine = DataframeSQL.__get_sqlite_engine(db_path_name)
        with engine.connect() as conn:
            return pd.read_sql_query(query, conn)
