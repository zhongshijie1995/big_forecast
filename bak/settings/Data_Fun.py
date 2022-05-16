from bak.prepares import DealDataFrame


# 定义需要预处理的函数
def merge_for_sk_id(df_dict: dict) -> dict:
    df_dict['bureau_balance'] = df_dict['bureau_balance'].merge(
        df_dict['bureau'][['SK_ID_CURR', 'SK_ID_BUREAU']], on='SK_ID_BUREAU', how='left')
    return df_dict


# 定义分块读取
def set_idx(df_dict: dict) -> dict:
    df_dict = DealDataFrame.zip_dataset(df_dict)
    for df_name in df_dict:
        for c in df_dict[df_name]:
            if 'SK_ID' in c:
                df_dict[df_name][c] = df_dict[df_name][c].fillna(0).astype(DealDataFrame.np.int32)
    return df_dict
