"""
-*- coding: utf-8 -*-
@File  : submits.py
@Author: 钟世杰
@Date  : 2023/2/21
@Desc  : 
@Contact : zhongshijie1995@outlook.com
"""
import os
from typing import List


class Mix:
    @staticmethod
    def mix_csv(_file_path_list: List[str], _weight_list: List[float]):
        pass

    @staticmethod
    def tag_csv(_file_path: str, score: float) -> None:
        new_path_name = _file_path.replace('.csv', '-{}.csv'.format(score))
        os.rename(_file_path, new_path_name)
        return None
