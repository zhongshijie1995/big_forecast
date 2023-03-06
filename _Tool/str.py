"""
-*- coding: utf-8 -*-
@File  : str.py
@Author: 钟世杰
@Date  : 2023/1/28
@Desc  : 字符处理
@Contact : zhongshijie1995@outlook.com
"""
import re
from typing import List, Union


class StrList:
    """
    字符串列表操作
    """

    @staticmethod
    def filter_keys(
            _strs: List[str],
            _keys: List[str],
            _contain: bool = True,
            _re_flags: Union[int, re.RegexFlag] = 0,
    ) -> List[str]:
        """
        以关键词进行过滤

        :param _strs: 待过滤的字符串列表
        :param _keys: 关键词列表
        :param _contain: 过滤逻辑为包含
        :param _re_flags: 正则表达式标记
        :return:
        """
        result = []
        for _key in _keys:
            for _str in _strs:
                search_flag = False if re.search(_key, _str, flags=_re_flags) is None else True
                if not _contain ^ search_flag:
                    result.append(_str)
        return result
