import os
import pickle
from typing import Any, List, Dict, Tuple, Union

from loguru import logger


class FileIO:
    """
    文件存取
    """

    @staticmethod
    def get_pickle(_file_name: str) -> Any:
        """
        读取文件到变量
        :param _file_name: 读取的文件
        :return: Python任意类型的变量
        """
        with open(_file_name, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save_pickle(_file_name: str, _bin: Any) -> None:
        """
        保存变量和文件
        :param _file_name: 写入的文件
        :param _bin: 变量
        :return: 无
        """
        with open(_file_name, 'wb') as f:
            pickle.dump(_bin, f)
        return None

    @staticmethod
    def get_file_list(_path: str, _reverse: bool = False) -> List[str]:
        """
        获取文件列表
        :param _path: 路径
        :param _reverse: 逆序
        :return: 文件列表
        """
        result = []
        for dir_path, dir_names, file_names in os.walk(_path):
            for file_name in file_names:
                result.append(os.path.join(dir_path, file_name))
        result.sort(reverse=_reverse)
        return result

    @staticmethod
    def get_base_name(_path: str) -> str:
        """
        给定文件路径，获取文件基础名

        :param _path: 目录

        :return: 文件名列表
        """
        return os.path.basename(os.path.splitext(_path)[0])

    class SplitTxt:
        """
        文本文件分割操作类
        """

        def __init__(self, old_path, new_path, chunk_size=1024):
            """
            文本分割工具

            :param old_path: 旧路径
            :param new_path: 新路径
            :param chunk_size: 分块大小
            """
            self.chunk_size = chunk_size
            self.old_path = old_path
            self.new_path = new_path

        def split_file(self):
            """
            将文件分割成大小为chunk_size的块

            :return:
            """
            if not os.path.exists(self.new_path):
                os.makedirs(self.new_path, exist_ok=True)
            chunk_num = 0
            input_file = open(self.old_path, 'rb')
            try:
                while True:
                    chunk = input_file.read(self.chunk_size)
                    if not chunk:
                        break
                    chunk_num += 1
                    filename = os.path.join(self.new_path, ("%02d.txt" % chunk_num))
                    file_obj = open(filename, 'wb')
                    file_obj.write(chunk)
                    file_obj.close()
            except IOError:
                logger.info("文件读取错误！")
                raise IOError
            finally:
                input_file.close()
            return chunk_num

        def merge_file(self):
            """
            将多个文件合并成一个文件

            :return:
            """
            if not os.path.exists(self.new_path):
                logger.info("待合并文件路径不存在，请输入正确路径！")
                raise IOError
            files = os.listdir(self.new_path)
            files.sort()
            with open(self.old_path, 'w') as output:
                for each_file in files:
                    if 'ipynb_checkpoints' in each_file:
                        continue
                    logger.info(os.path.join(self.new_path, each_file))
                    filepath = os.path.join(self.new_path, each_file)
                    with open(filepath, 'r') as infile:
                        data = infile.read()
                        output.write(data)


class VarIO:
    @staticmethod
    def filter_by_type(context: dict, want_types: Union[Any, Tuple[Any]]) -> Dict[str, Any]:
        """
        根据变量类型进行变量过滤
        :param context: 变量字典
        :param want_types: 想要的类型
        :return:
        """
        result = {}
        for k, v in context.items():
            if str(k).startswith('_'):
                continue
            if isinstance(v, want_types):
                result[k] = v
                logger.debug('通过[{}]提取到变量[{}]', ','.join([x.__name__ for x in want_types]), k)
        return result

    @staticmethod
    def filter_by_name(context: dict, want_names: Tuple[str]) -> Dict[str, Any]:
        """
        根据变量名进行变量过滤
        :param context:
        :param want_names:
        :return:
        """
        result = {}
        for k, v in context.items():
            if k in want_names:
                result[k] = v
                logger.debug('通过[{}]提取到变量[{}]', '名称', k)
        return result
