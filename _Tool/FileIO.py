import base64
import os
import pickle
from typing import Any, Dict, List

from loguru import logger


class B64IO:
    """
    Base64输入输出
    """

    @staticmethod
    def file_to_b64(file_path: str, b64_path: str) -> str:
        """
        二进制文件转Base64文件

        :param file_path: 二进制文件路径
        :param b64_path: Base64文件路径
        :return: Base64文本
        """
        with open(file_path, 'rb') as f:
            result = base64.b64encode(f.read()).decode('utf-8')
        with open(b64_path, 'w') as f:
            f.write(result)
        return result

    @staticmethod
    def b64_to_file(b64_path: str, file_path: str) -> bytes:
        """
        Base64文件转二进制文件

        :param b64_path: Base64文件路径
        :param file_path: 二进制文件路径
        :return: 二进制字节流
        """
        with open(b64_path, 'r') as f:
            result = base64.b64decode(f.read())
        with open(file_path, 'wb') as f:
            f.write(result)
        return result

    @staticmethod
    def b64_split(b64_path: str, b64_part_path: str, part_size: int = 204800) -> str:
        """
        Base64文件分割

        :param b64_path: Base64文件路径
        :param b64_part_path: Base64分割文件路径
        :param part_size: 分区大小（1=1字符=1B，2048=2KB）
        :return:
        """
        with open(b64_path, 'r') as f:
            content = f.read()
        part_num = int(len(content) / part_size) + 1
        os.makedirs(b64_part_path, exist_ok=True)
        for i in range(part_num):
            with open(os.path.join(b64_part_path, '%09d.b64' % i), 'w+') as f:
                f.write(content[i * part_size: (i + 1) * part_size])
        return b64_part_path

    @staticmethod
    def b64_merge(b64_part_path: str, b64_path: str):
        """
        Base64文件合并

        :param b64_part_path: Base64分割文件路径
        :param b64_path: Base64文件路径
        :return:
        """
        file_path_list = [os.path.join(b64_part_path, i) for i in os.listdir(b64_part_path)]
        file_path_list.sort()
        result = ''
        for i in file_path_list:
            with open(i, 'r') as f:
                result += f.read()
        with open(b64_path, 'w+') as f:
            f.write(result)
        return result


class TxtIO:
    """
    txt输入输出
    """

    @staticmethod
    def txt_merge(txt_part_path: str, txt_path: str, new_line: bool = True):
        """
        txt文件合并

        :param txt_part_path: txt分割文件路径
        :param txt_path: txt文件路径
        :param new_line: 文件换行
        :return: 合并结果
        """
        file_path_list = [os.path.join(txt_part_path, i) for i in os.listdir(txt_part_path)]
        file_path_list.sort()
        result = ''
        for i in file_path_list:
            with open(i, 'r') as f:
                new_content = f.read()
                if new_line and not new_content.endswith('\n'):
                    new_content += '\n'
                result += new_content
        with open(txt_path, 'w+') as f:
            f.write(result)
        return result

    @staticmethod
    def txt_replace_line(txt_path: str, line_idx: int, target_str: str):
        """
        txt文件替换指定行号的内容

        :param txt_path: 文件路径
        :param target_str: 替换后的字符
        :param line_idx: 行号，从0开始
        :return:
        """
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        lines[line_idx] = target_str + os.linesep
        logger.info('替换[{}]的第[{}]行内容为[{}]', txt_path, line_idx, target_str)
        with open(txt_path, 'w') as f:
            f.writelines(lines)


class FileNameIO:
    """
    文件名输入输出
    """

    @staticmethod
    def batch_rename(rename_dict: Dict[str, str], base_path: str = None) -> None:
        """
        批量重命名文件

        :param rename_dict: 重命名字典[‘原文件路径’， ‘新文件路径’]
        :param base_path: 统一的父目录，若没有，则不提供
        :return:
        """
        for k, v in rename_dict.items():
            f = os.path.join(base_path, k) if base_path is not None else k
            t = os.path.join(base_path, v) if base_path is not None else v
            if os.path.exists(f):
                logger.info('重命名文件[{}]->[{}]', f, t)
                os.rename(f, t)
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


class PickleIO:
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
