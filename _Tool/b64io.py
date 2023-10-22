import base64
import os


class B64IO:
    """
    Base64输入输出工具
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
    def b64file_split(b64_path: str, b64_part_path: str, part_size: int = 204800) -> str:
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
    def b64file_merge(b64_part_path: str, b64_path: str):
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
