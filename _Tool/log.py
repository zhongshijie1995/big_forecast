import time
from typing import Callable, Any, Dict, Tuple

from loguru import logger


class FuncLog:
    """
    函数日志打印器
    """

    @staticmethod
    def cost(func: Callable) -> Any:
        """
        开销记录日志

        :param func: 待执行函数

        :return:
        """

        def flat_args(args: Tuple[Any], kwargs: Dict[str, Any]) -> str:
            """
            压平参数

            :param args: 普通参数
            :param kwargs: 关键字参数

            :return: 返回参数后的字符
            """
            tmp = []
            for arg in args:
                tmp.append(str(arg).replace('\n', ''))
            for kw, arg in kwargs.items():
                tmp.append('{}={}'.format(kw, str(arg).replace('\n', '')))
            return ', '.join(tmp)

        def wrapper(*args, **kwargs) -> Any:
            """
            切面函数

            :param args: 传入参数
            :param kwargs: 传入参数

            :return: 函数执行结果
            """
            # 获取函数名称
            func_name = str(func.__name__)
            # 获取函数文档摘要
            func_docs = str(func.__doc__)
            summary = '' if func_docs is None else func_docs.strip().split()[0]
            # 打印开始标记
            logger.debug('执行[{}]({})，传入参数[{}]', func_name, summary, flat_args(args, kwargs))
            # 登记开始执行时间
            start_time = time.time()
            # 执行函数
            result = func(*args, **kwargs)
            # 登记结束时间
            end_time = time.time()
            # 计算耗时
            cost_time = round(end_time - start_time, 1)
            # 打印结束标记
            logger.debug('结束[{}]，共耗时[{}]秒', func_name, cost_time)
            return result

        # 返回切面函数
        return wrapper


if __name__ == '__main__':
    pass
