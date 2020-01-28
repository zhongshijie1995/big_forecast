import logging
import psutil
import os


def init(filename=None, level='DEBUG'):
    """
    根据传入文件名、日志级别，初始化日志输出工具

    :param filename: 日志输出文件名
    :param level: 日志级别
    :return: 无
    """
    level_name = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET']
    level_int = [50, 40, 30, 20, 10, 0]
    if str(level) in level_name:
        level_set = level_int[level_name.index(str(level))]
    else:
        level_set = level
        warn('您配置了不存在的日志级别 {} ，将不进行日志输出'.format(str(level)))
    log_fmt = '%(asctime)s - %(levelname)s - %(processName)s/%(threadName)s - %(message)s'
    logging.basicConfig(filename=filename, level=level_set, format=log_fmt)
    info('运行时日志初始化完成！')


def debug(*kwargs):
    """
    输出 DEBUG 日志

    :param kwargs: 日志参数
    :return: 无
    """
    logging.debug(*kwargs)


def info(*kwargs):
    """
    输出 INFO 日志

    :param kwargs: 日志参数
    :return: 无
    """
    logging.info(*kwargs)


def warn(*kwargs):
    """
    输出 WARN 日志

    :param kwargs: 日志参数
    :return: 无
    """
    logging.warning(*kwargs)


def error(*kwargs):
    """
    输出 ERROR 日志

    :param kwargs: 日志参数
    :return: 无
    """
    logging.error(*kwargs)


def critical(*kwargs):
    """
    输出 CRITICAL 日志

    :param kwargs: 日志参数
    :return: 无
    """
    logging.critical(*kwargs)


def memory_used():
    """
    输出 内存占用 信息

    :return: 无
    """
    logging.debug('目前内存消耗 {} {}'.format(psutil.Process(os.getpid()).memory_info().rss/1024/1024, 'Mb'))


# 默认运行日志参数
# init(filename=Run_Val.log_file_path_name, level=Run_Val.log_level)
