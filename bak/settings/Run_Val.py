from bak.utils import Log

# --------------------1. 日志配置 --------------------
"""
log_file_path_name:
    - 描述：日志输出
    - 取值：默认None，可填写文件绝对路径
log_level:
    - 描述：日志级别
    - 取值：默认DEBUG，可选: 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'
"""
log_file_path_name = None
log_level = 'DEBUG'


# --------------------2. 性能配置 --------------------
"""
split:
    - 描述：性能配置
    - 取值：值越大，数据分块粒度越小，速度越快，内存占用越高
"""
split = 2000

# --------------------运行时共享变量--------------------
dataset_names = set([])


def init():
    Log.init(log_file_path_name, log_level)
