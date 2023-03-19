import sys
from loguru import logger


class LoggerFormat:
    @staticmethod
    def set(format_str_list: str = None) -> None:
        if format_str_list is None:
            format_str_list = [
                '<green>{time:YYYY-MM-DD HH:mm:ss}</green>',
                '<cyan>{module}.{function}:{line}</cyan>',
                '<yellow>{message}</yellow>'
            ]
        logger.remove()
        logger.add(sys.stderr, format='| '.join(format_str_list))
        logger.info('日志格式指定成功')
        return None
