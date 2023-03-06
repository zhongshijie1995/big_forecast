from typing import Dict, Any, Callable


class Choose:
    """
    交互选择
    """

    @staticmethod
    def choose_func(choose_dict: Dict[int, Callable] = None, kv: Dict[str, Any] = None) -> None:
        """
        选择函数执行

        :param choose_dict: 手动指定选择字典
        :param kv: 自动选择字典（本地变量）
        :return:
        """
        if choose_dict is None:
            choose_dict = {}
        if kv is not None:
            func_list = []
            kvs = kv.copy()
            for k, v in kvs.items():
                if str(type(v)).startswith('<class \'function\''):
                    func_list.append(v)
            func_list += list(choose_dict.values())
            for i, v in enumerate(func_list):
                choose_dict[i + 1] = v
        print('--------')
        choose_desc = '\n'.join([
            '[{:>3}] {:<35s} {:<35s}'.format(k, v.__name__, str(v.__doc__).strip().split()[0])
            for k, v in choose_dict.items()
        ])
        choose = input('{}\n{}: '.format(choose_desc, '请输入需要执行的操作'))
        print('--------')
        choose_dict.get(int(choose))()
        return None
