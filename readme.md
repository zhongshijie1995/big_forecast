# tabular_forecast
表格类数据预测的机器学习自动化框架，只需几行代码解决预测问题。
## 框架说明
### 1. 适用场景
- **有监督学习**的**多表联合**的数据**预测**类型问题（分类、回归）
### 2. 主要功能
- 配置式自动化读取、预处理数据
- 配置式自动化特征工程
- 配置式自动化机器学习模型训练、调参、预测

## 安装说明
### 1. 克隆项目
```shell script
git clone https://gitee.com/zhongshijie/tabular_forecast
```
### 2. 安装必要依赖
```shell script
cd tabular_forecast
pip install -r requirements.txt
```
- 感谢依赖打包项目：[pipreqs](https://github.com/bndr/pipreqs)
- 相关核心开源项目：[featuretools](https://docs.featuretools.com/en/stable/)、 [MLBox](https://mlbox.readthedocs.io/en/latest/)、[dask](https://docs.dask.org/en/latest/install.html)
## 使用说明
### 1. 配置运行参数
```shell script
vi ./settings/Run_Val.py
```
结合注释：
- 根据需要编辑```1. 日志配置```
- 根据需要编辑```2. 性能配置```
### 2. 开发自定义处理函数（可选）
```shell script
vi ./settings/Data_Fun.py
```
- 结合对数据情况的掌握，开发合适的函数，用于在```数据参数配置```中进行使用，例如：明确已经的错误数据替换。
### 3. 配置数据参数
```shell script
vi ./settings/Data_Val.py
```
结合注释：
- 根据需要编辑```1. 参数配置```
- 根据需要编辑```2. 调教配置```
### 4. 运行主程序
```shell script
python ./Main.py
```
程序将全自动完成数据读取、特征工程、训练、预测，运行完成后，你可以根据配置参数中设置的预测结果路径来获得预测结果。
    

## Demo
项目本身已经处于配置完成状态，针对内容为：[Kaggle | Competitions | Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)，做了如下处理，你可以作为参考：
1. 准备工作
    - 确定数据目录，分析各项必备内容：
        - 源数据目录（读）：```D:\\99_Data\\02_home-credit-default-risk```
        - 分块数据目录（写）：```D:\\99_Data\\02_home-credit-default-risk-partitions```
        - 标签列名：```TARGET```
        - ...
2. 自定义函数开发(```./settings/Data_Fun.py```)
    - 根据数据分析，开发了如下自定义函数：
        - merge_for_sk_id
        - set_idx
3. 运行参数配置
    - 根据自己的调试需要和性能需求设置：
        - ```log_level = 'DEBUG'```
        - ```split = 2000```
4. 数据参数配置
    - 将 ```1.准备工作```和```2.自定义函数开发```的相应内容进行填写
        - ```dp = 'D:\\99_Data\\02_home-credit-default-risk'```
        - ```sp = ['feature_matrix_article.csv', 'HomeCredit_columns_description.csv', 'sample_submission.csv', 'p.csv']```
        - ```rs = {6365243: np.nan}```
        - ...

## 开发文档
### 模块：prepares
#### DealDataFile
##### ```get_data_dict_by_path(base_path: str, skips=None) -> dict```
##### ```get_all_suffix_files(base_path: str, suffix: str, skips: list = None) -> list```
##### ```get_base_name_dict(base_path: str, path_name_list: list) -> dict```
##### ```merge_p_by_path(base_path: str, p_name: str, output_path: str) -> None```
#### DealDataFrame
##### ```replace_val_from_df_dict(data: dict, replace_dict: dict) -> dict```
##### ```replace_types_in_df_dict(data: dict, replace_dict: dict) -> dict```
##### ```set_index_in_df_dict(data: dict, index_col_name: str) -> dict```
##### ```mix_dataset(data: dict, d_name: set = None) -> dict```
##### ```convert_types(df: pd.DataFrame) -> pd.DataFrame```
##### ```zip_dataset(df_dict: dict) -> dict```
#### FeatureEngineer
##### ```create_entity_set(dp: str, sp: list, esc: list, rls: list, od: Any, mt: str, only_get_es: bool = False) -> Any```
##### ```feature_matrix_from_entity_set(es: ft.EntitySet, dp: str, mt: str) -> None```
##### ```compute_feature_defs(dp: str, sp: list, esc: list, rls: list, od: Any, mt: str, oge: bool = True) -> None```
##### ```get_feature_matrix_dask(op: str, sp: list, esc: list, rls: list, od: Any, mt: str, ck: int) -> None```
#### PartitionsDataFile
##### ```read_data_and_make_partitions(dp: str, sp: str, rs: dict, ds: set, dd: Any, ic: str, mt: str, op: str, oi: list) -> int```
##### ```create_partition(data: dict, row_list: list, part_num: int, output_path: str, oi: list) -> None```
##### ```intelligent_partition(data: dict, table_name: str, output_path: str = None, oi: list = None) -> int```
### 模块：fits
#### MLBoxFits
##### ```go()```
### 模块：setting
#### Data_Fun
##### 自定义函数1
##### 自定义函数2
##### ...
#### Data_Val
##### 1.参数配置
##### 2.调教配置
#### Run_Val
##### 1.日志配置
##### 2.性能配置
### 模块：utils
#### Log
##### ```init(filename=None, level='DEBUG')```
##### ```debug(*kwargs)```
##### ```info(*kwargs)```
##### ```warn(*kwargs)```
##### ```error(*kwargs)```
##### ```critical(*kwargs)```
##### ```memory_used()```
