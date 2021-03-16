
# DeepCrossing框架
# 去掉操作DataFrame时的警告
import warnings
warnings.filterwarnings("ignore")

# 这个用的是Keras（第一次用这个框架）
from tensorflow import keras
# layers和model的引入
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import pandas as pd
import numpy as np
import tensorflow as tf

# 声明数字特征和类别特征的具体信息
from utils import SparseFeat, VarLenSparseFeat, DenseFeat
# 用于数据的预处理
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 模型的函数的定义
# 数据的预处理函数
def data_process(data_df, dense_features, sparse_features):
    """
        简单处理特征，包括填充缺失值，数值处理，类别编码
        数值特征的缺失值填充使用的是补零
        类别特征的缺失值填充使用的是补充字符串“-1”
        param data_df: DataFrame格式的数据
        param dense_features: 数值特征名称列表
        param sparse_features: 类别特征名称列表
        """
    # 先是数值的缺失值（数值的特征无需进过标签化的转换）
    # 缺失值还是补零（太简单的处理感觉）
    # 每一列的数据列内排序
    data_df[dense_features] = data_df[dense_features].fillna(0.0)
    # 数值特征数据的下标，通过下标来处理数据，pandas的处理，数值的特征数据的log输出
    for f in dense_features:
        data_df[f] = data_df[f].apply(lambda x: np.log(x+1) if x > -1 else -1)

    # 类别特征数据，要过嵌入层（类别数据的缺失值的填充的方式是字符串类型的-1）
    # 缺失值
    data_df[sparse_features] = data_df[sparse_features].fillna("-1")
    # 类别特征数据的下标的分析
    for f in sparse_features:
        # 进一个skleran.encoder
        # 标准化标签，将标签值统一转换成range(标签值个数-1)范围内
        # 也就是将标非数字型标签转化为下标数字的标签
        # 定义一个转换器
        lbe = LabelEncoder()
        # 非数字型标签的标准化
        # 使用的流程，先fit再transform
        # 标准化标签值的反转方法 .inverse_transform()
        data_df[f] = lbe.fit_transform(data_df[f])

    # 最后返回的处理好的DataFrame，用于进一步的嵌入和拼接
    return data_df[dense_features + sparse_features]

# 数据输入层
def build_input_layers(feature_columns):
    """
        构建输入层
        param feature_columns: 数据集中的所有特征对应的特征标记集
    """
    # 构建Input层字典，并以dense和sparse两类字典的形式返回
    # 从而将两类的特征分离，进行进一步的处理
    dense_inputs_dict, sparse_inputs_dict = {}, {}

    for fc in feature_columns:
        # 是类别特征的话，加入类别特征的字典
        if isinstance(fc, SparseFeat):
            # Input():用来实例化一个keras张量
            sparse_inputs_dict[fc.name] = Input(shape=(1,), name=fc.name)
        elif isinstance(fc, DenseFeat):
            dense_inputs_dict[fc.name] = Input(shape=(fc.dimension,), name=fc.name)

    # 分离开输入层的两种特征数据
    return dense_inputs_dict, sparse_inputs_dict


# 嵌入层，类别特征数据将进行一下嵌入化
def build_embedding_layers(feature_columns, input_layers_dict, is_linear):
    pass







if __name__ == "__main__":
    pass