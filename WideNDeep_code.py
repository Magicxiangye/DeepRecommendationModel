
# Wide and Deep model
import warnings
warnings.filterwarnings("ignore")
# 进度条
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 声明数字特征和类别特征的具体信息(具名元组)
from utils import SparseFeat, VarLenSparseFeat, DenseFeat

# 数据的预处理
def data_process(data_df, dense_features, sparse_features):
    """
            简单处理特征，包括填充缺失值，数值处理，类别编码
            数值特征的缺失值填充使用的是补零
            类别特征的缺失值填充使用的是补充字符串“-1”
            param data_df: DataFrame格式的数据
            param dense_features: 数值特征名称列表
            param sparse_features: 类别特征名称列表
            """
    # 先是dense数值数据
    data_df[dense_features] = data_df[dense_features].fillna(0.0)
    for f in dense_features:
        data_df[f] = data_df[f].apply(lambda x: np.log(x+1) if x > -1 else -1)

    # 类别数据的填充和编码标签方法
    data_df[sparse_features] = data_df[sparse_features].fillna("-1")
    for f in sparse_features:
        # 变为数字标签的编码
        encode = LabelEncoder()
        data_df[f] = encode.fit_transform(data_df[f])

    # return 处理好的DataFrame
    return data_df[dense_features + sparse_features]

# 生成输入层的函数
def build_input_layers(feature_columns):
    # 还是字典的形式返回
    dense_input_dict, sparse_input_dict = {}, {}

    # 使用具名元组依次定义
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            # 输入层张量的shape在具名元组都有定义
            sparse_input_dict[fc.name] = Input(shape=(1,), name=fc.name)
        elif isinstance(fc, DenseFeat):
            dense_input_dict[fc.name] = Input(shape=(fc.dimension,), name=fc.name)
    return dense_input_dict, sparse_input_dict

# 定义每个特征嵌入时使用的字典
def build_embedding_layers(feature_columns, input_layers_dict, is_linear):
    # 定义一个存储embedding层的字典
    embedding_layers_dict = dict()

    # 筛选出类别特征，生成嵌入层
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) \
        if feature_columns else []
    # 如果是用于线性部分的embedding层，其维度为1
    # 否则维度就是自己定义的embedding维度
    if is_linear:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, 1, name='1d_emb_' + fc.name)
    else:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim, name='kd_emb_' + fc.name)

    return embedding_layers_dict

# 定义wide的方法



if __name__ == "__main__":
    pass
