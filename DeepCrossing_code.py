
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
    # 定义一个embedding层对应的字典
    embedding_layers_dict = dict()
    # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
    # 该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，
    # 然后返回 True 或 False，最后将返回 True 的元素放到新列表中。
    # 将特征中的sparse特征筛选出来
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))\
        if feature_columns else []
    # 如果是用于线性部分的embedding层，其维度为1，
    # 否则维度就是自己定义的embedding维度
    if is_linear:
        # 用于线性部分的维度直接为1
        for fc in sparse_feature_columns:
            # Kreas的嵌入层
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size + 1, 1, name='1d_emb_' + fc.name)
        else:
            for fc in sparse_feature_columns:
                # 不是线性的就是自定义的维度
                embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size + 1,
                                                           fc.embedding_dim, name='kd_emb' + fc.name)

        # 输出嵌入层的字典
        return embedding_layers_dict


# 网络的特征数据拼接
def concat_embedding_list(feature_columns, input_layer_dict, embedding_layer_dict, flatten= False):
    # 将sparse(类别特征筛选出来)
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))

    embedding_list = []
    for fc in sparse_feature_columns:
        # 获取网络所需要的输入
        # 获取输入层
        _input = input_layer_dict[fc.name]
        # B x 1 x dim  获取对应的embedding层
        _embed = embedding_layer_dict[fc.name]
        # B x dim  将input层输入到embedding层中
        embed = _embed(_input)
        # Flatten层用来将输入“压平”，即把多维的输入一维化，
        # 常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
        # 是否需要flatten, 如果embedding列表最终是直接输入到Dense层中，
        # 需要进行Flatten，否则不需要(直接为stacking层则不需要压平)
        if flatten:
            embed = Flatten()(embed)
        # 嵌入层的生成数据
        embedding_list.append(embed)
    # 输出
    return embedding_list







if __name__ == "__main__":
    pass