
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
    # 一部分变为一维的，一部分变为多维的嵌入矩阵
    if is_linear:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, 1, name='1d_emb_' + fc.name)
    else:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim, name='kd_emb_' + fc.name)

    return embedding_layers_dict

# 定义wide的方法（线性的logits）
def get_linear_logits(dense_inputs_dict, sparse_input_dict, sparse_feature_columns):
    # tf.concat()拼接的张量只会改变一个维度，其他维度是保存不变的
    # 先把dense——inputs拼接，进入一个全连接层
    concat_dense_inputs = Concatenate(axis=1)(list(dense_inputs_dict.values()))
    dense_logits_output = Dense(1)(concat_dense_inputs)

    # 获取sparse_feature线性部分的嵌入特征
    # 这里使用embedding的原因是：
    # 对于linear部分直接将特征进行one-hot然后通过一个全连接层，
    # 当维度特别大的时候，计算比较慢
    # 使用embedding层的好处就是可以通过查表的方式获取到哪些非零的元素对应的权重，
    # 然后在将这些权重相加，效率比较高
    # 返回的是字典，以字典的形式进行返回便于使用
    linear_embedding_layers = build_embedding_layers(sparse_feature_columns, sparse_input_dict, is_linear=True)

    # 将一维的embedding向量进行拼接，这里要使用的是一个Flatten层
    # 把多维的输入一维化 使维度对应（以便于进入全连接层）
    sparse_1d_embed = []
    for fc in sparse_feature_columns:
        feat_input = sparse_input_dict[fc.name]
        # 最后的大小（B*1）B应该是batch_size
        embed = Flatten()(linear_embedding_layers[fc.name](feat_input))
        # 添加
        sparse_1d_embed.append(embed)

    #  embedding中查询得到的权重就是对应one-hot向量中一个位置的权重，
    #  所以后面不用再接一个全连接了，本身一维的embedding就相当于全连接
    #  只不过是这里的输入特征只有0和1，
    #  所以直接向非零元素对应的权重相加就等同于进行了全连接操作(非零元素部分乘的是1)
    sparse_logits_output = Add()(sparse_1d_embed)

    # wide部分的最后的加和
    # 最终将dense特征和sparse特征对应的logits相加，得到最终linear的logits
    linear_logits = Add()([dense_logits_output, sparse_logits_output])
    return linear_logits

# deep层中使用的方法 将K维的sparse-feature嵌入向量进行拼接
def concat_embedding_list(feature_columns, input_layer_dict, embedding_layer_dict, flatten=False):
    # 先筛选特征
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))

    embedding_list = []
    for fc in sparse_feature_columns:
        # 输入层(向量是一维的张量)
        _input = input_layer_dict[fc.name]
        # 获取相应的嵌入层（ Batch_size x 1 x dim ）
        _embed = embedding_layer_dict[fc.name]
        # 将input层输入到embedding层中(# 输出看作B x dim)
        embed = _embed(_input)
        # 是否需要flatten,
        # 如果embedding列表最终是直接输入到Dense层中，需要进行Flatten，
        # 否则不需要
        if flatten:
            embed = Flatten()(embed)

        embedding_list.append(embed)
    return embedding_list


# deep模块中的dnn_logits功能
def get_dnn_logit(dense_input_dict, sparse_input_dict, sparse_feature_columns, dnn_embedding_layers):
    # 模型中，dense-feature全连接层后与sparse-feature进行连接和压缩进入DNN
    # (Batch_size * dense_feature_num *dense-feature_dimension)
    concat_dense_inputs = Concatenate(axis=1)(list(dense_input_dict.values()))

    sparse_kd_embed = concat_embedding_list(sparse_feature_columns, sparse_input_dict, dnn_embedding_layers, flatten=True)
    concat_sparse_kd_embed = Concatenate(axis=1)(sparse_kd_embed)
    # 所有嵌入层的拼接
    # B x (n2k + n1)(列的拼接)
    dnn_input = Concatenate(axis=1)([concat_dense_inputs, concat_sparse_kd_embed])
    # dnn的设置（dropout参数也有设置）
    dnn_out = Dropout(0.5)(Dense(1024, activation='relu')(dnn_input))
    # dnn的层数可以自己去定义
    dnn_out = Dropout(0.3)(Dense(512, activation='relu')(dnn_out))
    dnn_out = Dropout(0.1)(Dense(256, activation='relu')(dnn_out))

    dnn_logits = Dense(1)(dnn_out)

    return dnn_logits

# WideNDeep模型的流程
def WideNDeep(linear_feature_columns, dnn_feature_columns):
    # 分为要进两个部分的输入特征（线性的进wdie， 高阶的进dnn部分）
    # 构建输入层，即所有特征对应的Input()层，
    # 这里使用字典的形式返回，方便后续构建模型
    # 第一步，输入层（构建好的是输入层的字典)
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns + dnn_feature_columns)
    # 在线性部分中，把离散的特征（sparse_feature先分离出来）
    # 要先进行嵌入加压缩与linear_dense_feature进行拼接
    linear_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), linear_feature_columns))

    # 模型先是要构建输入层（模型的输入层不能是字典的形式）
    # 应该将字典的形式转换成列表的形式
    # 注意：这里实际的输入与Input()层的对应，
    # 是通过模型输入时候的字典数据的key与对应name的Input层
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())
    # 模型的Wid部分使用的特征比较简单，并且得到的特征非常的稀疏，
    # 所以使用了FTRL优化Wide部分（这里没有实现FTRL）
    # 模型中是将所有的可能用到的特征都输入到Wide部分，具体的细节可以根据需求进行修改
    linear_logits = get_linear_logits(dense_input_dict, sparse_input_dict, linear_sparse_feature_columns)

    # Deep部分，首先就是要构建sparse_feature的embedding层
    # embedding层先定义
    embedding_layers = build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear=False)

    # 还是先分离出进入dnn网络的 sparse_feature
    dnn_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))
    # 在Wide&Deep模型中，
    # deep部分的输入是将dense特征和embedding特征拼在一起输入到dnn中
    dnn_logits = get_dnn_logit(dense_input_dict, sparse_input_dict, dnn_sparse_feature_columns, embedding_layers)

    # 将linear,dnn的logits相加作为最终的logits
    output_logits = Add()([linear_logits, dnn_logits])

    # 最后要输出的是一个几率（激活函数用到的就是sigmoid）
    output_layer = Activation("sigmoid")(output_logits)

    model = Model(input_layers, output_layer)
    return model


# 使用的流程
if __name__ == "__main__":
    # 训练的流程
    # 先是读取和预处理数据
    data = pd.read_csv('./data/criteo_sample.txt')

    # 先划分数值特征和类别特征
    columns = data.columns.values
    dense_features = [feat for feat in columns if 'I' in feat]
    sparse_features = [feat for feat in columns if 'C' in feat]

    # 简单的数据预处理
    train_data = data_process(dense_features, sparse_features)
    train_data['label'] = data['label']

    # 将特征分组，分成linear部分和dnn部分(根据实际场景进行选择)
    # 生成线性和离散的特征的具名元组
    #先是提取线性的特征的具名元组
    linear_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, ) for feat in dense_features]

    # 高阶的特征数据
    dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                           for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,) for feat in dense_features]

    # 模型的构建
    history = WideNDeep(linear_feature_columns, dnn_feature_columns)
    history.summary()
    # 使用函数和监控的对象
    history.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_crossentropy",
                                                                           tf.keras.metrics.AUC(name='auc')])

    # 输入数据的重组
    # 将输入数据转化成字典的形式输入
    train_model_input = {name: data[name] for name in dense_features + sparse_features}
    # 输入数据,开始训练
    history.fit(train_model_input, train_data['label'].values, batch_size=64, epoch=5, validation_split=0.2)
