# NFM模型的代码

import warnings
warnings.filterwarnings("igonre")
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import namedtuple

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from utils import SparseFeat, VarLenSparseFeat, DenseFeat


# 还是老样子 训练方法的流程
# 先是数据的预处理
def data_process(data_df, dense_features, sparse_features):
    # 先是数值特征的处理（dense_feature）
    # 先是填补缺失值
    data_df[dense_features] = data_df[dense_features].fillna(0.0)
    for f in dense_features:
        data_df[f] = data_df[f].apply(lambda x: np.log(x+1) if x> -1 else -1)

    # 离散特征的处理（sparse_features）
    data_df[sparse_features] = data_df[sparse_features].fillna("-1")
    # 将类别特征编码为标准化的标签
    for f in sparse_features:
        lbe = LabelEncoder()
        # 编码
        data_df[f] = lbe.fit_transform(data_df[f])

    return data_df[dense_features + sparse_features]

# 构建的网络的输入层
def build_input_layers(feature_columns):
    # 返回是以字典的形式返回的，key对应的是每一个的特征的名字
    dense_input_dict, sparse_input_dict = {}, {}

    # feature_columns里的每一个元素都是namedtuple格式的
    for fc in feature_columns:
        # dense_feature 和s sparse_feature的Input的格式是不一样的
        if isinstance(fc, SparseFeat):
            sparse_input_dict[fc.name] = Input(shape=(1,), name=fc.name)
        elif isinstance(fc, DenseFeat):
            dense_input_dict[fc.name] = Input(shape=(fc.dimension,), name=fc.name)

    return dense_input_dict, sparse_input_dict


# 构建嵌入层，也是分为了linear embedding层和K-dimension的嵌入层
def build_embedding_layers(feature_columns, input_layers_dict, is_linear):
    # 定义一个embedding层对应的字典
    embedding_layers_dict = dict()

    # 筛选出sparse_features
    sparse_feature_columns = list(filter(lambda x: isinstance(x,SparseFeat), feature_columns)) if feature_columns else []
    # 嵌入层的维度分为线性的和K维高阶的两种不同的嵌入向量
    if is_linear:
        for fc in sparse_feature_columns:
            # 线性的就设置为一维的嵌入层
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, 1, name='1d_emb_' + fc.name)
    else:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim, name='kd_emb_'+ fc.name)

    return embedding_layers_dict


# 模型中的两大块功能中的线性特征的学习模块
def get_linear_logits(dense_input_dict, sparse_input_dict, sparse_feature_columns):
    # 数值特征要先进行一次的全连接层之后，再与类别特征进行加和
    concat_dense_inputs = Concatenate(axis=1)(list(dense_input_dict.values()))
    # 全连接层
    dense_logits_output = Dense(1)(concat_dense_inputs)
    # sparse_feature相当过一个one-hot乘上了嵌入矩阵
    # embedding中查询得到的权重就是对应onehot向量中一个位置的权重，所以后面不用再接一个全连接了，本身一维的embedding就相当于全连接
    # 只不过是这里的输入特征只有0和1，所以直接向非零元素对应的权重相加就等同于进行了全连接操作(非零元素部分乘的是1)
    linear_embedding_layers = build_embedding_layers(sparse_feature_columns, sparse_input_dict, is_linear=True)
    # 要加上一个压缩层使得两种feature的维度相同
    sparse_1d_embed = []
    # 将一维的embedding拼接，注意这里需要使用一个Flatten层，使维度对应
    for fc in sparse_feature_columns:
        feat_input = sparse_input_dict[fc.name]
        embed = Flatten()(linear_embedding_layers[fc.name](feat_input))
        sparse_1d_embed.append(embed)

    # 连接所有的sparse_feature的嵌入结果
    sparse_logits_output = Add()(sparse_1d_embed)
    # 最终将dense特征和sparse特征对应的logits相加，得到最终linear的logits
    linear_part = Add()([dense_logits_output, sparse_logits_output])
    return linear_part

# 高维特征学习种最重要的层
# 池化层中的方法
# 特征交叉池化层
# 还是继承的keras.Layer
class BiInteractionPooling(Layer):
    def __init__(self):
        super(BiInteractionPooling, self).__init__()

    # call方法
    def call(self, inputs):
        # 简化的FM公式1/2（和的平方减去平方的和） =>> B x k
        concated_embeds_value = inputs # B x n x k

        # 和的平方
        # B x k
        square_of_sum = tf.square(tf.reduce_sum(concated_embeds_value,axis=1 ,keepdims=False))
        # 平方的和
        # 也是维度变为B x k
        sum_of_square = tf.reduce_sum(concated_embeds_value * concated_embeds_value, axis=1, keepdims=False)
        # B x k
        cross_term = 0.5 * (square_of_sum - sum_of_square)

        return cross_term

    def compute_output_shape(self, input_shape):
        # 因为压缩的是第二维，所以output的维度是第三维的维度
        return (None, input_shape[2])

# 池化层的输出
def get_bi_interaction_pooling_output(sparse_input_dict, sparse_feature_columns, dnn_embedding_layers):
    # 只考虑sparse的二阶交叉，将所有的embedding拼接到一起
    # 这里在实际运行的时候，其实只会将那些非零元素对应的embedding拼接到一起
    # 并且将非零元素对应的embedding拼接到一起本质上相当于已经乘了x, 因为x中的值是1(公式中的x)
    sparse_kd_embed = []
    for fc in sparse_feature_columns:
        feat_input = sparse_input_dict[fc.name]
        # B x 1 x k
        _embed = dnn_embedding_layers[fc.name](feat_input)
        sparse_kd_embed.append(_embed)

    # 将所有sparse的embedding拼接起来，
    # 得到 (n, k)的矩阵，其中n为特征数，k为embedding大小
    # 所有用户的矩阵为B x n x k
    concat_sparse_kd_embed = Concatenate(axis=1)(sparse_kd_embed)

    pooling_out = BiInteractionPooling()(concat_sparse_kd_embed)

    return pooling_out

# dnn_part的输出
# 接收的输入是池化层的输入
def get_dnn_logits(pooling_out):
    # dnn层，这里的Dropout参数，Dense中的参数都可以自己设定, 论文中还说使用了BN,
    # 但是个人觉得BN(Batch Normalization),和dropout同时使用
    # 可能会出现一些问题，感兴趣的可以尝试一些，这里就先不加上了
    # 全连接层多少层，自己决定就行
    dnn_out = Dropout(0.5)(Dense(1024, activation='relu')(pooling_out))
    dnn_out = Dropout(0.3)(Dense(512, activation='relu')(dnn_out))
    dnn_out = Dropout(0.1)(Dense(256, activation="relu")(dnn_out))

    dnn_logits = Dense(1)(dnn_out)

    return dnn_logits

# 模型的流程
def NFM(linear_feature_columns, dnn_feature_columns):
    # 构建输入层，即所有特征对应的Input()层，这里使用字典的形式返回，
    dense_input_dict , sparse_input_dict = build_input_layers(linear_feature_columns + dnn_feature_columns)
    # 把sparse_feaature中的线性部分的特征过滤出来
    linear_sparse_feature_columns  = list(filter(lambda x: isinstance(x, SparseFeat), linear_feature_columns))

    # 构建模型的输入层
    # 实际的输入与Input()层的对应，是通过模型输入时候的字典数据的key与对应name的Input层
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())
    # 输出是两部分的加和（线性的和高阶的）
    # 先是线性的结果
    linear_logits = get_linear_logits(dense_input_dict, sparse_input_dict, linear_sparse_feature_columns)

    # 构建维度为K的隐藏层，用于深度的学习
    embedding_layers = build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear=False)

    # 将输入到dnn中的sparse特征筛选出来
    dnn_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))

    # 先进特征交叉的池化层
    # B x (n(n-1)/2)
    pooling_output = get_bi_interaction_pooling_output(sparse_input_dict, dnn_sparse_feature_columns, embedding_layers)

    # 论文里加上了BN在池化层之后
    pooling_output = BatchNormalization()(pooling_output)

    dnn_logits = get_dnn_logits(pooling_output)

    # 线性和dnn的logits相加成为最后的logits
    output_logits = Add()([linear_logits, dnn_logits])
    # 激活函数使用的还是sigmoid
    output_layers = Activation("sigmoid")(output_logits)

    model = Model(input_layers, output_layers)
    return model


if __name__ == "__main__":
    # 读取数据
    data = pd.read_csv('./data/criteo_sample.txt')
    # 划分一下两类的数据
    columns = data.columns.values
    dense_features = [feat for feat in columns if "I" in feat]
    sparse_features = [feat for feat in columns if "C" in feat]

    # 简单的数据处理
    train_data = data_process(data, dense_features, sparse_features)
    train_data['label'] = data['label']

    # 将特征进行分组
    # 分成linear部分和dnn部分(根据实际场景进行选择)，
    # 并将分组之后的特征做标记（使用DenseFeat, SparseFeat）
    linear_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1) for feat in dense_features]

    # 深度神经网络需要的数据
    dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat], embedding_dim=4)
                           for feat in enumerate(sparse_features)] + [DenseFeat(feat, 1) for feat in dense_features]

    # 构建NFM模型
    history = NFM(linear_feature_columns, dnn_feature_columns)
    history.summary()
    # 检验的条件
    history.compile(optimizer="adam", loss="binary_crossentropy", metrics=["bianry_crossentropy", tf.keras.metrics.AUC(name='auc')])

    # 将输入数据变为字典的形式进行导入
    train_model_input = {name: data[name] for name in dense_features + sparse_features}

    # 模型的训练
    history.fit(train_model_input, train_data['label'].values, batch_size=64, epochs=5, validation_split=0.2)