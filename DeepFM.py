# DeepFMN模型的代码

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import namedtuple

# 深度学习框架的引入
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

# 数据处理的工具
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 具名元组的工具包
from utils import SparseFeat, DenseFeat, VarLenSparseFeat

# 定义的模型中使用的函数
def data_process(data_df, dense_features, sparse_features):
    # 先是把缺失值处理一下
    data_df[dense_features] = data_df[dense_features].fillna(-1)
    # 数值特征先用log限定一个范围
    for f in dense_features:
        data_df[f] = data_df[f].apply(lambda x: np.log(x+1) if x > -1 else -1)

    data_df[sparse_features] = data_df[sparse_features].fillna("-1")
    # 类别特征的编码（先编码再赋值）
    for f in sparse_features:
        lbe = LabelEncoder()
        data_df[f] = lbe.transform(data_df[f])

    return data_df[dense_features + sparse_features]

# 构建输入层
def build_input_layers(feature_columns):
    # 构建输入层的字典，每个特征的名称就是一个key
    # 以dense和sparse两类字典的形式返回
    dense_input_dict, sparse_input_dict = {}, {}

    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            # 定义这个类别特征的输入层
            sparse_input_dict[fc.name] = Input(shape=(1,), name=fc.name)
        elif isinstance(fc, DenseFeat):
            dense_input_dict[fc.name] = Input(shape=(fc.dimension, ), name=fc.name)

    return dense_input_dict, sparse_input_dict

# 定义嵌入层（用于处理Sparse_feature的）
def build_embedding_layers(feature_columns, input_layers_dict, is_linear):
    # is_linear是用于有的sparse_feature要嵌入成一维，用于get_linear_logits
    # 嵌入层的输出也是使用的相应的字典
    embedding_layers_dict = {}
    # 先将特征中的sparse_feature筛选出来
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    # 用于线性结果的提取的，嵌入的维度将为1
    # 用于dnn层结果的训练的，嵌入到维度就是自己定义的维度
    # 先是线性的
    if is_linear:
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, 1, name='1d_emb_' + fc.name)
    else:
        # 进入深度网络的嵌入
        for fc in sparse_feature_columns:
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size, fc.embedding_dim, name="kd_emb_" + fc.name)
    # 最后返回的是嵌入层结果的字典
    return embedding_layers_dict

# 线性特征的学习
def get_linear_logits(dense_input_dict, sparse_input_dict, sparse_feature_columns):
    # 将所有的dense特征的Input层，然后经过一个全连接层得到dense特征的logits
    # 先连接
    concat_dense_inputs = Concatenate(axis=1)(list(dense_input_dict.values()))
    # 过全连接层
    dense_logits_output = Dense(1)(concat_dense_inputs)

    # sparse_feature的线性特征的计算
    # 获取linear部分sparse特征的embedding层，这里使用embedding的原因是：
    # 对于linear部分直接将特征进行onehot然后通过一个全连接层，当维度特别大的时候，计算比较慢
    # 使用embedding层的好处就是可以通过查表的方式获取到哪些非零的元素对应的权重，
    # 然后在将这些权重相加，效率比较高
    linear_embedding_layers = build_embedding_layers(sparse_feature_columns, sparse_input_dict, is_linear=True)
    # 一维的类别特征的嵌入向量拼接然后和数值特征进行加和
    sparse_1d_embed = []
    for fc in sparse_feature_columns:
        # 输入层
        feat_input = sparse_input_dict[fc.name]
        # 嵌入和压缩
        embed = Flatten()(linear_embedding_layers[fc.name](feat_input))# B x 1(B应该是数据的条数)
        sparse_1d_embed.append(embed)
    # embedding中查询得到的权重就是对应one-hot向量中一个位置的权重，
    # 所以后面不用再接一个全连接了，本身一维的embedding就相当于全连接
    # 嵌入向量就相当于one-hot向量乘上了嵌入矩阵（就相当于过了一层全连接）
    sparse_logits_output = Add()(sparse_1d_embed)
    # 最终将dense特征和sparse特征对应的logits相加，
    # 得到最终linear的logits
    linear_logits = Add()([dense_logits_output, sparse_logits_output])
    return linear_logits

# DeepFM特有的FM层
# 继承的是Keras的Layer
# 自定义的神经网络层
# 进入的是自定义维度的嵌入层的输出
class FM_Layer(Layer):
    def __init__(self):
        super(FM_Layer, self).__init__()

    def call(self, inputs):
        # 优化后的公式为： 0.5 * 求和（和的平方-平方的和）
        # > B x 1
        concated_embeds_value = inputs # B x n x k

        square_of_sum = tf.square(tf.reduce_sum(concated_embeds_value, axis=1, keepdims=True))# B x 1 x k
        # B x1 xk
        sum_of_square = tf.reduce_sum(concated_embeds_value * concated_embeds_value, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square# B x 1 x k
        # B x 1 (压缩下)
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)# B x 1

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)

# FM_logits
# 嵌入矩阵进入FM-layer输出的结果
def get_fm_logits(sparse_input_dict, sparse_feature_columns, dnn_embedding_layers):
    # 将特征中的sparse特征筛选出来
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), sparse_feature_columns))
    # 只考虑sparse的二阶交叉，将所有的embedding拼接到一起进行FM计算
    # 因为类别型数据输入的只有0和1所以不需要考虑将隐向量与x相乘，直接对隐向量进行操作即可
    sparse_kd_embed = []
    for fc in sparse_feature_columns:
        # 输入层
        feat_input = sparse_input_dict[fc.name]
        # 嵌入层
        _embed = dnn_embedding_layers[fc.name](feat_input)# B x 1 x k
        sparse_kd_embed.append(_embed)

    # 将所有sparse的embedding拼接起来，得到 (n, k)的矩阵，其中n为特征数，k为embedding大小
    # B x n x k
    concat_sparse_kd_embed = Concatenate(axis=1)(sparse_kd_embed)
    # 进入自定义的FM层
    fm_cross_out = FM_Layer()(concat_sparse_kd_embed)

    return fm_cross_out

# 神经网络的logits
def get_dnn_logits(sparse_input_dict, sparse_feature_columns, dnn_embedding_layers):
    # 将特征中的sparse特征筛选出来
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), sparse_feature_columns))

    # 将所有非零的sparse特征对应的embedding拼接到一起
    sparse_kd_embed = []
    for fc in sparse_feature_columns:
        feat_input = sparse_input_dict[fc.name]
        # 嵌入
        # B x 1 x k
        _embed = dnn_embedding_layers[fc.name](feat_input)
        # 压缩进入全连接层
        # B x k
        _embed = Flatten()(_embed)
        sparse_kd_embed.append(_embed)

    # 压缩完进行连接
    # B x nk   (n为sparse_feature的数量)
    concat_sparse_kd_embed = Concatenate(axis=1)(sparse_kd_embed)

    # 进入深度视神经网络层
    # dnn层，这里的Dropout参数，Dense中的参数都可以自己设定，以及Dense的层数都可以自行设定
    mlp_out = Dropout(0.5)(Dense(256, activation='relu')(concat_sparse_kd_embed))
    mlp_out = Dropout(0.3)(Dense(256, activation='relu')(mlp_out))
    mlp_out = Dropout(0.1)(Dense(256, activation='relu')(mlp_out))

    dnn_out = Dense(1)(mlp_out)
    return dnn_out

# 总的流程
def DeepFM(linear_feature_columns, dnn_feature_columns):
    # 构建输入层，即所有特征对应的Input()层，这里使用字典的形式返回，方便后续构建模型
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns + dnn_feature_columns)

    # 将linear部分的特征中sparse特征筛选出来，后面用来做1维的embedding
    linear_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), linear_feature_columns))

    # 构建模型的输入层，模型的输入层不能是字典的形式，应该将字典的形式转换成列表的形式
    # 注意：这里实际的输入与Input()层的对应，是通过模型输入时候的字典数据的key与对应name的Input层
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())
    # 先是线性特征的计算
    linear_logits = get_linear_logits(dense_input_dict, sparse_input_dict, linear_sparse_feature_columns)

    #构建自定义维度为k的嵌入层，使用嵌入层来进行FMlayer和Dnn-logits的计算
    # 先构建嵌入层的定义
    embedding_layers = build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear=False)
    # 将要输入到嵌入层的sparse_feature筛选出来进入嵌入层
    dnn_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))
    # 进入Fm_layer的计算
    fm_logits = get_fm_logits(sparse_input_dict, dnn_sparse_feature_columns, embedding_layers)

    # 接下来是深度神经网络层的学习
    # 将所有的Embedding都拼起来，一起输入到dnn中
    dnn_logits = get_dnn_logits(sparse_input_dict, dnn_sparse_feature_columns, embedding_layers)

    # 线性，FM，dnn的输出相加作为最终的logits
    output_logits = Add()([linear_logits, fm_logits, dnn_logits])
    # 这里的激活函数使用sigmoid
    output_layers = Activation("sigmoid")(output_logits)

    # 建立模型
    model = Model(input_layers, output_layers)
    return model


# 实验的主函数
if __name__ == "__main__":
    # 先是读取数据
    data = pd.read_csv('./data/criteo_sample.txt')

    # 区分出数值特征和类别特征
    columns = data.columns.values
    dense_features = [feat for feat in columns if "I" in feat]
    sparse_features = [feat for feat in columns if "C" in feat]

    # 数据的预处理
    train_data = data_process(data, dense_features, sparse_features)
    # 训练数据的结果
    train_data['label'] = data['label']

    # 将每个块需要的特征进行分组
    # 分成linear部分和dnn部分(根据实际场景进行选择)，
    # 并将分组之后的特征做标记（使用DenseFeat, SparseFeat）
    # 线性的特征
    linear_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, ) for feat in dense_features]
    # 深层的特征
    dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                           for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, ) for feat in dense_features]

    #构建模型，开始训练
    history = DeepFM(linear_feature_columns, dnn_feature_columns)
    history.summary()
    # 需要监测的性能指标
    history.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])
    # 将输入数据转化成字典的形式输入
    train_model_input = {name: data[name] for name in dense_features + sparse_features}

    # 开始训练
    history.fit(train_model_input, train_data['label'].values, batch_size=64, epoch=5, validation_split=0.2)






