
# DeepCrossing框架
# 去掉操作DataFrame时的警告
import warnings
warnings.filterwarnings("ignore")

# 进度条的显示
import itertools
from tqdm import tqdm

# 具名以元组
from collections import namedtuple

# 这个用的是Keras（第一次用这个框架）(写的有点乱)
from tensorflow import keras
# layers和model的引入
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import pandas as pd
import numpy as np
import tensorflow as tf

# 声明数字特征和类别特征的具体信息
from utils import SparseFeat, VarLenSparseFeat, DenseFeat
# 用于数据的预处理
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 数据集的分割方式
from sklearn.model_selection import train_test_split
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
# Keras:Input()函数
# 作用：初始化深度学习网络输入层的tensor
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
            # 每一个的tensor的name是独一无二的
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
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    # 如果是用于线性部分的embedding层，其维度为1，
    # 否则维度就是自己定义的embedding维度
    if is_linear:
        # 用于线性部分的维度直接为1
        for fc in sparse_feature_columns:
            # Kreas的嵌入层
            # Embedding层的前两个参数是词汇表大小和词向量的维度
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size + 1, 1, name='1d_emb_' + fc.name)
    else:
        for fc in sparse_feature_columns:
            # 不是线性的就是自定义的维度
            embedding_layers_dict[fc.name] = Embedding(fc.vocabulary_size + 1, fc.embedding_dim, name='kd_emb_' + fc.name)

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

# DNN残差块的定义
# 通过继承Layer来实现自己的自定义层。
class ResidualBlock(Layer):
    # units表示的是DNN隐藏层神经元数量
    def __init__(self, units):
        # 自己变为自己的父类，调用自己的初始化方法
        super(ResidualBlock, self).__init__()
        self.units = units

    # DNN残差块的内部结构
    # 使用的是多层的全连接的网络
    # 定义权重的方法
    def build(self, input_shape):
        # 输出数据的维度
        out_dim = input_shape[-1]
        # 两层的神经网络的结构
        # 使用的是tensorflow.keras.layer.Dense
        # 它是Keras定义网络层的基本方法，(具体的参数用到的去看官方的文档)
        self.dnn1 = Dense(self.units, activation='relu')
        # 保证输入的维度和输出的维度一致才能进行残差连接
        self.dnn2 = Dense(out_dim, activation='relu')

    # 残差块的使用结构流程
    # call(x)是定义层功能的
    # 定义了具体的计算过程,x为输入值
    # 除非你希望你写的层支持masking，
    # 否则你只需要关心call的第一个参数：输入张量。
    # #注意：如果输出形状不变，则不需要；
    # 如果输出的形状发生改变，一定要return 出最后的shape
    # def compute_output_shape(self, input_shape):
    #     return (input_shape[0], 165, 165)
    def call(self, inputs):
        x = inputs
        x = self.dnn1(x)
        x = self.dnn2(x)
        # 残差的连接操作
        x = Activation('relu')(x + inputs)
        return x

# 最后的Scoring层
# 使用的是逻辑回归的分类方法
# block_nums表示DNN残差块的数量
def get_dnn_logits(dnn_inputs, block_nums=3):
    dnn_out = dnn_inputs
    for i in range(block_nums):
        dnn_out = ResidualBlock(64)(dnn_out)
    # 将dnn的输出转化为logits
    dnn_logits = Dense(1, activation='sigmoid')(dnn_out)

    return dnn_logits

# 网络的具体构架
def DeepCrossing(dnn_feature_columns):
    # 构建输入层，即所有特征对应的Input()层，这里使用字典的形式返回，
    # 方便后续构建模型
    dense_input_dict, sparse_input_dict = build_input_layers(dnn_feature_columns)
    # 构建模型的输入层，模型的输入层不能是字典的形式，
    # 应该将字典的形式转换成列表的形式
    # 注意：这里实际的输入与Input()层的对应，
    # 是通过模型输入时候的字典数据的key与对应name的Input层
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())
    # 构建维度为k的embedding层(K为数据集中类别特征的个数)，这里使用字典的形式返回，
    # 方便后面搭建模型
    embedding_layer_dict = build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear=False)
    # 先将dense特征拼接在一起
    dense_dnn_list = list(dense_input_dict.values())
    # axis拼接的维度
    # B x n (n表示数值特征的数量,B为多少条数据)
    dense_dnn_inputs = Concatenate(axis=1)(dense_dnn_list)
    # 因为需要将其与dense特征拼接到一起所以需要Flatten
    # 不进行Flatten的Embedding层输出的维度为：B x 1 x dim
    sparse_dnn_list = concat_embedding_list(dnn_feature_columns, sparse_input_dict, embedding_layer_dict, flatten=True)
    # B x m*dim (n表示类别特征的数量，dim表示embedding的维度)
    sparse_dnn_inputs = Concatenate(axis=1)(sparse_dnn_list)

    # 将dense和sparse数据最后拼接
    # 最后的维度  B x (n + m*dim)
    dnn_inputs = Concatenate(axis=1)([dense_dnn_inputs, sparse_dnn_inputs])
    # 输入到dnn中，需要提前定义要的几个残差块
    output_layer = get_dnn_logits(dnn_inputs, block_nums=3)

    # 定义模型
    model = Model(input_layers, output_layer)
    return model

# 使用的操作
if __name__ == "__main__":
    # 读取数据
    data = pd.read_csv('./data/criteo_sample.txt')

    # 划分dense和sparse特征
    columns = data.columns.values
    # I为数值特征
    # C为类别特征
    dense_features = [feat for feat in columns if 'I' in feat]
    sparse_features = [feat for feat in columns if 'C' in feat]

    # 简单的数据预处理
    # 补缺失值，以及标签化类别特征的值
    train_data = data_process(data, dense_features, sparse_features)
    # 最后的标签的定义还是要传输的
    # Python的浅拷贝子对象还是指向统一对象
    # data is train_data    >>>True
    # 深度拷贝需要引入copy模块   b = copy.deepcopy(a)---深度拷贝的这两个是完全独立的
    train_data['label'] = data['label']

    # 将特征做标记
    # 具名元组可以像类一样定义
    # vocabulary_size是每列特征的唯一值的统计量
    dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                           for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                           for feat in dense_features]

    # 构建DeepCrossing模型
    # 返回的是model
    history = DeepCrossing(dnn_feature_columns)

    # model.summary()输出模型各层的参数状况
    history.summary()
    # 模型优化的定义，损失函数，优化器的选择
    # metrics: 评价函数,与损失函数类似,只不过评价函数的结果不会用于训练过程中,可以传递已有的评价函数名称,或者传递一个自定义的
    history.compile(optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["binary_crossentropy", tf.keras.metrics.AUC(name='auc')])

    # 将输入数据转化成字典的形式输入
    train_model_input = {name: data[name] for name in dense_features + sparse_features}
    # 模型训练
    # keras.model.fit()函数
    history.fit(train_model_input, train_data['label'].values,
                batch_size=64, epochs=5, validation_split=0.2, )