---
title: "深度推荐模型学习小记（一）"
tag: 
  - DeepLearning
  - python
  - RecommdationModel
  - dataWhale
date: 2021-03-14
---

# 深度推荐模型学习小记（一）

## Task01---DeepCrossing的学习和理解

**简介**

这个模型的第一个真正的把深度学习的架构应用于推荐系统的模型。（2016-微软） 完整的**解决了特征工程、稀疏向量稠密化， 多层神经网络进行优化目标拟合**等一系列深度学习再推荐系统的应用问题。

这个模型的技术比较基础，在传统的神经网络模型加入嵌入（embedding）且结构简单。

模型的初始应用场景：微软搜索引擎Bing中的搜索广告推荐， 用户在输入搜索词之后， 搜索引擎除了返回相关结果， 还返回与搜索词相关的广告，Deep Crossing的优化目标就是预测对于某一广告， 用户是否会点击，依然是点击率预测的一个问题。

### 1.模型的结构及原理

为完成端到端的训练，DC模型需要在网络结构中解决的几个问题：

1. 离散类特征编码（要进行的是one-hot的独热编码）后**过于稀疏**， 不利于直接输入神经网络训练， **需要解决稀疏特征向量稠密化的问题**
2. 如何**解决特征自动交叉组合的问题**
3. 如何在输出层中达成问题设定的优化目标

多层的神经网络结构如下图：

![多层神经网络结构](https://gitee.com/magicye/blogimage/raw/master/img/image-20210314202735654.png)

#### 1.1 Embedding Layer

将稀疏的类别型特征转成稠密的Embedding向量,在深度学习的自然语言处理中，也有类似的操作，将one-hot向量与嵌入矩阵相结合，生成的嵌入向量用于下一步网络的处理。

在DeepCrossing的神经网络的结构中，类别特征(one-hot编码后的稀疏特征向量）是需要进入到嵌入层先进行处理**。数值型特征，不用embedding， 直接到了Stacking Layer。**

关于Embedding Layer的实现， 往往一个全连接层即可，Tensorflow中有实现好的层可以直接用。

#### 1.2 Stacking Layer

该层通常也称为连接层

是把不同的Embedding特征和数值型特征拼接在一起，**形成新的包含全部特征的特征向量。**

具体的实现的思路如下：

先将所有的数值特征拼接起来，然后将所有的Embedding拼接起来，最后将数值特征和Embedding特征拼接起来作为DNN的输入，TF是通过**Concatnate层进行拼接**。

扩展：Concatnate层--深度特征融合

![融合方式](https://gitee.com/magicye/blogimage/raw/master/img/20190308174348375.png)

concatenate是通道数的合并，也就是说描述图像本身的特征增加了，而每一特征下的信息是没有增加。

**Flatten层：**用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。

```python
#将所有的dense特征拼接到一起
dense_dnn_list = list(dense_input_dict.values())
# numpy按照哪一个维度进行连接
dense_dnn_inputs = Concatenate(axis=1)(dense_dnn_list) # B x n (n表示数值特征的数量)

# 因为需要将其与dense特征拼接到一起所以需要Flatten，不进行Flatten的Embedding层输出的维度为：Bx1xdim
sparse_dnn_list = concat_embedding_list(dnn_feature_columns, sparse_input_dict, embedding_layer_dict, flatten=True) 

sparse_dnn_inputs = Concatenate(axis=1)(sparse_dnn_list) # B x m*dim (m表示类别特征的数量，dim表示embedding的维度)

# 将dense特征和Sparse特征拼接到一起
dnn_inputs = Concatenate(axis=1)([dense_dnn_inputs, sparse_dnn_inputs]) # B x (n + m*dim)
```

#### 1.3 Multiple Residual Units Layer

该层的主要结构是MLP(多残差单元层)， 但DeepCrossing采用了残差网络进行的连接**。通过多层残差网络对特征向量各个维度充分的交叉组合**， 使得模型能够抓取更多的非线性特征和组合特征信息， 增加模型的表达能力。残差网络结构如下图所示：

![残差网络结构](https://gitee.com/magicye/blogimage/raw/master/img/image-20210314215203985.png)

Deep Crossing模型**使用稍微修改过的残差单元**，它不使用卷积内核，**改为了两层神经网络。我们可以看到，残差单元是通过两层ReLU变换再将原输入特征相加回来实现的。**具体代码实现如下：

```python
# DNN残差块的定义
class ResidualBlock(Layer):
    def __init__(self, units): # units表示的是DNN隐藏层神经元数量
        super(ResidualBlock, self).__init__()
        self.units = units

    def build(self, input_shape):
        out_dim = input_shape[-1]
        self.dnn1 = Dense(self.units, activation='relu')
        self.dnn2 = Dense(out_dim, activation='relu') # 保证输入的维度和输出的维度一致才能进行残差连接
    def call(self, inputs):
        x = inputs
        x = self.dnn1(x)
        x = self.dnn2(x)
        x = Activation('relu')(x + inputs) # 残差操作
        return x
```

#### 1.4Scoring Layer

这个作为输出层，为了拟合优化目标存在。 对于**CTR预估二分类问题**， Scoring往往采用逻辑回归，模型通过叠加多个残差块加深网络的深度，最后将结果转换成一个概率值输出。

```python
# block_nums表示DNN残差块的数量
def get_dnn_logits(dnn_inputs, block_nums=3):
    dnn_out = dnn_inputs
    for i in range(block_nums):
        dnn_out = ResidualBlock(64)(dnn_out)
    
    # 将dnn的输出转化成logits
    dnn_logits = Dense(1, activation='sigmoid')(dnn_out)

    return dnn_logits
```

**CTR预估**:是对每次广告的点击情况做出预测，预测用户是点击还是不点击。具体定义可以参考 CTR. CTR预估和很多因素相关，比如历史点击率、广告位置、时间、用户等。CTR预估模型就是综合考虑各种因素、特征，在大量历史数据上训练得到的模型。CTR预估的训练样本一般从历史log、离线特征库获得。样本标签相对容易，用户点击标记为1，没有点击标记为0.

### 2.总结

这就是DeepCrossing的结构了，比较清晰和简单，**没有引入特殊的模型结构，只是常规的Embedding+多层神经网络**。但这个网络模型的出现，有革命意义。DeepCrossing模型中没有任何人工特征工程的参与，只需要简单的特征处理，原始特征经Embedding Layer输入神经网络层，自主交叉和学习。 相比于FM，FFM只具备二阶特征交叉能力的模型，**DeepCrossing可以通过调整神经网络的深度进行特征之间的“深度交叉”**，这也是Deep Crossing名称的由来。 

如果是用于点击率预估模型的损失函数就是对数损失函数：

![对数损失函数公式](https://gitee.com/magicye/blogimage/raw/master/img/image-20210314224204189.png)

其中，y是表示的是真实的标签（点击或未点击），p表示Scoring Layer输出的结果。但是在实际应用中，根据不同的需求可以灵活替换为其他目标函数。

### 3.代码的实现解析