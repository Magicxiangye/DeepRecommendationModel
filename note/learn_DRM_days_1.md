---
title: "深度推荐模型学习小记（二）"
tag: 
  - DeepLearning
  - python
  - RecommendationModel
  - dataWhale
date: 2021-03-17
---

# 深度推荐模型学习小记（二）

## Task02---Wide&Deep的学习和理解

**简介**

原文文献：[1] Cheng, Heng-Tze, Koc, Levent, Harmsen, Jeremiah,等. Wide & Deep Learning for Recommender Systems[J]. 2016.

FTRL算法paper:[2] H. Brendan McMahan, Gary Holt, D. Sculley, Michael Young, Dietmar Ebner, Julian Grady, Lan Nie, Todd Phillips, Eugene Davydov, Daniel Golovin, Sharat Chikkerur, Dan Liu, Martin Wattenberg, Arnar Mar Hrafnkelsson, Tom Boulos, Jeremy Kubica, Ad Click Prediction: a View from the Trenches, Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD) (2013)

在以往的CTR（点击通过率）预估任务中利用手工构造的交叉组合特征来使线性模型具有“记忆性”，**使模型记住共现频率较高的特征组合**，往往也能达到一个不错的baseline，且可解释性强。但这种方式有着较为明显的缺点：

1. 特征工程需要耗费太多精力。
2. **模型是强行记住这些组合特征的**，**对于未曾出现过的特征组合，**权重系数为0，**无法进行泛化。**

为加强泛化的能力，研究人员引入了DNN结构，使用嵌入层将高维度的稀疏特征编码变为低维度稠密的Embedding Vector（嵌入向量）但是，**基于Embedding的方式可能因为数据长尾分布**，导致长尾的一些特征值无法被充分学习，其对应的Embedding vector是不准确的，**这便会造成模型泛化过度。**

PS:**数据的长尾化分布**：在自然情况下，数据往往都会呈现如下相同的长尾分布。这种趋势同样出现在从自然科学到社会科学的各个领域各个问题中，参考Zipf's Law或者我们常说的28定律。直接利用长尾数据来训练的分类和识别系统，往往会对头部数据过拟合，从而在预测时忽略尾部的类别。

![长尾分布](https://gitee.com/magicye/blogimage/raw/master/img/v2-3c2009cd25376e7bd63b40cee7aa3de6_720w.jpg)

Wide&Deep模型：就是围绕记忆性和泛化性进行讨论的，**模型能够从历史数据中学习到高频共现的特征组合的能力，称为是模型的Memorization。**能够利用特征之间的传递性去探索历史数据中从未出现过的特征组合。这也被称为是模型的Generalization。

Wide&Deep兼顾Memorization与Generalization，实践表明wide&deep框架显著提高了移动app store 的app下载率，同时满足了训练和服务的速度要求。

### 一、模型的结构原理分析

![模型的结构](https://pic4.zhimg.com/v2-509773d865632c1183f339833c585fd3_r.jpg)

wide&deep模型本身的结构是简单，但是如何根据自己的场景去**选择那些特征放在Wide部分，哪些特征放在Deep部分**(这个模型的重要部分)就需要理解这篇论文提出者当时对于设计该模型不同结构时的意图了，所以这也是用好这个模型的一个前提。

**如何理解Wide部分有利于增强模型的“记忆能力”，Deep部分有利于增强模型的“泛化能力”？**

- **wide部分**是![[公式]](https://www.zhihu.com/equation?tex=y%3Dw%5ETx%2Bb)，一个广义的线性模型，x= [x1, x2,x3,...xd]是特征向量，输入的特征向量主要有两部分组成，**一部分是原始的部分特征，另一部分是原始特征的交叉特征**(cross-product transformation)，最重要的转换之一是交叉乘积转换，对于交互特征可以定义为：

  ![交互特征公式](https://gitee.com/magicye/blogimage/raw/master/img/image-20210317173706369.png)

公式中的参数的定义：

**C_ki**：是一个布尔变量，当**第i个特征属于第k个特征组合时**，C_ki的值为1，否则为0

**X_i:**是第i个特征的值

**只有是符合特征组合的特征，才会进入这个等式。**相当于对于二值特征，是两个特征都同时为1这个新的特征才能为1，否则就是0，说白了就是一个特征组合。

对于wide部分训练时候使用的优化器是带L1正则的FTRL算法(Follow-the-regularized-leader),FTRL 算法综合考虑了 FOBOS 前向后向切分（FOBOS，Forward Backward Splitting）和 RDARDA（Regularized Dual Averaging Algorithm）叫做正则对偶平均算法，对于梯度和正则项的优势和不足。L1 FTLR是非常注重模型稀疏性质的，也就是说W&D模型采用L1 FTRL**是想让Wide部分变得更加的稀疏**，即Wide部分的大部分参数都为0，这就大大压缩了模型权重及特征向量的维度。**Wide部分模型训练完之后留下来的特征都是非常重要的，那么模型的“记忆能力”就可以理解为发现"直接的"，“暴力的”，“显然的”关联规则的能力。**

![算法伪代码](http://images.cnitblog.com/i/417893/201406/262049050552702.jpg)

- Deep部分是一个DNN模型，输入的特征主要分为两大类，**一类是数值特征(可直接输入DNN)，一类是类别特征(需要经过Embedding之后才能输入到DNN中)**，Deep部分的数学形式如下

![简单的神经网络隐藏层](https://gitee.com/magicye/blogimage/raw/master/img/image-20210317202742177.png)

对于Deep部分的DNN模型作者使用了深度学习常用的**优化器AdaGrad**

**Wide部分与Deep部分的结合**

W&D模型是**将两部分输出的结果结合起来联合训练**，将deep和wide部分的输出**重新使用一个逻辑回归模型做最终的预测**，输出概率值。联合训练的数学形式如下：需要注意的是，因为Wide侧的数据是高维稀疏的，所以作者使用了FTRL算法优化，而Deep侧使用的是 Adagrad。

![分类概率的公式](https://gitee.com/magicye/blogimage/raw/master/img/image-20210317203236369.png)

其中，Y是二值分类标签， ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%28.%29) 是sigmoid函数， ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi%28x%29) 是原始特征x的跨产品变换，b是偏置项， ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bwide%7D) 是wide模型的权重向量， ![[公式]](https://www.zhihu.com/equation?tex=w_%7Bdeep%7D) 是用于最终激活函数 ![[公式]](https://www.zhihu.com/equation?tex=a%5E%7B%28l_f%29%7D) 的权重。

### 二、代码实现

Wide侧记住的是历史数据中那些**常见、高频**的模式，但在实际上Wide侧没有发现新的模式，只是学习到这些模式之间的权重，做一些模式的筛选。正因为Wide侧不能发现新模式，因此我们需要**根据人工经验、业务背景，将我们认为有价值的、显而易见的特征及特征组合，喂入Wide侧。**

**Deep侧就是DNN**，通过embedding的方式将categorical/id特征映射成稠密向量，让DNN学习到这些特征之间的**深层交叉**，以增强扩展能力。

模型的结构是由deep和wide两个部分组成的。在wide部分，加入了有可能的一阶特征，包括数值特征和类别特征的one-hot都加进去了。**只要能够发现高频、常见模式的特征都可以放在wide侧**，对于Deep部分，在本数据中放入了数值特征和类别特征的embedding特征，实际应用也需要根据需求进行选择。

模型的结构流程

![流程结构](https://gitee.com/magicye/blogimage/raw/master/img/%E5%9B%BE%E7%89%87image-20210228160557072.png)

```python
# Wide&Deep 模型的wide部分及Deep部分的特征选择，应该根据实际的业务场景去确定哪些特征应该放在Wide部分，哪些特征应该放在Deep部分

# data_input_layer(), input_layer() embedding_layer()的方法和上一任务的函数一样。
def WideNDeep(linear_feature_columns, dnn_feature_columns):
    # 构建输入层，即所有特征对应的Input()层，这里使用字典的形式返回，分别生成wide和deep所需要的input_dict
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns + dnn_feature_columns)

    # 将需要进入linear部分的，在sparse特征里筛选出来，后面用来做1维的embedding
    linear_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), linear_feature_columns))

    # 构建模型的输入层，模型的输入层不能是字典的形式，应该将字典的形式转换成列表的形式
    # 注意：这里实际的输入与Input()层的对应，是通过模型输入时候的字典数据的key与对应name的Input层
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())

    # Wide&Deep模型论文中Wide部分使用的特征比较简单，并且得到的特征非常的稀疏，所以使用了FTRL优化Wide部分（这里没有实现FTRL）
    # 但是是根据他们业务进行选择的，我们这里将所有可能用到的特征都输入到Wide部分，具体的细节可以根据需求进行修改
    linear_logits = get_linear_logits(dense_input_dict, sparse_input_dict, linear_sparse_feature_columns)
    
    # 构建维度为k的embedding层，这里使用字典的形式返回，方便后面搭建模型
    embedding_layers = build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear=False)

    dnn_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))

    # 在Wide&Deep模型中，deep部分的输入是将dense特征和embedding特征拼在一起输入到dnn中
    dnn_logits = get_dnn_logits(dense_input_dict, sparse_input_dict, dnn_sparse_feature_columns, embedding_layers)
    
    # 将linear,dnn的logits相加作为最终的logits
    output_logits = Add()([linear_logits, dnn_logits])

    # 这里的激活函数使用sigmoid
    output_layer = Activation("sigmoid")(output_logits)

    model = Model(input_layers, output_layer)
    return model
```

