---
title: "深度推荐模型学习小记（三）"
tag: 
  - DeepLearning
  - python
  - RecommendationModel
  - dataWhale
date: 2021-03-20
---

# 深度推荐模型学习小记（三）

## Task03---DeepFM的学习和理解

### **1.简介**

对于CTR问题，被证明的最有效的提升任务表现的策略是特征组合(Feature Interaction)。一个模型，能够更好的学习特征组合，更加精确的描述数据的特点。是一个好的推荐系统模型的一个重要的思想。

前人研究过组合二阶特征，三阶甚至更高阶，但是面临一个问题就是随着阶数的提升，复杂度就成几何倍的升高。这样即使模型的表现更好了，但是推荐系统在实时性的要求也不能满足了。这也就引出了一个关键的问题：如何更高效的学习特征组合？

为了解决上述问题，出现了FM和FFM来优化Logistic Regression的特征组合较差这一个问题。

知识补充：**FM**（Factorization Machines，因子分解机）它是一种通用的预测方法，在即使数据非常稀疏的情况下，依然能估计出可靠的参数进行预测。**FFM**(Field Factorization Machine)是在FM的基础上引入了“场（Field）”的概念而形成的新模型。在FM中计算特征 ![[公式]](https://www.zhihu.com/equation?tex=x_i) 与其他特征的交叉影响时，使用的都是同一个隐向量 ![[公式]](https://www.zhihu.com/equation?tex=V_i) 。而FFM将特征按照事先的规则分为多个场(Field)，特征 ![[公式]](https://www.zhihu.com/equation?tex=x_i) 属于某个特定的场f。每个特征将被映射为多个隐向量 ![[公式]](https://www.zhihu.com/equation?tex=V_%7Bi1%7D%2C%E2%80%A6%2CV_%7Bif%7D) ，每个隐向量对应一个场。

并且在这个时候科学家们已经发现了DNN在特征组合方面的优势，所以又出现了FNN和PNN等使用深度网络的模型。但与此同时DNN也存在局限性。

- DNN的局限：

当我们使用DNN网络解决推荐问题的时候**存在网络参数过于庞大的问题**，这是因为**在进行特征处理的时候我们需要使用one-hot编码来处理离散特征**，这会导致输入的维度猛增。

![网络参数](https://gitee.com/magicye/blogimage/raw/master/img/image-20210320203950321.png)

这样庞大的参数量也是不实际的。为了解决DNN参数量过大的局限性，可以采用非常经典的Field思想，将One-Hot特征转换为Dense Vector

![dense层的分治](https://gitee.com/magicye/blogimage/raw/master/img/image-20210320204225642.png)

此时通过增加全连接层就可以实现高阶的特征组合

![全连接层的组合](https://gitee.com/magicye/blogimage/raw/master/img/image-20210320204429434.png)

- 早期的方法：

在DeepFM之前有FNN，虽然在影响力上可能并不如DeepFM，但是了解FNN的思想对我们理解DeepFM的特点和优点是很有帮助的。![FNN结构](https://gitee.com/magicye/blogimage/raw/master/img/%E5%9B%BE%E7%89%872021-02-22-10-12-19.png)

FNN是使用预训练好的FM模块，得到隐向量，然后把隐向量作为DNN的输入，但是经过实验进一步发现，在Embedding layer和hidden layer1之间增加一个product层（如上图所示）可以提高模型的表现，所以提出了PNN，使用product layer替换FM预训练层。

- Wide&Deep

FNN和PNN模型仍然有一个比较明显的尚未解决的缺点：**对于低阶组合特征学习到的比较少，这一点主要是由于FM和DNN的串行方式导致的，**也就是虽然FM学到了低阶特征组合，但是DNN的全连接结构导致低阶特征并不能在DNN的输出端较好的表现。看来我们已经找到问题了，**将串行方式改进为并行方式能比较好的解决这个问题。于是Google提出了Wide&Deep模型**，但是如果深入探究Wide&Deep的构成方式，虽然将整个模型的结构调整为了并行结构，在实际的使用中**Wide Module中的部分需要较为精巧的特征工程，换句话说人工处理对于模型的效果具有比较大的影响**（这一点可以在Wide&Deep模型部分得到验证）。

<img src="http://ryluo.oss-cn-chengdu.aliyuncs.com/Javaimage-20200910214310877.png" alt="image-20200910214310877" style="zoom:65%;" />

如上图所示，该模型仍然存在问题：**在output Units阶段直接将低阶和高阶特征进行组合，很容易让模型最终偏向学习到低阶或者高阶的特征，而不能做到很好的结合。**

### 2.模型的结构和原理

![模型结构](https://gitee.com/magicye/blogimage/raw/master/img/%E5%9B%BE%E7%89%87image-20210225180556628.png)

前面的Field和Embedding处理是和前面的方法是相同的，如上图中的绿色部分；DeepFM将Wide部分替换为了FM layer如上图中的蓝色部分。

#### 2.1FM

下图是FM的一个结构图，从图中大致可以看出FM Layer是**由一阶特征和二阶特征Concatenate到一起在经过一个Sigmoid得到logits分类**（结合FM的公式一起看），所以在实现的时候需要单独考虑linear部分和FM交叉特征部分。

![FM公式](https://gitee.com/magicye/blogimage/raw/master/img/image-20210320211741339.png)

![FM结构](https://gitee.com/magicye/blogimage/raw/master/img/%E5%9B%BE%E7%89%87image-20210225181340313.png)

#### 2.2Deep

Deep架构图

![Deep层](https://gitee.com/magicye/blogimage/raw/master/img/%E5%9B%BE%E7%89%87image-20210225181010107.png)

Deep Module是为了学习高阶的特征组合，在上图中使用用全连接的方式将Dense Embedding输入到Hidden Layer，**这里面Dense Embeddings就是为了解决DNN中的参数爆炸问题**，这也是推荐模型中常用的处理方法。

Dense Embedding层的公式分析：

Embedding层的输出是将所有id类特征对应的embedding向量连接到一起，再输入到DNN中。其中V_i表示第i个field的embedding，m是field的数量。

![公式2](https://gitee.com/magicye/blogimage/raw/master/img/image-20210320214356408.png)

连接的向量，作为隐藏层的输入，进入网络的全连接层，全连接层的计算方法：

![公式3](https://gitee.com/magicye/blogimage/raw/master/img/image-20210320214631300.png)

就是权重，偏置，和激活函数的使用。

最后从隐藏层到输出层，要使用sigmod激活函数进行激活：

![公式3](https://gitee.com/magicye/blogimage/raw/master/img/image-20210320214827356.png)

### 3.代码分析

![代码流程设计图](https://gitee.com/magicye/blogimage/raw/master/img/%E5%9B%BE%E7%89%87image-20210228161135777.png)

在上述的结构途中，模型主要分为的是两个部分：**FM和DNN**。DNN和之前的wide&Deep模型中的Deep部分的结构相似。而FM则分为了两个部分--分为了分析一阶特征和二阶交叉特征两个部分。所以上图将模型**分为了三个大块**。分别是**一阶特征处理linear部分，二阶特征交叉FM以及DNN的高阶特征交叉。**

- **FM线性部分--linear_logits 线性计算的方法**，是FM模块的一个部分。通过get_linear_logits()函数的线性的计算，这个函数的结果，与类别特征的一维one-hot编码线性的组合（PS：大佬指示，这里的dense特征并不是必须的，有可能会将数值特征进行分桶，然后再当做类别特征来处理）最后就是FM线性部分的输出。
- **FM离散特征的模型部分**：对于离散型的特征，都要过一遍嵌入层，然后使用FM特征交叉的方式，两两特征进行交叉，得到新的特征向量，最后计算交叉特征的logits。
- **DNN的离散特征部分**：这个部分的输入也是sparse-feature，首先过embedding，然后将得到的embedding拼接成一个向量。

```python
# 和之前一样的流程
def DeepFM(linear_feature_columns, dnn_feature_columns):
    # 构建输入层，即所有特征对应的Input()层，这里使用字典的形式返回，方便后续构建模型
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns + dnn_feature_columns)

    # 将linear部分的特征中sparse特征筛选出来，后面用来做1维的embedding
    linear_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), linear_feature_columns))

    # 构建模型的输入层，模型的输入层不能是字典的形式，应该将字典的形式转换成列表的形式
    # 注意：这里实际的输入与Input()层的对应，是通过模型输入时候的字典数据的key与对应name的Input层
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())

    # linear_logits由两部分组成，分别是dense特征的logits和sparse特征的logits
    linear_logits = get_linear_logits(dense_input_dict, sparse_input_dict, linear_sparse_feature_columns)

    # 构建维度为k的embedding层，这里使用字典的形式返回，方便后面搭建模型
    # embedding层用户构建FM交叉部分和DNN的输入部分
    embedding_layers = build_embedding_layers(dnn_feature_columns, sparse_input_dict, is_linear=False)

    # 将输入到dnn中的所有sparse特征筛选出来
    dnn_sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns))

    fm_logits = get_fm_logits(sparse_input_dict, dnn_sparse_feature_columns, embedding_layers) # 只考虑二阶项

    # 将所有的Embedding都拼起来，一起输入到dnn中
    dnn_logits = get_dnn_logits(sparse_input_dict, dnn_sparse_feature_columns, embedding_layers)
    
    # 将linear,FM,dnn的logits相加作为最终的logits
    output_logits = Add()([linear_logits, fm_logits, dnn_logits])

    # 这里的激活函数使用sigmoid
    output_layers = Activation("sigmoid")(output_logits)

    model = Model(input_layers, output_layers)
    return model
```

