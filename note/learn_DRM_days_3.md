---
title: "深度推荐模型学习小记（四）"
tag: 
  - DeepLearning
  - python
  - RecommendationModel
  - dataWhale
date: 2021-03-20
---

# 深度推荐模型学习小记（四）

## Task04---NFM的学习

### **1.简介**

NFM(Neural Factorization Machines)是2017年由新加坡国立大学的何向南教授等人在SIGIR会议上提出的一个模型，**传统的FM模型仅局限于线性表达和二阶交互，** 无法胜任生活中各种具有复杂结构和规律性的真实数据， 针对FM的这点不足， **作者提出了一种将FM融合进DNN的策略，通过引进了一个特征交叉池化层的结构**，使得FM与DNN进行了完美衔接，这样就**组合了FM的建模低阶特征交互能力和DNN学习高阶特征交互和非线性的能力，形成了深度学习时代的神经FM模型(NFM)**。

先是NFM的公式定义：

![NFM定义公式](https://gitee.com/magicye/blogimage/raw/master/img/image-20210323200835765.png)

![公式对比](https://gitee.com/magicye/blogimage/raw/master/img/%E5%9B%BE%E7%89%871.png)

两个公式只有第三项有不同，传统的FM的模型仅局限于线性表达和二阶交互，所以在公式中的体现就是**只能到二阶交叉， 且是线性模型**。要研究更为复杂的阶数交互，就必须使用**一个表达能力更强的函数来替代原FM中二阶隐向量内积的部分。**

作者换为了一个神经网络来当可以拟合任何复杂能力的函数**当然不是一个简单的DNN， 而是依然底层考虑了交叉，然后高层使用的DNN网络，** 这个也就是最终的NFM网络了：

![模型结构图](https://gitee.com/magicye/blogimage/raw/master/img/%E5%9B%BE%E7%89%872.png)

### 2.模型的结构和原理

![图片NFM模块化](https://gitee.com/magicye/blogimage/raw/master/img/%E5%9B%BE%E7%89%87NFM_aaaa.png)

#### 2.1 Input 和 Embedding层

$$
O\left(k N_{x}\right)O\left(k N_{x}\right)O\left(k N_{x}\right)
$$

输入层的特征， **文章指定了稀疏离散特征居多**， 这种特征我们也知道**一般是先one-hot, 然后会通过embedding，处理成稠密低维的**。 （所以输入层输入到嵌入层的输出的流程和之前的代码是一样的）假设![](https://gitee.com/magicye/blogimage/raw/master/img/image-20210323204011789.png)为第 i 个特征的embedding向量， 那么![](https://gitee.com/magicye/blogimage/raw/master/img/image-20210323203913140.png)表示的下一层的输入特征。这里带上了X_i是因为很多X_i转成了One-hot之后，出现很多为0的， 这里的![](https://gitee.com/magicye/blogimage/raw/master/img/image-20210323204200543.png)是X_i不等于0的那些特征向量。一维的sparse-feature也是进入到 get_linear_logits中与dense-feature进行加和。

#### 2.2 Bi-Interaction Pooling layer

**在Embedding层和神经网络之间加入了特征交叉池化层是本网络的核心创新了**，正是因为这个结构，实现了FM与DNN的无缝连接， 组成了一个大的网络，且能够正常的反向传播。假设{V_x}是所有特征embedding的集合， 那么在特征交叉池化层的操作：

![向量的元素集集合](https://gitee.com/magicye/blogimage/raw/master/img/image-20210323205556922.png)

![](https://gitee.com/magicye/blogimage/raw/master/img/image-20210323205750093.png)表示两个向量的元素积操作,**两个向量对应维度相乘得到的元素积向量**（可不是点乘呀）,其中第K维的操作：

![第K维的元素积操作](https://gitee.com/magicye/blogimage/raw/master/img/image-20210323210200913.png)

这便定义了在embedding空间特征的二阶交互，这个不仔细看会和感觉FM的最后一项很像，但是不一样，**一定要注意这个地方不是两个隐向量的内积，而是元素积，也就是这一个交叉完了之后k个维度不求和，最后会得到一个K维向量，而FM那里内积的话最后得到一个数**， 在进行**两两Embedding元素积之后，对交叉特征向量取和， 得到该层的输出向量**， 很显然， 输出是一个K维的向量。

注意， 之前的FM到这里其实就完事了， 上面就是输出了，**而这里很大的一点改进就是加入特征池化层之后， 把二阶交互的信息合并， 且上面接了一个DNN网络， 这样就能够增强FM的表达能力了**， 因为FM只能到二阶， 而这里的DNN可以进行多阶且非线性，只要FM把二阶的学习好了， DNN这块学习来会更加容易， 作者在论文中也说明了这一点，且通过后面的实验证实了这个观点。

如果不加DNN， NFM就退化成了FM，所以改进的关键就在于加了一个这样的层，组合了一下二阶交叉的信息，然后又给了DNN进行高阶交叉的学习，**成了一种“加强版”的FM。**

Bi-Interaction层不需要额外的模型学习参数，更重要的是它在一个线性的时间内完成计算，和FM一致的，即时间复杂度为![](https://gitee.com/magicye/blogimage/raw/master/img/image-20210323211846825.png)，N_x为embedding向量的数量。参考FM，可以将上式转化为（平方公式的展开）：

![变化公式](https://gitee.com/magicye/blogimage/raw/master/img/image-20210323211958646.png)

后面代码复现NFM就是用的这个公式直接计算，比较简便且清晰。

#### 2.3 隐藏层

也就是DNN layer 这一层就是全连接的神经网络， DNN在进行特征的高层非线性交互上有着天然的学习优势，公式如下(可多层):

![全连接公式](https://gitee.com/magicye/blogimage/raw/master/img/image-20210323212442115.png)

#### 2.4 预测层

这个就是最后一层的结果直接过一个隐藏层，但注意由于这里是回归问题，没有加sigmoid激活：

![隐藏层](https://gitee.com/magicye/blogimage/raw/master/img/image-20210323214326501.png)

所以， NFM模型的前向传播过程总结如下：

![前向传播的公式](https://gitee.com/magicye/blogimage/raw/master/img/image-20210323214428684.png)

这就是NFM模型的全貌， NFM相比较于其他模型的核心创新点是特征交叉池化层，基于它，实现了FM和DNN的无缝连接，使得DNN可以在底层就学习到包含更多信息的组合特征，这时候，就会减少DNN的很多负担，只需要很少的隐藏层就可以学习到高阶特征信息。NFM相比之前的DNN， 模型结构更浅，更简单，但是性能更好，训练和调参更容易。集合FM二阶交叉线性和DNN高阶交叉非线性的优势，非常适合处理稀疏数据的场景任务。在对NFM的真实训练过程中，也会用到像Dropout和Batch-Normalization这样的技术来缓解过拟合和在过大的改变数据分布。

### 3.参考资料

- [论文原文](https://arxiv.org/pdf/1708.05027.pdf)

- [deepctr](https://github.com/shenweichen/DeepCTR)

- [AI上推荐 之 FNN、DeepFM与NFM(FM在深度学习中的身影重现)](https://blog.csdn.net/wuzhongqiang/article/details/109532267?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161442951716780255224635%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=161442951716780255224635&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_v1~rank_blog_v1-1-109532267.pc_v1_rank_blog_v1&utm_term=NFM)

- 王喆 - 《深度学习推荐系统》

