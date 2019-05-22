## caffe中的Batch Normalization实现

这边我们使用在mnist数据集上训练的Lenet模型用来讲解batch normalization层的前向传播计算和反向传播计算过程，Lenet训练模型的位置为（examples/mnist/lenet_train_test.prototxt）。我们在其中添加一个bn1层。

我们在pool1层和conv2层之间添加一个bn层，只要确定好bottom和top的名字就好。

```c++
layer{
    name: "bn1"
    type: "BatchNorm"
    top:  "bn1"
    bottom: "pool1"
}
```

#### 数学原理分析

##### 前向传播过程

Batch Normalization的意思是将所有batch在同一个通道上的feature map做的正规化运算。所以BN扫描的维度是在$(N,H,W)$这三个维度上，这点非常重要，因为后面介绍如何将反向传播向量化的时候需要用大这一个知识点。那下面是BN的前向传播过程，整个过程看起来比较简单，我们主要注意一下几个重要量的维度就可以了。
$$
\begin{equation}
\mu_{\mathcal{B}} \leftarrow \frac{1}{m} \sum_{i=1}^{m} x_{i}
\end{equation}
$$

$$
\begin{equation}
\sigma_{\mathcal{B}}^{2} \leftarrow \frac{1}{m} \sum_{i=1}^{m}\left(x_{i}-\mu_{\mathcal{B}}\right)^{2}
\end{equation}
$$

$$
\begin{equation}
\widehat{x}_{i} \leftarrow \frac{x_{i}-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}}
\end{equation}
$$

$$
\begin{equation}
y_{i} \leftarrow \gamma \widehat{x}_{i}+\beta \equiv \mathrm{B} \mathrm{N}_{\gamma, \beta}\left(x_{i}\right)
\end{equation}
$$

$\mu_{\mathcal{B}} $、$\sigma_{\mathcal{B}}^{2}$、$\gamma$以及$\beta$是一个$C$长度的数组，这点需要注意一下。

##### 反向传播过程

$$
\begin{equation}
\frac{\partial \ell}{\partial \widehat{x}_{i}}=\frac{\partial \ell}{\partial y_{i}} \cdot \gamma
\end{equation}
$$

$$
\begin{equation}
\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^{2}}=\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}} \cdot\left(x_{i}-\mu_{\mathcal{B}}\right) \cdot \frac{-1}{2}\left(\sigma_{\mathcal{B}}^{2}+\epsilon\right)^{-3 / 2}
\end{equation}
$$

$$
\begin{equation}
\frac{\partial \ell}{\partial \mu_{\mathcal{B}}}=\left(\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}} \cdot \frac{-1}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}}\right)+\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^{2}} \cdot \frac{\sum_{i=1}^{m}-2\left(x_{i}-\mu_{\mathcal{B}}\right)}{m}
\end{equation}
$$

$$
\begin{equation}
\frac{\partial \ell}{\partial x_{i}}=\frac{\partial \ell}{\partial \widehat{x}_{i}} \cdot \frac{1}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}}+\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^{2}} \cdot \frac{2\left(x_{i}-\mu_{\mathcal{B}}\right)}{m}+\frac{\partial \ell}{\partial \mu_{\mathcal{B}}} \cdot \frac{1}{m}
\end{equation}
$$

$$
\begin{equation}
\frac{\partial \ell}{\partial \gamma}=\sum_{i=1}^{m} \frac{\partial \ell}{\partial y_{i}} \cdot \widehat{x}_{i}
\end{equation}
$$

$$
\begin{equation}
\frac{\partial \ell}{\partial \beta}=\sum_{i=1}^{m} \frac{\partial \ell}{\partial y_{i}}
\end{equation}
$$

上述6个公式是BN论文给出来的反向传播函数，这个公式理解起来比较容易，但是代码实现起来却没那么简单。这边需要将反向传播的过程向量化，那我们看一下具体如何向量化。首先我们看一下(10)这个公式如何将其向量化。

这边我们需要对公式(8)中的$\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^{2}}$进行相应的变换，我们用公式(5)来代替其中的某些项，则公式(8)变为：
$$
\begin{equation}
\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^{2}}=\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}} \cdot \hat{x}_{i} \cdot \frac{-1}{2}\left(\sigma_{\mathcal{B}}^{2}+\epsilon\right)^{-1}
\end{equation}
$$
注意一点，这边累加量$i$活动的范围是$(N,H,W)$这三个维度。所以和$\sigma_{\mathcal{B}}^{2}$没关系，我们可以将$\left(\sigma_{\mathcal{B}}^{2}+\epsilon\right)^{-1}$给移动出来，即得到：
$$
\begin{equation}
\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^{2}}= \frac{-1}{2}\left(\sigma_{\mathcal{B}}^{2}+\epsilon\right)^{-1}\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}} \cdot \hat{x}_{i} 
\end{equation}
$$
那我们将变换之后的$\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^{2}}$带入公式(10)的$\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^{2}} \cdot \frac{2\left(x_{i}-\mu_{\mathcal{B}}\right)}{m}$中，则有：
$$
\begin{equation}
\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^{2}} \cdot \frac{2\left(x_{i}-\mu_{\mathcal{B}}\right)}{m}= -1\left(\sigma_{\mathcal{B}}^{2}+\epsilon\right)^{\frac{-1}{2}}        
\frac{\hat{x}_{i}
\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}} \cdot \hat{x}_{i} }{m}
\end{equation}
$$
那我们看$\frac{\partial \ell}{\partial \mu_{\mathcal{B}}}$的变换情况，我们观察公式(1)和(7)，我们可以很清楚的发现$\frac{\partial \ell}{\partial \mu_{\mathcal{B}}}$后面一项是0，这就省去了大量的工作，那么$\frac{\partial \ell}{\partial \mu_{\mathcal{B}}} \cdot \frac{1}{m}$就变换为：
$$
\frac{\partial \ell}{\partial \mu_{\mathcal{B}}} \cdot \frac{1}{m} =\frac{-1}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}} 
\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}}
$$
嗯，那我们所有的转换过程都已经完成了，那我们看一下$\frac{\partial \ell}{\partial x_{i}}$变换为:
$$
\frac{\partial \ell}{\partial x_{i}} = \frac{1}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}} \left(\frac{\partial \ell}{\partial \widehat{x}_{i}} - 
\frac{1}{m} \left(\hat{x}_{i}
\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}} \cdot \hat{x}_{i}   +  
\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}}
\right)
\right)
$$
记住公式(15)，我们后面会在反向传播中看它是如何实现的。

#### 代码分析

我们先来看一下BatchNormLayer层的成员变量

```c++
     63   Blob<Dtype> mean_, variance_, temp_, x_norm_;
     64   bool use_global_stats_;
     65   Dtype moving_average_fraction_;
     66   int channels_;
     67   Dtype eps_;
     68 
     69   // extra temporarary variables is used to carry out sums/broadcasting
     70   // using BLAS
     71   Blob<Dtype> batch_sum_multiplier_;
     72   Blob<Dtype> num_by_chans_;
     73   Blob<Dtype> spatial_sum_multiplier_;
```

参数解释，mean_存放feature map在batch上的平均值，variance\_记录feature map在batch上的标准差，temp\_一个中间容器，x\_norm\_存正则化之后的数据，use\_global\_stats\_一般是测试的时候会用，moving\_average\_fraction\_是移动平均，channels\_通道数，eps\_微小量。

batch\_sum\_multiplier\_是最终获得batch层面上的中间量（我一下子比较难给它起个好名字）。

num\_by\_chans\_记录$N\times C$的结果。

spatial\_sum\_multiplier\_是用于获取feature map层面上的中间量。

##### LayerSetUp函数

该函数主要确定一些量的大小，具体代码我就不放了，下面列一下各个量的维度情况。

$channels\_ = C$。

blobs\_[0]和blobs\_[1]的是一个长度为$C$的数组，blobs\_[2]是一个标量。

##### Reshape函数

mean\_，variance\_是一个大小为$C$的数组，temp\_和x\_norm\_的维度为$(N,C,H,W)$。

batch\_sum\_multiplier\_是一个大小为$N$的数组，里面的值都是1。

spatial\_sum\_multiplier\_的维度为$(H, W)$，里面的值都是1。

num\_by\_chans\_是一个大小为$N\times C$的数组。

##### Forward\_cpu函数

**第一步：计算batch平均值**

首先计算每个feature map的平均值，总共有$N \times C$个feature map，我们将其存放到num\_by\_chans\_这个blob中。

```c++
    108     caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
    109         1. / (num * spatial_dim), bottom_data,
    110         spatial_sum_multiplier_.cpu_data(), 0.,
    111         num_by_chans_.mutable_cpu_data());
```

然后我们计算batch的平均值，注意这边我们的num\_by\_chans\_这个矩阵要转置一下，因为我们最终的获得的mean\_的维度是一个$C$长度的数组，而num\_by\_chans\_的矩阵形式的维度为$(N \times C)$

```c++
    112     caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
    113         num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
    114         mean_.mutable_cpu_data());
```

**第二步：将所有的像素点减去相应的平均值**

嗯嗯，我们现在的平均值都是存放在长度为$C$的mean\_数组上，但要做向量减法操作，我们必须要进行广播操作，将相应的平均值最终广播到各个feature map当中，我们来看一下具体是则么做的吧。

首先先将mean\_数组扩展为num×channels\_的矩阵，这里矩阵乘法注意batch\_sum\_multiplier\_在前面，mean\_在后面，将扩展之后的结果存放到num\_by\_chans\_中。

```c++
    118   caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
    119       batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
    120       num_by_chans_.mutable_cpu_data());
```

后面我们将num\_by\_chans\_数组扩展为$N\times H \times C \times W$的blob中。注意这里的矩阵乘法函数的$\beta$值为1，$\alpha$值为-1，那么经过该矩阵乘法运算之后每个像素点都减去相应的平均值。

```c++
    121   caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
    122       spatial_dim, 1, -1, num_by_chans_.cpu_data(),
    123       spatial_sum_multiplier_.cpu_data(), 1., top_data);
```

**第三步：计算方差和标准差**

计算$(X-E(X))^2$的结果

```c++
    127     caffe_sqr<Dtype>(top[0]->count(), top_data,
    128                      temp_.mutable_cpu_data());  // (X-EX)^2
```

计算feature map层面上的方差，将结果存在num\_by\_chans\_中

```c++
    129     caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
    130         1. / (num * spatial_dim), temp_.cpu_data(),
    131         spatial_sum_multiplier_.cpu_data(), 0.,
    132         num_by_chans_.mutable_cpu_data());
```

计算batch层面的方差，将结果存在variance\_中，和前面求mean值一样，注意矩阵转置情况。

```c++
    133     caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
    134         num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
    135         variance_.mutable_cpu_data());  // E((X_EX)^2)
```

然后计算标准差

```c++
    150   caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
    151   caffe_sqrt(variance_.count(), variance_.cpu_data(),
    152              variance_.mutable_cpu_data());
```

**第四步，将前面减去的均值的数据除以标准差**

这里整个过程和上面的减去均值的操作类似，需要一个反向扩展的操作，最终扩展到$N\times H \times C \times W$大小的blob数据，然后做向量除法操作。

```c++
    155   caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
    156       batch_sum_multiplier_.cpu_data(), variance_.cpu_data(), 0.,
    157       num_by_chans_.mutable_cpu_data());
    158   caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
    159       spatial_dim, 1, 1., num_by_chans_.cpu_data(),
    160       spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
    161   caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
```

##### Backward_cpu函数

计算得到$\frac{\partial \ell}{\partial \widehat{x}_{i}}\hat{x}_{i}$，这是对每个元素进行向量化乘法操作。

```c++
    200   caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
```

下面该矩阵线程是进行一个归约的操作，即计算得到$\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}}\hat{x}_{i}$的结果，当然这个操作是在$N,H,W$这三个维度上做的归约操作，得到的是一个长度为$C$的数组。

```c++
    201   caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
    202       bottom_diff, spatial_sum_multiplier_.cpu_data(), 0.,
    203       num_by_chans_.mutable_cpu_data());
    204   caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
    205       num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
    206       mean_.mutable_cpu_data());
```

接下来我们将结果广播到$N,H,W$这三个空间中，这样我们的bottom_diff上面每个元素都是$\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}}\hat{x}_{i}$的值了。

```c++
    209   caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
    210       batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
    211       num_by_chans_.mutable_cpu_data());
    212   caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
    213       spatial_dim, 1, 1., num_by_chans_.cpu_data(),
    214       spatial_sum_multiplier_.cpu_data(), 0., bottom_diff);
```

做一个向量化乘法，得到$\hat{x}_{i}\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}}\hat{x}_{i}$

```c++
    217   caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);
```

那同理下面的也是通过归约操作和广播操作得到bottom_diff
$$
\left(\hat{x}_{i}
\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}} \cdot \hat{x}_{i}   +  
\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}}
\right)
$$

```c
    220   caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
    221       top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
    222       num_by_chans_.mutable_cpu_data());
    223   caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
    224       num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
    225       mean_.mutable_cpu_data());
    226   // reshape (broadcast) the above to make
    227   // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
    228   caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
    229       batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
    230       num_by_chans_.mutable_cpu_data());
    231   caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
    232       spatial_dim, 1, 1., num_by_chans_.cpu_data(),
    233       spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);
```

最后通过下面两部向量化的操作，我们最终得到我们的反向传播结果：
$$
\frac{\partial \ell}{\partial x_{i}} = \frac{1}{\sqrt{\sigma_{\mathcal{B}}^{2}+\epsilon}} \left(\frac{\partial \ell}{\partial \widehat{x}_{i}} - 
\frac{1}{m} \left(\hat{x}_{i}
\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}} \cdot \hat{x}_{i}   +  
\sum_{i=1}^{m} \frac{\partial \ell}{\partial \widehat{x}_{i}}
\right)
\right)
$$

```c++
    236   caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff,
    237       Dtype(-1. / (num * spatial_dim)), bottom_diff);
    238 
    239   // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
    240   // pass.
    241   caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
```

ok，这就是BN层中正规化操作的后向传播，这个后向传播可能一开始理解起来比较困难，需要对矩阵归约的代码操作和反向传播公式的向量化改造有非常清晰的认识，如果没有代码参考，我感觉我也很难想出操作BN层的向量化代码，有时候一想这就是阅读源码的乐趣吧，看看大神们是如何使用一些巧妙的技巧来解决的一些复杂的问题。

#### MVN层

在caffe框架中还有一个MVN层的正规化操作，它和这个BN原理差不多。但是它所处理的维度是在feature map的的维度上，即$(H, W)$这两个维度，如果看懂了BN层的代码那么看MVN层的代码就非常的简单了。关于MVN层的代码这里就不细讲了，可以自己推导一遍MVN层的公式以及理解一遍它的反向传播算法。

