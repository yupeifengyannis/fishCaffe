### ResNet

在分析ResNet网络之前我们先来看一下caffe中相关的层。

#### eltwise层

在ResNet中最关键的是一个残差模块，其结构示意图如下：

<img src="C:\Users\yupei\Desktop\caffe源码\图片\残差模块.png" style="zoom:50%" />

主要的意思是将前面的feature map加到后面的feature map，反向传播的时候$x$这个feature map的反向误差会累计两次。
$$
H(x) = F(x) + x
$$
梯度计算：
$$
\frac{\partial{E}}{\partial{x}} = \frac{\partial{E}}{{\partial{H(x)}}} (\frac{\partial{H(x)}}{\partial{x}}+
\frac{\partial{H(x)}}{\partial F(x)}\frac{\partial{F(x)}}{\partial(x)}
)
$$
这边在ResNet中$x$和$F(x)$是元素之间做加法，但是实际上我们可以在元素之间做减法，选最大值，做乘法运算。这些操作EltwiseLayer中都是已经支持了，我们来具体看一下吧。

##### 成员变量

(include/caffe/layers/eltwise_layer.hpp)

```c++
     42   EltwiseParameter_EltwiseOp op_;     //我们在proto文件中可以看一下这个类成员
     43   vector<Dtype> coeffs_;           //如果是做加法，则这个是每个变量的系数
     44   Blob<int> max_idx_;			//记录最大值所在的位置
     45 
     46   bool stable_prod_grad_;		//乘法梯度信息
```

我们从proto文件中来具体看一下EltwiseParameter这个类吧。

（src/caffe/proto/caffe.proto）

```c++
    720 message EltwiseParameter {
    721   enum EltwiseOp {
    722     PROD = 0;
    723     SUM = 1;
    724     MAX = 2;
    725   }
    726   optional EltwiseOp operation = 1 [default = SUM]; // element-wise operation
    727   repeated float coeff = 2; // blob-wise coefficient for SUM operation
    728 
    729   // Whether to use an asymptotically slower (for >2 inputs) but stabler method
    730   // of computing the gradient for the PROD operation. (No effect for SUM op.)
    731   optional bool stable_prod_grad = 3 [default = true];
    732 }
```

我们可以看到一般默认的方式是采用sum操作，ResNet的残差模块就是基于这个操作的，关于其他的的product和max操作目前没有在哪些论文上有看到这一块的内容。

##### LayerSetUp函数

在这里该函数没啥特殊的，主要是确定操作类型，系数项coeff等成员变量。

##### Reshape函数

首先我们的EltwiseLayer层的bottom输入的blob个数一般情况下要大于等于两个，其实一般情况下都是两个blob。那么这两个blob的维度必须是要一样的，如果维度不一样的化则eltwise按元素处理就不能处理了，同时该层的输出的top blob的维度和bottom的blob维度也是要一样的。

##### Forwad_cpu函数

**product前向传播：**

```c++
     55         case EltwiseParameter_EltwiseOp_PROD:
     56             caffe_mul(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);
     57             for (int i = 2; i < bottom.size(); ++i) {
     58                 caffe_mul(count, top_data, bottom[i]->cpu_data(), top_data);
     59             }
     60             break;
```

将bottom中每个blob的对应位置的元素进行相乘，然后将其放到top_data当中，原理非常简单。

**sum前向传播：**

```c++

```

代码很简单，将每个bottom blob的元素乘以相应的系数然后填充到top中。

**max前向传播：**

```c++
     68         case EltwiseParameter_EltwiseOp_MAX:
     69             // Initialize
     70             mask = max_idx_.mutable_cpu_data();
     71             caffe_set(count, -1, mask);
     72             caffe_set(count, Dtype(-FLT_MAX), top_data);
     73             // bottom 0 & 1
     74             bottom_data_a = bottom[0]->cpu_data();
     75             bottom_data_b = bottom[1]->cpu_data();
     76             for (int idx = 0; idx < count; ++idx) {
     77                 if (bottom_data_a[idx] > bottom_data_b[idx]) {
     78                     top_data[idx] = bottom_data_a[idx];  // maxval
     79                     mask[idx] = 0;  // maxid
     80                 } else {
     81                     top_data[idx] = bottom_data_b[idx];  // maxval
     82                     mask[idx] = 1;  // maxid
     83                 }
     84             }
     85             // bottom 2++
     86             for (int blob_idx = 2; blob_idx < bottom.size(); ++blob_idx) {
     87                 bottom_data_b = bottom[blob_idx]->cpu_data();
     88                 for (int idx = 0; idx < count; ++idx) {
     89                     if (bottom_data_b[idx] > top_data[idx]) {
     90                         top_data[idx] = bottom_data_b[idx];  // maxval
     91                         mask[idx] = blob_idx;  // maxid
     92                     }
     93                 }
     94             }
     95             break;
```

首先max操作就是在几个blob中选取最大的值，这个和我们之前的max pooling很类似。因此我们需要有一个和top blob相同大小的blob来记录相应的位置。注意一点，我们输入的bottom blob的个数可能不止2个。

**Backward_cpu函数**

那我们可以看一下上述三种正向传播的反向传播算法是怎么样的吧。

**product反向传播**

```c++
    113                 case EltwiseParameter_EltwiseOp_PROD:
    114                     if (stable_prod_grad_) {
    115                         bool initialized = false;
    116                         for (int j = 0; j < bottom.size(); ++j) {
    117                             if (i == j) { continue; }
    118                             if (!initialized) {
    119                                 caffe_copy(count, bottom[j]->cpu_data(), bottom_diff);
    120                                 initialized = true;
    121                             } else {
    122                                 caffe_mul(count, bottom[j]->cpu_data(), bottom_diff,
    123                                         bottom_diff);
    124                             }
    125                         }
    126                     } else {
    127                         caffe_div(count, top_data, bottom_data, bottom_diff);
    128                     }
    129                     caffe_mul(count, bottom_diff, top_diff, bottom_diff);
    130                     break;
```

这里的product的反向传播有两种情况，但是这两种方式怎么感觉都是一样的。前面一种是将微分的系数一个个累成到bottom_diff中，而后一种是将top_data除以自己的值从而得到系数，**难道是因为用了除法而导致求的梯度不稳定吗？**

**sum反向传播**

```c++
    131                 case EltwiseParameter_EltwiseOp_SUM:
    132                     if (coeffs_[i] == Dtype(1)) {
    133                         caffe_copy(count, top_diff, bottom_diff);
    134                     } else {
    135                         caffe_cpu_scale(count, coeffs_[i], top_diff, bottom_diff);
    136                     }
    137                     break;
```

sum反向传播比较简单，只要让top blob那边传来的梯度乘以相应的系数。

**max反向传播**

```c++

```

注意只有共享了最大值的那个位置有梯度传回来，否则梯度都是0。