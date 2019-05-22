### GoogLeNet

我们在看googlenet的时候，可以关注到它里面核心的模块就是它的**inception**模块，其示意图如下：

<img src="C:\Users\yupei\Desktop\caffe源码\图片\googlenet.png" style="zoom:100%" />

这里比较核心的一个点是如何将几个feature map模块给合并在一起。那在caffe中我们可以看到ConcatLayer就支持这一操作。我们来具体看一下它是如何操作的吧。

##### ConcatLayer层

首先bottom输入的blob个数肯定是大于1的，输入的blob的数据为

$input\_1$的维度为$(N\times C \times H\times W)$。

$input\_2$的维度为$(N\times C \times H\times W)$。

.......

$input\_k$的维度为$(N\times C \times H \times W)$。

如果我们以轴0进行concat的话，则输出维度为$(kN\times C \times H \times W)$。

如果我们是以轴1进行concat的话，则输出维度为$(N\times kC \times H \times W)$。**常用的都是这种**

那么整个前向传播过程可以写为下式：
$$
y = [\begin{array}{cccc} x_1 & x_2 & ... & x_K \end{array}]
$$
反向传播则如下，其实都非常的简单。
$$
\left[ \begin{array}{cccc}
         \frac{\partial E}{\partial x_1} &
         \frac{\partial E}{\partial x_2} &
          ... &
         \frac{\partial E}{\partial x_K}
     \end{array} \right] =
      \frac{\partial E}{\partial y}
$$
**成员变量**

(include/caffe/layers/concat_layer.hpp)

```c++
     79   int count_;		//top blob元素个数
     80   int num_concats_;	 //concat的次数
     81   int concat_input_size_;	//concat input的大小
     82   int concat_axis_;	//基于哪个轴进行concat
```

关于num_concats_的含义，如果我们是按照轴0进行concat的话，则次数为$N$次。如果我们按照轴1进行concat的话，则次数为$N\times C$次。

那concat_input_size这个参数的含义为，如果轴为0的话，则大小为$C\times H \times W$，如果轴为1的话，则大小为$H\times W$。

同理我们得看一下proto文件中关于ConcatParameter参数的定义：我们发现很简单，我们可以知道一般都是在channel这个轴上进行concat，这就和我们理解的Inception模块合并过程就一样了。

(src/caffe/proto/caffe.proto)

```c++
    515 message ConcatParameter {
    516   // The axis along which to concatenate -- may be negative to index from the
    517   // end (e.g., -1 for the last axis).  Other axes must have the
    518   // same dimension for all the bottom blobs.
    519   // By default, ConcatLayer concatenates blobs along the "channels" axis (1).
    520   optional int32 axis = 2 [default = 1];
    521 
    522   // DEPRECATED: alias for "axis" -- does not support negative indexing.
    523   optional uint32 concat_dim = 1 [default = 1];
    524 }
```

##### LayerSetUp函数

该函数非常简单，简单一看就好。

##### Reshape函数

我们来简单的看一下代码它是如何得到top blob的维度大小的。

```c++
     32     // Initialize with the first blob.
     33     vector<int> top_shape = bottom[0]->shape();
     34     num_concats_ = bottom[0]->count(0, concat_axis_); //需要进行concat的次数
     35     concat_input_size_ = bottom[0]->count(concat_axis_ + 1); 
     36     int bottom_count_sum = bottom[0]->count();
     37     for (int i = 1; i < bottom.size(); ++i) {
     38         CHECK_EQ(num_axes, bottom[i]->num_axes())
     39             << "All inputs must have the same #axes.";
     40         for (int j = 0; j < num_axes; ++j) {
         			//检车除了其他轴的维度是否相等，如果不相等说明无法concat
     41             if (j == concat_axis_) { continue; }
     42             CHECK_EQ(top_shape[j], bottom[i]->shape(j))
     43                 << "All inputs must have the same shape, except at concat_axis.";
     44         }
     45         bottom_count_sum += bottom[i]->count();
         		//轴所在的维度累加，将每个bottom的轴长度相加
     46         top_shape[concat_axis_] += bottom[i]->shape(concat_axis_);
     47     }
     48     top[0]->Reshape(top_shape);   //构造top blob的维度
     49     CHECK_EQ(bottom_count_sum, top[0]->count());
			//如果我们的输入的bottom只有一个blob，则bottom和top的所有数据都是一模一样的。
     50     if (bottom.size() == 1) {
     51         top[0]->ShareData(*bottom[0]);
     52         top[0]->ShareDiff(*bottom[0]);
     53     }
     54 }
```

##### Forward_cpu函数

```
     56 template <typename Dtype>
     57 void ConcatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
     58         const vector<Blob<Dtype>*>& top) {
     59     if (bottom.size() == 1) { return; }
     60     Dtype* top_data = top[0]->mutable_cpu_data();
     61     int offset_concat_axis = 0;
     62     const int top_concat_axis = top[0]->shape(concat_axis_);
     63     for (int i = 0; i < bottom.size(); ++i) {
     64         const Dtype* bottom_data = bottom[i]->cpu_data();
     65         const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
     66         for (int n = 0; n < num_concats_; ++n) {
     67             caffe_copy(bottom_concat_axis * concat_input_size_,
     68                     bottom_data + n * bottom_concat_axis * concat_input_size_,
     69                     top_data + (n * top_concat_axis + offset_concat_axis)
     70                     * concat_input_size_);
     71         }
     72         offset_concat_axis += bottom_concat_axis;
     73     }
     74 }
```

