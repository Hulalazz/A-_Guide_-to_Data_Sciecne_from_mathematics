## Network Compression and Acceleration

[To revolutionize deep learning with real-time AI solutions that scale from the edge to the datacenter](https://wavecomp.ai/).
The parameters of deep neural networks are tremendous. And deep learning is matrix-computation intensive. Specific hardware  such as GPU or TPU is used to speed up the computation of deep learning in training or inference.
The optimization methods are used to train the deep neural network. After training, the parameters of the deep neural network are fixed and in inference, we would do much matrix multiplication via the saved fixed parameters of deep neural network.  
<https://blogs.nvidia.com/blog/2016/08/22/difference-deep-learning-training-inference-ai/>

|Evolution of Model Compression and Acceleration|
|:-----:|
|Computer Architecture: TPUs|
|Compilers: [TVM](https://docs.tvm.ai/tutorials/)|
|Model Re-design: MobileNet|
|Re-parameterization: Pruning|
|Transfer Learning|

When the computation resource is limited such as embedded or mobile system,
can we deploy deep learning models? Definitely yes.

* [Hanlab: ACCELERATED DEEP LEARNING COMPUTING
Hardware, AI and Neural-nets](https://hanlab.mit.edu/)
* [模型压缩之deep compression](https://littletomatodonkey.github.io/2018/10/10/2018-10-10-%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E4%B9%8Bdeep%20compression/)
* [论文笔记《A Survey of Model Compression and Acceleration for Deep Neural Networks》](https://blog.csdn.net/song_pipi/article/details/79154539)
* https://zhuanlan.zhihu.com/p/67508423
* [Distiller in Github](https://nervanasystems.github.io/distiller/index.html)
* [Network Speed and Compression](https://github.com/mrgloom/Network-Speed-and-Compression)
* [PocketFlow in Github](https://github.com/Tencent/PocketFlow)
* https://pocketflow.github.io/
* https://hanlab.mit.edu/projects/tsm/
* [Model Compression and Acceleration](https://www.jiqizhixin.com/articles/2018-05-18-4)
* [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)
* [Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://arxiv.org/abs/1712.01887)
+ https://github.com/BlueWinters/research


### Sys for Deep Learning

* http://impact.crhc.illinois.edu/default.aspx
* https://www.sigarch.org/call-participation/ml-benchmarking-tutorial/
* https://sites.google.com/g.harvard.edu/mlperf-bench/home
* [Computer Systems Colloquium (EE380) Schedule](https://web.stanford.edu/class/ee380/)
* [DNN builder](https://www.c3sr.com/publication/2018/iccad_dnnbuilder/)
* [System for Machine Learning](https://dlsys.cs.washington.edu/)
* [Papers Reading List of *Embedded Neural Network*](https://github.com/ZhishengWang/Embedded-Neural-Network)
* [SigDL -- Deep Learning for IoT Device and Edge Computing Embedded Targets](https://github.com/signalogic/SigDL#DeepLearningModelCompression)
* https://developer.nvidia.com/tensorrt
* [Deep Compression and EIE](https://web.stanford.edu/class/ee380/Abstracts/160106-slides.pdf)
 


<img src=https://littletomatodonkey.github.io/img/post/20181010-DC-%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E6%B5%81%E7%A8%8B%E5%9B%BE.png width=80% />

### TVM and Versatile Tensor Accelerator (VTA)

<img src="https://raw.githubusercontent.com/uwsampl/web-data/master/vta/blogpost/vta_stack.png" width="70%" />

<img src="https://raw.githubusercontent.com/tvmai/tvmai.github.io/master/images/main/stack_tvmlang.png" width= "70%" />

* https://sampl.cs.washington.edu/
* https://homes.cs.washington.edu/~haichen/
* https://tqchen.com/
* https://www.cs.washington.edu/people/faculty/arvind
* [TVM: End to End Deep Learning Compiler Stack](https://tvm.ai/)
* [TVM and Deep Learning Compiler Conference](https://sampl.cs.washington.edu/tvmconf/)
* [VTA Deep Learning Accelerator](https://sampl.cs.washington.edu/projects/vta.html)

### Network Pruning

Pruning is to prune the connections in deep neural network in order to reduce the number of weights.

* https://nervanasystems.github.io/distiller/pruning/index.html
* https://github.com/yihui-he/channel-pruning
* https://pocketflow.github.io/cp_learner/
* [A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers](https://arxiv.org/abs/1804.03294)

### Quantization

Quantization is to quantize the weights in order to store the weights with less bits.
[Neural Network Quantization](https://jackwish.net/neural-network-quantization-introduction-chn.html)

* https://nervanasystems.github.io/distiller/quantization/index.html
* [PocketFlow is an open-source framework for compressing and accelerating deep learning models with minimal human effort.](https://pocketflow.github.io/uq_learner/)
* [Neural Network Quantization Resources](https://jackwish.net/neural-network-quantization-resources.html)

### Huffman Encoding

[Huffman coding](https://www.wikiwand.com/en/Huffman_coding) is a code scheme.

### Knowledge Distillation

* https://nervanasystems.github.io/distiller/knowledge_distillation/index.html
* https://github.com/dkozlov/awesome-knowledge-distillation
* https://github.com/lhyfst/knowledge-distillation-papers
* https://pocketflow.github.io/distillation/

<img src=https://pocketflow.github.io/pics/framework_design.png width=80% />

### Low-rank Approximation
  
[通用矩阵乘和卷积优化](https://jackwish.net/gemm-optimization-and-convolution.html)
Note that the deep learning models are composite of linear and non-linear maps. And linear maps are based on matrices.

The matrix $A_{m\times n}$ can be decompsed as the multiplication of two matrices such as $A_{m\times n}=Q_{m\times r}R_{r\times n}$, so that the storage is from $O(m\times n)$ to $O(m+n)\times O(r)$.

* https://en.wikipedia.org/wiki/Low-rank_approximation
* [Low Rank Matrix Approximation](http://www.cs.yale.edu/homes/el327/papers/lowRankMatrixApproximation.pdf)
* [On Compressing Deep Models by Low Rank and Sparse Decomposition](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_On_Compressing_Deep_CVPR_2017_paper.pdf)
* https://www.cnblogs.com/zhonghuasong/p/7821170.html

***

* https://srdas.github.io/DLBook/intro.html#effective
* https://zhuanlan.zhihu.com/p/48420428
* https://cognitiveclass.ai/courses/accelerating-deep-learning-gpu/
* https://vast.cs.ucla.edu/projects/acceleration-deep-learning-cloud-and-edge-computing
* https://github.com/songhan/Deep-Compression-AlexNet
* https://github.com/sun254/awesome-model-compression-and-acceleration
* https://github.com/chester256/Model-Compression-Papers
* [gab41.lab41.org](https://gab41.lab41.org/lab41-reading-group-deep-compression-9c36064fb209)
* [CS 598 LAZ: Cutting-Edge Trends in Deep Learning and Recognition](http://slazebni.cs.illinois.edu/spring17/)
* http://slazebni.cs.illinois.edu/spring17/lec06_compression.pdf
* http://slazebni.cs.illinois.edu/spring17/reading_lists.html#lec06
* https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning
* [CS236605: Deep Learning](https://vistalab-technion.github.io/cs236605/lectures/)