## Network Compression and Acceleration
https://www.comp.nus.edu.sg/~hebs/publication.html
<img src="https://cs217.stanford.edu/assets/img/4___JET.gif" width="50%" />

[To revolutionize deep learning with real-time AI solutions that scale from the edge to the datacenter](https://wavecomp.ai/).
The parameters of deep neural networks are tremendous. And deep learning is matrix-computation intensive. Specific hardware  such as GPU or TPU is used to speed up the computation of deep learning in training or inference.
The optimization methods are used to train the deep neural network. After training, the parameters of the deep neural network are fixed and in inference, we would do much matrix multiplication via the saved fixed parameters of deep neural network.  
<https://blogs.nvidia.com/blog/2016/08/22/difference-deep-learning-training-inference-ai/>

|Evolution of Model Compression and Acceleration|
|:-----:|
|Computer Architecture: TPUs|
|Compilers: [TVM](https://docs.tvm.ai/tutorials/)|
|Model Re-design: [EfficientNet](https://arxiv.org/pdf/1905.11946v1.pdf)|
|Re-parameterization: Pruning|
|Transfer Learning|

When the computation resource is limited such as embedded or mobile system,
can we deploy deep learning models? Definitely yes.

* [Hanlab: ACCELERATED DEEP LEARNING COMPUTING
Hardware, AI and Neural-nets](https://hanlab.mit.edu/)
* [模型压缩之deep compression](https://littletomatodonkey.github.io/2018/10/10/2018-10-10-%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E4%B9%8Bdeep%20compression/)
* [论文笔记《A Survey of Model Compression and Acceleration for Deep Neural Networks》](https://blog.csdn.net/song_pipi/article/details/79154539)
* https://zhuanlan.zhihu.com/p/67508423
* [Distiller an open-source Python package for neural network compression research](https://nervanasystems.github.io/distiller/index.html)
* [Network Speed and Compression](https://github.com/mrgloom/Network-Speed-and-Compression)
* [PocketFlow： An Automatic Model Compression (AutoMC) framework for developing smaller and faster AI applications ](https://github.com/Tencent/PocketFlow)
* https://pocketflow.github.io/
* https://hanlab.mit.edu/
* [Model Compression and Acceleration](https://www.jiqizhixin.com/articles/2018-05-18-4)
* [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)
* [Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://arxiv.org/abs/1712.01887)
+ https://github.com/BlueWinters/research


### Sys for Deep Learning

* https://www.xilinx.com/
* https://wavecomp.ai/
* [An in-depth look at Google’s first Tensor Processing Unit (TPU)](https://cloud.google.com/blog/products/gcp/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu)
* [Eyeriss: An Energy-Efficient Reconfigurable Accelerator
for Deep Convolutional Neural Networks ](http://eyeriss.mit.edu/)
* http://impact.crhc.illinois.edu/default.aspx
* https://www.alphaics.ai/
* http://www.cambricon.com/
* https://en.wikipedia.org/wiki/AI_accelerator
* https://www.sigarch.org/call-participation/ml-benchmarking-tutorial/
* [BENCHMARKING DEEP LEARNING SYSTEMS](https://sites.google.com/g.harvard.edu/mlperf-bench/home)
* [Computer Systems Colloquium (EE380) Schedule](https://web.stanford.edu/class/ee380/)
* [DNN builder](https://www.c3sr.com/publication/2018/iccad_dnnbuilder/)
* [System for Machine Learning @.washington.edu/](https://dlsys.cs.washington.edu/)
* [Workshop on Systems for ML and Open Source Software at NeurIPS 2018](http://learningsys.org/nips18/schedule.html)
* [Papers Reading List of *Embedded Neural Network*](https://github.com/ZhishengWang/Embedded-Neural-Network)
* [SigDL -- Deep Learning for IoT Device and Edge Computing Embedded Targets](https://github.com/signalogic/SigDL#DeepLearningModelCompression)
* [Deep Compression and EIE](https://web.stanford.edu/class/ee380/Abstracts/160106-slides.pdf)
* [Hardware Accelerators for Machine Learning (CS 217)](https://cs217.stanford.edu/)
* [Acceleration of Deep Learning for Cloud and Edge Computing@UCLA](https://vast.cs.ucla.edu/projects/acceleration-deep-learning-cloud-and-edge-computing)


<img src=https://littletomatodonkey.github.io/img/post/20181010-DC-%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E6%B5%81%E7%A8%8B%E5%9B%BE.png width=80% />


* [Programmable Inference Accelerator](https://developer.nvidia.com/tensorrt)
* https://github.com/NervanaSystems/ngraph

### Compilers for Deep Learning

#### TVM and Versatile Tensor Accelerator (VTA)

<img src="https://raw.githubusercontent.com/uwsampl/web-data/master/vta/blogpost/vta_stack.png" width="70%" />


Matrix computation dense application like deep neural network would take the advantages of specific architecture design.

<img src="https://raw.githubusercontent.com/tvmai/tvmai.github.io/master/images/main/stack_tvmlang.png" width= "100%" />

* https://sampl.cs.washington.edu/
* https://homes.cs.washington.edu/~haichen/
* https://tqchen.com/
* https://www.cs.washington.edu/people/faculty/arvind
* [TVM: End to End Deep Learning Compiler Stack](https://tvm.ai/)
* [TVM and Deep Learning Compiler Conference](https://sampl.cs.washington.edu/tvmconf/)
* [VTA Deep Learning Accelerator](https://sampl.cs.washington.edu/projects/vta.html)

+ https://en.wikipedia.org/wiki/Zeroth_(software)

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

* [Huffman coding](https://www.wikiwand.com/en/Huffman_coding) is a code scheme.
* [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)
* [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)
* [XNOR-Net论文解读](https://zhuanlan.zhihu.com/p/65103916)
* [Joseph Chet Redmon: author of XORNet](https://pjreddie.com/)

### Knowledge Distillation

* https://nervanasystems.github.io/distiller/knowledge_distillation/index.html
* https://github.com/dkozlov/awesome-knowledge-distillation
* https://github.com/lhyfst/knowledge-distillation-papers
* https://pocketflow.github.io/distillation/

<img src=https://pocketflow.github.io/pics/framework_design.png width=80% />

### Low-rank Approximation

[通用矩阵乘和卷积优化](https://jackwish.net/gemm-optimization-and-convolution.html)

Note that the deep learning models are composite of linear and non-linear maps. And linear maps are based on matrices.

The matrix $A_{m\times n}$ can be decomposed as the multiplication of two matrices such as $A_{m\times n}= Q_{m\times r}R_{r\times n}$, so that the storage is from $O(m\times n)$ to $O(m+n)\times O(r)$.

And `Toeplitz Matrix` can be applied to approximate  the  weight matrix
$$
W = {\alpha}_1T_{1}T^{−1}_{2} + {\alpha}_2 T_3 T_{4}^{-1} T_{5}
$$

where ${M}$ is the square weight matrix, $T_1, T_2, T_3, T_4, T_5$ are square *Toeplitz matrix*.

<img title="espnets" src="https://prior.allenai.org/assets/project-content/espnets/esp-unit.jpg" width="80%" />

* https://en.wikipedia.org/wiki/Low-rank_approximation
* [Low Rank Matrix Approximation](http://www.cs.yale.edu/homes/el327/papers/lowRankMatrixApproximation.pdf)
* [On Compressing Deep Models by Low Rank and Sparse Decomposition](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_On_Compressing_Deep_CVPR_2017_paper.pdf)
* [ESPNets for Computer Vision
Efficient CNNs for Edge Devices](https://prior.allenai.org/projects/espnet)
* https://www.cnblogs.com/zhonghuasong/p/7821170.html

***

* https://srdas.github.io/DLBook/intro.html#effective
* https://zhuanlan.zhihu.com/p/48420428
* https://cognitiveclass.ai/courses/accelerating-deep-learning-gpu/
* https://github.com/songhan/Deep-Compression-AlexNet
* https://github.com/sun254/awesome-model-compression-and-acceleration
* https://github.com/chester256/Model-Compression-Papers
* [gab41.lab41.org](https://gab41.lab41.org/lab41-reading-group-deep-compression-9c36064fb209)
* [CS 598 LAZ: Cutting-Edge Trends in Deep Learning and Recognition](http://slazebni.cs.illinois.edu/spring17/)
* http://slazebni.cs.illinois.edu/spring17/lec06_compression.pdf
* http://slazebni.cs.illinois.edu/spring17/reading_lists.html#lec06
* https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning
* [CS236605: Deep Learning](https://vistalab-technion.github.io/cs236605/lectures/)
* https://mlperf.org/
