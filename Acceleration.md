# Network Compression and Acceleration

<img src="https://cs217.stanford.edu/assets/img/4___JET.gif" width="50%" />

* https://github.com/1duo/awesome-ai-infrastructures
* [FairNAS: Rethinking Evaluation Fairness of Weight Sharing Neural Architecture Search](https://github.com/fairnas/FairNAS)
* [VISUAL COMPUTING SYSTEMS](http://graphics.cs.cmu.edu/courses/15769/fall2016/lectures)
* https://zhuanlan.zhihu.com/jackwish
* https://machinethink.net/blog/compressing-deep-neural-nets/
* [Rethinking Deep Learning: Architectures and Algorithms](https://nickhigham.files.wordpress.com/2019/05/talk12-constantinides.pdf)
* https://arxiv.org/abs/1904.00938
* https://github.com/ChanChiChoi/awesome-model-compression
* https://github.com/fengbintu/Neural-Networks-on-Silicon
* https://vast.cs.ucla.edu/

[To revolutionize deep learning with real-time AI solutions that scale from the edge to the datacenter](https://wavecomp.ai/).

The parameters of deep neural networks are tremendous. And deep learning is matrix-computation intensive. Specific hardware  such as GPU or TPU is used to speed up the computation of deep learning in training or inference.
The optimization methods are used to train the deep neural network.
To boost the training of deep learning, we would like to design faster optimization methods such as `ADAM` and delicate architectures of neural network such as `ResNet`.
After training, the parameters of the deep neural network are fixed and used for inference, we would do much matrix multiplication via the saved fixed parameters of deep neural network.  
From [What’s the Difference Between Deep Learning Training and Inference?](https://blogs.nvidia.com/blog/2016/08/22/difference-deep-learning-training-inference-ai/)

<img src="https://blogs.nvidia.com/wp-content/uploads/2016/08/ai_difference_between_deep_learning_training_inference.jpg" width="80%">

|Evolution of Model Compression and Acceleration|
|:-----:|
|Computer Architecture: TPUs, GPUs|
|Compilers: [TVM](https://docs.tvm.ai/tutorials/)|
|Model Re-design: [EfficientNet](https://arxiv.org/pdf/1905.11946v1.pdf)|
|Re-parameterization: Pruning|
|Transfer Learning|

When the computation resource is limited such as embedded or mobile system,
can we deploy deep learning models? Definitely yes.


* [Awesome model compression and acceleration](https://github.com/memoiry/Awesome-model-compression-and-acceleration)
* [Acceleration and Model Compression by Handong](https://handong1587.github.io/deep_learning/2015/10/09/cnn-compression-acceleration.html)
* [模型压缩之deep compression](https://littletomatodonkey.github.io/2018/10/10/2018-10-10-%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E4%B9%8Bdeep%20compression/)
* [论文笔记《A Survey of Model Compression and Acceleration for Deep Neural Networks》](https://blog.csdn.net/song_pipi/article/details/79154539)
* https://zhuanlan.zhihu.com/p/67508423
* [Network Speed and Compression](https://github.com/mrgloom/Network-Speed-and-Compression)
* [Model Compression and Acceleration](https://www.jiqizhixin.com/articles/2018-05-18-4)
* [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)
* [Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://arxiv.org/abs/1712.01887)
* [AutoML 的十大开源库](https://www.52cs.com/archives/3138)
* [TensorFlow模型压缩和Inference加速](https://zhuanlan.zhihu.com/p/31023153)
+ https://github.com/BlueWinters/research
+ [Reference workloads for modern deep learning methods](https://github.com/rdadolf/fathom)



## Sys for Deep Learning

Over the past few years, deep learning has become an important technique to successfully solve problems in many different fields, such as vision, NLP, robotics. An important ingredient that is driving this success is the development of deep learning systems that efficiently support the task of learning and inference of complicated models using many devices and possibly using distributed resources. The study of how to build and optimize these deep learning systems is now an active area of research and commercialization.

Matrix computation dense application like deep neural network would take the advantages of specific architecture design. Thus it is really close to `high performance computational science` when sloving some computation dense problems.

<img src="https://pooyanjamshidi.github.io/mls/_images/mls-logo.jpg" width="69%" />


* [Workshop on AI Systems](http://learningsys.org/sosp19/)
* http://learningsys.org/nips18/
* http://learningsys.org/sosp17/
* https://sites.google.com/site/mlsys2016/
* [Programmable Hardware Accelerators (Winter 2019)](https://cmpe293-winter19-01.courses.soe.ucsc.edu/home)
* [Hardware Accelerators for Machine Learning (CS 217)](https://cs217.stanford.edu/)
* [CSCE 790/590: Machine Learning Systems](https://github.com/pooyanjamshidi/mls)
* [Illinois Microarchitecture Project using Algorithms and Compiler Technology](http://impact.crhc.illinois.edu/default.aspx)
* [Deep Learning for Computer Architects](https://www.morganclaypool.com/doi/abs/10.2200/S00783ED1V01Y201706CAC041)
* [System for Machine Learning @.washington.edu/](https://dlsys.cs.washington.edu/)
* [Hanlab: ACCELERATED DEEP LEARNING COMPUTING Hardware, AI and Neural-nets](https://hanlab.mit.edu/)
* [Bingsheng He's publication on GPU](https://www.comp.nus.edu.sg/~hebs/publication.html)
* [Workshop on Systems for ML and Open Source Software at NeurIPS 2018](http://learningsys.org/nips18/schedule.html)
* [Papers Reading List of *Embedded Neural Network*](https://github.com/ZhishengWang/Embedded-Neural-Network)
* [Computer Systems Colloquium (EE380) Schedule](https://web.stanford.edu/class/ee380/)
* [ML Benchmarking Tutorial](https://www.sigarch.org/call-participation/ml-benchmarking-tutorial/)
* [The ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE)](https://2018.fseconference.org/home)
* [The fastest path to machine learning integration](https://intel.github.io/dffml/)
* https://www.sigarch.org/call-participation/ml-benchmarking-tutorial/
* [DNNBuilder: an Automated Tool for Building High-Performance DNN Hardware Accelerators for FPGAs](https://www.c3sr.com/publication/2018/iccad_dnnbuilder/)
* [ISCA 2016 in Seoul](http://isca2016.eecs.umich.edu/)
* [Deep Compression and EIE](https://web.stanford.edu/class/ee380/Abstracts/160106-slides.pdf)
* [Acceleration of Deep Learning for Cloud and Edge Computing@UCLA](https://vast.cs.ucla.edu/projects/acceleration-deep-learning-cloud-and-edge-computing)
* [Hot Chips: A Symposium on High Performance Chips](http://hotchips.org)
* https://paco-cpu.github.io/paco-cpu/
* [Programmable Inference Accelerator](https://developer.nvidia.com/tensorrt)
* [Fair and useful benchmarks for measuring training and inference performance of ML hardware, software, and services.](https://mlperf.org/)
* https://hanlab.mit.edu/
* http://yanjoy.win/

<img src="https://www.researchgate.net/profile/Gu_Yeon_Wei/publication/306398249/figure/fig2/AS:614016141512719@1523404264555/Breakdown-of-execution-time-by-operation-type-for-each-Fathom-workload.png" width="80%" />

* https://www.alphaics.ai/
* http://www.cambricon.com/
* https://www.sigarch.org/
* https://www.xilinx.com/
* https://wavecomp.ai/
* https://www.graphcore.ai/
* https://www.alphaics.ai/
* https://www.wikiwand.com/en/Hardware_acceleration
* [An in-depth look at Google’s first Tensor Processing Unit (TPU)](https://cloud.google.com/blog/products/gcp/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu)
* [BENCHMARKING DEEP LEARNING SYSTEMS](https://sites.google.com/g.harvard.edu/mlperf-bench/home)
* [EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML](https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html)
* [GPU，CUDA，cuDNN的理解](https://blog.csdn.net/u014380165/article/details/77340765)

## Numerical algorithms for high-performance computational science

Several key themes emerged across multiple talks in [Royal Society Discussion Meeting](https://constantinides.net/2019/04/23/royal-society-discussion-meeting/), all in the context of today’s high performance computing landscape in which processor clock speeds have stagnated (with the end of Moore’s law) and exascale machine are just two or three years away.

* An important way of accelerating computations is through the use of `low precision floating-point arithmetic`—in particular by exploiting a hierarchy of precisions.
* We must exploit `low rank matrix structure` where it exists, for example in hierarchical (H-matrix) form, combining it with randomized approximations.
* Minimizing `data movement (communication)` is crucial, because of its increasing costs relative to the costs of floating-point arithmetic.
* `Co-design` (the collaborative and concurrent development of hardware, software, and numerical algorithms, with knowledge of applications) is increasingly important for numerical computing.

+ [Numerical Algorithms for High-Performance Computational Science: Highlights of the Meeting](https://nickhigham.wordpress.com/2019/05/07/numerical-algorithms-for-high-performance-computational-science-highlights/)
+ [Numerical algorithms for high-performance computational science](https://royalsociety.org/science-events-and-lectures/2019/04/high-performance-computing/)
+ [Reflections on the Royal Society’s “Numerical Algorithms for High-performance Computational Science” Discussion Meeting](https://sinews.siam.org/Details-Page/reflections-on-the-royal-societys-numerical-algorithms-for-high-performance-computational-science-discussion-meeting)
+ [Overview of Microsoft HPC Pack 2016](https://docs.microsoft.com/zh-cn/powershell/high-performance-computing/overview?view=hpc16-ps)
+ [Document Library: High Performance Computing Fabrics](https://www.intel.com/content/www/us/en/high-performance-computing-fabrics/library.html)
+ https://researchcomputing.lehigh.edu/
+ https://library.columbia.edu/libraries/dsc/hpc.html
+ [Open MPI: Open Source High Performance Computing](https://www.open-mpi.org/)
+ https://ltsnews.lehigh.edu/node/115
+ https://developer.nvidia.com/cuda-zone
+ [NumFOCUS](https://numfocus.org/sponsored-projects)
+ http://www.mit.edu/~kepner/D4M/
+ [Butterflies Are All You Need: A Universal Building Block for Structured Linear Maps](https://dawn.cs.stanford.edu/2019/06/13/butterfly/)

### Automatic Differentiation



Many algorithms in machine learning, computer vision, physical simulation, and other fields require the calculation of `gradients and other derivatives`. Manual derivation of gradients can be time consuming and error-prone. `Automatic Differentiation (AD)` is a technology for automatically augmenting computer programs, including arbitrarily complex simulations, with statements for the computation of derivatives, also known as sensitivities. Automatic differentiation comprises a set of techniques to calculate the derivative of a numerical computation expressed as a computer program. These techniques are commonly used in atmospheric sciences and computational fluid dynamics, and have more recently also been adopted by machine learning researchers.

Practitioners across many fields have built a wide set of automatic differentiation tools, using different programming languages, computational primitives and intermediate compiler representations. Each of these choices comes with positive and negative trade-offs, in terms of their usability, flexibility and performance in specific domains.

In the ideal case, automatically generated derivatives should be competitive with manually generated ones and run at near-peak performance on modern hardware, but the most expressive systems for autodiff which can handle arbitrary, Turing-complete programs, are unsuited for performance-critical applications, such as large-scale machine learning or physical simulation. Alternatively, the most performant systems are not designed for use outside of their designated application space, e.g. graphics or neural networks.

All numerical gradient-based optimization methods benifits from faster computation of gradients specially `backprop`.


+ https://autodiff-workshop.github.io/
+ https://autodiff-workshop.github.io/2016.html
+ https://program-transformations.github.io/
+ http://www.autodiff.org/
+ https://autodiff.github.io/
+ https://github.com/google/jax
+ https://github.com/google/tangent
+ https://en.wikipedia.org/wiki/Automatic_differentiation
+ http://www.admb-project.org/
+ https://github.com/rjhogan/Adept
+ http://www.met.reading.ac.uk/clouds/adept/
+ [AD computation with Template Model Builder (TMB)](https://github.com/kaskr/adcomp)
+ https://non-contradiction.github.io/autodiffr/
+ https://srijithr.gitlab.io/post/autodiff/
+ https://fl.readthedocs.io/en/latest/autograd.html
+ https://pymanopt.github.io/
+ https://yiduai.sg/tensorflow-workshop/

### Generalized Matrix Multiplication Optimization

If the computation speed  of matrix operation is boosted, the inference of deep learning model is accelerated.
Matrix multiplication  $C_{M\times N}=A_{M\times K}B_{K\times N}$ via dot product is defined as
$$C[m,n]=\left< A[m,:], B[:, m]\right>=\sum_{k=1}^{K}A[m, k]\times B[k, n]$$

which is esentially product-sum.
```
for (int m = 0; m < M; m++) {
  for (int n = 0; n < N; n++) {
    C[m][n] = 0;
    for (int k = 0; k < K; k++) {
      C[m][n] += A[m][k] * B[k][n];
    }
  }
}
```

It  needs $O(MKN)$ multiplication.

##### Strassen Algorithms

It is based on block-multiplication. It is required that $C\in\mathbb R^{2^n\times 2^n}$.

The matrice are rearranged as blocks:
$$
\mathbf{A} =
  \begin{bmatrix}
   \mathbf{A}_{1,1} & \mathbf{A}_{1,2} \\
   \mathbf{A}_{2,1} & \mathbf{A}_{2,2}
  \end{bmatrix},
\mathbf{B} =
  \begin{bmatrix}
   \mathbf{B}_{1,1} & \mathbf{B}_{1,2} \\
   \mathbf{B}_{2,1} & \mathbf{B}_{2,2}
  \end{bmatrix},\\
\mathbf{C} =
  \begin{bmatrix}
   \mathbf{C}_{1,1} & \mathbf{C}_{1,2} \\
   \mathbf{C}_{2,1} & \mathbf{C}_{2,2}
  \end{bmatrix} =
  \begin{bmatrix}
   \mathbf{A}_{1,1}\mathbf{B}_{1,1} & \mathbf{A}_{1,2}\mathbf{B}_{1,2} \\
   \mathbf{A}_{2,1}\mathbf{B}_{2,1} & \mathbf{A}_{2,2}\mathbf{B}_{2,2}
  \end{bmatrix}.
  $$

Submatrix(blocks) multiplication is performed in the following way:
$$
\mathbf{M}_{1} =\left(\mathbf{A}_{1,1}+\mathbf{A}_{2,2}\right)\left(\mathbf{B}_{1,1}+\mathbf{B}_{2,2}\right) \\
\mathbf{M}_{2} =\left(\mathbf{A}_{2,1}+\mathbf{A}_{2,2}\right) \mathbf{B}_{1,1} \\
\mathbf{M}_{3} =\mathbf{A}_{1,1}\left(\mathbf{B}_{1,2}-\mathbf{B}_{2,2}\right) \\
\mathbf{M}_{4} =\mathbf{A}_{1,2}\left(\mathbf{B}_{2,1}-\mathbf{B}_{1,1}\right) \\
\mathbf{M}_{5} =\left(\mathbf{A}_{1,1}+\mathbf{A}_{1,2}\right) \mathbf{B}_{2,2} \\
\mathbf{M}_{6} =\left(\mathbf{A}_{2,1}-\mathbf{A}_{1,1}\right)\left(\mathbf{B}_{1,1}+\mathbf{B}_{1,2}\right) \\
\mathbf{M}_{7} =\left(\mathbf{A}_{1,2}-\mathbf{A}_{2,2}\right)\left(\mathbf{B}_{2,1}+\mathbf{B}_{2,2}\right)
$$

And then
$$
\mathbf{C}_{1,1} =\mathbf{M}_{1}+\mathbf{M}_{4}-\mathbf{M}_{5}+\mathbf{M}_{7} \\
\mathbf{C}_{1,2} =\mathbf{M}_{3}+\mathbf{M}_{5} \\
\mathbf{C}_{2,1} =\mathbf{M}_{2}+\mathbf{M}_{4} \\
\mathbf{C}_{2,2} =\mathbf{M}_{1}-\mathbf{M}_{2}+\mathbf{M}_{3}+\mathbf{M}_{6}
$$

#### Coppersmith–Winograd Algorithms

For matrix multiplication $U = V = W = \mathbb F^{n \times n}$, and we write this bilinear map as $\phi=<n, n, n>$ where $\phi(\cdot, \cdot): U\times V\mapsto W$ is bilinear.

The tensor corresponding to the multiplication of an $m\times n$ matrix by an $n\times  p$ matrix is
$$\left<m, n, p\right>=\sum_{i=1}^{m}\sum_{j=1}^{p}\sum_{k=1}^{n} a_{ik}\otimes b_{kj}\otimes c_{ij}.$$

One can define in a natural way the tensor product of two tensors. In particular, for matrix multipli-
cation tensors, we obtain the following identity: for any positive integers $m, m_0, n, n_0, p, p_0$,
$$\left<m, n, p\right>\otimes \left<m_0, n_0, p_0\right>=\left<m m_0, n n_0,  p p_0\right>.$$

Consider three vector spaces $U$, $V$ and $W$ over the field $\mathbb F$ and
* $U=span\{x_1, \dots, x_{dim(U)}\}$,
* $V=span\{y_1, \dots, y_{dim(U)}\}$,
* $W=span\{z_1, \dots, z_{dim(U)}\}$.

A tensor over $(U, V, W)$ is an element of $U\otimes V\otimes W$ i.e., a formal sum
$$T=\sum_{u=1}^{dim(U)}\sum_{v=1}^{dim(V)}\sum_{w=1}^{dim(W)}\underbrace{d_{uvw}}_{\in\mathbb F} x_{u}\otimes y_{v}\otimes z_{w}.$$


<img src="https://jackwish.net/images/2019/qnnpack/qnnpack-gemm-reduce.jpg" width="70%" />
<img src="https://jackwish.net/images/2019/gemm-opt/gemm-1x4.svg" width="60%" />

* [Powers of Tensors and Fast Matrix Multiplication](https://simons.berkeley.edu/sites/default/files/docs/2438/slideslegall.pdf)
* https://www.kkhaydarov.com/matrix-multiplication-algorithms/
* [BLISlab: A Sandbox for Optimizing GEMM](https://github.com/flame/blislab)
* http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/gemm/
* [通用矩阵乘和卷积优化](https://jackwish.net/gemm-optimization-and-convolution.html)
* [Fast Matrix Multiplication Algorithms](https://www.ics.uci.edu/~fastmm/)
* [Anatomy of high-performance matrix multiplication](https://dl.acm.org/citation.cfm?id=1356053)
* [The Indirect Convolution Algorithm](https://arxiv.org/abs/1907.02129)
* https://github.com/flame/how-to-optimize-gemm/wiki
* https://en.wikipedia.org/wiki/Matrix_multiplication#Complexity
* [Coppersmith-Winograd Algorithm](https://www.gabormelli.com/RKB/Coppersmith-Winograd_Algorithm)
* [Matrix multiplication via arithmetic progressions](https://www.sciencedirect.com/science/article/pii/S0747717108800132)
* [On the Coppersmith–Winograd method](http://www.cs.toronto.edu/~yuvalf/Limitations.pdf)
* [Breaking the Coppersmith-Winograd barrier](https://www.cs.rit.edu/~rlc/Courses/Algorithms/Papers/matrixMult.pdf)
* [Adaptive Winograd’s Matrix Multiplications](https://www.ics.uci.edu/~fastmm/FMM-Reference/dalberto-nicolau.winograd.TOMS.pdf)
* [Multiplying matrices faster than Coppersmith-Winograd](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.297.2680&rep=rep1&type=pdf)
* https://www.wikiwand.com/en/Coppersmith%E2%80%93Winograd_algorithm
* https://www.wikiwand.com/en/Matrix_multiplication_algorithm
* [Limits on All Known (and Some Unknown) Approaches to Matrix Multiplication](https://simons.berkeley.edu/talks/virginia)

#### Fixed-point arithmetic and Approximate Computing

Today’s computing systems are designed to deliver only exact solutions at high energy cost, while many of the algorithms that are run on data are at their heart statistical, and thus do not require exact answers.

It turns out that it is sometimes possible to get high-accuracy solutions from low-precision training—and here we'll describe a new variant of stochastic gradient descent (SGD) called high-accuracy low precision (HALP) that can do it. HALP can do better than previous algorithms because it reduces the two sources of noise that limit the accuracy of low-precision SGD: gradient variance and round-off error.

<img src="https://jackwish.net/images/2019/quantization/mixed-fp32int8-pure-int8.svg" width="80%"/>

* https://www.wikiwand.com/en/Fixed-point_arithmetic
* [Approximate Computing](http://moimani.weebly.com/approximate-computing.html)
* [PACO: The Worlds First Approximate Computing General Purpose CPU](https://paco-cpu.github.io/paco-cpu/)
* [System Energy Efficiency Lab](http://seelab.ucsd.edu/)
* http://www.oliviervalery.com/publications/pdp2018
* [Training deep neural networks with low precision multiplications](https://arxiv.org/abs/1412.7024)
* [Ultra-Low-Precision Training of Deep Neural Networks](https://www.ibm.com/blogs/research/2019/05/ultra-low-precision-training/)
* [HALP: High-Accuracy Low-Precision Training](https://dawn.cs.stanford.edu/2018/03/09/low-precision/)
* [Quantized Neural Network PACKage - mobile-optimized implementation of quantized neural network operators ](https://github.com/pytorch/QNNPACK)
* http://proceedings.mlr.press/v37/gupta15.pdf
* [Making Neural Nets Work With Low Precision](https://sahnimanas.github.io/2018/06/24/quantization-in-tf-lite.html)
* http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
* https://devblogs.nvidia.com/int8-inference-autonomous-vehicles-tensorrt/


## Compilers for Deep Learning

|DNN Acceleratation Framework|
|---|
|[NVIDIA cuDNN GPU Accelerated Deep Learning](https://developer.nvidia.com/cudnn)|
|[cupy](https://cupy.chainer.org/), [ideep](https://github.com/intel/ideep)|
|[Menoh](https://github.com/pfnet-research/menoh)|
|[Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)](https://github.com/intel/mkl-dnn)|
|[Reference workloads for modern deep learning methods.](https://rdadolf.github.io/fathom/)|
|[nGraph](https://www.ngraph.ai/),[PlaidML](https://www.intel.ai/plaidml/)|
|[Minerva: a fast and flexible tool for deep learning on multi-GPU.](https://github.com/dmlc/minerva)|
|[SigDL -- Deep Learning for IoT Device and Edge Computing Embedded Targets](https://github.com/signalogic/SigDL#DeepLearningModelCompression)|

* https://arxiv.org/pdf/1602.01528.pdf


### TVM and Versatile Tensor Accelerator (VTA)

> TVM is an open deep learning compiler stack for CPUs, GPUs, and specialized accelerators. It aims to close the gap between the productivity-focused deep learning frameworks, and the performance- or efficiency-oriented hardware backends. TVM provides the following main features:

> Compilation of deep learning models in Keras, MXNet, PyTorch, Tensorflow, CoreML, DarkNet into minimum deployable modules on diverse hardware backends.
Infrastructure to automatic generate and optimize tensor operators on more backend with better performance.

<img src="https://raw.githubusercontent.com/tvmai/tvmai.github.io/master/images/main/stack_tvmlang.png" width= "80%" />


The Versatile Tensor Accelerator (VTA) is an extension of the TVM framework designed to advance deep learning and hardware innovation. VTA is a programmable accelerator that exposes a RISC-like programming abstraction to describe compute and memory operations at the tensor level. We designed VTA to expose the most salient and common characteristics of mainstream deep learning accelerators, such as tensor operations, DMA load/stores, and explicit compute/memory arbitration.


<img src="https://raw.githubusercontent.com/uwsampl/web-data/master/vta/blogpost/vta_stack.png" width="70%" />



* https://sampl.cs.washington.edu/
* https://homes.cs.washington.edu/~haichen/
* https://tqchen.com/
* https://www.cs.washington.edu/people/faculty/arvind
* [TVM: End to End Deep Learning Compiler Stack](https://tvm.ai/)
* [TVM and Deep Learning Compiler Conference](https://sampl.cs.washington.edu/tvmconf/)
* [VTA Deep Learning Accelerator](https://sampl.cs.washington.edu/projects/vta.html)
* https://sampl.cs.washington.edu/
* https://docs.tvm.ai/vta/index.html
* [如何利用TVM快速实现超越Numpy(MKL)的GEMM蓝色](https://zhuanlan.zhihu.com/p/75203171)
* [使用TVM支持TFLite（下)](https://zhuanlan.zhihu.com/p/57147430)

#### nGraph

`nGraph` is an end to end deep learning compiler for inference and training with extensive framework and hardware support.

<img src="https://www.ngraph.ai/sites/default/files/2019-08/main_diagram_fw_hw.png" width="70%" />

+ https://www.ngraph.ai/
+ https://github.com/NervanaSystems/ngraph
- https://en.wikipedia.org/wiki/Zeroth_(software)

#### XLA

The XLA compilation framework is invoked on subgraphs of TensorFlow computations. The framework requires all tensor shapes to be fixed, so compiled code is specialized to concrete shapes. This means, for example, that the compiler may be invoked multiple times for the same subgraph if it is executed on batches of different sizes.

- https://www.tensorflow.org/versions/master/experimental/xla/
- https://developers.googleblog.com/2017/03/xla-tensorflow-compiled.html
- https://www.tensorflow.org/xla/overview
- https://autodiff-workshop.github.io/slides/JeffDean.pdf
- [XLA: The TensorFlow compiler framework](https://haosdent.gitbooks.io/tensorflow-document/content/resources/xla_prerelease.html)

#### JAX: Autograd and XLA

With its updated version of Autograd, JAX can automatically differentiate native Python and NumPy functions. It can differentiate through loops, branches, recursion, and closures, and it can take derivatives of derivatives of derivatives. It supports reverse-mode differentiation (a.k.a. backpropagation) via grad as well as forward-mode differentiation, and the two can be composed arbitrarily to any order.

<img src="https://raw.githubusercontent.com/google/jax/master/images/lifecycle.png" width="60%" />

- https://github.com/google/jax

#### Multi-Level Intermediate Representation

The Multi-Level Intermediate Representation (MLIR) is intended for easy expression and optimization of computations involving deep loop nests and dense matrices of high dimensionality. It is thus well-suited to deep learning computations in particular. Yet it is general enough to also represent arbitrary sequential computation. The representation allows high-level optimization and parallelization for a wide range of parallel architectures including those with deep memory hierarchies --- general-purpose multicores, GPUs, and specialized neural network accelerators.

- https://github.com/tensorflow/mlir
- https://llvm.org/devmtg/2019-04/slides/Keynote-ShpeismanLattner-MLIR.pdf

#### Glow

Glow is a machine learning compiler and execution engine for hardware accelerators. It is designed to be used as a backend for high-level machine learning frameworks. The compiler is designed to allow state of the art compiler optimizations and code generation of neural network graphs. This library is in active development.

- https://arxiv.org/pdf/1805.00907.pdf
- https://ai.facebook.com/tools/glow/
- https://github.com/pytorch/glow


----

### Compression and Acceleration of CNN

Theme Name | Description | Application | More Details
----|----|----|----
Parameter pruning and sharing | Reducing redundant parameters which are not sensitive to the performance | Convolutional layer and fully connected layer| Robust to various setting, can achieve good performance, can support both train from scratch and pre-trained model
Low-rank factorization| Using matrix/tensor decomposition to estimate the information parameters | Convolutional layer and fully connected layer| Standardized pipeline, easily to be implemented, can support both train from scratch and pre-trained model
Transferred/compact convolutional filters | Designing special structural convolutional filter to save parameters | Convolutional layer  only | Algorithms are dependent on applications, usually achieve good performance, only support train from scratch
Knowledge distillation |Training a compact neural network with distilled knowledge of a large model |Convolutional layer and fully connected layer| Model performances are sensitive to applications and network structure only support train from scratch

* https://jackwish.net/convolution-neural-networks-optimization.html
* [Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks ](http://eyeriss.mit.edu/)
* [深度学习如何进行模型压缩？](https://www.zhihu.com/question/64987081/answer/684375500)
* [Caffeine: Towards Uniformed Representation and Acceleration for Deep Convolutional Neural Networks](https://vast.cs.ucla.edu/publications/caffeine-towards-uniformed-representation-and-acceleration-deep-convolutional-neural)
* [Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks](https://vast.cs.ucla.edu/publications/optimizing-fpga-based-accelerator-design-deep-convolutional-neural-networks)
* [Automated Systolic Array Architecture Synthesis for High Throughput CNN Inference on FPGAs](https://vast.cs.ucla.edu/publications/automated-systolic-array-architecture-synthesis-high-throughput-cnn-inference-fpgas)

#### Parameter Pruning and Sharing

Pruning is to prune the connections in deep neural network in order to reduce the number of weights.

* Learn the connectivity via normal network training
* Prune the low-weight connections
* Retrain the sparse network

<img src=https://littletomatodonkey.github.io/img/post/20181010-DC-%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E6%B5%81%E7%A8%8B%E5%9B%BE.png width=80% />

----
* https://nervanasystems.github.io/distiller/pruning/index.html
* https://github.com/yihui-he/channel-pruning
* https://pocketflow.github.io/cp_learner/
* [A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers](https://arxiv.org/abs/1804.03294)

##### Quantization and Binarization

Network quantization compresses the original network by
reducing the number of bits required to represent each weight.

Uniform quantization is widely used for model compression and acceleration. Originally the weights in the network are represented by 32-bit floating-point numbers. With uniform quantization, low-precision (e.g. 4-bit or 8-bit) fixed-point numbers are used to approximate the full-precision network. For k-bit quantization, the memory saving can be up to $32/k$​. For example, 8-bit quantization can reduce the network size by 4 folds with negligible drop of performance.
The lth quantized ReLU $\sigma(x_l, \alpha_l)$ acts element-wise on vector $x_l$ from a previous layer and is parameterized by trainable scalar $\alpha_l>0$.  
In uniform quantization,
$$
\sigma (x,\alpha ) = \begin{cases}
0, \quad & \mathrm{if}\quad x \leq 0,\\
k\, \alpha, \quad & \mathrm{if}\quad \left(k-1\right)\alpha < x \leq k\, \alpha, \; k = 1, 2, \dots, 2^{b_a}-1,\\
\left(2^{b_a}-1\right)\alpha,\quad & \mathrm{if}\quad x > \left(2^{b_a}-1\right)\alpha, \tag1
\end{cases}
$$
where $x$ is the scalar input, $b_a$ is the bit-width, and $k$ is the quantization level. For a 4-bit quantization, $b_a=4$ and $2^{b_a}=16$ levels exist, including zero.

<img src="https://jackwish.net/images/2019/quantization/calibration-and-quantization-aware-training.jpg" width="70%" />

* https://zhuanlan.zhihu.com/p/38328685
* https://nervanasystems.github.io/distiller/quantization/index.html
* https://jackwish.net/neural-network-quantization-resources.html
* https://jackwish.net/neural-network-quantization-introduction.html
* [Neural Network Quantization](https://jackwish.net/neural-network-quantization-introduction-chn.html)
* [PocketFlow is an open-source framework for compressing and accelerating deep learning models with minimal human effort.](https://pocketflow.github.io/uq_learner/)
* [Neural Network Quantization Resources](https://jackwish.net/neural-network-quantization-resources.html)
* [Low Precision Arithmetic Simulation in PyTorch](https://github.com/Tiiiger/QPyTorch)
* [Training Quantized Deep Neural Networks and Applications with Blended Coarse Gradient Descent](https://sinews.siam.org/Details-Page/training-quantized-deep-neural-networks-and-applications-with-blended-coarse-gradient-descent)

##### Binarized Neural Network, Ternary Weight Networks, XOR-Net

<img title="XNOR Net" src="https://pic1.zhimg.com/80/v2-a5bcc5b680ec296aeb706ca4f2fe2c90_hd.jpg" width="80%" />


* [Boolean Circuits are Neural Networks](https://constantinides.net/2019/04/26/boolean-circuits-are-neural-networks/)
* https://www.sciencedirect.com/science/article/pii/S0925231217315655
* https://arxiv.org/abs/1602.02830
* https://pjreddie.com/media/files/papers/xnor.pdf
* [XNOR-Net论文解读](https://zhuanlan.zhihu.com/p/65103916)

<img src="https://jackwish.net/images/2019/quantization/fp-distribution.png" width="50%"/>

##### Huffman Encoding

Huffman code is a type of optimal prefix code that is commonly used for loss-less data compression.
It produces a variable-length code table for encoding source symbol.
The table is derived from the occurrence
probability for each symbol.
As in other entropy encoding methods, more common symbols are represented with fewer bits than less common symbols, thus save the total space.

* [Huffman coding](https://www.wikiwand.com/en/Huffman_coding) is a code scheme.
* [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)
* [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)
* [Joseph Chet Redmon: author of XORNet](https://pjreddie.com/)

#### Knowledge Distillation

Distillation (Hinton et al., 2015) is a kind of model compression approaches in which a pre-trained large model teaches a smaller model to achieve the similar prediction performance. It is often named as the "teacher-student" training, where the large model is the teacher and the smaller model is the student.

With distillation, knowledge can be transferred from the teacher model to the student by minimizing a loss function to recover the distribution of class probabilities predicted by the teacher model. In most situations, the probability of the correct class predicted by the teacher model is very high, and probabilities of other classes are close to 0, which may not be able to provide extra information beyond ground-truth labels. To overcome this issue, a commonly-used solution is to raise the temperature of the final softmax function until the cumbersome model produces a suitably soft set of targets.

* https://nervanasystems.github.io/distiller/knowledge_distillation/index.html
* https://github.com/dkozlov/awesome-knowledge-distillation
* https://github.com/lhyfst/knowledge-distillation-papers
* https://pocketflow.github.io/distillation/

<img src=https://pocketflow.github.io/pics/framework_design.png width=80% />

#### Transferred/Compact Convolutional Filters

Transfer learning methods have demonstrated state-of-the-art performance on various small-scale image classification tasks. This is generally achieved by exploiting the information from an ImageNet convolution neural network (ImageNet CNN). However, the transferred CNN model is generally with high computational complexity and storage requirement. It raises the issue for real-world applications, especially for some portable devices like phones and tablets without high-performance GPUs. Several approximation methods have been proposed to reduce the complexity by reconstructing the linear or non-linear filters (responses) in convolutional layers with a series of small ones.

+ https://arxiv.org/pdf/1905.11946v1.pdf
+ [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
+ [Compact Convolutional Neural Network Transfer Learning For Small-Scale Image Classification](https://kar.kent.ac.uk/55053/)

#### Low-rank Approximation


Note that the deep learning models are composite of linear and non-linear maps. And linear maps are based on matrices.

**Singular value decomposition**

The matrix $A_{m\times n}$ can be decomposed as the multiplication of two matrices such as $A_{m\times n}= Q_{m\times r}R_{r\times n}$, so that the storage is from $O(m\times n)$ to $O(m+n)\times O(r)$.

To explore a low-rank subspace combined with a sparse structure for the weight matrix $W$, we assume that $W \approx L+S$,
where $L$ is a low-rank component and $S$ is a sparse matrix. Then, to
compress the weight matrix, we have the following model:
$$\min_{L, S}\frac{1}{2}{\|W-L-S\|}_F^2,\\
s.t.\quad rnak(L) \leq r,\\
card(S)\leq c,
$$
where $rank(L)$ denotes the rank of $L$ and $card(S)$ denotes the cardinality of matrix $S$.

And `Toeplitz Matrix` can be applied to approximate  the  weight matrix
$$
W = {\alpha}_1T_{1}T^{−1}_{2} + {\alpha}_2 T_3 T_{4}^{-1} T_{5}
$$

where ${M}$ is the square weight matrix, $T_1, T_2, T_3, T_4, T_5$ are square *Toeplitz matrix*.

<img title="espnets" src="https://prior.allenai.org/assets/project-content/espnets/esp-unit.jpg" width="80%" />

* https://en.wikipedia.org/wiki/Low-rank_approximation
* [Low Rank Matrix Approximation](http://www.cs.yale.edu/homes/el327/papers/lowRankMatrixApproximation.pdf)
* [On Compressing Deep Models by Low Rank and Sparse Decomposition](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_On_Compressing_Deep_CVPR_2017_paper.pdf)
* [ESPNets for Computer Vision Efficient CNNs for Edge Devices](https://prior.allenai.org/projects/espnet)
* https://www.cnblogs.com/zhonghuasong/p/7821170.html
* https://github.com/chester256/Model-Compression-Papers

|Model Compression|
|---|
|[Distiller](https://nervanasystems.github.io/distiller/index.html)|
|[PocketFlow](https://pocketflow.github.io/)|
|[PocketFlow中的模型压缩算法](https://zhuanlan.zhihu.com/c_1041626714043949056)|
|[PERMDNN: Efficient Compressed DNN Architecture with Permuted Diagonal Matrices](http://alchem.usc.edu/portal/static/download/permdnn.pdf)|
|[knowledge-distillation-pytorch](https://github.com/peterliht/knowledge-distillation-pytorch)|
|[keras_compressor](https://github.com/DwangoMediaVillage/keras_compressor)|
|[TensorFlow Lite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite), [tensorflow-compression](https://tensorflow.github.io/compression/)|

***

* https://srdas.github.io/DLBook/intro.html#effective
* https://cognitiveclass.ai/courses/accelerating-deep-learning-gpu/
* https://github.com/songhan/Deep-Compression-AlexNet
* [awesome-model-compression-and-acceleration](https://github.com/sun254/awesome-model-compression-and-acceleration)
* [gab41.lab41.org](https://gab41.lab41.org/lab41-reading-group-deep-compression-9c36064fb209)
* [CS 598 LAZ: Cutting-Edge Trends in Deep Learning and Recognition](http://slazebni.cs.illinois.edu/spring17/)
* http://slazebni.cs.illinois.edu/spring17/lec06_compression.pdf
* http://slazebni.cs.illinois.edu/spring17/reading_lists.html#lec06
* https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning
* [CS236605: Deep Learning](https://vistalab-technion.github.io/cs236605/lectures/)
* https://mlperf.org/



###  Compressing Recurrent Neural Network

All techniques above can be used to fully-connected networks or generally feed-forward network.
RNN is feedback network where there is rings in its computational graph.

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S1568494619301851-fx1_lrg.jpg" width="80%" />

- [Dynamically Hierarchy Revolution: DirNet for Compressing Recurrent Neural Network on Mobile Devices](https://www.ijcai.org/proceedings/2018/0429.pdf)
- [Compressing Recurrent Neural Network with Tensor Train](https://arxiv.org/pdf/1705.08052.pdf)
- [Learning Compact Recurrent Neural Networks with Block-Term Tensor Decomposition](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ye_Learning_Compact_Recurrent_CVPR_2018_paper.pdf)
- [ANTMAN: SPARSE LOW-RANK COMPRESSION TO ACCELERATE RNN INFERENCE](https://openreview.net/pdf?id=BJgsN3R9Km)
- [Deep Compresion](https://web.stanford.edu/class/ee380/Abstracts/160106-slides.pdf)
- [Run-Time Efficient RNN Compression for Inference on Edge Devices](https://www.groundai.com/project/run-time-efficient-rnn-compression-for-inference-on-edge-devices/1)
- [LSTM compression for language modeling](https://aspirantura.hse.ru/data/2017/05/06/1171468475/2017-04-27-grachev.pdf)
- https://arxiv.org/ftp/arxiv/papers/1612/1612.00891.pdf
- https://ai.google/research/pubs/pub44632
- https://vast.cs.ucla.edu/sites/default/files/publications/CLINK_ISLPED%202018%20publication.pdf


https://github.com/kedartatwawadi/NN_compression
