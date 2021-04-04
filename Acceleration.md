# Network Compression and Acceleration

It is about how to accelerate the training and inference of deep learning(generally machine learning) 
except numerical optimization methods including the following topics:

* compiler optimization for computation intensive programs;
* system architecture design for computation intensive programs;
* network model compression.

<img src="https://cs217.stanford.edu/assets/img/4___JET.gif" width="50%" />

- [Network Compression and Acceleration](#network-compression-and-acceleration)
  - [Resource on ML Sys](#resource-on-ml-sys)
    - [Workshop and Conference](#workshop-and-conference)
    - [Patents and Products](#patents-and-products)
    - [Courses and Labs](#courses-and-labs)
  - [System for Deep Learning](#system-for-deep-learning)
    - [Parallel Architectures and Special Hardware](#parallel-architectures-and-special-hardware)
      - [GPU](#gpu)
      - [TPU](#tpu)
      - [NPU](#npu)
  - [Compilers for Deep Learning](#compilers-for-deep-learning)
    - [TVM and Versatile Tensor Accelerator (VTA)](#tvm-and-versatile-tensor-accelerator-vta)
    - [XLA](#xla)
    - [JAX: Autograd and XLA](#jax-autograd-and-xla)
    - [Multi-Level Intermediate Representation](#multi-level-intermediate-representation)
    - [Glow](#glow)
    - [Cilk](#cilk)
    - [Triton](#triton)
    - [nGraph](#ngraph)
    - [DLR](#dlr)
    - [TASO](#taso)
    - [DLVM](#dlvm)
    - [Jittor](#jittor)
    - [CHET](#chet)
    - [Butterflies: A Universal Building Block for Structured Linear Maps](#butterflies-a-universal-building-block-for-structured-linear-maps)
    - [halide](#halide)
    - [taichi](#taichi)
  - [Numerical algorithms for high-performance computational science](#numerical-algorithms-for-high-performance-computational-science)
    - [Fast Matrix-vector Multiplication](#fast-matrix-vector-multiplication)
    - [Computation of Matrix Chain Products](#computation-of-matrix-chain-products)
    - [Generalized Matrix to Matrix Multiplication](#generalized-matrix-to-matrix-multiplication)
      - [Strassen Algorithms](#strassen-algorithms)
      - [Coppersmith–Winograd Algorithms](#coppersmithwinograd-algorithms)
      - [Linear Algebra Packages](#linear-algebra-packages)
    - [Automatic Differentiation, Differentiable Programming and Program Transformations](#automatic-differentiation-differentiable-programming-and-program-transformations)
      - [Automatic Differentiation](#automatic-differentiation)
      - [Differentiable Programming](#differentiable-programming)
      - [Program Transformations](#program-transformations)
    - [Open Auto-differentiation Library](#open-auto-differentiation-library)
  - [Deep Model Compression](#deep-model-compression)
    - [Fixed-point Arithmetic and Approximate Computing](#fixed-point-arithmetic-and-approximate-computing)
    - [Huffman Encoding](#huffman-encoding)
    - [Knowledge Distillation](#knowledge-distillation)
    - [Parameter Pruning and Sharing](#parameter-pruning-and-sharing)
    - [Quantization and  Fixed-point Arithmetic](#quantization-and--fixed-point-arithmetic)
    - [Low Bit Neural Network](#low-bit-neural-network)
      - [8-bit-training](#8-bit-training)
      - [Binarized Neural Network, Ternary Weight Networks, XOR-Net](#binarized-neural-network-ternary-weight-networks-xor-net)
        - [Binarized Neural Network](#binarized-neural-network)
        - [Ternary Weight Networks](#ternary-weight-networks)
        - [XOR-Net](#xor-net)
      - [Mixed Precision Training](#mixed-precision-training)
      - [AdderNet](#addernet)
      - [Blended Coarse Gradient Descent](#blended-coarse-gradient-descent)
      - [Low-precision Training](#low-precision-training)
        - [High-accuracy Low Precision](#high-accuracy-low-precision)
        - [Ultra-Low Precision Training](#ultra-low-precision-training)
        - [ADMM-NN](#admm-nn)
    - [Transferred/Compact Convolutional Filters](#transferredcompact-convolutional-filters)
    - [Tensor Methods](#tensor-methods)
    - [Compressing Recurrent Neural Network](#compressing-recurrent-neural-network)
    - [Compressing GANs](#compressing-gans)
      - [GAN-pruning](#gan-pruning)
    - [Compressed Transformer](#compressed-transformer)
      - [Compressed BERT](#compressed-bert)
    - [Hashing-accelerated neural networks](#hashing-accelerated-neural-networks)
  - [Distributed Training](#distributed-training)
    - [Accelerating Deep Learning Workloads](#accelerating-deep-learning-workloads)
      - [PipeDream](#pipedream)
      - [AdaptDL](#adaptdl)
    - [Efficient Communication  for Distributed Training](#efficient-communication--for-distributed-training)
      - [DeepSpeed](#deepspeed)
      - [NCCL](#nccl)
    - [Gradient Code and Compression](#gradient-code-and-compression)
      - [Gradient Code and Approximate Gradient Coding](#gradient-code-and-approximate-gradient-coding)
      - [Gradient Compression](#gradient-compression)
      - [Deep Gradient Compression @ MIT](#deep-gradient-compression--mit)
        - [Gradient Compression @ epfl](#gradient-compression--epfl)
        - [Gradient Compression @ Edinburgh](#gradient-compression--edinburgh)
        - [Gradient Compression @ kaust](#gradient-compression--kaust)
      - [Count-Sketches](#count-sketches)
    - [Synthetic gradient](#synthetic-gradient)
    - [Gradient Centralization](#gradient-centralization)
    - [Privacy and Security](#privacy-and-security)
    - [Distributed deep learning libraries](#distributed-deep-learning-libraries)
      - [Deep learning + Spark](#deep-learning--spark)
  - [Products and Packages](#products-and-packages)
    - [TensorFlow](#tensorflow)
    - [PyTorch](#pytorch)
    - [MNN](#mnn)
    - [TNN](#tnn)
    - [PaddlePaddle](#paddlepaddle)
      - [DasyDL](#dasydl)
    - [MindSpore](#mindspore)
      - [ModelArts](#modelarts)
    - [MegEngine](#megengine)
    - [Oneflow](#oneflow)
  - [Edge Computation](#edge-computation)
    - [Mobile Deep Learning](#mobile-deep-learning)
      - [Toolkits](#toolkits)
  - [Tool kits](#tool-kits)

|The World of Neural Network Acceleration|
|:----:|
|Choice of Algorithm|
|Parallelism|
|Distributed Computing|
|Hardware Architectures|

* [Trax — your path to advanced deep learning](https://github.com/google/trax)
* [THE 5TH ANNUAL SCALEDML CONFERENCE](http://scaledml.org/2020/)
* https://www.atlaswang.com/
* https://faculty.ucmerced.edu/mcarreira-perpinan/research/MCCO.html
* https://github.com/1duo/awesome-ai-infrastructures
* https://duvenaud.github.io/learning-to-search/
* http://www.eecs.harvard.edu/htk/publications/
* http://www.eecs.harvard.edu/htk/courses/
* [FairNAS: Rethinking Evaluation Fairness of Weight Sharing Neural Architecture Search](https://github.com/fairnas/FairNAS)
* [VISUAL COMPUTING SYSTEMS](http://graphics.cs.cmu.edu/courses/15769/fall2016/lectures)
* [Tutorial on Hardware Accelerators for Deep Neural Networks](http://eyeriss.mit.edu/tutorial.html)
* [Survey and Benchmarking of Machine Learning Accelerators](https://arxiv.org/abs/1908.11348)
* https://zhuanlan.zhihu.com/jackwish
* https://machinethink.net/blog/compressing-deep-neural-nets/
* [Rethinking Deep Learning: Architectures and Algorithms](https://nickhigham.files.wordpress.com/2019/05/talk12-constantinides.pdf)
* https://girishvarma.in/teaching/efficient-cnns/
* https://github.com/ChanChiChoi/awesome-model-compression
* https://github.com/fengbintu/Neural-Networks-on-Silicon
* https://vast.cs.ucla.edu/
* [Blade Benchmark Suite(BBS)简介](https://help.aliyun.com/document_detail/140558.html)
* https://web.northeastern.edu/yanzhiwang/research/
* http://shivaram.org/#teaching
* https://neuralmagic.com/blog/
* https://github.com/PredictiveModelingMachineLearningLab/MA598
* https://c3dti.ai/
* https://statistics.wharton.upenn.edu/profile/dobriban/
* https://ml-retrospectives.github.io/
* https://data.berkeley.edu/
* https://github.com/mcanini/SysML-reading-list

[To revolutionize deep learning with real-time AI solutions that scale from the edge to the data center](https://wavecomp.ai/).

The parameters of deep neural networks are tremendous. And deep learning is matrix-computation intensive. 
Specific hardware  such as GPU or TPU is used to speed up the computation of deep learning in training or inference.
The optimization methods are used to train the deep neural network.
To boost the training of deep learning, 
we would like to design faster optimization methods such as `ADAM` and delicate architectures of neural network such as `ResNet`.
After training, the parameters of the deep neural network are fixed and used for inference, 
we would do much matrix multiplication via the saved fixed parameters of deep neural network.  
From [What’s the Difference Between Deep Learning Training and Inference?](https://blogs.nvidia.com/blog/2016/08/22/difference-deep-learning-training-inference-ai/)

<img src="https://blogs.nvidia.com/wp-content/uploads/2016/08/ai_difference_between_deep_learning_training_inference.jpg" width="80%">

Training | Inference
---|---
Acceleration | Compression
https://web.stanford.edu/~perdavan/DNNTrain/|https://www.intel.ai/accelerating-tensorflow-inference-with-intel-deep-learning-boost-on-2nd-gen-intel-xeon-scalable-processors/
[Tutorial on Hardware Accelerators for Deep Neural Networks](http://eyeriss.mit.edu/tutorial.html)|[Accelerating Large Scale Deep Learning Inference through DeepCPU at Microsoft](https://www.usenix.org/system/files/opml19papers-zhang.pdf)

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
* [Accelerating Deep Learning Inference via Freezing](https://www.usenix.org/conference/hotcloud19/presentation/kumar)
* [模型压缩之deep compression](https://littletomatodonkey.github.io/2018/10/10/2018-10-10-%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E4%B9%8Bdeep%20compression/)
* [论文笔记《A Survey of Model Compression and Acceleration for Deep Neural Networks》](https://blog.csdn.net/song_pipi/article/details/79154539)
* https://zhuanlan.zhihu.com/p/67508423
* [Network Speed and Compression](https://github.com/mrgloom/Network-Speed-and-Compression)
* [Model Compression and Acceleration](https://www.jiqizhixin.com/articles/2018-05-18-4)
* [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)
* [Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://arxiv.org/abs/1712.01887)
* [AutoML 的十大开源库](https://www.52cs.com/archives/3138)
* [TensorFlow模型压缩和Inference加速](https://zhuanlan.zhihu.com/p/31023153)
* https://zhuanlan.zhihu.com/DNN-on-Chip
+ https://github.com/BlueWinters/research
+ [Reference workloads for modern deep learning methods](https://github.com/rdadolf/fathom)
+ [How to Train for and Run Machine Learning Models on Edge Devices](https://www.simonwenkel.com/2018/12/08/How-to-run-Deep-Learning-Models-on-Edge-devices.html)
+ http://mvapich.cse.ohio-state.edu/
+ http://people.eecs.berkeley.edu/~reddy/


## Resource on ML Sys

- https://www.microsoft.com/en-us/research/project/astra-custom-wired-dnns/

### Workshop and Conference

* http://www.sysml.cc/
* https://mlperf.org/
* https://sosp19.rcs.uwaterloo.ca/program.html
* http://learningsys.org/sosp19/schedule.html
* https://cs.stanford.edu/people/zhihao/
* [Workshop on AI Systems](http://learningsys.org/sosp19/)
* [Systems for ML](http://learningsys.org/neurips19/)
* [Workshop on ML for Systems at NeurIPS 2019](http://mlforsystems.org/)
* http://learningsys.org/nips18/
* http://learningsys.org/sosp17/
* https://sites.google.com/site/mlsys2016/
* [Workshop on Systems for ML and Open Source Software at NeurIPS 2018](http://learningsys.org/nips18/schedule.html)
* [Computer Systems Colloquium (EE380) Schedule](https://web.stanford.edu/class/ee380/)
* [ML Benchmarking Tutorial](https://www.sigarch.org/call-participation/ml-benchmarking-tutorial/)
* [The ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE)](https://2018.fseconference.org/home)
* [The fastest path to machine learning integration](https://intel.github.io/dffml/)
* https://www.sigarch.org/call-participation/ml-benchmarking-tutorial/
* [DNNBuilder: an Automated Tool for Building High-Performance DNN Hardware Accelerators for FPGAs](https://www.c3sr.com/publication/2018/iccad_dnnbuilder/)
* [ISCA 2016 in Seoul](http://isca2016.eecs.umich.edu/)
* [Acceleration of Deep Learning for Cloud and Edge Computing@UCLA](https://vast.cs.ucla.edu/projects/acceleration-deep-learning-cloud-and-edge-computing)
* [Hot Chips: A Symposium on High Performance Chips](http://hotchips.org)
* https://paco-cpu.github.io/paco-cpu/
* [Programmable Inference Accelerator](https://developer.nvidia.com/tensorrt)
* [Fair and useful benchmarks for measuring training and inference performance of ML hardware, software, and services.](https://mlperf.org/)
* https://dlonsc19.github.io/



### Patents and Products

* https://www.nextplatform.com/
* https://iq.opengenus.org/neural-processing-unit-npu/
* https://www.csail.mit.edu/event/domain-specific-accelerators
* https://www.alphaics.ai/
* http://www.cambricon.com/
* https://www.sigarch.org/
* https://www.xilinx.com/
* https://wavecomp.ai/
* https://www.graphcore.ai/
* https://www.wikiwand.com/en/Hardware_acceleration
* https://en.wikichip.org/wiki/WikiChip
* https://patents.google.com/patent/US8655815B2/en
* https://www.intel.ai/blog/
* https://www.arm.com/solutions/artificial-intelligence
* https://mlperf.org/index.html#companies
* [Accelerating Deep Learning with Memcomputing](https://arxiv.org/pdf/1801.00512.pdf)
* http://maxpumperla.com/
* http://accelergy.mit.edu/
* https://eng.uber.com/sbnet-sparse-block-networks-convolutional-neural-networks/

### Courses and Labs

* [Modern Numerical Computing](https://courses.csail.mit.edu/18.337/2018/)
* [Papers Reading List of *Embedded Neural Network*](https://github.com/ZhishengWang/Embedded-Neural-Network)
* [Deep Compression and EIE](https://web.stanford.edu/class/ee380/Abstracts/160106-slides.pdf)
* [Programmable Hardware Accelerators (Winter 2019)](https://cmpe293-winter19-01.courses.soe.ucsc.edu/home)
* [Hardware Accelerators for Training Deep Neural Networks ](https://web.stanford.edu/~perdavan/DNNTrain/)
* [Illinois Microarchitecture Project using Algorithms and Compiler Technology](http://impact.crhc.illinois.edu/default.aspx)
* [Deep Learning for Computer Architects](https://www.morganclaypool.com/doi/abs/10.2200/S00783ED1V01Y201706CAC041)
* [System for Machine Learning @.washington.edu/](https://dlsys.cs.washington.edu/)
* [Hanlab: ACCELERATED DEEP LEARNING COMPUTING Hardware, AI and Neural-nets](https://hanlab.mit.edu/)
* [Bingsheng He's publication on GPU](https://www.comp.nus.edu.sg/~hebs/publication.html)
* [Architecture Lab for Creative High-performance Energy-efficient Machines](http://alchem.usc.edu/portal/machinelearning.html)
* [HIGH PERFORMANCE POWER EFFICIENT NEURAL NETWORK IMPLEMENTATIONS ON EMBEDDED DEVICES](https://events.vtools.ieee.org/m/211223)
* https://eiclab.net/
* https://parsa.epfl.ch/~falsafi/
* [swDNN: A Library for Accelerating Deep Learning Applications on Sunway TaihuLight Supercomputer](https://github.com/THUHPGC/swDNN)
* [ High Performance Distributed Computing (HPDC](https://www.vuhpdc.net/)
* https://readingxtra.github.io/
* https://nextcenter.org/
* http://yanjoy.win/
* https://dlonsc19.github.io/
* http://hibd.cse.ohio-state.edu/
* http://www.federated-ml.org/
* https://dai.lids.mit.edu/research/publications/
* https://aiforgood.itu.int/
* https://www.linayao.com/publications/
* https://www.deepstack.ai/
* https://pvs.ifi.uni-heidelberg.de/home

## System for Deep Learning

Over the past few years, deep learning has become an important technique to successfully solve problems in many different fields, such as vision, NLP, robotics. 
An important ingredient that is driving this success is the development of deep learning systems that efficiently support the task of learning and inference of complicated models using many devices and possibly using distributed resources. 
The study of how to build and optimize these deep learning systems is now an active area of research and commercialization.

Matrix computation dense application like deep neural network would take the advantages of specific architecture design. 
Thus it is really close to `high performance computational science` when solving some computation dense problems.

<img src="https://pooyanjamshidi.github.io/mls/_images/mls-logo.jpg" width="69%" />


<img src="https://www.researchgate.net/profile/Gu_Yeon_Wei/publication/306398249/figure/fig2/AS:614016141512719@1523404264555/Breakdown-of-execution-time-by-operation-type-for-each-Fathom-workload.png" width="80%" />


* [BENCHMARKING DEEP LEARNING SYSTEMS](https://sites.google.com/g.harvard.edu/mlperf-bench/home)
* [Facebook AI Performance Evaluation Platform](https://github.com/facebook/FAI-PEP)
* [Hardware Accelerators for Machine Learning (CS 217) Stanford University, Fall 2018](https://cs217.stanford.edu/readings)
* [CSCE 790/590: Machine Learning Systems](https://github.com/pooyanjamshidi/mls)
* [DeepDream: Accelerating Deep Learning With Hardware](https://www.svds.com/deepdream-accelerating-deep-learning-hardware/)
* http://ece-research.unm.edu/jimp/codesign/
* http://learningsys.org/sosp19/
* https://determined.ai/
* https://web.stanford.edu/~rezab/

### Parallel Architectures and Special Hardware

Parallel Architectures for Parallel Processing as co-design is a subfield of system for machine learning.

<img src="https://vistalab-technion.github.io/cs236605/assets/images/lec10/lec10-2.png" width="80%" />

- https://vistalab-technion.github.io/cs236605/lectures/lecture_8/
- https://vistalab-technion.github.io/cs236605/lectures/lecture_9/
- https://vistalab-technion.github.io/cs236605/lectures/lecture_10/
- https://iq.opengenus.org/gpu-vs-tpu-vs-fpga/
- [Introduction to Parallel Computing Author: Blaise Barney, Lawrence Livermore National Laboratory](https://computing.llnl.gov/tutorials/parallel_comp/)
- [Parallel Architectures for Artificial Neural Networks: Paradigms and Implementations N. Sundararajan, P. Saratchandran](https://www.wiley.com/WileyCDA/WileyTitle/productCd-0818683996,miniSiteCd-IEEE_CS2.html)
- [Parallel Computer Architecture and Programming (CMU 15-418/618)](http://www.math-cs.gordon.edu/courses/cps343/)
- https://twimlcon.com/
- [Accelerating Deep Learning with a Parallel Mechanism Using CPU + MIC](https://link.springer.com/article/10.1007%2Fs10766-017-0535-9)
- [Papers on Big Data Meets New Hardware](https://readingxtra.github.io/)
* [LUTNet: Rethinking Inference in FPGA Soft Logic](https://arxiv.org/abs/1904.00938)
#### GPU

This GPU architecture works well on applications with massive parallelism, such as matrix multiplication in a neural network.
 Actually, you would see order of magnitude higher throughput than CPU on typical training workload for deep learning. 
 This is why the GPU is the most popular processor architecture used in deep learning at time of writing.

But, the GPU is still a general purpose processor that has to support millions of different applications and software. 
This leads back to our fundamental problem, the von Neumann bottleneck. 
For every single calculation in the thousands of ALUs, GPU need to access registers or shared memory to read and store the intermediate calculation results. 
Because the GPU performs more parallel calculations on its thousands of ALUs, [it also spends proportionally more energy accessing memory and also increases footprint of GPU for complex wiring.](https://cloud.google.com/blog/products/ai-machine-learning/what-makes-tpus-fine-tuned-for-deep-learning)

<img src="https://storage.googleapis.com/gweb-cloudblog-publish/original_images/image2.gif" width="70%"/>

* [GPU，CUDA，cuDNN的理解](https://blog.csdn.net/u014380165/article/details/77340765)
* https://developer.nvidia.com/cuda-zone
* https://arxiv.org/pdf/1410.0759.pdf

#### TPU

TPUs can't run word processors, control rocket engines, or execute bank transactions, 
but they can handle the massive multiplications and additions for neural networks, at blazingly fast speeds while consuming much less power and inside a smaller physical footprint.

The key enabler is a major reduction of the von Neumann bottleneck. 
Because the primary task for this processor is matrix processing, hardware designer of the TPU knew every calculation step to perform that operation. 
So they were able to place thousands of multipliers and adders and connect them to each other directly to form a large physical matrix of those operators. 
This is called systolic array architecture.

<img src="https://deliveryimages.acm.org/10.1145/3160000/3154484/f2.jpg" width="70%"/>

<img src="https://storage.googleapis.com/gweb-cloudblog-publish/original_images/image4_5PFB45w.gif" width="50%"/>
<img src="https://storage.googleapis.com/gweb-cloudblog-publish/original_images/image1_2PdcvlE.gif"  width="50%"/>

* [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8192463)
* https://www.mlq.ai/tpu-machine-learning/
* [EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML](https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html)
* [An in-depth look at Google’s first Tensor Processing Unit (TPU)](https://cloud.google.com/blog/products/gcp/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu)
* [A Domain-Specific Architecture for Deep Neural Networks](https://cacm.acm.org/magazines/2018/9/230571-a-domain-specific-architecture-for-deep-neural-networks/fulltext)

#### NPU

A neural processing unit (NPU) is a microprocessor that specializes in the acceleration of machine learning algorithms, 
typically by operating on predictive models such as artificial neural networks (ANNs) or random forests (RFs). 
[It is, also, known as neural processor.](https://iq.opengenus.org/neural-processing-unit-npu/)

NPU are required for the following purpose:

1. Accelerate the computation of Machine Learning tasks by several folds (nearly 10K times) as compared to GPUs
2. Consume low power and improve resource utilization for Machine Learning tasks as compared to GPUs and CPUs




## Compilers for Deep Learning

* https://aiotworkshop.github.io/2020/program.html
* https://www.incose.org/
* [Compilers by SOE-YCSCS1 STANFORD SCHOOL OF ENGINEERING](https://online.stanford.edu/courses/soe-ycscs1-compilers)
* [EIE: Efficient Inference Engine on Compressed Deep Neural Network](https://arxiv.org/pdf/1602.01528.pdf)
* [A modern compiler infrastructure for deep learning systems with adjoint code generation in a domain-specific IR](http://dowobeha.github.io/papers/autodiff17.pdf)
* [TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](https://arxiv.org/abs/1603.04467)
* [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://papers.nips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf)
* https://ai-techsystems.com/dnn-compiler/

### TVM and Versatile Tensor Accelerator (VTA)

> TVM is an open deep learning compiler stack for CPUs, GPUs, and specialized accelerators. It aims to close the gap between the productivity-focused deep learning frameworks, and the performance- or efficiency-oriented hardware backends. 
> TVM provides the following main features:

> Compilation of deep learning models in Keras, MXNet, PyTorch, Tensorflow, CoreML, DarkNet into minimum deployable modules on diverse hardware backends.
Infrastructure to automatic generate and optimize tensor operators on more backend with better performance.

<img src="https://raw.githubusercontent.com/tvmai/tvmai.github.io/master/images/main/stack_tvmlang.png" width= "80%" />


The Versatile Tensor Accelerator (VTA) is an extension of the TVM framework designed to advance deep learning and hardware innovation. 
VTA is a programmable accelerator that exposes a RISC-like programming abstraction to describe compute and memory operations at the tensor level. 
We designed VTA to expose the most salient and common characteristics of mainstream deep learning accelerators, such as tensor operations, DMA load/stores, and explicit compute/memory arbitration.


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


### XLA

The XLA compilation framework is invoked on subgraphs of TensorFlow computations. 
The framework requires all tensor shapes to be fixed, so compiled code is specialized to concrete shapes. 
This means, for example, that the compiler may be invoked multiple times for the same subgraph if it is executed on batches of different sizes.

- https://www.tensorflow.org/versions/master/experimental/xla/
- https://developers.googleblog.com/2017/03/xla-tensorflow-compiled.html
- https://www.tensorflow.org/xla/overview
- https://autodiff-workshop.github.io/slides/JeffDean.pdf
- [XLA: The TensorFlow compiler framework](https://haosdent.gitbooks.io/tensorflow-document/content/resources/xla_prerelease.html)

### JAX: Autograd and XLA

With its updated version of Autograd, JAX can automatically differentiate native Python and NumPy functions.
It can differentiate through loops, branches, recursion, and closures, and it can take derivatives of derivatives of derivatives. 
It supports reverse-mode differentiation (a.k.a. backpropagation) via grad as well as forward-mode differentiation,
 and the two can be composed arbitrarily to any order.

<img src="https://raw.githubusercontent.com/google/jax/master/images/lifecycle.png" width="60%" />

- https://github.com/google/jax
- https://github.com/hips/autograd

### Multi-Level Intermediate Representation

The `Multi-Level Intermediate Representation (MLIR)` is intended for easy expression and optimization of computations involving deep loop nests and dense matrices of high dimensionality. 
It is thus well-suited to deep learning computations in particular. 
Yet it is general enough to also represent arbitrary sequential computation. 
The representation allows high-level optimization and parallelization for a wide range of parallel architectures including those with deep memory hierarchies --- general-purpose multicores, GPUs, and specialized neural network accelerators.

- https://github.com/tensorflow/mlir
- https://llvm.org/devmtg/2019-04/slides/Keynote-ShpeismanLattner-MLIR.pdf

### Glow

`Glow` is a machine learning compiler and execution engine for hardware accelerators. 
It is designed to be used as a backend for high-level machine learning frameworks. 
The compiler is designed to allow state of the art compiler optimizations and code generation of neural network graphs. 
This library is in active development.

- https://arxiv.org/pdf/1805.00907.pdf
- https://ai.facebook.com/tools/glow/
- https://github.com/pytorch/glow


### Cilk

[Cilk aims to make parallel programming a simple extension of ordinary serial programming.](http://cilk.mit.edu/)
Other concurrency platforms, such as Intel’s Threading Building Blocks (TBB) and OpenMP, share similar goals of making parallel programming easier. 
But Cilk sets itself apart from other concurrency platforms through its simple design and implementation and its powerful suite of provably effective tools. 
These properties make Cilk well suited as a platform for next-generation multicore research.
[Tapir enables effective compiler optimization of parallel programs with only minor changes to existing compiler analyses and code transformations.](https://www.csail.mit.edu/event/tapir-embedding-recursive-fork-join-parallelism-llvms-intermediate-representation)
Tapir uses the serial-projection property to order logically parallel fine-grained tasks in the program's control-flow graph. 
This ordered representation of parallel tasks allows the compiler to optimize parallel codes effectively with only minor modifications.
- http://cilk.mit.edu/tapir/
- https://llvm.org/
- https://zhuanlan.zhihu.com/p/64903359

### Triton

The aim of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA, but also with higher flexibility than other existing DSLs.

- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- https://github.com/ptillet/triton
- https://pldi19.sigplan.org/details/mapl-2019-papers/1/Triton-An-Intermediate-Language-and-Compiler-for-Tiled-Neural-Network-Computations
- http://www.federated-ml.org/tutorials/globecom2020/part4.pdf


### nGraph

`nGraph` is an end to end deep learning compiler for inference and training with extensive framework and hardware support.

<img src="https://www.ngraph.ai/sites/default/files/2019-08/main_diagram_fw_hw.png" width="70%" />

+ https://www.ngraph.ai/
+ https://github.com/NervanaSystems/ngraph
+ https://github.com/plaidml/plaidml
- https://en.wikipedia.org/wiki/Zeroth_(software)


### DLR

DLR is a compact, common runtime for deep learning models and decision tree models compiled by AWS SageMaker Neo, TVM, or Treelite. DLR uses the TVM runtime, Treelite runtime, NVIDIA TensorRT™, and can include other hardware-specific runtimes. DLR provides unified Python/C++ APIs for loading and running compiled models on various devices. DLR currently supports platforms from Intel, NVIDIA, and ARM, with support for Xilinx, Cadence, and Qualcomm coming soon.

- https://github.com/neo-ai/neo-ai-dlr
- https://github.com/dmlc/treelite
- http://treelite.io/
- https://aws.amazon.com/cn/sagemaker/neo/

### TASO

TASO optimizes the computation graphs of DNN models using automatically generated and verified graph transformations. For an arbitrary DNN model, TASO uses the auto-generated graph transformations to build a large search space of potential computation graphs that are equivalent to the original DNN model. TASO employs a cost-based search algorithm to explore the space, and automatically discovers highly optimized computation graphs.

* https://cs.stanford.edu/people/zhihao/
* https://github.com/jiazhihao/TASO
* http://theory.stanford.edu/~aiken/

### DLVM

We present DLVM, a design and implementation of a compiler infrastructure with a linear algebra intermediate representation, algorithmic differentiation by adjoint code generation, domain-specific optimizations, and a code generator targeting GPU via LLVM.

- [DLVM: A MODERN COMPILER INFRASTRUCTURE FOR DEEP LEARNING SYSTEMS](https://arxiv.org/pdf/1711.03016.pdf)
- https://github.com/dlvm-team
- https://github.com/Jittor/jittor
- http://dlvm.org/
- http://dowobeha.github.io/papers/autodiff17.pdf
- http://dowobeha.github.io/
- https://roshandathathri.github.io/
- https://www.clsp.jhu.edu/workshops/19-workshop/

### Jittor

Jittor is a high-performance deep learning framework based on JIT compiling and meta-operators. The whole framework and meta-operators are compiled just-in-time. A powerful op compiler and tuner are integrated into Jittor. It allowed us to generate high-performance code with specialized for your model. Jittor also contains a wealth of high-performance model libraries, including: image recognition, detection, segmentation, generation, differentiable rendering, geometric learning, reinforcement learning, etc.。

- https://github.com/Jittor
- https://cg.cs.tsinghua.edu.cn/jittor/about/

### CHET


> CHET is a domain-specific optimizing compiler designed to make the task of programming FHE applications easier. Motivated by the need to perform neural network inference on encrypted medical and financial data, CHET supports a domain-specific language for specifying tensor circuits. It automates many of the laborious and error prone tasks of encoding such circuits homomorphically, including encryption parameter selection to guarantee security and accuracy of the computation, determining efficient tensor layouts, and performing scheme-specific optimizations.


- https://roshandathathri.github.io/publication/2019-pldi
- [CHET: An Optimizing Compiler for Fully-Homomorphic Neural-Network Inferencing](https://www.cs.utexas.edu/~roshan/CHET.pdf)
- https://www.cs.utexas.edu/~pingali/
- https://roshandathathri.github.io/
- https://iss.oden.utexas.edu/

------------


### Butterflies: A Universal Building Block for Structured Linear Maps

Fast linear transforms are ubiquitous in machine learning, including the discrete Fourier transform, discrete cosine transform, 
and other structured transformations such as convolutions. 
All of these transforms can be represented by dense matrix-vector multiplication, yet each has a specialized and highly efficient (subquadratic) algorithm. 
We ask to what extent hand-crafting these algorithms and implementations is necessary, what structural priors they encode, 
and how much knowledge is required to automatically learn a fast algorithm for a provided structured transform. 
Motivated by a characterization of `fast matrix-vector multiplication` as products of sparse matrices, 
we introduce a parameterization of divide-and-conquer methods that is capable of representing a large class of transforms. 
This generic formulation can automatically learn an efficient algorithm for many important transforms; 
for example, it recovers the $O(N\log N)$ Cooley-Tukey FFT algorithm to machine precision, for dimensions N up to 1024. 
Furthermore, our method can be incorporated as a lightweight replacement of generic matrices in machine learning pipelines 
to learn efficient and compressible transformations. 
On a standard task of compressing a single hidden-layer network, 
our method exceeds the classification accuracy of unconstrained matrices on CIFAR-10 by 3.9 points
[---the first time a structured approach has done so---with 4X faster inference speed and 40X fewer parameters.](https://arxiv.org/abs/1903.05895)

+ [Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations](https://arxiv.org/abs/1903.05895)
+ [ Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations](https://github.com/HazyResearch/learning-circuits/)
+ [Butterflies Are All You Need: A Universal Building Block for Structured Linear Maps](https://dawn.cs.stanford.edu/2019/06/13/butterfly/)
- https://github.com/stanford-futuredata/Willump
- [A Two Pronged Progress in Structured Dense Matrix Multiplication](https://arxiv.org/abs/1611.01569)

### halide

Halide is a programming language designed to make it easier to write high-performance image and array processing code on modern machines. 

- https://halide-lang.org/
- https://halide-lang.org/#publications
- https://people.csail.mit.edu/tzumao/

### taichi

Taichi (太极) is a programming language designed for high-performance computer graphics. It is deeply embedded in Python, and its just-in-time compiler offloads compute-intensive tasks to multi-core CPUs and massively parallel GPUs

- http://taichi.graphics/
- https://github.com/yuanming-hu/taichi
- http://taichi.graphics/wp-content/uploads/2019/09/taichi_lang.pdf\

## Numerical algorithms for high-performance computational science

Several key themes emerged across multiple talks in [Royal Society Discussion Meeting](https://constantinides.net/2019/04/23/royal-society-discussion-meeting/), all in the context of today’s high performance computing landscape in which processor clock speeds have stagnated (with the end of Moore’s law) and exascale machine are just two or three years away.

* An important way of accelerating computations is through the use of `low precision floating-point arithmetic`—in particular by exploiting a hierarchy of precisions.
* We must exploit `low rank matrix structure` where it exists, for example in hierarchical (H-matrix) form, combining it with randomized approximations.
* Minimizing `data movement (communication)` is crucial, because of its increasing costs relative to the costs of floating-point arithmetic.
* `Co-design` (the collaborative and concurrent development of hardware, software, and numerical algorithms, with knowledge of applications) is increasingly important for numerical computing.

For more on high performance computation on GPU see [https://hgpu.org/](https://hgpu.org/).



+ https://hgpu.org/
+ [Numerical Algorithms for High-Performance Computational Science: Highlights of the Meeting](https://nickhigham.wordpress.com/2019/05/07/numerical-algorithms-for-high-performance-computational-science-highlights/)
+ [Numerical algorithms for high-performance computational science](https://royalsociety.org/science-events-and-lectures/2019/04/high-performance-computing/)
+ [Reflections on the Royal Society’s “Numerical Algorithms for High-performance Computational Science” Discussion Meeting](https://sinews.siam.org/Details-Page/reflections-on-the-royal-societys-numerical-algorithms-for-high-performance-computational-science-discussion-meeting)
+ [Overview of Microsoft HPC Pack 2016](https://docs.microsoft.com/zh-cn/powershell/high-performance-computing/overview?view=hpc16-ps)
+ [Document Library: High Performance Computing Fabrics](https://www.intel.com/content/www/us/en/high-performance-computing-fabrics/library.html)
+ [MVAPICH: MPI over InfiniBand, Omni-Path, Ethernet/iWARP, and RoCE](http://mvapich.cse.ohio-state.edu/)
+ https://researchcomputing.lehigh.edu/
+ https://library.columbia.edu/libraries/dsc/hpc.html
+ [Open MPI: Open Source High Performance Computing](https://www.open-mpi.org/)
+ https://ltsnews.lehigh.edu/node/115
+ [NumFOCUS](https://numfocus.org/sponsored-projects)
+ http://www.mit.edu/~kepner/D4M/
+ [CSCS-ICS-DADSi Summer School: Accelerating Data Science with HPC, September 4 – 6, 2017 ](https://github.com/probprog/CSCS-summer-school-2017)
+ [MS&E 317: Algorithms for Modern Data Models: Spring 2014, Stanford University ](https://stanford.edu/~rezab/amdm/)
+ [Distributed Machine Learning and Matrix Computations: A NIPS 2014 Workshop](https://stanford.edu/~rezab/nips2014workshop/index.html)
+ [Large Scale Matrix Analysis and Inference: A NIPS 2013 Workshop](http://stanford.edu/~rezab/nips2013workshop/)
+ [Breakthrough! Faster Matrix Multiply](https://www.i-programmer.info/news/112-theory/3453-breakthrough-faster-matrix-multiply.html)
+ [Performance Engineering of Software Systems](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-172-performance-engineering-of-software-systems-fall-2018/)
+ https://zhuanlan.zhihu.com/p/94653447

[Why GEMM is at the heart of deep learning](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)

<img src="https://yongyuan.name/imgs/posts/gemm_cup_gpu.png" width="70%">

[General Matrix Multiply (GEMM) is a common algorithm in linear algebra, machine learning, statistics, and many other domains.](https://spatial-lang.org/gemm)

- https://github.com/pytorch/QNNPACK
- https://github.com/XiaoMi/mace

###  Fast Matrix-vector Multiplication

Matrix-vector multiplication is a special matrix multiplication:
$$\mathbb R^m\mapsto \mathbb R^n: Mv\to u \\
u=Mv=\sum_{i=1}A^{(i)}v_i$$
where $M\in\mathbb R^{m\times n}, u\in\mathbb R^n$; each column $M^{(i)}$
can, metaphorically, indicate one address or house and each $v(i)$ a letter addressed to it.

- https://people.csail.mit.edu/rrw/
- [Matrix-Vector Multiplication in Sub-Quadratic Time (Some Preprocessing Required)](https://people.csail.mit.edu/rrw/mat-vec3.pdf)
- [The Mailman algorithm for matrix vector multiplication](http://www.cs.yale.edu/homes/el327/papers/matrixVectorApp.pdf)
- [Matrix-vector multiplication using the FFT](http://math.mit.edu/icg/resources/teaching/18.085-spring2015/toeplitz.pdf)
- [On fast matrix-vector multiplication with a Hankel matrix in multiprecision arithmetics](https://arxiv.org/pdf/1402.5287.pdf)
- [Optimizing Large Matrix-Vector Multiplications](https://simulationcorner.net/index.php?page=fastmatrixvector)
- [Faster Matrix vector multiplication ON GeForce](https://www.nvidia.com/docs/IO/47905/fujimoto_lspp2008.pdf)
- [Fast Implementation of General Matrix-Vector Multiplication (GEMV) on Kepler GPUs](https://ieeexplore.ieee.org/document/7092787)
- [FAST ALGORITHMS TO COMPUTE MATRIX-VECTOR PRODUCTS FOR PASCAL MATRICES](http://users.umiacs.umd.edu/~ramani/pubs/tang_pascal_updated.pdf)
- [Fast algorithms to compute matrix-vector products for Toeplitz and Hankel matrices](http://pe.org.pl/articles/2012/8/47.pdf)
- [Fast High Dimensional Vector Multiplication Face Recognition](http://openaccess.thecvf.com/content_iccv_2013/papers/Barkan_Fast_High_Dimensional_2013_ICCV_paper.pdf)
- [Faster Online Matrix-Vector Multiplication](https://cs.au.dk/~larsen/papers/omv.pdf)
- [A fast matrix–vector multiplication method for solving the radiosity equation](http://homepage.math.uiowa.edu/~atkinson/ftp/atkchien_fast.pdf)
- [Computational Science and Engineering Spring 2015 syllabus](http://math.mit.edu/icg/resources/teaching/18.085-spring2015/)

### Computation of Matrix Chain Products

Generations of students have learned that the product $xy^Tz$, where $x, y,$ and $z$ are n-vectors, should be written and evaluated as $x(y^Tz)$ ($O(n)$ flops) rather than $(xy^T)z$ ($O(n^2)$) flops). 
More generally, deciding where to put the parentheses in a matrix product $A_1A_2\dots A_k$ to minimize the number of operations in the evaluation is a nontrivial problem, known as the `matrix chain multiplication problem`.

A special case is when $A_1=A_2=\dots =A_k$ the problem is to compute the $A^k=\underbrace{A\cdots A}_{k}$.

- [The World’s Most Fundamental Matrix Equation](https://sinews.siam.org/Details-Page/the-worlds-most-fundamental-matrix-equation)
- [Computation of Matrix Chain Products, PART I, PART II](http://i.stanford.edu/pub/cstr/reports/cs/tr/81/875/CS-TR-81-875.pdf)
- [CS3343/3341 Analysis of Algorithms  Matrix-chain  Multiplications](http://www.cs.utsa.edu/~wagner/CS3343/dp/mat.html)
- [rosettacode: Matrix chain multiplication](https://rosettacode.org/wiki/Matrix_chain_multiplication)
- http://www.columbia.edu/~cs2035/courses/csor4231.F11/matrix-chain.pdf
- https://www.geeksforgeeks.org/matrix-chain-multiplication-dp-8/
- https://www.geeksforgeeks.org/matrix-chain-multiplication-a-on2-solution/
- https://home.cse.ust.hk/~dekai/271/notes/L12/L12.pdf


### Generalized Matrix to Matrix Multiplication

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
<img src="https://jackwish.net/images/2019/gemm-opt/gemm-1x4.svg" width="60%" />

It  needs $O(MKN)$ multiplication.

The picture below visualizes the computation of a single element in the result matrix $C$. 
Each element in the result matrix $C$ is the sum of element-wise multiplication of a row from $A$ and a column from $B$.

<img src="https://gist.githubusercontent.com/nadavrot/5b35d44e8ba3dd718e595e40184d03f0/raw/23dc2fdf78e88ef7fa2f00028bb735ee70429d6d/zsimple.png" width="60%">

Our program is memory bound, which means that the multipliers are not active most of the time because they are waiting for memory.

* [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/users/pingali/CS378/2008sp/papers/gotoPaper.pdf)
* [Geometry and the complexity of matrix multiplication](https://www.ams.org/journals/bull/2008-45-02/S0273-0979-08-01176-2/home.html)
* [High-Performance Matrix Multiplication](https://gist.github.com/nadavrot/5b35d44e8ba3dd718e595e40184d03f0)
* [Fast Matrix Multiplication @mathoverflow](https://mathoverflow.net/questions/34173/fast-matrix-multiplication)
* [Powers of Tensors and Fast Matrix Multiplication](https://simons.berkeley.edu/sites/default/files/docs/2438/slideslegall.pdf)
* https://www.kkhaydarov.com/matrix-multiplication-algorithms/
* [BLISlab: A Sandbox for Optimizing GEMM](https://github.com/flame/blislab)
* [MAGMA: Matrix Algebra for GPU and Multicore Architectures](https://icl.cs.utk.edu/projectsfiles/magma/doxygen/index.html)
* http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/gemm/
* [通用矩阵乘和卷积优化](https://jackwish.net/gemm-optimization-and-convolution.html)
* [Fast Matrix Multiplication Algorithms](https://www.ics.uci.edu/~fastmm/)
* [Anatomy of high-performance matrix multiplication](https://dl.acm.org/citation.cfm?id=1356053)
* [The Indirect Convolution Algorithm](https://arxiv.org/abs/1907.02129)
* https://github.com/flame/how-to-optimize-gemm/wiki
* https://en.wikipedia.org/wiki/Matrix_multiplication#Complexity
* [Matrix multiplication via arithmetic progressions](https://www.sciencedirect.com/science/article/pii/S0747717108800132)
* [Fast sparse matrix multiplication](http://www.cs.tau.ac.il/~zwick/papers/sparse.pdf)
* [Part I: Performance of Matrix multiplication in Python, Java and C++](https://martin-thoma.com/matrix-multiplication-python-java-cpp/)
* [Part III: Matrix multiplication on multiple cores in Python, Java and C++](https://martin-thoma.com/part-iii-matrix-multiplication-on-multiple-cores-in-python-java-and-c/)
* https://github.com/MartinThoma/matrix-multiplication
* http://jianyuhuang.com/
* [SPATIAL: A high-level language for programming accelerators](https://spatial-lang.org/gemm)
* https://github.com/OAID/Tengine
* [Performance of Classic Matrix Multiplication Algorithm on Intel® Xeon Phi™ Processor System](https://software.intel.com/en-us/articles/performance-of-classic-matrix-multiplication-algorithm-on-intel-xeon-phi-processor-system)
* [Low-precision matrix multiplication](https://github.com/google/gemmlowp)

#### Strassen Algorithms

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

- [Strassen’s Matrix Multiplication Algorithm | Implementation](https://www.geeksforgeeks.org/strassens-matrix-multiplication-algorithm-implementation/)
- [Part II: The Strassen algorithm in Python, Java and C++](https://martin-thoma.com/strassen-algorithm-in-python-java-cpp/)
- [Using Strassen’s Algorithm to Accelerate the Solution of Linear System](https://www.davidhbailey.com/dhbpapers/strassen.pdf)
- https://shivathudi.github.io/jekyll/update/2017/06/15/matr-mult.html
- http://jianyuhuang.com/papers/sc16.pdf
- [Comparative Study of Strassen’s Matrix Multiplication Algorithm](http://www.ijcst.com/vol31/4/juby.pdf)
- http://andrew.gibiansky.com/blog/mathematics/matrix-multiplication/
- http://jianyuhuang.com/

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

[We use tensors (and their low-rank decompositions) to multiply matrices faster than $O(n^3)$.](https://dustingmixon.wordpress.com/2014/07/17/introduction-to-fast-matrix-multiplication/)
Define the matrix multiplication tensor as follows:
$$
M_{(a, b), (c, d), (e, f)}^{(n)}
=\begin{cases}
1, &\text{if $b=c, d=e, f=a$}, \\
0, &\text{otherwise}.
\end{cases}
$$
Suppose $\mathrm{rank}(M^{(n)})\leq r$, i.e.,

$$\displaystyle{M_{(a,b),(c,d),(e,f)}^{(n)}=\sum_{\ell=1}^r x_{ab}^\ell y_{cd}^\ell z_{ef}^\ell.}$$

Then we can use this decomposition to re-express matrix multiplication:
$$
{(AB)}_{ik}=\sum_{j}A_{ij}B_{jk}\\
=\sum_{j}M_{(i, j), (j, k), (k, i)}^{(n)}A_{ij}B_{jk}\\
= \sum_{a=1}^n\sum_{b=1}^n\sum_{c=1}^n\sum_{d=1}^n M_{(a, b), (c, d), (k, i)}^{(n)}A_{ab}B_{cd}\\
= \sum_{a=1}^n\sum_{b=1}^n\sum_{c=1}^n\sum_{d=1}^n (\sum_{l=1}^{r} x_{ab}^l y_{cd}^l z_{ki}^l)A_{ab}B_{cd}\\
= \sum_{l=1}^r z_{ki}^l(\sum_{a=1}^n\sum_{b=1}^n x_{ab}^l A_{ab})(\sum_{c=1}^n\sum_{d=1}^n y_{cd}^l B_{cd})
$$

<img src="https://jackwish.net/images/2019/qnnpack/qnnpack-gemm-reduce.jpg" width="70%" />

* [Fast Matrix Multiplication = Calculating Tensor Rank, Caltech Math 10 Presentation](https://people.eecs.berkeley.edu/~nirkhe/talks/ma10presentation2.pdf)
* http://users.wfu.edu/ballard/pdfs/CSE17.pdf
* https://jackwish.net/reveal-qnnpack-implementation.html
* [Coppersmith-Winograd Algorithm](https://www.gabormelli.com/RKB/Coppersmith-Winograd_Algorithm)
* [On the Coppersmith–Winograd method](http://www.cs.toronto.edu/~yuvalf/Limitations.pdf)
* [Adaptive Winograd’s Matrix Multiplications](https://www.ics.uci.edu/~fastmm/FMM-Reference/dalberto-nicolau.winograd.TOMS.pdf)
* https://www.wikiwand.com/en/Matrix_multiplication_algorithm
* [Introduction to fast matrix multiplication](https://dustingmixon.wordpress.com/2014/07/17/introduction-to-fast-matrix-multiplication/)
* [Practical Fast Matrix Multiplication Algorithms](http://jianyuhuang.com/papers/thesis.pdf)

<img src="https://oscimg.oschina.net/oscnet/0338ded39791cc2a4a5702c65e0c91f5533.jpg" width="70%"/>

* [Breaking the Coppersmith-Winograd barrier](https://www.cs.rit.edu/~rlc/Courses/Algorithms/Papers/matrixMult.pdf)
* [Multiplying matrices faster than Coppersmith-Winograd](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.297.2680&rep=rep1&type=pdf)
* [Limits on All Known (and Some Unknown) Approaches to Matrix Multiplication](https://simons.berkeley.edu/talks/virginia)
* [A New Fast Recursive Matrix Multiplication Algorithm](https://link.springer.com/article/10.1007/s10559-019-00163-2)


- https://cqa.institute/2015/08/07/accelerating-matrix-multiplication/
- [Distinguished Lecture Series III: Avi Wigderson, “Algebraic computation”](https://terrytao.wordpress.com/tag/fast-matrix-multiplication/)
- http://www.cs.utexas.edu/~rvdg/


#### Linear Algebra Packages

- [Halide: a language for fast, portable computation on images and tensors](https://halide-lang.org/)
- https://eigen.tuxfamily.org/dox/index.html
- [FLAME project ](https://www.cs.utexas.edu/~flame/web/)
- [A high-level language for programming accelerators](https://spatial-lang.org/)
- http://people.ece.umn.edu/users/parhi/
- https://www.cs.utexas.edu/~flame/web/publications.html
- [PLAPACK: Parallel Linear Algebra Package](http://www.cs.utexas.edu/users/plapack/)
- https://spatial-lang.readthedocs.io/en/latest/
- [Matrix Computations and Optimization in Apache Spark](https://stanford.edu/~rezab/papers/linalg.pdf)
- [Cusp is a library for sparse linear algebra and graph computations based on Thrust.](https://cusplibrary.github.io/index.html)
- [The Hierarchical Computations on Manycore Architectures (HiCMA) library aims to redesign existing dense linear algebra libraries to exploit the data sparsity of the matrix operator. ](https://ecrc.github.io/hicma/)
- [A generalized multidimensional matrix multiplication](http://tamivox.org/redbear/gen_matrix_mult/index.html)
- https://dspace.mit.edu/handle/1721.1/85943
- [BLIS Retreat 2017](http://www.cs.utexas.edu/users/flame/BLISRetreat2017/program.html)

### Automatic Differentiation, Differentiable Programming and Program Transformations

#### Automatic Differentiation

All numerical gradient-based optimization methods benefits from faster computation of gradients specially `backprop`.

> Many algorithms in machine learning, computer vision, physical simulation, and other fields require the calculation of `gradients and other derivatives`. 
> Manual derivation of gradients can be time consuming and error-prone. 
> `Automatic Differentiation (AD)` is a technology for automatically augmenting computer programs, including arbitrarily complex simulations, with statements for the computation of derivatives, also known as sensitivities. 
> Automatic differentiation comprises a set of techniques to calculate the derivative of a numerical computation expressed as a computer program. 
> These techniques are commonly used in atmospheric sciences and computational fluid dynamics, and have more recently also been adopted by machine learning researchers.
> Practitioners across many fields have built a wide set of automatic differentiation tools, using different programming languages, computational primitives and intermediate compiler representations. Each of these choices comes with positive and negative trade-offs, in terms of their usability, flexibility and performance in specific domains.

> In the ideal case, automatically generated derivatives should be competitive with manually generated ones and run at near-peak performance on modern hardware, 
but the most expressive systems for autodiff which can handle arbitrary, Turing-complete programs, are unsuited for performance-critical applications, such as large-scale machine learning or physical simulation. 
[Alternatively, the most performant systems are not designed for use outside of their designated application space, e.g. graphics or neural networks.](https://autodiff-workshop.github.io/)

+ https://autodiff-workshop.github.io/
+ https://autodiff-workshop.github.io/2016.html
+ http://www.autodiff.org/
+ [Tools for Automatic Differentiation](http://www.autodiff.org/?module=Tools)
+ https://arxiv.org/abs/1611.01652
+ [The simple essence of automatic differentiation](http://conal.net/papers/essence-of-ad/)

“What does AD mean, independently of implementation?” An answer arises in the form of naturality of sampling a function and its derivative. 
Automatic differentiation flows out of this naturality condition, 
together with the chain rule. Graduating from first-order to higher-order AD corresponds to sampling all derivatives instead of just one. 
Next, the setting is expanded to arbitrary vector spaces, in which derivative values are linear maps. 
[The specification of AD adapts to this elegant and very general setting, which even simplifies the development.](http://conal.net/papers/beautiful-differentiation/)


#### Differentiable Programming

Deep learning may look like another passing fad, in the vein of "expert systems" or "big data." 
But it's based on two timeless ideas (back-propagation and weight-tying), and while differentiable programming is a very new concept, 
it's a natural extension of these ideas that may prove timeless itself. 
Even as specific implementations, architectures, and technical phrases go in and out of fashion, 
[these core concepts will continue to be essential to the success of AI.](https://www.edge.org/response-detail/26794)

<img src="https://skymind.ai/images/wiki/differentiable_probabilistic.jpg" width="70%">

[Constructing neural networks using pure and higher-order differentiable functions and training them using reverse-mode automatic differentiation is unsurprisingly called Differentiable Programming. ](https://www.goto10.se/evenemang/the-principles-behind-differentiable-programming/)

+ [What Is Differentiable Programming?](https://fluxml.ai/2019/02/07/what-is-differentiable-programming.html)
+ https://www.lokad.com/differentiable-programming
+ [Google Summer of Code Projects](https://fluxml.ai/gsoc.html)
+ [Flux: The Julia Machine Learning Library](https://fluxml.ai/Flux.jl/stable/)
+ [DiffEqFlux.jl – A Julia Library for Neural Differential Equations](https://julialang.org/blog/2019/01/fluxdiffeq)
+ [Demystifying Differentiable Programming: Shift/Reset the Penultimate Backpropagator](https://arxiv.org/abs/1803.10228)
+ [Diòerentiable Visual Computing by Tzu-Mao Li](https://people.csail.mit.edu/tzumao/phdthesis/phdthesis.pdf)

______
---|Deep Learning |Differentiable Programming
---|---|---
Primary purpose|Learning|Learning+Optimization
Typical usage|Learn-once, Eval-many|Learn-once, Eval-once
Input granularity|Fat objects (images, voice sequences, lidar scans, full text pages)|Thin objects (products, clients, SKUs, prices)
Input variety|Homogeneous objects (e.g. images all having the same height/width ratio)|Heterogeneous objects (relational tables, graphs, time-series)

+ [Probabilistic & Differentiable Programming Summit](https://probabilisticdifferentiablepro.splashthat.com/)
+ [Differentiable Programming for Image Processing and Deep Learning in Halide](https://people.csail.mit.edu/tzumao/gradient_halide/)
+ https://github.com/sunze1/Differential-Programming
+ [Differentiable Programming: A Semantics Perspective](https://barghouthi.github.io/2018/05/01/differentiable-programming/)
+ https://fixpointsandcoffee.com/computer-science/169/
+ [Zygote: A Differentiable Programming System to Bridge Machine Learning and Scientific Computing](https://www.groundai.com/project/zygote-a-differentiable-programming-system-to-bridge-machine-learning-and-scientific-computing/)
+ [Differentiable Programming for Image Processing and Deep Learning in Halide](https://people.csail.mit.edu/tzumao/gradient_halide/)
+ https://github.com/Hananel-Hazan/bindsnet
+ https://skymind.ai/wiki/differentiableprogramming
+ https://people.csail.mit.edu/tzumao/
+ https://people.eecs.berkeley.edu/~jrk/
+ http://people.csail.mit.edu/fredo/

#### Program Transformations

[Program Transformations for Machine Learning](https://program-transformations.github.io/)- Workshop at NeurIPS 2019 – December 13 or 14 2019, Vancouver, Canada - claims that
> Machine learning researchers often express complex models as a program, relying on program transformations to add functionality. 
> New languages and transformations (e.g., TorchScript and TensorFlow AutoGraph) are becoming core capabilities of ML libraries. 
> However, existing transformations, such as `automatic differentiation` (AD or autodiff), inference in `probabilistic programming languages` (PPLs), and `optimizing compilers` are often built in isolation, and limited in scope. 
> This workshop aims at viewing program transformations in ML in a unified light, making these capabilities more accessible, and building entirely new ones.

> Program transformations are an area of active study. AD transforms a program performing numerical computation into one computing the gradient of those computations. 
> In probabilistic programming, a program describing a sampling procedure can be modified to perform inference on model parameters given observations. 
> Other examples are vectorizing a program expressed on one data point, and learned transformations where ML models use programs as inputs or outputs.

> This workshop will bring together researchers in the fields of `AD, probabilistic programming, programming languages, compilers, and ML` with the goal of understanding the commonalities between disparate approaches and views, and sharing ways to make these techniques broadly available. 
> It would enable ML practitioners to iterate faster on novel models and architectures (e.g., those naturally expressed through high-level constructs like recursion).

+ https://popl19.sigplan.org/track/lafi-2019#About
+ https://program-transformations.github.io/
+ https://uncertainties-python-package.readthedocs.io/en/latest/
+ https://conf.researchr.org/track/POPL-2017/pps-2017
+ https://gustavoasoares.github.io/
+ https://kedar-namjoshi.github.io/
+ [Learning Syntactic Program Transformations from Examples](https://alexpolozov.com/papers/icse2017-refactoring.pdf)
+ https://alexpolozov.com/
+ https://kedar-namjoshi.github.io/
+ https://vega.github.io/vega/


###  Open Auto-differentiation Library

+ https://coin-or.github.io/CppAD/doc/cppad.htm
+ http://simweb.iwr.uni-heidelberg.de/~darndt/files/doxygen/deal.II/index.html
+ [AD-Suite: A Test Suite for Algorithmic Differentiation](http://www.autodiff.org/ad16/Oral/Narayanamurthi_ADSuite.pdf)
+ [autodiff is a C++17 library for automatic computation of derivatives](https://autodiff.github.io/)
+ [DiffSharp: Differentiable Functional Programming](http://diffsharp.github.io/DiffSharp/)
+ https://www.mcs.anl.gov/OpenAD/
+ https://en.wikipedia.org/wiki/Automatic_differentiation
+ http://www.admb-project.org/
+ https://fluxml.ai/Zygote.jl/latest/
+ http://www.met.reading.ac.uk/clouds/adept/
+ [AD computation with Template Model Builder (TMB)](https://github.com/kaskr/adcomp)
+ https://www.juliadiff.org/
+ [autodiffr for Automatic Differentiation in R through Julia](https://non-contradiction.github.io/autodiffr/)
+ https://srijithr.gitlab.io/post/autodiff/
+ https://fl.readthedocs.io/en/latest/autograd.html
+ https://pymanopt.github.io/
+ https://yiduai.sg/tensorflow-workshop/
+ https://enzyme.mit.edu/
+ [Automatic Differentiation in Swift](https://github.com/tensorflow/swift/blob/master/docs/AutomaticDifferentiation.md)
+ https://github.com/wsmoses/Enzyme
+ https://github.com/Functional-AutoDiff/STALINGRAD
+ https://github.com/NVIDIA/MinkowskiEngine
+ https://github.com/rjhogan/Adept
+ https://github.com/google/tangent
+ https://github.com/autodiff/autodiff
+ https://nvidia.github.io/MinkowskiEngine/




##  Deep Model Compression

- https://github.com/he-y/Awesome-Pruning
- https://www.microsoft.com/en-us/research/search/?q=Deep+Compression
- https://www.tinyml.org/summit/slides/tinyMLSummit2020-4-1-Choi.pdf
- https://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_On_Compressing_Deep_CVPR_2017_paper.pdf
- https://github.com/mightydeveloper/Deep-Compression-PyTorch
- https://www.oki.com/en/rd/tt/dl/
- [Deep Compression, DSD Training and EIE: Deep Neural Network Model Compression, Regularization and Hardware Acceleration](https://www.microsoft.com/en-us/research/video/deep-compression-dsd-training-and-eie-deep-neural-network-model-compression-regularization-and-hardware-acceleration/)


-----------------------------

CNN is the most wisely used deep learning models in computer vision.

Theme Name | Description | Application | More Details
----|----|----|----
Parameter pruning and sharing | Reducing redundant parameters which are not sensitive to the performance | Convolutional layer and fully connected layer| Robust to various setting, can achieve good performance, can support both train from scratch and pre-trained model
Low-rank factorization| Using matrix/tensor decomposition to estimate the information parameters | Convolutional layer and fully connected layer| Standardized pipeline, easily to be implemented, can support both train from scratch and pre-trained model
Transferred/compact convolutional filters | Designing special structural convolutional filter to save parameters | Convolutional layer  only | Algorithms are dependent on applications, usually achieve good performance, only support train from scratch
Knowledge distillation |Training a compact neural network with distilled knowledge of a large model |Convolutional layer and fully connected layer| Model performances are sensitive to applications and network structure only support train from scratch

* [Stanford Compression Forum](https://compression.stanford.edu/2019-stanford-compression-workshop)
* https://jackwish.net/convolution-neural-networks-optimization.html
* [Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks ](http://eyeriss.mit.edu/)
* [深度学习如何进行模型压缩？](https://www.zhihu.com/question/64987081/answer/684375500)
* [Caffeine: Towards Uniformed Representation and Acceleration for Deep Convolutional Neural Networks](https://vast.cs.ucla.edu/publications/caffeine-towards-uniformed-representation-and-acceleration-deep-convolutional-neural)
* [Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks](https://vast.cs.ucla.edu/publications/optimizing-fpga-based-accelerator-design-deep-convolutional-neural-networks)
* [Automated Systolic Array Architecture Synthesis for High Throughput CNN Inference on FPGAs](https://vast.cs.ucla.edu/publications/automated-systolic-array-architecture-synthesis-high-throughput-cnn-inference-fpgas)
* [Efficient Deep Learning for Computer Vision CVPR 2019](https://sites.google.com/view/ecv2019/home)
* [Model Compression and Acceleration for Deep Neural Networks: The Principles, Progress, and Challenges](https://ieeexplore.ieee.org/document/8253600)
* https://www.ibm.com/blogs/research/2018/02/deep-learning-training/


### Fixed-point Arithmetic and Approximate Computing

Today’s computing systems are designed to deliver only exact solutions at high energy cost, while many of the algorithms that are run on data are at their heart statistical, and thus do not require exact answers.

It turns out that it is sometimes possible to get high-accuracy solutions from low-precision training—
and here we'll describe a new variant of stochastic gradient descent (SGD) called high-accuracy low precision (HALP) that can do it. 
HALP can do better than previous algorithms 
because it reduces the two sources of noise that limit the accuracy of low-precision SGD: gradient variance and round-off error.

<img src="https://jackwish.net/images/2019/quantization/mixed-fp32int8-pure-int8.svg" width="80%"/>

* https://www.wikiwand.com/en/Fixed-point_arithmetic
* [Approximate Computing](http://moimani.weebly.com/approximate-computing.html)
* [PACO: The Worlds First Approximate Computing General Purpose CPU](https://paco-cpu.github.io/paco-cpu/)
* [System Energy Efficiency Lab](http://seelab.ucsd.edu/)
* http://www.oliviervalery.com/publications/pdp2018
* https://devblogs.nvidia.com/int8-inference-autonomous-vehicles-tensorrt/
* https://nvidia.github.io/apex/
* [A Multiprecision World](https://sinews.siam.org/Details-Page/a-multiprecision-world)
* https://arxiv.org/abs/1806.00875v1

### Huffman Encoding

Huffman code is a type of optimal prefix code that is commonly used for loss-less data compression.
It produces a variable-length code table for encoding source symbol.
The table is derived from the occurrence
probability for each symbol.
As in other entropy encoding methods, more common symbols are represented with fewer bits than less common symbols, thus save the total space.

* [Huffman coding](https://www.wikiwand.com/en/Huffman_coding) is a code scheme.

### Knowledge Distillation

[Distillation (Hinton et al., 2015) is a kind of model compression approaches in which a pre-trained large model teaches a smaller model to achieve the similar prediction performance. It is often named as the "teacher-student" training, where the large model is the teacher and the smaller model is the student.](https://pocketflow.github.io/distillation/)

Initially, [it is used to compress the knowledge in an ensemble into a single model  which is much easier to deploy.](https://arxiv.org/pdf/1503.02531.pdf)

The core idea is that [an obvious way to transfer the generalization ability of the cumbersome model to a small model is to use the class probabilities produced by the cumbersome model as “soft targets” for training the small model. When the soft targets have high entropy, they provide much more information per training case than hard targets and much less variance in the gradient between training cases, so the small model can often be trained on much less data than the original cumbersome model and using a much higher learning rate.](https://arxiv.org/pdf/1503.02531.pdf)

[With distillation, knowledge can be transferred from the teacher model to the student by minimizing a loss function to recover the distribution of class probabilities predicted by the teacher model. In most situations, the probability of the correct class predicted by the teacher model is very high, and probabilities of other classes are close to 0, which may not be able to provide extra information beyond ground-truth labels. To overcome this issue, a commonly-used solution is to raise the temperature of the final softmax function until the cumbersome model produces a suitably soft set of targets.](https://pocketflow.github.io/distillation/)
The soften probability $q_i$ of class $i$ is calculated from the logit $z_i$
$$q_i = \frac{\exp \left( z_i / T \right)}{\sum_j{\exp \left( z_j / T \right)}}$$

where $T$ is the temperature. As $T$ grows, the probability distribution is more smooth, providing more information as to which classes the cumbersome model more similar to the predicted class. It is better to include the standard loss ($T=1$) between the predicted class probabilities and ground-truth labels. The overall loss function is given by:

$$L(x;W)=H(y,\sigma(z_s;T=1))+\alpha⋅H(\sigma(z_t;T=\tau),\sigma(z_s,T=\tau))$$
where $x$ is the input, $W$ are parameters of the distilled small model and $y$ is ground-truth labels, $\sigma$ is the softmax parameterized by temperature $T, H$ is the cross-entropy loss, and $\alpha$ is the coefficient of distillation loss.


* https://nervanasystems.github.io/distiller/knowledge_distillation/index.html
* [Awesome Knowledge Distillation](https://github.com/dkozlov/awesome-knowledge-distillation)
* [knowledge distillation papers](https://github.com/lhyfst/knowledge-distillation-papers)
* https://pocketflow.github.io/distillation/
* [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)


[Knowledge Distillation (KD) is a widely used approach to transfer the output information from a heavy network to a smaller network for achieving higher performance.](http://www.noahlab.com.hk/#/news/new1909_1) The student network can be optimized using the following loss function based on knowledge distillation:
$$\mathcal L_{KD}=\frac{1}{n}H(y_S^i, y_T^i).$$

Therefore, utilizing the knowledge transfer technique, a portable network can be optimized without the specific architecture of the given network.

<img src="http://www.noahlab.com.hk/mockdata/news/new1909_1/img/datafree7.png" width="70%"/>

* [Great Breakthrough! Huawei Noah's Ark Lab first pioneers a novel knowledge distillation technique without training data.](http://www.noahlab.com.hk/#/news/new1909_1)
* [DAFL: Data-Free Learning of Student Networks](https://arxiv.org/abs/1904.01186)
* https://github.com/huawei-noah/DAFL
* https://blog.csdn.net/xbinworld/article/details/83063726

### Parameter Pruning and Sharing

Pruning is to prune the connections in deep neural network in order to reduce the number of weights.

* Learn the connectivity via normal network training
* Prune the low-weight connections
* Retrain the sparse network

<img src=https://littletomatodonkey.github.io/img/post/20181010-DC-%E6%A8%A1%E5%9E%8B%E5%8E%8B%E7%BC%A9%E6%B5%81%E7%A8%8B%E5%9B%BE.png width=80% />

----
* [Pruning deep neural networks to make them fast and small](https://jacobgil.github.io/deeplearning/pruning-deep-learning)
* https://nervanasystems.github.io/distiller/pruning/index.html
* https://github.com/yihui-he/channel-pruning
* https://pocketflow.github.io/cp_learner/
* [A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers](https://arxiv.org/abs/1804.03294)

<img src=https://pocketflow.github.io/pics/framework_design.png width=80% />

### Quantization and  Fixed-point Arithmetic

`Network quantization compresses the original network by
reducing the number of bits required to represent each weight.`


[`Uniform quantization` is widely used for model compression and acceleration. Originally the weights in the network are represented by 32-bit floating-point numbers. ](https://pocketflow.github.io/uq_learner/)
With uniform quantization, low-precision (e.g. 4-bit or 8-bit) fixed-point numbers are used to approximate the full-precision network. 
For k-bit quantization, the memory saving can be up to $32/k$​. For example, 8-bit quantization can reduce the network size by 4 folds with negligible drop of performance.
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

Given a pre-defined full-precision model, the learner inserts quantization nodes and operations into the computation graph of the model. 
With activation quantization enabled, quantization nodes will also be placed after activation operations (e.g. ReLU).

In the training phase, both full-precision and quantized weights are kept. 
In the forward pass, quantized weights are obtained by applying the quantization function on full-precision weights. 
To update full-precision weights in the backward pass, since gradients w.r.t. quantized weights are zeros almost everywhere,
 we use the straight-through estimator (STE, Bengio et al., 2015) to pass gradients of quantized weights directly to full-precision weights for update.



* [Quantized Neural Network PACKage - mobile-optimized implementation of quantized neural network operators ](https://github.com/pytorch/QNNPACK)
* [TensorQuant: A TensorFlow toolbox for Deep Neural Network Quantization](https://github.com/cc-hpc-itwm/TensorQuant)
* https://dominikfhg.github.io/TensorQuant/
* https://zhuanlan.zhihu.com/p/38328685
* [Distiller is an open-source Python package for neural network compression research.](https://nervanasystems.github.io/distiller/quantization/index.html)
* [Neural Network Quantization](https://jackwish.net/neural-network-quantization-introduction-chn.html)
* [Neural Network Quantization Resources](https://jackwish.net/neural-network-quantization-resources.html)
* [FINN-R: An End-to-End Deep-Learning Framework for Fast Exploration of Quantized Neural Networks](https://arxiv.org/abs/1809.04570v1)
* [Making Neural Nets Work With Low Precision](https://sahnimanas.github.io/post/quantization-in-tflite/)
* [Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks](https://arxiv.org/abs/1908.05033)
* [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
* [Lower Numerical Precision Deep Learning Inference and Training](https://software.intel.com/en-us/articles/lower-numerical-precision-deep-learning-inference-and-training)

<img src="https://jackwish.net/images/2019/quantization/fp-distribution.png" width="50%"/>

`Fixed-point Arithmetic`

[The precision of a fixed-point number is the number of digits to the right of the decimal point, and it normally stays the same when computations are performed on the number.](http://www.efgh.com/software/fixed.htm)

- [A Fixed-Point Arithmetic Package](http://www.efgh.com/software/fixed.htm)
- [http://hackage.haskell.org/package/fixed-point](http://hackage.haskell.org/package/fixed-point)
- https://courses.cs.washington.edu/courses/cse467/08au/labs/l5/fp.pdf
- [Fixed Point Arithmetic and Tricks](http://x86asm.net/articles/fixed-point-arithmetic-and-tricks/)

### Low Bit Neural Network

- [Convergence of a Relaxed Variable Splitting Coarse Gradient Descent Method for Learning Sparse Weight Binarized Activation Neural Network](https://www.math.uci.edu/~jxin/RVSCGD_2020.pdf)
- https://www.math.uci.edu/~jxin/xue.pdf
- https://arxiv.org/abs/1711.07354v1
- https://www.math.uci.edu/~jxin/signals.html
- [Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM](https://arxiv.org/abs/1707.09870)
- [Toward Extremely Low Bit and Lossless Accuracy in DNNs with Progressive ADMM](https://arxiv.org/abs/1905.00789)


#### 8-bit-training

The state-of-the-art hardware platforms for training Deep Neural Networks (DNNs) are moving from traditional single precision (32-bit) computations towards 16 bits of precision -- in large part due to the high energy efficiency and smaller bit storage associated with using reduced-precision representations. 
However, unlike inference, training with numbers represented with less than 16 bits has been challenging due to the need to maintain fidelity of the gradient computations during back-propagation. 
Here we demonstrate, for the first time, the successful training of DNNs using 8-bit floating point numbers while fully maintaining the accuracy on a spectrum of Deep Learning models and datasets. 
[In addition to reducing the data and computation precision to 8 bits, we also successfully reduce the arithmetic precision for additions (used in partial product accumulation and weight updates) from 32 bits to 16 bits through the introduction of a number of key ideas including chunk-based accumulation and floating point stochastic rounding.](https://arxiv.org/abs/1812.08011)
- [8-Bit Precision for Training Deep Learning Systems](https://www.ibm.com/blogs/research/2018/12/8-bit-precision-training/)
- [Extending 8-bit training breakthroughs to the toughest deep learning models](https://www.ibm.com/blogs/research/2019/12/8-bit-training-breakthroughs-deep-learning/)
- [Hybrid 8-bit Floating Point (HFP8) Training and Inference for Deep Neural Networks](https://papers.nips.cc/paper/8736-hybrid-8-bit-floating-point-hfp8-training-and-inference-for-deep-neural-networks)
- [Training Deep Neural Networks with 8-bit Floating Point Numbers](https://arxiv.org/abs/1812.08011)
- https://www.ibm.com/blogs/research/author/xiao-sun/

#### Binarized Neural Network, Ternary Weight Networks, XOR-Net

- https://github.com/uber-research/sbnet
- [BinaryConnect](https://github.com/MatthieuCourbariaux/BinaryConnect)

##### Binarized Neural Network

Binary neural networks are networks with binary weights and activations at run time. 
At training time these weights and activations are used for computing gradients; however, the gradients and true weights are stored in full precision. 
[This procedure allows us to effectively train a network on systems with fewer resources.](https://software.intel.com/en-us/articles/binary-neural-networks)

`Forward Binarization`

For forward propagation, we need two binary matrices; we thus binarize the weight matrix and the incoming activation from the previous layer.

<img src="https://software.intel.com/sites/default/files/managed/c0/e0/webops10048-fig1-binarization-procedure.png" width="80%"/>
<img src="https://mohitjainweb.files.wordpress.com/2018/07/bnn-forward-propagation.png?w=414&h=382"/>

[A key to the success of BNNs it the binary activation function, which clamps
all negatives inputs to −1 and all positive inputs to 1.](http://www.eecs.harvard.edu/~htk/publication/2017-ewsn-mcdanel-teerapittayanon-kung.pdf)
There  are two binarized functions:

deterministic| stochastic
---|---
<img src="https://mohitjainweb.files.wordpress.com/2018/07/bnn-sign-function.png?w=206&h=44">|<img src="https://mohitjainweb.files.wordpress.com/2018/07/bnn-stochastic-binarization-function.png?w=213&h=44">

Here $\sigma(x)$ is is the “hard sigmoid” function: $\sigma(x)=\max(0, min(1, \frac{x+1}{2}))$.
The stochastic binarization is better than the Sign function but is harder to implement. 
As a result, the deterministic Sign function is used more often.

- https://mohitjain.me/2018/07/14/bnn/

`Gradient Propagation Through Discretization`

The derivative of the sign function is zero almost everywhere, making it incompatible with backpropagation. 
Thus, a straight-through estimator is used. This preserves the gradient's information and cancels large gradients.

<img src="https://software.intel.com/sites/default/files/managed/c0/e0/webops10048-fig2-gradientPropagationProcedure.png" width="80%" />
<img src="https://mohitjainweb.files.wordpress.com/2018/07/bnn-backward-propagation.png?w=312&h=216" width="50%"/>

While updating the weights, the following is done:

Each real valued weight, $w^r$, is constrained to remain between -1 and +1. 
If a weight update brings $w^r$ outside $[-1, 1]$, it is clipped. This is done because otherwise, 
the real-valued weights will grow very large without having any impact on the binary weights, $w^b$.
The new updated binary weights are then calculated as $w^b = Sign(w^r)$.

This network has the following layers:

<img title="Layer map" src="https://software.intel.com/sites/default/files/managed/c0/e0/webops10048-fig4-network-layers.png" width="80%"/>


* Fully connected (128)
* Ramp - rectified linear unit (ReLU) activation function
* Binarize activations
* Fully connected (128)
* Ramp - ReLU activation function
* Binarize activations
* Fully connected (10)
* Sigmoid activation function

- https://blog.csdn.net/stdcoutzyx/article/details/50926174
- https://duanyzhi.github.io/Binary-Network/
- [BinaryConnect： Training Deep Neural Networks with binary weights during propagations](https://github.com/MatthieuCourbariaux/BinaryConnect)
- [Neural Networks with Few Multiplications](https://arxiv.org/abs/1510.03009)
- [Binary Neural Networks](https://software.intel.com/en-us/articles/binary-neural-networks)
- [Code: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://github.com/MatthieuCourbariaux/BinaryNet)
- [Code Sample: Optimizing Binarized Neural Networks on Intel® Xeon® Scalable Processors](https://software.intel.com/en-us/articles/optimizing-binarized-neural-networks-on-intel-xeon-scalable-processors)
- [Accelerating Neural Networks with Binary Arithmetic](https://www.intel.ai/accelerating-neural-networks-binary-arithmetic/)
- [ ] [The High-Dimensional Geometry of Binary Neural Networks](https://arxiv.org/abs/1705.07199)
- [ ] [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)
- [ ] [Boolean Circuits are Neural Networks](https://constantinides.net/2019/04/26/boolean-circuits-are-neural-networks/)
- [ ] [FP-BNN: Binarized neural network on FPGA](https://www.sciencedirect.com/science/article/pii/S0925231217315655)
- [ ] [Embedded Binarized Neural Networks](http://www.eecs.harvard.edu/~htk/publication/2017-ewsn-mcdanel-teerapittayanon-kung.pdf)
- [ ] [Deep Learning for Real-Time Crime Forecasting and Its Ternarization](https://arxiv.org/pdf/1711.08833.pdf)
- [ ] https://www.ibm.com/blogs/research/2019/04/2-bit-precision/


##### Ternary Weight Networks

Ternary weight networks (TWNs) is the  neural networks with weights
constrained to `+1, 0 and -1`.
This network can date up to the paper [Learning algorithms with neural network with ternary weights](http://id3490.securedata.net/rod/pdf/RG.Paper.CP24.pdf) in 1988.

- [Learning algorithms with neural network with ternary weights](http://id3490.securedata.net/rod/pdf/RG.Paper.CP24.pdf)
- [Ternary Weight Networks](https://arxiv.org/abs/1605.04711)
- [TBN: Convolutional Neural Network with Ternary Inputs and Binary Weights](http://openaccess.thecvf.com/content_ECCV_2018/papers/Diwen_Wan_TBN_Convolutional_Neural_ECCV_2018_paper.pdf)
- [Trained Ternary Quantization](https://www.arxiv-vanity.com/papers/1612.01064/)
- https://iceory.github.io/2018/04/04/ternary-weight-networks/
- https://ieeexplore.ieee.org/document/8581485

##### XOR-Net

In Binary-WeightNetworks, the filters are approximated with binary values resulting in 32× memory saving. 
In XNOR-Networks, both the filters and the input to convolutional layers are binary. 
[XNOR-Networks approximate convolutions using primarily binary operations. This results in 58× faster convolutional operations and 32× memory savings.](https://pjreddie.com/media/files/papers/xnor.pdf)

<img title="XNOR Net" src="https://pic1.zhimg.com/80/v2-a5bcc5b680ec296aeb706ca4f2fe2c90_hd.jpg" width="80%" />

* [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://pjreddie.com/media/files/papers/xnor.pdf)
* [GXNOR-Net: Training deep neural networks with ternary weights and activations without full-precision memory under a unified discretization framework](https://www.sciencedirect.com/science/article/abs/pii/S0893608018300108)
* [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)
* [Joseph Chet Redmon: author of XORNet](https://pjreddie.com/)
* [XNOR-Net论文解读](https://zhuanlan.zhihu.com/p/65103916)
* https://github.com/liuzechun/Bi-Real-net
* http://iis-projects.ee.ethz.ch/index.php/Bringing_XNOR-nets_(ConvNets)_to_Silicon
* http://web.eng.tau.ac.il/deep_learn/wp-content/uploads/2017/03/Binary-Deep-Learning.pdf
* https://papers.nips.cc/paper/6638-towards-accurate-binary-convolutional-neural-network.pdf
* https://arxiv.org/pdf/2001.05936v1.pdf
_____

* [Low Precision Arithmetic Simulation in PyTorch](https://github.com/Tiiiger/QPyTorch)
* [Deep Learning with Limited Numerical Precision](http://proceedings.mlr.press/v37/gupta15.pdf)
* [Making Neural Nets Work With Low Precision](https://sahnimanas.github.io/2018/06/24/quantization-in-tf-lite.html)
* [8-bit Inference with TensorRT](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)


#### Mixed Precision Training

[Mixed-precision training lowers the required resources by using lower-precision arithmetic, which has the following benefits.](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/)

+ Decrease the required amount of memory. Half-precision floating point format (FP16) uses 16 bits, compared to 32 bits for single precision (FP32). Lowering the required memory enables training of larger models or training with larger minibatches.
+ Shorten the training or inference time. Execution time can be sensitive to memory or arithmetic bandwidth. Half-precision halves the number of bytes accessed, thus reducing the time spent in memory-limited layers.

<img src="https://devblogs.nvidia.com/wp-content/uploads/2017/10/ssd_ag_log_histo_coarse.png" width="70%"/>

The Y-axis is the percentage of all values on a log scale. The X-axis is the log scale of absolute values, as well as a special entry for zeros. For example, in this training session 66.8% of values were zero, whereas 4% of values were between $2^{-32}$ and $2^{-30}$.
A very efficient way to ensure that gradients fall into the range representable by half precision is to multiply the training loss with the scale factor. 
This adds just a single multiplication and by the chain rule it ensures that all the gradients are scaled up (or shifted up) at no additional cost. 
Loss scaling ensures that relevant gradient values lost to zeros are recovered. 
Weight gradients need to be scaled down by the same factor $S$ before the weight update. 
[The scale-down operation could be fused with the weight update itself (resulting in no extra memory accesses) or carried out separately.](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/)

Additions to the traditional iteration procedure are in bold.

1. **Make an FP16 copy of the weights**
2. Forward propagate using FP16 weights and activations
3. **Multiply the resulting loss by the scale factor S**
4. Backward propagate using FP16 weights, activations, and their gradients
5. **Multiply the weight gradients by 1/S**
6. Optionally process the weight gradients (gradient clipping, weight decay, etc.)
7. Update the master copy of weights in FP32

* https://nnabla.readthedocs.io/en/latest/python/tutorial/mixed_precision_training.html
* https://nvidia.github.io/OpenSeq2Seq/html/mixed-precision.html
* https://pdc.one/2019/05/14/Mixed-precision-training/
* [Paulius Micikevicius's talk "Training Neural Networks with Mixed Precision: Theory and Practice" (GTC 2018, S8923).](http://on-demand.gputechconf.com/gtc-cn/2018/pdf/CH8302.pdf)
* [Experimental Evaluation of Mixed Precision Training for End to End Applications](http://research.baidu.com/Blog/index-view?id=103)
* [Apex (A PyTorch Extension)](https://nvidia.github.io/apex/)
* https://blog.masterliu.net/tensorflow/tensorflow-mixed-precision/
* [Mixed-Precision Training of Deep Neural Networks](https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/)
* [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
* [User Guide](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)
* [TRAINING WITH MIXED PRECISION](http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf)
* [NVIDIA Apex: Tools for Easy Mixed-Precision Training in PyTorch](https://devblogs.nvidia.com/apex-pytorch-easy-mixed-precision-training/)
* https://developer.nvidia.com/vulkan-driver



#### AdderNet

[In AdderNets, we take the $\ell_1$-norm distance between filters and input feature as the output response.](https://github.com/huawei-noah/AdderNet)

The convolution in CNN is replaced by the calculating $\ell_1$-norm distance distance between the filter and the input.


<img src="https://raw.githubusercontent.com/huawei-noah/AdderNet/master/figures/visualization.png"  width="80%">

- https://github.com/huawei-noah/AdderNet
- https://papers.nips.cc/paper/2020/file/1cf44d7975e6c86cffa70cae95b5fbb2-Paper.pdf
- [Kernel Based Progressive Distillation for Adder Neural Networks](https://arxiv.org/abs/2009.13044)
- https://www.zhihu.com/column/EdgeDeepLearning
- [AdderSR: Towards Energy Efficient Image Super-Resolution](https://arxiv.org/abs/2009.08891)
- https://arxiv.org/abs/2101.10015
- https://github.com/dlut-dimt/PODM
- [A Bridging Framework for Model Optimization and Deep Propagation](https://papers.nips.cc/paper/2018/file/1a0a283bfe7c549dee6c638a05200e32-Paper.pdf)


#### Blended Coarse Gradient Descent

Coarse gradient is generally not a gradient of any function but an artificial ascent direction. 
The weight update of BCGD goes by coarse gradient correction of a weighted average of the full-precision weights and their quantization (the so-called blending), 
[which yields sufficient descent in the objective value and thus accelerates the training.](https://www.math.uci.edu/~jxin/BCGD_RMS_2018.pdf)

* https://www.math.uci.edu/~jxin/
* [Blended coarse gradient descent for full quantization of deep neural networks](https://www.math.uci.edu/~jxin/BCGD_RMS_2018.pdf)
* [Training Quantized Deep Neural Networks and Applications with Blended Coarse Gradient Descent](https://sinews.siam.org/Details-Page/training-quantized-deep-neural-networks-and-applications-with-blended-coarse-gradient-descent)
* [Blended Coarse Gradient Descent for Full Quantization of Deep Neural Networks](https://arxiv.org/abs/1808.05240)
* https://dominikfhg.github.io/TensorQuant/
* [Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations](https://arxiv.org/abs/1609.07061v1)
* [Median Binary-Connect Method and a Binary Convolutional Neural Nework for Word Recognition](https://arxiv.org/abs/1811.02784v1)
* [BinaryConnect: Training Deep Neural Networks with binary weights during propagations](https://arxiv.org/abs/1511.00363)
* [Binary Relax: A Relaxation Approach For Training Deep Neural Networks With Quantized Weights](https://www.math.uci.edu/~jxin/binaryrelax_final.pdf)
* [Quantization and Training of Low Bit-Width Convolutional Neural Networks for Object Detectio](https://www.math.uci.edu/~jxin/LBW-JCM.pdf)
- https://www.math.uci.edu/~jxin/signals.html
- https://www.researchgate.net/scientific-contributions/43925792_Dimitris_S_Papailiopoulos


####  Low-precision Training

It turns out that DNNs can work with smaller datatypes, with less precision, such as 8-bit integers. 
Roughly speaking, we’re trying to work with a number line looking closer to the sparse one on the bottom. 
[The numbers are quantized, i.e. discretized to some specific values, which we can then represent using integers instead of floating-point numbers.](https://sahnimanas.github.io/post/quantization-in-tflite/)

- https://arxiv.org/abs/1906.03193
- https://arxiv.org/abs/1903.05662
- https://arxiv.org/pdf/1609.07061.pdf

##### High-accuracy Low Precision

High-accuracy low precision (HALP) is our algorithm which runs SVRG and uses bit centering with a full gradient at every epoch to update the low-precision representation.
It can do better than previous algorithms because it reduces the two sources of noise that limit the accuracy of low-precision SGD: gradient variance and round-off error.

- To reduce noise from gradient variance, HALP uses a known technique called `stochastic variance-reduced gradient (SVRG)`. 
   SVRG periodically uses full gradients to decrease the variance of the gradient samples used in SGD.
- To reduce noise from quantizing numbers into a low-precision representation, HALP uses a new technique we call `bit centering`.
  The intuition behind bit centering is that as we get closer to the optimum, the gradient gets smaller in magnitude and in some sense carries less information, 
  so we should be able to compress it. 
  By dynamically re-centering and re-scaling our low-precision numbers, we can lower the quantization noise as the algorithm converges.

* [High-Accuracy Low-Precision Training](https://arxiv.org/pdf/1803.03383.pdf)
* [Training deep neural networks with low precision multiplications](https://arxiv.org/abs/1412.7024)
* [HALP: High-Accuracy Low-Precision Training](https://dawn.cs.stanford.edu/2018/03/09/low-precision/)

##### Ultra-Low Precision Training

There are three primary challenges that make it difficult to scale precision below 16 bits while fully preserving model accuracy. 
Firstly, when all the operands (i.e., weights, activations, errors, and gradients) for general matrix multiplication (GEMM) and convolution computations are simply reduced to 8 bits, most DNNs suffer noticeable accuracy degradation. Secondly, reducing the bit precision of accumulations in GEMM from 32 bits to 16 bits significantly impacts the convergence of DNN training. 
This is why commercially available hardware platforms exploiting scaled precision for training (including GPUs) still continue to use 32 bits of precision for accumulation. 
Reducing accumulation bit precision below 32 bits is critically important for reducing the area and power of 8-bit hardware. 
Finally, reducing the bit precision of weight updates to 16-bit floating-point impacts accuracy, 
[while 32-bit weight updates, used in today’s systems, require an extra copy of the high-precision weights and gradients to be kept in memory, which is expensive.](https://www.ibm.com/blogs/research/2018/12/8-bit-precision-training/)

* [Accumulation Bit-Width Scaling For Ultra-Low Precision Training Of Deep Networks](https://arxiv.org/abs/1901.06588)
* [Ultra-Low-Precision Training of Deep Neural Networks](https://www.ibm.com/blogs/research/2019/05/ultra-low-precision-training/)
* [8-Bit Precision for Training Deep Learning Systems](https://www.ibm.com/blogs/research/2018/12/8-bit-precision-training/)
* [Training High-Performance and Large-Scale Deep Neural Networks with Full 8-bit Integers](https://arxiv.org/pdf/1909.02384.pdf)
* [Highly Accurate Deep Learning Inference with 2-bit Precision](https://www.ibm.com/blogs/research/2019/04/2-bit-precision/)
* [OptQuant: Distributed training of neural networks with optimized quantization mechanisms](https://www.sciencedirect.com/science/article/pii/S0925231219302735)
* https://www.ibm.com/blogs/research/author/chia-yuchen/
* https://www.ibm.com/blogs/research/author/naigangwang/


##### ADMM-NN

We can apply alteranting direction method of mulipliers(ADMM) to train deep neural networks.
The first part of ADMM-NN is a systematic, joint framework of DNN weight pruning and quantization using ADMM. 
It can be understood as a smart regularization technique with regularization target dynamically updated in each ADMM iteration, 
thereby resulting in higher performance in model compression than prior work. 
The second part is hardware-aware DNN optimizations to facilitate hardware-level implementations. 
Without accuracy loss, we can achieve 85\timesand 24\timespruning on LeNet-5 and AlexNet models, respectively, significantly higher than prior work. 
The improvement becomes more significant when focusing on computation reductions. 
[Combining weight pruning and quantization, we achieve 1,910\timesand 231\timesreductions in overall model size on these two benchmarks, when focusing on data storage. Highly promising results are also observed on other representative DNNs such as VGGNet and ResNet-50.](https://scinapse.io/papers/2906775339)

- https://web.northeastern.edu/yanzhiwang/publications/
- [Deep ADMM-Net for Compressive Sensing MRI](https://papers.nips.cc/paper/6406-deep-admm-net-for-compressive-sensing-mri)
- [Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM](https://arxiv.org/abs/1707.09870)
- [Toward Extremely Low Bit and Lossless Accuracy in DNNs with Progressive ADMM](https://arxiv.org/abs/1905.00789)
- [ADMM-based Weight Pruning for Real-Time Deep Learning Acceleration on Mobile Devices](https://cse.buffalo.edu/~wenyaoxu/papers/conference/xu-glvlsi2019.pdf)
- [ADMM-NN: An Algorithm-Hardware Co-Design Framework of DNNs Using Alternating Direction Method of Multipliers](https://scinapse.io/papers/2906775339)
- [StructADMM: A Systematic, High-Efficiency Framework of Structured Weight Pruning for DNNs](https://arxiv.org/abs/1807.11091)
- https://blog.csdn.net/XSYYMY/article/details/81904882
- https://arxiv.org/abs/1706.06197
- https://csyhhu.github.io/
- https://www.ntu.edu.sg/home/sinnopan/
- https://ywang393.expressions.syr.edu/
- https://cse.buffalo.edu/~wenyaoxu/
- http://gr.xjtu.edu.cn/web/jiansun/publications


### Transferred/Compact Convolutional Filters

Transfer learning methods have demonstrated state-of-the-art performance on various small-scale image classification tasks. 
This is generally achieved by exploiting the information from an ImageNet convolution neural network (ImageNet CNN). However, the transferred CNN model is generally with high computational complexity and storage requirement. 
It raises the issue for real-world applications, especially for some portable devices like phones and tablets without high-performance GPUs. 
Several approximation methods have been proposed to reduce the complexity by reconstructing the linear or non-linear filters (responses) in convolutional layers with a series of small ones.

+ [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946v1.pdf)
+ [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
+ [Compact Convolutional Neural Network Transfer Learning For Small-Scale Image Classification](https://kar.kent.ac.uk/55053/)
+ https://zhuanlan.zhihu.com/p/35405071
+ https://www.jianshu.com/u/4a95cf020626
+ https://github.com/MG2033/ShuffleNet

### Tensor Methods

Note that the deep learning models are composite of linear and non-linear maps. And linear maps are based on matrices.

These methods take a layer and decompose it into several smaller layers. 
Although there will be more layers after the decomposition, 
the total number of floating point operations and weights will be smaller. 
Some reported results are on the order of x8 for entire networks (not aimed at large tasks like imagenet, though), or x4 for specific layers inside imagenet. 
[My experience was that with these decompositions I was able to get a speedup of between x2 to x4, depending on the accuracy drop I was willing to take.](https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning)

- http://www.tensorlet.com/
- [Optimal Low Rank Tensor Factorization for Deep Learning](https://link.springer.com/chapter/10.1007%2F978-981-13-2372-0_42)
- https://github.com/jacobgil/pytorch-tensor-decompositions
- [Accelerating deep neural networks with tensor decompositions](https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning)
- [Workshop: “Deep Learning and Tensor/Matrix Decomposition for Applications in Neuroscience”, November 17th, Singapore](https://crei.skoltech.ru/cdise/icdm-2018-workshop/)
- http://www.deeptensor.ml/index.html
- http://tensorly.org/stable/home.html
- http://www.vision.jhu.edu/
- https://wabyking.github.io/talks/DL_tensor.pdf

**Singular value decomposition**

The matrix $A_{m\times n}$ can be decomposed as the multiplication of two matrices such as $A_{m\times n}= Q_{m\times r}R_{r\times n}$, so that the storage is from $O(m\times n)$ to $O(m+n)\times O(r)$.

To explore a low-rank subspace combined with a sparse structure for the weight matrix $W$, we assume that $W \approx L+S$,
where $L$ is a low-rank component and $S$ is a sparse matrix. Then, to
compress the weight matrix, we have the following model:
$$
\min_{L, S}\frac{1}{2}{\|W-L-S\|}_F^2,\\
s.t.\quad rnak(L) \leq r,\\
card(S)\leq c,$$
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
* https://arxiv.org/pdf/1712.01887.pdf

***

* https://srdas.github.io/DLBook/intro.html#effective
* https://cognitiveclass.ai/courses/accelerating-deep-learning-gpu/
* https://github.com/songhan/Deep-Compression-AlexNet
* [awesome-model-compression-and-acceleration](https://github.com/sun254/awesome-model-compression-and-acceleration)
* [gab41.lab41.org](https://gab41.lab41.org/lab41-reading-group-deep-compression-9c36064fb209)
* [CS 598 LAZ: Cutting-Edge Trends in Deep Learning and Recognition](http://slazebni.cs.illinois.edu/spring17/)
* http://slazebni.cs.illinois.edu/spring17/lec06_compression.pdf
* http://slazebni.cs.illinois.edu/spring17/reading_lists.html#lec06
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
- [Parameter Compression of Recurrent Neural Networks and Degradation of Short-term Memory](https://arxiv.org/ftp/arxiv/papers/1612/1612.00891.pdf)
- https://ai.google/research/pubs/pub44632
- [CLINK: Compact LSTM Inference Kernel for Energy Efficient Neurofeedback Devices](https://vast.cs.ucla.edu/sites/default/files/publications/CLINK_ISLPED%202018%20publication.pdf)

### Compressing GANs

* https://arxiv.org/abs/1902.00159
* https://github.com/mit-han-lab/gan-compression
* https://hanlab.mit.edu/projects/gancompression/
* https://github.com/amitadate/gan-compression
* [Adversarial Network Compression](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11132/Belagiannis_Adversarial_Network_Compression_ECCVW_2018_paper.pdf)
* [Model Compression with Generative Adversarial Networks](https://arxiv.org/pdf/1812.02271v2.pdf)


- [ ]  https://github.com/kedartatwawadi/NN_compression
- [ ]  [Optimizing Data-Intensive Computations in Existing Libraries with Split Annotations](https://dawn.cs.stanford.edu/2019/10/22/split-annotations/)


####  GAN-pruning

* https://github.com/huawei-noah/GAN-pruning
* https://daiwk.github.io/posts/cv-gan-pruning.html

### Compressed Transformer

-  https://github.com/khakhulin/compressed-transformer
-  http://nla.skoltech.ru/projects/files/presentations/team_21.pdf
-  https://shaojiejiang.github.io/post/en/compressive-transformers/
-  https://github.com/khakhulin/compressed-transformer

#### Compressed BERT

- https://github.com/szhangtju/The-compression-of-Transformer
- https://github.com/intersun/PKD-for-BERT-Model-Compression
- https://huggingface.co/transformers/model_doc/mobilebert.html

### Hashing-accelerated neural networks

[Our approach is compellingly simple: we use a hash function to group network connections into hash buckets uniformly at random such that all connections grouped to the i th hash bucket share the same weight value w_i. Our parameter hashing is akin to prior work in feature hashing and requires no additional memory overhead. The backpropagation algorithm can naturally tune the hash bucket parameters and take into account the random weight sharing within the neural network architecture.](https://www.cse.wustl.edu/~ychen/HashedNets/)

<img src="https://www.cse.wustl.edu/~ychen/HashedNets/hashednets.png">
  
- https://www.cse.wustl.edu/~ychen/HashedNets/
- [Compressing Neural Networks with the Hashing Trick](http://proceedings.mlr.press/v37/chenc15.pdf)
- https://spectrum.ieee.org/tech-talk/computing/hardware/algorithms-and-hardware-for-deep-learning
- https://www2021.thewebconf.org/papers/hashing-accelerated-graph-neural-networks-for-link-prediction/


## Distributed Training

The problem of deep learning $T(x;\Theta)$ is the big model and big data, i.e., $\Theta$ may be in too extra-high dimensional space to store in a single laptop computer.
And the training process is to find the optimal parameters $\arg\min_{\Theta}\sum_{i}L(T(x_i;\Theta), y_i)$, which requires sufficient data size.

Sometimes we need partition the model or the data into different machines.
In another world, the model or the data are distributed in a few machines.

Distributed training of deep learning models is a branch of distributed computation.

Training advanced deep learning models is challenging. Beyond model design, model scientists also need to set up the state-of-the-art training techniques such as distributed training, mixed precision, gradient accumulation, and checkpointing. 
[Yet still, scientists may not achieve the desired system performance and convergence rate. Large model sizes are even more challenging: a large model easily runs out of memory with pure data parallelism and it is difficult to use model parallelism.](https://www.deepspeed.ai/)

It is really important to reduce the cost of communication in distributed computation
including the communication time, communication frequency, communication content and latency.


- https://www.cs.rice.edu/~as143/COMP640_Fall16/
- https://www.cs.rice.edu/~as143/
- https://www.deepspeed.ai/
- https://shivaram.org/
- https://www.comp.hkbu.edu.hk/~chxw/
- [High Performance Distributed Deep Learning](http://prace.it4i.cz/en/HPDDL-01-2018)
- [Tutorial: High Performance Distributed Deep Learning](https://ppopp20.sigplan.org/details/PPoPP-2020-Workshops-and-Tutorials/3/Tutorial-High-Performance-Distributed-Deep-Learning)

### Accelerating Deep Learning Workloads

[The more we know about the resource usage patterns of workloads, the better we can allocate resources.](https://www.usenix.org/conference/hotcloud20/presentation/berral)

- https://itpeernetwork.intel.com/accelerating-deep-learning-workloads/#gs.ny4nke
- [A Generic Communication Scheduler for Distributed DNN Training Acceleration](https://i.cs.hku.hk/~cwu/papers/yhpeng-sosp19.pdf)
- https://arxiv.org/pdf/1806.03377.pdf
- http://www.iiswc.org/iiswc2019/index.html
- https://cs.stanford.edu/~matei/
- [Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations](http://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf)
- https://zhuanlan.zhihu.com/p/87515411
- https://zhuanlan.zhihu.com/p/84862107
- [Accelerating Deep Learning Workloads through Efficient Multi-Model Execution](https://cs.stanford.edu/~matei/papers/2018/mlsys_hivemind.pdf)
- [Multi-tenant GPU Clusters for Deep Learning Workloads: Analysis and Implications](https://www.microsoft.com/en-us/research/uploads/prod/2018/05/gpu_sched_tr.pdf)
- https://cs.stanford.edu/~matei/

#### PipeDream

- [Characterizing Deep Learning Training Workloads on Alibaba-PAI](https://arxiv.org/abs/1910.05930)
- https://www.microsoft.com/en-us/research/project/fiddle/
- https://github.com/bytedance/byteps
- https://github.com/msr-fiddle/pipedream
- https://github.com/msr-fiddle/philly-trace
- https://www.microsoft.com/en-us/research/project/fiddle/
- [PipeMare: Asynchronous Pipeline Parallel DNN Training](https://arxiv.org/abs/1910.05124)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [XPipe: Efficient Pipeline Model Parallelism for Multi-GPU DNN Training](https://arxiv.org/pdf/1911.04610.pdf)
- https://arxiv.org/pdf/1806.03377.pdf

[PipeDream, a system developed as part of Microsoft Research’s Project Fiddle, introduces pipeline parallelism, a new way to parallelize DNN training by combining traditional intra-batch parallelism (model and data parallelism) with inter-batch parallelism (pipelining).](https://www.microsoft.com/en-us/research/blog/pipedream-a-more-effective-way-to-train-deep-neural-networks-using-pipeline-parallelism/)

- [Multi-tenant GPU Clusters for Deep Learning Workloads: Analysis and Implications](https://www.microsoft.com/en-us/research/uploads/prod/2018/05/gpu_sched_tr.pdf)
- [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://www.microsoft.com/en-us/research/blog/pipedream-a-more-effective-way-to-train-deep-neural-networks-using-pipeline-parallelism/)


#### AdaptDL

AdaptDL is a resource-adaptive deep learning (DL) training and scheduling framework. The goal of AdaptDL is to make distributed DL easy and efficient in dynamic-resource environments such as shared clusters and the cloud.

- https://adaptdl.readthedocs.io/
- https://github.com/petuum/adaptdl


### Efficient Communication  for Distributed Training

The communication cost of distributed training depends on the content whcih the distributed machiens share.

- https://www.usenix.org/conference/hotedge18/presentation/tao
- [eSGD: Communication Efficient Distributed Deep Learning on the Edge](https://www.usenix.org/conference/hotedge18/presentation/tao)
- https://www.run.ai/
- https://theaisummer.com/distributed-training/
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a.html)
- https://www.usenix.org/system/files/conference/atc17/atc17-zhang.pdf
- https://embedl.ai/
- [Communication-Efficient Federated Deep Learning With Layerwise Asynchronous Model Update and Temporally Weighted Aggregation](https://ieeexplore.ieee.org/document/8945292)
- [Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters](https://web.kaust.edu.sa/Faculty/MarcoCanini/classes/CS345/S19/papers/optimus.pdf)
- [CS-345, Spring 2019: Advanced Distributed and Networked Systems](https://web.kaust.edu.sa/Faculty/MarcoCanini/classes/CS345/S19/)
- http://www.cs.cmu.edu/~muli/file/parameter_server_nips14.pdf
- http://www1.se.cuhk.edu.hk/~htwai/oneworld/pdf/ji_SP.pdf
- [GPU Direct RDMA](https://clustarai.com/coreTech/0)

#### DeepSpeed

The DeepSpeed API is a lightweight wrapper on PyTorch. 
This means that you can use everything you love in PyTorch and without learning a new platform. 
In addition, DeepSpeed manages all of the boilerplate state-of-the-art training techniques, such as `distributed training, mixed precision, gradient accumulation, and checkpoints` so that you can focus on your model development. 
Most importantly, you can leverage the distinctive efficiency and effectiveness benefit of DeepSpeed to boost speed and scale with just a few lines of code changes to your PyTorch models.

- [ZeRO-2 & DeepSpeed: Shattering barriers of deep learning speed & scale](https://www.microsoft.com/en-us/research/blog/zero-2-deepspeed-shattering-barriers-of-deep-learning-speed-scale/)
- https://www.deepspeed.ai/

#### NCCL

NCCL (pronounced "Nickel") is a stand-alone library of standard communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, as well as any send/receive based communication pattern. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

- https://github.com/NVIDIA/nccl
- https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html

### Gradient Code and Compression

Gradient code and compression is to accelerate the distributed training of deep learning models by reducing the communication cost.

- [Memory-Efficient Adaptive Optimization](https://papers.nips.cc/paper/9168-memory-efficient-adaptive-optimization.pdf)

#### Gradient Code and Approximate Gradient Coding

Some nodes may be much slower than others parallelized or distributed computation enviroment.
With redundancy, we may not need every node to finish.
See [prior articles](https://zachcharles.com/files/Presentations/sbc_slides.pdf) for more areticles on this topic.

`Approximate gradient coding` allows us to tolerate more stragglers with less work.

- [Gradient Coding](https://arxiv.org/abs/1612.03301)
- [Reducing the Average Delay in Gradient Coding](https://www.ntu.edu.sg/home/hmkiah/docs/papers/averageDelayGradientCoding.pdf)
- [Tree Gradient Coding](https://www.ece.ucsb.edu/~ramtin/Tree_Gradient_Coding_final.pdf)
- [Communication-computation efficient gradient coding](https://www.researchwithnj.com/en/publications/communication-computation-efficient-gradient-coding)
- [Approximate Gradient Coding via Sparse Random Graphs](https://arxiv.org/abs/1711.06771)
- [ErasureHead: Distributed Gradient Descent without Delays Using Approximate Gradient Coding](https://hwang595.github.io/publications/erasurehead_2019/)
- https://users.oden.utexas.edu/~leiqi/stragglers_nips.pdf
- https://zachcharles.com/files/Presentations/sbc_slides.pdf
- [Ternary Gradients to Reduce Communication in Distributed Deep Learning (TensorFlow)](https://github.com/wenwei202/terngrad)
***
* https://hwang595.github.io/publications/
* http://papail.io/
* http://kangwooklee.com/
* https://www.ntu.edu.sg/home/hmkiah/
* https://users.oden.utexas.edu/~leiqi/
* https://zachcharles.com/

#### Gradient Compression

In `distributed training` of machine learning models with stochastic optimization, the exchange of parameter updates between workers often is a bottleneck that limits the scalability of distributed training. 
This is especially true for models with a large parameter space, such as neural networks. 
Several techniques have been proposed to enhance scalability by `compressing gradients`, 
e.g. by sending a sparse set of coordinates only, or by quantization. 
We study the gradient compression literature from both sides: on the one hand, [we study properties of these algorithms in a distributed setting, and their effectiveness for speed and scalability. On the other hand, we explore properties of the minima found by these algorithms, such as robustness or generalisation.](https://memento.epfl.ch/event/gradient-compression-techniques-to-accelerate-dist/)

#### Deep Gradient Compression @ MIT

- [Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://arxiv.org/abs/1712.01887)
- https://blog.csdn.net/a609640147/article/details/90054754
- https://songhan.mit.edu/
- [Deep Leakage from Gradients](https://arxiv.org/pdf/1906.08935.pdf)
- https://littleorange.site/2019/02/deep-gradient-compression.html
- [Training Kinetics in 15 Minutes: Large-scale Distributed Training on Videos](https://arxiv.org/abs/1910.00932)

##### Gradient Compression @ epfl

We study gradient compression methods to alleviate the communication bottleneck in data-parallel distributed optimization. 
Despite the significant attention received, current compression schemes either do not scale well or fail to achieve the target test accuracy. 
[We propose a new low-rank gradient compressor based on power iteration that can](https://github.com/epfml/powersgd) 
i) compress gradients rapidly, ii) efficiently aggregate the compressed gradients using all-reduce, and iii) achieve test performance on par with SGD. 
The proposed algorithm is the only method evaluated 
that achieves consistent wall-clock speedups when benchmarked against regular SGD with an optimized communication backend. 
We demonstrate reduced training times for convolutional networks as well as LSTMs on common datasets.


- [Gradient Compression Techniques to Accelerate Distributed Training of Neural Networks](https://memento.epfl.ch/event/gradient-compression-techniques-to-accelerate-dist/)
- [QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding](https://arxiv.org/pdf/1610.02132.pdf)
- [Sparse Binary Compression: Towards Distributed Deep Learning with minimal Communication](https://arxiv.org/abs/1805.08768)
- [ATOMO: Communication-efficient Learning via Atomic Sparsification](https://arxiv.org/abs/1806.04090)
- [Error Feedback Fixes SignSGD and other Gradient Compression Schemes](https://arxiv.org/abs/1901.09847)
- https://github.com/epfml/error-feedback-SGD
- [PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization](https://arxiv.org/pdf/1905.13727.pdf)
- https://github.com/epfml/powersgd
- [Decentralized SGD and Consensus with Communication Compression](https://github.com/epfml/ChocoSGD)

----

- https://sstich.ch/
- https://tvogels.nl/
- https://parsa.epfl.ch/~falsafi/
- https://lts4.epfl.ch/people/people-current/frossard-2/
- http://www.martinjaggi.ch/
- http://www.sysml.cc/
- https://sstich.ch/
- https://www.epfl.ch/labs/mlo/

##### Gradient Compression @ Edinburgh

* [Sparse Communication for Distributed Gradient Descent](https://aclweb.org/anthology/D17-1045/)
* https://www.kheafield.com/
* https://www.kheafield.com/papers/
* http://fusharblog.com/archive/

##### Gradient Compression @ kaust

- [Gradient compression for distributed training of machine learning models](https://vsrp.kaust.edu.sa/Pages/Gradient%20compression%20for%20distributed%20training%20of%20machine%20learning%20models.aspx)
- https://www.maths.ed.ac.uk/~prichtar/i_papers.html
* https://richtarik.org/i_papers.html
* https://richtarik.org/i_team.html
* https://www.maths.ed.ac.uk/~prichtar/i_seminar.html
* [Better Communication Complexity for Local SGD](https://arxiv.org/abs/1909.04746)
* [Gradient Descent with Compressed Iterates](https://arxiv.org/abs/1909.04716)


#### Count-Sketches

Sketch is a class of algorithms using a probabilistic data structure to approximate the distribution of input data.

[The count-sketches streaming algorithm instantiates the following framework](https://stackoverflow.com/questions/6811351/explaining-the-count-sketch-algorithm):

1. Find a randomized streaming algorithm whose output (as a random variable) has the desired expectation but usually high variance (i.e., noise).
2. To reduce the variance/noise, run many independent copies in parallel and combine their outputs.

- [Compressing Gradient Optimizers via Count-Sketches](https://arxiv.org/abs/1902.00179)
- [MISSION: Ultra Large-Scale Feature Selection using Count-Sketches](https://arxiv.org/abs/1806.04310)
- [Buffered Count-Min Sketch on SSD: Theory and Experiments](https://arxiv.org/abs/1804.10673)
- https://github.com/rdspring1/Count-Sketch-Optimizers
- http://net.pku.edu.cn/~yangtong/
- https://dblp.org/pers/hd/k/Kyrillidis:Anastasios
- https://www.csauthors.net/vijai-mohan/
- [SketchML: Accelerating Distributed Machine Learning with Data Sketches](http://net.pku.edu.cn/~yangtong/uploads/SketchML.pdf)

### Synthetic gradient

> It’s a simple idea: rather than compute gradients through backpropagation, we can train a model to predict what those gradients will be, and use our prediction to update our weights. It’s dynamic programming for neural networks.

-  https://github.com/andrewliao11/dni.pytorch
-  [Understanding Synthetic Gradients and Decoupled Neural Interfaces](https://arxiv.org/pdf/1703.00522.pdf)
-  [Synthetic Gradients with Tensorflow](https://r2rt.com/synthetic-gradients-with-tensorflow.html)
-  http://deliprao.com/archives/187
-  [Exploring the Use of Synthetic Gradients for Distributed Deep Learning across Cloud and Edge Resources](https://www.usenix.org/conference/hotedge19/presentation/chen)
-  [Decoupled Neural Interfaces using Synthetic Gradients](https://arxiv.org/pdf/1608.05343.pdf)
-  http://cnichkawde.github.io/SyntheticGradients.html
-  [A Bird's Eye View of Synthetic Gradients](https://greydanus.github.io/2016/11/26/synthetic-gradients/)
-  https://iamtrask.github.io/2017/03/21/synthetic-gradients/

### Gradient Centralization

> Gradient Centralization (GC) is a simple and effective optimization technique for Deep Neural Networks (DNNs), which operates directly on gradients by centralizing the gradient vectors to have zero mean. It can both speedup training process and improve the final generalization performance of DNNs.

- http://www.breloff.com/
- https://github.com/Yonghongwei/Gradient-Centralization- [Gradient Centralization: A New Optimization Technique for Deep Neural Networks](https://www4.comp.polyu.edu.hk/~cslzhang/paper/conf/ECCV20/ECCV_GC.pdf)

Ranger now uses Gradient Centralization by default, and applies it to all conv and fc layers by default. However, everything is customizable so you can test with and without on your own datasets. (Turn on off via "use_gc" flag at init).

- [Ranger - a synergistic optimizer combining RAdam (Rectified Adam) and LookAhead, and now GC (gradient centralization) in one optimization](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
- https://github.com/lessw2020/Ranger-Mish-ImageWoof-5

### Privacy and Security


Existing security mechanisms for high-performance and distributed computing infrastructure are complex and difficult to deploy.
As a result, many high-performance and distributed computing facilities do no deploy sufficient security mechanisms. 
This has prevented privacy-sensitive applications, such as those in the medical fields, and security-sensitive applications from using such facilities.
[In this project, we will develop and deploy DICE, Data Insurance in the Cluster Environment, to enhance the security in HPC and distributed computing clusters.](http://explorer.cs.fsu.edu/research/) 
DICE will consist of three major components: a container-based virtual cluster, a component to defend against side-channel attacks, and a secure execution ledger for auditing. 
The container-based virtual cluster will be developed based on the Docker Linux container. 
The `Docker security` mechanism will be enhanced by deploying an effective key management scheme for groups and by reducing the attack surface exposed to containers. 
Novel defense mechanisms will be developed and deployed to defend against side-channel attacks in the cluster environment by exploiting new security features in the recent processors. 
The secure execution ledger will provide a global holistic view of program execution in the whole system, enabling auditing the behavior of individual user as well as user groups. 
DICE essentially creates a two-level security model: on the (physical) cluster level, a group of (mostly) mutually trusted users share a single virtual cluster for their jobs; and inside the virtual cluster, the group may use existing security mechanisms of their software-of-choice to further refine security.

<img src="- https://www.microsoft.com/en-us/research/uploads/prod/2019/10/Figure3_pipedream.png" width="80%"/>


- [Deep Leakage from Gradients](https://arxiv.org/abs/1906.08935)
- http://www.cs.cornell.edu/~shmat/research.html


### Distributed deep learning libraries

- https://github.com/amplab/drizzle-spark
- https://github.com/horovod/horovod
- http://maxpumperla.com/elephas/
- https://horovod.ai/
- http://explorer.cs.fsu.edu/

####  Deep learning + Spark

BigDL is a distributed deep learning library for Apache Spark; with BigDL, users can write their deep learning applications as standard Spark programs, which can directly run on top of existing Spark or Hadoop clusters. To makes it easy to build Spark and BigDL applications, a high level Analytics Zoo is provided for end-to-end analytics + AI pipelines.

- [BigDL is a distributed deep learning library for Apache Spark](https://bigdl-project.github.io/master/)
- https://github.com/intel-analytics/BigDL/

> Drizzle is a low latency execution engine for Apache Spark that is targeted at stream processing and iterative workloads. Currently, Spark uses a BSP computation model, and notifies the scheduler at the end of each task. Invoking the scheduler at the end of each task adds overheads and results in decreased throughput and increased latency.

- https://github.com/amplab/drizzle-spark
- https://shivaram.org/drafts/drizzle.pdf

> Horovod can additionally run on top of Apache Spark, making it possible to unify data processing and model training into a single pipeline. Once Horovod has been configured, the same infrastructure can be used to train models with any framework, making it easy to switch between TensorFlow, PyTorch, MXNet, and future frameworks as machine learning tech stacks continue to evolve.


- https://horovod.ai/
- https://eng.uber.com/horovod/
- https://github.com/horovod/horovod

Elephas is an extension of Keras, which allows you to run distributed deep learning models at scale with Spark.

- http://maxpumperla.com/elephas/
- https://github.com/maxpumperla/elephas

## Products and Packages

- https://zhuanlan.zhihu.com/p/87392811
- https://zhuanlan.zhihu.com/p/87458316
- https://www.dataiku.com/
- https://www.fritz.ai/

### TensorFlow

TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML-powered applications.

- https://www.tensorflow.org/resources/tools
-  https://github.com/tensorflow/tensorflow
-  https://www.tensorflow.org/lite?hl=zh-cn
-  https://github.com/tensorflow/tflite-support

### PyTorch

PyTorch is a Python package that provides two high-level features:

- Tensor computation (like NumPy) with strong GPU acceleration
- Deep neural networks built on a tape-based autograd system

PyTorch has minimal framework overhead. We integrate acceleration libraries such as Intel MKL and NVIDIA (cuDNN, NCCL) to maximize speed. At the core, its CPU and GPU Tensor and neural network backends (TH, THC, THNN, THCUNN) are mature and have been tested for years.

Hence, PyTorch is quite fast – whether you run small or large neural networks.

The memory usage in PyTorch is extremely efficient compared to Torch or some of the alternatives. We've written custom memory allocators for the GPU to make sure that your deep learning models are maximally memory efficient. This enables you to train bigger deep learning models than before.

- https://pytorch.org/
- https://github.com/pytorch/pytorch
- https://pytorch-lightning.readthedocs.io/en/0.5.3.2/Trainer/Distributed%20training/
- https://pytorch.org/mobile/home/
- https://github.com/rusty1s/pytorch_geometric

### MNN

MNN is a highly efficient and lightweight deep learning framework. It supports inference and training of deep learning models, and has industry leading performance for inference and training on-device. At present, MNN has been integrated in more than 20 apps of Alibaba Inc, such as Taobao, Tmall, Youku, Dingtalk, Xianyu and etc., covering more than 70 usage scenarios such as live broadcast, short video capture, search recommendation, product searching by image, interactive marketing, equity distribution, security risk control. In addition, MNN is also used on embedded devices, such as IoT.

- [开源背后 | 面对端侧推理引擎的挑战，阿里工程师如何应对？ - 阿里技术的文章 - 知乎](https://zhuanlan.zhihu.com/p/71816894)
- https://github.com/alibaba/MNN
- https://www.mnn.zone/

### TNN

TNN is a high-performance and lightweight inference framework for mobile devices. It provides lots of advanced features such as cross-platform, model-compression, and code-pruning. TNN, inspired by mainstream open-source industry frameworks, integrates and leverages Youtu Lab's Rapidnet, ncnn framework.


- https://github.com/Tencent/TNN
  
> ncnn is a high-performance neural network inference computing framework optimized for mobile platforms. ncnn is deeply considerate about deployment and uses on mobile phones from the beginning of design. ncnn does not have third party dependencies. it is cross-platform, and runs faster than all known open source frameworks on mobile phone cpu. Developers can easily deploy deep learning algorithm models to the mobile platform by using efficient ncnn implementation, create intelligent APPs, and bring the artificial intelligence to your fingertips. ncnn is currently being used in many Tencent applications, such as QQ, Qzone, WeChat, Pitu and so on.

- https://github.com/Tencent/ncnn


###  PaddlePaddle

PaddlePaddle, as the only independent R&D deep learning platform in China, has been officially open-sourced to professional communities since 2016. It is an industrial platform with advanced technologies and rich features that cover core deep learning frameworks, basic model libraries, end-to-end development kits, tools & components as well as service platforms. PaddlePaddle is originated from industrial practices with dedication and commitments to industrialization. It has been widely adopted by a wide range of sectors including manufacturing, agriculture, enterprise service, and so on while serving more than 2.3 million developers. With such advantages, PaddlePaddle has helped an increasing number of partners commercialize AI.

- https://www.paddlepaddle.org.cn/
- https://github.com/PaddlePaddle/Paddle

#### DasyDL

- https://ai.baidu.com/easydl/


### MindSpore

>  MindSpore is a new open source deep learning training/inference framework that could be used for mobile, edge and cloud scenarios. MindSpore is designed to provide development experience with friendly design and efficient execution for the data scientists and algorithmic engineers, native support for Ascend AI processor, and software hardware co-optimization. At the meantime MindSpore as a global AI open source community, aims to further advance the development and enrichment of the AI software/hardware application ecosystem.

- https://github.com/mindspore-ai/mindspore
- https://www.mindspore.cn/
- https://gitee.com/mindspore/mindspore

#### ModelArts

- https://www.huaweicloud.com/product/modelarts.html


### MegEngine
MegEngine is a fast, scalable and easy-to-use deep learning framework, with auto-differentiation.


- https://megengine.org.cn/
- https://www.openi.org.cn/html/2020/Framework_0325/18.html
- https://github.com/MegEngine/MegEngine

### Oneflow

Oneflow is an open source deep learning platform with whole new frame design and the world's leading technology for distributed system.

- https://github.com/Oneflow-Inc/oneflow
- http://www.oneflow.org/index.html
- http://docs.oneflow.org/

## Edge Computation

Edge computation is to perform some computation on the edge devices such as the monior in order to send less raw data to the computation center.


> Machine learning models for edge devices need to have a small footprint in terms of storage, prediction latency, and energy. One instance of where such models are desirable is resource-scarce devices and sensors in the Internet of Things (IoT) setting. Making real-time predictions locally on IoT devices without connecting to the cloud requires models that fit in a few kilobytes.

- https://github.com/microsoft/EdgeML
- https://github.com/fritzlabs/Awesome-Mobile-Machine-Learning
- https://github.com/Bisonai/awesome-edge-machine-learning
- https://github.com/hoangcuong2011/Good-Papers
- https://github.com/rkirkendall/MLEdgeDeploy
- https://github.com/sig-ml/bleedml
- https://www.usenix.org/conference/hotedge18
- https://project-sparthan.github.io/


### Mobile Deep Learning

Mobile deep learning is aimed to run deep learning models (training or inference) on the mobile phones.

It is necessary to compress the deep learning models in order to run it in mobile phones.

- https://www.fritz.ai/resources/mobile-ml-github-repositories.html
- https://www.sigmobile.org/mobisys/2019/workshops/deepmobile19/
- https://github.com/amusi/mobile-deep-learning
- https://github.com/pytorch/qnnpack
- https://mi.inf.ed.ac.uk/
- https://www-users.cs.umn.edu/~fengqian/
- http://homepages.inf.ed.ac.uk/ppatras/
- https://www.cs.ucr.edu/~jiasi/
- https://ix.cs.uoregon.edu/~jiao/
- https://arxiv.org/abs/1904.09274
- https://www.zhqiang.org/deep-learning-mobile/
- https://bdtechtalks.com/2019/12/30/computer-vision-applications-deep-learning/
- [MuffNet: Multi-Layer Feature Federation for Mobile Deep Learning](http://openaccess.thecvf.com/content_ICCVW_2019/papers/CEFRL/Chen_MuffNet_Multi-Layer_Feature_Federation_for_Mobile_Deep_Learning_ICCVW_2019_paper.pdf)
- [Deep Learning for Mobile Multimedia: A Survey](https://www.iti.gr/~bmezaris/publications/tomm17_survey_preprint.pdf)

####  Toolkits


- https://github.com/pytorch/qnnpack
- https://github.com/PaddlePaddle/Paddle-Lite
- https://paddlepaddle.github.io/Paddle-Lite/
- https://github.com/alibaba/MNN
- https://www.tensorflow.org/lite
- https://project-sparthan.github.io/papers/acm19.pdf
- https://github.com/XiaoMi/mace
- https://mace.readthedocs.io/en/latest/
- https://github.com/OAID/Tengine
****

|DNN Acceleratation Framewore|
|---|
|https://hgpu.org/|
|NVIDIA|
|https://developer.nvidia.com/cudnn)|
|http://nvdla.org/|
|https://docs.nvidia.com/cuda/|
|https://developer.nvidia.com/tensorrt|
|[cupy](https://cupy.chainer.org/)|
|intel|
|[ideep](https://github.com/intel/ideep)
|[Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)](https://github.com/intel/mkl-dnn)|
|https://github.com/intel/onnxruntime|
|[nGraph](https://www.ngraph.ai/),[PlaidML](https://www.intel.ai/plaidml/)|
|https://intel.github.io/mkl-dnn/|
|[Reference workloads for modern deep learning methods.](https://rdadolf.github.io/fathom/)|
|[Minerva: a fast and flexible tool for deep learning on multi-GPU.](https://github.com/dmlc/minerva)|
|[SigDL -- Deep Learning for IoT Device and Edge Computing Embedded Targets](https://github.com/signalogic/SigDL#DeepLearningModelCompression)|
|[Menoh: fast DNN inference library with multiple programming language support](https://github.com/pfnet-research/menoh)|
|[trillium](https://www.arm.com/products/silicon-ip-cpu/machine-learning/project-trillium)|
|https://github.com/alibaba/MNN|
|https://github.com/sql-machine-learning/elasticdl|
|https://github.com/Tencent/TNN|


*****

|Model Compression Packages|
|---|
|[Distiller is an open-source Python package for neural network compression research.](https://nervanasystems.github.io/distiller/index.html)|
|[PocketFlow](https://pocketflow.github.io/)|
|[PocketFlow中的模型压缩算法](https://zhuanlan.zhihu.com/c_1041626714043949056)|
|[BNN](https://yq.aliyun.com/articles/230662)|
|[PERMDNN: Efficient Compressed DNN Architecture with Permuted Diagonal Matrices](http://alchem.usc.edu/portal/static/download/permdnn.pdf)|
|[knowledge-distillation-pytorch](https://github.com/peterliht/knowledge-distillation-pytorch)|
|[keras_compressor](https://github.com/DwangoMediaVillage/keras_compressor)|
|[TensorFlow Lite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite), [tensorflow-compression](https://tensorflow.github.io/compression/)|
|[TensorRT](https://github.com/NVIDIA/TensorRT)|
|https://github.com/Tencent/ncnn|
|[Introduction to Intel® Deep Learning Deployment Toolkit](https://docs.openvinotoolkit.org/latest/_docs_IE_DG_Introduction.html)|

## Tool kits

- https://www.wandb.com/
- https://github.com/fastnlp/fitlog
- https://optuna.org/
- https://preferred.jp/en/projects/optuna/
- https://lutzroeder.github.io/netron/
- https://netron.app/
- https://github.com/lutzroeder/netron
- https://github.com/onnx/onnx
- https://github.com/daquexian/onnx-simplifier

-------

- https://github.com/drcut/CPD
- https://github.com/purine/purine2
- https://dlonsc19.github.io/
- http://prace.it4i.cz/en/HPDDL-01-2018
