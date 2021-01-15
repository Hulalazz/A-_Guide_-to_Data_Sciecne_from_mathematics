# AutoML

AutoML is aimed to find the best algorithm for the specific task.

AutoML draws on many disciplines of machine learning, prominently including

* Bayesian optimization;
* Meta learning;
* Transfer learning;
* Regression models for structured data and big data;
* Combinatorial optimization.

Bayesian optimization is introduced in Bayesian Learning. Meta learning will extends beyond the stacking generalization.

AutoML is regarded as an approach to general artificial intelligence.
Different from the supervised machine learning, AutoML is to tune the hyper-parameters,
which is handed tuned in usual algorithm.
Evolutionary or genetic algorithms are applied to solve some unstructured problems.

* https://www.ml4aad.org/
* https://www.automl.org/
* https://www.automl.org/automl/auto-sklearn/
* https://www.automl.org/wp-content/uploads/2019/05/AutoML_Book.pdf
* https://mlbox.readthedocs.io/en/latest/
* [Introduction to Auto-Keras](https://www.simonwenkel.com/2018/08/29/introduction-to-autokeras.html)
* [EasyDL定制化训练和服务平台](http://ai.baidu.com/easydl/)
* [HyperparameterHunter 3.0](https://zhuanlan.zhihu.com/p/80038497)
* https://www.autodl.com/home
* [Taking Human out of Learning Applications: A Survey on Automated Machine Learning](https://arxiv.org/abs/1810.13306)
* [automlk: Automated and distributed machine learning toolkit](https://github.com/pierre-chaville/automlk)
* [Awesome-AutoML-Papers](https://github.com/hibayesian/awesome-automl-papers)
* [AutoML总结](https://www.jianshu.com/p/8178bb4d2ec3)
* https://github.com/kevinzakka/hypersearch
* https://github.com/skorch-dev/skorch
* https://github.com/automl/Auto-PyTorch
* https://github.com/microsoft/nni
* https://github.com/hibayesian/awesome-automl-papers
* https://www.simonwenkel.com/2018/08/28/automated-machine-learning.html
* [Introduction to Automated Machine Learning Frameworks](https://www.simonwenkel.com/projects/introduction-to-automated-machine-learning-frameworks.html)
* [Does AutoML (Automated Machine Learning) lead to better models and fulfill legal requirements?](https://www.simonwenkel.com/2018/09/06/does-automl-lead-to-better-models-and-fulfill-legal-requirements.html)
* https://zhuanlan.zhihu.com/c_141907721
* https://zhuanlan.zhihu.com/c_1005865351275573248
* https://www.stat.purdue.edu/~chengg/
* https://www.datavisor.com/blog/automated-feature-engineering/


## Automated Feature Engineer



* [AutoCross: Automatic Feature Crossing for Tabular Data in Real-World Applications](https://arxiv.org/abs/1904.12857v1)
* [Neural Input Search for Large Scale Recommendation Models](https://arxiv.org/abs/1907.04471)
* https://dblp.uni-trier.de/pers/hd/t/Tang:Ruiming
* [Automatic Feature Engineering From Very High Dimensional Event Logs Using Deep Neural Networks](https://dlp-kdd.github.io/assets/pdf/a13-hu.pdf)
* https://www.featuretools.com/
* http://workshops.inf.ed.ac.uk/nips2016-ai4datasci/papers/NIPS2016-AI4DataSci_paper_13.pdf
* https://github.com/cod3licious/autofeat
* https://dspace.mit.edu/handle/1721.1/119919
* https://simplecore.intel.com/nervana/wp-content/uploads/sites/53/2018/05/IntelAIDC18_Xin-Hunt_Tron_052418_final.pdf
* https://github.com/chu-data-lab

## Meta-Learning: Learning to Learn

* [Workshop on Meta-Learning (MetaLearn 2019)](http://metalearning.ml/2019/)
* [awesome-meta-learning](https://github.com/dragen1860/awesome-meta-learning)
* [Meta-Learning: Learning to Learn Fast](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html)
* [Learning to Learn by Chelsea Finn, Jul 18, 2017](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/)
* [Learning to Learn with Probabilistic Task Embeddingsby Kate Rakelly, Jun 10, 2019](https://bair.berkeley.edu/blog/2019/06/10/pearl/)

### Learning to Optimize

[We may find that the following facts:](https://bair.berkeley.edu/blog/2017/09/12/learning-to-optimize-with-rl/)
>> Machine learning has enjoyed tremendous success and is being applied to a wide variety of areas, both in AI and beyond. This success can be attributed to the data-driven philosophy that underpins machine learning, which favours automatic discovery of patterns from data over manual design of systems using expert knowledge.

>> Yet, there is a paradox in the current paradigm: the algorithms that power machine learning are still designed manually. This raises a natural question: can we learn these algorithms instead? This could open up exciting possibilities: we could find new algorithms that perform better than manually designed algorithms, which could in turn improve learning capability.


For some optimization problems, we are lucky to obtain the closed form solver such as the projector or the proximity operator.


- https://epfl-lts2.github.io/unlocbox-html/doc/prox/
- http://proximity-operator.net/
- http://foges.github.io/pogs/

The iterative optimization methods such as stochastic gradient descent, ADMM map the current parameters to the new ones closer to the optimum.
In terms of mathematics,  these methods is a fixed point mapping:
$$x^{(t+1)}=M(x^{(t)})$$
so that $f(x^{(t+1)})\leq f(x^{(t)})$ and $\lim_{t\to\infty}f(x^{(t)})=\min f(x)$under some proper conditions.

- https://deepai.org/publication/learning-to-optimize
- https://halide-lang.org/papers/halide_autoscheduler_2019.pdf
- https://ieeexplore.ieee.org/document/8444648
- http://web.stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf
- https://arxiv.org/abs/1606.01885
- https://www2.isye.gatech.edu/~tzhao80/
- https://smartech.gatech.edu/handle/1853/62075
- https://www.dataquest.io/blog/course-optimize-algorithm-complexity-for-data-engineering/
- https://www.math.ias.edu/~ke.li/
- https://arxiv.org/abs/1703.00441
- https://lib.dr.iastate.edu/etd/17329/
- https://papers.nips.cc/paper/2018/file/8b5700012be65c9da25f49408d959ca0-Paper.pdf
- https://openaccess.thecvf.com/content_CVPR_2020/papers/Gao_Learning_to_Optimize_on_SPD_Manifolds_CVPR_2020_paper.pdf
- https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Learning_to_Optimize_Non-Rigid_Tracking_CVPR_2020_paper.pdf

## Learning-based methods

- https://www.microsoft.com/en-us/research/publication/learning-partial-differential-equations-for-computer-vision/
- https://zero-lab-pku.github.io/
- http://people.inf.ethz.ch/liup/
- http://www.yongxu.org/
- http://www.iro.umontreal.ca/~memisevr/talks/memisevicIcisp2016.pdf
- https://www.pantechsolutions.net/blog/image-processing-projects-2019/

### Learnable Optimization

Several recent studies build deep structures by unrolling a particular optimization model that involves task information, i.e., `learning to optime`.

Like gradient boost decision tree, we can optimize a cost function with a machine learning algorithms to fit the gradients- that is so-called gradient boost machine.
In another hand, it is expected that machine learning could learn/approximate the ierative formula of any optimization algorithms.

- [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474)

`learning-based iterative methods `

[Numerous tasks at the core of statistics, learning and vision areas are specific cases of ill-posed inverse problems. Recently, learning-based (e.g., deep) iterative methods have been empirically shown to be useful for these problems. Nevertheless, integrating learnable structures into iterations is still a laborious process, which can only be guided by intuitions or empirical insights. Moreover, there is a lack of rigorous analysis about the convergence behaviors of these reimplemented iterations, and thus the significance of such methods is a little bit vague. This paper moves beyond these limits and proposes Flexible Iterative Modularization Algorithm (FIMA), a generic and provable paradigm for nonconvex inverse problems. Our theoretical analysis reveals that FIMA allows us to generate globally convergent trajectories for learning-based iterative methods. Meanwhile, the devised scheduling policies on flexible modules should also be beneficial for classical numerical methods in the nonconvex scenario. Extensive experiments on real applications verify the superiority of FIMA.](http://dutmedia.org/FIMA/)

- https://github.com/Heyi007/FIMA
- https://ieeexplore.ieee.org/abstract/document/8727950
- [Proximal Alternating Direction Network: A Globally Converged Deep Unrolling Framework](https://arxiv.org/abs/1711.07653)
- [A Bridging Framework for Model Optimization and Deep Propagation](http://papers.nips.cc/paper/7685-a-bridging-framework-for-model-optimization-and-deep-propagation)
- [On the Convergence of Learning-based Iterative Methods for Nonconvex Inverse Problems](http://dutmedia.org/FIMA/)
- https://locuslab.github.io/2019-10-28-cvxpylayers/
- https://arxiv.org/abs/1204.4145
- https://github.com/pepper-johnson/Learnable-Optimization
- http://stanford.edu/~qysun/NIPS2018-neural-proximal-gradient-descent-for-compressive-imaging.pdf

### Deep Unrolling



[The move from hand-designed features to learned features in machine learning has been wildly successful. In spite of this, optimization algorithms are still designed by hand. In this paper we show how the design of an optimization algorithm can be cast as a learning problem, allowing the algorithm to learn to exploit structure in the problems of interest in an automatic way. Our learned algorithms, implemented by LSTMs, outperform generic, hand-designed competitors on the tasks for which they are trained, and also generalize well to new tasks with similar structure. We demonstrate this on a number of tasks, including simple convex problems, training neural networks, and styling images with neural art.](https://arxiv.org/abs/1606.04474)



- [Deep unrolling](https://zhuanlan.zhihu.com/p/44003318)
- http://dutmedia.org/
- http://dlutir.dlut.edu.cn/Scholar/Detail/6711
- https://dblp.uni-trier.de/pers/hd/l/Liu:Risheng
- https://github.com/dlut-dimt
- https://www.researchgate.net/project/optimization-numerical-computation-optimal-control
- https://ankita-shukla.github.io/
- [Neural-network-based iterative learning control of nonlinear systems](https://www.sciencedirect.com/science/article/abs/pii/S0019057819303908)
- http://bcmi.sjtu.edu.cn/home/niuli/download/intro_deep_unrolling.pdf
- https://arxiv.org/abs/1912.10557
- https://papers.nips.cc/paper/7685-a-bridging-framework-for-model-optimization-and-deep-propagation.pdf
- https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_DNU_Deep_Non-Local_Unrolling_for_Computational_Spectral_Imaging_CVPR_2020_paper.pdf
- https://github.com/ethliup/DeepUnrollNet
- http://faculty.dlut.edu.cn/rsliu/zh_CN/zdylm/983095/list/index.htm
- https://scholar.google.com/citations?user=DzuhImQAAAAJ&hl=en
- https://dblp.org/pers/l/Liu:Risheng.html

## Transfer Learning

> This half-day workshop (in conjunction with IEEE Big Data 2019) is a continuation of our past Big Data Transfer Learning (BDTL) workshops (1st BDTL, 2nd BDTL, 3rd BDTL) 
> which will provide a focused international forum to bring together researchers and research groups to review the status of transfer learning on both conventional vectorized features and heterogeneous information networks. 

* https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf
* http://www.cad.zju.edu.cn/home/dengcai/
* http://clopinet.com/isabelle/Projects/ICML2011/
* http://emergingtechnet.org/DTL2019/default.php
* https://lld-workshop.github.io/
* http://www.cis.umassd.edu/~mshao/BDTL2018/index.html
* https://www.computefest.seas.harvard.edu/transfer-learning
* https://sites.google.com/view/icml2019-generalization/home
* [迁移学习 Transfer Learning](http://transferlearning.xyz/)
* [NVIDIA Transfer Learning Toolkit: High level SDK for tuning of domain specific DNNs](https://developer.nvidia.com/transfer-learning-toolkit)

## AutoDL

* https://autodl.chalearn.org/
* https://autodl-community.github.io/autodl-irssi/
* [NeurIPS AutoDL challenges:  AutoDL 2019](https://autodl.chalearn.org/)
* http://imarkserve.iws.in/blog/
* https://zhuanlan.zhihu.com/p/82192726
* [Discovering the best neural architectures in the continuous space](https://www.microsoft.com/en-us/research/blog/discovering-the-best-neural-architectures-in-the-continuous-space/?OCID=msr_blog_neuralarchitecture_neurips_hero)
* https://arxiv.org/abs/1905.00424


### NAS

- [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)
- [Progressive Neural Architecture Search](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf)
- https://www.bbsmax.com/A/QW5Y4a1eJm/
- http://valser.org/article-336-1.html

### Differentiable Neural Architecture Search

The algorithm is based on `continuous relaxation and gradient descent` in the architecture space. 
[It is able to efficiently design high-performance convolutional architectures for image classification (on CIFAR-10 and ImageNet) and recurrent architectures for language modeling (on Penn Treebank and WikiText-2).](https://github.com/quark0/darts)

- https://github.com/quark0/darts
- https://github.com/chenxin061/pdarts
- https://github.com/xiaomi-automl/
- https://github.com/1adrianb/binary-nas
- https://github.com/xiangning-chen/SmoothDARTS
- https://www.eye-on.ai/ai-research-watch-papers/2020/7/13/2020713-neural-papers
- https://www.ml4aad.org/automl/literature-on-neural-architecture-search/
- https://deepai.org/publication/stabilizing-differentiable-architecture-search-via-perturbation-based-regularization

### Optimization based Pruning for DNN Compression

- http://www.cse.ust.hk/~qyaoaa/papers/talk4thpara.pdf
- https://www.ijcai.org/Proceedings/2018/330
- http://www.lamda.nju.edu.cn/qianc/ijcai18-olmp.pdf
- https://github.com/IdiosyncraticDragon/OLMP
- https://papers.nips.cc/paper/9521-autoprune-automatic-network-pruning-by-regularizing-auxiliary-parameters.pdf
- https://csyhhu.github.io/data/L-DNQ.pdf


#### AOGNet

An AOGNet consists of a number of stages each of which is composed of a number of AOG building blocks. 
An AOG building block splits its input feature map into $N$ groups along feature channels and then treat it as a sentence of $N$ words. 
It then jointly realizes a phrase structure grammar and a dependency grammar in bottom-up parsing the “sentence” for better feature exploration and reuse. 
[It provides a unified framework for the best practices developed in state-of-the-art DNNs.](https://arxiv.org/pdf/1711.05847.pdf)

`We first need to understand the underlying wisdom in designing better network architectures: It usually lies in finding network structures
which can support flexible and diverse information flows for exploring new features, reusing existing features in previous layers 
and back-propagating learning signals (e.g., gradients).`

- [AOGNets: Compositional Grammatical Architectures for Deep Learning](https://arxiv.org/pdf/1711.05847.pdf)
- http://www.stat.ucla.edu/~tfwu/
- http://www.stat.ucla.edu/~tfwu//project_posts/iRCNN/
- https://github.com/xilaili/AOGNet
- https://github.com/iVMCL/AOGNets
- http://www.stat.ucla.edu/~tfwu/project_posts/AOGNets/
- [New Framework Improves Performance of Deep Neural Networks](https://research.ece.ncsu.edu/ivmcl/2019/05/29/new-framework-improves-performance-of-deep-neural-networks/)
- http://bbs.cvmart.net/topics/959
- http://www.stat.ucla.edu/~tfwu/project_posts/AOGNets/
- https://github.com/tfwu

### Weight Agnostic Neural Networks

- https://arxiv.org/pdf/1906.04358.pdf
- http://www.sohu.com/a/320257682_100024677
- https://weightagnostic.github.io/
- https://weightagnostic.github.io/slides/wann_slides.pdf
- https://ai.googleblog.com/2019/08/exploring-weight-agnostic-neural.html
- https://chinagdg.org/2019/08/exploring-weight-agnostic-neural-networks/
- https://torontoai.org/
- https://bioreports.net/weight-agnostic-neural-networks/

  
## AutoNLP

- https://mc.ai/automl-and-autodl-simplified/
- [AMIR 2019 : 1st Interdisciplinary Workshop on Algorithm Selection and Meta-Learning in Information Retrieval](http://amir-workshop.org/)
- https://github.com/DeepBlueAI/AutoNLP
- https://www.4paradigm.com/competition/autoNLP2019
- https://autodl.lri.fr/competitions/35

# Empirical Model Learning

* https://sites.google.com/view/boostingopt2018/
* [Empirical Model Learning: Embedding Machine Learning Models in Optimization](https://emlopt.github.io/)

# Active Learning

* https://modal-python.readthedocs.io/en/latest/
* https://www.apres.io/
* [Active Learning: Optimization != Improvement](https://www.lighttag.io/blog/active-learning-optimization-is-not-imporvement/)
* [Accelerate Machine Learning with Active Learning](https://becominghuman.ai/accelerate-machine-learning-with-active-learning-96cea4b72fdb)
* [Active Learning Tutorial, ICML 2009](http://hunch.net/~active_learning/)
* [ACTIVE LEARNING: THEORY AND APPLICATIONS](http://www.robotics.stanford.edu/~stong/papers/tong_thesis.pdf)

# Sequential Learning

* [Learning Strategic Behavior in Sequential Decision Tasks](http://www.cs.utexas.edu/users/ai-lab/?nsfri09)
* [Sequential Learning](https://webdocs.cs.ualberta.ca/~rgreiner/RESEARCH/seq.html)
* [Sequential Learning Using Incremental Import Vector Machines for Semantic Segmentation](https://d-nb.info/1043056424/34)
* [Continuous Online Sequence Learning with an Unsupervised Neural Network Model](https://numenta.com/neuroscience-research/research-publications/papers/continuous-online-sequence-learning-with-an-unsupervised-neural-network-model/)
* [SEQUENTIAL LEARNING: Iterative machine learning helps you reach your experimental design goals faster](https://citrine.io/platform/sequential-learning/)
  
<img src="https://1hrkl410nh36441q7v2112ft-wpengine.netdna-ssl.com/wp-content/uploads/2018/11/SEQ_LEARNING_FLOW_CHART_R5-768x689.jpg" width=80%/>

[A Python implementation of Online Sequential Extreme Machine Learning (OS-ELM) for online machine learning](https://github.com/leferrad/pyoselm)


