#  Back-propagation and Beyond

### Beyond Back-propagation

Training deep learning models does not require gradients such as `ADMM, simulated annealing`.

- [BinaryConnect: Training Deep Neural Networks with binary weights during propagations](https://arxiv.org/abs/1511.00363)
- [Bidirectional Backpropagation](http://sipi.usc.edu/~kosko/B-BP-SMC-Revised-13January2018.pdf)
- [Beyond Backprop: Online Alternating Minimization with Auxiliary Variables](https://www.ibm.com/blogs/research/2019/06/beyond-backprop/)
- [Beyond Backpropagation: Uncertainty Propagation](http://videolectures.net/iclr2016_lawrence_beyond_backpropagation/)
- [Beyond Feedforward Models Trained by Backpropagation: a Practical Training Tool for a More Efficient Universal Approximator](https://www.memphis.edu/clion/pdf-papers/0710.4182.pdf)
- [BEYOND BACKPROPAGATION: USING SIMULATED ANNEALING FOR TRAINING NEURAL NETWORKS](http://people.missouristate.edu/RandallSexton/sabp.pdf)
- [Main Principles of the General Theory of Neural Network with Internal Feedback](http://worldcomp-proceedings.com/proc/p2015/ICA6229.pdf)
- [Eigen Artificial Neural Networks](https://arxiv.org/pdf/1907.05200.pdf)
- [Deep Learning as a Mixed Convex-Combinatorial Optimization Problem](https://arxiv.org/abs/1710.11573)
- [Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation](https://arxiv.org/abs/1602.05179)
- https://www.ncbi.nlm.nih.gov/pubmed/28522969
- https://www.computer.org/10.1109/CVPR.2016.165
- [Efficient Training of Very Deep Neural Networks for Supervised Hashing](https://arxiv.org/abs/1511.04524)
- https://zhuanlan.zhihu.com/p/67782029
- [Biologically-Plausible Learning Algorithms Can Scale to Large Datasets](https://arxiv.org/pdf/1811.03567.pdf)
- [A Biologically Plausible Learning Algorithm for Neural Networks](https://www.ibm.com/blogs/research/2019/04/biological-algorithm/)
- [DEEP LEARNING AS A MIXED CONVEXCOMBINATORIAL OPTIMIZATION PROBLEM](https://homes.cs.washington.edu/~pedrod/papers/iclr18.pdf)
- [An Alternating Minimization Method to Train Neural Network Models for Brain Wave Classification](http://evoq-eval.siam.org/Portals/0/Publications/SIURO/Volume%2011/An_Alternating_Minimization_Method_to_Train_Neural_Network_Models.pdf?ver=2018-02-27-134920-257)

#### Operator Splitting Methods For Training Deep Neural Networks

##### ADMM

ADMM is based on the constraints of  successive layers in neural networks.
Recall the feedforward neural networks:
$$O=\sigma(W^nx^{N}+b_{N})\\
x^{n}=\sigma(W^{n-1}x^{n-1}+b_{n-1})\quad\forall n=1,\cdots, N 
$$
[The parameters in each layer are updated backward and then forward so that the parameter information in each layer is exchanged efficiently. The time complexity is reduced from cubic to quadratic in (latent) feature dimensions via a dedicated algorithm design for subproblems that enhances them utilizing iterative quadratic approximations and backtracking. Finally, we provide the first proof of global convergence for an ADMM-based method (dlADMM) in a deep neural network problem under mild conditions. Experiments on benchmark datasets demonstrated that our proposed dlADMM algorithm outperforms most of the comparison methods.](https://arxiv.org/abs/1905.13611)

- [Training Neural Networks Without Gradients: A Scalable ADMM Approach](https://arxiv.org/abs/1605.02026)
- [ADMM for Efficient Deep Learning with Global Convergence](https://arxiv.org/abs/1905.13611)
- [ADMM-CSNet: A Deep Learning Approach for Image Compressive Sensing](https://ieeexplore.ieee.org/document/8550778/)
- https://github.com/KaiqiZhang/ADAM-ADMM
- [ADMM-NN: An Algorithm-Hardware Co-Design Framework of DNNs Using Alternating Direction Method of Multipliers](https://ui.adsabs.harvard.edu/abs/2018arXiv181211677R/abstract)
- [ALTERNATING DIRECTION METHOD OF MULTIPLIERS FOR SPARSE CONVOLUTIONAL NEURAL NETWORKS](https://arxiv.org/pdf/1611.01590.pdf)
- https://patents.google.com/patent/US20170147920/fi


##### Lifted Proximal Operator Machine (LPOM) 

By rewriting the activation function as an equivalent proximal operator, 
we approximate a feed-forward neural network by adding the proximal operators to the objective function as penalties, 
hence we call the lifted proximal operator machine (LPOM). LPOM is block multi-convex in all layer-wise weights and activations. 
This allows us to use block coordinate descent to update the layer-wise weights and activations in parallel.
Most notably, we only use the mapping of the activation function itself, rather than its derivatives, thus avoiding the gradient vanishing or blow-up issues in gradient based training methods.
So our method is applicable to various non-decreasing Lipschitz continuous activation functions, which can be saturating and non-differentiable. LPOM does not require more auxiliary variables than the layer-wise activations, thus using roughly the same amount of memory as stochastic gradient descent (SGD) does. 
[We further prove the convergence of updating the layer-wise weights and activations. Experiments on MNIST and CIFAR-10 datasets testify to the advantages of LPOM.](https://arxiv.org/abs/1811.01501v1)

- [Optimization and Deep Neural Networks by Zhouchen Lin](https://slides.games-cn.org/pdf/Games201991%E6%9E%97%E5%AE%99%E8%BE%B0.PDF)
- https://zhouchenlin.github.io/
- [Lifted Proximal Operator Machines](https://arxiv.org/abs/1811.01501v1)
- [一种提升邻近算子机神经网络优化方法 [发明]](http://cprs.patentstar.com.cn/Search/Detail?ANE=9GED7DEA8EDAAICA9CGA9EFB9HHGAGGADGIA9AIFDHEA9IHH)

#### Mixed Integer Optimization

[Network models and integer programs are applicable for an enormous known variety of decision problems. Some of these decision problems are really physical problems, such as transportation or flow of commodities. Many network problems are more of an abstract representations of processes or activities, such as the critical path activity network in project management. These problems are easily illustrated by using a network of arcs, and nodes.](http://home.ubalt.edu/ntsbarsh/opre640A/partIII.htm)

- http://home.ubalt.edu/ntsbarsh/opre640A/partIII.htm
- [Strong mixed-integer programming formulations for trained neural networks by Joey Huchette1](http://www.cas.mcmaster.ca/~deza/slidesRIKEN2019/huchette.pdf)
- [Deep neural networks and mixed integer linear optimization](https://link.springer.com/article/10.1007/s10601-018-9285-6)
- [Matteo Fischetti, University of Padova](http://www.dei.unipd.it/~fisch/papers/slides/2018%20Dagstuhl%20%5BFischetti%20on%20DL%5D.pdf)
- [Deep Neural Networks as 0-1 Mixed Integer Linear Programs: A Feasibility Study](https://arxiv.org/abs/1712.06174)
- https://www.researchgate.net/profile/Matteo_Fischetti
- [A Mixed Integer Linear Programming Formulation to Artificial Neural Networks](http://www.amp.i.kyoto-u.ac.jp/tecrep/ps_file/2019/2019-001.pdf)
- [ReLU Networks as Surrogate Models in Mixed-Integer Linear Programs](http://www.optimization-online.org/DB_FILE/2019/07/7276.pdf)
- [Strong mixed-integer programming formulations for trained neural networks](http://www.optimization-online.org/DB_FILE/2018/11/6911.pdf)
- [Deep Learning in Computational Discrete Optimization CO 759, Winter 2018](https://www.math.uwaterloo.ca/~bico/co759/2018/index.html)
- https://sites.google.com/site/mipworkshop2017/program
- [Second Conference on Discrete Optimization and Machine Learning](https://aip.riken.jp/events/doml190729-31/)
- [Mixed integer programming (MIP) for machine learning](http://www.normastic.fr/wp-content/uploads/2014/12/Main_MIP_Rouen_2019.pdf)
- https://atienergyworkshop.wordpress.com/
- http://www.doc.ic.ac.uk/~dletsios/publications/gradient_boosted_trees.pdf
- https://minoa-itn.fau.de/
- http://www.me.titech.ac.jp/technicalreport/h26/2014-1.pdf
- [Training Binarized Neural Networks using MIP and CP](http://www.cs.toronto.edu/~lillanes/papers/ToroIcarteICCMB-cp2019-training-preprint.pdf)

#### Lagrangian Propagator

It has been showed that Neural Networks can be embedded in a `Constraint Programming` model 
by simply encoding each neuron as a global constraint, 
which is then propagated individually. 
[Unfortunately, this decomposition approach may lead to weak bounds.](https://link.springer.com/article/10.1007/s10601-015-9234-6)

- [A Lagrangian Propagator for Artificial Neural Networks in Constraint Programming](https://bitbucket.org/m_lombardi/constraints-15-ann-lag-resources/src/master/)
- https://link.springer.com/article/10.1007/s10601-015-9234-6
- https://www.researchgate.net/profile/Michele_Lombardi
- [Embedding Machine Learning Models in Optimization](https://emlopt.github.io/)
- https://www.unibo.it/sitoweb/michele.lombardi2/pubblicazioni
- https://people.eng.unimelb.edu.au/pstuckey/papers.html
- https://cis.unimelb.edu.au/agentlab/publications/
- [A New Propagator for Two-Layer Neural Networks in Empirical Model Learning](https://link.springer.com/chapter/10.1007/978-3-642-40627-0_35)

#### Layer-wise Relevance Propagation

Layer-wise Relevance Propagation (LRP) is a method 
that identifies important pixels by running a backward pass in the neural network. 
The backward pass is a conservative relevance redistribution procedure, 
where neurons that contribute the most to the higher-layer receive most relevance from it. 
[The LRP procedure is shown graphically in the figure below.](http://www.heatmapping.org/)

<img src="http://www.heatmapping.org/lrpgraph.png" width="50%"/>

- [Layer-wise Relevance Propagation for Deep Neural Network Architectures](http://iphome.hhi.de/samek/pdf/BinICISA16.pdf)
- https://github.com/gentaman/LRP
- [Tutorial: Implementing Layer-Wise Relevance Propagation](http://www.heatmapping.org/tutorial/)
- http://www.heatmapping.org/
- http://iphome.hhi.de/same

####  Target Propagation

Back-propagation has been the workhorse of recent successes of deep learning
but it relies on infinitesimal effects (partial derivatives) in order to perform credit
assignment. This could become a serious issue as one considers deeper and more
non-linear functions, e.g., consider the extreme case of non-linearity where the relation between parameters and cost is actually discrete. Inspired by the biological implausibility of back-propagation, a few approaches have been proposed in the past that could play a similar credit assignment role as backprop.
In this spirit, we explore a novel approach to credit assignment in deep networks that we call target propagation.
`The main idea is to compute targets rather than gradients, at each layer. Like gradients, they are propagated backwards.
In a way that is related but different from previously proposed proxies for back-propagation which rely on a backwards network with symmetric weights, target propagation relies on auto-encoders at each layer`.
Unlike back-propagation, it can be applied even when units exchange stochastic bits rather than real numbers.
[We show that a linear correction for the imperfectness of the auto-encoders is very effective to make target propagation actually work, along with adaptive learning rates.](http://www2.cs.uh.edu/~ceick/7362/T3-3.pdf)

- [TARGET PROPAGATION](http://www2.cs.uh.edu/~ceick/7362/T3-3.pdf)
- [Training Language Models Using Target-Propagation](https://arxiv.org/abs/1702.04770)
- http://www2.cs.uh.edu/~ceick/
- http://www2.cs.uh.edu/~ceick/7362/7362.html
- [Difference Target Propagation](https://arxiv.org/abs/1412.7525)


####  Gradient Target Propagation

We report a learning rule for neural networks that computes how much each neuron should contribute to minimize a giving cost function via the estimation of its target value.
By theoretical analysis, we show that this learning rule contains backpropagation, Hebbian learning, and additional terms. 
We also give a general technique for weights initialization.
[Our results are at least as good as those obtained with backpropagation.](https://arxiv.org/pdf/1810.09284.pdf)

- [Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations](https://arxiv.org/pdf/1609.07061.pdf)
- [Gradient target propagation](https://arxiv.org/abs/1810.09284)
- https://github.com/tiago939/target
- https://qdata.github.io/deep2Read//MoreTalksTeam/Un17/Muthu-OptmTarget.pdf

### Capsule Networks and More

Capsule Networks provide a way to detect parts of objects in an image and represent spatial relationships between those parts. This means that capsule networks are able to recognize the same object in a variety of different poses even if they have not seen that pose in training data.

- https://www.edureka.co/blog/capsule-networks/
- https://cezannec.github.io/Capsule_Networks/
- https://jhui.github.io/2017/11/03/Dynamic-Routing-Between-Capsules/
- [Awesome Capsule Networks](https://github.com/sekwiatkowski/awesome-capsule-networks)
- [Capsule Networks Explained](https://kndrck.co/posts/capsule_networks_explained/)
- [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)
- http://people.missouristate.edu/RandallSexton/sabp.pdf
- [Neuronal Dynamics: From single neurons to networks and models of cognition](https://neuronaldynamics.epfl.ch/book.html)

### Deep Stochastic Configuration Networks

In contrast to known randomized learning algorithms for single layer feed-forward neural networks (e.g., random vector functional-link networks), 
Stochastic Configuration Networks (SCNs) randomly assign the input weights and biases of the hidden nodes in the light of a supervisory mechanism, 
while the output weights are analytically evaluated in a constructive or selective manner.

Current experimental results indicate that SCNs outperform other randomized neural networks in terms of required human intervention, selection of the scope of random parameters, and fast learning and generalization. 
`Deep sctochastic configuration networks (DeepSCNs)' have been mathematically proved as universal approximators for continous nonlinear functions defined over compact sets. 
They can be constructed efficiently (much faster than other deep neural networks) 
and share many great features, such as learning representation and consistency property between learning and generalization.

[This website](http://www.deepscn.com/index.php) collect some introductory material on DeepSCNs, most notably a brief selection of publications and some software to get started.

- http://www.deepscn.com/references.php

### MIND-Net

From the  analytical perspective, the ad hoc nature of deep learning renders  its success  at the mercy of  trial-and-errors.  
To rectify this problem, we advocate a methodic  learning paradigm, MIND-Net,  which is computationally efficient in  training the networks and yet mathematically feasible to  analyze.  
MIND-Net hinges upon the use of an effective optimization metric, called Discriminant Information (DI).  
It will be used as a surrogate  of the popular metrics such as  0-1 loss or  prediction accuracy. Mathematically, DI is equivalent or closely related to Gauss’ LSE, Fisher’s FDR, and Shannon’s Mutual Information.  
[We shall explain why is that higher DI means higher linear separability, i.e. higher DI means that  the data are more discriminable.  In fact, it can be shown that, both theoretically and empirically,  a high DI score usually implies a high prediction accuracy.](http://www.it.fudan.edu.cn/En/Data/View/2519)

- https://datasciencephd.eu/
- https://ieeexplore.ieee.org/document/8682208
- https://www.researchgate.net/scientific-contributions/9628663_Sun-Yuan_Kung
- https://dblp.uni-trier.de/pers/hd/k/Kung:Sun=Yuan
- http://www.zhejianglab.com/mien/active_info/75.html
- https://ee.princeton.edu/people/sun-yuan-kung
- 
- https://ieeexplore.ieee.org/author/37273489000
- [Scalable Kernel Learning via the Discriminant Information](https://arxiv.org/abs/1909.10432)
- [METHODICAL DESIGN AND TRIMMING OF DEEP LEARNING NETWORKS: ENHANCING EXTERNAL BP LEARNING WITH INTERNAL OMNIPRESENT-SUPERVISION TRAINING PARADIGM](http://150.162.46.34:8080/icassp2019/ICASSP2019/pdfs/0008058.pdf)


### Dynamic Hierarchical Mimicking

- https://arxiv.org/abs/2003.10739
- https://github.com/d-li14/DHM

### DLphi


- http://www.pc-petersen.eu/
- http://voigtlaender.xyz/
- https://math.ethz.ch/sam/research/reports.html
- https://arxiv.org/abs/1901.05744
- https://faculty.washington.edu/kutz/

