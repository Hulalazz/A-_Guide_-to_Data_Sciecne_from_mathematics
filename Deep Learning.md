# Deep Learning

<img title="https://www.artsky.com/read/524" src="https://www.artsky.com/r/s/pic/info/2015/12/1451038754807.jpg" width="60%" />

https://jsideas.net/snapshot_ensemble/
https://www.deeplearningpatterns.com/doku.php?id=ensembles

* [2019 Fall Semester: Program on Deep Learning](https://www.samsi.info/programs-and-activities/semester-long-programs/2019-fall-semester-program-on-deep-learning/)
* https://www.mis.mpg.de/montufar/index.html
* [Deep Learning Theory Kickoff Meeting](https://www.mis.mpg.de/calendar/conferences/2019/kickoff.html)
* [PIMS CRG Summer School: Deep Learning for Computational Mathematics](https://www.pims.math.ca/scientific-event/190722-pcssdlcm)
* [Imperial College Mathematics department Deep Learning course](https://www.deeplearningmathematics.com/)
* [International Summer School on Deep Learning 2017 ](http://grammars.grlmc.com/DeepLearn2017/)
* [3rd International Summer School on Deep Learning](https://deeplearn2019.irdta.eu/)
* [Open Source Innovation in Artificial Intelligence, Machine Learning, and Deep Learning](https://aifiddle.io/)
* https://www.scienceofintelligence.de/
* https://lilianweng.github.io/lil-log/
* https://lfai.foundation/
* [Framework for Better Deep Learning](https://machinelearningmastery.com/framework-for-better-deep-learning/)
* [Distributed Deep Learning, Part 1: An Introduction to Distributed Training of Neural Networks](https://blog.skymind.ai/distributed-deep-learning-part-1-an-introduction-to-distributed-training-of-neural-networks/)
* [Large Scale Distributed Deep Networks 中译文](http://blog.sina.com.cn/s/blog_81f72ca70101kuk9.html)
* http://static.ijcai.org/2019-Program.html#paper-1606


Deep learning is the modern version of artificial neural networks full of tricks and techniques.
In mathematics, it is nonlinear non-convex and composite of many functions.
Its name -deep learning- is to distinguish from the classical machine learning "shallow" methods.
However, its complexity makes it yet engineering even art far from science.
There is no first principle in deep learning but trial and error.
In theory, we do not clearly understand how to design more robust and efficient network architecture;
in practice, we can apply it to diverse fields. It is considered as one approach to artificial intelligence.

Deep learning is a typical hierarchical machine learning model, consists of `hierarchical representation of input data,  non-linear evaluation and non-convex optimization`.
The application of deep learning are partial listed in

* [Awesome deep learning](https://github.com/ChristosChristofidis/awesome-deep-learning);
* [Opportunities and obstacles for deep learning in biology and medicine: 2019 update](https://greenelab.github.io/deep-review/);
* [Deep interests](https://github.com/Honlan/DeepInterests);
* [Awesome DeepBio](https://github.com/gokceneraslan/awesome-deepbio).

***
|[Father of Deep Learning](https://www.wikiwand.com/en/Alexey_Ivakhnenko)|
|:----------------------------------------------------------------------:|
|![Father of Deep Learning](https://tse2.mm.bing.net/th?id=OIP.RPMZM_oYzqfEvUISXL6aCQAAAA&pid=Api)|
|[A history of deep learning](https://www.import.io/post/history-of-deep-learning/)|
|[Three Giants' Survey in *(Nature 521 p 436)*](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)|
|[Critique of Paper by "Deep Learning Conspiracy" (Nature 521 p 436)](http://people.idsia.ch/~juergen/deep-learning-conspiracy.html) |
|<http://principlesofdeeplearning.com/>|
|[**Deep Learning in Neural Networks: An Overview**](http://people.idsia.ch/~juergen/deep-learning-overview.html)|
|[AI winter](https://www.wikiwand.com/en/AI_winter)|
|[Deep learning in wiki](https://www.wikiwand.com/en/Deep_learning), [Deep Learning in Scholarpedia](http://www.scholarpedia.org/article/Deep_Learning).|

***

| More |
|:---:|
|[A Brief History of Deep Learning (Part One)](https://www.bulletproof.net.au/a-brief-history-deep-learning-part-one/)|
|[On the Origin of Deep Learning](https://arxiv.org/abs/1702.07800)|
|![history of nn](https://www.import.io/wp-content/uploads/2017/06/Import.io_quote-image5-170525.jpg)|
|![nn_timeline](http://beamandrew.github.io//images/deep_learning_101/nn_timeline.jpg)|
|https://mitpress.mit.edu/books/deep-learning-revolution|
|https://blog.keras.io/the-limitations-of-deep-learning.html|
|[MIT Deep Learning Book](https://github.com/janishar/mit-deep-learning-book-pdf)|

***

Deep learning origins from neural networks and applies to many fields including computer vision, natural language processing.
The **architecture** and **optimization** are the core content of deep learning models. We will focus on the first one.
The optimization methods are almost the content of **stochastic/incremental gradient descent**.

![](http://www.deeplearningpatterns.com/lib/exe/fetch.php?media=deeplearning_overview_9_.jpg)



![](https://raw.githubusercontent.com/hunkim/deep_architecture_genealogy/master/Neural_Net_Arch_Genealogy.png)

## Artificial Neural Network

Artificial neural networks are most easily visualized in terms of a **directed graph**.
In the case of sigmoidal units, node ${s}$ represents sigmoidal unit  and directed edge $e=(u,v)$ indicates that one of sigmoidal unit ${v}$'s inputs is the output of sigmoidal unit ${u}$.
And there are connection between the node and itself in some kinds of neural networks.
The way or topology that nodes connected is an important part of deep neural network architecture. The other part is choice of **activation function**.
And any deep neural network can be expressed in the form of **computational graph**.
We will talk all the details after we give a glimpse to neural network zoo.

<img src=http://www.asimovinstitute.org/wp-content/uploads/2016/09/neuralnetworks.png width = 60% />
<img src=http://www.asimovinstitute.org/wp-content/uploads/2016/12/neuralnetworkcells.png width = 80% />


***

* [The Wikipedia page on ANN](https://www.wikiwand.com/en/Artificial_neural_network)
* [History of deep learning](http://beamandrew.github.io/deeplearning/2017/02/23/deep_learning_101_part1.html)
* https://brilliant.org/wiki/artificial-neural-network/
* https://www.asimovinstitute.org/neural-network-zoo/
* https://www.asimovinstitute.org/neural-network-zoo-prequel-cells-layers/
* [Computational graph](https://blog.csdn.net/zxl55/article/details/83537144)

### Perceptron

[Perceptron](https://www.wikiwand.com/en/Perceptron) can be seen as  a generalized linear model.
In mathematics, it is a map
$$
f:\mathbb{R}^n\rightarrow\mathbb{R}.
$$

It can be decomposed into $2$ steps:

1. Aggregate all the information: $z=\sum_{i=1}^{n}w_ix_i+b_0=(x_1,x_2,\cdots,x_n,1)\cdot (w_1,w_2,\cdots,w_n,b_0)^{T}.$
2. Transform the information to activate something: $y=\sigma(z)$, where $\sigma$ is nonlinear such as step function.

It can solve the linearly separate problem.   
![](http://i2.wp.com/abhay.harpale.net/blog/wp-content/uploads/perceptron-picture.png)


#### Learning  algorithm

We draw it from the [Wikipedia page](https://en.wikipedia.org/w/index.php?title=Perceptron&action=edit&section=3).
Learning is to find optimal parameters of the model. Before that, we must feed data into the model.
The training data of $n$ sample size is
$$
D=\{(\mathbf{x}_i, d_i)\}_{i=1}^{n}
$$

where

* $\mathbf{x}_i$ is the $n$-dimensional input vector;
* $d_i$ is the desired output value of the perceptron for that input $\mathbf{x}_i$.

***

1. Initialize the weights and the threshold. Weights may be initialized to 0 or to a small random value. In the example below, we use 0.
2. For each example $j$ in our training set $D$, perform the following steps over the input $\mathbf{x}_{j}$ and desired output $d_{j}$:  

   * Calculate the actual output：
      $$y_{j}(t)= f(\left<\mathbf{w}(t), \mathbf{x}_{j}\right>).$$
   * Update the weights:
       $w_{i}(t+1) = w_{i}(t) + r\cdot (d_{j}-y_{j}(t))x_{(j,i)},$
       for all features $[0\leq i\leq n]$, is the learning rate.
3. For [offline learning](https://www.wikiwand.com/en/Offline_learning), the second step may be repeated until the iteration error $\frac{1}{s}\sum_{j=1}^{s}|d_{j}-y_{j}(t)|$ is less than a user-specified error threshold $\gamma$, or a predetermined number of iterations have been completed, where _s_ is again the size of the sample set.

$\color{lime}{Note}$: the perceptron model is linear classifier, i.e. the training data set $D$ is linearly separable such that the learning algorithm can converge.

***
|----|----|
|:-------------:|:-----------------:|
|<img src=https://www.i-programmer.info/images/stories/Core/AI/DeepLearning/neuron.jpg width = 80% />|<img src=https://s.hswstatic.com/gif/brain-neuron-types-a.gif width = 80% />|
|[Perceptrons](https://www.wikiwand.com/en/Perceptrons_(book))|[人工神经网络真的像神经元一样工作吗？](https://www.jqr.com/article/000595)|
|<img src=https://weltbild.scene7.com/asset/vgw/perceptrons-195682049.jpg width = 80% /> | ![](https://tse4.mm.bing.net/th?id=OIP.96P534YMnRYWdiFQIv7WrgAAAA&pid=Api&w=200&h=450&rs=1&p=0)|

More in [Wikipedia page](https://www.wikiwand.com/en/Perceptron).

It is the first time to model cognition.


#### Activation functions

The nonlinear function $\sigma$ is conventionally called activation function.
There are some activation functions in history.

* Sign function
   $$
   f(x)=
      \begin{cases}
        1,&\text{if $x > 0$}\\
        -1,&\text{if $x < 0$}
     \end{cases}
   $$
* Step function
    $$
    f(x)=\begin{cases}1,&\text{if $x\geq0$}\\
                      0,&\text{otherwise}\end{cases}
    $$

* Sigmoid function
    $$
    \sigma(x)=\frac{1}{1+e^{-x}}.
    $$

* Radical base function
    $$
    \rho(x)=e^{-\beta(x-x_0)^2}.
    $$

* TanH function
   $$
   tanh(x)=2\sigma(2x)-1=\frac{2}{1+e^{-2x}}-1.
   $$

* ReLU function
   $$
   ReLU(x)={(x)}_{+}=\max\{0,x\}=\begin{cases}x,&\text{if $x\geq 0$}\\
                                 0,&\text{otherwise}\end{cases}.
   $$

***

* [神经网络激励函数的作用是什么？有没有形象的解释?](https://www.zhihu.com/question/22334626/answer/465380541)
* [Activation function in Wikipedia](https://www.wikiwand.com/en/Activation_function)
* 激活函数<https://blog.csdn.net/cyh_24/article/details/50593400>
* [可视化超参数作用机制：一、动画化激活函数](https://www.jqr.com/article/000161)
* https://towardsdatascience.com/hyper-parameters-in-action-a524bf5bf1c
* https://www.cnblogs.com/makefile/p/activation-function.html
* http://www.cnblogs.com/neopenx/p/4453161.html
* https://blog.paperspace.com/vanishing-gradients-activation-function/
* https://machinelearningmastery.com/exploding-gradients-in-neural-networks/

### Feed-forward Neural Network

#### Representation of Feedforward Neural Network

Given that the function of a single neuron is rather simple, it subdivides the input space into two regions by a hyperplane, the complexity must come from having more layers of neurons involved in a complex action (like recognizing your grandmother in all possible situations).
The "squashing" functions introduce critical non-linearities in the system, without their presence multiple layers would still create linear functions.
Organized layers are very visible in the human cerebral cortex, the part of our brain which plays a key role in memory, attention, perceptual awareness, thought, language, and consciousness.[^13]

The **feed-forward neural network** is also called multilayer perceptron. [The best way to create complex functions from simple functions is by **composition**.](http://math.mit.edu/~gs/learningfromdata/SIAM03.pdf)
In mathematics, it  can be considered as multi-layered non-linear composite function:

$$
X\to \sigma\circ (W_1X+b_1)=H_1\to \sigma\circ(W_2 H_1+b_2)=H_2 \to\cdots \sigma(WH+b)=y
$$

where the notation $\circ$, $M_1,b_1,M_2,b_2,\cdots, W,b$ mean pointwise operation, the parameters in the affine mapping, respectively. Thus the data flow in the form of the chain:
$$
\begin{align}
f = H_1 \circ {H_2}\circ \cdots \circ{\sigma} \qquad   &\text{Composite form}           \\
X\stackrel{\sigma}\to H_1 \stackrel{\sigma}\to H_2 \stackrel{\sigma} \to \cdots\stackrel{\sigma}\to\,y \qquad &\text{Hierarchy form}         \\
\mathbb{R}^p\to \mathbb{R}^{l_1}\to \mathbb{R}^{l_2}\to \cdots\to \mathbb{R}  \qquad&  \text{Dimension}
\end{align}
$$

where the circle notation $\circ$ means forward composite or as the input of afterward operation.
In hierarchy form, we omit the affine map.
It is can be written in the *recursive form*:
$$
\begin{align}
\mathbf{z}_{i} &= W_{i}H_{i-1}+b_i, \\
        H_{i}  &=\sigma\circ (\mathbf{z}_{i}),
\forall i\{1,2,\dots,D\} \tag 3
\end{align}
$$
where $H_{0}=X\in\mathbb{R}^p$, $b_i$ is a vector and $W_{i}$ is matrix. And the number of recursive times $D$ is called the depth of network.

1. In the first layer, we feed the input vector $X\in\mathbb{R}^{p}$ and connect it to each unit in the next layer $W_1X+b_1\in\mathbb{R}^{l_1}$ where $W_1\in\mathbb{R}^{n\times l_1}, b_1\in\mathbb{R}^{l_1}$. The output of the first layer is $H_1=\sigma\circ(M_1X+b)$, or in another word the output of $j$th unit in the first (hidden) layer is $h_j=\sigma{(W_1X+b_1)}_j$ where ${(W_1X+b_1)}_j$ is the $j$th element of $l_1$-dimensional vector $W_1X+b_1$.
2. In the second layer, its input is the output of first layer,$H_1$, and apply linear map to it: $W_2H_1+b_2\in\mathbb{R}^{l_2}$, where $W_2\in\mathbb{R}^{l_1\times l_2}, b_2\in\mathbb{R}^{l_2}$. The output of the second layer is $H_2=\sigma\circ(W_2H_1+b_2)$, or in another word the output of $j$th unit in the second (hidden) layer is $h_j=\sigma{(W_2H_1+b_2)}_j$ where ${(W_2H_1+b_2)}_j$ is the $j$th element of $l_2$-dimensional vector $W_2H_1+b_2$.
3. The map between the second layer and the third layer is similar to (1) and (2): the linear maps datum to different dimensional space and the nonlinear maps extract better representations.
4. In the last layer, suppose the input data is $H\in\mathbb{R}^{l}$. The output may be vector or scalar values and $W$ may be a matrix or vector as well as $y$.

***
The ordinary feedforward neural networks take the *sigmoid* function $\sigma(x)=\frac{1}{1+e^{-x}}$ as the nonlinear activation function
while the *RBF networks* take the [Radial basis function](https://www.wikiwand.com/en/Radial_basis_function) as the activation function such as $\sigma(x)=e^{c{\|x\|}_2^2}$.

In theory, the `universal approximation theorem` show the power of feed-forward neural network
if we take some proper activation functions such as sigmoid function.

* https://www.wikiwand.com/en/Universal_approximation_theorem
* http://mcneela.github.io/machine_learning/2017/03/21/Universal-Approximation-Theorem.html
* http://neuralnetworksanddeeplearning.com/chap4.html

<img src="https://www.cse.unsw.edu.au/~cs9417ml/MLP2/MainPage.gif" width="70%">

#### Evaluation and Optimization in Multilayer Perceptron

The problem is how to find the optimal parameters $W_1, b_1, W_2, b_2,\cdots, W, b$ ?
The multilayer perceptron is as one example of supervised learning, which means that we feed datum
$D=\{(\mathbf{x_i},d_i)\}_{i=1}^{n}$ to it and evaluate it.

The general form of the evaluation is given by:
$$
J(\theta)=\frac{1}{n}\sum_{i=1}^{n}\mathbb{L}[f(\mathbf{x}_i|\theta),\mathbf{d}_i]
$$
where $\mathbf{d}_i$ is the desired value of the input $\mathbf{x}_i$ and $\theta$ is the parameters of multilayer perceptron. The notation $f(\mathbf{x}_i|\theta)$ is the output given parameters $\theta$. The function $\mathbb{L}$ is **loss function** to measure the discrepancy between the predicted value $f(\mathbf{x}_i|\theta)$ and the desired value $\mathbf{d}_i$.

In general, the number of parameters $\theta$ is less than the sample size $n$. And the objective function $J(\theta)$ is not convex.  

We will solve it in the next section *Backpropagation, Optimization and Regularization*.

***

|The diagram of MLP|
|:----------------:|
|![](https://www.hindawi.com/journals/jcse/2012/389690.fig.0011.jpg)|
|[Visualizing level surfaces of a neural network with raymarching](https://arogozhnikov.github.io/3d_nn/)|

* https://devblogs.nvidia.com/deep-learning-nutshell-history-training/
* [Deep Learning 101 Part 2](http://beamandrew.github.io/deeplearning/2017/02/23/deep_learning_101_part2.html)
* https://www.wikiwand.com/en/Multilayer_perceptron
* https://www.wikiwand.com/en/Feedforward_neural_network
* https://www.wikiwand.com/en/Radial_basis_function_network
* https://www.cse.unsw.edu.au/~cs9417ml/MLP2/


##### Evaluation for different tasks

Evaluation is to judge the models with different parameters in some sense via objective function.
In maximum likelihood estimation, it is likelihood or log-likelihood;
in parameter estimation, it is bias or mean square error;
in regression, it depends on the case.

In machine learning, evaluation is aimed to  measure the discrepancy between the predicted values and trues value by **loss function**.
It is expected to be continuous smooth and differential but it is snot necessary.
The principle is the loss function make the optimization or learning tractable.
For classification, the loss function always is cross entropy;
for regression, the loss function can be any norm function such as the $\ell_2$ norm;
in probability models, the loss function always is joint probability or logarithm of joint probability.

In classification, the last layer is to predict the degree of belief of the labels via [softmax function](http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/),
i.e.

$$
softmax(z)=(\frac{\exp(z_1)}{\sum_{i=1}^{n}\exp(z_i)},\frac{\exp(z_2)}{\sum_{i=1}^{n} \exp(z_i)}, \cdots, \frac{\exp(z_n)}{\sum_{i=1}^{n}\exp(z_i)})
$$

where $n$ is the number of total classes. The labels are encoded as the one hot vector such as $\mathrm{d}=(1,0,0,\cdots,0)$. The [cross entropy](https://www.cnblogs.com/smartwhite/p/8601477.html) is defined as:

$$
\mathbf{H}(d,p)=-\sum_{i=1}^{n}d_i\log(p_i)=\sum_{i=1}^{n}d_i\log(\frac{1}{p_i}),
$$

where $d_i$ is the $i$th element of the one-hot vector $d$ and $p_i=\frac{\exp(z_i)}{\sum_{j=1}^{n}\exp(z_j)}$ for all $i=1,2\dots, n.$

Suppose $\mathrm{d}=(1,0,0,\cdots,0)$, the cross entropy is $\mathbf{H}(d,p)=-\log(p_1)=\log \sum_{i=1}^{n}\exp(z_i)-z_1$. The cost function is $\frac{1}{n}\sum_{i=1}^{n}\mathbf{H}(d^{i},p^{i})$ in the training data set $\{(\mathbf{x}_i,d^i)\}_{i=1}^{n}$ where $\mathbf{x}_i$ is the features of $i$th sample and $d^i$ is the desired true target label encoded in **one-hot** vector meanwhile $p^{i}$ is the predicted label of $\mathbf{x}_i$.
See the following links for more information on cross entropy and softmax.


|VISUALIZING THE LOSS LANDSCAPE OF NEURAL NETS||
|:-------------------------------------------:|---|
|![VGG](https://raw.githubusercontent.com/tomgoldstein/loss-landscape/master/doc/images/resnet56_noshort_small.jpg)|![ResNet](https://raw.githubusercontent.com/tomgoldstein/loss-landscape/master/doc/images/resnet56_small.jpg)|

Cross entropy is an example of [Bregman divergence](http://mark.reid.name/blog/meet-the-bregman-divergences.html).

![](https://pic3.zhimg.com/80/v2-cbbe99689224cdd829003483938af50e_hd.png)

* [Hinge loss function](https://www.wikiwand.com/en/Hinge_loss)

  It is a loss function for binary classification, of which the   output is $\{1, -1\}$.

  $$
      Hinge(x)=max\{0, 1-tx\}
  $$
  where $t=+1$ or $t=-1$.

* Negative logarithm likelihood function

  It is always the loss function in  probabilistic models.

[Loss Functions for Binary Class Probability Estimation and Classification: Structure and Applications
](http://stat.wharton.upenn.edu/~buja/PAPERS/paper-proper-scoring.pdf)

***

* <https://blog.csdn.net/u014380165/article/details/77284921>;
* <https://blog.csdn.net/u014380165/article/details/79632950>;
* <http://rohanvarma.me/Loss-Functions/>;
* <https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/>;
* <http://mark.reid.name/blog/meet-the-bregman-divergences.html>;
* <https://ieeexplore.ieee.org/document/5693441>;
* <https://www.zhihu.com/question/65288314>.

In regression, the loss function may simply be the squared $\ell_2$ norm, i.e. $\mathbb{L}(d,p)=(d-p)^{2}$ where ${d}$ is the desired target and $p$ is the predicted result. And the cost function is *mean squared error*:

$$
J(\theta)=\frac{1}{n}\sum_{i=1}^{n}[f(\mathbf{x}_i|\theta)-\mathrm{d}_i]^2.
$$

In **robust statistics**, there are more loss functions such as *Huber loss* and *Tukey loss*.
***

* [Huber loss function](https://www.wikiwand.com/en/Huber_loss)
  $$
      Huber_{\delta}(x)=
      \begin{cases}
          \frac{|x|^2}{2},&\text{if $|x|\leq\delta$}\\
          \delta(|x|-\frac{1}{2}\delta),&\text{otherwise}
      \end{cases}
  $$
* Tukey loss function

  $$
    Tukey_{\delta}(x)=
    \begin{cases}
         (1-[1-\frac{x^2}{\delta^2}]^3)\frac{\delta^2}{6}, &\text{if $|x|\leq\delta$}\\
         \frac{\delta^2}{6},                      &\text{otherwise}
   \end{cases}
  $$

   where its derivative is called [Tukey's Biweight](http://mathworld.wolfram.com/TukeysBiweight.html)：
   $$
   \phi(x)=
       \begin{cases}
         x(1-\frac{x^2}{\delta^2})^2 , &\text{if $|x|\leq\delta$}\\
         0,              &\text{otherwise}
      \end{cases}
   $$
   if we do not consider the points $x=\pm \delta$.
***

It is important to choose or design loss function or more generally objective function,
which can select variable as LASSO or confirm prior information as Bayesian estimation.
Except the *representation* or model, it is the objective function that affects the usefulness of learning algorithms.

The **smoothly clipped absolute deviation (SCAD) penalty** for `variable selection` in *high dimensional statistics* is defined by its derivative:
$$
p_{\lambda}^{\prime} (\theta) = \lambda \{ \mathbb{I}(\theta \leq \lambda)+\frac{{(a\lambda-\theta)}_{+}}{(a-1)\lambda}\mathbb{I}(\theta > \lambda) \}
$$

or in another word

$$
p_{\lambda}^{\prime} (\theta) =
 \begin{cases}
   \lambda  & \,\quad\text{if $\theta \leq \lambda$;} \\
   \frac{ {(a\lambda-\theta)}_{+} }{ (a-1)\lambda } &\,\quad\text{otherwise}
 \end{cases}
$$

for some $a > 2$ and $\theta > 0$.

The [minimax concave penalties(MCP)](https://projecteuclid.org/euclid.aos/1266586618) provides the convexity of the penalized loss in sparse regions to
the greatest extent given certain thresholds for variable selection and unbiasedness.
It is defined as
$$\rho(t;\lambda) = \lambda \int_{0}^{t}{(1-\frac{x}{\gamma\lambda})}_{+} \mathrm{d}x.$$

The function defined by its derivative is convenient for us to minimize the cost function via gradient-based methods.

***
The two page paper [Eliminating All Bad Local Minima from Loss Landscapes
Without Even Adding an Extra Unit](https://arxiv.org/pdf/1901.03909.pdf) modified the loss function $L(\theta)$ to eliminate all the bad local minima:

$$
\overline{L}(\theta, a,b) = L(\theta)(1+(a\exp(b)-1)^2)+\lambda a^2
$$

where $a,b\in\mathbb{R}$ are are auxiliary parameters, and and $\lambda\in\mathbb{R}^{+}$ is a regularization hyperparameter.

Its gradient with respect to the auxiliary parameters ${a,b}$ is given by
$$
\frac{\partial \overline{L}(\theta, a,b)}{\partial a} = 2 L(\theta)(a\exp(b)-1)\exp(b) + 2\lambda a \\
\frac{\partial \overline{L}(\theta, a,b)}{\partial b} = 2a L(\theta)(a\exp(b)-1)\exp(b)
$$

so that setting them to be 0s, we could get $a=0, L(\theta) = 0$ if $|b|<\infty$.

For more on **loss function** see:

* <https://blog.algorithmia.com/introduction-to-loss-functions/>;
* <https://arxiv.org/abs/1701.03077>;
* https://www.wikiwand.com/en/Robust_statistics
* https://www.wikiwand.com/en/Huber_loss
* https://www.e-sciencecentral.org/articles/SC000012498
* <https://arxiv.org/abs/1811.03962>
* https://arxiv.org/pdf/1901.03909.pdf
* <https://www.cs.umd.edu/~tomg/projects/landscapes/>.
* [optimization beyond landscape at offconvex.org](http://www.offconvex.org/2018/11/07/optimization-beyond-landscape/)

### Backpropagation, Training and Regularization

#### Backpropagation

Automatic differentiation is the generic name for techniques that use the computational representation of a function to produce **analytic** values for the derivatives.
Automatic differentiation techniques are founded on the observation that any function, no matter how complicated, is evaluated by performing a sequence of simple elementary operations involving just one or two arguments at a time.
Backpropagation is one special case of automatic differentiation, i.e. *reverse-mode automatic differentiation*.

The backpropagation procedure to compute the gradient of an objective function with respect to the weights of a multilayer stack of modules is nothing more than a practical application of the **chain rule in terms of partial derivatives**.
Suppose $g:\mathbb{R}^{p}\to\mathbb{R}^n$ and $f:\mathbb{R}^{n}\to\mathbb{R}^{m}$, and let $b=g(a)$, $c=f(b)$, `Chain rule` says that
$$\frac{\partial c_i}{\partial a_j}=\sum_{k}\frac{\partial c_i}{\partial b_k}\frac{\partial b_k}{\partial a_j}.$$

The key insight is that the derivative (or gradient) of the objective with respect to the input of a module can be computed by working backwards from the gradient with respect to the output of that module (or the input of the subsequent module).
The backpropagation equation can be applied repeatedly to
propagate gradients through all modules, starting from the output at the top (where the network produces its prediction) all the way to the bottom (where the external input is fed).
Once these gradients have been computed, it is straightforward to compute the gradients with respect to the weights of each module.

***
Suppose that $f(x)={\sigma}\circ(WH + b)$,where
* $H  =\sigma\circ(W_4H_3 + b_4)$,
* $H_3=\sigma\circ(W_3H_2 + b_3)$,
* $H_2=\sigma\circ(W_2H_1 + b_2)$,
* $H_1=\sigma\circ(W_1x + b_1)$,

we want to compute the gradient $L(x_0,d_0)=\|f(x_0)-d_0\|^{2}_2$ with respect to all weights $W_1,W_2,W_3,W$:

$$
\frac{\partial L(x_0,d_0)}{\partial W_n^i}=\frac{\partial L(x_0,d_0)}{\partial f(x_0)}\frac{\partial f(x_0)}{\partial W_n^i}\forall i\in\{1,2,\dots,l_n\}, \forall\,n\in\{1,2,3,4\}
$$

and it is fundamental to compute the gradient with respect to the last layer as below.

- [ ] the gradient of loss function with respect to the prediction function:
$$\frac{\partial L(x_0,d_0)}{\partial f(x_0)}=2[f(x_0)-d_0],$$

- [ ] the gradient of each unit in prediction function with respect to the weight in the last layer:
$$
\frac{\partial f^{j}(x_0)}{\partial W^j}=
\frac{\partial \sigma(W^jH+b^j)}{\partial W^j}=
{\sigma}^{\prime}(W^jH+b^j) H \,\,\forall j\in\{1,2,\dots,l\},
$$

- [ ] the gradient of prediction function with respect to the last hidden state:
$$
\frac{\partial f^{j}(x_0)}{\partial H}  =
\frac{\partial \sigma(W^jH + b^j)}{\partial H}  =
{\sigma}^{\prime}(W^jH + b^j) W^j \,\,\forall j\in\{1,2,\dots,l\},
$$
where $f^{j}(x_0)$, $W^{j}$, $b^j$ and $\sigma^{\prime}(z)$ is the j-th element of $f(x_0)$, the j-th row of matrix $W$, the  j-th element of vector ${b}$ and $\frac{\mathrm{d}\sigma(z)}{\mathrm{d} z}$, respectively.

|The Architecture of Feedforward Neural Networks|
|:---------------------------------------------:|
|![](https://www.hindawi.com/journals/jcse/2012/389690.fig.0011.jpg)|
|Each connection ,the black line, is attached with a weight parameter.|
***
Recall the chain rule with more variables:
$$
\frac{\partial f(m(x_0),n(x_0))}{\partial x_0}=\frac{\partial f(m(x_0),n(x_0))}{\partial m(x_0)}\frac{\partial m(x_0)}{\partial x_0} + \frac{\partial f(m(x_0),n(x_0))}{\partial n(x_0)}\frac{\partial n(x_0)}{\partial x_0}.
$$

Similarly , we can compute following gradients:
$$\frac{\partial H^j}{\partial W_4^j}  =\frac{\partial \sigma(W_4^j H_3+b_4^j)}{\partial W_4^j}  =[\sigma^{\prime}(W_4^j H_3+b_4^j)H_3]^T    \qquad\forall j\in\{1,2,\dots,l\};$$

$$\frac{\partial H^j}{\partial H_3}    =\frac{\partial \sigma(W_4^j H_3+b_4^j)}{\partial H_3}  =[\sigma^{\prime}(W_4^j H_3+b_4^j)W_4^j]^T  \qquad\forall j\in\{1,2,\dots,l_4\};$$

$$\frac{\partial H_3^j}{\partial W_3^j}=\frac{\partial \sigma(W_3^j H_2+b_3^j)}{\partial W_3^j}  =[\sigma^{\prime}(W_3^j H_2+b_3^j)H_2 ]^T   \qquad\forall j\in\{1,2,\dots,l_3\};$$

$$\frac{\partial H_3^j}{\partial H_2}  =\frac{\partial \sigma(W_3^j H_2+b_3^j)}{\partial H_2}  =[\sigma^{\prime}(W_3^j H_2+b_3^j)W_3^j]^T  \qquad\forall j\in\{1,2,\dots,l_3\};$$

$$\frac{\partial H_2^j}{\partial W_2^j}=\frac{\partial \sigma(W_2^j H_1+b_2^j)}{\partial W_2^j}  =[\sigma^{\prime}(W_2^j H_1+b_2^j)H_1 ]^T   \qquad\forall j\in\{1,2,\dots,l_2\};$$

$$\frac{\partial H_2^j}{\partial H_1}  =\frac{\partial \sigma(W_2^j H_1+b_2^j)}{\partial H_1}    =[\sigma^{\prime}(W_2^j H_1+b_2^j)W_2^j]^T  \qquad\forall j\in\{1,2,\dots,l_2\};$$

$$\frac{\partial H_1^j}{\partial W_1^j}=\frac{\partial \sigma(W_1^j x_0+b_1^j)}{\partial W_1^j}  =[\sigma^{\prime}(W_1^j x_0+b_1^j)x_0]^T    \qquad\forall j\in\{1,2,\dots,l_1\}.$$


***
The multilayer perceptron $f(x)$ can be written in a chain form:
$$
X\stackrel{\sigma}{\to} H_1 \stackrel{\sigma}{\to} H_2\stackrel{\sigma} \to H_3 \stackrel{\sigma} \to H_4 \stackrel{\sigma} \to H\stackrel{\sigma}\to\, f(x)  \\
X\rightarrow W_1 X \rightarrow W_2H_1 \rightarrow W_3H_2 \rightarrow W_4H_3 \rightarrow WH \rightarrow y       \\
\mathbb{R}^{p}\to \mathbb{R}^{l_1}\to \mathbb{R}^{l_2}\to \mathbb{R}^{l_3}\to \mathbb{R}^{l}\to \mathbb{R}^{o}
$$

while the backpropagation to compute the gradient is in the reverse order:

$$
\frac{\partial y}{\partial W}\to \frac{\partial y}{\partial H}\to \frac{\partial H}{\partial W_4}\to \frac{\partial H}{\partial H_3}\to \frac{\partial H_3}{\partial W_3}\to \frac{\partial H_3}{\partial H_2}\to \frac{\partial H_2}{\partial W_2}\to \frac{\partial H_2}{\partial W_1}\to \frac{\partial H_1}{\partial W_1}.
$$

In general, the gradient of any weight can be computed by *backpropagation* algorithm.

The first step is to compute the gradient of squared loss function with respect to the output $f(x_0)=y\in\mathbb{R}^{o}$, i.e.
$\frac{\partial L(x_0, d_0)}{\partial f(x_0)}=2(f(x_0)-d_0)=2(\sigma\circ(WH+b)-d_0)$
, of which the $i$th element is $2(y^{i}-d_0^i)=2(\sigma(W^{i}H+b^{i})-d_0^{i})\,\forall i\{1,2,\dots,o\}$.
Thus
$$\frac{\partial L(x_0, d_0)}{\partial W^{i}}=\frac{\partial L(x_0, d_0)}{\partial y^{i}}\frac{\partial y^{i}}{\partial W^{i}}=2(y^{i} -d_0^i)\sigma^{\prime}(W^iH+b^i)[H]^T,\\
\frac{\partial L(x_0, d_0)}{\partial b^{i}}=\frac{\partial L(x_0, d_0)}{\partial y^{i}}\frac{\partial y^{i}}{\partial b^{i}}=2(y^{i}-d_0^i)\sigma^{\prime}(W^iH + b^i) b^i.$$
Thus we can compute all the gradients of $W$ columns. Note that $H$ has been computed through forwards propagation in that layer.

And $H=\sigma\circ(W_4H_3+b_3)$, of which the $i$ th element is
$$H^{i}=\sigma(W_4 H_3 +b_4)^{i}=\sigma(W_4^{i} H_3+b_4^{i}).$$

And we can compute the gradient of columns of $W_4$:

$$\frac{\partial L(x_0,y_0)}{\partial W_4^i}$$
where $y_0=f(x_0)=\sigma\circ[W\underbrace{\sigma\circ(W_4H_3+b_4)}_{H}+b]$.

and by the `chain rule`  we obtain
$$\frac{\partial L(x_0,y_0)}{\partial W_4^i}
=\sum_{j=1}^{o}\underbrace{\frac{\partial L(x_0,y_0)}{\partial y^j}}_{\text{computed in last layer} }
\sum_{n}(\overbrace{\frac{\partial y^j}{\partial H^n}}^{\text{the hidden layer}}
\frac{\partial H^n}{\partial W_4^i})$$

$$
= \color{aqua}{
\sum_{j=1}^{o} \frac{\partial L}{\partial y^j}}\color{blue}{
\sum_{n}(\underbrace{\frac{\partial\, y^j}{\partial (W^jH+b^j)}
\frac{\partial (W^jH + b^j)}{\partial H^{n}}}_{\color{purple}{\frac{\partial y^j}{ \partial H^i}} }
\frac{\partial (H^{n})}{\partial W_4^i}) } \\
= \sum_{j=1}^{o} \frac{\partial L}{\partial y^j}
\sum_{n}(\frac{\partial\, y^j}{\partial (W^jH+b^j)}
\frac{\partial (W^jH + b^j)}{\partial H^{n}}
\color{green}{\underbrace{\sigma^{\prime}(W^i_4 H_3+b^i_4)[H_3]^T}_{\text{computed after forward computation}} })
,
$$

where $W^{j,i}$ is the $i$th element of $j$th row in matrix $W$.

$$
\frac{\partial L(x_0,y_0)}{\partial W_3^i}=
\sum_{j=1}^{o}
\frac{\partial L}{\partial y^j}
\{\sum_m[\sum_{n}(\frac{\partial y^j}{\partial H^n}
\overbrace{\frac{\partial H^n}{\partial H_3^m} }^{\triangle})]
\underbrace{\frac{\partial H_3^m}{\partial W_3^i} }_{\triangle} \}
$$
where all the partial derivatives or gradients have been computed or accessible. It is nothing except to add or multiply these values in the order when we compute the weights of hidden layer.

$$
\frac{\partial L(x_0,y_0)}{\partial W_2^i}=
\sum_{j=1}^{o}\frac{\partial L}{\partial y^j}
\fbox{$\sum_{l}\{\underbrace{\sum_m[\sum_{n}(\frac{\partial y^j}{\partial H^n}
\frac{\partial H^n}{\partial H_3^m} )] }_{\text{computed in last layer}}
\frac{\partial H_3^m}{\partial H_2^l} \}  \frac{\partial H_2^l}{\partial W_2^i}$}
$$

And the gradient of the first layer is computed by
$$
\frac{\partial L(x_0,y_0)}{\partial W_1^i}
=\sum_{j=1}^{o}\frac{\partial L}{\partial y^j}
\big(  \sum_{p}\left(\sum_{l}\{\sum_m[\sum_{n}(\frac{\partial y^j}{\partial H^n}
\frac{\partial H^n}{\partial H_3^m} )]
\frac{\partial H_3^m}{\partial H_2^l} \} \frac{\partial H_2^l}{\partial H_1^p} \right)\frac{\partial H_1^p}{\partial W_1^i} \big).
$$

See more information on backpropagation in the following list

* [Back-propagation, an introduction at offconvex.org](http://www.offconvex.org/2016/12/20/backprop/);
* [Backpropagation on Wikipedia](https://www.wikiwand.com/en/Backpropagation);
* [Automatic differentiation on Wikipedia](https://www.wikiwand.com/en/Automatic_differentiation);
* [backpropagation on brilliant](https://brilliant.org/wiki/backpropagation/);
* [Who invented backpropagation ?](http://people.idsia.ch/~juergen/who-invented-backpropagation.html);
* An introduction to automatic differentiation at <https://alexey.radul.name/ideas/2013/introduction-to-automatic-differentiation/>;
* Reverse-mode automatic differentiation: a tutorial at <https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation>.
* [Autodiff Workshop: The future of gradient-based machine learning software and techniques, NIPS 2017](https://autodiff-workshop.github.io/)
* http://www.autodiff.org/
* [如何直观地解释 backpropagation 算法？ - 景略集智的回答 - 知乎](https://www.zhihu.com/question/27239198/answer/537357910)
* The chapter 2 *How the backpropagation algorithm works* at the online book <http://neuralnetworksanddeeplearning.com/chap2.html>
* For more information on automatic differentiation see the book *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation, Second Edition* by Andreas Griewank and Andrea Walther_ at <https://epubs.siam.org/doi/book/10.1137/1.9780898717761>.
* [Building auto differentiation library](https://maciejkula.github.io/2018/07/18/building-an-autodifferentiation-library/)
* [The World’s Most Fundamental Matrix Equation](https://sinews.siam.org/Details-Page/the-worlds-most-fundamental-matrix-equation)

<img src=http://ai.stanford.edu/~tengyuma/forblog/weight5.jpg width=50% />



##### Beyond Back-propagation

Training deep learning models does not require gradients such as `ADMM, simulated annealing`.

- [Main Principles of the General Theory of Neural Network with Internal Feedback](http://worldcomp-proceedings.com/proc/p2015/ICA6229.pdf)
- https://arxiv.org/pdf/1907.05200.pdf
- [Layer-wise Relevance Propagation for Deep Neural Network Architectures](http://iphome.hhi.de/samek/pdf/BinICISA16.pdf)
- https://github.com/gentaman/LRP
- [Tutorial: Implementing Layer-Wise Relevance Propagation](http://www.heatmapping.org/tutorial/)
- [ADMM for Efficient Deep Learning with Global Convergence](https://arxiv.org/abs/1905.13611)
- https://neuronaldynamics.epfl.ch/book.html
- [Deep Learning as a Mixed Convex-Combinatorial Optimization Problem](https://arxiv.org/abs/1710.11573)
- [Bidirectional Backpropagation](http://sipi.usc.edu/~kosko/B-BP-SMC-Revised-13January2018.pdf)
- [Difference Target Propagation](https://arxiv.org/abs/1412.7525)
- [Gradient target propagation](https://arxiv.org/abs/1810.09284)
- [Beyond Backprop: Online Alternating Minimization with Auxiliary Variables](https://www.ibm.com/blogs/research/2019/06/beyond-backprop/)
- [Beyond Backpropagation: Uncertainty Propagation](http://videolectures.net/iclr2016_lawrence_beyond_backpropagation/)
- [Beyond Feedforward Models Trained by Backpropagation: a Practical Training Tool for a More Efficient Universal Approximator](https://www.memphis.edu/clion/pdf-papers/0710.4182.pdf)
- [A Biologically Plausible Learning Algorithm for Neural Networks](https://www.ibm.com/blogs/research/2019/04/biological-algorithm/)
- [Awesome Capsule Networks](https://github.com/sekwiatkowski/awesome-capsule-networks)
- [Capsule Networks Explained](https://kndrck.co/posts/capsule_networks_explained/)
- [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)
- http://people.missouristate.edu/RandallSexton/sabp.pdf
- [BEYOND BACKPROPAGATION: USING SIMULATED ANNEALING FOR TRAINING NEURAL NETWORKS](http://people.missouristate.edu/RandallSexton/sabp.pdf)
#### Fundamental Problem of Deep Learning and Activation Functions

`Gradients vanishing` is the fundamental problem of deep neural networks according to [Juergen](http://people.idsia.ch/~juergen/).
The the 1991 diploma thesis of [Sepp Hochreiter](http://www.bioinf.jku.at/people/hochreiter/) formally showed that [deep neural networks are hard to train, because they suffer from the now famous problem of vanishing or exploding gradients: in typical deep or recurrent networks, back-propagated error signals either shrink rapidly, or grow out of bounds. In fact, they decay exponentially in the number of layers, or they explode.](http://people.idsia.ch/~juergen/fundamentaldeeplearningproblem.html)

<img src="http://people.idsia.ch/~juergen/seppdeep754x466.jpg" width="70%" />

This fundamental problem makes it impossible to decrease the total loss function via gradient-based optimization problems.
One direction solution is to replace the sigmoid functions in order to take the advantages of gradient-based methods.
Another approach is to minimize the cost function via gradient-free optimization methods such as simulated annealing.

The sigmoid function as the first function to replace the step function actually is a cumulative density function (CDF), which satisfy the following conditions:

-  The function, $\sigma(x) = \frac{1}{1+exp(-x)}$,  is left  continuous;
-  $0 \leq \sigma(x) \leq 1, \forall x\in\mathbb{R}$;
-  $\lim_{x \to - \infty}\sigma(x)=0, \lim_{x\to\infty}\sigma(x)=1$.

It is called *logistic function* in statistics. This function is easy to explain as a continuous alternative to the step function. And it is much easier to evaluate than the common normal distribution.

And the derivative function of this function is simple to compute if we know the function itself:

$$\sigma^{\prime}(x) = \sigma(x)(1 -  \sigma(x)).$$

It is clear that $\arg\max_{x} \sigma^{\prime}(x) = \frac{1}{4}< 1$, which can make the `gradients vanish` during the back-propagation.

<img src=https://blog.paperspace.com/content/images/2018/06/sigmoid700.png width=50% />

By `Back-Propagation`, we obtain that
$$
\frac{\partial (\sigma(\omega^T x + b))}{\partial \omega}
=\frac{\partial (\sigma(\omega^T x + b))}{\partial (\omega^Tx + b)}\cdot\frac{\partial (\omega^T x + b)}{\partial \omega}
\\= [\sigma(\omega^Tx + b)\cdot(1-\sigma(\omega^T x + b))] x.
$$

The problems of `Vanishing gradients` can be worsened by saturated neurons. Suppose, that pre-activation ${\omega}^T x + b$ that is fed to a neuron with a Sigmoid activation is either very high or very low. The gradient of sigmoid at very high or low values is almost 0. Any gradient update would hardly produce a change in the weights $\omega$ and the bias $b$, and it would take a lot of steps for the neuron to get modify weights so that the pre-activation falls in an area where the gradient has a substantial value.

And if all the data points fed into layers are close to ${0}$, it may help to solve the vanishing gradients problems. And the sigmoid function $\sigma(x)$ is always non-negative and is not symmetric about ${0}$ so we may modify this function into

$$
tanh(x) = 2\sigma(2x) - 1
= \frac{2}{1 + \exp(-2x)}-1
\\= \frac{1- \exp(-2x)}{1 + \exp(-2x)}
\\=\frac{\exp(x) - \exp(-x)}{\exp(x)+\exp(-x)}\in(-1, 1).
$$
Sometimes it is alos called `Bipolar Sigmoid`.

And its gradient is
$$
tanh^{\prime}(x)= 4\sigma^{\prime}(x) = 4\sigma(x)(1-\sigma(x))\in (0,1].
$$


<img src=https://pic3.zhimg.com/80/v2-3528e66d0e12b35f778fe0ed21d2ced2_hd.jpg width=50% />

Another profit of `Exponential function` of feature vector $\exp(x)$ is that complex interaction of the features although it is expensive at computation.

And  an alternative function is so-called `hard tanh`: $f(x)=\max(-1, \min(1, x))$.

<img src=https://i.stack.imgur.com/CJnMI.png width=40% />

***
The first attempt at curbing the problem of vanishing gradients in a general deep network setting (LSTMs were introduced to combat this as well, but they were restricted to recurrent models) was the introduction of the `ReLU` activation function:
$$
ReLU(x)= \max\{x, 0\}={(x)}_{+}
\\=\begin{cases}
x, & \text{if}\quad x\geq 0; \\
0, & \text{otherwise}.
\end{cases}
$$

<img src=https://blog.paperspace.com/content/images/2018/06/relu.png width=80% />

If we do not consider ${0}$, the gradient of ReLU is computed as
$$
ReLU^{\prime}(x)=
\begin{cases}
1, & \text{if}\quad x > 0; \\
0, & \text{if}\quad x < 0.
\end{cases}
$$

And to be technological, ReLU is not continuous at 0.

The product of gradients of ReLU function doesn't end up converging to 0 as the value is either 0 or 1. If the value is 1, the gradient is back propagated as it is. If it is 0, then no gradient is backpropagated from that point backwards.

We had a two-sided saturation in the sigmoid functions. That is the activation function would saturate in both the positive and the negative direction. In contrast, ReLUs provide one-sided saturations.

ReLUs come with their own set of shortcomings. While sparsity is a computational advantage, too much of it can actually hamper learning. Normally, the pre-activation also contains a bias term. If this bias term becomes too negative such that $\omega^T x + b < 0$, then the gradient of the ReLU activation during backward pass is 0. **Therefore, the weights and the bias causing the negative pre-activations cannot be updated.**

If the weights and bias learned is such that the pre-activation is negative for the entire domain of inputs, the neuron never learns, causing a sigmoid-like saturation. This is known as the `dying ReLU problem`.

In order to combat the problem of dying ReLUs, the `leaky ReLU` was proposed. A Leaky ReLU is same as normal ReLU, except that instead of being 0 for $x \leq 0$, it has a small negative slope for that region.

$$
f(x) =
 \begin{cases}
   x, & \text{if}\quad x > 0;\\
   \alpha x, & \text{otherwise}.
 \end{cases}
$$

`Randomized Leaky ReLU`

$$
f(x) =
   \begin{cases}
      x, & \text{if} \quad x > 0;\\
      \alpha x, & \text{otherwise}.
   \end{cases} \\
\alpha\sim U[0,1]
$$

`Parametric ReLU`

$$
f(x) =
\begin{cases}
x, & \text{if} \quad x > 0;\\
\alpha x, & \text{otherwise}.
\end{cases}
$$
where this $\alpha$ can be learned since you can backpropagate into it.

![](https://i.stack.imgur.com/1BX7l.png)

`ELU(Exponential Linear Units)` is an alternative of ReLU:

$$
ELU(x)=
\begin{cases}
   x, &\text{if}\quad x > 0;\\
   \alpha(\exp(x)-1), &\text{otherwise}.
\end{cases}
$$

<img src=https://pic2.zhimg.com/80/v2-604be114fa0478f3a1059923fd1022d1_hd.png width=80% />

***
`SWISH`:

$$
\sigma(x) = x\cdot sigmoid(x) = \frac{x}{1+e^{-x}} \in (0,+\infty)
$$

The derivative function of SWISH is given by
$$
\sigma^{\prime}(x)= sigmoid(x) + x\cdot {sigmoid^{\prime}(x)}
\\=\frac{1}{1+e^{-x}}[1+x(1-\frac{1}{1+e^{-x}})]
\\=\frac{1+e^{-x} + xe^{-x}}{(1+e^{-x})^2}.
$$
And $\lim_{x\to +\infty}\frac{SWISH}{x}=\lim_{x\to +\infty}\frac{1}{1+e^{-x}}=1$.

![](https://www.learnopencv.com/wp-content/uploads/2017/10/swish.png)

`Soft Plus`:
It is also known as  Smooth Rectified Linear Unit, Smooth Max or Smooth Rectifier.
$$
f(x)=\log(1+e^{x})\in(0,+\infty).
$$
And $\lim_{x\to\infty}\frac{f(x)}{x}=1$.

Its derivative function is the sigmoid function:
$$
f^{\prime}(x)=\frac{e^x}{1 + e^x} = \frac{1}{1 + e^{-x}}\in (0,1).
$$


`SoftSign`:

$$
f(x) =  \frac{x}{1+|x|} \in (-1, 1) \\
f^{\prime}(x)=\frac{1}{(1+|x|)^2}\in(0, 1)
$$

The sign function  is defined as

$$
sgn(x)=
   \begin{cases}
     1, &\text{if $x > 0$}\\
     0, &\text{if $x=0$}  \\
     - 1, &\text{if $x < 0$}
   \end{cases}
\\=\frac{x}{|x|}\quad\text{if $x\not= 0$}.
$$

<img src=https://www.gabormelli.com/RKB/images/3/35/softsign.png width=50% />

The soft sign function is not continuous at ${0}$ as well as sign function while it is smoother than sign function and without any leap points.

Softsign is another alternative to Tanh activation. Like Tanh, it's anti-symmetrical, zero centered, differentiable and returns a value between -1 and 1. Its flatter shape and more slowly declining derivative suggest that it may learn more efficiently. On the other hand, calculation of the derivative is more computationally cumbersome than Tanh.

`Bent identity`:
$$
f(x)=\frac{\sqrt{x^2+1}-1}{2}+x \in(-\infty, +\infty) \\
f^{\prime}(x) = \frac{x}{2\sqrt{x^2+1}} + 1\in(0.5, 1.5)
$$

A sort of compromise between Identity and ReLU activation, Bent Identity allows non-linear behaviours, while its non-zero derivative promotes efficient learning and overcomes the issues of dead neurons associated with ReLU. As its derivative can return values either side of 1, it can be susceptible to both exploding and vanishing gradients.

|Sigmoid Function|
|:---:|
|<img src=https://cloud.githubusercontent.com/assets/14886380/22743102/1ddd6a88-ee54-11e6-98ea-6b67011e091b.png width=80% />|

|Sigmoidal function| Non-saturation function |
|---|---|
| Sigmoid | ReLU|
| Tanh | ELU |
| Hard Tanh | Leaky ReLU|
| Sign | SWISH|
| SoftSign | Soft Plus|

The functions in the right column are  approximate to the identity function in $\mathbb{R}^{+}$, i.e., $\lim_{x\to +\infty}\frac{f(x)}{x} = 1$. And activation function ${f}$ is potential to result in gradient vanishing if its derivative is always less than 1: $f^{\prime}(x) < 1 \quad\forall x$. The non-saturation function can result in gradient exposition.
[Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent](https://arxiv.org/abs/1902.06720)

- [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
- [Swish: a Self-Gated Activation Function](https://arxiv.org/abs/1710.05941v1)
- [Deep Learning using Rectified Linear Units (ReLU)](https://arxiv.org/abs/1803.08375)
- [Softsign as a Neural Networks Activation Function](https://sefiks.com/2017/11/10/softsign-as-a-neural-networks-activation-function/)
- [Softsign Activation Function](https://www.gabormelli.com/RKB/Softsign_Activation_Function)
- [A Review of Activation Functions in SharpNEAT](http://sharpneat.sourceforge.net/research/activation-fn-review/activation-fn-review.html)
- [Activation Functions](https://rpubs.com/shailesh/activation-functions)
- [第一十三章 优化算法](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch13_%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/%E7%AC%AC%E5%8D%81%E4%B8%89%E7%AB%A0_%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95.md)
- [ReLU and Softmax Activation Functions](https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions).

****
* http://people.idsia.ch/~juergen/fundamentaldeeplearningproblem.html
* http://neuralnetworksanddeeplearning.com/chap5.html
* https://blog.csdn.net/cppjava_/article/details/68941436
* https://adventuresinmachinelearning.com/vanishing-gradient-problem-tensorflow/
* https://www.jefkine.com/general/2018/05/21/2018-05-21-vanishing-and-exploding-gradient-problems/
* https://golden.com/wiki/Vanishing_gradient_problem
* https://blog.paperspace.com/vanishing-gradients-activation-function/
* https://machinelearningmastery.com/exploding-gradients-in-neural-networks/
* https://www.zhihu.com/question/49812013
* https://nndl.github.io/chap-%E5%89%8D%E9%A6%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.pdf
* https://www.learnopencv.com/understanding-activation-functions-in-deep-learning/
* https://sefiks.com/2018/12/01/using-custom-activation-functions-in-keras/
* https://isaacchanghau.github.io/post/activation_functions/
* https://dashee87.github.io/deep%20learning/visualising-activation-functions-in-neural-networks
* <http://laid.delanover.com/activation-functions-in-deep-learning-sigmoid-relu-lrelu-prelu-rrelu-elu-softmax/>;
* https://explained.ai/matrix-calculus/index.html

#### Training Methods

The training is to find the optimal parameters of the model based on the **training data set**. The training methods are usually based on the gradient of cost function as well as back-propagation algorithm  in deep learning.
See **Stochastic Gradient Descent** in **Numerical Optimization** for details.
In this section, we will talk other optimization tricks such as **Normalization**.

| Concepts | Interpretation|
|:--------:|:-------------:|
|*Overfitting* and *Underfitting*| See [Overfitting](https://www.wikiwand.com/en/Overfitting) or [Overfitting and Underfitting With Machine Learning Algorithms](https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/)|
|*Memorization* and *Generalization*|Memorizing, given facts, is an obvious task in learning. This can be done by storing the input samples explicitly, or by identifying the concept behind the input data, and memorizing their general rules. The ability to identify the rules, to generalize, allows the system to make predictions on unknown data. Despite the strictly logical invalidity of this approach, the process of reasoning from specific samples to the general case can be observed in human learning. From <https://www.teco.edu/~albrecht/neuro/html/node9.html>.|
|*Normalization* and *Standardization*| *Normalization* is to scale the data into the interval [0,1] while *Standardization* is to rescale the datum with zero mean $0$ and unit variance $1$. See [Standardization vs. normalization](http://www.dataminingblog.com/standardization-vs-normalization/).|

![](https://srdas.github.io/DLBook/DL_images/INI2.png)

[Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent](https://arxiv.org/abs/1902.06720)

- https://srdas.github.io/DLBook/ImprovingModelGeneralization.html
- https://srdas.github.io/DLBook/HyperParameterSelection.html
- https://github.com/scutan90/DeepLearning-500-questions/tree/master/ch13_%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95
- https://arxiv.org/pdf/1803.09820.pdf
- https://github.com/kmkolasinski/deep-learning-notes/tree/master/seminars/2018-12-Improving-DL-with-tricks

See **Improve the way neural networks learn** at <http://neuralnetworksanddeeplearning.com/chap3.html>.
See more on nonconvex optimization at <http://sunju.org/research/nonconvex/>.

For some specific or given tasks, we would choose some proper models before training them.

<img title = model src=https://srdas.github.io/DLBook/DL_images/HPO3.png width=80% />

Given machine learning models, optimization methods are used to minimize the cost function or maximize the performance index.

<img title = algorithm src=https://srdas.github.io/DLBook/DL_images/HPO4.png width=80% />
<img title = "manual tuning" src="https://srdas.github.io/DLBook/DL_images/HPO2.png" width="80%" />


##### Initialization and More

As any optimization methods, initial values affect the convergence.
Some methods require that the initial values must locate at the convergence region, which means it is close to the optimal values in some sense.

[Initialization can have a significant impact on convergence in training deep neural networks. Simple initialization schemes have been found to accelerate training, but they require some care to avoid common pitfalls. In this post, we'll explain how to initialize neural network parameters effectively.](http://www.deeplearning.ai/ai-notes/initialization/)

<img title = "SGD" src="https://image.jiqizhixin.com/uploads/editor/5aa8c27f-c832-4105-83de-954be7420763/1535523576187.png" width="80%" />

**Initialization**

* https://srdas.github.io/DLBook/GradientDescentTechniques.html#InitializingWeights
* [A Mean-Field Optimal Control Formulation of Deep Learning](https://arxiv.org/abs/1807.01083)
* [An Empirical Model of Large-Batch Training Gradient Descent with Random Initialization: Fast Global Convergence for Nonconvex Phase Retrieva](http://www.princeton.edu/~congm/Publication/RandomInit/main.pdf)
* [Gradient descent and variants](http://www.cnblogs.com/yymn/articles/4995755.html)
* [REVISITING SMALL BATCH TRAINING FOR DEEP NEURAL NETWORKS@graphcore.ai](https://www.graphcore.ai/posts/revisiting-small-batch-training-for-deep-neural-networks)
* [可视化超参数作用机制：二、权重初始化](https://zhuanlan.zhihu.com/p/38315135)
* [第6章 网络优化与正则化](https://nndl.github.io/chap-%E7%BD%91%E7%BB%9C%E4%BC%98%E5%8C%96%E4%B8%8E%E6%AD%A3%E5%88%99%E5%8C%96.pdf)
* [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf)
* [Choosing Weights: Small Changes, Big Differences](https://intoli.com/blog/neural-network-initialization/)
* [Weight initialization tutorial in TensorFlow](https://adventuresinmachinelearning.com/weight-initialization-tutorial-tensorflow/)
* http://www.deeplearning.ai/ai-notes/initialization/

**Learning Rate**

* https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
* https://www.jeremyjordan.me/nn-learning-rate/
* https://sgugger.github.io/the-1cycle-policy.html
* https://arxiv.org/abs/1506.01186
* https://arxiv.org/abs/1708.07120
* https://arxiv.org/abs/1702.04283

**Batch Size**

- https://supercomputersfordl2017.github.io/Presentations/DLSC_talk.pdf
- https://openreview.net/pdf?id=B1Yy1BxCZ
- https://arxiv.org/abs/1711.00489

##### Distributed Training of Neural Networks

Data

Model


* https://github.com/wenwei202/terngrad
* [MVAPICH: MPI over InfiniBand, Omni-Path, Ethernet/iWARP, and RoCE](http://mvapich.cse.ohio-state.edu/)
* [Tutorial on Hardware Accelerators for Deep Neural Networks](http://eyeriss.mit.edu/tutorial.html)
* https://stanford.edu/~rezab/
* [CME 323: Distributed Algorithms and Optimization](https://stanford.edu/~rezab/dao/)
* [Distributed Deep Learning, Part 1: An Introduction to Distributed Training of Neural Networks](https://blog.skymind.ai/distributed-deep-learning-part-1-an-introduction-to-distributed-training-of-neural-networks/)
* [Distributed Deep Learning with DL4J and Spark](https://deeplearning4j.org/docs/latest/deeplearning4j-scaleout-intro)
* [A Hitchhiker’s Guide On Distributed Training of Deep Neural Networks](https://www.groundai.com/project/a-hitchhikers-guide-on-distributed-training-of-deep-neural-networks/1)
* [Distributed training of neural networks](https://www.beyondthelines.net/machine-learning/distributed-training-of-neural-networks/)
* [A Network-Centric Hardware/Algorithm Co-Design to Accelerate Distributed Training of Deep Neural Networks](https://www.cc.gatech.edu/~hadi/doc/paper/2018-micro-inceptionn.pdf)
* [Parallel and Distributed Deep Learning](https://stanford.edu/~rezab/classes/cme323/S16/projects_reports/hedge_usmani.pdf)
* [Network Design Projects: Parallel and Distributed Deep Learning Harvard CS 144r/244r Spring 2019 ](http://www.eecs.harvard.edu/htk/courses/)
* [DIANNE is a modular software framework for designing, training and evaluating artificial neural networks](http://dianne.intec.ugent.be/)
* [BytePS : a high performance and general distributed training framework.](https://github.com/bytedance/byteps)
* [[GBDT] The purposes of using parameter server in GBDT](https://github.com/Angel-ML/angel/issues/7)

#### Regularization

In mathematics, statistics, and computer science, particularly in the fields of machine learning and inverse problems, regularization is a process of introducing additional information in order to solve an ill-posed problem or to prevent over-fitting.
In general, regularization is a technique that applies to objective functions in ill-posed optimization problems.
It changes the objective function or more generally the optimization procedure. However, it is not crystal clear that what is the relationship between the optimization techniques and generalization ability.
See the following links for more information on optimization and generalization.

* https://www.inference.vc/sharp-vs-flat-minima-are-still-a-mystery-to-me/
* https://arxiv.org/abs/1506.02142
* https://arxiv.org/abs/1703.04933
* https://arxiv.org/abs/1810.05369
* http://www.offconvex.org/2017/12/08/generalization1/
* http://www.offconvex.org/2018/02/17/generalization2/
* http://www.offconvex.org/2017/03/30/GANs2/
* https://machinelearningmastery.com/blog/
* http://www.mit.edu/~9.520/fall16/
* http://lcsl.mit.edu/courses/regml/regml2016/
* https://chunml.github.io/ChunML.github.io/tutorial/Regularization/
* https://blog.csdn.net/xzy_thu/article/details/80732220
* https://srdas.github.io/DLBook/ImprovingModelGeneralization.html#Regularization

##### Parameter norm penalty

The $\ell_2$ norm penalty  is to add the squares of $\ell_2$ norm of parameters to the objective function $J(\theta)$ to reduce the parameters(or weights) as shown in ridge regression with regular term coefficient $\lambda$, i.e.
$J(\theta)+\lambda {\|\theta\|}_{2}^{2}.$
Suppose  $E(\theta)=J(\theta)+\lambda {\|\theta\|}_{2}^{2}$, the gradient descent take approximate (maybe inappropriate)  form
$$
\theta=\theta-\eta\frac{\partial E(\theta)}{\partial \theta}=\theta -\eta\frac{\partial J(\theta)}{\partial \theta}-2\eta\lambda \theta
$$
thus
$$
\frac{\partial J(\theta)}{\partial \theta} = -2\lambda\theta\implies J(\theta)=e^{-2\lambda \theta}.
$$

So $\lim_{\lambda\to \infty}J(\theta)\to 0$.
If we want to find the minima of $E(\theta)$, $\theta$ will decay to $0$.
It extends to the following iterative formula:
$$
\theta^{t+1} = (1-\lambda)\theta^{t}-\alpha_{t}\frac{\partial J(\theta^{t})}{\partial \theta},
$$
where $\lambda$  determines how you trade off the original cost $J(\theta)$ with the large weights penalization.
The new term $\lambda$ coming from the regularization causes the weight to decay in proportion to its size.

* https://stats.stackexchange.com/questions/70101/neural-networks-weight-change-momentum-and-weight-decay
* https://metacademy.org/graphs/concepts/weight_decay_neural_networks
* https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-neural-networks-with-weight-constraints-in-keras/
* https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/
* https://papers.nips.cc/paper/4409-the-manifold-tangent-classifier.pdf

The $\ell_1$ norm penalty is also used in deep learning as in **LASSO**. It is to solve the following optimization problem:
  $$\min_{\theta}J(\theta)+\lambda{\|\theta\|}_1,$$
where $\lambda$ is a hyperparameter. Sparsity  brings to the model as shown as in **LASSO**.
* [深度学习训练中是否有必要使用L1获得稀疏解?](https://www.zhihu.com/question/51822759/answer/675969996)

`Max norm constraints` is to set an upper bound to regularize the networks, i.e., it is to minimize the Constrained cost function
$$
J(\theta), \qquad s.t. \qquad \|\theta \| \leq c.
$$

Consider the fact that the parameters or weights are always in the matrix form, i.e.,
$$\theta=\{W_1, W_2, \dots, W_n\}$$

the regularization terms sometimes are in the sum of norm of matrix in each layer.

`Tangent prop` is to minimize the cost function with penalty on gradient:
$$ J(\theta)+\sum_{i} [(\nabla_x f(x)^T v^{(i)})]^2 $$

<img title = "The Manifold Tangent Classifier" src="http://res.cloudinary.com/hrscywv4p/image/upload/c_limit,fl_lossy,h_1440,w_720,f_auto,q_auto/v1/29096/manifold_copy_sycxlo.png" width="70%" />

##### Early stop

Its essential is to make a balance in memorization and generalization.
Early stopping is to stop the procedure before finding the minima of cost in training data. It is one direct application of **cross validation**.

![](https://srdas.github.io/DLBook/DL_images/RL4.png)

* https://www.wikiwand.com/en/Early_stopping
* https://www.wikiwand.com/en/Cross-validation_(statistics)
* https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
* https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/

##### Dropout

It is to cripple the connections stochastically, which  is often used in visual tasks. See the original paper [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf).
<img src="https://srdas.github.io/DLBook/DL_images/dropoutRegularization.png" width="80%" />

* https://www.zhihu.com/question/24529483
* https://www.jiqizhixin.com/articles/2018-11-10-7
* https://www.jiqizhixin.com/articles/112501
* https://www.jiqizhixin.com/articles/2018-08-27-12
* https://yq.aliyun.com/articles/68901
* https://www.wikiwand.com/en/Regularization_(mathematics)
* [CNN tricks](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html)
* https://www.jeremyjordan.me/deep-neural-networks-preventing-overfitting/
* https://www.doc.ic.ac.uk/~nd/surprise_96/journal/vol4/cs11/report.html#An%20engineering%20approach
* https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/

##### Data augmentation

Data augmentation is to augment the training datum specially in visual recognition.
**Overfitting** in supervised learning is data-dependent. In other words, the model may generalize better if the data set is more diverse.
It is to collect more datum in the statistical perspective.

* [The Effectiveness of Data Augmentation in Image Classification using Deep Learning](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf)
* http://www.cnblogs.com/love6tao/p/5841648.html

***

|Feed forward and Propagate backwards|
|:----------------------------------:|
|![TF](http://beamandrew.github.io//images/deep_learning_101/tensors_flowing.gif)|

#### Ablation Studies

Ablation studies have been widely used in the field of neuroscience to tackle complex biological systems such as the extensively studied Drosophila central nervous system, the vertebrate brain and more interestingly and most delicately, the human brain. In the past, these kinds of studies were utilized to uncover structure and organization in the brain, i.e. a mapping of features inherent to external stimuli onto different areas of the neocortex. considering the growth in size and complexity of state-of-the-art artificial neural networks (ANNs) and the corresponding growth in complexity of the tasks that are tackled by these networks, the question arises whether ablation studies may be used to investigate these networks for a similar organization of their inner representations. In this paper, we address this question and performed two ablation studies in two fundamentally different ANNs to investigate their inner representations of two well-known benchmark datasets from the computer vision domain. We found that features distinct to the local and global structure of the data are selectively represented in specific parts of the network. Furthermore, some of these representations are redundant, awarding the network a certain robustness to structural damages. We further determined the importance of specific parts of the network for the classification task solely based on the weight structure of single units. Finally, we examined the ability of damaged networks to recover from the consequences of ablations by means of recovery training.


* [Ablation Studies in Artificial Neural Networks](https://arxiv.org/abs/1901.08644)
* [Ablation of a Robot’s Brain:
Neural Networks Under a Knife](https://arxiv.org/pdf/1812.05687.pdf)
* [Using ablation to examine the structure of artificial neural networks](https://techxplore.com/news/2018-12-ablation-artificial-neural-networks.html)
* [What is ablation study in machine learning](http://qingkaikong.blogspot.com/2017/12/what-is-ablation-study-in-machine.html)


## Convolutional Neural Network

Convolutional neural network is originally aimed to solve visual tasks. In so-called [Three Giants' Survey](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf), the history of ConvNet and deep learning is curated.
[Deep, Deep Trouble--Deep Learning’s Impact on Image Processing, Mathematics, and Humanity](https://sinews.siam.org/Details-Page/deep-deep-trouble-4) tells us the  mathematicians' impression on ConvNet in image processing.

### Convolutional Layer

Convolutional layer consists of padding, convolution, pooling.

#### Convolution Operation

Convolution operation is the basic element of convolution neural network.
We only talk the convolution operation in 2-dimensional space.

The image is represented as matrix or tensor in computer:

$$
M=
\begin{pmatrix}
    x_{11} & x_{12} & x_{13} & \cdots & x_{1n} \\
    x_{21} & x_{22} & x_{23} & \cdots & x_{2n} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    x_{m1} & x_{m2} & x_{m3} & \cdots & x_{mn} \\
\end{pmatrix}
$$
where each entry $x_{ij}\in\{1,2\cdots,256\},i\in\{1,2,\cdots,m\},j\in\{1,2,\cdots,n\}$.
***

* Transformation as matrix multiplication
   + The $M$ under the transformation $A$ is their product -$MA$- of which each column is the linear combination of columns of the matrix $M$. The spatial information or the relationship of the neighbors is lost.
* Spatial information extraction
   + [Kernel in image processing ](https://www.wikiwand.com/en/Kernel_(image_processing)) takes the relationship of the neighboring entries into consideration. It transforms the neighboring entries into one real value. It is the local pattern that we can learn.

In mathematics, the matrix space $\mathbb{M}^{m\times n}$ in real domain, which consists of the matrix with $m\times n$ size, is `linear isomorphic` to the linear space $\mathbb{R}^{m\times n}$.
And thus in 2-dimenional space, convolution corresponds to $\color{aqua}{\text{doubly block circulant matrix}}$ if the matrix is flatten.

<img title = "sparse representation of CNN" src="http://jermmy.xyz/images/2017-12-16/convolution-mlp-mapping.png" width="70%" />

Let $\otimes$ denote the convolution operator and $K\in\mathbb{R}^{w\times l}$, then we obtain that
$$
F=M\otimes K \in \mathbb{R}^{(m-1)\times (n-1)},
$$

where $F_{ij}=\sum_{l=0}^{w-1}\sum_{m=0}^{l-1} M_{(i+l),(j+m)}K_{(l+1),(m+1)}$.

So that $\frac{\partial F_{ij}}{\partial K_{l,m}}=M_{(i+l-1),(j+m-1)}$.

|The illustration of convolution operator|
|:---:|
|![](https://pic4.zhimg.com/v2-15fea61b768f7561648dbea164fcb75f_b.gif)|

|The Effect of Filter|
|:---:|
|(http://cs231n.github.io/assets/conv-demo/index.html)|
|![](https://ujwlkarn.files.wordpress.com/2016/08/giphy.gif?w=748)|

***
As similar as the inner product of vector, the convolution operators  can compute the similarity between the submatrix of images and the kernels (also called filters).

The convolution operators play the role as *parameter sharing* and *local connection*.

For more information on `convolution`, click the following links.

* [One by one convolution](https://iamaaditya.github.io/2016/03/one-by-one-convolution/)
* [conv arithmetic](https://github.com/vdumoulin/conv_arithmetic)
* [Understanding Convolution in Deep Learning](http://timdettmers.com/2015/03/26/convolution-deep-learning/)
* https://zhuanlan.zhihu.com/p/28749411
* http://colah.github.io/posts/2014-12-Groups-Convolution/

#### Padding

The standard convolution  operation omit the information of the boundaries. Padding is to add some $0$s outside the boundaries of the images.

|Zero padding|
|:----------:|
|![](https://blog.xrds.acm.org/wp-content/uploads/2016/06/Figure_3.png)|

https://zhuanlan.zhihu.com/p/36278093  

#### Activation

As in feedforward neural networks, an additional non-linear  operation called **ReLU** has been used after every Convolution operation.

#### Pooling as Subsampling

Pooling as subsampling is to make the model/network more robust or transformation invariant.
Spatial Pooling (also called subsampling or down-sampling) is to use some summary statistic that extract from spatial neighbors,
which reduces the dimensionality of each feature map but retains the most important information.
The function of Pooling is

- [ ] to progressively reduce the spatial size of the input representation and induce the size of receptive field.
- [ ] makes the network invariant to small transformations, distortions and translations in the input image.
- [ ] helps us arrive at an almost scale invariant representation of our image (the exact term is "equivariant").

See more in the following links:

* https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
* http://deeplearning.stanford.edu/tutorial/supervised/Pooling/
* https://machinelearning.wtf/terms/pooling-layer/

#### Max Pooling

It is to use the maximum to represent the local information.

![](https://upload.wikimedia.org/wikipedia/commons/e/e9/Max_pooling.png)

See <https://www.superdatascience.com/convolutional-neural-networks-cnn-step-2-max-pooling/>.
* https://computersciencewiki.org/index.php/Max-pooling_/_Pooling

#### Sum Pooling

It is to use the sum to represent the local information.

#### Average Pooling

It is to use the average to represent the local information.

#### Random Pooling

It is to draw a sample from the receptive field to represent the local information.
https://www.cnblogs.com/tornadomeet/p/3432093.html

### CNN

Convolution neural network (conv-net or CNN for short) is the assembly of convolution, padding, pooling and full connection , such as
$$
M\stackrel{Conv 1}{\to}H_1 \stackrel{Conv 2}{\to} H_2 \dots \stackrel{Conv l}{\to}{H} \stackrel{\sigma}{\to} y.
$$
In the $i$th layer of convolutional neural network, it can be expressed as

$$
\hat{H}_{i} = P\oplus H_{i-1}         \\
\tilde{H_i} = C_i\otimes(\hat{H}_{t})   \\
Z_i = \mathrm{N}\cdot  \tilde{H_i} \\
H_i = Pooling\cdot (\sigma\circ Z_i)
$$


where $\otimes,\oplus,\cdot$ represent convolution operation, padding and pooling, respectively.

|Diagram of Convolutional neural network|
|:-----------------------------------:|
|![](http://www.linleygroup.com/mpr/h/2016/11561/U26_F3.png)|

The outputs of each layer are matrices or tensors rather than real vectors in CNN.
The vanila CNN has no feedback or loops in the architecture.

* [CS231n Convolutional Neural Network for Visual Recognition](http://vision.stanford.edu/teaching/cs231n/index.html)
* [CNN(卷积神经网络)是什么？有入门简介或文章吗？ - 机器之心的回答 - 知乎](https://www.zhihu.com/question/52668301/answer/131573702)
* [能否对卷积神经网络工作原理做一个直观的解释？ - YJango的回答 - 知乎](https://www.zhihu.com/question/39022858/answer/194996805)
* [Awesome deep vision](https://github.com/kjw0612/awesome-deep-vision)
* [解析深度学习——卷积神经网络原理与视觉实践](http://lamda.nju.edu.cn/weixs/book/CNN_book.html)
* [Interpretable Convolutional Neural Networks](http://qszhang.com/index.php/icnn/)
* [An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)
* [Convolutional Neural Network Visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
* [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/)
* https://www.vicarious.com/2017/10/20/toward-learning-a-compositional-visual-representation/
* https://nndl.github.io/chap-%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.pdf
* https://srdas.github.io/DLBook/ConvNets.html
* https://mlnotebook.github.io/post/CNN1/
* http://scs.ryerson.ca/~aharley/vis/conv/flat.html
* https://wiki.tum.de/pages/viewpage.action?pageId=22578448


#### Optimization in CNNs


* [Bag of Tricks for Image Classification with Convolutional Neural Networks](http://arxiv.org/abs/1812.01187)
* [Applying Gradient Descent in Convolutional Neural Networks](https://iopscience.iop.org/article/10.1088/1742-6596/1004/1/012027/pdf)

##### Backpropagation in CNNs

Convolutional neural networks extract the features of the images by trial and error to tune the convolutions $\{C_i \mid i=1,2,\dots, n\}$.

At each convolutional layer,  we feed forwards the information like:

$$
\hat{H}_{i} = P \oplus H_{i-1}         \\
\tilde{H_i} = C_i \otimes(\hat{H}_{t})   \\
Z_i = \mathrm{N}\cdot  \tilde{H_i} \\
H_i = Pooling\cdot (\sigma\circ Z_i)
$$

In order to design proper convolutions for specific task automatically according to the labelled images via gradient-based optimization methods, we should compute the gradient of error with respect to convolutions:

$$
\frac{\partial L}{\partial H_n}   \to
\frac{\partial H_n}{\partial C_n} \to
\frac{\partial H_n}{\partial H_{n-1}}\to\cdots \to
\frac{\partial H_1}{\partial M}.
$$
In fully connected layer, the backpropagation is as the same as in feedforward neural networks.

In convolutional layer, it is made up of padding, convolution, activation, normalization and pooling.

Note that even that there is only one  output of the convolutional layer (for example when the kernels or filters are in the same form of input without pooling), it is accessible to compute the gradient of the kernels like the inner product.

For example, we compute the gradient with respect to the convolution:

$$
F:\mathbb{R}^{m\times n}  \to \mathbb{R}^{\hat{m}\times \hat{n}}\\
F = Pooling \cdot \mathrm{N} \cdot \sigma[(P \oplus M) \otimes C],
$$

which is a composite of operators/functions.
So that by the chain rule of derivatives

$$
\frac{\partial F}{\partial K} = \frac{\partial Pooling}{\partial n}
\frac{\partial n}{\partial \sigma}
\frac{\partial \sigma}{\partial C}
\frac{\partial C}{\partial M}.
$$


* http://jermmy.xyz/2017/12/16/2017-12-16-cnn-back-propagation/
* http://andrew.gibiansky.com/blog/machine-learning/convolutional-neural-networks/
* https://www.cnblogs.com/tornadomeet/p/3468450.html
* http://www.cnblogs.com/pinard/p/6494810.html

**Backward Propagation of the Pooling Layers**

$$\frac{\partial Pooling}{\partial n}$$
![BP in Pooling](https://lanstonchu.files.wordpress.com/2018/08/avg-pool.gif?w=614&zoom=2)

* [Convolutional Neural Network (CNN) – Backward Propagation of the Pooling Layers](https://lanstonchu.wordpress.com/2018/09/01/convolutional-neural-network-cnn-backward-propagation-of-the-pooling-layers/)

**Backward Propagation of the Normalization Layers**

$$\frac{\partial n}{\partial \sigma}$$

**Batch Normalization**

It is an effective way to accelerate deep  learning training.

![](https://img-blog.csdn.net/20161128135254463)

And `Batch Normalization` transformation is differentiable so that we can compute the gradient in backpropagation.
For example, let $\ell$ be the cost function to be minimized, then we could compute the gradients

$$
\frac{\partial \ell}{\partial \hat{x}_i} = \frac{\partial \ell}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} =
\frac{\partial \ell}{\partial y_i} \gamma
$$

****

$$
\frac{\partial \ell}{\partial \sigma_B^2} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial \sigma_B^2} = \sum_{i=1}^{m}\frac{\partial \ell}{\partial \hat{x}_i}\frac{-\frac{1}{2}(x_i - \mu_B)}{(\sigma_B^2 + \epsilon)^{\frac{3}{2}}}
$$

***

$$
\frac{\partial \ell}{\partial \mu_B} = \frac{\partial \ell}{\partial \sigma_B^2} \frac{\partial \sigma_B^2}{\partial \mu_B} + \sum_{i=1}^{m}\frac{\partial \ell}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial \mu_B} =
\frac{\partial \ell}{\partial \sigma_B^2}[\frac{2}{m}\sum_{i=1}^{m}(x_i-\mu_B)] + \sum_{i=1}^{m}\frac{\partial \ell}{\partial \hat{x}_i}\frac{-1}{\sqrt{(\sigma_B^2 + \epsilon)}}
$$

***
$$
\frac{\partial \ell}{\partial {x}_i} = \frac{\partial \ell}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial {x}_i} + \frac{\partial \ell}{\partial \mu_B} \frac{\partial \mu_B}{\partial {x}_i} + \frac{\partial \ell}{\partial \sigma^2_B}  \frac{\partial \sigma^2_B}{\partial {x}_i}
$$

$$
\frac{\partial \ell}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i} \hat{x}_i \\
\frac{\partial \ell}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i} \hat{x}_i
 $$

|Batch Normalization in Neural|
|:---:|
|![BN](https://srdas.github.io/DLBook/DL_images/BatchNormalization.png)|

| Batch Normalization in Neural Network |
|:---:|
|<img title="BN in NN" src="https://image.jiqizhixin.com/uploads/editor/4f055102-a306-4a76-9dec-d0c69ef60e6f/1535523577800.png" width="80%" />|

**Layer Normalization**

Different from the Batch Normalization, `Layer Normalization` compute the average $\mu$ or variance $\sigma$ for all the inputs of the given hidden layer, as
$$
\mu = \frac{1}{n}\sum_{i}^{n} x_i \\
\sigma = \sqrt{\frac{1}{n}\sum_{i}^{n}(x_i - \mu)^2 + \epsilon}
$$
where ${n}$ is the number of units in the given hidden layer.

|Layer Normalization|
|:---:|
|![LN](https://pic1.zhimg.com/80/v2-2f1ad5749e4432d11e777cf24b655da8_hd.jpg)|

|Layer Normalization in FNN|
|:---:|
|<img title="LN+FNN" src="https://image.jiqizhixin.com/uploads/editor/eaf495c1-9254-4418-b15d-53ba4b0b09c4/1535523579078.png" width="60%" />|

**Weight Normalization**

<img title="WN in NN" src="https://pic2.zhimg.com/80/v2-93d904e4fff751a0e5b940ab3c27b6d5_hd.jpg" width="70%" />

* [Batch normalization 和 Instance normalization 的对比？ - Naiyan Wang的回答 - 知乎](https://www.zhihu.com/question/68730628/answer/277339783)
* [Weight Normalization 相比 batch Normalization 有什么优点呢？](https://www.zhihu.com/question/55132852/answer/171250929)
* [深度学习中的Normalization模型](https://www.jiqizhixin.com/articles/2018-08-29-7)
* [Group Normalization](https://arxiv.org/abs/1803.08494)
* [Busting the myth about batch normalization at paperspace.com](https://blog.paperspace.com/busting-the-myths-about-batch-normalization/)
* https://zhuanlan.zhihu.com/p/33173246
* The original paper *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* at <https://arxiv.org/pdf/1502.03167.pdf>.
* https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/
* [An Overview of Normalization Methods in Deep Learning](http://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/)
* [Batch Normalization](https://srdas.github.io/DLBook/GradientDescentTechniques.html#BatchNormalization)
* https://arxiv.org/abs/1902.08129
* [Normalization in Deep Learning](https://calculatedcontent.com/2017/06/16/normalization-in-deep-learning/)

**Backward Propagation of the Activation Layers**

$$\frac{\partial \sigma}{\partial C}$$

| $\sigma$|$\frac{\partial \sigma}{\partial C}$|
|---|---|
|activation function | the partial derivatives after convolution |

**Backward Propagation of the Convolution Layers**

$$\frac{\partial C}{\partial M}$$


##### Data Augmentation

- [深度学习中的Data Augmentation方法和代码实现](https://absentm.github.io/2016/06/14/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84Data-Augmentation%E6%96%B9%E6%B3%95%E5%9)
- [The Effectiveness of Data Augmentation in Image Classification using Deep Learning](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf)
- [500 questions on deep learning](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch04_%E7%BB%8F%E5%85%B8%E7%BD%91%E7%BB%9C/%E7%AC%AC%E5%9B%9B%E7%AB%A0_%E7%BB%8F%E5%85%B8%E7%BD%91%E7%BB%9C.md)
- [算法面试](https://github.com/imhuay/Algorithm_Interview_Notes-Chinese/blob/master/B-%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/B-%E4%B8%93%E9%A2%98-%E5%9F%BA%E6%9C%AC%E6%A8%A1%E5%9E%8B.md)
- [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision)
- [some research blogs](https://world4jason.gitbooks.io/research-log/content/deepLearning/Deep%20Learning.html)
- [SOTA](http://openresearch.ai/tags/sota)
- [Deep Neural Networks Motivated by Partial Differential Equations](https://arxiv.org/pdf/1804.04272.pdf)
***

<img title="DL Approach" src="http://1reddrop.com/wp-content/uploads/2016/08/deeplearning.jpg" width="80%" />

- https://grzegorzgwardys.wordpress.com/2016/04/22/8/
- https://lanstonchu.wordpress.com/category/deep-learning/

### Visualization /Interpretation of CNN

[It has shown](https://openreview.net/pdf?id=Bygh9j09KX) that
> ImageNet trained CNNs are strongly biased towards recognising `textures` rather than `shapes`,
which is in stark contrast to human behavioural evidence and reveals fundamentally different classification strategies.

* [Deep Visualization](http://yosinski.com/deepvis)
* [Interpretable Representation Learning for Visual Intelligence](http://bzhou.ie.cuhk.edu.hk/publication/thesis.pdf)
* [IMAGENET-TRAINED CNNS ARE BIASED TOWARDS TEXTURE; INCREASING SHAPE BIAS IMPROVES ACCURACY AND ROBUSTNESS](https://openreview.net/pdf?id=Bygh9j09KX)
* [2017 Workshop on Visualization for Deep Learning](https://icmlviz.github.io/)
* [Understanding Neural Networks Through Deep Visualization](http://yosinski.com/deepvis)
* [vadl2017: Visual Analysis of Deep Learning](https://vadl2017.github.io/)
* [Multifaceted feature visualization: Uncovering the different types of features learned by each neuron in deep neural networks](http://www.evolvingai.org/mfv)
* [Interactive Visualizations for Deep Learning](http://predictive-workshop.github.io/papers/vpa2014_1.pdf)
* https://www.zybuluo.com/lutingting/note/459569
* https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
* [卷积网络的可视化与可解释性（资料整理） - 陈博的文章 - 知乎](https://zhuanlan.zhihu.com/p/36474488)
* https://zhuanlan.zhihu.com/p/24833574
* https://zhuanlan.zhihu.com/p/30403766
* https://zhuanlan.zhihu.com/p/28054589
* https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
* http://people.csail.mit.edu/bzhou/ppt/presentation_ICML_workshop.pdf
* https://www.robots.ox.ac.uk/~vedaldi//research/visualization/visualization.html
* https://www.graphcore.ai/posts/what-does-machine-learning-look-like
* https://srdas.github.io/DLBook/ConvNets.html#visualizing-convnets
* [CNN meets PDEs](https://deqefw538d79t.cloudfront.net/api/file/jSBt1r2nTeP0ItVIVY9o?cache=true)
* [Deep Neural Network motivated by PDEs](https://gateway.newton.ac.uk/sites/default/files/asset/doc/1805/2018-DeepLearning-beamer_0.pdf)
* https://www.semanticscholar.org/author/Lars-Ruthotto/2557699
****
|graphcore.ai|
|:----------:|
| <img title="DL Approach" src="https://www.graphcore.ai/hubfs/images/alexnet_label%20logo.jpg?t=1541693113453" width="60%" /> |
****

## Recurrent Neural Networks and Long Short-Time Memory

[601.765 Machine Learning: Linguistic & Sequence Modeling](https://seq2class.github.io/)
[Scientia est Potentia](http://complx.me/)

Recurrent neural networks are aimed to handle sequence data such as time series.
Recall the recursive form of feedforward neural networks:
$$
\begin{align}
\mathbf{z}_i &= W_i H_{i-1}+b_i,     \\
    H_{i}    &= \sigma\circ(\mathbf{z}_i),
\end{align}
$$

where $W_i\in\mathbb{R}^{l_{i}\times l_{i-1}}$, $H_i (\text{as well as}\, b_i)\in \mathbb{R}^{l_i}\forall i\in\{1, 2, \dots, D\}$ and $\sigma$ is activation function.
In convention, $H_0$ is defined as input $X\in\mathbb{R}^{p}$.
In short, the formula in the $i$th layer of feedforward neural network is given by
$$
H_i=\sigma\circ(W_i H_{i-1}+b_i).
$$

<img title="RNN" src="http://www.dam.brown.edu/people/mraissi/assets/img/RNN.png" width="60%" />

It may suit the **identically independently distributed** data set.
However, the sequence data is not **identically independently distributed** in most cases. For example, the outcome of current decision determines the next decision.

<img title="RNN diagram" src="https://srdas.github.io/DLBook/DL_images/rnn1b.png" width="60%" />

In mathematics it can be expressed as

$$
H_{t}=\sigma\circ(X_t,H_{t-1})=\sigma\circ(W H_{t-1} + U X_{t}+b),
$$

where $X_{t}\in\mathbb{R}^{p}$ is the output $\forall t\in\{1,2\dots,\tau\}$.
The Hidden Variable sequence  $H_{t}$  captures the lossy history of the  $X_t$  sequence, and hence serves as a type of memory.
We can compute the gradient with respect to the parameters $W, U, b$:

$$
\frac{\partial H_t}{\partial W} = \sigma^{\prime} \circ H_{t-1}\\
\frac{\partial H_t}{\partial U} = \sigma^{\prime} \circ X_t \\
\frac{\partial H_t}{\partial b} =  \sigma^{\prime} \circ b  .
$$

|RNN Cell|
|:------:|
|![](http://imgtec.eetrend.com/sites/imgtec.eetrend.com/files/201810/blog/18051-37153-6.gif)|

For each step from $t=1$ to $t=\tau$, the complete update equations of RNN:

$$
\begin{align}
H_{t} &=\sigma\circ(W H_{t-1} + U X_{t} + b) \\
O_{t} &= \mathrm{softmax}(V H_{t} + c)
\end{align}
$$

where the parameters are the bias vectors $b$ and $c$ along with the weight matrices
$U$, $V$ and $W$, respectively for input-to-hidden, hidden-to-output and hidden-to-hidden connections.

![](http://5b0988e595225.cdn.sohucs.com/images/20181017/a8fdc5e0afd147fa8043473a04d7127e.gif)

|Types of RNN|
|:---:|
|<img title="RNN Types" src="https://devblogs.nvidia.com/wp-content/uploads/2015/05/Figure_3-624x208.png" width="60%" />|
|<img title="RNN Types" src="https://www.altoros.com/blog/wp-content/uploads/2017/01/Deep-Learning-Using-TensorFlow-recurrent-neural-networks.png" width="68%" />|

* https://srdas.github.io/DLBook/RNNs.html
* [A Beginner’s Guide on Recurrent Neural Networks with PyTorch](https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/)
* [Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
* [Recurrent Neural Networks Tutorial, Part 2 – Implementing a RNN with Python, Numpy and Theano](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)
* [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)
* <https://github.com/kjw0612/awesome-rnn>
* <https://blog.acolyer.org/2017/03/03/rnn-models-for-image-generation/>
* <https://nlpoverview.com/>

### The Back Propagation Through Time (BPTT) Algorithm

<img title="RNN Types" src="https://srdas.github.io/DLBook/DL_images/rnn3.png" width="60%" />


- https://srdas.github.io/DLBook/RNNs.html#Training

### Bi-directional RNN

For each step from $t=1$ to $t = \tau$, the complete update equations of RNN:

$$
\begin{align}
\stackrel{\rightarrow} H_{t} &= \sigma\circ(W_1\stackrel{\rightarrow}H_{t-1}+U_1X_{t}+b_1) \\
\stackrel{\leftarrow}H_{t}  &= \sigma\circ(W_2\stackrel{\leftarrow}H_{t+1}+U_2X_{t}+b_2) \\
O_{t} &= \mathrm{softmax}(VH_{t}+c)
\end{align}
$$

where $H_{t} = [\stackrel{\rightarrow} H_{t};\stackrel{\leftarrow} H_{t}]$.

|Bi-directional RNN|
|:----------------:|
|<img src="http://www.wildml.com/wp-content/uploads/2015/09/bidirectional-rnn.png" width = "60%" />|
|The bold line(__) is computed earlier than the dotted line(...).|

[deepai.org](https://deepai.org/machine-learning-glossary-and-terms/bidirectional-recurrent-neural-networks)

- [ ] http://building-babylon.net/2018/05/08/siegelmann-sontags-on-the-computational-power-of-neural-nets/

### LSTM

There are several architectures of LSTM units. A common architecture is composed of a *memory cell, an input gate, an output gate and a forget gate*.
It is the first time to solve the **gradient vanishing problem** and
**long-term dependencies** in deep learning.

|LSTM block|
|:---:|
|See *LSTM block* at(https://devblogs.nvidia.com/wp-content/uploads/2016/03/LSTM.png)|
|<img src="http://5b0988e595225.cdn.sohucs.com/images/20181017/cd0e8107c3f94e849bd82e0fd0123776.jpeg" width = "60%" />|

* forget gate

$$f_{t} = \sigma(W_{f}[h_{t-1},x_t]+b_f). \tag{forget gate}$$

|forget gate|
|:---:|
|<img title = "forget gate" src="http://5b0988e595225.cdn.sohucs.com/images/20181017/87957492ade3445ea90871dda02c92ca.gif" width = "60%" />|

* input gate

$$
i_{t} = \sigma(W_i[h_{t-1}, x_t] + b_i),
\tilde{C}_{t} = tanh(W_C[h_{t-1}, x_t] + b_C).\tag{input gate}
$$

|input gate|
|:---:|
|<img title = " input gate" src ="http://5b0988e595225.cdn.sohucs.com/images/20181017/dac15a9da4164b3da346ac891d10ff9e.gif" width = "60%" />|

* memory cell

$$C_{t} = f_t \odot c_{t-1}+i_{t} \otimes {\tilde{C}_{t}}.\tag{memory cell}$$

|memory cell|
|:---:|
|<img title = " input gate" src ="http://5b0988e595225.cdn.sohucs.com/images/20181017/ad6435e481064d57833a4733e716fa8f.gif" width = "60%" />|

* output gate

$$o_{t} = \sigma(W_{O}[h_{t-1},x_t]+b_{O}),\tag{ouput gate 1}$$
and
$$Z_{t} = O_{t}\odot tanh(c_{t}).\tag{ouput gate 2}$$

|output gate|
|:---:|
|<img title = "output gate" src ="http://5b0988e595225.cdn.sohucs.com/images/20181017/bca51dcc89f14b1c9a2e9076f419540a.gif" width = "60%" />|

|[Inventor of LSTM](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735?journalCode=neco)|
|:---:|
|<img title = "Juergen Schmidhuber" src ="https://www.analyticsindiamag.com/wp-content/uploads/2018/09/jurgen-banner.png" width = "60%" /> more on (http://people.idsia.ch/~juergen/)|
|![Internals of a LSTM Cell](https://srdas.github.io/DLBook/DL_images/rnn22.png)|


* [LSTM in Wikipeida](https://www.wikiwand.com/en/Long_short-term_memory)
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) and its Chinese version <https://www.jianshu.com/p/9dc9f41f0b29>.
* [LTSMvis](http://lstm.seas.harvard.edu/)
* [Jürgen Schmidhuber's page on Recurrent Neural Networks](http://people.idsia.ch/~juergen/rnn.html)
* [Exporing LSTM](http://blog.echen.me/2017/05/30/exploring-lstms/)
* [Essentials of Deep Learning : Introduction to Long Short Term Memory](https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/)
* [LSTM为何如此有效？](https://www.zhihu.com/question/278825804)

### Gated Recurrent Units (GRUs)

GRUs are a recently proposed alternative to LSTMs, that share its good properties, i.e., they are also designed to avoid the Vanishing Gradient problem. The main difference from LSTMs is that GRUs don’t have a cell memory state  $C_t$ , but instead are able to function effectively using the Hidden State  $Z_t$  alone.

<img title = "GURs" src ="https://srdas.github.io/DLBook/DL_images/rnn6.png" width = "60%" />

* Update Gate
$$
\begin{equation}
o_t = \sigma(W^o X_t + U^o Z_{t-1}). \tag{ Update Gate}
\end{equation}$$

* Reset Gate
$$
\begin{equation}
r_t = \sigma(W^r X_t + U^r Z_{t-1}). \tag{Reset Gate}
\end{equation}
$$

* Hidden State
$$
\begin{equation}
{\tilde Z}_t = \tanh(r_t\odot U Z_{t-1} + W X_t), \\
Z_t = (1 - o_t)\odot {\tilde Z}_t + o_t\odot Z_{t-1}). \tag{Hidden State}
\end{equation}
$$
***
* https://wugh.github.io/posts/2016/03/cs224d-notes4-recurrent-neural-networks-continue/
* https://srdas.github.io/DLBook/RNNs.html#GRU
* https://d2l.ai/chapter_recurrent-neural-networks/gru.html

### Deep RNN

Deep RNN is composed of *RNN cell* as MLP is composed of perceptrons.
For each step from $t=1$ to $t=\tau$, the complete update equations of deep $d$-RNN  at the $i$th layer:
$$
\begin{align}
H_{t}^{i} &= \sigma\circ(W_{i} H_{t-1} + U_{i} X_{t} + b) \\
O_{t} &= \mathrm{softmax}(V H_{d} + c)
\end{align}
$$
where the parameters are the bias vectors $b$ and $c$ along with the weight matrices
$U_i$, $V$ and $W_i$, respectively for input-to-hidden, hidden-to-output and hidden-to-hidden connections for $i\in \{1,2,\cdots,d\}$.

Other RNN cells also can compose deep RNN via this stacking way such as deep Bi-RNN networks.

|Deep Bi-RNN|
|:---:|
|![brnn](http://opennmt.net/OpenNMT/img/brnn.png)|

* [Application of Deep RNNs](http://blog.songru.org/posts/notebook/Opinion_Mining_with_Deep_Recurrent_Neural_Networks_NOTE/)
* http://opennmt.net/OpenNMT/training/models/
* https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

***

* [A Tour of Recurrent Neural Network Algorithms for Deep Learning](https://machinelearningmastery.com/recurrent-neural-network-algorithms-for-deep-learning/)
* [Recurrent Neural Networks Tutorial, Part 1 – Introduction to RNNs](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
* [Recurrent Neural Networks Tutorial, Part 2 – Implementing a RNN with Python, Numpy and Theano](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)
* [当我们在谈论 Deep Learning：RNN 其常见架构](https://zhuanlan.zhihu.com/p/27485750)
* [RNN in Wikipeida](https://en.wikipedia.org/wiki/Recurrent_neural_network)
* [Awesome RNN](https://github.com/kjw0612/awesome-rnn)
* [RNN in metacademy](https://metacademy.org/graphs/concepts/recurrent_neural_networks)
* https://zhuanlan.zhihu.com/p/49834993
* [RNNs in Tensorflow, a Practical Guide and Undocumented Features](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [Visualizations of RNN units
Diagrams of RNN unrolling, LSTM and GRU.](https://kvitajakub.github.io/2016/04/14/rnn-diagrams/)
* http://www.zmonster.me/notes/visualization-analysis-for-rnn.html
* http://imgtec.eetrend.com/d6-imgtec/blog/2018-10/18051.html
* https://arxiv.org/pdf/1801.01078.pdf
* http://www.sohu.com/a/259957763_610300
* https://skymind.ai/wiki/lstm
* [循环神经网络(RNN, Recurrent Neural Networks)介绍](https://blog.csdn.net/heyongluoyao8/article/details/48636251)
* https://arxiv.org/pdf/1506.02078.pdf
* https://srdas.github.io/DLBook/RNNs.html#Applications

### Attention Mechanism

An attention model is a method that takes $n$ arguments $y_1, \dots, y_n$  and a context $c$. It return a vector $z$ which is supposed to be the  **summary** of the ${y}_i\in \mathbb{R}^{d}$, focusing on information linked to the context $c$. More formally, it returns a weighted arithmetic mean of the $y_i$, and the weights are chosen according the relevance of each $y_i$ given the context $c$.
In mathematics, it can be expressed as:

$$
{\alpha}_i = softmax [s(y_i,c)]               \\
z = \sum_{i=1}^{n} {\alpha}_i y_i
$$

where $s(\cdot, \cdot)$ is the attention scoring function.
The attention scoring function $s({y}_i, c)$ is diverse, such as:

* the additive model $s({y}_i, c) = v^{T} tanh\circ (W {y}_i + U c)$, where $v \in \mathbb{R}^{d}$, $W \in \mathbb{R}^{d\times d}$, $U \in \mathbb{R}^{d}$ are parameters to learn;
* the inner product model $s({y}_i, c) = \left< {y}_i, c \right>$, i.e. the inner product of ${y}_i, c$;
* the scaled inner product model $s({y}_i, c) = \frac{\left< {y}_i, c \right>}{d}$,where $d$ is the dimension of input ${y}_i$;
* the bilinear model $s({y}_i, c) = {y}_i^{T} W c$, where $W\in \mathbb{R}^{d\times d}$ is parameter matrix to learn.

It is always as one component of some complex network as normalization.

<img title="Attension" src="https://srdas.github.io/DLBook/DL_images/rnn32.png" width = "80%" />

***

* [第 8 章 注意力机制与外部记忆](https://nndl.github.io/chap-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E4%B8%8E%E5%A4%96%E9%83%A8%E8%AE%B0%E5%BF%86.pdf)
* https://srdas.github.io/DLBook/RNNs.html#Memory
* https://skymind.ai/wiki/attention-mechanism-memory-network
* https://distill.pub/2016/augmented-rnns/
* https://blog.heuritech.com/2016/01/20/attention-mechanism/
* [Attention mechanism](https://github.com/philipperemy/keras-attention-mechanism)
* [Attention and Memory in Deep Learning and NLP](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)
* http://www.deeplearningpatterns.com/doku.php?id=attention
* [细想神毫注意深-注意力机制 - 史博的文章 - 知乎](https://zhuanlan.zhihu.com/p/51747716)
* http://www.charuaggarwal.net/Chap10slides.pdf
* https://d2l.ai/chapter_attention-mechanism/seq2seq-attention.html


***
* [What is DRAW (Deep Recurrent Attentive Writer)?](http://kvfrans.com/what-is-draw-deep-recurrent-attentive-writer/)
* [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
* [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/)

## Recursive Neural Network

* http://www.iro.umontreal.ca/~bengioy/talks/gss2012-YB6-NLP-recursive.pdf
* https://www.wikiwand.com/en/Recursive_neural_network
* https://cs224d.stanford.edu/lectures/CS224d-Lecture10.pdf
* https://devblogs.nvidia.com/recursive-neural-networks-pytorch/
* http://sei.pku.edu.cn/~luyy11/slides/slides_141029_RNN.pdf

|Diagram of Recursive Neural Network|
|:---------------------------------:|
|<img title = "dr" src ="http://www.cs.cornell.edu/~oirsoy/files/drsv/deep-recursive.png" width = "60%" />|


## Generative Models

<img title = "gan" src ="https://reiinakano.github.io/images/wp/s1/shard-painting.jpg" width = "60%" />


* [CSC 2541 Fall 2016: Differentiable Inference and Generative Models](https://www.cs.toronto.edu/~duvenaud/courses/csc2541/index.html)
* [Learning Discrete Latent Structure](https://duvenaud.github.io/learn-discrete/)
* [The Living Thing / Notebooks : Reparameterisation tricks in differentiable inference](https://livingthing.danmackinlay.name/reparameterisation_diff.html)
- https://grzegorzgwardys.wordpress.com/2016/06/19/convolutional-autoencoder-for-dummies/

### Generative Adversarial Network

http://unsupervised.cs.princeton.edu/deeplearningtutorial.html

It origins from <http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf>.
It is a generative model via an adversarial process.
It trains a generative model $G$ that captures the data distribution, and a discriminative model $D$ that estimates
the probability that a sample came from the training data rather than $G$.
The training procedure for $G$ is to maximize the probability of $D$ making a mistake.
This framework corresponds to a minimax two-player game. In the space of arbitrary
functions $G$ and $D$, a unique solution exists, with $G$ recovering the training data
distribution and $D$ equal to $\frac{1}{2}$ everywhere.

It is not to minimize the cost function or errors as that in supervised machine learning.
In mathematics, it is saddle point optimization.
Thus some optimization or regularization techniques are not suitable for this framework.
It requires some methods to find the proper generator $G$ and discriminator $D$.

|Generative Adversarial Network|
|:----------------------------:|
|<img title = "gan loss" src ="https://image.slidesharecdn.com/generativeadversarialnetworks-161121164827/95/generative-adversarial-networks-11-638.jpg" width = "70%" />|

As a generative model, it is really important to evaluate the quantity of the model output.
Suppose there is a model used to write Chinese traditional poem, how does the machine know it is a fantastic masterpiece? How does it write a novel poem in a given topic or form? The loss function or evaluation is implicit.    
One solution is to train another program to evaluate the performance of generative model.

The idea behind the GAN:

* Idea 1: Deep nets are good at recognizing images, then let it judge of the outputs of a generative model;
* Idea 2: If a good discriminator net has been trained, use it to provide “gradient feedback” that improves the generative model.
* Idea 3: Turn the training of the generative model into a game of many moves or alternations.

In mathematics, it is in the following form
$$\min_{G}\max_{D}\mathbb{E}_{x\sim P} [f(D(x))] + \mathbb{E}_{h}[f(1 - D(G(h)))]$$

where $G$ is the generator and $D$ is the discriminator.

`Cycle-GAN`
****

* https://skymind.ai/wiki/generative-adversarial-network-gan
* [千奇百怪的GAN变体，都在这里了（持续更新嘤） - 量子学园的文章 - 知乎](https://zhuanlan.zhihu.com/p/26491601)
* [生成模型中的左右互搏术：生成对抗网络GAN——深度学习第二十章（四） - 川陀学者的文章 - 知乎](https://)https://zhuanlan.zhihu.com/p/37846221)
* [Really awesome GANs](https://github.com/nightrome/really-awesome-gan)
* [GAN zoo](https://github.com/hindupuravinash/the-gan-zoo)
* [Open Questions about Generative Adversarial Networks](https://distill.pub/2019/gan-open-problems/)
* https://gandissect.csail.mit.edu/
* https://poloclub.github.io/ganlab
* https://github.com/nndl/Generative-Adversarial-Network-Tutorial
* https://danieltakeshi.github.io/2017/03/05/understanding-generative-adversarial-networks/
* http://aiden.nibali.org/blog/2016-12-21-gan-objective/
* http://www.gatsby.ucl.ac.uk/~balaji/Understanding-GANs.pdf
* https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html
* https://www.cs.princeton.edu/courses/archive/spring17/cos598E/GANs.pdf
* https://seas.ucla.edu/~kao/nndl/lectures/gans.pdf

![](https://gandissect.csail.mit.edu/img/framework-d2.svg)
More generative models include GLOW, variational autoencoder and energy-based models.

### Variational Autoencoder

- https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
- http://kvfrans.com/variational-autoencoders-explained/
- https://ermongroup.github.io/cs228-notes/extras/vae/
- https://www.jeremyjordan.me/variational-autoencoders/
- http://anotherdatum.com/vae-moe.html
- http://anotherdatum.com/vae.html
- [STYLE TRANSFER: VAE](https://www.andrewszot.com/blog/machine_learning/deep_learning/variational_autoencoders)
- [Pixel Art generation using VAE](https://mlexplained.wordpress.com/2017/05/06/pixel-art-generation-using-vae/)
- [Pixel Art generation Part 2. Using Hierarchical VAE](https://mlexplained.wordpress.com/2017/07/27/pixel-art-generation-part-2-using-hierarchical-vae/)


### PixelRNN

- https://zhuanlan.zhihu.com/p/25299749
- http://vsooda.github.io/2016/10/30/pixelrnn-pixelcnn/
- http://www.qbitlogic.com/challenges/pixel-rnn
***
* https://worldmodels.github.io/
* https://openai.com/blog/glow/
* https://github.com/wiseodd/generative-models
* http://www.gg.caltech.edu/genmod/gen_mod_page.html
* https://deepgenerativemodels.github.io/
* [Flow-based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)
* [Deep Generative Models](https://ermongroup.github.io/generative-models/)
* [Teaching agents to paint inside their own dreams](https://reiinakano.github.io/2019/01/27/world-painters.html)
* [Optimal Transformation and Machine Learning](http://otml17.marcocuturi.net/)

- http://computationalcreativity.net/home/

## Geometric Deep Learning

### Graph Convolution Network

Graph can be represented as `adjacency matrix` as shown in *Graph Algorithm*. However, the adjacency matrix only describe the connections between the nodes. The feature of the nodes does not appear. The node itself really matters.
For example, the chemical bonds can be represented as `adjacency matrix` while the atoms in molecule really determine the properties of the molecule.

A naive approach is to concatenate the `feature matrix` $X\in \mathbb{R}^{N\times E}$ and `adjacency matrix` $A\in \mathbb{R}^{N\times N}$, i.e. $X_{in}=[X, A]\in \mathbb{R}^{N\times (N+E)}$. And what is the output?

How can deep learning apply to them?

> For these models, the goal is then to learn a function of signals/features on a graph $G=(V,E)$ which takes as input:

> * A feature description $x_i$ for every node $i$; summarized in a $N\times D$ feature matrix $X$ ($N$: number of nodes, $D$: number of input features)
> * A representative description of the graph structure in matrix form; typically in the form of an adjacency matrix $A$ (or some function thereof)

> and produces a node-level output $Z$ (an $N\times F$ feature matrix, where $F$ is the number of output features per node). Graph-level outputs can be modeled by introducing some form of pooling operation (see, e.g. [Duvenaud et al., NIPS 2015](http://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints)).

Every neural network layer can then be written as a non-linear function
$${H}_{i+1} = \sigma \circ ({H}_{i}, A)$$
with ${H}_0 = {X}_{in}$ and ${H}_{d} = Z$ (or $Z$ for graph-level outputs), $d$ being the number of layers.
The specific models then differ only in how $\sigma$ is chosen and parameterized.

For example, we can consider a simple form of a layer-wise propagation rule
$$
{H}_{i+1} = \sigma \circ ({H}_{i}, A)=\sigma \circ(A {H}_{i} {W}_{i})
$$
where ${W}_{i}$ is a weight matrix for the $i$-th neural network layer and $\sigma (\cdot)$ is is a non-linear activation function such as *ReLU*.

* But first, let us address two limitations of this simple model: multiplication with $A$ means that, for every node, we sum up all the feature vectors of all neighboring nodes but not the node itself (unless there are self-loops in the graph). We can "fix" this by enforcing self-loops in the graph: we simply add the identity matrix ${I}$ to ${A}$.

* The second major limitation is that $A$ is typically not normalized and therefore the multiplication with $A$ will completely change the scale of the feature vectors (we can understand that by looking at the eigenvalues of $A$).Normalizing $A$ such that all rows sum to one, i.e. $D^{−1}A$, where $D$ is the diagonal node degree matrix, gets rid of this problem.

In fact, the propagation rule introduced in [Kipf & Welling (ICLR 2017)](https://arxiv.org/abs/1609.02907) is given by:
$$
{H}_{i+1} = \sigma \circ ({H}_{i}, A)=\sigma \circ(\hat{D}^{-\frac{1}{2}} \hat{A} \hat{D}^{-\frac{1}{2}} {H}_{i} {W}_{i}),
$$
with $\hat{A}=A+I$, where ${I}$ is the identity matrix and $\hat{D}$ is the diagonal node degree matrix of $\hat{A}$.
See more details at [Multi-layer Graph Convolutional Network (GCN) with first-order filters](http://tkipf.github.io/graph-convolutional-networks/).

Like other neural network, GCN is also composite of linear and nonlinear mapping. In details,\

1. $\hat{D}^{-\frac{1}{2}} \hat{A} \hat{D}^{-\frac{1}{2}}$ is to normalize the graph structure;
2. the next step is to multiply node properties and weights;
3. Add nonlinearities by activation function $\sigma$.

[See more at experoinc.com](https://www.experoinc.com/post/node-classification-by-graph-convolutional-network).

<img title = "GCN" src ="http://tkipf.github.io/graph-convolutional-networks/images/gcn_web.png" width = "70%" />
<img title = "CNN VS. GCNN" src ="https://research.preferred.jp/wp-content/uploads/2017/12/cnn-gcnn.png" width = "70%" />


* http://deeploria.gforge.inria.fr/thomasTalk.pdf
* https://arxiv.org/abs/1812.04202
* https://skymind.ai/wiki/graph-analysis
* [Graph-based Neural Networks](https://github.com/sungyongs/graph-based-nn)
* [Geometric Deep Learning](http://geometricdeeplearning.com/)
* [Deep Chem](https://deepchem.io/)
* [GRAM: Graph-based Attention Model for Healthcare Representation Learning](https://arxiv.org/abs/1611.07012)
* https://zhuanlan.zhihu.com/p/49258190
* https://www.experoinc.com/post/node-classification-by-graph-convolutional-network
* http://sungsoo.github.io/2018/02/01/geometric-deep-learning.html
* https://sites.google.com/site/deepgeometry/slides-1
* https://rusty1s.github.io/pytorch_geometric/build/html/notes/introduction.html
* [.mp4 illustration](http://tkipf.github.io/graph-convolutional-networks/images/video.mp4)
* [Deep Graph Library (DGL)](https://www.dgl.ai/)
* https://github.com/alibaba/euler
* https://github.com/alibaba/euler/wiki/%E8%AE%BA%E6%96%87%E5%88%97%E8%A1%A8
* https://www.experoinc.com/post/node-classification-by-graph-convolutional-network
* https://www.groundai.com/project/graph-convolutional-networks-for-text-classification/
* https://datawarrior.wordpress.com/2018/08/08/graph-convolutional-neural-network-part-i/
* https://datawarrior.wordpress.com/2018/08/12/graph-convolutional-neural-network-part-ii/
* http://www.cs.nuim.ie/~gunes/files/Baydin-MSR-Slides-20160201.pdf
* http://colah.github.io/posts/2015-09-NN-Types-FP/
* https://www.zhihu.com/question/305395488/answer/554847680
* https://www-cs.stanford.edu/people/jure/pubs/graphrepresentation-ieee17.pdf
* https://blog.acolyer.org/2019/02/06/a-comprehensive-survey-on-graph-neural-networks/

$\color{navy}{\text{Graph convolution network is potential to}}\, \mathcal{reasoning}$ as the blend of $\mathfrak{\text{probabilistic graph model}}$ and $\mit{\text{deep learning}}$.

GCN can be regarded as the counterpart of CNN for graphs so that the optimization techniques such as normalization, attention mechanism and even the adversarial version can be extended to the graph structure.

### ChebNet, CayleyNet, MotifNet

In the previous post, the convolution of the graph Laplacian is defined in its **graph Fourier space** as outlined in the paper of Bruna et. al. (arXiv:1312.6203). However, the **eigenmodes** of the graph Laplacian are not ideal because it makes the bases to be graph-dependent. A lot of works were done in order to solve this problem, with the help of various special functions to express the filter functions. Examples include Chebyshev polynomials and Cayley transform.

Defining filters as polynomials applied over the eigenvalues of the `graph Laplacian`, it is possible
indeed to avoid any eigen-decomposition and realize convolution by means of efficient sparse routines
The main idea behind CayleyNet is to achieve some sort of spectral zoom property by means of Cayley transform.

**CayleyNet**

Defining filters as polynomials applied over the eigenvalues of the `graph Laplacian`, it is possible
indeed to avoid any eigen-decomposition and realize convolution by means of efficient sparse routines
The main idea behind `CayleyNet` is to achieve some sort of spectral zoom property by means of Cayley transform:
$$
C(\lambda) = \frac{\lambda - i}{\lambda + i}
$$

Instead of Chebyshev polynomials, it approximates the filter as:
$$
g(\lambda) = c_0 + \sum_{j=1}^{r}[c_jC^{j}(h\lambda) + c_j^{\star} C^{j^{\star}}(h\lambda)]
$$
where $c_0$ is real and other $c_j$’s are generally complex, and ${h}$ is a zoom parameter, and $\lambda$’s are the eigenvalues of the graph Laplacian.
Tuning ${h}$ makes one find the best zoom that spread the top eigenvalues. ${c}$'s are computed by training. This solves the problem of unfavorable clusters in ChebNet.

**MotifNet**

`MotifNet` is aimed to address the direted graph convolution.

* https://datawarrior.wordpress.com/2018/08/12/graph-convolutional-neural-network-part-ii/
* https://github.com/thunlp/GNNPapers
* http://mirlab.org/conference_papers/International_Conference/ICASSP%202018/pdfs/0006852.pdf
* [graph convolution network有什么比较好的应用task？ - superbrother的回答 - 知乎](https://www.zhihu.com/question/305395488/answer/554847680)
* https://arxiv.org/abs/1704.06803
* https://github.com/alibaba/euler

### Graph Embedding

- https://zhuanlan.zhihu.com/p/47489505
- http://blog.lcyown.cn/2018/04/30/graphencoding/
- https://blog.csdn.net/NockinOnHeavensDoor/article/details/80661180
- http://building-babylon.net/2018/04/10/graph-embeddings-in-hyperbolic-space/


[Differential geometry based geometric data analysis (DG-GDA) of molecular datasets](https://weilab.math.msu.edu/DG-GL/)

## DeepRL

Reinforcement Learning (RL) has achieved many successes over the years in training autonomous agents to perform simple tasks. However, one of the major remaining challenges in RL is scaling it to high-dimensional, real-world applications.

Although many works have already focused on strategies to scale-up RL techniques and to find solutions for more complex problems with reasonable successes, many issues still exist. This workshop encourages to discuss diverse approaches to accelerate and generalize RL, such as the use of approximations, abstractions, hierarchical approaches, and Transfer Learning.

Scaling-up RL methods has major implications on the research and practice of complex learning problems and will eventually lead to successful implementations in real-world applications.

![DeepRL](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Reinforcement_learning_diagram.svg/375px-Reinforcement_learning_diagram.svg.png)

+ http://surl.tirl.info/
+ https://srdas.github.io/DLBook/ReinforcementLearning.html
+ https://katefvision.github.io/
+ https://spinningup.openai.com/en/latest/
+ http://rll.berkeley.edu/deeprlcoursesp17/
+ [A Free course in Deep Reinforcement Learning from beginner to expert.](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)
+ https://fullstackdeeplearning.com/march2019
https://sites.ualberta.ca/~szepesva/RLBook.html
+ https://nervanasystems.github.io/coach/
+ [Deep Reinforcement Learning (DRL)](http://primo.ai/index.php?title=Deep_Reinforcement_Learning_(DRL))
+ https://deeplearning4j.org/deepreinforcementlearning.html
+ https://openai.com/blog/spinning-up-in-deep-rl/
+ [Deep Reinforcement Learning NUS SoC, 2018/2019, Semester II](https://www.comp.nus.edu.sg/~kanmy/courses/6101_1820/)
+ [CS 285 at UC Berkeley: Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/)
+ https://theintelligenceofinformation.wordpress.com/

## Deep Learning Ensemble

Deep learning and ensemble learning share some similar guide line.

- [ ] [Neural Network Ensembles](https://www.computer.org/csdl/journal/tp/1990/10/i0993/13rRUyv53Gg)
- [ ] [A selective neural network ensemble classification for incomplete data](https://link.springer.com/article/10.1007/s13042-016-0524-0)
- [ ] [Deep Neural Network Ensembles](https://arxiv.org/abs/1904.05488)
- [ ] [Ensemble Learning Methods for Deep Learning Neural Networks](https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/)
- [ ] [Stochastic Weight Averaging — a New Way to Get State of the Art Results in Deep Learning](https://pechyonkin.me/stochastic-weight-averaging/)
- [ ] [Ensemble Deep Learning for Speech Recognition](https://www.isca-speech.org/archive/archive_papers/interspeech_2014/i14_1915.pdf)
- [ ] http://ruder.io/deep-learning-optimization-2017/
- [ ] https://arxiv.org/abs/1704.00109v1
- [ ] [Blending and deep learning](http://jtleek.com/advdatasci/17-blending.html)
- [ ] https://arxiv.org/abs/1708.03704
- [ ] [Better Deep Learning: Train Faster, Reduce Overfitting, and Make Better Predictions](https://machinelearningmastery.com/better-deep-learning/)
- https://machinelearningmastery.com/framework-for-better-deep-learning/
- https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/

### Selective Ensemble

[An ensemble is generated by training multiple component learners for a same task and then combining their predictions. In most ensemble algorithms, all the trained component learners are employed in constituting an ensemble. But recently, it has been shown that when the learners are neural networks, `it may be better to ensemble some instead of all of the learners`. In this paper, this claim is generalized to situations where the component learners are decision trees. Experiments show that ensembles generated by a selective ensemble algorithm, which selects some of the trained C4.5 decision trees to make up an ensemble, may be not only smaller in the size but also stronger in the generalization than ensembles generated by non-selective algorithms.](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/rsfdgrc03.pdf)

- [Selective Ensemble](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/rsfdgrc03.pdf)

### Snapshot

In contrast to traditional ensembles (produce an ensemble of multiple neural networks), the goal of this work is training a single neural network, converging to several local minima along its optimization path and saving the model parameters to obtain a ensembles model. It is clear that the number of possible local minima grows exponentially with the number of parameters and different local minima often have very similar error rates, the corresponding neural networks tend to make different mistakes.

![snapshot](http://ruder.io/content/images/2017/11/snapshot_ensembles.png)

`Snapshot Ensembling` generate an ensemble of accurate and diverse models from a single training with an optimization process which visits several local minima before converging to a final solution. In each local minima, they save the parameters as a model and then take model snapshots at these various minima, and average their predictions at test time.

- [Snapshot Ensembles: Train 1, get M for free](https://arxiv.org/abs/1704.00109)
- [Snapshot Ensembles in Torch](https://github.com/gaohuang/SnapshotEnsemble)
- [Snapshot Ensemble in Keras](https://github.com/titu1994/Snapshot-Ensembles)
- [Snapshot Ensembles: Train 1, get M for free Reviewed on Mar 8, 2018 by Faezeh Amjad](https://vitalab.github.io/article/2018/03/08/snapshot-distillation.html)

### Fast Geometric Ensembling (FGE)

- [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026)
- https://github.com/timgaripov/dnn-mode-connectivity
- https://izmailovpavel.github.io/files/curves/nips_poster.pdf
- https://bayesgroup.github.io/bmml_sem/2018/Garipov_Loss%20Surfaces.pdf

### Stochastic Weight Averaging (SWA)

- https://github.com/timgaripov/swa
- [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)
- https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
- https://izmailovpavel.github.io/

## Bayesian Deep Learning

[The abstract of Bayesian Deep learning](http://bayesiandeeplearning.org/) put that:

> While deep learning has been revolutionary for machine learning, most modern deep learning models cannot represent their uncertainty nor take advantage of the well studied tools of probability theory. This has started to change following recent developments of tools and techniques combining Bayesian approaches with deep learning. The intersection of the two fields has received great interest from the community over the past few years, with the introduction of new deep learning models that take advantage of Bayesian techniques, as well as Bayesian models that incorporate deep learning elements [1-11]. In fact, the use of Bayesian techniques in deep learning can be traced back to the 1990s’, in seminal works by Radford Neal [12], David MacKay [13], and Dayan et al. [14]. These gave us tools to reason about deep models’ confidence, and achieved state-of-the-art performance on many tasks. However earlier tools did not adapt when new needs arose (such as scalability to big data), and were consequently forgotten. Such ideas are now being revisited in light of new advances in the field, yielding many exciting new results
> Extending on last year’s workshop’s success, this workshop will again study the advantages and disadvantages of such ideas, and will be a platform to host the recent flourish of ideas using Bayesian approaches in deep learning and using deep learning tools in Bayesian modelling. The program includes a mix of invited talks, contributed talks, and contributed posters. It will be composed of five themes: deep generative models, variational inference using neural network recognition models, practical approximate inference techniques in Bayesian neural networks, applications of Bayesian neural networks, and information theory in deep learning. Future directions for the field will be debated in a panel discussion.
> This year’s main theme will focus on applications of Bayesian deep learning within machine learning and outside of it.

1. Kingma, DP and Welling, M, "Auto-encoding variational Bayes", 2013.
2. Rezende, D, Mohamed, S, and Wierstra, D, "Stochastic backpropagation and approximate inference in deep generative models", 2014.
3. Blundell, C, Cornebise, J, Kavukcuoglu, K, and Wierstra, D, "Weight uncertainty in neural network", 2015.
4. Hernandez-Lobato, JM and Adams, R, "Probabilistic backpropagation for scalable learning of Bayesian neural networks", 2015.
5. Gal, Y and Ghahramani, Z, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning", 2015.
6. Gal, Y and Ghahramani, G, "Bayesian convolutional neural networks with Bernoulli approximate variational inference", 2015.
7. Kingma, D, Salimans, T, and Welling, M. "Variational dropout and the local reparameterization trick", 2015.
8. Balan, AK, Rathod, V, Murphy, KP, and Welling, M, "Bayesian dark knowledge", 2015.
9. Louizos, C and Welling, M, “Structured and Efficient Variational Deep Learning with Matrix Gaussian Posteriors”, 2016.
10. Lawrence, ND and Quinonero-Candela, J, “Local distance preservation in the GP-LVM through back constraints”, 2006.
11. Tran, D, Ranganath, R, and Blei, DM, “Variational Gaussian Process”, 2015.
12. Neal, R, "Bayesian Learning for Neural Networks", 1996.
13. MacKay, D, "A practical Bayesian framework for backpropagation networks", 1992.
14. Dayan, P, Hinton, G, Neal, R, and Zemel, S, "The Helmholtz machine", 1995.
15. Wilson, AG, Hu, Z, Salakhutdinov, R, and Xing, EP, “Deep Kernel Learning”, 2016.
16. Saatchi, Y and Wilson, AG, “Bayesian GAN”, 2017.
17. MacKay, D.J.C. “Bayesian Methods for Adaptive Models”, PhD thesis, 1992.

***

* [Towards Bayesian Deep Learning: A Framework and Some Existing Methods](https://arxiv.org/abs/1608.06884)
* http://www.wanghao.in/mis.html
* https://github.com/junlulocky/bayesian-deep-learning-notes
* https://github.com/robi56/awesome-bayesian-deep-learning
* https://alexgkendall.com/computer_vision/phd_thesis/
* http://bayesiandeeplearning.org/
* http://www.cs.ox.ac.uk/people/yarin.gal/website/blog.html
* http://twiecki.github.io/blog/2016/06/01/bayesian-deep-learning/
* https://uvadlc.github.io/lectures/apr2019/lecture9-bayesiandeeplearning.pdf

## Theories of Deep Learning

[ICML 2017](https://www.padl.ws/) organized a workshop on **Principled Approaches to Deep Learning**:
> The recent advancements in deep learning have revolutionized the field of machine learning, enabling unparalleled performance and many new real-world applications. Yet, the developments that led to this success have often been driven by empirical studies, and little is known about the theory behind some of the most successful approaches. While theoretically well-founded deep learning architectures had been proposed in the past, they came at a price of increased complexity and reduced tractability. Recently, we have witnessed considerable interest in principled deep learning. This led to a better theoretical understanding of existing architectures as well as development of more mature deep models with solid theoretical foundations. In this workshop, we intend to review the state of those developments and provide a platform for the exchange of ideas between the theoreticians and the practitioners of the growing deep learning community. Through a series of invited talks by the experts in the field, contributed presentations, and an interactive panel discussion, the workshop will cover recent theoretical developments, provide an overview of promising and mature architectures, highlight their challenges and unique benefits, and present the most exciting recent results.

Topics of interest include, but are not limited to:

* Deep architectures with solid theoretical foundations
* Theoretical understanding of deep networks
* Theoretical approaches to representation learning
* Algorithmic and optimization challenges, alternatives to backpropagation
* Probabilistic, generative deep models
* Symmetry, transformations, and equivariance
* Practical implementations of principled deep learning approaches
* Domain-specific challenges of principled deep learning approaches
* Applications to real-world problems

There are more mathematical perspectives to deep learning: dynamical system, thermodynamics, Bayesian statistics, random matrix, numerical optimization, algebra and differential equation.

The information theory or code theory helps to accelerate the deep neural network inference as well as computer system design.

The limitation and extension of deep learning methods is also discussed such as F-principle, capsule-net, biological plausible methods.
The deep learning method is more engineer. The computational evolutionary adaptive  cognitive intelligence does not occur until now.

* [DALI 2018 - Data, Learning and Inference](http://dalimeeting.org/dali2018/workshopTheoryDL.html)
* https://www.msra.cn/zh-cn/news/people-stories/wei-chen
* https://www.microsoft.com/en-us/research/people/tyliu/
* [On Theory@http://www.deeplearningpatterns.com ](http://www.deeplearningpatterns.com/doku.php?id=theory)
* https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/85815724
* [UVA DEEP LEARNING COURSE](https://uvadlc.github.io/)
* [Understanding Neural Networks by embedding hidden representations](https://rakeshchada.github.io/Neural-Embedding-Animation.html)
* [Tractable Deep Learning](https://www.cs.washington.edu/research/tractable-deep-learning)
* [Theories of Deep Learning (STATS 385)](https://stats385.github.io/)
* [Topics Course on Deep Learning for Spring 2016 by Joan Bruna, UC Berkeley, Statistics Department](https://github.com/joanbruna/stat212b)
* [Mathematical aspects of Deep Learning](http://elmos.scripts.mit.edu/mathofdeeplearning/)
* [MATH 6380p. Advanced Topics in Deep Learning Fall 2018](https://deeplearning-math.github.io/)
* [CoMS E6998 003: Advanced Topics in Deep Learning](https://www.advancedtopicsindeeplearning.com/)
* [Deep Learning Theory: Approximation, Optimization, Generalization](http://www.mit.edu/~9.520/fall17/Classes/deep_learning_theory.html)
* [Theory of Deep Learning, ICML'2018](https://sites.google.com/site/deeplearningtheory/)
* [Deep Neural Networks: Approximation Theory and Compositionality](http://www.mit.edu/~9.520/fall16/Classes/deep_approx.html)
* [Theory of Deep Learning, project in researchgate](https://www.researchgate.net/project/Theory-of-Deep-Learning)
* [THE THEORY OF DEEP LEARNING - PART I](https://physicsml.github.io/blog/DL-theory.html)
* [Magic paper](http://cognitivemedium.com/magic_paper/index.html)
* [Principled Approaches to Deep Learning](https://www.padl.ws/)
* [The Science of Deep Learning](http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_5058.html)
* [The thermodynamics of learning](https://phys.org/news/2017-02-thermodynamics.html)
* [A Convergence Theory for Deep Learning via Over-Parameterization](https://arxiv.org/pdf/1811.03962.pdf)
* [MATHEMATICS OF DEEP LEARNING, NYU, Spring 2018](https://github.com/joanbruna/MathsDL-spring18)
* [Advancing AI through cognitive science](https://github.com/brendenlake/AAI-site)
* [DALI 2018, Data Learning and Inference](http://dalimeeting.org/dali2018/workshopTheoryDL.html)
* [Deep unrolling](https://zhuanlan.zhihu.com/p/44003318)
* [WHY DOES DEEP LEARNING WORK?](https://calculatedcontent.com/2015/03/25/why-does-deep-learning-work/)
* [Deep Learning and the Demand for Interpretability](http://stillbreeze.github.io/Deep-Learning-and-the-Demand-For-Interpretability/)
* https://beenkim.github.io/
* [Integrated and detailed image understanding](https://www.robots.ox.ac.uk/~vedaldi//research/idiu/idiu.html)
* [Layer-wise Relevance Propagation (LRP)](http://www.heatmapping.org/)
* [ICCV 2019 Tutorial on Interpretable Machine Learning for Computer Vision](http://networkinterpretability.org/)
* [6.883 Science of Deep Learning: Bridging Theory and Practice -- Spring 2018](https://people.csail.mit.edu/madry/6.883/)
* https://interpretablevision.github.io/
****
* [Open Source Deep Learning Curriculum, 2016](https://www.deeplearningweekly.com/blog/open-source-deep-learning-curriculum/)
* [Short Course of Deep Learning 2016 Autumn, PKU](http://www.xn--vjq503akpco3w.top/)
* [Website for UVA Qdata Group's Deep Learning Reading Group](https://qdata.github.io/deep2Read/)
* [Foundation of deep learning](https://github.com/soumyadsanyal/foundations_for_deep_learning)
* [深度学习名校课程大全 - 史博的文章 - 知乎](https://zhuanlan.zhihu.com/p/31988246)
* [Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)
* [CS 598 LAZ: Cutting-Edge Trends in Deep Learning and Recognition](http://slazebni.cs.illinois.edu/spring17/)
* [Hugo Larochelle’s class on Neural Networks](https://sites.google.com/site/deeplearningsummerschool2016/)
* [Deep Learning Group @microsoft](https://www.microsoft.com/en-us/research/group/deep-learning-group/)
* [Silicon Valley Deep Learning Group](http://www.svdlg.com/)
* http://blog.qure.ai/notes/visualizing_deep_learning
* http://blog.qure.ai/notes/deep-learning-visualization-gradient-based-methods
* https://zhuanlan.zhihu.com/p/45695998
* https://www.zhihu.com/question/265917569
* https://www.ias.edu/ideas/2017/manning-deep-learning
* https://www.jiqizhixin.com/articles/2018-08-03-10
* https://cloud.tencent.com/developer/article/1345239
* http://cbmm.mit.edu/publications
* https://stanford.edu/~shervine/l/zh/teaching/cs-229/cheatsheet-deep-learning
* https://stanford.edu/~shervine/teaching/cs-230.html
* https://cordis.europa.eu/project/rcn/214602/factsheet/en
* http://clgiles.ist.psu.edu/IST597/index.html
* https://zhuanlan.zhihu.com/p/44003318
* https://deepai.org/
* https://deepnotes.io/deep-clustering
* http://www.phontron.com/class/nn4nlp2019/schedule.html
* https://deeplearning-cmu-10707.github.io/

***

* [A guide to deep learning](https://yerevann.com/a-guide-to-deep-learning/)
* [500 Q&A on Deep Learning](https://github.com/scutan90/DeepLearning-500-questions)
* [Deep learning Courses](https://handong1587.github.io/deep_learning/2015/10/09/dl-courses.html)
* [Deep Learning note](https://github.com/brylevkirill/notes/blob/master/Deep%20Learning.md)
* [Deep learning from the bottom up](https://metacademy.org/roadmaps/rgrosse/deep_learning)
* [Deep Learning Papers Reading Roadmap](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap)
* Some websites on deep learning:
    + [http://colah.github.io/]
    + [https://distill.pub/]
    + [http://www.wildml.com/]
    + [https://www.fast.ai/]
* [Deep learning 101](https://markus.com/deep-learning-101/)
* [Design pattern of deep learning](http://www.deeplearningpatterns.com/doku.php?id=overview)
* [A Quick Introduction to Neural Networks](https://ujjwalkarn.me/2016/08/09/quick-intro-neural-networks/)
* [A Primer on Deep Learning](https://blog.datarobot.com/a-primer-on-deep-learning)
* [Deep learning and neural network](http://ucanalytics.com/blogs/deep-learning-and-neural-networks-simplified-part-1/)
* [An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)
* [Recurrent Neural Network](https://developer.nvidia.com/discover/recurrent-neural-network)
* [THE ULTIMATE GUIDE TO RECURRENT NEURAL NETWORKS (RNN)](https://www.superdatascience.com/the-ultimate-guide-to-recurrent-neural-networks-rnn/)
* [LSTM at skymind.ai](https://skymind.ai/wiki/lstm)
* [Deep Learning Resources](https://sebastianraschka.com/deep-learning-resources.html)
* [Deep Learning meeting](http://www.cs.tau.ac.il/~wolf/deeplearningmeeting/home.html#)
* [Online courses](https://deeplearningcourses.com/)
* [Some personal experience in deep learning](http://people.idsia.ch/~juergen/firstdeeplearner.html)
* [**11-485/785** Introduction to Deep Learning](http://deeplearning.cs.cmu.edu/)
* [**Deep Learning: Do-It-Yourself!**](https://www.di.ens.fr/~lelarge/dldiy/)
* [**Deep Learning course: lecture slides and lab notebooks**](https://m2dsupsdlclass.github.io/lectures-labs/)
* [EE-559 – DEEP LEARNING (SPRING 2018)](https://fleuret.org/ee559/)
* [NSF Center for Big Learning Creating INtelligence](http://nsfcbl.org/)
* [National Academy of Sicence Colloquia: The Science of Deep Learning](http://www.nasonline.org/programs/nas-colloquia/completed_colloquia/science-of-deep-learning.html)
* [MRI: Development of an Instrument for Deep Learning Research](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1725729&HistoricalAwards=false)
* [National Center for Supercomputing Application](http://www.ncsa.illinois.edu/)
* [Deep Learning and Representation Learning](https://www.microsoft.com/en-us/research/project/deep-learning-and-representation-learning/)
* https://nsfcbl.cs.uoregon.edu/
* http://www.andyli.ece.ufl.edu/
* https://deeplearninganalytics.org/
* https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-deep-learning
* https://www.deeplearningwizard.com/
* https://www.stateoftheart.ai/
* [神经网络与深度学习](https://nndl.github.io/)
* https://mchromiak.github.io/articles/2017/Sep/01/Primer-NN/#.XBXb42h3hPY
* https://www.european-big-data-value-forum.eu/program/explainable-artificial-intelligence/
* http://staff.ustc.edu.cn/~lgliu/Resources/DL/What_is_DeepLearning.html

## Application

IN fact, deep learning is applied widely in prediction and classification in diverse domain from image recognition to drug discovery even scientific computation though there is no first principle to guide how to apply deep learning to these domain.
The history of deep learning begins with its performance in image recognition and spoken language recognition over human being. Current progress is in natural language processing with BERT, XLNET and more.

Even deep learning is young and cut-edge, some pioneers contribute to its development.

* [The Deep Learning for Science Workshop](https://dlonsc.github.io/)
* [The Functions of Deep Learning](https://sinews.siam.org/Details-Page/the-functions-of-deep-learning)
* [Internet-Scale Deep Learning for Bing Image Search](https://blogs.bing.com/search-quality-insights/May-2018/Internet-Scale-Deep-Learning-for-Bing-Image-Search)
* [What can deep learning do for you?](https://machinethink.net/)
* https://www.microsoft.com/en-us/research/group/deep-learning-group/
* https://www.interspeech2019.org/program/schedule/
* https://courses.engr.illinois.edu/ie534/fa2019/
* https://jasirign.github.io/
* https://xbpeng.github.io/
* https://delug.github.io/
* https://github.com/Honlan/DeepInterests
* [Deep Learning for Physical Sciences](https://dl4physicalsciences.github.io/)

### Computer Vision

- [Deep learning and visualization infrastructure (EVL)](https://cs.uic.edu/news-stories/997k-nsf-grant-deep-learning-and-visualization-infrastructure-evl/)
- http://llcao.net/cu-deeplearning17/
- https://missinglink.ai/blog/
- [deep learning for computer vision](https://cvhci.anthropomatik.kit.edu/600_1682.php)
- [deep learning for vision](https://sif-dlv.github.io/)
- [SOD - An Embedded Computer Vision & Machine Learning Library](https://sod.pixlab.io/)
- [VisualData: Discover Computer Vision Datasets](https://www.visualdata.io)
- https://xueyangfu.github.io/
- https://njuhaozhang.github.io/
- https://matbc.github.io/

### Spoken Language Processing

- https://www.interspeech2019.org/program/schedule/
- https://irasl.gitlab.io/
- [CS224S / LINGUIST285 - Spoken Language Processing](http://web.stanford.edu/class/cs224s/)
- https://pairlabs.ai/


### Natural Language Processing

- https://www.comp.nus.edu.sg/~kanmy/courses/6101_1810/
- https://web.stanford.edu/class/cs224n/
- https://deep-spin.github.io/tutorial/
- [flavor with NLP](https://chokkan.github.io/deeplearning/)
- https://yscacaca.github.io/


### Finance

- [Deep Learning In Finance](https://rohitghosh.github.io/deep_learning_in_finance/)
- https://iknowfirst.com/
- https://algorithmxlab.com/
- https://github.com/firmai/financial-machine-learning
- https://www.firmai.org

### Brain and Cognition Science

- [Exploring Computational Creativity with Neural Networks](https://blog.floydhub.com/humans-of-ml-kalai-ramea/)
- [Deep Learning and Brain](https://elsc.huji.ac.il/events/elsc-conference-10)
- [Deep Learning for Cognitive Computing, Theory (Course code: TIES4910) 5 ECTS, Autumn Semester](http://www.cs.jyu.fi/ai/vagan/DL4CC.html)
- [Theoretical Neuroscience and Deep Learning Theory](http://videolectures.net/deeplearning2017_ganguli_deep_learning_theory/)
- [Bridging Neuroscience and Deep Machine Learning, by building theories that work in the Real World.](https://ankitlab.co/)
- https://ankitlab.co/talks/
- https://lynnsunxmu.github.io/

<img src="http://www.cs.jyu.fi/ai/vagan/DL4CC_files/image003.gif" width="50%" />

* [Program 2019 - Deep Learning for Human Brain Mapping](https://brainhack101.github.io/IntroDL/)
* [Courses of Vagan Terziyan](http://www.cs.jyu.fi/ai/vagan/courses.html)
* [Neural Networks, Types, and Functional Programming](http://colah.github.io/posts/2015-09-NN-Types-FP/)
* [AI and Neuroscience: A virtuous circle](https://deepmind.com/blog/ai-and-neuroscience-virtuous-circle/)
* [Neuroscience-Inspired Artificial Intelligence](http://www.columbia.edu/cu/appliedneuroshp/Papers/out.pdf)
* [深度神经网络（DNN）是否模拟了人类大脑皮层结构？ - Harold Yue的回答 - 知乎](https://www.zhihu.com/question/59800121/answer/184888043)
* Connectionist models of cognition <https://stanford.edu/~jlmcc/papers/ThomasMcCIPCambEncy.pdf>
* http://fourier.eng.hmc.edu/e161/lectures/nn/node3.html
* [PSYCH 209: Neural Network Models of Cognition: Principles and Applications](https://web.stanford.edu/class/psych209/)
* [Deep Learning: Branching into brains](https://elifesciences.org/articles/33066)
* [BRAIN INSPIRED](https://braininspired.co/about/)
* [http://www.timkietzmann.de](http://www.timkietzmann.de/)
* http://www.brain-ai.jp/organization/

## The Future

The ultimate goal is general artificial intelligence.

* http://www.iro.umontreal.ca/~bengioy/papers/ftml_book.pdf
* https://cbmm.mit.edu/publications
* https://www.ctolib.com/pauli-space-foundations_for_deep_learning.html
* https://blog.keras.io/the-future-of-deep-learning.html
* https://github.com/sekwiatkowski/awesome-capsule-networks
* http://www.thetalkingmachines.com/article/neural-programmer-interpreters
* https://barghouthi.github.io/2018/05/01/differentiable-programming/
* https://darioizzo.github.io/d-CGP/
* https://aifuture2016.stanford.edu/
* [Harnessing the Data Revolution (HDR): Institutes for Data-Intensive Research in Science and Engineering - Ideas Labs  (I-DIRSE-IL)](https://www.nsf.gov/funding/pgm_summ.jsp?pims_id=505614)
* https://wsdm2019-dapa.github.io/
* https://leon.bottou.org/slides/mlss13/mlss-nn.pdf
* [Physics Informed Deep Learning](http://www.dam.brown.edu/people/mraissi/research/1_physics_informed_neural_networks/)
* http://www.dam.brown.edu/people/mraissi/teaching/1_deep_learning_tutorial/
* https://ieeexplore.ieee.org/abstract/document/1035030
* https://arxiv.org/abs/1805.10451
* https://deepai.org/machine-learning/researcher/na-lei
* [Differential Geometry in Computer Vision and Machine Learning ](https://diffcvml2018.wordpress.com/)
* http://primo.ai/index.php?title=PRIMO.ai
* https://vistalab-technion.github.io/cs236605/
* http://www.cs.jyu.fi/ai/vagan/DL4CC.html
* [The Deep Learning on Supercomputers Workshop](https://dlonsc19.github.io/)
* https://deep-learning-security.github.io/
* https://www.surrey.ac.uk/events/20190723-ai-summer-school
* https://smartech.gatech.edu/handle/1853/56665
* http://alchemy.cs.washington.edu/
