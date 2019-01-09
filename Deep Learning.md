# Deep Learning

Deep learning is the modern version of artificial neural networks full of tricks and techniques.
In mathematics, it is nonlinear non-convex and composite of many functions. Its name -deep learning- is to distinguish from the classical machine learning "shallow" methods.
However, its complexity makes it yet engineering even art far from science. There is no first principle in deep learning but trial and error.
In theory, we do not clearly understand how to design more robust and efficient network architecture; in practice, we can apply it to diverse fields. It is considered as one approach to artificial intelligence

Deep learning is a typical hierarchy model.
The application of deep learning are partial listed in **[Awesome deep learning](https://github.com/ChristosChristofidis/awesome-deep-learning)**,[MIT Deep Learning Book](https://github.com/janishar/mit-deep-learning-book-pdf) and [Deep interests](https://github.com/Honlan/DeepInterests).

***
|[Father of Deep Learning](https://www.wikiwand.com/en/Alexey_Ivakhnenko)|
|:----------------------------------------------------------------------:|
|![Father of Deep Learning](https://tse2.mm.bing.net/th?id=OIP.RPMZM_oYzqfEvUISXL6aCQAAAA&pid=Api)|
|[A history of deep learning](https://www.import.io/post/history-of-deep-learning/)|
|[Three Giants' Survey in *(Nature 521 p 436)*](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)|
|[Critique of Paper by "Deep Learning Conspiracy" (Nature 521 p 436)](http://people.idsia.ch/~juergen/deep-learning-conspiracy.html) |
|http://principlesofdeeplearning.com/|
|[**Deep Learning in Neural Networks: An Overview**](http://people.idsia.ch/~juergen/deep-learning-overview.html)|
|[AI winter](https://www.wikiwand.com/en/AI_winter)|
|[Deep learning in wiki](https://www.wikiwand.com/en/Deep_learning) and [Deep Learning in Scholarpedia](http://www.scholarpedia.org/article/Deep_Learning)|
|[A Brief History of Deep Learning (Part One)](https://www.bulletproof.net.au/a-brief-history-deep-learning-part-one/)|
|[On the Origin of Deep Learning](https://arxiv.org/abs/1702.07800)|
|![history of nn](https://www.import.io/wp-content/uploads/2017/06/Import.io_quote-image5-170525.jpg)|
|![Deep Learning Roadmap](http://www.deeplearningpatterns.com/lib/exe/fetch.php?media=deeplearning_overview_9_.jpg)|
|![nn_timeline](http://beamandrew.github.io//images/deep_learning_101/nn_timeline.jpg)|
![Neural_Net_Arch_Genealogy](https://raw.githubusercontent.com/hunkim/deep_architecture_genealogy/master/Neural_Net_Arch_Genealogy.png)
https://mitpress.mit.edu/books/deep-learning-revolution

***
The **architecture** and **optimization** are the core content of deep learning models. We will focus on the first one.

## Artificial Neural Network

Artificial neural networks are most easily visualized in terms of a **directed graph**. In the case of sigmoidal units, node $s$ represents sigmoidal unit  and directed edge $e=(u,v)$ indicates that one of sigmoidal unit $v$'s inputs is the output of sigmoidal unit $u$.

![The Neural Network Zoo](http://www.asimovinstitute.org/wp-content/uploads/2016/09/neuralnetworks.png)
***
* [The Wikipedia page on ANN](https://www.wikiwand.com/en/Artificial_neural_network)
* [History of deep learning](http://beamandrew.github.io/deeplearning/2017/02/23/deep_learning_101_part1.html)
* https://brilliant.org/wiki/artificial-neural-network/
* https://www.asimovinstitute.org/neural-network-zoo/

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

#### Activation function

The nonlinear function $\sigma$ is conventionally called activation function.
There are some activation functions in history.

* Sign function
   $$
   f(x)=\begin{cases}1,&\text{if $x > 0$}\\
                    -1,&\text{if $x < 0$}\end{cases}
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

#### Learning  algorithm

We draw it from the [Wikipedia page](https://en.wikipedia.org/w/index.php?title=Perceptron&action=edit&section=3).
Learning is to find optimal parameters of the model. Before that, we must feed data into the model.
The training data of $n$ sample size is
$$
D=\{(\mathbf{x}_i,d_i)\}_{i=1}^{n}
$$
where

* $\mathbf{x}_i$ is the $n$-dimensional input vector;
* $d_i$ is the desired output value of the perceptron for that input $\mathbf{x}_i$.

***

1. Initialize the weights and the threshold. Weights may be initialized to 0 or to a small random value. In the example below, we use 0.
2. For each example $j$ in our training set $D$, perform the following steps over the input $\mathbf{x}_{j}$ and desired output $d_{j}$:  
   * Calculate the actual output：
      $$
      \begin{align}
         y_{j}(t) & =f[\mathbf{w}(t)\cdot\mathbf{x}_{j}]\\
       ​           & =f[w_{0}(t)x_{j,0}+w_{1}(t)x_{j,1}+w_{2}(t)x_{j,2}+\dotsb +w_{n}(t)x_{j,n}]
      \end{align}
      $$

   * Update the weights:
       $w_{i}(t+1)=w_{i}(t)+r\cdot (d_{j}-y_{j}(t))x_{(j,i)},$
       for all features $[0\leq i\leq n]$, is the learning rate.
3. For [offline learning](https://www.wikiwand.com/en/Offline_learning), the second step may be repeated until the iteration error $\frac{1}{s}\sum_{j=1}^{s}|d_{j}-y_{j}(t)|$ is less than a user-specified error threshold $\gamma$, or a predetermined number of iterations have been completed, where _s_ is again the size of the sample set.

$\color{lime}{Note}$: the perceptron model is linear classifier, i.e. the training data set $D$ is linearly separable such that the learning algorithm can converge.
***
|||
|:-------------:|:-----------------:|
|![](https://www.i-programmer.info/images/stories/Core/AI/DeepLearning/neuron.jpg)|![](https://s.hswstatic.com/gif/brain-neuron-types-a.gif)|
|[Perceptrons](https://www.wikiwand.com/en/Perceptrons_(book))|[人工神经网络真的像神经元一样工作吗？](https://www.jqr.com/article/000595)|
|![](https://weltbild.scene7.com/asset/vgw/perceptrons-195682049.jpg)|![](https://tse4.mm.bing.net/th?id=OIP.96P534YMnRYWdiFQIv7WrgAAAA&pid=Api&w=300&h=450&rs=1&p=0)|

More in [Wikipedia page](https://www.wikiwand.com/en/Perceptron).

It is the first time to model cognition.

* [Neural Networks, Types, and Functional Programming](http://colah.github.io/posts/2015-09-NN-Types-FP/)
* [AI and Neuroscience: A virtuous circle](https://deepmind.com/blog/ai-and-neuroscience-virtuous-circle/)
* [Neuroscience-Inspired Artificial Intelligence](http://www.columbia.edu/cu/appliedneuroshp/Papers/out.pdf)
* [深度神经网络（DNN）是否模拟了人类大脑皮层结构？ - Harold Yue的回答 - 知乎](https://www.zhihu.com/question/59800121/answer/184888043)
* Connectionist models of cognition <https://stanford.edu/~jlmcc/papers/ThomasMcCIPCambEncy.pdf>
* https://stats385.github.io/blogs

### Feed-forward Neural Network

#### Representation of Feedforward Neural Network

Given that the function of a single neuron is rather simple, it subdivides the input space into two regions by a hyperplane, the complexity must come from having more layers of neurons involved in a complex action (like recognizing your grandmother in all possible situations).The "squashing" functions introduce critical nonlinearities in the system, without their presence multiple layers would still create linear functions.
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

In theory, the universal approximation theorem show the power of feedforward neural network
if we take some proper activation functions such as sigmoid function.

* https://www.wikiwand.com/en/Universal_approximation_theorem
* http://mcneela.github.io/machine_learning/2017/03/21/Universal-Approximation-Theorem.html
* http://neuralnetworksanddeeplearning.com/chap4.html

#### Evaluation and Optimization in Multilayer Perceptron

The problem is how to find the optimal parameters $W_1, b_1, W_2, b_2,\cdots, W, b$ ?
The multilayer perceptron is as one example of supervised learning, which means that we feed datum
$D=\{(\mathbf{x_i},d_i)\}_{i=1}^{n}$ to it and evaluate it.

The general form of the evaluation is given by:
$$
J(\theta)=\frac{1}{n}\sum_{i=1}^{n}\mathbb{L}[f(\mathbf{x}_i|\theta),\mathbf{d}_i]
$$
where $\mathbf{d}_i$ is the desired value of the input $\mathbf{x}_i$ and $\theta$ is the parameters of multilayer perceptron. The notation $f(\mathbf{x}_i|\theta)$ is the output given paramaters $\theta$. The function $\mathbb{L}$ is **loss function** to measure the discrpency between the predicted value $f(\mathbf{x}_i|\theta)$ and the desired value $\mathbf{d}_i$.

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
\mathbf{H}(d,p)=-\sum_{i=1}^{n}d_i\log(p_i)=\sum_{i=1}^{n}d_i\log(\frac{1}{p_i}),$$
where $d_i$ is the $i$th element of the one-hot vector $d$ and $p_i=\frac{\exp(z_i)}{\sum_{j=1}^{n}\exp(z_j)}$ for all $i=1,2\dots, n.$

Suppose $\mathrm{d}=(1,0,0,\cdots,0)$, the cross entropy is $\mathbf{H}(d,p)=-\log(p_1)=\log \sum_{i=1}^{n}\exp(z_i)-z_1$. The cost function is $\frac{1}{n}\sum_{i=1}^{n}\mathbf{H}(d^{i},p^{i})$ in the training data set $\{(\mathbf{x}_i,d^i)\}_{i=1}^{n}$ where $\mathbf{x}_i$ is the features of $i$th sample and $d^i$ is the desired true target label encoded in **one-hot** vector meanwhile $p^{i}$ is the predicted label of $\mathbf{x}_i$.
See the following links for more information on cross entropy and softmax.

|VISUALIZING THE LOSS LANDSCAPE OF NEURAL NETS||
|:-------------------------------------------:|---|
|![VGG](https://raw.githubusercontent.com/tomgoldstein/loss-landscape/master/doc/images/resnet56_noshort_small.jpg)|![ResNet](https://raw.githubusercontent.com/tomgoldstein/loss-landscape/master/doc/images/resnet56_small.jpg)|

* <https://blog.csdn.net/u014380165/article/details/77284921>;
* <https://blog.csdn.net/u014380165/article/details/79632950>;
* <https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/>;
* <https://www.zhihu.com/question/65288314>.

In regression, the loss function may simply be the squared $\ell_2$ norm, i.e. $\mathbb{L}(d,p)=(d-p)^{2}$ where $d$ is the desired target and $p$ is the predicted result. And the cost function is *mean squared error*:
$$ J(\theta)=\frac{1}{n}\sum_{i=1}^{n}[f(\mathbf{x}_i|\theta)-\mathrm{d}_i]^2.$$
In **robust statistics**, there are more loss functions such as *Huber loss*, *hinge loss*, *Tukey loss*.
***

* [Huber loss function](https://www.wikiwand.com/en/Huber_loss)
   $$
      Huber_{\delta}(x)=\begin{cases}
                      \frac{|x|}{2},&\text{if $|x|\leq\delta$}\\
                      \delta(|x|-\frac{1}{2}\delta),&\text{otherwise}
                      \end{cases}
   $$

* [Hinge loss function](https://www.wikiwand.com/en/Hinge_loss)
  $$
      Hinge(x)=max\{0, 1-tx\}
  $$
  where $t=+1$ or $t=-1$.

* Tukey loss function
   $$
    Tukey_{\delta}(x)=\begin{cases}
         (1-[1-x^2/\delta^2]^3)\frac{\delta^2}{6},&\text{if $|x|\leq\delta$}\\
         \frac{\delta^2}{6},                      &\text{otherwise}
                      \end{cases}
   $$
***

It is important to choose or design loss function or more generally objective function,
which can select variable as LASSO or confirm prior information as Bayesian estimation.
Except the *representation* or model, it is the objective function that affects the usefulness of learning algorithms.

For more on **loss function** see:

* <https://blog.algorithmia.com/introduction-to-loss-functions/>;
* <https://www.learnopencv.com/understanding-activation-functions-in-deep-learning/>;
* <http://laid.delanover.com/activation-functions-in-deep-learning-sigmoid-relu-lrelu-prelu-rrelu-elu-softmax/>;
* https://www.wikiwand.com/en/Robust_statistics
* https://www.wikiwand.com/en/Huber_loss
* <https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions>;
* <https://www.cs.umd.edu/~tomg/projects/landscapes/>.

### Backpropagation, Training and Regularization

#### Backpropagation

Automatic differentiation is the generic name for techniques that use the computational representation of a function to produce **analytic** values for the derivatives.
Automatic differentiation techniques are founded on the observation that any function, no matter how complicated, is evaluated by performing a sequence of simple elementary operations involving just one or two arguments at a time.
Backpropagation is one special case of automatic differentiation, i.e. *reverse-mode automatic differentiation*.

The backpropagation procedure to compute the gradient of an objective function with respect to the weights of a multilayer stack of modules is nothing more than a practical application of the **chain rule for derivatives**.
The key insight is that the derivative (or gradient) of the objective with respect to the input of a module can be computed by working backwards from the gradient with respect to the output of that module (or the input of the subsequent module).
The backpropagation equation can be applied repeatedly to
propagate gradients through all modules, starting from the output at the top (where the network produces its prediction) all the way to the bottom (where the external input is fed).
Once these gradients have been computed, it is straightforward to compute the gradients with respect to the weights of each module.[^10]
***
Suppose that $f(x)={\sigma}\circ(WH + b)$,where $H=\sigma\circ(W_4H_3 + b_4)$, $H_3=\sigma\circ(W_3H_2 + b_3)$,$H_2=\sigma\circ(W_2H_1 + b_2),$ $H_1=\sigma\circ(W_1x + b_1)$,
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
\frac{\partial \sigma(W^jH+b^j)}{\partial H}  =
{\sigma}^{\prime}(W^jH+b^j) W^j \,\,\forall j\in\{1,2,\dots,l\},
$$
where $f^{j}(x_0)$, $W^{j}$, $b^j$ and $\sigma^{\prime}(z)$is the $j$th element of $f(x_0)$, the $j$-th row of matrix $W$, the $j$th element of vector $b$ and $\frac{\mathrm{d}\sigma(z)}{\mathrm{d} z}$, respectively.

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
$$\frac{\partial H^j}{\partial W_4^j}  =\frac{\partial \sigma(W_4^j H_3+b_4^j)}{\partial W_4^j}  =\sigma^{\prime}(W_4^j H_3+b_4^j)H_3    \qquad\forall j\in\{1,2,\dots,l\};$$
$$\frac{\partial H^j}{\partial H_3}    =\frac{\partial \sigma(W_4^j H_3+b_4^j)}{\partial H_3}  =\sigma^{\prime}(W_4^j H_3+b_4^j)W_4^j  \qquad\forall j\in\{1,2,\dots,l_4\};$$
$$\frac{\partial H_3^j}{\partial W_3^j}=\frac{\partial \sigma(W_3^j H_2+b_3^j)}{\partial W_3^j}  =\sigma^{\prime}(W_3^j H_2+b_3^j)H_2    \qquad\forall j\in\{1,2,\dots,l_3\};$$
$$\frac{\partial H_3^j}{\partial H_2}  =\frac{\partial \sigma(W_3^j H_2+b_3^j)}{\partial H_2}  =\sigma^{\prime}(W_3^j H_2+b_3^j)W_3^j  \qquad\forall j\in\{1,2,\dots,l_3\};$$
$$\frac{\partial H_2^j}{\partial W_2^j}=\frac{\partial \sigma(W_2^j H_1+b_2^j)}{\partial W_2^j}  =\sigma^{\prime}(W_2^j H_1+b_2^j)H_1    \qquad\forall j\in\{1,2,\dots,l_2\};$$
$$\frac{\partial H_2^j}{\partial H_1}  =\frac{\partial \sigma(W_2^j H_1+b_2^j)}{\partial H_1}    =\sigma^{\prime}(W_2^j H_1+b_2^j)W_2^j  \qquad\forall j\in\{1,2,\dots,l_2\};$$
$$\frac{\partial H_1^j}{\partial W_1^j}=\frac{\partial \sigma(W_1^j x_0+b_1^j)}{\partial W_1^j}  =\sigma^{\prime}(W_1^j x_0+b_1^j)x_0    \qquad\forall j\in\{1,2,\dots,l_1\}.$$


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
The first step is to compute the gradient of loss function with respect to the output $f(x_0)=y\in\mathbb{R}^{o}$, i.e.
$\frac{\partial L(x_0, d_0)}{\partial f(x_0)}=2(f(x_0)-d_0)=2(\sigma\circ(WH+b)-d_0)$
, of which the $i$th element is $2(y^{i}-d_0^i)=2(\sigma(W^{i}H+b^{i})-d_0^{i})\,\forall i\{1,2,\dots,o\}$.
Thus
$$\frac{\partial L(x_0, d_0)}{\partial W^{i}}=\frac{\partial L(x_0, d_0)}{\partial y^{i}}\frac{\partial y^{i}}{\partial W^{i}}=2(y^{i}-d_0^i)\sigma^{\prime}(W^iH+b^i)H.$$
Thus we can compute all the gradients of $W$ columns. Note that $H$ has been computed through forwards propagation in that layer.

And $H=\sigma\circ(W_4H_3+b_3)$, of which the $i$th element is $H^{i}=\sigma(W_4 H_3 +b_4)^{i}=\sigma(W_4^{i} H_3+b_4^{i})$.

And we can compute the gradient of columns of $W_4$:
$$
\frac{\partial L(x_0,y_0)}{\partial W_4^i}= \sum_{j=1}^{o}
\frac{\partial L(x_0,y_0)}{\partial f^j (x_0)}
\frac{\partial f^j (x_0)}{\partial z}
\frac{\partial z}{\partial W_4^i}
=\sum_{j=1}^{o}\frac{\partial L(x_0,y_0)}{\partial y^j}
\frac{\partial y^j}{\partial H^i}
\frac{\partial H^i}{\partial W_4^i} \\
= \color{aqua}{
\sum_{j=1}^{o} \frac{\partial L}{\partial y^j}
\frac{\partial\, y^j}{\partial (W^jH+b^j)}
\frac{\partial (W^jH+b^j)}{\partial H^{i}}
\frac{\partial (H^{i})}{\partial W_4^i} } \\
= \sum_{j=1}^{l}\frac{\partial L}{\partial y^j}\,\sigma^{\prime}(W^j H+b^j)\,W^{j,i}\,\sigma^{\prime}(W^i_4 H_3+b^i_4)H_3,
$$
where $W^{j,i}$ is the $i$th element of $j$th column in matrix $W$.

$$
\frac{\partial L(x_0,y_0)}{\partial W_3^i}=\sum_{j=1}^{o}
\frac{\partial L(x_0,y_0)}{\partial f^j (x_0)}
[\frac{\partial f^j (x_0)}{\partial z}]
\frac{\partial z}{\partial W_3^i}
=\sum_{j=1}^{o}
\frac{\partial L}{\partial y^j }
[\frac{\partial y^j}{\partial H_3^i}]
\frac{\partial H_3^i}{\partial W_3^i}\\
=\sum_{j=1}^{o}
\frac{\partial L}{\partial y^j }
[\sum_{k=1}\frac{\partial y^j}{\partial H^k} \frac{\partial H^k}{\partial H_3^i}]
\frac{\partial H_3^i}{\partial W_3^i}
$$
where all the partial derivatives or gradients have been computed or accessible. It is nothing except to add or multiply these values in the order when we compute the weights of hidden layer.


$$
\frac{\partial L(x_0,y_0)}{\partial W_2^i}=\sum_{j=1}^{o}
\frac{\partial L(x_0,y_0)}{\partial f^j (x_0)}
[\frac{\partial f^j (x_0)}{\partial z}]
\frac{\partial z}{\partial W_2^i}
=\sum_{j=1}^{l}
\frac{\partial L}{\partial y^j }
[\frac{\partial y^j}{\partial H_2^i}]
\frac{\partial H_2^i}{\partial W_2^i}\\
=\sum_{j=1}^{o}
\frac{\partial L}{\partial y^j }
\{\sum_{k=1}\frac{\partial y^j}{\partial H^k}
[\sum_{m}\frac{\partial H^k}{\partial H_3^m}\frac{\partial H_3^m}{\partial H_2^i}]\}
\frac{\partial H_2^i}{\partial W_2^i}
$$

And the gradient of the first layer is computed by
$$
\frac{\partial L(x_0,y_0)}{\partial W_1^i}
=\sum_{j}\frac{\partial L(x_0,y_0)}{\partial y^j}\frac{\partial y^j}{\partial z}\frac{\partial z}{\partial W_1^i}                              \\
=\sum_{j}\frac{\partial L}{\partial y^j}
[\sum_{k}\frac{\partial y^j}{\partial H^k}
\sum_{m}\frac{\partial H^k}{\partial H_3^m}
\sum_{n}\frac{\partial H_3^k}{\partial H_2^n}
\sum_{r}\frac{\partial H_2^n}{\partial H_1^r}]
\frac{\partial H_1^i}{\partial W_1^i}.
$$
See more information on backpropagation in the following list

* [Back-propagation, an introduction at offconvex.org](http://www.offconvex.org/2016/12/20/backprop/);
* [Backpropagation on Wikipedia](https://www.wikiwand.com/en/Backpropagation);
* [Automatic differentiation on Wikipedia](https://www.wikiwand.com/en/Automatic_differentiation);
* [backpropagation on brilliant](https://brilliant.org/wiki/backpropagation/);
* An introduction to automatic differentiation at <https://alexey.radul.name/ideas/2013/introduction-to-automatic-differentiation/>;
* Reverse-mode automatic differentiation: a tutorial at <https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation>.
* [Who invented backpropagation ?](http://people.idsia.ch/~juergen/who-invented-backpropagation.html);
* [Autodiff Workshop](https://autodiff-workshop.github.io/)
* [如何直观地解释 backpropagation 算法？ - 景略集智的回答 - 知乎](https://www.zhihu.com/question/27239198/answer/537357910)
* The chapter 2 *How the backpropagation algorithm works* at the online book <http://neuralnetworksanddeeplearning.com/chap2.html>
* For more information on automatic differentiation see the book *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation, Second Edition* by Andreas Griewank and Andrea Walther_ at <https://epubs.siam.org/doi/book/10.1137/1.9780898717761>.

![](http://ai.stanford.edu/~tengyuma/forblog/weight5.jpg)


#### Training Methods

The training is to find the optimal parameters of the model based on the **training data set**. The training methods are usually based on the gradient of cost function as well as back-propagation algorithm  in deep learning.
See **Stochastic Gradient Descent** in **Numerical Optimization** for details.
In this section, we will talk other optimization tricks such as **Normalization**.

| Concepts | Interpretation|
|:--------:|:-------------:|
|*Overfitting* and *Underfitting*| See [Overfitting](https://www.wikiwand.com/en/Overfitting) or [Overfitting and Underfitting With Machine Learning Algorithms](https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/)|
|*Memorization* and *Generalization*|Memorizing, given facts, is an obvious task in learning. This can be done by storing the input samples explicitly, or by identifying the concept behind the input data, and memorizing their general rules. The ability to identify the rules, to generalize, allows the system to make predictions on unknown data. Despite the strictly logical invalidity of this approach, the process of reasoning from specific samples to the general case can be observed in human learning. From <https://www.teco.edu/~albrecht/neuro/html/node9.html>.|
|*Normalization* and *Standardization*| *Normalization* is to scale the data into the interval [0,1] while *Standardization* is to rescale the datum with zero mean $0$ and unit variance $1$. See [Standardization vs. normalization](http://www.dataminingblog.com/standardization-vs-normalization/).|

##### Initialization and More

* https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
* https://github.com/kmkolasinski/deep-learning-notes/tree/master/seminars/2018-12-Improving-DL-with-tricks
* [An Empirical Model of Large-Batch Training Gradient Descent with Random Initialization: Fast Global Convergence for Nonconvex Phase Retrieva](http://www.princeton.edu/~congm/Publication/RandomInit/main.pdf)
* [Gradient descent and variants](http://www.cnblogs.com/yymn/articles/4995755.html)
* [optimization beyond landscape at offconvex.org](http://www.offconvex.org/2018/11/07/optimization-beyond-landscape/)
* [graphcore.ai](https://www.graphcore.ai/posts/revisiting-small-batch-training-for-deep-neural-networks)
* [可视化超参数作用机制：二、权重初始化](https://zhuanlan.zhihu.com/p/38315135)
* [第6章 网络优化与正则化](https://nndl.github.io/chap-%E7%BD%91%E7%BB%9C%E4%BC%98%E5%8C%96%E4%B8%8E%E6%AD%A3%E5%88%99%E5%8C%96.pdf)
* [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf)

##### Normalization

* [Batch normalization 和 Instance normalization 的对比？ - Naiyan Wang的回答 - 知乎](https://www.zhihu.com/question/68730628/answer/277339783)
* [Weight Normalization 相比 batch Normalization 有什么优点呢？](https://www.zhihu.com/question/55132852/answer/171250929)
* [深度学习中的Normalization模型](https://www.jiqizhixin.com/articles/2018-08-29-7)
* [Group Normalization](https://arxiv.org/abs/1803.08494)
* [Busting the myth about batch normalization at paperspace.com](https://blog.paperspace.com/busting-the-myths-about-batch-normalization/)
* The original paper *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* at <https://arxiv.org/pdf/1502.03167.pdf>.

See **Improve the way neural networks learn** at <http://neuralnetworksanddeeplearning.com/chap3.html>.
See more on nonconvex optimization at <http://sunju.org/research/nonconvex/>.

#### Regularization

In mathematics, statistics, and computer science, particularly in the fields of machine learning and inverse problems, regularization is a process of introducing additional information in order to solve an ill-posed problem or to prevent over-fitting.
In general, regularization is a technique that applies to objective functions in ill-posed optimization problems.
It changes the objective function or more generally the optimization procedure. However, it is not crystal clear that what is the relationship between the optimization techniques and generalization ability.
See the following links for more information on optimization and generalization.

* https://www.inference.vc/sharp-vs-flat-minima-are-still-a-mystery-to-me/
* https://arxiv.org/abs/1703.04933
* https://arxiv.org/abs/1810.05369
* https://blog.csdn.net/xzy_thu/article/details/80732220
* http://www.offconvex.org/2017/12/08/generalization1/
* http://www.offconvex.org/2018/02/17/generalization2/
* http://www.offconvex.org/2017/03/30/GANs2/
* https://machinelearningmastery.com/blog/
* http://lcsl.mit.edu/courses/regml/regml2016/
* https://chunml.github.io/ChunML.github.io/tutorial/Regularization/
* http://www.mit.edu/~9.520/fall16/
* https://arxiv.org/pdf/1506.02142.pdf

##### Parameter norm penalty

The $\ell_2$ norm penalty  is to add the squares of $\ell_2$ norm of parameters to the objective function $J(\theta)$ to reduce the parameters(or weights) as shown in ridge regression with regular term coefficient $\lambda$, i.e.
$J(\theta)+\lambda {\|\theta\|}_{2}^{2}.$
Suppose  $E(\theta)=J(\theta)+\lambda {\|\theta\|}_{2}^{2}$, the gradient descent take approximate (maybe inappropriate)  form
$$
\theta=\theta-\eta\frac{\partial E(\theta)}{\partial \theta}=\theta -\eta\frac{\partial J(\theta)}{\partial \theta}-2\eta\lambda \theta
$$
thus
$$
\frac{\partial J(\theta)}{\partial \theta}=-2\lambda\theta\implies J(\theta)=e^{-2\lambda \theta}.
$$

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
*

The $\ell_1$ norm penalty is also used in deep learning as in **LASSO**. It is to solve the following optimization problem:
  $$\min_{\theta}J(\theta)+\lambda{\|\theta\|}_1,$$
where $\lambda$ is a hyperparameter. Sparsity  brings to the model as shown as in **LASSO**.

##### Early stop

Its essential is to make a balance in memorization and generalization.
Early stopping is to stop the procedure before finding the minima of cost in training data. It is one direct application of **cross validation**.

* https://www.wikiwand.com/en/Early_stopping
* https://www.wikiwand.com/en/Cross-validation_(statistics)
* https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
* https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/

##### Dropout

It is to cripple the connections stochastically, which  is often used in visual tasks. See the original paper [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf).

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

##### Data Augmentation

Data augmentation is to augment the training datum specially in visual recognition.
**Overfitting** in supervised learning is data-dependent. In other words, the model may generalize better if the data set is more diverse.
It is to collect more datum in the statistical perspective.

* [The Effectiveness of Data Augmentation in Image Classification using Deep Learning](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf)
* http://www.cnblogs.com/love6tao/p/5841648.html

***

|Feed forward and Propagate backwards|
|:----------------------------------:|
|![TF](http://beamandrew.github.io//images/deep_learning_101/tensors_flowing.gif)|

## Convolutional Neural Network

Convolutional neural network is originally aimed to solve visual tasks. In so-called [Three Giants' Survey](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf), the history of ConvNet and deep learning is curated.
[Deep, Deep Trouble--Deep Learning’s Impact on Image Processing, Mathematics, and Humanity](https://sinews.siam.org/Details-Page/deep-deep-trouble-4) tells us the  mathematicians' impression on ConvNet in image processing.

### Convolutional layer

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
   + [Kernel in image processing ](https://www.wikiwand.com/en/Kernel_(image_processing)) takes the relationship of the neighboring entries into consideration. It transforms the neighboring entries into one real value. In 2-dimenional space, convolution corresponds to $\color{aqua}{\text{doubly block circulant matrix}}$ if the matrix is flatten. It is local pattern that we can learn.

|The illustration of convolution operator|
|:---:|
|![](https://pic4.zhimg.com/v2-15fea61b768f7561648dbea164fcb75f_b.gif)|
|(http://cs231n.github.io/assets/conv-demo/index.html)|
|![](https://ujwlkarn.files.wordpress.com/2016/08/giphy.gif?w=748)|

***
As similar as the inner product of vector, the convolution operators  can compute the similarity between the submatrix of images and the kernels (also called filters).
The convolution operators play the role as *parameter sharing* and *local connection*.
For more information on CNN, click the following links.

* [CNN(卷积神经网络)是什么？有入门简介或文章吗？ - 机器之心的回答 - 知乎](https://www.zhihu.com/question/52668301/answer/131573702)
* [能否对卷积神经网络工作原理做一个直观的解释？ - YJango的回答 - 知乎](https://www.zhihu.com/question/39022858/answer/194996805)
* [One by one convolution](https://iamaaditya.github.io/2016/03/one-by-one-convolution/)
* [conv arithmetic](https://github.com/vdumoulin/conv_arithmetic)
* [Convolution deep learning](http://timdettmers.com/2015/03/26/convolution-deep-learning/)

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
Spatial Pooling (also called subsampling or downsampling) is to use some summary statistic that extract from spatial neighbors,
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
\tilde{H_i}=C_i\otimes(P\oplus H_{i-1})   \\
H_i = Pooling\cdot (\sigma\circ \tilde{H_i})
$$
where $\otimes,\oplus,\cdot$ represent convolution operation, padding and pooling, respectively.

|Diagram of Convolutional neural network|
|:-----------------------------------:|
|![](http://www.linleygroup.com/mpr/h/2016/11561/U26_F3.png)|

******

* [CS231n Convolutional Neural Network for Visual Recognition](http://vision.stanford.edu/teaching/cs231n/index.html)
* The *Wikipeida* page <https://www.wikiwand.com/en/Convolutional_neural_network>
* [Awesome deep vision](https://github.com/kjw0612/awesome-deep-vision)
* [解析深度学习——卷积神经网络原理与视觉实践](http://lamda.nju.edu.cn/weixs/book/CNN_book.html)
* [Interpretable Convolutional Neural Networks](http://qszhang.com/index.php/icnn/)
* [An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)
* [Convolutional Neural Network Visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
* [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/)
* https://www.vicarious.com/2017/10/20/toward-learning-a-compositional-visual-representation/

***

https://zhuanlan.zhihu.com/p/28749411
![](http://1reddrop.com/wp-content/uploads/2016/08/deeplearning.jpg)

### Visualization of CNN

* [Deep Visualization](http://yosinski.com/deepvis)
* [Interpretable Representation Learning for Visual Intelligence](http://bzhou.ie.cuhk.edu.hk/publication/thesis.pdf)
* https://www.zybuluo.com/lutingting/note/459569
* https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
* [卷积网络的可视化与可解释性（资料整理） - 陈博的文章 - 知乎](https://zhuanlan.zhihu.com/p/36474488)
* https://zhuanlan.zhihu.com/p/24833574
* https://zhuanlan.zhihu.com/p/30403766
* https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
* http://people.csail.mit.edu/bzhou/ppt/presentation_ICML_workshop.pdf
* https://www.robots.ox.ac.uk/~vedaldi//research/visualization/visualization.html
* https://www.graphcore.ai/posts/what-does-machine-learning-look-like

****
|graphcore.ai|
|:----------:|
|![](https://www.graphcore.ai/hubfs/images/alexnet_label%20logo.jpg?t=1541693113453)|
****

## Recurrent Neural Networks and Long Short-Time Memory

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

It may suit the **identically independently distributed** data set.
However, the sequence data is not **identically independently distributed** in most cases. For example, the outcome of current decision determines the next decision.
In mathematics it can be expressed as
$$
H_{t}=\sigma\circ(X_t,H_{t-1})=\sigma\circ(W H_{t-1} + U X_{t}+b),
$$
where $X_{t}\in\mathbb{R}^{p}$ is the output $\forall t\in\{1,2\dots,\tau\}$.

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
|![](https://devblogs.nvidia.com/wp-content/uploads/2015/05/Figure_3-624x208.png)|\
|(https://www.altoros.com/blog/wp-content/uploads/2017/01/Deep-Learning-Using-TensorFlow-recurrent-neural-networks.png)|

### Bi-directional RNN

For each step from $t=1$ to $t = \tau$, the complete update equations of RNN:
$$
\begin{align}
\stackrel{\rightarrow}H_{t} &=\sigma\circ(W_1\stackrel{\rightarrow}H_{t-1}+U_1X_{t}+b_1) \\
\stackrel{\leftarrow}H_{t}  &=\sigma\circ(W_2\stackrel{\leftarrow}H_{t+1}+U_2X_{t}+b_2) \\
O_{t} &= \mathrm{softmax}(VH_{t}+c)
\end{align}
$$
where $H_{t} = [\stackrel{\rightarrow} H_{t};\stackrel{\leftarrow} H_{t}]$.

|Bi-directional RNN|
|:----------------:|
|![](http://www.wildml.com/wp-content/uploads/2015/09/bidirectional-rnn.png)|
|The bold line(__) is computed earlier than the dotted line(...).|

[deepai.org](https://deepai.org/machine-learning-glossary-and-terms/bidirectional-recurrent-neural-networks)

### LSTM

There are several architectures of LSTM units. A common architecture is composed of a *memory cell, an input gate, an output gate and a forget gate*.
It is the first time to solve the **gradient vanishing problem** and
**long-term dependencies** in deep learning.

|LSTM block|
|:---:|
|See *LSTM block* at(https://devblogs.nvidia.com/wp-content/uploads/2016/03/LSTM.png)|
|![](http://5b0988e595225.cdn.sohucs.com/images/20181017/cd0e8107c3f94e849bd82e0fd0123776.jpeg)|

* forget gate
$f_{t}=\sigma(W_{f}[h_{t-1},x_t]+b_f)$
![forget gate](http://5b0988e595225.cdn.sohucs.com/images/20181017/87957492ade3445ea90871dda02c92ca.gif)
* input gate
$i_{t}=\sigma(W_i[h_{t-1},x_t]+b_i)$,
$\tilde{C}_{t}=tanh(W_C[h_{t-1},x_t]+b_C)$
![](http://5b0988e595225.cdn.sohucs.com/images/20181017/dac15a9da4164b3da346ac891d10ff9e.gif)
* memory cell
$c_{t}=f_t\times c_{t-1}+i_{t}\times{\tilde{C}_{t}}$
![](http://5b0988e595225.cdn.sohucs.com/images/20181017/ad6435e481064d57833a4733e716fa8f.gif)
* output gate
$O_{t}=\sigma(W_{O}[h_{t-1},x_t]+b_{O})$ and
$h_{t}=O_{t}\times tanh(c_{t})$
![](http://5b0988e595225.cdn.sohucs.com/images/20181017/bca51dcc89f14b1c9a2e9076f419540a.gif)


|[Inventor of LSTM](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735?journalCode=neco)|
|:---:|
|![Juergen Schmidhuber](https://www.analyticsindiamag.com/wp-content/uploads/2018/09/jurgen-banner.png) more on (http://people.idsia.ch/~juergen/)|


* [LSTM in Wikipeida](https://www.wikiwand.com/en/Long_short-term_memory)
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) and its Chinese version <https://www.jianshu.com/p/9dc9f41f0b29>.
* [LTSMvis](http://lstm.seas.harvard.edu/)
* [Jürgen Schmidhuber's page on Recurrent Neural Networks](http://people.idsia.ch/~juergen/rnn.html)
* https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/

### Deep RNN

Deep RNN is composed of *RNN cell* as MLP is composed of perceptrons.
For each step from $t=1$ to $t=\tau$, the complete update equations of deep $d$-RNN  at the $i$th layer:
$$
\begin{align}
H_{t}^{i} &=\sigma\circ(W_{i} H_{t-1} + U_{i} X_{t} + b) \\
O_{t} &= \mathrm{softmax}(V H_{d} + c)
\end{align}
$$
where the parameters are the bias vectors $b$ and $c$ along with the weight matrices
$U_i$, $V$ and $W_i$, respectively for input-to-hidden, hidden-to-output and hidden-to-hidden connections for $i\in \{1,2,\cdots,d\}$.

Other RNN cells also can compose deep RNN via this stacking way such as deep Bi-RNN networks.

|Deep Bi-RNN|
|:---:|
|![](http://opennmt.net/OpenNMT/img/brnn.png)|

* <http://blog.songru.org/posts/notebook/Opinion_Mining_with_Deep_Recurrent_Neural_Networks_NOTE/>
* http://opennmt.net/OpenNMT/training/models/
* https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

***

* https://machinelearningmastery.com/recurrent-neural-network-algorithms-for-deep-learning/
* http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/
* http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
* https://zhuanlan.zhihu.com/p/27485750
* [RNN in Wikipeida](https://en.wikipedia.org/wiki/Recurrent_neural_network)
* [Awesome RNN](https://github.com/kjw0612/awesome-rnn)
* [RNN in metacademy](https://metacademy.org/graphs/concepts/recurrent_neural_networks)
* https://zhuanlan.zhihu.com/p/49834993
* [RNNs in Tensorflow, a Practical Guide and Undocumented Features](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* https://kvitajakub.github.io/2016/04/14/rnn-diagrams/
* http://www.zmonster.me/notes/visualization-analysis-for-rnn.html
* http://imgtec.eetrend.com/d6-imgtec/blog/2018-10/18051.html
* https://arxiv.org/pdf/1801.01078.pdf
* http://www.sohu.com/a/259957763_610300
* https://skymind.ai/wiki/lstm
* [循环神经网络(RNN, Recurrent Neural Networks)介绍](https://blog.csdn.net/heyongluoyao8/article/details/48636251)
* https://arxiv.org/pdf/1506.02078.pdf

## Attention Mechanism

An attention model is a method that takes $n$ arguments $y_1, \dots, y_n$  and a context $c$. It return a vector $z$ which is supposed to be the  **summary** of the ${y}_i\in \mathbb{R}^{d}$, focusing on information linked to the context $c$. More formally, it returns a weighted arithmetic mean of the $y_i$, and the weights are chosen according the relevance of each $y_i$ given the context $c$.
In mathematics, it can be expressed as:

$$
{\alpha}_i = softmax[s(y_i,c)]               \\
         z = \sum_{i=1}^{n} {\alpha}_i y_i
$$

where $s(\cdot, \cdot)$ is the attention scoring function.
The attention scoring function $s({y}_i, c)$ is diverse, such as:

* the additive model $s({y}_i, c) = v^{T} tanh\circ (W {y}_i + U c)$, where $v \in \mathbb{R}^{d}$, $W \in \mathbb{R}^{d\times d}$, $U \in \mathbb{R}^{d}$ are parameters to learn;
* the inner product model $s({y}_i, c) = \left< {y}_i, c \right>$, i.e. the inner product of ${y}_i, c$;
* the scaled inner product model $s({y}_i, c) = \frac{\left< {y}_i, c \right>}{d}$,where $d$ is the dimension of input ${y}_i$;
* the bilinear model $s({y}_i, c) = {y}_i^{T} W c$, where $W\in \mathbb{R}^{d\times d}$ is parameter matrix to learn.

It is always as one component of some complex network as normalization.

***

* [第 8 章 注意力机制与外部记忆](https://nndl.github.io/chap-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6%E4%B8%8E%E5%A4%96%E9%83%A8%E8%AE%B0%E5%BF%86.pdf)
* https://skymind.ai/wiki/attention-mechanism-memory-network
* https://distill.pub/2016/augmented-rnns/
* https://blog.heuritech.com/2016/01/20/attention-mechanism/
* [Attention mechanism](https://github.com/philipperemy/keras-attention-mechanism)
* [Attention and Memory in Deep Learning and NLP](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)
* http://www.deeplearningpatterns.com/doku.php?id=attention
* [细想神毫注意深-注意力机制 - 史博的文章 - 知乎](https://zhuanlan.zhihu.com/p/51747716)

## Recursive Neural Network

* http://www.iro.umontreal.ca/~bengioy/talks/gss2012-YB6-NLP-recursive.pdf
* https://www.wikiwand.com/en/Recursive_neural_network
* https://cs224d.stanford.edu/lectures/CS224d-Lecture10.pdf
* https://devblogs.nvidia.com/recursive-neural-networks-pytorch/
* http://sei.pku.edu.cn/~luyy11/slides/slides_141029_RNN.pdf

|Diagram of Recursive Neural Network|
|-----------------------------------|
|![](http://www.cs.cornell.edu/~oirsoy/files/drsv/deep-recursive.png)|

## Graph Convolution Network

Graph can be represented as `adjacency matrix` as shown in *Graph Algorithm*. However, the adjacency matrix only describe the connections between the nodes. The feature of the nodes does not appear. The node itself really matters.
For example, the chemical bonds can be representd as `adjacency matrix` while the atoms in molecule really determine the properties of the molecule.

A naive approach is to concatenate the `feature matrix` $X\in \mathbb{R}^{N\times E}$ and `adjacency matrix` $A\in \mathbb{R}^{N\times N}$, i.e. $X_{in}=[X, A]\in \mathbb{R}^{N\times (N+E)}$. And what is the output?

How can deep learning apply to them?

> For these models, the goal is then to learn a function of signals/features on a graph $G=(V,E)$ which takes as input:

> * A feature description $x_i$ for every node $i$; summarized in a $N\times D$ feature matrix $X$ ($N$: number of nodes, $D$: number of input features)
> * A representative description of the graph structure in matrix form; typically in the form of an adjacency matrix $A$ (or some function thereof)

> and produces a node-level output $Z$ (an $N\times F$ feature matrix, where $F$ is the number of output features per node). Graph-level outputs can be modeled by introducing some form of pooling operation (see, e.g. [Duvenaud et al., NIPS 2015](http://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints)).

Every neural network layer can then be written as a non-linear function
$${H}_{i+1} = \sigma \circ ({H}_{i}, A)$$
with ${H}_0 = {X}_{in}$ and ${H}_{d} = Z$ (or $Z$ for graph-level outputs), $d$ being the number of layers. The specific models then differ only in how $\sigma$ is chosen and parameterized.

For example, we can consider a simple form of a layer-wise propagation rule
$$
{H}_{i+1} = \sigma \circ ({H}_{i}, A)=\sigma \circ(A {H}_{i} {W}_{i})
$$
where ${W}_{i}$ is a weight matrix for the $i$-th neural network layer and $\sigma (\cdot)$ is is a non-linear activation function such as *ReLU*.

* But first, let us address two limitations of this simple model: multiplication with $A$ means that, for every node, we sum up all the feature vectors of all neighboring nodes but not the node itself (unless there are self-loops in the graph). We can "fix" this by enforcing self-loops in the graph: we simply add the identity matrix $I$ to $A$.

* The second major limitation is that $A$ is typically not normalized and therefore the multiplication with $A$ will completely change the scale of the feature vectors (we can understand that by looking at the eigenvalues of $A$).Normalizing $A$ such that all rows sum to one, i.e. $D^{−1}A$, where $D$ is the diagonal node degree matrix, gets rid of this problem.

In fact, the propagation rule introduced in [Kipf & Welling (ICLR 2017)](https://arxiv.org/abs/1609.02907) is given by:
$$
{H}_{i+1} = \sigma \circ ({H}_{i}, A)=\sigma \circ(\hat{D}^{-\frac{1}{2}} \hat{A} \hat{D}^{-\frac{1}{2}} {H}_{i} {W}_{i}),
$$
with $\hat{A}=A+I$, where $I$ is the identity matrix and $\hat{D}$ is the diagonal node degree matrix of $\hat{A}$.
See more details at [Multi-layer Graph Convolutional Network (GCN) with first-order filters](http://tkipf.github.io/graph-convolutional-networks/).

Like other neural network, GCN is also composite of linear and nonlinear mapping. In details,
1. $\hat{D}^{-\frac{1}{2}} \hat{A} \hat{D}^{-\frac{1}{2}}$ is to normalize the graph structure;
2. the next step is to multiply node properties and weights;
3. Add nonlinearities by activation function $\sigma$.

[See more at experoinc.com](https://www.experoinc.com/post/node-classification-by-graph-convolutional-network).

![GCN](http://tkipf.github.io/graph-convolutional-networks/images/gcn_web.png)



![CNN VS. GCNN](https://research.preferred.jp/wp-content/uploads/2017/12/cnn-gcnn.png)

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
* https://www.experoinc.com/post/node-classification-by-graph-convolutional-network
* https://www.groundai.com/project/graph-convolutional-networks-for-text-classification/
* https://datawarrior.wordpress.com/2018/08/08/graph-convolutional-neural-network-part-i/
* https://datawarrior.wordpress.com/2018/08/12/graph-convolutional-neural-network-part-ii/
* http://blog.lcyown.cn/2018/04/30/graphencoding/


$\color{navy}{\text{Graph convolution network is potential to}}\, \cal{reasoning}$ as the blend of $\frak{\text{probabilistic graph model}}$ and $\mit{\text{deep learning}}$.

GCN can be regarded as the counterpart of CNN for graphs so that the optimization techniques such as normalization, attention mechanism and even the adversarial version can be extended to the graph structure.

### ChebNet, CayleyNet, MotifNet

In the previous post, the convolution of the graph Laplacian is defined in its **graph Fourier space** as outlined in the paper of Bruna et. al. (arXiv:1312.6203). However, the **eigenmodes** of the graph Laplacian are not ideal because it makes the bases to be graph-dependent. A lot of works were done in order to solve this problem, with the help of various special functions to express the filter functions. Examples include Chebyshev polynomials and Cayley transform.

https://datawarrior.wordpress.com/2018/08/12/graph-convolutional-neural-network-part-ii/

## Generative Adversarial Network

It origins from <http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf>.

* https://skymind.ai/wiki/generative-adversarial-network-gan
* [千奇百怪的GAN变体，都在这里了（持续更新嘤） - 量子学园的文章 - 知乎](https://zhuanlan.zhihu.com/p/26491601)
* [生成模型中的左右互搏术：生成对抗网络GAN——深度学习第二十章（四） - 川陀学者的文章 - 知乎](https://)https://zhuanlan.zhihu.com/p/37846221)
* [Really awesome GANs](https://github.com/nightrome/really-awesome-gan)
* [GAN zoo](https://github.com/hindupuravinash/the-gan-zoo)
* https://gandissect.csail.mit.edu/
* https://poloclub.github.io/ganlab/
* https://github.com/nndl/Generative-Adversarial-Network-Tutorial

|Generative Adversarial Network|
|:----------------------------:|
|![](https://image.slidesharecdn.com/generativeadversarialnetworks-161121164827/95/generative-adversarial-networks-11-638.jpg?cb=1480242452)|

## Network Compression

* [Distiller](https://nervanasystems.github.io/distiller/index.html)
* [Deep Compression and EIE](https://web.stanford.edu/class/ee380/Abstracts/160106-slides.pdf)
* [Network Speed and Compression](https://github.com/mrgloom/Network-Speed-and-Compression)
* https://arxiv.org/pdf/1712.01887.pdf
* https://hanlab.mit.edu/projects/tsm/
* [Papers Reading List of *Embeded Neural Network*](https://github.com/ZhishengWang/Embedded-Neural-Network)
* [SigDL -- Deep Learning for IoT Device and Edge Computing Embedded Targets](https://github.com/signalogic/SigDL#DeepLearningModelCompression)
* https://arxiv.org/abs/1804.03294

## Bayesian Deep Learning

* https://github.com/junlulocky/bayesian-deep-learning-notes
* https://github.com/robi56/awesome-bayesian-deep-learning#theory
* https://alexgkendall.com/computer_vision/phd_thesis/
* http://bayesiandeeplearning.org/
* http://twiecki.github.io/blog/2016/06/01/bayesian-deep-learning/

## Theories of Deep Learning

* [Short Course of Deep Learning 2016 Autumn, PKU](http://www.xn--vjq503akpco3w.top/)
* [深度学习名校课程大全 - 史博的文章 - 知乎](https://zhuanlan.zhihu.com/p/31988246)
* [Theories of Deep Learning (STATS 385)](https://stats385.github.io/)
* [Topics Course on Deep Learning for Spring 2016 by Joan Bruna, UC Berkeley, Statistics Department](https://github.com/joanbruna/stat212b)
* [Mathematical aspects of Deep Learning](http://elmos.scripts.mit.edu/mathofdeeplearning/)
* [MATH 6380p. Advanced Topics in Deep Learning Fall 2018](https://deeplearning-math.github.io/)
* [CoMS E6998 003: Advanced Topics in Deep Learning](https://www.advancedtopicsindeeplearning.com/)
* [Deep Learning Theory: Approximation, Optimization, Generalization](http://www.mit.edu/~9.520/fall17/Classes/deep_learning_theory.html)
* [Theory of Deep Learning, ICML'2018](https://sites.google.com/site/deeplearningtheory/)
* [Deep Neural Networks: Approximation Theory and Compositionality](http://www.mit.edu/~9.520/fall16/Classes/deep_approx.html)
* [Neural Networks, Manifolds, and Topology](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)
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
* http://blog.qure.ai/notes/visualizing_deep_learning
* http://blog.qure.ai/notes/deep-learning-visualization-gradient-based-methods
* http://stillbreeze.github.io/Deep-Learning-and-the-Demand-For-Interpretability/
* https://beenkim.github.io/
* https://www.robots.ox.ac.uk/~vedaldi//research/idiu/idiu.html
* http://networkinterpretability.org/
* https://interpretablevision.github.io/
* https://zhuanlan.zhihu.com/p/45695998
* https://www.zhihu.com/question/265917569
* https://www.ias.edu/ideas/2017/manning-deep-learning
* https://www.jiqizhixin.com/articles/2018-08-03-10
* https://cloud.tencent.com/developer/article/1345239
* http://www.deeplearningpatterns.com/doku.php?id=theory
* http://cbmm.mit.edu/publications
* https://stanford.edu/~shervine/l/zh/teaching/cs-229/cheatsheet-deep-learning
* https://stanford.edu/~shervine/teaching/cs-230.html

***
|Deep Dream|
|:--------:|
|![Deep Dream](http://grayarea.org/wp-content/uploads/2016/03/3057368-inline-i-1-inside-the-first-deep-dream-art-show.jpg)|
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
* [EE-559 – DEEP LEARNING (SPRING 2018)](https://documents.epfl.ch/users/f/fl/fleuret/www/dlc/)
* [The Functions of Deep Learning](https://sinews.siam.org/Details-Page/the-functions-of-deep-learning)
* https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-deep-learning
* https://www.deeplearningwizard.com/
* https://www.stateoftheart.ai/
* [神经网络与深度学习](https://nndl.github.io/)
* https://mchromiak.github.io/articles/2017/Sep/01/Primer-NN/#.XBXb42h3hPY
