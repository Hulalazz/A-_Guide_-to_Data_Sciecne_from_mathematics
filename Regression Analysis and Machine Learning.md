## Regression Analysis

Regression is a method for studying the relationship between a response variable $Y$ and a covariates $X$. The covariate is also called a **predictor** variable or a **feature**.

Regression is not function fitting. In function fitting,  it is well-defined - $f(x_i)$ is fixed when $x_i$ is fixed; in regression, it is not always so.

Linear regression is the "hello world" in statistical learning. It is the simplest model to fit the datum. We will induce it in the maximum likelihood estimation perspective.
See this link for more information <https://www.wikiwand.com/en/Regression_analysis>.

### Ordinary Least Squares

#### Representation of Ordinary Least Squares

A linear regression model assumes that the regression function $E(Y|X)$ is
linear in the inputs $X_1,\cdots, X_p$. They are simple and often
provide an adequate and interpretable description of how the inputs affect the output.
Suppose that the datum $\{(x_i, y_i)\}_{i=1}^{n}$,
$$
{y}_{i} = f({x}_{i})+{\epsilon}_{i},
$$
where the function $f$ is linear, i.e. $f(x)=w^{T}x + b$.
Let $\epsilon = y - f(x)$.Then $\mathbb{E}(\epsilon|X) = \mathbb{E}(y - f(x)|x)=0$
and the residual errors $\{{\epsilon}_{i}|{\epsilon}_{i} = {y}_{i} - f(x_i)\}_{i=1}^{n}$ are **i.i.d. in standard Gaussian distribution**.

By convention (**very important**!):

* $\mathrm{x}$ is assumed to be standardized (mean 0, unit variance);
* $\mathrm{y}$ is assumed to be centered.

For the linear regression, we could assume $\mathrm{x}$ is in Gaussian distribution.

#### Evaluation of Ordinary Least Squares

The likelihood of the errors are  
$$
L(\epsilon_1,\epsilon_2,\cdots,\epsilon_n)=\prod_{i=1}^{n}\frac{1}{\sqrt{2\pi}}e^{-\epsilon_i^2}.
$$

In MLE, we have shown that it is equivalent to
$$
  \arg\max \prod_{i=1}^{n}\frac{1}{\sqrt{2\pi}}e^{-\epsilon_i^2}=\arg\min\sum_{i=1}^{n}\epsilon_i^2=\arg\min\sum_{i=1}^{n}(y_i-f(x_i))^2.
$$

|Linear Regression and Likelihood Maximum Estimation|
|:-------------------------------------------------:|
|![<reliawiki.org>](http://reliawiki.org/images/2/28/Doe4.3.png)|
***

#### Optimization of Ordinary Least Squares

For linear regression, the function $f$ is linear, i.e. $f(x) = w^Tx$ where $w$ is the parameters to tune. Thus $\epsilon_i = y_i-w^Tx_i$ and $\sum_{i=1}^{n}(y_i-f(x_i))^2=\sum_{i=1}^{n}(y_i - w^Tx_i)^2$. It is also called *residual sum of squares* in statistics or *objective function* in optimization theory.
In a compact form,
$$
\sum_{i=1}^{n}(y_i - w^T x_i)^2=\|Y-Xw\|^2\,\tag 0,
$$
where $Y=(y_1,y_2,\cdots, y_n)^T, X=(x_1, x_2,\cdots,x_n)$.
Let the gradient of objective function $\|Y-Xw\|^2$ be 0, i.e.
$$
\nabla_{w}{\|Y-Xw\|^2}=2X^T(Y-Xw)=0\,\tag 1,
$$
then we gain that **$w=(X^TX)^{-1}X^TY$** if possible.

$\color{lime}{Note}$:

1. the residual error $\{\epsilon_i\}_{i=1}^{n}$ are i.i.d. in Gaussian distribution;
2. the inverse matrix $(X^{T}X)^{-1}$ may not exist in some extreme case.

See more on [Wikipedia page](https://www.wikiwand.com/en/Ordinary_least_squares).

### Ridge Regression and LASSO

When the matrix $X^{T}X$ is not inverse, ordinary least squares does not work.
And in ordinary least squares, the parameters $w$ is estimated by MLE rather more general Bayesian estimator.

In the perspective of computation, we would like to consider the *regularization* technique;
In the perspective of Bayesian statistics, we would like to consider more proper *prior* distribution of the parameters.

#### Ridge Regression As Regularization

It is to optimize the following objective function with parameter norm penalty
$$
PRSS_{\ell_2}=\sum_{i=1}^{n}(y_i-w^Tx_i)^2+\lambda w^{T}w=\|Y-Xw\|^2+\lambda\|w\|^2\,\tag {Ridge}.
$$
It is called penalized residual sum of squares.
Taking derivatives, we solve
$$
\frac{\partial PRSS_{\ell_2}}{\partial w}=2X^T(Y-Xw)+2\lambda w=0
$$
and we gain that
$$
w=(X^{T}X+\lambda I)^{-1}X^{T}Y
$$
where it is trackable if $\lambda$ is large enough.

#### LASSO as Regularization

LASSO  is the abbreviation of **Least Absolute Shrinkage and Selection Operator**.
1. It is to minimize the following objective function：
$$
PRSS_{\ell_1}=\sum_{i=1}^{n}(y_i-w^Tx_i)^2+\lambda{\|w\|}_{1} =\|Y-Xw\|^2+\lambda{\|w\|}_1\,\tag {LASSO}.
$$

2. the optimization form:
$$
\begin{align}
 \arg\min_{w}\sum_{i=1}^{n}(y_i-w^Tx_i)^2 &\qquad\text{Objective function} \\
 \text{subject to}\,{\|w\|}_1 \leq t      &\qquad\text{constraint}.
\end{align}
$$

3. the selection form:
$$
\begin{align}
 \arg\min_{w}{\|w\|}_1                                  \qquad &\text{Objective function} \\
 \text{subject to}\,\sum_{i=1}^{n}(y_i-w^Tx_i)^2 \leq t \qquad &\text{constraint}.
\end{align}
$$
where ${\|w\|}_1=\sum_{i=1}^{n}|w_i|$ if $w=(w_1,w_2,\cdots, w_n)^{T}$.

More solutions to this optimization problem:

* [Q&A in zhihu.com](https://www.zhihu.com/question/22332436/answer/21068494);
* [ADMM to LASSO](http://www.simonlucey.com/lasso-using-admm/);
* [Regularization: Ridge Regression and the LASSO](http://statweb.stanford.edu/~tibs/sta305files/Rudyregularization.pdf);
* [Least angle regression](https://www.wikiwand.com/en/Least-angle_regression);
* http://web.stanford.edu/~hastie/StatLearnSparsity/
* [历史的角度来看，Robert Tibshirani 的 Lasso 到底是不是革命性的创新？- 若羽的回答 - 知乎](https://www.zhihu.com/question/275196908/answer/533790835)
* http://www.cnblogs.com/xingshansi/p/6890048.html

|LASSO and Ridge Regression|
|:------------------------:|
|![](https://pic3.zhimg.com/80/v2-2a88e2acc009fa4de3edeb51e683ca02_hd.png)|
|[The LASSO Page](http://statweb.stanford.edu/~tibs/lasso.html) ,[Wikipedia page](https://www.wikiwand.com/en/Lasso_(statistics)) and [Q&A in zhihu.com](https://www.zhihu.com/question/275196908/answer/378776602)|
|[More References in Chinese blog](https://blog.csdn.net/godenlove007/article/details/11387977)|

#### Bayesian Perspective

If we suppose the prior distribution  of the parameters $w$ is in Gaussian distribution, i.e. $f_{W}(w)\propto e^{-\lambda\|w\|^{2}}$, we will deduce the ridge regression.
If we suppose the prior distribution  of the parameters $w$ is in Laplacian distribution, i.e. $f_{W}(w)\propto e^{-\lambda{\|w\|}_1}$, we will deduce LASSO.


* [Stat 305: Linear Models (and more)](http://statweb.stanford.edu/~tibs/sta305.html)
* [机器学习算法实践-岭回归和LASSO - PytLab酱的文章 - 知乎](https://zhuanlan.zhihu.com/p/30535220)

#### Elastic Net

When the sample size of training data $n$ is far less than the number of features $p$, the objective function is:
$$
PRSS_{\alpha}=\frac{1}{2n}\sum_{i=1}^{n}(y_i -w^T x_i)^2+\lambda\sum_{j=1}^{p}P_{\alpha}(w_j),\tag {Elastic net}
$$
where $P_{\alpha}(w_j)=\alpha {|w_j|}_1 + \frac{1}{2}(1-\alpha)(w_j)^2$.

See more on <http://web.stanford.edu/~hastie/TALKS/glmnet_webinar.pdf> or
<http://www.princeton.edu/~yc5/ele538_math_data/lectures/model_selection.pdf>.

We can deduce it by Bayesian estimation if we suppose the prior distribution  of the parameters $w$ is in mixture of  Gaussian distribution and Laplacian distribution, i.e.
$$f_{W}(w)\propto  e^{-\alpha{\|w\|}_1-\frac{1}{2}(1-\alpha)\|w\|^{2}}.$$

See **Bayesian lasso regression** at <http://faculty.chicagobooth.edu/workshops/econometrics/past/pdf/asp047v1.pdf>.


### Generalized Linear Model

In **ordinary least squares**, we assume that the errors $\{{\epsilon}_{i}\}_{i=1}^{n}$ are i.i.d. Gaussian, i.e. $\{\epsilon_{i}\} \stackrel{i.i.d}{\sim} N(0,1)$ for $i\in\{1,2, \cdots, n\}$. It is not necessary to assume that they are in Gaussian distribution.
In statistics, the generalized linear model (GLM) is a flexible generalization of ordinary linear regression that allows for response variables that have error distribution models other than a normal distribution. The GLM generalizes linear regression by allowing the linear model to be related to the response variable via a link function and by allowing the magnitude of the variance of each measurement to be a function of its predicted value.

#### Intuition

Ordinary linear regression predicts the expected value of a given unknown quantity (the response variable, a random variable) as a linear combination of a set of observed values (predictors). This implies that a constant change in a predictor leads to a constant change in the response variable (i.e. a linear-response model).
This is appropriate when the response variable has a normal distribution (intuitively, when a response variable can vary essentially indefinitely in either direction with no fixed "zero value", or more generally for any quantity that only varies by a relatively small amount, e.g. human heights).

However, these assumptions are inappropriate for some types of response variables. For example, in cases where the response variable is expected to be always positive and varying over a wide range,
constant input changes lead to geometrically varying, rather than constantly varying, output changes. As an example, a prediction model might predict that 10 degree temperature decrease would lead to 1,000 fewer people visiting the beach is unlikely to generalize well over both small beaches
(e.g. those where the expected attendance was 50 at a particular temperature) and large beaches (e.g. those where the expected attendance was 10,000 at a low temperature).
The problem with this kind of prediction model would imply a temperature drop of 10 degrees would lead to 1,000 fewer people visiting the beach, a beach whose expected attendance was 50 at a higher temperature would now be predicted to have the impossible attendance value of −950. Logically, a more realistic model would instead predict a constant rate of increased beach attendance (e.g. an increase in 10 degrees leads to a doubling in beach attendance, and a drop in 10 degrees leads to a halving in attendance).
Such a model is termed an exponential-response model (or log-linear model, since the logarithm of the response is predicted to vary linearly).

#### Model components

In a generalized linear model (GLM), each outcome $\mathrm{Y}$ of the dependent variables is assumed to be generated from a particular distribution in the **exponential family**, a large range of probability distributions that includes the normal, binomial, Poisson and gamma distributions, among others. The mean, $\mu$, of the distribution depends on the independent variables, $X$, through:
$$
\mathbb{E}(Y)=\mu=g^{-1}(X\beta)
$$
where $\mathbb{E}(Y)$ is the expected value of $\mathrm{Y}$; $X\beta$ is the linear predictor, a linear combination of unknown parameters $\beta$; $g$ is the link function and $g^{-1}$ is the inverse function of $g$.

The GLM consists of three elements:

|Model components|
|:--------------:|
|1. A probability distribution from the exponential family.|
|2. A linear predictor $\eta = X\beta$.|
|3. A link function $g$ such that $\mathbb{E}(Y) = \mu = g^{−1}(\eta)$.|

***

* https://xg1990.com/blog/archives/304
* https://zhuanlan.zhihu.com/p/22876460
* https://www.wikiwand.com/en/Generalized_linear_model
* **Roadmap to  generalized linear models in metacademy** at (https://metacademy.org/graphs/concepts/generalized_linear_models#lfocus=generalized_linear_models)
* [Course in Princeton](http://data.princeton.edu/wws509/notes/)
* [Exponential distribution familiy](https://www.wikiwand.com/en/Exponential_family)
* [14.1 - The General Linear Mixed Model](https://onlinecourses.science.psu.edu/stat502/node/207/)

#### Logistic Regression

The logistic distribution:
$$
y\stackrel{\triangle}=P(Y=1|X=x)=\frac{1}{1+e^{-w^{T}x}}=\frac{e^{w^{T}x}}{1+e^{w^{T}x}}
$$

where $w$ is the parameter vector. Thus, we can obtain:
$$
\log\frac{P(Y=1|X=x)}{P(Y \not= 1|X=x)}=w^{T}x,
$$

i.e.
$$
\log\frac{y}{1-y}=w^{T}x
$$
in theory.


* https://www.wikiwand.com/en/Logistic_regression
* http://www.omidrouhani.com/research/logisticregression/html/logisticregression.htm

### Poisson Regression

* http://www.cnblogs.com/kemaswill/p/3440780.html
* https://www.wikiwand.com/en/Poisson_regression


### Projection pursuit regression

https://www.wikiwand.com/en/Projection_pursuit_regression

### Nonparametric Regression

* https://www.wikiwand.com/en/Additive_model
* https://www.wikiwand.com/en/Generalized_additive_model
* http://iacs-courses.seas.harvard.edu/courses/am207/blog/lecture-20.html
* Nonparametric regression <https://www.wikiwand.com/en/Nonparametric_regression>
* Multilevel model <https://www.wikiwand.com/en/Multilevel_model>
* [A Tutorial on Bayesian Nonparametric Models](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/GershmanBlei2011.pdf)
* Hierarchical generalized linear model <https://www.wikiwand.com/en/Hierarchical_generalized_linear_model>
* http://doingbayesiandataanalysis.blogspot.com/2012/11/shrinkage-in-multi-level-hierarchical.html
* https://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/
* [http://www.stat.cmu.edu/~larry/=sml](http://www.stat.cmu.edu/~larry/=sml/nonpar.pdf)
* [http://www.stat.cmu.edu/~larry/](http://mlg.postech.ac.kr/~seungjin/courses/easyml/handouts/handout07.pdf)
* https://zhuanlan.zhihu.com/p/26830453

***
![https://blogs.ams.org/visualinsight](https://blogs.ams.org/visualinsight/files/2013/10/atomic_singular_inner_function.png)

***

## Machine Learning

The goal of machine learning is to program computers to use example data or past experience to solve a given problem. Many successful applications of machine learning exist already, including systems that analyze past sales data to predict customer behavior, recognize faces or spoken speech, optimize robot behavior so that a task can be completed using minimum resources, and extract knowledge from bioinformatics data.
From [ALPAYDIN, Ethem, 2004. Introduction to Machine Learning. Cambridge, MA: The MIT Press.](https://mitpress.mit.edu/books/introduction-machine-learning).

* http://www.machine-learning.martinsewell.com/
* https://arogozhnikov.github.io/2016/04/28/demonstrations-for-ml-courses.html
* https://www.analyticsvidhya.com/blog/2016/09/most-active-data-scientists-free-books-notebooks-tutorials-on-github/
* https://data-flair.training/blogs/machine-learning-tutorials-home/
* https://github.com/ty4z2008/Qix/blob/master/dl.md
* https://machinelearningmastery.com/category/machine-learning-resources/
* https://machinelearningmastery.com/machine-learning-checklist/  
* https://ml-cheatsheet.readthedocs.io/en/latest/index.html
* https://sgfin.github.io/learning-resources/
* https://cedar.buffalo.edu/~srihari/CSE574/index.html
* http://interpretable.ml/
* https://christophm.github.io/interpretable-ml-book/
* https://csml.princeton.edu/readinggroup
* https://www.seas.harvard.edu/courses/cs281/
* https://developers.google.com/machine-learning/guides/rules-of-ml/

![Road To Data Scientist](http://nirvacana.com/thoughts/wp-content/uploads/2013/07/RoadToDataScientist1.png)

### Dimension Reduction

https://www.wikiwand.com/en/Nonlinear_dimensionality_reduction

### Density Estimation

[Density estimation](https://www.wikiwand.com/en/Density_estimation)
https://www.cs.toronto.edu/~hinton/science.pdf

![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/KernelDensityGaussianAnimated.gif/700px-KernelDensityGaussianAnimated.gif)

### Classification

* https://en.wikipedia.org/wiki/Category:Classification_algorithms

### Clustering

***
There are two different contexts in clustering, depending on how the entities to be clustered are organized.
In some cases one starts from an **internal representation** of each entity (typically an $M$-dimensional vector $x_i$ assigned to entity $i$) and
derives mutual dissimilarities or mutual similarities from the internal representation.In
this case one can derive prototypes (or centroids) for each cluster, for example by averaging the characteristics of the contained entities (the vectors).
In other cases only an **external representation** of dissimilarities is available and the resulting model is an
**undirected and weighted graph** of entities connected by edges. From <https://www.intelligent-optimization.org/LIONbook/>.[^13]
***
The external representation of dissimilarity will be discussed in **Graph Algorithm**.

#### K-means

K-means is also called  **Lloyd’s algorithm**.

#### Hierarchical clustering

+ https://blog.csdn.net/qq_39388410/article/details/78240037
+ http://iss.ices.utexas.edu/?p=projects/galois/benchmarks/agglomerative_clustering
+ https://nlp.stanford.edu/IR-book/html/htmledition/hierarchical-agglomerative-clustering-1.html
+ https://www.wikiwand.com/en/Hierarchical_clustering

![](https://raw.githubusercontent.com/Hulalazz/hierarchical-clustering/master/Results/Centroid.png)

#### DBSCAN

* http://sklearn.apachecn.org/cn/0.19.0/modules/clustering.html
* https://www.wikiwand.com/en/Cluster_analysis
* https://www.toptal.com/machine-learning/clustering-algorithms


### Support Vector Machine

http://www.svms.org/history.html
http://www.svms.org/
http://web.stanford.edu/~hastie/TALKS/svm.pdf
http://bytesizebio.net/2014/02/05/support-vector-machines-explained-well/
![](https://rescdn.mdpi.cn/entropy/entropy-15-00416/article_deploy/html/images/entropy-15-00416-g001.png)


### Kernel Methods

+ [Kernel method at Wikipedia](https://www.wikiwand.com/en/Kernel_method);
+  http://www.kernel-machines.org/
+  http://onlineprediction.net/?n=Main.KernelMethods
+  https://arxiv.org/pdf/math/0701907.pdf

***
![](https://courses.cs.ut.ee/2011/graphmining/uploads/Main/pm.png)