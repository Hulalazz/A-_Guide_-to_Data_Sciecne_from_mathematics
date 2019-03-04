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
\arg\min_{w}\sum_{i=1}^{n}(y_i-w^Tx_i)^2 \qquad\text{objective function} \\
 \text{subject to}\,{\|w\|}_1 \leq t     \qquad\text{constraint}.
$$

3. the selection form:
$$
 \arg\min_{w}{\|w\|}_1                                  \qquad \text{objective function} \\
 \text{subject to}\,\sum_{i=1}^{n}(y_i-w^Tx_i)^2 \leq t \qquad \text{constraint}.
$$
where ${\|w\|}_1=\sum_{i=1}^{n}|w_i|$ if $w=(w_1,w_2,\cdots, w_n)^{T}$.


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

#### FISTA

- [ ]  http://www.seas.ucla.edu/~vandenbe/236C/lectures/fista.pdf
- [ ]  https://github.com/tiepvupsu/FISTA
- [ ]  https://www.math.ucdavis.edu/~sqma/MAT258A_Files/FISTA.pdf

More solutions to this optimization problem:

* [Q&A in zhihu.com](https://www.zhihu.com/question/22332436/answer/21068494);
* [LASSO using ADMM](http://www.simonlucey.com/lasso-using-admm/);
* [Regularization: Ridge Regression and the LASSO](http://statweb.stanford.edu/~tibs/sta305files/Rudyregularization.pdf);
* [Least angle regression](https://www.wikiwand.com/en/Least-angle_regression);
* http://web.stanford.edu/~hastie/StatLearnSparsity/
* [历史的角度来看，Robert Tibshirani 的 Lasso 到底是不是革命性的创新？- 若羽的回答 - 知乎](https://www.zhihu.com/question/275196908/answer/533790835)
* http://www.cnblogs.com/xingshansi/p/6890048.html
* https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf
* http://www.stat.cmu.edu/~ryantibs/papers/genlasso.pdfv


Generally, we may consider the penality residual squared sum
$$
PRSS_{\ell_p}=\sum_{i=1}^{n}(y_i-w^Tx_i)^2+\lambda{\|w\|}_{p}^{p} =\|Y-Xw\|^2+\lambda{\|w\|}_p^{p}\,\tag {LASSO}
$$
where ${\|w\|}_{p}=(\sum_{i=1}^{n}|w_i|^{p})^{1/p}, p\in (0,1)$.

[Dr. Zongben Xu](http://gr.xjtu.edu.cn/web/zbxucn/) proposed the $L_{1/2}$ regularization where $p=\frac{1}{2}$.

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

### Variable Selection: $p\gg N$

In this chapter we discuss prediction problems in which the number of
features ${p}$ is much larger than the number of observations ${N}$, often written
$p\gg N$. Such problems have become of increasing importance, especially in genomics and other areas of computational biology.

The outcome ${y}$ was generated according to a linear model of the feature vectir $x$:
$$
y = \left<w, x\right> +\sigma\epsilon
$$

where $w, x\in\mathbb{R}^p$,$\epsilon$ was generated from a standard Gaussian distribution.

Suppose  $Y=(y_1, y_2, \dots, y_N)^T\in\mathbb{R}^N$, $X=(x_1, x_2, \dots, x_N)^T\in\mathbb{R}^{P\times N}$, the linear equation system 
$Y=Xw$ is not posed-well because $p\gg N$. The regularization technique is necessary to solve this question if we do not select some important independent variables.
More generally, it is one part of the engineering - feature selection.


* http://www.biostat.jhsph.edu/~iruczins/teaching/jf/ch10.pdf
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5969114/
* https://newonlinecourses.science.psu.edu/stat508/