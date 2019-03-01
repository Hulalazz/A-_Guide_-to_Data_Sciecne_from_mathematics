### Generalized Linear Models

In **ordinary least squares**, we assume that the errors $\{ { \epsilon }_{i} \}_{i=1}^{n}$ are i.i.d. Gaussian, i.e. $\{ {\epsilon}_{i}\} \stackrel{i.i.d}{\sim} N(0,1)$ for $i\in\{1,2, \cdots, n\}$. It is not necessary to assume that they are in Gaussian distribution.
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
\mathbb{E}(Y)=\mu=g^{-1}(X^{T}\beta)
$$
where $\mathbb{E}(Y)$ is the expected value of $\mathrm{Y}$; $X^{T}\beta$ is the linear predictor, a linear combination of unknown parameters $\beta$; ${g}$ is the link function and $g^{-1}$ is the inverse function of $g$.

The GLM consists of three elements:

|Model components|
|:---------------|
|1. A probability distribution from the exponential family.|
|2. A linear predictor $\eta = X^{T}\beta$.|
|3. A link function $g$ such that $\mathbb{E}(Y) = \mu = g^{−1}(\eta)$.|

***

* https://xg1990.com/blog/archives/304
* https://zhuanlan.zhihu.com/p/22876460
* https://www.wikiwand.com/en/Generalized_linear_model
* **Roadmap to  generalized linear models in metacademy** at (https://metacademy.org/graphs/concepts/generalized_linear_models#lfocus=generalized_linear_models)
* [Course in Princeton](http://data.princeton.edu/wws509/notes/)
* [Exponential distribution familiy](https://www.wikiwand.com/en/Exponential_family)
* [14.1 - The General Linear Mixed Model](https://onlinecourses.science.psu.edu/stat502/node/207/)
* https://scikit-learn.org/stable/modules/linear_model.html#
* Multilevel model <https://www.wikiwand.com/en/Multilevel_model>
* Hierarchical generalized linear model <https://www.wikiwand.com/en/Hierarchical_generalized_linear_model>
* http://bactra.org/notebooks/regression.html

#### Exponential Families

An exponential family distribution has the following form
$$
p(x|\eta)=h(x) \exp(\eta^{T}t(x) - a(\eta))
$$
where

* a parameter vector $\eta$ is often referred to as the canonical or natural parameter;
* the statistic $t(X)$ is referred to as a sufficient statistic;
* the underlying measure $h(x)$ is a counting measure or Lebesgue measure;
* the log normalizer $a(\eta)=\log \int h(x) \exp(\eta^{T}t(x)) \mathrm{d}x$

For example, a Bernoulli random variable $X$, which assigns probability measure $\pi$ to the point $x = 1$ and
probability measure $1 − \pi$ to $x = 0$, can be rewritten in
$$
\begin{align}
p(x|\eta)
    & = {\pi}^{x}(1-\pi)^{1-x}\\
    & = \{\frac{\pi}{1-\pi}\}^{x} (1-\pi)\\
    & = \exp\{\log(\frac{\pi}{1-\pi})x+\log (1-\pi)\}
\end{align}
$$
so that $\eta=\log(\frac{\pi}{1-\pi})$.
***
The Poisson  distribution can be written in
$$
\begin{align}
p(x|\lambda)
     & =\frac{{\lambda}^{x}e^{-\lambda}}{x!} \\
     & = \frac{1}{x!}exp\{x\log(\lambda)-\lambda\}
\end{align}
$$
for $x=\{0,1,2,\dots\}$, such that

* $\eta=\log(\lambda)$;
* $t(X)=X$;
* $h(x)=\frac{1}{x!}$;
* $a(\eta)=\lambda=\exp(\eta)$.

***
The multinomial distribution can be written in
$$
p(x|\pi)
  =\frac{M!}{{x}_{1}!{x}_{2}!\cdots {x}_{K}!}{\pi}_{1}^{x_1}{\pi}_{2}^{x_2}\cdots {\pi}_{K}^{x_K}  \\
  = \frac{M!}{{x}_{1}!{x}_{2}!\cdots {x}_{K}!}\exp\{\sum_{i=1}^{K}x_i\log(\pi_i)\} \\
  = \frac{M!}{{x}_{1}!{x}_{2}!\cdots {x}_{K}!}\exp\{\sum_{i=1}^{K-1}x_i\log(\pi_i)+(M-\sum_{i=1}^{K-1}x_i) \log(1-\sum_{i=1}^{K-1} \pi_i)\} \\
  = \frac{M!}{{x}_{1}!{x}_{2}!\cdots {x}_{K}!}\exp\{\sum_{i=1}^{K-1}x_i\log(\frac{\pi_i}{1-\sum_{i=1}^{K-1} \pi_i}) + M \log(1-\sum_{i=1}^{K-1} \pi_i)\}
$$

where $\sum_{i=1}^{K}x_i=M$ and $\sum_{i=1}^{K} \pi_i = 1$,
such that

* $\eta_k =\log(\frac{\pi_k}{1-\sum_{i=1}^{K-1} \pi_i})=\log(\frac{\pi_k}{\pi_K})$ then $\pi_k = \frac{e^{\eta_k}}{\sum_{k=1}^{K}e^{\eta_k}}$ for $k\in\{1,2,\dots, K\}$ with $\eta_K = 0$;
* $t(X)=X=(X_1, X_2, \dots, X_{K-1})$;
* $h(x)=\frac{M!}{{x}_{1}!{x}_{2}!\cdots {x}_{K}!}$;
* $a(\eta)=-M \log(1-\sum_{i=1}^{K-1} \pi_i)=-M \log(\pi_K)$.

Note that $\eta_K = 0$ then $\pi_K = \frac{e^{\eta_K}}{\sum_{k=1}^{K}e^{\eta_k}} = \frac{1}{\sum_{k=1}^{K}e^{\eta_k}}$ and $a(\eta)=-M\log(\pi_K)=M\log(\sum_{k=1}^{K}e^{\eta_k})$.

* https://en.wikipedia.org/wiki/Exponential_family
* https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/exponential-families.pdf
* https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf

#### Logistic Regression

|Logistic Regression|
|:------------------|
|1. A Bernoulli random variable $Y$ is from the exponential family.|
|2. A linear predictor $\eta = \log(\frac{\pi}{1-\pi}) = X^{T}\beta$.|
|3. A link function $g$ such that $\mathbb{E}(Y) = \pi = g^{−1}(\eta)$, where $g^{-1}=\frac{1}{1+e^{-\eta}}$.|

The logistic distribution:
$$
\pi\stackrel{\triangle}=P(Y=1|X=x)=\frac{1}{1+e^{-x^{T}\beta}}=\frac{e^{x^{T}\beta}}{1+e^{x^{T}\beta}}
$$

where $w$ is the parameter vector. Thus, we can obtain:
$$
\log\frac{P(Y=1|X=x)}{P(Y = 0|X=x)} = x^{T}\beta,
$$

i.e. $\log\frac{\pi}{1-\pi}=x^{T}\beta$

where $\pi\in (0,1)$ and $x\in \mathbb{R}^{d+1}$ is an $(d + 1)$ - dimensional vector consisting of $d$ independent variables concatenated to a vector of ones in theory.

How we can do estimate the parameters $\beta$ ?
Because logistic regression predicts probabilities, rather than just classes, we can fit it
using likelihood.
The likelihood function for logistic regression is
$$
\begin{align}
   L(\beta)
     & ={\prod}_{i=1}^{n}{\pi}^{y_i}(1-\pi)^{1-y_i} \\
     & ={\prod}_{i=1}^{n}(\frac{\pi}{1-\pi})^{y_i}(1-\pi) \\
     & = {\prod}_{i=1}^{n} {\exp(x_i^{T}\beta)}^{y_i}\frac{1}{1 + \exp({x_i^{T}\beta})}.
\end{align}
$$


The log-likelihood turns products into sums
$$
\begin{align}
  \ell(\beta)
     & ={\sum}_{i=1}^{n}{y}_{i}\log(\pi)+(1-{y}_i)\log(1-\pi) \\
     & = {\sum}_{i=1}^{n}{y}_{i}\log(\frac{\pi}{1-\pi})+\log(1-\pi) \\
     & = {\sum}_{i=1}^{n}{y}_{i}(x_i^T\beta)-\log(1+\exp(x_i^T\beta))
\end{align}
$$

where ${(x_i,y_i)}_{i=1}^{n}$ is the input data set and $\eta$ is the parameter to learn.
We want to find the MLE of $\beta$, i.e.

$$
\begin{align}
  \hat{\beta}
     & = \arg\max_{\beta} L(\beta)=\arg\max_{\beta} \ell(\beta) \\
     & = \arg\max_{\beta} {\sum}_{i=1}^{n}{y}_{i}(x_i^T\beta)-\log(1+\exp(x_i^T\beta))
\end{align}
$$

which we can solve this optimization by numerical optimization methods such as Newton's method.

$$\arg\max_{\beta} L(\beta)={\sum}_{i=1}^{n}{y}_{i}\log(\pi)+(1-{y}_i)\log(1-\pi) $$

* https://www.wikiwand.com/en/Logistic_regression
* https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf
* https://machinelearningmastery.com/logistic-regression-for-machine-learning/
* http://www.omidrouhani.com/research/logisticregression/html/logisticregression.htm

#### Poisson Regression

Poisson regression assumes the response variable $Y$ has a Poisson distribution, and assumes the logarithm of its expected value can be modeled by a linear combination of unknown parameters. A Poisson regression model is sometimes known as a log-linear model, especially when used to model contingency tables.

|Poisson Regression|
|:-----------------|
|1. A Poisson random variable $Y$ is  from the exponential family.|
|2. A linear predictor $\eta = \log(\lambda) = X^{T}\beta$.|
|3. A link function $g$ such that $\mathbb{E}(Y) = \lambda = g^{−1}(\eta)$, where $g^{-1}(\eta) = \exp(\eta)$.|

Thus we obtain that
$\mathbb{E}(Y) = \lambda = g^{−1}(\eta)  = \exp({x^{T}\beta})$, $x\in \mathbb{R}^{d+1}$ is an $(d + 1)$ - dimensional vector consisting of $d$ independent variables concatenated to a vector of ones.
The likelihood function in terms of $\beta$ is
$$
\begin{align}
   L(\beta|X, Y)
     & = {\prod}_{i=1}^{n}\frac{\lambda^{y_i}e^{-\lambda}}{y_i!} \\
     & = {\prod}_{i=1}^{n} \frac{\exp(y_i x_i^T\beta)\exp[-\exp(x_i^T\beta)]}{y_i!}.
\end{align}
$$
The log-likelihood is

$$
\begin{align}
   \ell(\beta|X, Y)
     & = {\sum}_{i=1}^{n}[y_i x_i^T\beta -\exp(x_i^T\beta)-\log(y_i!)] \\
     & \propto {\sum}_{i=1}^{n} [y_i x_i^T\beta -\exp(x_i^T\beta)].
\end{align}
$$

The negative log-likelihood function $-\ell(\beta|X, Y)$ is convex so that we can apply standard convex optimization techniques such as gradient descent to find the optimal value of $\beta$.

* http://www.cnblogs.com/kemaswill/p/3440780.html
* https://www.wikiwand.com/en/Poisson_regression

#### Softmax Logistic Regression

|Softmax Regression|
|:-----------------|
|1. A multinomial random variable $Y=(Y_1, Y_2, \dots, Y_{K})^{T}$ is  from the exponential family.|
|2. A linear predictor ${\eta}_k = \log(\frac{{\pi}_k}{{\pi}_K}) = X^{T}{\beta}_k$.|
|3. A link function $g$ such that $\mathbb{E}(Y_k) = {\pi}_k = \frac{e^{\eta_k}}{\sum_{j=1}^{K}e^{\eta_j}}$.|

The log-likelihood in terms of $\beta = (\beta_1,\dots,\beta_K)^{T}$ is
$$
\begin{align}
\ell(\beta|X, Y)
     & = {\sum}_{i=1}^{n} {\sum}_{k=1}^{K} y_k^{(i)}\log(\pi_k) \\
     & = {\sum}_{i=1}^{n} [{\sum}_{k=1}^{K-1} y_k^{(i)}\log(\pi_k) + y_K\log(\pi_K)] \\
     & = {\sum}_{i=1}^{n} [{\sum}_{k=1}^{K-1} y_k^{(i)}\log(\pi_k) + (1-{\sum}_{k=1}^{K-1} y_k))\log(\pi_K)].
\end{align}
$$

* http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92
* https://www.wikiwand.com/en/Multinomial_logistic_regression
* https://www.wikiwand.com/en/Softmax_function
* https://metacademy.org/graphs/concepts/multinomial_logistic_regression

### Robust Regression

Generalized linear models extend the distribution of outputs so that the loss function(the likelihood function or log-likelihood function).

It is supposed that there is a link function $g$ such that $\mathbb{E}(Y) = \pi = g^{−1}(\eta)$. However, why it is the expectation $\mathbb{E}(Y)$ rather than others?

Robust regression can be used in any situation in which you would use least squares regression. When fitting a least squares regression, we might find some outliers or high leverage data points. We have decided that these data points are not data entry errors, neither they are from a different population than most of our data. So we have no compelling reason to exclude them from the analysis. Robust regression might be a good strategy since it is a compromise between excluding these points entirely from the analysis and including all the data points and treating all them equally in OLS regression. The idea of robust regression is to weigh the observations differently based on how well behaved these observations are. Roughly speaking, it is a form of weighted and reweighted least squares regression. Draw form[https://stats.idre.ucla.edu/r/dae/robust-regression/].

![](https://ww2.mathworks.cn/help/examples/stats/win64/CompareRobustAndLeastSquaresRegressionExample_01.png)

**M-estimators** are given by
$$\hat{\beta}_M=\arg\min_{\beta}{\sum}_{i}\rho({\epsilon}_i)$$
where  error term as ${\epsilon}_i(\beta)$ is to reflect the error term's dependency on the regression coefficients $\beta$ and  function $\rho(\cdot)$  is chosen which is acting on the residuals.
A reasonable $\rho$ should have the following properties

* Nonnegative, $\rho(e)\geq 0 \,\,\forall e$;
* Equal to zero when its argument is zero, $\rho(0)=0$;
* Symmetric, $\rho(e)=\rho(-e)$;
* Monotone in $|e|$, $\rho(e_1)\geq \rho(e_2)$ if $|e_1|>|e_2|$.

For example, $\rho$ can be absolute value function.

|Method|Loss function|
|:---|:---:|---:|
|Least-Squares|$\rho_{LS}(e)=e^2$|
|Huber | [Huber function](https://blog.csdn.net/lanchunhui/article/details/50427055)
|Bisquare|[mathworld](http://mathworld.wolfram.com/TukeysBiweight.html)|
|Winsorizing|[Wiki page](https://www.wikiwand.com/en/Winsorizing)|

![](https://upload.wikimedia.org/wikipedia/commons/c/c1/RhoFunctions.png)

Three common functions chosen in M-estimation are given in [Robust Regression Methods](https://newonlinecourses.science.psu.edu/stat501/node/353/).

And it is really close to supervised machine learning.

* https://stats.idre.ucla.edu/r/dae/robust-regression/
* http://users.stat.umn.edu/~sandy/courses/8053/handouts/robust.pdf
* https://newonlinecourses.science.psu.edu/stat501/node/351/
* https://stats.idre.ucla.edu/r/dae/robust-regression/
* https://www.r-bloggers.com/visual-contrast-of-two-robust-regression-methods/
* https://newonlinecourses.science.psu.edu/stat501/node/353/
* https://projecteuclid.org/euclid.aos/1534492823
* https://orfe.princeton.edu/~jqfan/papers/14/Robust14.pdf

Another application of design loss function is in feature selection or regularization such as **LASSO, SCAD**.

* https://econpapers.repec.org/article/besjnlasa/v_3a96_3ay_3a2001_3am_3adecember_3ap_3a1348-1360.htm

### Nonparametric Regression

In generalized linear model, the response $Y$ is specified its distribution family. In linear or generalized linear model, we must specify the distribution family of output or error such as Gaussian distribution in linear regression or Poisson distribution in Poisson regression.
It is not required to specify the distribution of the error as in ordinary  linear regression.

The assumption for the straight-line model is:
1. Our straight-line model is
$$
Y_i = \alpha + \beta x_i + e_i, i=1,2,\dots, n
$$
   where distinct  $x_i$ is known observed constants and $\alpha$ and $\beta$ are unknown parameters to estimate.

2. The random variables $e_i$ are random sample from a continuous population that has median 0.

Then we would make a  null hypothesis
$$H_0: \beta = \beta_0$$

 and Theil construct a statistics ${C}$:
 $$
 C = \sum_{i=1}^{n-1}\sum_{j=i+1}^{n} c(D_j-D_i)
 $$

 where $D_i = Y_i -\beta_0 x_i$,  $x_1<x_2\cdots <x_n$, and
 $$
 c(a)=
    \begin{cases}
     1, & \text{if $a>0$} \\
     0, & \text{if $a=0$} \\
    -1, & \text{if $a<0$}
    \end{cases}.
 $$

For different  alternative  hypothesis, we can test it via comparing with the p-values.

* https://jvns.ca/blog/2018/12/29/some-initial-nonparametric-statistics-notes/
* https://www.wikiwand.com/en/Additive_model
* https://www.wikiwand.com/en/Generalized_additive_model
* http://iacs-courses.seas.harvard.edu/courses/am207/blog/lecture-20.html
* Nonparametric regression <https://www.wikiwand.com/en/Nonparametric_regression>
* [http://www.stat.cmu.edu/~larry/=sml](http://www.stat.cmu.edu/~larry/=sml/nonpar.pdf)
* [http://www.stat.cmu.edu/~larry/](http://mlg.postech.ac.kr/~seungjin/courses/easyml/handouts/handout07.pdf)
* https://zhuanlan.zhihu.com/p/26830453
* http://wwwf.imperial.ac.uk/~bm508/teaching/AppStats/Lecture7.pdf
* http://web.stanford.edu/class/ee378a/books/book2.pdf

#### Bayesian Nonparametric Models

Bayesian nonparametric (BNP) approach is to fit a single model that can adapt its complexity
to the data. Furthermore, BNP models allow the complexity to grow as more
data are observed, such as when using a model to perform prediction.

Please remind the Bayesian formula:
$$
P(A|B)=\frac{P(B|A)P(A)}{P(B)}\\
P(A|B)\propto P(B|A)P(A)
$$

and in Bayesian everything can be measured in the belief degree in $[0, 1]$.

* [A Tutorial on Bayesian Nonparametric Models](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/GershmanBlei2011.pdf)
* https://www.quantstart.com/articles/Bayesian-Linear-Regression-Models-with-PyMC3
* http://doingbayesiandataanalysis.blogspot.com/2012/11/shrinkage-in-multi-level-hierarchical.html
* https://twiecki.github.io/blog/2013/08/12/bayesian-glms-1/
* https://twiecki.github.io/blog/2013/08/27/bayesian-glms-2/
* https://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/
* https://blog.applied.ai/bayesian-inference-with-pymc3-part-1/
* https://blog.applied.ai/bayesian-inference-with-pymc3-part-2/
* https://blog.applied.ai/bayesian-inference-with-pymc3-part-3/
* https://magesblog.com/post/2016-02-02-first-bayesian-mixer-meeting-in-london/
* https://www.meetup.com/Bayesian-Mixer-London/
* https://sites.google.com/site/doingbayesiandataanalysis/

### Smoothing Splines and Generalized Additive Model

Given samples $(x_i, y_i), i = 1, \cdots, n$, we can consider estimating the regression function
$r(x) = E(Y |X = x)$ by fitting a kth order `spline` with knots at some prespecified locations $t_1 < t_2 <\cdots < t_m$.
We want the function ${f}$ in the model $Y = f(X) + \epsilon$ where $f$ is the spline function.
The coefficients of the spline functions are just estimated by least squares:
$$\sum_{i=1}^{n}[y_i - f(x_i)]^2.$$

The regularization technique can be applied to control the model complexity
$$\sum_{i=1}^{n}[y_i - f(x_i)]^2 + \lambda\int \{f^{\prime\prime}(t)\}^2\mathrm{d}t$$
where $\lambda$ is a fixed smoothing parameter.
It is different from LASSSO or ridge regression that we constrain the smoothness of the function rather than the norm of coefficients.

It can date back to  spline interpolation in computational or numerical analysis. There may be some points such as $(x_1, y_1)$ and $(x_2, y_2)$ that $x_1=x_2$ while $y_1\not= y_2$ in smoothing splines.
And it is also different from `ploynomial regression`:
$$Y = \beta_0 +\beta_1 X +\beta_2 X^2+\cdots+\beta_h X^{h}+\epsilon$$
where ${h}$ is called the degree of the polynomial. 

* http://www.stat.cmu.edu/~ryantibs/advmethods/notes/smoothspline.pdf
* https://web.stanford.edu/class/stats202/content/lec17.pdf
* https://robjhyndman.com/etc5410/splines.pdf
* https://newonlinecourses.science.psu.edu/stat501/node/324/

### Projection pursuit regression

There is a section of **projection pursuit** and **neural networks** in **Element of Statistical Learning**.
Assume we have an input vector ${X}$ with ${p}$ components, and a target ${Y}$, the projection pursuit regression (PPR) model has the form:

$$
f(X) = \sum_{m=1}^{M} g_m ( \left<\omega_m, X \right>)
$$

where each parameter $\omega_m$ is unit ${p}$-vectors. The functions ${g_m}$ are unspecified and is called `projection indices`.

The function $g_m ( \left<\omega_m, X \right>)$ is called a ridge function in $\mathbb{R}^p$.  It varies only
in the direction defined by the vector $\omega_m$. The scalar variable $V_m = \left<\omega_m, X \right>$
is the projection of ${X}$ onto the unit vector $\omega_m$, and we seek $\omega_m$ so that the model fits well, hence the name "projection pursuit".

The $M = 1$ model is known as the `single index model` in econometrics.

We seek the approximate minimizers of the error function
$$
\sum_{n=1}^{N}[y_n - \sum_{m=1}^{M} g_m ( \left<\omega_m, x_i \right>)]^2
$$

over functions $g_m$ and direction vectors $\omega_m, m = 1, 2,\dots ,M$, which is different from other algorithms in statistics.

And there is a PHD thesis [Design and choice of projection indices](https://people.maths.bris.ac.uk/~magpn/Research/PP/phdthesis.pdf) in 1992.

* https://www.wikiwand.com/en/Projection_pursuit_regression
* https://projecteuclid.org/euclid.aos/1176349519
* https://projecteuclid.org/euclid.aos/1176349520
* https://projecteuclid.org/euclid.aos/1176349535
* https://people.maths.bris.ac.uk/~magpn/Research/PP/PP.html
* https://www.wikiwand.com/en/Projection_pursuit_regression
* http://www.stat.cmu.edu/~larry/=stat401/
* http://cis.legacy.ics.tkk.fi/aapo/papers/IJCNN99_tutorialweb/node23.html
* https://www.pnas.org/content/115/37/9151
* https://www.ncbi.nlm.nih.gov/pubmed/30150379


***
![https://blogs.ams.org/visualinsight](https://blogs.ams.org/visualinsight/files/2013/10/atomic_singular_inner_function.png)
