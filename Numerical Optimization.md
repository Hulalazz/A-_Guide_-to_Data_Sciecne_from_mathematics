# Numerical Optimization

IN [A Few Useful Things to Know about Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf), Pedro Domingos put up a relation:
$\color{aqua}{LEARNING}$ = $\color{green}{REPRESENTATION}$ + $\color{blue}{EVALUATION}$ + $\color{red}{OPTIMIZATION}$.

* Representation as the core of the note is the general (mathematical) **model** that computer can handle.
* Evaluation is  **criteria**. An evaluation function (also called objective function, cost function or scoring function) is needed to distinguish good classifiers from bad ones.
* Optimization is aimed to find the parameters that optimizes the evaluation function, i.e.
    $$
    \arg\min_{\theta\in \Theta} f(\theta)=\{\theta^{\ast}|f(\theta^{\ast})=\min f(\theta)\}\,\text{or}
    \\ \quad\arg\max_{\theta\in \Theta}f(\theta)=\{\theta^{\ast}|f(\theta^{\ast})=\max f(\theta)\}.
    $$


<img title="http://art.ifeng.com/2015/1116/2606232.shtml" src="http://upload.art.ifeng.com/2015/1116/1447668349594.jpg" width="70%"/>

The objective function to be minimized is also called cost function.

Evaluation is always attached with optimization; the evaluation which cannot be optimized is not a good evaluation in machine learning.

* https://www.wikiwand.com/en/Mathematical_optimization
* https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
* http://www.cs.cmu.edu/~pradeepr/convexopt/
* [An interactive tutorial to numerical optimization](https://www.benfrederickson.com/numerical-optimization/)
* [Patrick Louis' RECENT CONFERENCE TALKS  on optimization](https://pcombet.math.ncsu.edu/confab.html)
* [Northwestern University Open Text Book on Process Optimization](https://optimization.mccormick.northwestern.edu/index.php/Main_Page)
* [Introductory course on non-smooth optimisation](https://jliang993.github.io/nsopt.html)
* [CS4540: Simple Algorithms](https://algorithms2017.wordpress.com/)
* [CS 798: Convexity and Optimization](https://cs.uwaterloo.ca/~lapchi/cs798/index.html)
* [Survival Guide for Students of Optimization, Dianne P. O'Leary, September 2017](http://www.cs.umd.edu/users/oleary/survivalo.html)
* https://nlopt.readthedocs.io/en/latest/

The proof of convergence  or complexity is often based  on the convex cases where the objective function as well as the constrained set is convex, i.e.,
$$t x+(1-t)y\in\Theta,\\
f(t x+(1-t)y)\leq t f(x)+(1-t)f(y),\\
\quad t\in [0,1], \quad \forall x, y\in\Theta.$$

And this optimization is called convex optimization.
By the  way, the name `numerical optimization` means the variables or parameters to estimate are  in numeric format, which is far from performance optimization in concept.

**First optimal condition** is a necessary condition for the unconstrainted optimziation problems if the objective function is differentiable: if $\nabla f$ exists, and $x^{\ast}\in\arg\min_{x} f(x)$, we have $\nabla f(x^{\ast})=0$.

***

Wotao Yin wrote a summary on [First-order methods and operator splitting for optimization](http://www.math.ucla.edu/~wotaoyin/research.html):
> First-order methods are described and analyzed with gradients or subgradients, while second-order methods use second-order derivatives or their approximations.

> During the 70s–90s the majority of the optimization community focused on second-order methods since they are more efficient for those problems that have the sizes of interest at that time. Beginning around fifteen years ago, however, the demand to solve ever larger problems started growing very quickly. Many large problems are further complicated by non-differentiable functions and constraints. Because simple first-order and classic second-order methods are ineffective or infeasible for these problems, operator splitting methods regained their popularity.

> Operators are used to develop algorithms and analyze them for a wide spectrum of problems including optimization problems, variational inequalities, and differential equations. Operator splitting is a set of ideas that generate algorithms through decomposing a problem that is too difficult as a whole into two or more smaller and simpler subproblems. During the decomposition, complicated structures like non-differentiable functions, constraint sets, and distributed problem structures end up in different subproblems and thus can be handled elegantly. We believe ideas from operator splitting provide the most eﬀective means to handle such complicated structures for computational problem sizes of modern interest.

* [The world of optimization](http://awibisono.github.io/2016/06/06/world-of-optimization.html)
* [Gradient flow and gradient descent](http://awibisono.github.io/2016/06/13/gradient-flow-gradient-descent.html)
* [Accelerated gradient descent](http://awibisono.github.io/2016/06/20/accelerated-gradient-descent.html)
* [Accelerated gradient flow](http://awibisono.github.io/2016/06/27/accelerated-gradient-flow.html)
* [Stochastic gradient flow](http://awibisono.github.io/2016/09/05/stochastic-gradient-flow.html)
* [Aristotle vs. Newton](http://awibisono.github.io/2016/07/04/aristotle-newton.html)
* [Advanced non-smooth optimization](https://mathematical-coffees.github.io/mc12-advanced-optim/)
* [The 2014 International Workshop on Signal Processing, Optimization, and Control (SPOC 2014)](http://spoc2014.nudt.edu.cn/index.html)

## Gradient Descent and More

Each iteration of a line search method computes a search direction $p^{k}$ and then decides how
far to move along that direction. The iteration is given by

$$
x^{k+1}=x^{k}+\alpha_{k}p^{k}\tag{Line search}
$$

where the positive scalar $\alpha_{k}$ is called the step length. The success of a line search method
depends on effective choices of both the direction $p^{k}$ and the step length $\alpha_{k}$.

$\color{lime}{Note}$: we use the notation $x^{k}$ and $\alpha_k$ to represent the $k$th iteration of the vector variables $x$ and $k$ th step length, respectively.
Most line search algorithms require $p^k$ to be a descent direction — one for which
$\left< {p^k},\nabla f_k \right> < 0$ — because this property guarantees that the function $f$ can be reduced along this direction, where $\nabla f_k$ is the gradient of objective function $f$ at the $k$th iteration point $x_k$ i.e. $\nabla f_k=\nabla f(x^{k})$.

***

Gradient descent and its variants are to find the local solution of  the unconstrained optimization problem:

$$
\min_{x} f(x)
$$

where $x\in \mathbb{R}^{n}$.

It is not difficult to observe that
$$f(x) \approx f(x^k)+(x - x^k)^{T}\nabla f(x^{k}) + \frac{s_k}{2}{\|x-x^k\|}_2^2$$
by Taylor expansion of $f$ near the point $x^{k}$ for some constant $s_k$.

Let $x^{k+1}=\arg\min_{x} f(x^k) + (x - x^k)^{T}\nabla f(x^{k}) + \frac{s_k}{2}{\|x-x^k\|}_2^2$,  we will obtain $x^{k+1} = x^{k}-\frac{1}{s_k}{\nabla}_{x} f(x^k)$. However, the constant $s_k$ is difficult to estimate.

And the general gradient descent methods are given by

$$
x^{k+1}=x^{k}-\alpha_{k}{\nabla}_{x} f(x^k)
$$

where $x^{k}$ is the $k$th iterative result, $\alpha_{k}\in\{\alpha|f(x^{k+1})< f(x^{k})\}$ and particularly $\alpha_{k}=\arg\min_{\alpha}f(x^{k}-\alpha\nabla_{x}f(x^{k}))$ so that $f(x^{k+1})=\min_{\alpha} f(x^k - \alpha\nabla_x f(x^k))$.

$$
x^{k+1}=\fbox{$x^{k}$}-\alpha_{k}{\nabla}_{x} f(x^k)
= \fbox{$x^{k-1}-\alpha_{k-1}\nabla f(x_{k-1})$}-\alpha_{k}{\nabla}_{x} f(x^k)\\
= x^1-\sum_{n=0}^{k}\alpha_n{\nabla}_{x} f(x^n)
$$

- http://59.80.44.100/www.seas.ucla.edu/~vandenbe/236C/lectures/gradient.pdf
- http://wiki.fast.ai/index.php/Gradient_Descent


<img src="https://www.fromthegenesis.com/wp-content/uploads/2018/06/Gradie_Desce.jpg" width=50%>

There are many ways to choose some proper step or learning rate sequence $\{\alpha_k\}$.
***
The first-order Taylor approximation of $f(x + v)$ around ${x}$ is
$$f(x+v)\approx \hat{f}(x+v)=f(x)+\nabla_x f(x)^T v.$$

The second term on the righthand side, $\nabla f(x)^T v$, is the directional derivative of ${f}$ at ${x}$ in the direction ${v}$. It gives the approximate change in ${f}$ for a small step ${v}$.
The step ${v}$ is a descent direction if the directional derivative is negative. Since the directional derivative $\nabla_x f(x)^T v$ is linear in
${v}$, it can be made as negative as we like by taking ${v}$ large (provided ${v}$ is a descent
direction, i.e., $\nabla_x f(x)^T v< 0$). To make the question sensible we have to limit the
size of ${v}$, or normalize by the length of ${v}$.

We define a normalized steepest descent direction (with respect to the norm $\|\cdot \|$ in $\mathbb{R}^n$) as

$$\Delta x_{nsd}=\arg\min_{v}\{f(x)^T v\mid \|v\|=1\}.$$

It is also convenient to consider a steepest descent step $\Delta x_{sd}$ that is unnormalized,
by scaling the normalized steepest descent direction in a particular way:

$$\Delta x_{sd}={\|\nabla f(x)\|}_{\ast}\Delta x_{nsd}$$

where ${\| \cdot \|}_{\ast}$ denotes the dual norm.

(We say ‘a’ steepest descent direction because there can be multiple minimizers.)

***
**Algorithm  Steepest descent method**
* given a starting point $x \in domf$.
* repeat
   1. Compute steepest descent direction $\Delta x_{sd}$.
   2. Line search. Choose ${t}$ via backtracking or exact line search.   3. Update. $x := x + t \Delta x_{sd}$.
* until stopping criterion is satisfied.

If the variable ${x}$ is restricted in some bounded domain, i.e., $x\in D$, the steepest gradient descemt methods can be modified to `conditional gradient descent method` or `Frank-Wolfe algorithm`.
***
**Algorithm  Frank–Wolfe algorithm**
> * given a starting point $x \in domf$.
> * repeat
>   1. Find $s^k$: $s^k =\arg\min_{v}\{f(x^{k})^T v\mid v\in D \}$.
>   2. Set $t=\frac{2}{k+2}$ or $t=\arg\min\{f(x^k+t(s^k - x^k))\mid t\in [0, 1]\}$
>   3. Update. $x^{k+1}= x^k + t(s^k-x^k), k\leftarrow k+1$.
> * until stopping criterion is satisfied.

<img src="http://m8j.net/data/Website-Images/3d-FW-blue-small.jpg" width="40%"/>

* [梯度下降法和最速下降法的细微差别](https://blog.csdn.net/Timingspace/article/details/50963564)
* [An Introduction to Conditional Gradient](http://www.cs.cmu.edu/~yaoliang/mytalks/condgrad.pdf)
* [Frank–Wolfe algorithm ](https://www.wikiwand.com/en/Frank%E2%80%93Wolfe_algorithm)
* [Greedy Algorithms, Frank-Wolfe and Friends - A modern perspective](http://www.cmap.polytechnique.fr/~jaggi/NIPS-workshop-FW-greedy/)
* https://sites.google.com/site/nips13greedyfrankwolfe/
* [Revisiting Frank-Wolfe](http://m8j.net/(All)Revisiting%20Frank-Wolfe)
* [Nonsmooth Frank-Wolfe algorithm and Coresets](https://sravi-uwmadison.github.io/2017/08/21/corensfw/)

***

Some variants of gradient descent methods are not line search method.
For example, the **heavy ball methods or momentum methods**:

$$
x^{k+1}=x^{k}-\alpha_{k}\nabla_{x}f(x^k)+\rho_{k}(x^k-x^{k-1})
$$


where the momentum coefficient $\rho_k\in[0,1]$ generally and the step length $\alpha_k$ cannot be determined by line search.
We can unfold the recursion defining momentum  gradient descent and write:
$$
x^{k+1}=x^{k}-\alpha_{k}\nabla_{x}f(x^k)+\rho_{k}\underbrace{(x^k-x^{k-1})}_{\text{denoted as  $\Delta_{k}$ }}\\
\Delta_{k+1}= -\alpha_k\nabla_{x}f(x^k)+ \rho_{k}\Delta_{k}\\
\\
x^{k+1}=x^{k}-\alpha_{k}\nabla_{x}f(x^k)+\sum_{i=1}^{k-1}\{ - \prod_{j=0}^{i}\rho_{k-j}  \}\alpha_{k-j-1}\nabla f(\underbrace{\color{red}{x}^{k-i}}_{\triangle})+\rho_1\Delta_1.
$$

**Nesterov accelerated gradient method** at the $k$th step is given by:

$$
x^{k}=y^{k}-\alpha_{k+1}\nabla_{x}f(y^k) \qquad \text{Descent} \\
y^{k+1}=x^{k}+\rho_{k}(x^{k}-x^{k-1})   \qquad  \text{Momentum}
$$

where the momentum coefficient $\rho_k\in[0,1]$ generally.

$$
x^{k}=y^{k}-\alpha_{k+1}\nabla_{x}f(y^k) \\
y^{k}=x^{k-1}+\rho_{k-1}(x^{k-1}-x^{k-2})\\
x^{k+1}=x^{k}-\alpha_{k}\nabla_{x}f(y^k)+\sum_{i=1}^{k-1}\{ -\prod_{j=0}^{i}\rho_{k-j} \}\alpha_{k-j-1}\nabla f(\underbrace{\color{green}{y}^{k-i}}_{\triangle})+\rho_1\Delta_1.
$$

They are called as **inertial gradient methods** or **accelerated gradient methods**. [And there are some different forms.](https://jlmelville.github.io/mize/nesterov.html)

|Inventor of Nesterov accelerated Gradient|
|:---:|
|<img src=https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Nesterov_yurii.jpg/440px-Nesterov_yurii.jpg width = 60% />|


* [ORF523: Nesterov’s Accelerated Gradient Descent](https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/)
* [Nesterov’s Accelerated Gradient Descent for Smooth and Strongly Convex Optimization](https://blogs.princeton.edu/imabandit/2014/03/06/nesterovs-accelerated-gradient-descent-for-smooth-and-strongly-convex-optimization/)
* [Revisiting Nesterov’s Acceleration](https://blogs.princeton.edu/imabandit/2015/06/30/revisiting-nesterovs-acceleration/)
* [A short proof for Nesterov’s momentum](https://blogs.princeton.edu/imabandit/2018/11/21/a-short-proof-for-nesterovs-momentum/)
* [Nemirovski’s acceleration](https://blogs.princeton.edu/imabandit/2019/01/09/nemirovskis-acceleration/)
***
* https://zhuanlan.zhihu.com/p/41263068
* https://zhuanlan.zhihu.com/p/35692553
* https://zhuanlan.zhihu.com/p/35323828
* [On Gradient-Based Optimization: Accelerated, Asynchronous, Distributed and Stochastic](https://www.sigmetrics.org/sigmetrics2017/MI_Jordan_sigmetrics2017.pdf)
* [Nesterov Accelerated Gradient and Momentum](https://jlmelville.github.io/mize/nesterov.html)
* [Why Momentum Really Works](https://distill.pub/2017/momentum/)
* http://www.optimization-online.org/DB_FILE/2018/11/6938.pdf
* https://www.mat.univie.ac.at/~neum/glopt/mss/MasAi02.pdf
* https://www.cs.cmu.edu/~ggordon/10725-F12/slides/09-acceleration.pdf
* [WHAT IS GRADIENT DESCENT IN MACHINE LEARNING?](https://saugatbhattarai.com.np/what-is-gradient-descent-in-machine-learning/)
* https://www.fromthegenesis.com/gradient-descent-part1/
* https://www.fromthegenesis.com/gradient-descent-part-2/
* [Deep Learning From Scratch IV: Gradient Descent and Backpropagation](http://www.deepideas.net/deep-learning-from-scratch-iv-gradient-descent-and-backpropagation/)
* https://ee227c.github.io/notes/ee227c-lecture08.pdf
* [Monotonicity, Acceleration, Inertia, and the Proximal Gradient algorithm](http://www.iutzeler.org/pres/osl2017.pdf)


## Conjugate Gradient Methods

It is from the methods to solve the Symmetric positive definite linear systems $Ax=b, A=A^T\succ 0 \in\mathbb{R}^{n\times n}$.
$x^{\ast}=A^{-1}b$ is the solution and $x^{\ast}$ minimizes (convex function) $f(x)=\frac{1}{2}x^T Ax - b^T x$  thus $\nabla f(x) = Ax − b$ is gradient of $f$.
$r = b − Ax=-\nabla f(x) = -A(x-x^{\ast})$ is called the residual at $x$.

[The solution is to make the search directions-orthogonal instead of orthogonal.](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
Two vectors $u$ and $v$ are $A$ -orthogonal, or conjugate, i.e.,
$$u^TAv=0.$$

Krylov subspace $\cal K_k=\operatorname{span}\{b, Ab, \cdots, A^{k-1}b\}$ and
we define the Krylov sequence
$$ x^{k} =\arg\min_{x\in\cal K_k} f(x)=\arg\min_{x\in\cal K}{\|x-x^{\ast}\|}_A^2$$

the CG algorithm (among others) generates the Krylov sequence.
* $f\left(x^{(k+1)}\right) \leq f\left(x^{(k)}\right)$ but $\|r\|$ can increase.
* $x^{(n)}=x^{\star}\left(i.e., x^{\star} \in \mathcal{K}_{n}\right)$ even when  $\mathcal{K}_{n} \neq \mathbf{R}^{n}$
* $x^{(k)}=p_{k}(A) b,$ where  $p_{k}$ is a polynomial with $\operatorname{deg} p_{k}<k$.

There is a two-term recurrence
$$\fbox{$x^{k+1} = x^k + \alpha_k (x^k-x^{\ast}) + \beta_k (x^{k}-x^{k-1})$}$$

$$
{x :=0, r :=b,   \rho_{0} :={\|r\|}^2 } \\
{\text { for } k=1, \ldots, N_{\text {max }}} \\
{\text { quit if } \sqrt{\rho_{k}} \leq \epsilon {\|b\|}_{2} \text {or} \|r\| \leq \epsilon{\|b\|}_{2}} \\
{w := Ap} \\
{\alpha :=\frac{\rho_{k}}{w^{T} p}} \\
{x := r+\alpha p} \\
{r := r-\alpha w} \\
{\rho_k := {\|r\|}^2}
$$

**Preconditioned conjugate gradient**

with preconditioner  $M \approx A^{-1}$ (hopefully)
$$
\begin{array}{l}
{x :=0, r :=b-A x_{0},  p :=r  z :=M r,  \rho_{1} :=r^{T} z} \\
{\text { for } k=1, \ldots, N_{\text {max }}} \\
{\begin{array}{l}{\text { quit if } \sqrt{\rho_{k}} \leq \epsilon {\|b\|}_{2} \text {or} \|r\| \leq \epsilon{\|b\|}_{2}} \\
{w :=\frac{\rho_{k}}{w^{T} p}} \\
{\alpha :=\frac{\rho_{k}}{w^{T} p}} \\
{x :=r+\alpha p} \\
{r :=r-\alpha w} \\
{z :=M r} \\
\rho_{k+1} :=z^{T} r\\
{p := z+\frac{\rho_{k+1}}{\rho_{k}} p}\end{array}}
 \end{array}
$$


The gradient  descent  methods transforms the multiply optimization to univariate optimization.

Here is an outline of the nonlinear CG method:
* $d^{0}=r^{0}=-f^{\prime}(x^0)$
* Find $\alpha^{i}$ that minimizes $f(x^{i}+\alpha^{i} d^{i})$,
* $x^{i+1}=x^{i} + \alpha_{i} d^{i}$
* $r^{i+1}=-f^{\prime}(x^{i+1})$
* $\beta_{i+1}=\frac{\left<r^{i+1}, r^{i+1}\right>}{\left<r^{i}, r^{i}\right>}$ or $\beta_{i+1}=\max\{\frac{\left<r^{i+1}, r^{i+1}-r^{i}\right>}{\left<r^{i}, r^{i}\right>}, 0\}$
* $d^{i+1}=r^{i+1}+\beta_{i+1}d^{i}$


+ https://wiki.seg.org/wiki/The_conjugate_gradient_method
+ https://stanford.edu/class/ee364b/lectures/conj_grad_slides.pdf
+ [An Introduction to the Conjugate Gradient Method Without the Agonizing Pain](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
+ http://www.cs.umd.edu/users/oleary/cggg/historyindex.html
+ [A BRIEF INTRODUCTION TO THE  CONJUGATE GRADIENT METHOD](http://www.math.ust.hk/~mamu/courses/531/cg.pdf)

## Variable Metric Methods

### Newton's Methods

NEWTON’S METHOD and QUASI-NEWTON METHODS are classified to variable metric methods.

It is also to find the solution of unconstrained optimization problems, i.e.
$$\min f(x)$$
where $x\in \mathbb{R}^{n}$. Specially, the objective function $f(x)$ is convex so that the local minima is global minima.
***
If ${x^{\star}}$ is the extrema of the cost function $f(x)$, it is necessary that $\nabla f(x^{\star}) = 0$.
So if we can find all the solution of the equation system $\nabla f(x) = 0$, it helps us to find the solution to the optimization problem $\arg\min_{x\in\mathbb{R}^n} f(x)$.

**Newton's method** is one of the fixed-point methods to solve nonlinear equation system.

It is given by
$$
x^{k+1}=\arg\min_{x}f(x^k)+(x-x^k)^T \nabla_x f(x^k)+\frac{1}{2\alpha_{k+1}}(x-x^k)^T H(x^k)(x-x^k)
\\=x^{k}-\alpha_{k+1}H^{-1}(x^{k})\nabla_{x}\,{f(x^{k})}
$$
where $H^{-1}(x^{k})$ is inverse of the Hessian matrix of the function $f(x)$ at the point $x^{k}$.
It is called **Newton–Raphson algorithm** in statistics.
Especially when the log-likelihood function $\ell(\theta)$ is well-behaved,
a natural candidate for finding the MLE is the **Newton–Raphson algorithm** with quadratic convergence rate.

$$
x^{k+1}=x^{k}-\alpha_{k+1}H^{-1}(x^{k})\nabla_{x}\,{f(x^{k})}\\
= x^{k-1}-\alpha_{k}H^{-1}(x^{k-1})\nabla_{x}\,{f(x^{k-1})}-\alpha_{k+1}H^{-1}(x^{k})\nabla_{x}\,{f(x^{k})} \\
=x^{1}-\sum_{n=1}^{k} \alpha_{n+1}H^{-1}(x^{n})\nabla_{x}\,{f(x^{n})}
$$

- https://brilliant.org/wiki/newton-raphson-method/
- https://www.shodor.org/unchem/math/newton/
- http://fourier.eng.hmc.edu/e176/lectures/NM/node21.html
- https://www.cup.uni-muenchen.de/ch/compchem/geom/nr.html
- http://web.stanford.edu/class/cme304/docs/newton-type-methods.pdfs

### The Fisher Scoring Algorithm

In maximum likelihood estimation, the objective function is the log-likelihood function, i.e.
$$
\ell(\theta)=\sum_{i=1}^{n}\log{P(x_i|\theta)}
$$
where $P(x_i|\theta)$ is the probability of realization $X_i=x_i$ with the unknown parameter $\theta$.
However, when the sample random variable $\{X_i\}_{i=1}^{n}$ are not observed or realized, it is best to
replace negative Hessian matrix (i.e. -$\frac{\partial^2\ell(\theta)}{\partial\theta\partial\theta^{T}}$)
of the likelihood function with the  **observed information matrix**:
$$
J(\theta)=\mathbb{E}({\color{red}{-1}}\frac{\partial^2\ell(\theta)}{\partial\theta\partial\theta^{T}})
=\text{$\color{red}{-}$}\int\frac{\partial^2\ell(\theta)}{\partial\theta\partial\theta^{T}}f(x_1, \cdots, x_n|\theta)\mathrm{d}x_1\cdots\mathrm{d}x_n
$$
where $f(x_1, \cdots, x_n|\theta)$ is the joint probability density function of  $X_1, \cdots, X_n$ with unknown parameter $\theta$.

And the **Fisher scoring algorithm** is given by
$$
\theta^{k+1}=\theta^{k}+\alpha_{k}J^{-1}(\theta^{k})\nabla_{\theta} \ell(\theta^{k})
$$
where $J^{-1}(\theta^{k})$ is the inverse of observed information matrix at the point $\theta^{k}$.

See <http://www.stats.ox.ac.uk/~steffen/teaching/bs2HT9/scoring.pdf> or <https://wiseodd.github.io/techblog/2018/03/11/fisher-information/>.

**Fisher scoring algorithm** is regarded  as an example of **Natural Gradient Descent** in
information geometry as shown in <https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/>
and <https://www.zhihu.com/question/266846405>.

### Quasi-Newton Methods

Quasi-Newton methods, like steepest descent, require only the gradient of the objective
function to be supplied at each iterate.
By measuring the changes in gradients, they construct a model of the objective function
that is good enough to produce super-linear convergence.
The improvement over steepest descent is dramatic, especially on difficult
problems. Moreover, since second derivatives are not required, quasi-Newton methods are
sometimes more efficient than Newton’s method.[^11]

In optimization, quasi-Newton methods (a special case of **variable-metric methods**) are algorithms for finding local maxima and minima of functions. Quasi-Newton methods are based on Newton's method to find the stationary point of a function, where the gradient is 0.
In quasi-Newton methods the Hessian matrix does not need to be computed. The Hessian is updated by analyzing successive gradient vectors instead. Quasi-Newton methods are a generalization of the secant method to find the root of the first derivative for multidimensional problems.
In multiple dimensions the secant equation is under-determined, and quasi-Newton methods differ in how they constrain the solution, typically by adding a simple low-rank update to the current estimate of the Hessian.
One of the chief advantages of quasi-Newton methods over Newton's method is that the Hessian matrix (or, in the case of quasi-Newton methods, its approximation) $\mathbf{B}$ does not need to be inverted. The Hessian approximation $\mathbf{B}$ is chosen to satisfy
$$
\nabla f(x^{k+1})=\nabla f(x^{k})+B(x^{k+1}-x^{k}),
$$
which is called the **$\fbox{secant equation}$** (the Taylor series of the gradient itself).
In more than one dimension B is underdetermined. In one dimension, solving for B and applying the Newton's step with the updated value is equivalent to the [secant method](https://www.wikiwand.com/en/Secant_method).
The various quasi-Newton methods differ in their choice of the solution to the **secant equation** (in one dimension, all the variants are equivalent).

The unknown $x_{k}$ is updated applying the Newton's step calculated using the current approximate Hessian matrix $B_{k}$:

1. ${\displaystyle \Delta x_{k}=-\alpha_{k} B_{k}^{-1}\nabla f(x_{k})}$, with $\alpha$  chosen to satisfy the Wolfe conditions;
2. $x_{k+1}=x_{k}+\Delta x_{k}$;
3. The gradient computed at the new point $\nabla f(x_{k+1})$, and
$y_{k}=\nabla f(x_{k+1})-\nabla f(x_{k})$ is used to update the approximate Hessian $B_{k+1}$, or directly its inverse $H_{k+1} = B_{k+1}^{-1}$ using the Sherman–Morrison formula.

A key property of the BFGS and DFP updates is that if $B_{k}$ is positive-definite, and ${\alpha}_{k}$ is chosen to satisfy the Wolfe conditions, then $B_{k+1}$ is also positive-definite.

For example,

|Method|$\displaystyle B_{k+1}=$| $H_{k+1}=B_{k+1}^{-1}=$|
|---|---|---|
|DFP|$(I-\frac{y_k \Delta x_k^{\mathrm{T}}}{y_k^{\mathrm{T}} \Delta x_k}) B_k (I-\frac{ \Delta x_k y_k^{\mathrm{T}}}{y_k^{\mathrm{T}} \Delta x_k}) + \frac{y_k y_k^{\mathrm{T}}}{y_k^{\mathrm{T}}} \Delta x_k$|$H_k +\frac{\Delta x_k \Delta x_k^T}{\Delta x_k^T y_k} -\frac{H_k y_ky_k^T H_k}{y_K^T H_K y_k}$|
|BFGS|$B_k + \frac{y_k y_k^{\mathrm{T}}}{y_k^{\mathrm{T}}\Delta x_k} - \frac{B_k\Delta x_k(B_k\Delta x_k)^T}{\Delta x_k B_k \Delta x_k}$|$(I-\frac{ \Delta x_k^{\mathrm{T}} y_k}{ y_k^{\mathrm{T}} \Delta x_k}) H_k (I-\frac{y_k \Delta x_k^{\mathrm{T}}}{ y_k^{\mathrm{T}} \Delta x_k}) + \frac{\Delta x_k \Delta x_k^T}{y_k^T \Delta x_k}$|
|SR1|$B_{k} + \frac{(y_{k} - B_{k}\,\Delta x_{k} )(y_{k} - B_{k}\,\Delta x_{k})^{\mathrm{T}} }{(y_{k} - B_{k}\,\Delta x_{k})^{\mathrm{T} }\,\Delta x_{k}}$|	$H_{k} + \frac{(\Delta x_{k}-H_{k}y_{k}) (\Delta x_{k}  -H_{k} y_{k})^{\mathrm{T}} }{(\Delta x_{k}-H_{k}y_{k})^{\mathrm {T} }y_{k}}$|


<img src="http://aria42.com/images/bfgs.png" width = "60%" />

* [Quasi-Newton methods in Wikipedia page](https://www.wikiwand.com/en/Quasi-Newton_method)
* http://59.80.44.98/www.seas.ucla.edu/~vandenbe/236C/lectures/qnewton.pdf
* http://fa.bianp.net/teaching/2018/eecs227at/quasi_newton.html

#### The Barzilai-Borwein method

Consider the gradient iteration form
$$
x^{k+1}=x^{k}-\alpha_k \nabla f(x^k)
$$

which can be written as
$$
x^{k+1}=x^{k}- D_k \nabla f(x^k)
$$
where $D_k = \alpha_k I$.
In order to make the matrix $D_k$ have quasi-Newton property, we compute $\alpha_k$ such that
$$
\min \|s_{k-1}-D_k y_{k-1}\|
$$
which yields

$$
\alpha_k =\frac{\left<s_{k-1},y_{k-1}\right>}{\left<y_{k-1},y_{k-1}\right>}\tag 1
$$

where $s_{k-1}= x_k-x_{k-1}, y_{k-1}=\nabla f(x^k)-\nabla f(x^{k-1})$.

By symmetry, we may minimize $\|D_k^{-1}s_{k-1}- y_{k-1}\|$ with respect to $\alpha_k$ and get

$$
\alpha_k = \frac{\left<s_{k-1}, s_{k-1}\right>}{\left<s_{k-1}, y_{k-1}\right>}.\tag 2
$$

In short, the iteration formula of Barzilai-Borwein method is given by

$$
x^{k+1}=x^{k}-\alpha_k \nabla f(x^k)
$$
where $\alpha_k$ is determined by (1) or (2).

It is easy to see that in this method no matrix computations and no line searches (except $k = 0$) are required.

- https://mp.weixin.qq.com/s/G9HH29b2-VBnk_Sqze8pDg
- http://www.math.ucla.edu/~wotaoyin/math273a/slides/
- http://bicmr.pku.edu.cn/~wenzw/courses/WenyuSun_YaxiangYuan_BB.pdf
- https://www.math.lsu.edu/~hozhang/papers/cbb.pdf

#### L-BFGS

The BFGS quasi-newton approximation has the benefit of not requiring us to be able to analytically compute the Hessian of a function. However, we still must maintain a history of the $s_n$ and $y_n$ vectors for each iteration.

* https://www.wikiwand.com/en/Limited-memory_BFGS
* [On the limited memory BFGS method for large scale optimization](https://link.springer.com/article/10.1007%2FBF01589116)
* [Numerical Optimization: Understanding L-BFGS](http://aria42.com/blog/2014/12/understanding-lbfgs)
* [Unconstrained optimization: L-BFGS and CG](http://www.alglib.net/optimization/lbfgsandcg.php)

[L-BFGS algorithm builds and refines quadratic model of a function being optimized. Algorithm stores last M value/gradient pairs and uses them to build positive definite Hessian approximation. This approximate Hessian matrix is used to make quasi-Newton step. If quasi-Newton step does not lead to sufficient decrease of the value/gradient, we make line search along direction of this step.](http://www.alglib.net/optimization/lbfgsandcg.php)

[Essential feature of the algorithm is positive definiteness of the approximate Hessian. Independently of function curvature (positive or negative) we will always get SPD matrix and quasi-Newton direction will always be descent direction.](http://www.alglib.net/optimization/lbfgsandcg.php)

[Another essential property is that only last $M$ function/gradient pairs are used, where M is moderate number smaller than problem size N, often as small as 3-10. It gives us very cheap iterations, which cost just O(N·M) operations.](http://www.alglib.net/optimization/lbfgsandcg.php)

- [Optimizing Neural Networks with Kronecker-factored Approximate Curvature](https://arxiv.org/abs/1503.05671)

#### Gauss–Newton Method

A form of regression where the objective function is the sum of squares of nonlinear functions:
$$f(x)=\frac{1}{2}\sum_{j=1}^{m}(r_j(x))^2=\frac{1}{2}\sum_{j=1}^{m}{\|r(x)\|}^2$$
where The j-th component of the m-vector $r(x)$ is the residual $r_j(x)=(\phi(x_j;t_j)-y_j)^2$.

The Gauss-Newton Method generalizes Newton’s method for multiple dimensions with approximate Hessian matrix $\nabla^2 f_k \approx J_k^T J_k$.

The basic steps that the software will perform (note that the following steps are for a single iteration):

+ Make an initial guess $x^0$ for $x$,
+ Make a guess for $k = 1$,
+ Create a vector $f^k$ with elements $f_i(x^k)$,
+ Create a Jacobian matrix for $J_k$
+ Solve ($J^T_k J_k p_k = -J^T_k f_k$).
+ This gives you the probabilities $p$ for all $k$.
+ Find $s$. $F(x^k + s p_k)$ should satisfy the `Wolfe conditions` (these prove that step-lengths exist).
+ Set $x^{k+1} = x^k + sp^k$.
+ Repeat Steps 1 to 7 until convergence.

- https://www8.cs.umu.se/kurser/5DA001/
- http://iacs-courses.seas.harvard.edu/courses/am205/fall14/slides/am205_lec06.pdf
- [Applications of the Gauss-Newton Method](https://ccrma.stanford.edu/~wherman/tulane/gauss_newton.pdf)
- [Gauss-Newton / Levenberg-Marquardt Optimization](http://ethaneade.com/optimization.pdf)
- [Nonlinear Least-Squares Problems with the Gauss-Newton and Levenberg-Marquardt Methods](https://www.math.lsu.edu/system/files/MunozGroup1%20-%20Presentation.pdf)
- [Gauss-Newton algorithm for nonlinear models](http://fourier.eng.hmc.edu/e176/lectures/NM/node36.html)

***
* [Wikipedia page on Newton Method](https://www.wikiwand.com/en/Newton%27s_method_in_optimization)
* [Newton-Raphson Visualization (1D)](http://bl.ocks.org/dannyko/ffe9653768cb80dfc0da)
* [Newton-Raphson Visualization (2D)](http://bl.ocks.org/dannyko/0956c361a6ce22362867)
* [Using Gradient Descent for Optimization and Learning](http://www.gatsby.ucl.ac.uk/teaching/courses/ml2-2008/graddescent.pdf)
* ftp://lsec.cc.ac.cn/pub/home/yyx/papers/siamjna1987.pdf


### Natural Gradient Descent

The generalized natural gradient is the direction of steepest ascent in Riemannian space, which is invariant to parametrization, and is defined:
$$\nabla S\propto\lim_{\epsilon\to 0} \arg\max_{d:B(P_{\theta}, P_{\theta+d})=\epsilon} S(\theta+d, y)$$

where $B(P_{\theta}, P_{\theta+d})$ is the Bregman divergence of the probabilities $P_{\theta}, P_{\theta + d}$ and $S$ is a loss function on the probability $P_{\theta + d}$ and $y$ such as log-likelihood function.

Natural gradient descent is to solve the optimization problem $\min_{\theta} L(\theta)$ by
$$
\theta^{(t+1)}=\theta^{(t+1)}-\alpha_{(t)}F^{-1}(\theta^{(t)})\nabla_{\theta}L(\theta^{(t)})
$$
where $F^{-1}(\theta^{(t)})$ is the inverse of `Remiann metric` at the point $\theta^{(t)}$.
And **Fisher scoring** algorithm is a typical application of **Natural Gradient Descent** to statistics.  
**Natural gradient descent** for manifolds corresponding to
exponential families can be implemented as a first-order method through [**mirror descent**](https://www.stat.wisc.edu/~raskutti/publication/MirrorDescent.pdf).



| Originator of Information Geometry: Shunichi Amari |
|:----:|
|<img src="https://groups.oist.jp/sites/default/files/imce/u34/images/people/shun-ichi-amari.jpg" width = "70%" />|


* [Natural gradient descent and mirror descent](http://www.dianacai.com/blog/2018/02/16/natural-gradients-mirror-descent/)
* [Online Natural Gradient as a Kalman Filter](http://www.yann-ollivier.org/rech/publs/natkal.pdf)
* [New insights and perspectives on the natural gradient method](https://arxiv.org/pdf/1412.1193.pdf)
* https://www.zhihu.com/question/266846405
* [2016 PKU Mini-Course: Information Geometry](http://bicmr.pku.edu.cn/~dongbin/Conferences/Mini-Course-IG/index.html)
* [Information Geometry and Natural Gradients](http://ipvs.informatik.uni-stuttgart.de/mlr/wp-content/uploads/2015/01/mathematics_for_intelligent_systems_lecture12_notes_I.pdf)
* [Tutorial on Information Geometry and Algebraic Statistics](http://www.luigimalago.it/tutorials/algebraicstatistics2015tutorial.pdf)
* [True Asymptotic Natural Gradient Optimization
](http://www.yann-ollivier.org/rech/publs/tango.pdf)
* [Shun-ichi Amari's CV in RIKEN](http://www.brain.riken.jp/asset/img/researchers/cv/s_amari.pdf)
* [Energetic Natural Gradient Descent](https://people.cs.umass.edu/~pthomas/papers/Thomas2016b_ICML.pdf)
* [Natural gradients and K-FAC](https://www.depthfirstlearning.com/assets/k-fac-tutorial.pdf)
* http://www.divergence-methods.org/
* http://www.deeplearningpatterns.com/doku.php?id=natural_gradient_descent
* [NATURAL GRADIENTS AND STOCHASTIC VARIATIONAL INFERENCE](http://andymiller.github.io/2016/10/02/natural_gradient_bbvi.html)
* [谈谈优化算法 - 郑思座的文章 - 知乎](https://zhuanlan.zhihu.com/p/60088231)
* [Accelerating Natural Gradient with Higher-Order Invariance](https://ermongroup.github.io/blog/geo/)

## Higher-Order Derivatives Methods

[Higher-order methods, such as Newton, quasi-Newton and adaptive gradient descent methods, are extensively used in many scientific and engineering domains. At least in theory, these methods possess several nice features: they exploit local curvature information to mitigate the effects of ill-conditioning, they avoid or diminish the need for hyper-parameter tuning, and they have enough concurrency to take advantage of distributed computing environments. Researchers have even developed stochastic versions of  higher-order methods, that feature speed and scalability by incorporating curvature information in an economical and judicious manner. However, often higher-order methods are “undervalued.”](https://sites.google.com/site/optneurips19/)

[The key observation, which underlies all results of this paper, is that an appropriately regularized Taylor approximation of convex function is a convex multivariate polynomial. This is indeed a very natural property since this regularized approximation usually belongs to the epigraph of convex function. Thus, the auxiliary optimization problem in the high-order (or tensor) methods becomes generally solvable by many powerful methods of Convex Optimization.](https://ideas.repec.org/p/cor/louvco/2018005.html)

The most well-known third order optimization method may be Halley's method(Tangent Hyperbolas Method).

- [Higher-Order Derivatives in Computational Systems Engineering
Problem Solving](http://www.autodiff.org/ad08/talks/ad08_marquardt.pdf)
- [SECOND-ORDER OPTIMIZATION FOR NEURAL NETWORKS](http://www.cs.toronto.edu/~jmartens/docs/thesis_phd_martens.pdf)
- [On the use of higher order derivatives in optimization using Lagrange's expansion](https://www.sciencedirect.com/science/article/abs/pii/0362546X8390024X)
- [Sparsity in Higher Order Methods in Optimization*](https://cerfacs.fr/wp-content/uploads/2016/04/gundersen.pdf)
- [HoORaYs: High-order Optimization of Rating Distance for Recommender Systems](https://dl.acm.org/citation.cfm?id=3098019)
- https://akyrillidis.github.io/2019/08/05/WorkshopNeurIPS.html
- ftp://file.viasm.org/Web/TienAnPham-16/Preprint_1669.pdf
- [Implementable Tensor Methods in Unconstrained Convex Optimization](https://alfresco.uclouvain.be/alfresco/service/guest/streamDownload/workspace/SpacesStore/aabc2323-0bc1-40d4-9653-1c29971e7bd8/coredp2018_05web.pdf?guest=true)
- [Reachability of optimal convergence rate estimates for high-order numerical convex optimization methods](https://journals.eco-vector.com/0869-5652/article/view/12813)
- [The global rate of convergence for optimal tensor methods in smooth convex optimization](https://arxiv.org/abs/1809.00382)
- [Optimization for Tensor Models](https://www.math.ucla.edu/sites/default/files/dls/posters/UCLA_Lecture_3_Final.compressed.pdf)
- [Tensor Methods for Large, Sparse Unconstrained Optimization](https://www.semanticscholar.org/paper/Tensor-Methods-for-Large%2C-Sparse-Unconstrained-Bouaricha/88de96f75204e6d49849eaa69321b906b3675393)
- [Beyond First Order Methods in ML](https://sites.google.com/site/optneurips19/)
- [UNIFIED ACCELERATION OF HIGH-ORDER ALGORITHMS UNDER HOLDER CONTINUITY AND UNIFORM CONVEXITY](https://arxiv.org/pdf/1906.00582.pdf)
- [Lower Bounds for Higher-Order Convex Optimization](https://pdfs.semanticscholar.org/d32d/682349b09b2d3767448cfd6fafd199ccf92a.pdf)
- https://file.scirp.org/Html/9-7403428_73034.htm
- [On the Geometry of Halley's Method](https://ms.yccd.edu/Data/Sites/1/userfiles/facstaff/jthoo/cvandpubs/papers/halley.pdf)
- https://archive.lib.msu.edu/crcmath/math/math/h/h030.htm
- [A METHOD FOR OBTAINING THIRD-ORDER ITERATIVE FORMULAS](http://www.kurims.kyoto-u.ac.jp/EMIS/journals/NSJOM/Papers/38_2/NSJOM_38_2_195_207.pdf)
- [](https://rucore.libraries.rutgers.edu/rutgers-lib/58491/PDF/1/play/)
- [Some New Variants of Chebyshev-Halley Methods Free from Second Derivative](http://www.internonlinearscience.org/upload/papers/20110301100750315.pdf)


## Trust Region Methods

Trust-region methods define a region around
the current iterate within which they trust the model to be an adequate representation of
the objective function, and then choose the step to be the approximate minimizer of the
model in this region.
If a step is not acceptable, they reduce the size of the region and find a new minimizer.
In general, the direction of the step changes whenever the size of the trust region
is altered.

$$
\min_{p\in\mathbb{R}^n} m_k(p)=f_k+ g_k^T p+\frac{1}{2}p^T B_k p, \, \, s.t. \|p\|\leq {\Delta}_k,
$$
where $f_k=f(x^k), g_k = \nabla f(x^k)$;$B_k$ is some symmetric matrix as an
approximation to the Hessian $\nabla^2 f(x^k + tp)$; $\Delta_k>0$ is the trust-region radius.
In most of our discussions, we define $\|\cdot\|$ to be the Euclidean norm.
Thus, the trust-region approach requires us to solve a sequence of subproblems
in which the objective function and constraint are both quadratic.
Generally, we set $B_k = \nabla^2 f(x^k)$.

![](https://optimization.mccormick.northwestern.edu/images/c/c0/TRM_Dogleg.png)
* https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods
* [Concise Complexity Analyses for Trust-Region Methods](https://arxiv.org/abs/1802.07843)
* https://www.nmr-relax.com/manual/Trust_region_methods.html
* http://lsec.cc.ac.cn/~yyx/worklist.html

## Expectation Maximization Algorithm

**Expectation-Maximization algorithm**, popularly known as the  **EM algorithm** has become a standard piece in the statistician’s repertoire.
It is used in incomplete-data problems or latent-variable problems such as Gaussian mixture model in maximum likelihood  estimation.
The basic principle behind the **EM** is that instead of performing a complicated optimization,
one augments the observed data with latent data to perform a series of simple optimizations.

It is really popular for Bayesian statistician.

Let $\ell(\theta|Y_{obs})\stackrel{\triangle}=\log{L(\theta | Y_{obs})}$ denote the log-likelihood function of observed datum $Y_{obs}$.
We augment the observed data $Y_{obs}$ with latent variables $Z$ so that both the
complete-data log-likelihood $\ell(\theta | Y_{obs}, Z)$ and the conditional predictive distribution $f(z|Y_{obs}, \theta)$ are available.

Each iteration of the **EM** algorithm consists of an expectation step (E-step) and a maximization step (M-step)
Specifically, let $\theta^{(t)}$ be the current best guess at the MLE $\hat\theta$. The E-step
is to compute the **Q** function defined by
$$
\begin{align}
Q(\theta|\theta^{(t)})
        & \stackrel{\triangle}= \mathbb{E}(\ell(\theta | Y_{obs}, Z) |Y_{obs}, \theta^{(t)}) \\
        &= \int_{Z}\ell(\theta | Y_{obs}, Z)\times f(z | Y_{obs}, \theta^{(t)})\mathrm{d}z,
\end{align}
$$

and the M-step is to maximize **Q** with respect to $\theta$ to obtain

$$
\theta^{(t+1)}=\arg\max_{\theta} Q(\theta|\theta^{(t)}).
$$

* [BIOS731: Advanced Statistical Computing](http://www.haowulab.org/teaching/statcomp/statcomp.html)
* http://cs229.stanford.edu/notes/cs229-notes8.pdf
* [EM算法存在的意义是什么？ - 史博的回答 - 知乎](https://www.zhihu.com/question/40797593/answer/275171156)

|Diagram of EM algorithm|
|:---------------------:|
|![](https://i.stack.imgur.com/v5bqe.png)|


### Generalized EM Algorithm

Each iteration of the **generalized EM** algorithm consists of an expectation step (E-step) and a maximization step (M-step)
Specifically, let $\theta^{(t)}$ be the current best guess at the MLE $\hat\theta$. The E-step
is to compute the **Q** function defined by
$$
Q(\theta|\theta^{(t)})
        = \mathbb{E}[ \ell(\theta|Y_{obs}, Z)|Y_{obs},\theta^{(t)} ] \\
        = \int_{Z}\ell(\theta|Y_{obs}, Z)\times f(z|Y_{obs}, \theta^{(t)})\mathrm{d}z,
$$
and the another step is to find  $\theta$ that satisfies $Q(\theta^{t+1}|\theta^{t})>Q(\theta^{t}|\theta^{t})$, i.e.
$$
\theta^{(t+1)}\in \{\hat{\theta}|Q(\hat{\theta}|\theta^{(t)} \geq Q(\theta|\theta^{(t)}) \}.
$$

It is not to maximize the conditional expectation.

See more on the book [The EM Algorithm and Extensions, 2nd Edition by Geoffrey McLachlan , Thriyambakam Krishna](https://www.wiley.com/en-cn/The+EM+Algorithm+and+Extensions,+2nd+Edition-p-9780471201700).

<img title="projection " src="https://pic4.zhimg.com/80/v2-468b515b4d26ebc4765f82bf3ed1c3bf_hd.jpg" width="50%" />

* [The MM Algorithm by Kenneth Lange](https://www.stat.berkeley.edu/~aldous/Colloq/lange-talk.pdf)
* [MM Optimization Algorithms](https://epubs.siam.org/doi/book/10.1137/1.9781611974409)
* [Examples of MM Algorithms](http://hua-zhou.github.io/teaching/biostatm280-2018spring/slides/20-mm/deLeeuw.pdf)
* [Majorization-Minimization Algorithms in Signal Processing, Communications, and Machine Learning](https://palomar.home.ece.ust.hk/papers/2017/SunBabuPalomar-TSP2017%20-%20MM.pdf)
+ [AN ASSEMBLY AND DECOMPOSITION APPROACH FOR CONSTRUCTING SEPARABLE MINORIZING FUNCTIONS IN A CLASS OF MM ALGORITHMS](https://www.semanticscholar.org/paper/AN-ASSEMBLY-AND-DECOMPOSITION-APPROACH-FOR-IN-A-OF-Tian-Huang/fdffff8c1bf5cdd258f287136e0bbcd8ab0b7529)
+ [Generalized Majorization-Minimization](http://proceedings.mlr.press/v97/parizi19a/parizi19a.pdf)

## Projected Gradient Method and More

We will focus on  **projected gradient descent** and its some non-Euclidean generalization in order to solve some simply constrained optimization problems.

If not specified, all these methods are aimed to solve convex optimization problem with explicit constraints, i.e.
$$
\arg\min_{x\in\mathbb{S}}f(x)
$$

where $f$ is convex and differentiable, $\mathbb{S}\subset\mathbb{R}^n$  is convex.
The optimal condition for this constrained optimization problem is that `any feasible direction is not descent direction`: if $x^{\star}\in\mathbb{S}$ is the solution to the problem, we can  assert that `variational inequality` holds:
$$\forall x \in \mathbb{S}, \left<\nabla f(x^{\star}),x-x^{\star} \right> \geq 0.$$

And it is the optimal condition of constrained optimization problem.

We say the gradient $\nabla f$ of the convex function $f$ is a `monotone` operator because
$$\left<x-y, \nabla f(x)-\nabla f(y)\right>\geq 0. $$

### Projected Gradient Descent

**Projected gradient descent** has two steps:
$$
z^{k+1} = x^{k}-\alpha_k\nabla_x f(x^{k}) \qquad \text{Descent}\\
x^{k+1} = Proj_{\mathbb{S}}(z^{k+1})=\arg\min_{x\in \mathbb{S}}\|x-z^{k+1}\|^{2} \qquad\text{Projection}
$$

or in the compact form
$$
x^{k+1} = \arg\min_{x\in \mathbb{S}}\|x-(x^{k}-\alpha_k\nabla_x f(x^{k}))\|^{2}
\\= \arg\min_{x}\{\|x-(x^{k}-\alpha_k\nabla_x f(x^{k}))\|^{2} +\delta_{\mathbb{S}}(x) \}
$$

where $\delta_{\mathbb{S}}$ is the indictor function of the set $\mathbb{S}$
$$
h(x) =\delta_{\mathbb{S}}(x)=
 \begin{cases}
   0, & \text{if} \quad x \in \mathbb{S};\\
   \infty, & \text{otherwise}.
 \end{cases}
$$

Each projection is an optimization so that the iterative points satisfy the optimal conditions, which also restricts the projection  method into the case where the projection is available or simple to compute.
And it is natural to search better Descent step or Projection step.

The following links are recommended if you are interested in more theoretical proof:

* http://maths.nju.edu.cn/~hebma/slides/03C.pdf
* http://maths.nju.edu.cn/~hebma/slides/00.pdf

For the non-differentiable but convex function such as the absolute value function $f(x)=|x|$,  we therefore consider sub-gradients in place of the gradient, where sub-gradient $\phi$ at the point $\hat{x}$ is defined as the elements in the  domain of convex function ${f}$ (denoted as $D_f$), satisfying

$$
\left<\phi, x-\hat{x}\right>\leq f(x) -f(\hat{x}),\forall x \in D_f.
$$

### Mirror descent

**Mirror descent** can be regarded as the non-Euclidean generalization via replacing the $\ell_2$ norm or Euclidean distance in projected gradient descent by [Bregman divergence](https://www.mdpi.com/1099-4300/16/12/6338/htm).

Bregman divergence is induced by convex smooth function ${h}$:

$$
 B_h(x,y) = h(x) - h(y)-\left<\nabla h(y),x-y\right>
$$

where $\left<\cdot,\cdot\right>$ is inner product and it also denoted as $D_h$.

The function ${h}$ is usually required to be strongly convex. And if the convex function ${h}$ is not differentiable, one element of the sub-gradient ${\partial h(y)}$ may replace the gradient $\nabla h(y)$.

It is convex in $x$ and  $\frac{\partial B_h(x, y)}{\partial x} = \nabla h(x) - \nabla h(y)$.


Especially, when ${h}$ is quadratic function, the Bregman divergence induced by $h$ is
$$
 B_h(x, y)=x^2-y^2-\left<2y,x-y\right>=x^2+y^2-2xy=(x-y)^2
$$
i.e. the square of Euclidean distance.

A wonderful introduction to **Bregman divergence** is **Meet the Bregman Divergences**
by [Mark Reid](http://mark.reid.name/) at <http://mark.reid.name/blog/meet-the-bregman-divergences.html>.

The Bregman projection onto a convex set $C\subset \mathbb{R}^n$ given by
$$
y^{\prime}= \arg\min_{x\in C} B(x,y)
$$
is unique.

A `generalized Pythagorean theorem` holds: for convex $C\subset \mathbb{R}^n$ and for all $x\in C$ and $y\in \mathbb{R}^n$ we have
$$B(x,y)\geq B(x,y^{\prime}) + B(y^{\prime},y)$$
where $y^{\prime}$ is the Bregman projection of ${y}$, and equality holds
when the convex set C defining the projection $y^{\prime}$ is affine.

<img src = "https://upload.wikimedia.org/wikipedia/commons/2/2e/Bregman_divergence_Pythagorean.png" width="70%">

***
It is given in the projection form:

$$
z^{k+1} = x^{k}-\alpha_k\nabla_x f(x^{k}) \qquad \text{Gradient descent}；\\
x^{k+1} = \arg\min_{x\in\mathbb{S}}B_h(x,z^{k+1}) \qquad\text{Bregman projection}.
$$

In another compact form, mirror gradient can be described in the proximal form:
$$
x^{k+1} = \arg\min_{x\in\mathbb{S}} \{ f(x^k) + \left<g^k, x-x^k\right> + \frac{1}{\alpha_k} B_h(x,x^k)\}\tag {1}
$$
with $g^k=\nabla f(x^k)$.

Note that the next iteration point $x^{k+1}$ only depends the current iteration point $x^{k}$ no more previous iteration points.

By the optimal conditions of equation (1), the original "mirror" form of mirror gradient method is described as
$$
\nabla h(x^{k+1}) = \nabla h(x^k) - \alpha_k \nabla f(x^k), \\
= \nabla h(x^1) - \sum_{n=1}^{k}\alpha_n \nabla f(x^n) , x\in \mathbb{S},
$$
where the convex function ${h}$ induces the Bregman divergence.

One special method is called `entropic mirror descent(Exponential Gradient Descent)` when the Bregman divergence induced by  $e^x$ and  the constraint set $\mathbb{S}\subset\mathbb{R}^n$ is simplex, i.e. $\sum_{i=1}^{n}x_i =1, \forall x_i \geq 0$.

**Entropic descent method** at step ${k}$ is given as follows:

$$
{x_{i}^{k+1} = \frac{x_i^{k}\exp(-\alpha \nabla {f(x^k)}_{i})}{\sum_{j=1}^{n} x_j^{k}\exp(-\alpha \nabla  {f(x^k)}_{j})}}, i=1,2,\dots, n.
$$
it is obvious that entropic decscent methods are in the coordinate-wise update formula.
Whast is more , it can be rewritten as
$$x^{k+1}=\frac{x^{1}\exp(\sum_{n=1}^{k}-\alpha \nabla f(x^n))}{\prod_{n=1}^{k}\left<x^n, \exp(-\alpha \nabla f(x^n))\right>}\propto x^{1}\exp(\sum_{n=1}^{k}-\alpha \nabla f(x^n)).$$  

`Multiplicative Weights Update` is closely related with entropic descent method. See more on the following link list.

* [The Divergence Methods Web Site (under construction)](http://www.divergence-methods.org/)
* [Bregman Divergence and Mirror Descent, Xinhua Zhang(张歆华)](http://users.cecs.anu.edu.au/~xzhang/teaching/bregman.pdf)
* [CS 294 / Stat 260, Fall 2014: Learning in Sequential Decision Problems](https://www.stat.berkeley.edu/~bartlett/courses/2014fall-cs294stat260/lectures/mirror-descent-notes.pdf)
* [ELE522: Large-Scale Optimization for Data Science , Yuxin Chen, Princeton University, Fall 2019](http://www.princeton.edu/~yc5/ele522_optimization/lectures/mirror_descent.pdf)
* [Mirror Descent and the Multiplicative Weight Update Method, CS 435, 201, Nisheeth Vishnoi](https://nisheethvishnoi.files.wordpress.com/2018/05/lecture42.pdf)
* [Mirror descent: 统一框架下的first order methods](https://zhuanlan.zhihu.com/p/34299990)
* [ORF523: Mirror Descent, part I/II](https://blogs.princeton.edu/imabandit/2013/04/16/orf523-mirror-descent-part-iii/)
* [ORF523: Mirror Descent, part II/II](https://blogs.princeton.edu/imabandit/2013/04/18/orf523-mirror-descent-part-iiii/)
* [Thibaut Lienart: MIrror Descent](https://tlienart.github.io/pub/csml/cvxopt/mda.html)
* [Sinkhorn Algorithm as a Special Case of Stochastic Mirror Descent](https://konstmish.github.io/publication/19_sinkhorn/)
* https://web.stanford.edu/class/cs229t/2017/Lectures/mirror-descent.pdf
* https://www.cs.ubc.ca/labs/lci/mlrg/slides/mirrorMultiLevel.pdf
* https://konstmish.github.io/

### Proximal Gradient Method

Recall the projected gradient method,  it converts the constrained problem $\arg\min_{x}\{f(x)\mid x\in \mathbb{S}\}$ into unconstrained (also uncontinuous) problem $\arg\min_{x}\{f(x)+\delta_{\mathbb{S}}(x)\}$ literally
and it follows the following iteration:
$$
x^{k+1} = \arg\min_{x}\{\|x-(x^{k}-\alpha_k\nabla_x f(x^{k}))\|^{2} +\delta_{\mathbb{S}}(x) \}
$$
where the function $\delta_{\mathbb{S}}(x)$ is obviously non-differentiable, defined as follows
$$
h(x) =\delta_{\mathbb{S}}(x)=
 \begin{cases}
   0, & \text{if} \quad x \in \mathbb{S};\\
   \infty, & \text{otherwise}.
 \end{cases}
$$

If the function $\delta_{\mathbb{S}}(x)$ is replaced by general non-differential while  convex function $\mathbf{h}(x)$, the `proximal mapping (or prox-operator)` of a convex function $\mathbf{h}$ is defined as
$$
prox_h(x)=\arg\min_{u}\{\mathbf{h}(u)+\frac{1}{2} {\|x-u\|}_2^2\}
$$

Unconstrained problem with cost function split in two components:
$$minimize \qquad f(x) = g(x)+\mathbf{h}(x)$$

- ${g}$ is convex, differentiable;
- $\mathbf{h}$ is closed, convex, possibly non-differentiable while $prox_h(x)$ is inexpensive.

**Proximal gradient algorithm**
$$x^{k}=prox_{t_k h} \{x^{k-1}-t_k\nabla g(x^{k-1})\}$$

$t_k > 0$  is step size, constant or determined by line search.

$$x^+ = prox_{th}(x-t\nabla g(x))$$
from the definition of proximal mapping:
$$
x^+ = \arg\min_{u}( h(u)+\frac{1}{2}{\|u-(x-t\nabla g(x))\|}_2^2 )
\\= \arg\min_{u}( h(u) + g(x) + \nabla g(x)^T(u-x) +\frac{1}{2t} {\|u-x\|}_2^2 )
\\= \arg\min_{u}( h(u) + \nabla g(x)^T(u-x) +\frac{1}{2t} {\|u-x\|}_2^2 )
$$

$x^{+}$ minimizes $h(u)$ plus a simple quadratic local model of $g(u)$ around ${x}$.

And projected gradient method is a special case of proximal gradient method.

And it is natural to consider a more general algorithm by replacing the squared Euclidean distance in definition of `proximal mapping` with a Bregman distance:
$$
\arg\min_{u}\{h(u)+\frac{1}{2} {\|x-u\|}_2^2\}\to \arg\min_{u}\{h(u)+ B(x,u)\}.
$$
so that the primary proximal gradient methods are modified to the Bregman version,
which is called as `Bregman proximal gradient` method.

* [A collection of proximity operators implemented in Matlab and Python.](http://proximity-operator.net/)
* [L. Vandenberghe ECE236C (Spring 2019): 4. Proximal gradient method](http://www.seas.ucla.edu/~vandenbe/236C/lectures/proxgrad.pdf)
* https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture18.pdf
* [Accelerated Bregman Proximal Gradient Methods for Relatively Smooth Convex Optimization](https://arxiv.org/abs/1808.03045)
* [A unified framework for Bregman proximal methods: subgradient, gradient, and accelerated gradient schemes](https://arxiv.org/abs/1812.10198)
* [Proximal Algorithms, N. Parikh and S. Boyd](https://web.stanford.edu/~boyd/papers/prox_algs.html)
* [For more (non exhaustive) informations regarding the proximity operator and the associated proximal algorithms](http://proximity-operator.net/bibliography.html)

### Proximal and Projected Newton Methods

A fundamental difference between gradient descent and Newton’s
method was that the latter also iteratively minimized quadratic
approximations, but these used the local Hessian of the function in
question.

There are many such methods (`Ellipsoid method, AdaGrad`, ... ) with projection carried out in the $H_k$ metric to solve $\min_{x\in X}f(x)$:

1. get subgradient $g^{k}\in\partial f(x^K)$
2. update (diagonal) metric $H_k$
3. update $x^{k+1}=\operatorname{Proj}_{X}^{H_K}{\|x^k - H_k^{-1}g^k\|}_2^2$

where $\operatorname{Proj}_{X}^{H_K}(y)=\arg\min_{x\in X}{\|x-y\|}_{H_k}^2$ and ${\|x-y\|}_{H_K}^2=(x-y)^TH_k(x-y)$.

- [PROJECTED NEWTON METHODS FOR OPTIMIZATION PROBLEMS WITH SIMPLE CONSTRAINTS*](http://www.mit.edu/~dimitrib/ProjectedNewton.pdf)
- [Projected Newton Methods](http://www.mit.edu/~dimitrib/Gafni_Newton.pdf)
- [Mirror Descent and Variable Metric Methods](http://web.stanford.edu/class/ee364b/lectures/mirror_descent_slides.pdf)


`Proximal Newton method` can be also applied to such optimization problem:
$$
y^{k} = prox_{H_{k-1}}(x^{k-1}-H_{k-1}^{-1}\nabla g(x^{k-1}))\\
x^{k} = x^{k-1}+t_{k}(y^{k}-x^{k-1}).
$$

- [Proximal and Projected Newton Methods, Ryan Tibshirani, Convex Optimization 10-725/36-725](http://www.stat.cmu.edu/~ryantibs/convexopt-S15/lectures/24-prox-newton.pdf)


### Douglas–Rachford method

$$\min_{x}\{f(x)+g(x)\}$$
where $f$ and $g$ are closed convex functions.
Douglas–Rachford iteration: start at any $y^0$ and repeat for $k = 0, 1, \cdots,$
$$
x^{k+1} = prox_f(y^k)\\
y^{k+1} = y^k + prox_g(2x^{k+1} − y^k) − x^{k+1}
$$
+ useful when $f$ and $g$ have inexpensive prox-operators;
+ $x^k$ converges to a solution of $0 \in \partial f (x) + \partial g(x)$ (if a solution exists);
+ not symmetric in $f$ and $g$.

The iteration can be written as fixed-point iteration
$$y^{k+1} = F(y^k)$$
where $F(y)=y + prox_g(2prox_f(y) − y) − prox_f(y)$.

- http://www.seas.ucla.edu/~vandenbe/236C/lectures/dr.pdf

### Proximal point algorithms and its Beyond

Now, let us consider the simple convex optimization
$$\min_{x}\{\theta(x)+f(x)\mid x \in\mathcal X\},\tag{2.1}$$
where $\theta(x)$ and $f(x)$ are convex but $\theta(x)$ is not necessary smooth, $\mathcal X$ is a closed convex set.
For solving (2.1), the k-th iteration of the proximal point
algorithm (abbreviated to PPA) begins with a given $x^k$, offers the new
iterate $x^{k+1}$ via the recursion
$$x^{k+1}=\arg\min_{x}\{\theta(x)+f(x)+\frac{r}{2}{\|x-x^k\|}_2^2\mid x \in\mathcal X\}.\tag{2.2}$$

Since $x^{k+1}$ is the optimal solution of (2.2), it follows from optimal condition that
$$\theta(x)-\theta(x^{k+1})+(x-x^{k+1})^{T}\{ \nabla f(x^{k+1}) + r( f(x^{k+1}) - f(x^k))  \} \quad\forall x\in\mathcal X.$$

It is proved that The sequence $\{x^k\}$ generated by PPA is `Fejer monotone`, i.e.,
$\|x^{k+1}-x^{\ast}\|^2\leq \|x^{k}-x^{\ast}\|^2-\|x^{k+1}-x^{k}\|^2$.

- http://maths.nju.edu.cn/~hebma/Talk/VI-PPA.pdf
- https://lostella.github.io/software/
- [The proximal point method revisited](https://arxiv.org/pdf/1712.06038.pdf)

$\color{quad}{Note}$: the materials in this section is taken from the lectures of [Bingshneg He](http://maths.nju.edu.cn/~hebma/Talk/VI-PPA.pdf).

#### Prediction-Correction Methods

The optimal condition of the linearly constrained convex optimization is
characterized as a mixed monotone variational inequality:

$$w^{\ast}\in\Omega, \theta(u)-\theta(u^{\ast})+(w - w^{\ast})^{T} \nabla F(w^{\ast}) \quad\forall w\in\mathcal \Omega.$$

***
[Prediction Step.]
: With given $v^k$, find a vector $\tilde{w}^k\in \Omega$ such that
$$\theta(u)-\theta(\tilde u^{k})+(w - \tilde w^{k})^{T} \nabla F(w^{\ast})\geq (v-\tilde v^{k})^T Q (v-\tilde v^{k}),\quad w\in\Omega$$
where the matrix $Q$ is not necessary symmetric, but $Q^T+Q$ is positive definite.

[Correction Step.]
: The new iterate $v^{k+1}$ by
$$v^{k+1} =v^{k}- \alpha M(v^{k}-\tilde v^{k+1}).$$
***

Convergence Conditions
: For the matrices $Q$ and $M$ above, there is a positive definite matrix $H$ such that
$$HM = Q. \tag{2.20a}$$
Moreover, the matrix
$$G = Q^T + Q − \alpha M^T H M \tag{2.20b}$$
is positive semi-definite.

- https://www.seas.upenn.edu/~spater/assets/papers/conference/c_2019_paternain_et_al.pdf
- http://export.arxiv.org/pdf/1709.05850
- https://www.hindawi.com/journals/aaa/2013/845459/
- https://arxiv.org/pdf/1709.05850.pdf


#### Customized PPA

Now, let us consider the simple convex optimization
$$\min_{u}\{\theta(u)\mid Au=b, u \in\mathcal U\}.\tag{2.3}$$

The related variational inequality of the saddle point of the Lagrangian function is
$$w^{\ast}\in \Omega, \theta(u) - \theta(u^{\ast})+(w-w^{\ast})^T F(w^{\ast})\geq 0 \quad\forall w\in\Omega$$
where $w=(u, \lambda)^T, F(w)=(-A^T\lambda, Au-b)^T$.

For given $v^k = w^k = (u^k, \lambda^k)$, the predictor is given by
$$
\begin{aligned} \tilde{u}^{k} &=\arg \min \left\{L\left(u, \lambda^{k}\right)+\frac{r}{2}\left\|u-u^{k}\right\|^{2} | u \in \mathcal{U}\right\}, \\
\tilde{\lambda}^{k} &=\arg \max \left\{L\left(\left[2 \tilde{u}^{k}-u^{k}\right], \lambda\right)-\frac{s}{2}\left\|\lambda-\lambda^{k}\right\|^{2}\right\}. \end{aligned}
$$

* https://link.springer.com/article/10.1007/s10589-013-9616-x
* https://core.ac.uk/display/23878067
* http://www.optimization-online.org/DB_FILE/2017/03/5922.pdf

***
$\color{aqua}{Note}$: the projection from a point $x^0$ into a subset $C\subset\mathbb{R}^n$ is defined in proximal operator as

$$
x^+ =\arg\min_{x}\{\delta_C(x)+\frac{1}{2}{\|x-x^0\|}_2^2\}
$$

while it can also written in the following form:
$$
x^{+}=\arg\min_{x}\{\frac{1}{1_C(x)}\cdot {\|x-x^0\|}_2^2\}
$$

where
$$
1_C(x)=
 \begin{cases}
   1, & \text{if} \quad x \in C,\\
   0, & \text{otherwise}.
 \end{cases}
$$
How we can generalize this form into the proximal form? And what  is the difference with the original addition proximal operator?

If $x^0$ is in the set ${C}$, the projection can be rewritten in proximal operator:
$$
x^{+}=\arg\min_{x}\{\exp[\delta_C(x)]\cdot \frac{1}{2}{\|x-x^0\|}_2^2\}.
$$

How we can generalize the function $\delta_{C}$ to more general convex function? What are the advantages of this generalization?
As likelihood and log-likelihood, we can transfer the product of some functions to sum by taking logarithm.

$$
x^{+} = \arg\min_{x}\{\exp[\delta_C(x)]\cdot \frac{1}{2}{\|x-x^0\|}_2^2\}
\\ =  \arg\min_{x}\log\{\exp[\delta_C(x)]\cdot \frac{1}{2}{\|x-x^0\|}_2^2\}
\\=  \arg\min_{x} \{\delta_C(x)+\log({\|x-x^0\|}_2^2)\}.
$$

## Penalty/Barrier Function Methods

In projected gradient method,  it converts the constrained problem into  an unsmooth problem.

***
constrained problem|unconstrained problem|desired properties
---|---|---
$\arg\min_{x}\{f(x)\mid x\in \mathbb{S}\}$| $\arg\min_{x}\{f(x)+\delta_{\mathbb{S}}(x)\}$| unsmooth, inexpensive
$\arg\min_{x}\{f(x)\mid x\in \mathbb{S}\}$|$\arg\min_{x}\{f(x) + g_{\mathbb{S}}(x)\}$| smooth, inexpensive
$\arg\min_{x}\{f(x)\mid g(x)\in \mathbb{S}\}$| $\arg\min_{x}\{f(x)+\fbox{?}\}$|smooth, inexpensive



[In constrained optimization, a field of mathematics, a barrier function is a continuous function whose value on a point increases to infinity as the point approaches the boundary of the feasible region of an optimization problem.](https://en.wikipedia.org/wiki/Barrier_function)

Recall the indicator function of constrained set in projected methods:
 $$
\delta_{\mathbb{S}}(x)=
  \begin{cases}
    0, & \text{if} \quad x \in \mathbb{S};\\
    \infty, & \text{otherwise},
  \end{cases}
 $$
there is a dramatic leap between the boundary of $\mathbb S$ and the outer.

The basic idea of the barrier method is to approximate the indicator function by some functions ${f}$ satisfying:
* convex and differentiable;
* $f(x)< \infty$ if $x\in\mathbb S$;
* for every point constraint on the boundary $f(x)\to\infty$ as $x\to\partial\mathbb S$.

For example, [Barrier Method](http://www.stat.cmu.edu/~ryantibs/convexopt-F15/lectures/15-barr-method.pdf) in [Convex Optimization: Fall 2018](http://www.stat.cmu.edu/~ryantibs/convexopt/) pushs the inequality constraints in the
problem in a smooth way, reducing the problem to an equality-constrained problem.
Consider now the following minimization problem:
$$
\min_{x} f(x) \\
\text{subject to} \quad h_n \leq 0, n=1,2,\cdots, N \\
Ax=b
$$
where $f, h_1, h_2,\cdots, h_N$ are assumed to be convex and twice differential functions in $\mathbb R^p$.
The log-barrier function is defined as
$$\phi(x)=-\sum_{n=1}^{N}\log(-h_n(x)).$$

When we ignore equality constraints, the problem above
can be written by incorporating the inequality with the identity function, which in turn can be approximated
by the log-barrier functions as follows
$$\min_{x} f(x)+\sum_{n=1}^{N}\mathbb{I}_{h_n(x)\leq 0}\approx \min_{x} f(x)+\frac{1}{t}\phi(x)$$

for $t > 0$ being a large number.

[A penalty method replaces a constrained optimization problem by a series of unconstrained problems whose solutions ideally converge to the solution of the original constrained problem. The unconstrained problems are formed by adding a term, called a penalty function, to the objective function that consists of a penalty parameter multiplied by a measure of violation of the constraints. The measure of violation is nonzero when the constraints are violated and is zero in the region where constraints are not violated.](https://www.wikiwand.com/en/Penalty_method)

In the context of the equality-constrained problem
$$\min_{x} f(x), \,\, s.t. c_n(x)=0, n = 1,2,\cdots, N,$$

the quadratic penalty function $Q(x;\mu)$ for this formulation is
$$Q(x;\mu)=f(x)+\frac{\mu}{2}\sum_{n=1}^{N}c_n^2(x)$$
where $\mu>0$ is the `penalty parameter`. By driving $\mu\to\infty$, we penalize the constraint violations with increasing severity.

**Quadratic Penalty Method**

* Given $\mu_0 > 0$, a nonnegative sequence ${\tau_k}$ with $\tau_k\to 0$, and a starting point $x^s_0$;
* __for__ $k = 0, 1, 2, . . .$:
   * Find an approximate minimizer $x^k=\arg\min_{x}Q(x; \mu_k)$, starting at $x^s_k$ and  terminating when ${\|\nabla_x Q(x;\mu_k)\|}_2\leq \tau_k$;
   * __if__ final convergence test satisfied,
       * __stop__ with approximate solution $x^k$;
   * end __if__
   * Choose new penalty parameter $\mu_{k+1} > \mu_{k}$;
   * Choose new starting point $x^s_{k+1}$
 * end (__for__)

- [ ] [Line Search Procedures for the Logarithmic Barrier Function](https://epubs.siam.org/doi/abs/10.1137/0804013)
- [ ] [Penalty method](https://en.wikipedia.org/wiki/Penalty_method)
- http://s-mat-pcs.oulu.fi/~keba/Optimointi/OP_penalty_engl.pdf
- https://www.me.utexas.edu/~jensen/ORMM/supplements/units/nlp_methods/const_opt.pdf
- http://users.jyu.fi/~jhaka/opt/TIES483_constrained_indirect.pdf

## Path Following Methods

The main ideas of path following by predictor–corrector and piecewise-linear methods, and their application in the direction of homotopy methods and nonlinear eigenvalue problems are reviewed. Further new applications to areas such as polynomial systems of equations, linear eigenvalue problems, interior methods for linear programming, parametric programming and complex bifurcation are surveyed. Complexity issues and available software are also discussed.

- [Continuation and path following](https://www.cambridge.org/core/journals/acta-numerica/article/continuation-and-path-following/4368C662C0FA6F729FA4B2A5C1B60085)
- https://nisheethvishnoi.files.wordpress.com/2018/05/lecture71.pdf
- http://www.stat.cmu.edu/~ryantibs/convexopt-S15/lectures/16-primal-dual.pdf
- [L. Vandenberghe EE236C (Spring 2016): 17. Path-following methods](http://www.seas.ucla.edu/~vandenbe/236C/lectures/pf.pdf)

## Lagrange Multipliers and Duality

It is to solve the constrained optimization problem
$$\arg\min_{x}f(x), \quad s.t.\quad g(x)=b.$$

The barrier or penalty function methods are to add some terms to $f(x)+\Omega(g(x)-b)$ [converting
constrained problems into unconstrained problems by introducing an artificial penalty for
violating the constraint.](https://web.stanford.edu/group/sisl/k12/optimization/MO-unit5-pdfs/5.6penaltyfunctions.pdf)
For example,
$$P(x, \lambda)= f(x) + \lambda {\|g(x)-b\|}_2^2$$
where the penalty function $\Omega(x) = {\|x\|}_2^2$, $\lambda\in\mathbb{R}^{+}$.
We can regard it as a surrogate loss technique.
Although the penalty function is convex and differentiable, it is more difficult than directly optimizing $f(x)$ when the constrain is complicated.

### Lagrange Multipliers and Generalized Lagrange Function

The penalty function methods do not take the optimal conditions into consideration although it works.
If $x^{\star}$ is in the solution set of the optimization problem above, it is obvious that $Ax^{\star}=b$ and $L(x^{\star}, \lambda)=f(x^{\star})$ where
$$L(x, \lambda)= f(x) + \lambda^T(Ax-b).$$

In another direction, we want to prove that $x^{\star}$ is the optima of the optimization problem if
$$
L(x^{\star}, {\lambda}^{\star})=\quad \min_{x}[\max_{\lambda} L(x, \lambda)].
$$

By the definition,
$L(x^{\star}, {\lambda}^{\star})\geq L(x^{\star}, \lambda)=f(x^{\star})+\lambda^T(Ax^{\star}-b)$, which implies that $\lambda^T(Ax^{\star}-b)=0$ i.e., $Ax^{\star}=b$. And $L(x^{\star}, {\lambda}^{\star}) = f(x^{\star})\leq L(x, {\lambda}^{\star})\forall x$ if $Ax=b$
thus $x^{\star}$ is the solution to the primary problem.

It is dual problem is in the following form:
$$\max_{\lambda}[\min_{x} L(x, \lambda)].$$

Note that
$$
\min_{x} L(x, \lambda)\leq L(x,\lambda)\leq \max_{\lambda}L(x,\lambda)
$$
implies
$$\\
\max_{\lambda}[\min_{x} L(x, \lambda)]\leq \max_{\lambda} L(x, \lambda)\leq \min_{x}[\max_{\lambda} L(x, \lambda)].
$$

And note that necessary condition of extrema is that the gradients are equal to 0s:
$$
\frac{\partial L(x,\lambda)}{\partial x}= \frac{\partial f(x)}{\partial x}+ A\lambda = 0 \\
\frac{\partial L(x,\lambda)}{\partial \lambda} = Ax-b=0
$$

<img src="https://www.thefamouspeople.com/profiles/images/joseph-louis-lagrange-2.jpg" width="60%">

**Dual Ascent** takes advantages of this properties:

> 1. $x^{k+1}=\arg\min_{x} L(x,\lambda)$;
> 2. ${\lambda}^{k+1}= {\lambda}^{k}+\alpha_k(Ax^{k+1}-b)$.

***
[Entropy minimization algorithms](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-253-convex-analysis-and-optimization-spring-2012/lecture-notes/MIT6_253S12_lec24.pdf):
$$
x^{k+1}\in \arg\min_{x}\{f(x)+\frac{1}{c_k} \sum_{i=1}^{n} x^{i}(\ln(\frac{x_i}{x^{k}_{i}})-1)\}.
$$


- [Duality](http://www.ece.ust.hk/~palomar/ELEC5470_lectures/07/slides_Lagrange_duality.pdf)
- https://cs.stanford.edu/people/davidknowles/lagrangian_duality.pdf
- https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/lecture7.pdf
- https://www.svm-tutorial.com/2016/09/duality-lagrange-multipliers/
- https://www.cs.jhu.edu/~svitlana/papers/non_refereed/optimization_1.pdf
- [Constrained Optimization and Lagrange Multiplier Methods](http://web.mit.edu/dimitrib/www/lagr_mult.html)
- https://zhuanlan.zhihu.com/p/50823110
- [ ] [The proximal augmented Lagrangian method for nonsmooth composite optimization](https://arxiv.org/abs/1610.04514)

**Exponential Augmented Lagrangian Method**

A special case for the convex problem
$$
\text{minimize}\quad f(x),\\
s.t.  g_1(x) \leq 0, g_2(x) \leq 0, \cdots, g_r(x) \leq 0, x\in X
$$

is the [**exponential augmented Lagrangean method**](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-253-convex-analysis-and-optimization-spring-2012/lecture-notes/MIT6_253S12_lec24.pdf).

It consists of unconstrained minimizations:

$$
x^{k}\in \arg\min_{x\in X}\{f(x)+\frac{1}{c_k} \sum_{j=1}^{r} {\mu}^{k}_{j}\exp(c_k g_j(x))
$$
followed by the multiplier iterations
$$
{\mu}^{k+1}_{j} = {\mu}^{k}_{j}\exp(c_k g_j(x^k)).
$$

If the constraints are more complex, **KKT theorem** may be necessary.

* [An exponential augmented Lagrangian method with second order convergence](https://impa.br/wp-content/uploads/2016/12/maria_daniela_abstract.pdf)
* [On the convergence of the exponential
multiplier method for convex programming ](https://web.mit.edu/dimitrib/www/Expmult.pdf)

**Generalized Lagrangian function**

Minimize $f(x)$, subject to $g_i(x) \ge 0, i = 1,\cdots, m$.
Here, $x \in\Omega$, and $\Omega$ is a subset of the Euclidean space $E$. We assume that $f(x)$ and $g_i(x)$ are twice continuously differentiable.
A generalized Lagrangian function is defined  as
$$L(x, \sigma) = f(x) - G[g(x), \sigma]. $$

- [A generalized Lagrangian function and multiplier method](https://link.springer.com/article/10.1007%2FBF00933876)
- [Generalized Lagrange Function and Generalized Weak Saddle Points for a Class of Multiobjective Fractional Optimal Control Problems](https://link.springer.com/article/10.1007/s10957-012-0007-8)
- https://blog.csdn.net/shayashi/article/details/82529816
- https://suzyahyah.github.io/calculus/optimization/2018/04/07/Lagrange-Multiplier.html
- [Lagrange Multipliers without Permanent Scarring](https://people.eecs.berkeley.edu/~klein/papers/lagrange-multipliers.pdf)

**KKT condition**

Given general problem
$$
\begin{aligned}
\min_{x \in \mathbb{R}^{n}} f(x) & \\
\text { subject to } & h_{i}(x) \leq 0, \quad i=1, \ldots m \\
& \ell_{j}(x)=0, \quad j=1, \ldots r \end{aligned}
$$

We defined the Lagrangian:
$$L(x, u, v) = f(x) +\sum_{i=1}^{m}u_ih_i(x)+\sum_{j=1}^r v_j \ell_j.$$

The Karush-Kuhn-Tucker conditions or KKT conditions are:
$$\begin{aligned}
&\bullet\quad 0 \in \partial f(x)+\sum_{i=1}^{m} u_{i} \partial h_{i}(x)+\sum_{j=1}^{T} v_{j} \partial \ell_{j}(x) &\text{(stationarity)} \\
&\bullet\quad u_{i} \cdot h_{i}(x)=0 \text { for all } i  &\text{ (complementary slackness) } \\
&\bullet\quad h_{i}(x) \leq 0, \ell_{j}(x)=0 \text { for all } i, j  &\text{(primal feasibility) } \\
&\bullet\quad u_{i} \geq 0 \text{  for all  } i &\text{ (dual feasibility) }
\end{aligned}$$

I learnt this theorem in functional analysis at graduate level course.

- https://mitmgmtfaculty.mit.edu/rfreund/educationalactivities/
- [Nonlinear Programming by Robert M. Freund](https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/)
- https://www.cs.cmu.edu/~ggordon/10725-F12/slides/16-kkt.pdf

**Conjugate Duality**

Consider the standard form convex optimization problem in the absence of data uncertainty
$$\min_{x} f(x)\tag{P}$$

where $f$ is a proper lower semi-continuous convex functions.

This problem can be embedded into a family of parameterized problems
$$\min_{x} g(x, y)\tag{$P_y$}$$
where the function $g(x, y)$ satisfies $g(x, 0) = f(x)$.



* https://www.zhihu.com/question/58584814/answer/823769937
- [Conjugate Duality and Optimization](https://sites.math.washington.edu/~rtr/papers/rtr054-ConjugateDuality.pdf)
- [LECTURE 5: CONJUGATE DUALITY](https://www.ise.ncsu.edu/fuzzy-neural/wp-content/uploads/sites/9/2015/07/Lecture5-1.pdf)
- [Duality Theory of Constrained Optimization by Robert M. Freund](https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec18_duality_thy.pdf)
- [Gauge optimization, duality, and applications](https://www.researchgate.net/publication/257592332_Gauge_optimization_duality_and_applications)
- [Robust Conjugate Duality for Convex Optimization under Uncertainty with Application to Data Classification](https://web.maths.unsw.edu.au/~gyli/papers/ljl-conjugate-revised-final-18-11-10.pdf)
- https://mitmgmtfaculty.mit.edu/rfreund/educationalactivities/
- http://www.mit.edu/~mitter/publications/6_conjugate_convex_IS.pdf


### Splitting Methods

In following context we will talk the optimization problem with linear constraints:
$$
\arg\min_{x} f(x) \\
  s.t. Ax = b
$$
where $f(x)$ is always convex.

#### ADMM

Alternating direction method of multipliers is called **ADMM** shortly.
It is aimed to solve the following convex optimization problem:
$$
\min F(x,y) \{=f(x)+g(y)\}  \\
        \text{subject to }\quad  Ax+By =b
$$
where $f(x)$ and $g(y)$ is convex; ${A}$ and ${B}$ are matrices.

Define the augmented Lagrangian:

$$
L_{\beta}(x, y)=f(x)+g(y) - \lambda^{T}(Ax + By -b)+ \frac{\beta}{2}{\|Ax + By - b\|}_{2}^{2}.
$$

* [Augmented Lagrangian method](https://www.semanticscholar.org/topic/Augmented-Lagrangian-method/11373)
***

Augmented Lagrange Method at step $k$ is described as following:

> 1. $(x^{k+1}, y^{k+1})=\arg\min_{x\in\mathbf{X}}L_{\beta}(x, y,\lambda^{\color{aqua}{k}});$
> 2. $\lambda^{k+1} = \lambda^{k} - \beta (Ax^{\color{red}{k+1}} + By^{\color{red}{k+1}}-b).$

***
ADMM at step $t$ is described as following:

> 1. $x^{k+1}=\arg\min_{x\in\mathbf{X}}L_{\beta}(x,y^{\color{aqua}{k}},\lambda^{\color{aqua}{k}});$
> 2. $y^{k+1}=\arg\min_{y\in\mathbf{Y}} L_{\beta}(x^{\color{red}{k+1}}, y, \lambda^{\color{aqua}{k}});$
> 3. $\lambda^{k+1} = \lambda^{k} - \beta (Ax^{\color{red}{k+1}} + By^{\color{red}{k+1}}-b).$


The convergence  proof of ADMM in convex optimization can be reduced to [verifying the stability of a dynamical system](https://arxiv.org/pdf/1502.02009.pdf) or based on the optimal condition `variational inequality` like [On the O(1/t) convergence rate of alternating direction method](http://www.optimization-online.org/DB_FILE/2011/09/3157.pdf).

**Linearized ADMM**

Note that the $x$ subproblem in ADMM
$$
\begin{align}
\arg\min_{x}L_{\beta}(x,y^{\color{aqua}{k}},\lambda^{\color{aqua}{k}})\\
=\arg\min_{x}\{f(x)+g(y^k)+{\lambda^k}^{T}(Ax+By^{k}-b)+\frac{\beta}{2}{\|Ax+By^{k}-b\|}_{2}^{2}\} \\
=\arg\min_{x}f(x)+\frac{\beta}{2}{\|Ax+By^{k}-b-\frac{1}{\beta}\lambda\|}_{2}^{2}\tag 1
\end{align}
$$

However, the
solution of the subproblem (1) does not have the closed form solution because of the
general structure of the matrix ${A}$. In this case, we linearize the quadratic term of
$$\frac{\beta}{2}{\|Ax+By^{k}-b-\frac{1}{\beta}\lambda^k\|}_{2}^{2}$$

at $x^k$ and add a proximal term $\frac{r}{2}{\|x-x^k\|}_2^2$ to the objective function.
In another word, we solve the following ${x}$ subproblem if ignoring the constant term of the objective function:
$$ \min_{x}f(x)+\beta(Ax)^T(A x^k + B y^k - b-\frac{1}{\lambda^k}) + \frac{r}{2}{\|x - x^k\|}_2^2. $$

> 1. $x^{k+1}=\arg\min_{x\in \mathbf{X}} f(x) + \beta(A x)^T (A x^k + B y^k - b -\frac{1}{\lambda^k})+ \frac{r}{2}{\|x - x^k\|}_2^2$,
> 2. $y^{k+1}=\arg\min_{y\in\mathbf{Y}} L_{\beta}( x^{\color{red}{k+1}}, y, \lambda^{\color{aqua}{k}} )$,
> 3. $\lambda^{k+1} = \lambda^{k} - \beta (Ax^{\color{red}{k+1}} + By^{\color{red}{k+1}} - b).$

For given $\beta > 0$, choose ${r}$ such that
the matrix $rI_{1}-\beta A^TA$ is definitely positive, i.e.,
$$rI_{1}-\beta A^TA\geq 0.$$

We can also linearize  the ${y}$ subproblem:

> 1. $x^{k+1}=\arg\min_{x\in\mathbf{X}}L_{\beta}(x, y^k, \lambda^k)$,
> 2. $y^{k+1}=\arg\min_{y\in\mathbf{Y}} g(y) + \beta (By)^{T}(Ax^{\color{red}{k+1}} + B y^k - b -\frac{1}{\beta}\lambda^k) + \frac{r}{2}\|y - y^k\|^2$,
> 3. $\lambda^{k+1} = \lambda^{k} - \beta (Ax^{\color{red}{k+1}} + By^{\color{red}{k+1}}-b).$

For given $\beta > 0$, choose ${r}$ such that
the matrix $rI_{2}-\beta B^T B$ is definitely positive, i.e.,
$$rI_{2}-\beta B^T B\geq 0.$$

***
Taking $\mu\in(0, 1)$ (usually $\mu=0.9$), the **Symmetric ADMM** is described as

> 1. $x^{k+1}=\arg\min_{x\in\mathbf{X}}L_{\beta}(x, y^{\color{aqua}{k}},\lambda^{\color{aqua}{k}})$,
> 2. $\lambda^{k + \frac{1}{2}} = \lambda^{k} - \mu\beta (Ax^{\color{red}{k+1}} + By^{\color{red}{k}}-b)$,
> 3. $y^{k+1}=\arg\min_{y\in\mathbf{Y}} L_{\beta}(x^{\color{red}{k+1}}, y, \lambda^{\color{aqua}{k+\frac{1}{2}}})$,
> 4. $\lambda^{k+1} = \lambda^{\color{red}{k+\frac{1}{2}}} - \mu\beta (A x^{\color{red}{k+1}} + B y^{\color{red}{k+1}}-b)$.

* http://www.optimization-online.org/DB_FILE/2015/05/4925.pdf


$\color{aqua}{\text{Thanks to Professor He Bingsheng who taught me those.}}$[^9]

![He Bingsheng](https://pic1.zhimg.com/v2-bc583f2c01d8ac2a346982b1133753f9_1200x500.jpg)

- [Some recent advances in the linearized ALM, ADMM and Beyond Relax the crucial parameter requirements](http://maths.nju.edu.cn/~hebma/Talk/OptimalParameter.pdf)
- [Bing-Sheng He](https://www.researchgate.net/profile/Bing-Sheng_He)
***

One of the particular ADMM is also called `Split Bregman` methods. And `Bregman ADMM` replace the quadratic penalty function with Bregman divergence:
$$
L_{\beta}^{\phi}(x, y)=f(x)+g(y) - \lambda^{T}(Ax + By - b) + \frac{\beta}{2} B_{\phi}(b- Ax, By).
$$

where $B_{\phi}$ is the Bregman divergence induced by the convex function $\phi$.

**BADMM**

> 1. $x^{k+1}=\arg\min_{x\in\mathbf{X}}L_{\beta}^{\phi}(x,y^{\color{aqua}{k}},\lambda^{\color{aqua}{k}});$
> 2. $y^{k+1}=\arg\min_{y\in\mathbf{Y}} L_{\beta}^{\phi}(x^{\color{red}{k+1}}, y, \lambda^{\color{aqua}{k}});$
> 3. $\lambda^{k+1} = \lambda^{k} - \beta (Ax^{\color{red}{k+1}} + By^{\color{red}{k+1}}-b).$

* [Bregman Alternating Direction Method of Multipliers](https://arxiv.org/abs/1306.3203)
* https://www.swmath.org/software/20288


**Relaxed, Inertial and Fast ADMM**

$$\min_{x}f(x)+g(Ax)$$
The penalty parameter
is $\rho > 0$ and the relaxation parameter is $\alpha\in (0, 2)$. Standard ADMM is recovered with
$\alpha = 1$.
> Family of relaxed A-ADMM algorithms for the above problem
>
> 1. $x^{k+1}=\arg\min_{x\in\mathbf{X}}f(x)+\frac{\rho}{2}{\|Ax- y^k+ \lambda^k\|}^2$,
> 2. $y^{k+1}=\arg\min_{y\in\mathbf{Y}} g(y) + \frac{\rho}{2}{\|\alpha Ax^{\color{red}{k+1} }+(1-\alpha_k)y^k-y+\lambda^k\|}^2$,
> 3. $\lambda^{k+1} = \lambda^{k} +\alpha Ax^{\color{red}{k+1}}+(1-\alpha_k)y^k-z^{\color{red}{k+1}}$.


> Family of relaxed ADMM algorithms for the above problem
The damping constant is $r \geq 3$.
>
> 1. $x^{k+1}=\arg\min_{x\in\mathbf{X}}f(x)+\frac{\rho}{2}{\|Ax-\hat y^k+\lambda^k\|}^2$,
> 2. $y^{k+1}=\arg\min_{y\in\mathbf{Y}} g(y) + \frac{\rho}{2}{\|\alpha Ax^{\color{red}{k+1} }+(1-\alpha_k)\hat y^k-y+\lambda^k\|}^2$,
> 3. $\lambda^{k+1} = \hat\lambda^{k} +\alpha Ax^{\color{red}{k+1}}+(1-\alpha_k)\hat y^k-z^{\color{red}{k+1}}$,
> 4. $\gamma_{k+1}=\frac{k}{k+r}$,
> 5. $\hat\lambda^{k+1}=\lambda^{k+1}+\gamma_{k+1}(\lambda^{k+1}-\lambda^{k})$,
> 6. $\hat y^{k+1}=y^{k+1}+\gamma_{k+1}(y^{k+1}-y^{k})$.
____
> Fast ADMM
> 1. $x^{k+1}=\arg\min_{x\in\mathbf{X}}L_{\beta}(x,\hat y^{\color{aqua}{k}},\hat\lambda^{\color{aqua}{k}});$
> 2. $y^{k+1}=\arg\min_{y\in\mathbf{Y}} L_{\beta}(x^{\color{red}{k+1}}, y, \hat\lambda^{\color{aqua}{k}});$
> 3. $\lambda_{k+1} = \lambda^{k} - \beta (Ax^{\color{red}{k+1}} + By^{\color{red}{k+1}}-b).$
> 4. $\alpha_{k+1}=\frac{1+\sqrt{1+4\alpha_k^2}}{2}$
> 5. $\hat y^{k+1}=y^{k+1}+\frac{\alpha_{k}-1}{\alpha_{k+1}}(y^{k}-y^{k-1})$
> 6. $\hat \lambda^{k+1}=\lambda^{k+1}+\frac{\alpha_{k}-1}{\alpha_{k+1}}(\lambda^{k}-\lambda^{k-1})$


* [FAST ALTERNATING DIRECTION OPTIMIZATION METHODS](https://www.mia.uni-saarland.de/Publications/goldstein-cam12-35.pdf)
* [The Classical Augmented Lagrangian Method and Nonexpansiveness](http://rutcor.rutgers.edu/pub/rrr/reports2012/32_2012.pdf)
* [Relative-error inertial-relaxed inexact versions of Douglas-Rachford and ADMM splitting algorithms](https://arxiv.org/abs/1904.10502)
* [Relax, and Accelerate: A Continuous Perspective on ADMM](https://arxiv.org/abs/1808.04048v1)
* https://www.semanticscholar.org/author/Guilherme-Fran%C3%A7a/145512630
* [An explicit rate bound for over-relaxed ADMM](https://ieeexplore.ieee.org/document/7541670)

#### Multi-Block ADMM

Firstly we consider the following optimization problem

$$
\min f_1(x_1) + f_2(x_2) + \cdots + f_n(x_n)\\
s.t.\quad A_1x_1 + A_2x_2 + \cdots + A_n x_n = b, \\
x_i\in\mathop{X_i}\in\mathbb{R}^{d_i}, i=1,2,\cdots, n.
$$

We defined its augmented Lagrangian multipliers as

$$
L_{\beta}^{n}(x_1,x_2,\cdots, x_n\mid \lambda)=\sum_{i=1}^{n} f_i(x_i) -\lambda^T (\sum_{i=1}^{n} A_i x_i - b) + \frac{\beta}{2} {({\|\sum_{i=1}^{n} A_i x_i - b\|})}_{2}^{2}.
$$


Particularly, we firstly consider the case when $n=3$:

$$
L_{\beta}^{3}(x, y, z\mid \lambda)=f_1(x) + f_2(y) + f_3(z)-\lambda^T (A_1 x + A_2 y + A_3 z - b)
\\+\frac{\beta}{2}{\|A_1 x + A_2 y + A_3 z - b\|}_2^2.
$$

[It is natural and computationally beneficial to extend the original ADMM directly to solve the general n-block problem](https://web.stanford.edu/~yyye/MORfinal.pdf).
[A counter-example shows that this method diverges.](https://link.springer.com/article/10.1007/s10107-014-0826-5)

***
And [Professor Bingsheng He](http://maths.nju.edu.cn/~hebma/), who taught me this section in his class, and his coauthors proposed some schemes for this problem based on his unified frame work for convex optimization and monotonic variational inequality.

[Parallel splitting augmented Lagrangian method](https://link.springer.com/article/10.1007/s10589-007-9109-x) (abbreviated to `PSALM`) is described as follows:
> 1. $x^{k+1}=\arg\min_{x}\{L_{\beta}^3(x,y^k,z^k,\lambda^k)\mid x\in\mathbb{X}\}$;
> 2. $y^{k+1}=\arg\min_{x}\{L_{\beta}^3(x^{\color{red}{k+1}},y,z^k,\lambda^k)\mid y\in\mathbb{Y}\}$;
> 3. $z^{k+1}=\arg\min_{x}\{L_{\beta}^3(x^{\color{red}{k+1}},y^{\color{yellow}{k}},z,\lambda^k)\mid z\in\mathbb{Z}\}$;
> 4. $\lambda^{k+1} = {\lambda}^{k}-\beta(A_1x^{k+1}+A_2y^{k+1}+A_3z^{k+1}-b)$.

We can add one more correction step
$$
v^{k+1} := v^{k}-\alpha(v^k - v^{k+1}),\alpha\in (0, 2 -\sqrt{2})
$$

where $v^{k+1}=(y^{k+1},z^{k+1},\lambda^{k+1})$.

Another approach is to add an regularized terms:

> 1. $x^{k+1}=\arg\min_{x}\{L_{\beta}^3(x, y^k, z^k, \lambda^k)\mid x\in\mathbb{X}\}$,
> 2. $y^{k+1}=\arg\min_{x}\{L_{\beta}^3(x^{\color{red}{k+1}}, y, z^k,\lambda^k)+\color{red}{\frac{\tau}{2}\beta{\|A_2(y-y^k)\|}^{2}}\mid y\in\mathbb{Y}\}$,
> 3. $z^{k+1}=\arg\min_{x}\{L_{\beta}^3(x^{\color{red}{k+1}}, y^{\color{yellow}{k}}, z, \lambda^k)+\color{blue}{\frac{\tau}{2}\beta{\|A_3(z - z^k)\|}^{2}}\mid z\in\mathbb{Z}\}$,
> 4. $\lambda^{k+1} = {\lambda}^{k} - \beta(A_1 x^{k+1} + A_2 y^{k+1} + A_3 z^{k+1}-b)$,

where $\tau>1$.

- http://scis.scichina.com/en/2018/122101.pdf
- http://maths.nju.edu.cn/~hebma/slides/17C.pdf
- http://maths.nju.edu.cn/~hebma/slides/18C.pdf
- https://link.springer.com/article/10.1007/s10107-014-0826-5

****
**Davis-Yin three operator splitting**

If $f_1$ is strongly convex, then apply Davis-Yin (to dual problem) gives:

> 1. $x^{k+1}=\arg\min_{x}\{L^3_{\beta}(\color{green}{x},y^k,z^k,\lambda^k)\mid x\in\mathbb{X}\}$;
> 2. $y^{k+1}=\arg\min_{x}\{L_{\beta}^3(x^{k+1},\color{green}{y},z^k,\lambda^k)\mid y\in\mathbb{Y}\}$;
> 3. $z^{k+1}=\arg\min_{x}\{L_{\beta}^3(x^{k+1},y^{k+1},\color{green}{z},\lambda^k)\mid z\in\mathbb{Z}\}$;
> 4. $\lambda^{k+1} = {\lambda}^{k}-\beta(A_1x^{k+1}+A_2y^{k+1}+A_3z^{k+1}-b)$.

where the notation $L^3(x, y, z, \lambda)$ is deefined by
$$L^3(x, y, z, \lambda) = f_1(x) + f_2(y) + f_3(z)-{\lambda}^T(A_1 x + A_2 y + A_3 z - b)$$
is the Lagrangian rather than  augmented Lagrangian.

- [Three-Operator Splitting](http://fa.bianp.net/blog/2018/tos/)
- [A Three-Operator Splitting Scheme and its Optimization Applications](https://link.springer.com/article/10.1007%2Fs11228-017-0421-z)
- [Three-Operator Splitting and its Optimization Applications](http://www.math.ucla.edu/~wotaoyin/papers/pdf/three_op_splitting_wotao_yin_40_min.pdf)
- [A Three-Operator Splitting Scheme and its Optimization Applications](ftp://ftp.math.ucla.edu/pub/camreport/cam15-13.pdf)
- [A Three-Operator Splitting Scheme and its Applications](http://www.math.ucla.edu/~wotaoyin/papers/pdf/three_operator_splitting_ICCM16.pdf)

****

`Randomly Permuted ADMM` given initial values at round $k$ is described as follows:

1. Primal update
    * Pick a permutation $\sigma$ of ${1,.. ., n}$ uniformly at random;
    * For $i = 1,2,\cdots, n$, compute ${x}^{k+1}_{\sigma(i)}$ by
      $$
      x^{k+1}_{\sigma(i)}=\arg\min_{x_{\sigma(i)}} L(x^{k+1}_{\sigma(1)},\cdots, x^{k+1}_{\sigma(i-1)}, x_{\sigma(i)}, x^{k+1}_{\sigma(i+1)},\cdots\mid \lambda^{k}).
      $$
2. Dual update. Update the dual variable by
   $${\lambda}^{k+1}={\lambda}^{k}-\mu(\sum_{i=1}^{n}A_i x_i -b).$$

- [Randomly Permuted ADMM](https://web.stanford.edu/~yyye/MORfinal.pdf)
- [On the Expected Convergence of Randomly Permuted ADMM](http://opt-ml.org/oldopt/papers/OPT2015_paper_47.pdf)
- [On the Efficiency of Random Permutation for ADMM and Coordinate Descent](https://arxiv.org/abs/1503.06387)
- [Multi-Block ADMM and its Convergence Random Permutation Helps-A talk by Ye](https://community.apan.org/cfs-file/__key/docpreview-s/00-00-01-07-11/Ye.pdf)

****

**dlADMM**


$$
\begin{array}{l}{\mathrm{ Problem 1. }} \\
{\min _{W_{l}, b_{l}, z_{l}, a_{l}} R\left(z_{L} ; y\right)+\sum_{l=1}^{L} \Omega_{l}\left(W_{l}\right)} \\
{\text { s.t. } z_{l}=W_{l} a_{l-1}+b_{l}(l=1, \cdots, L), a_{l}=f_{l}\left(z_{l}\right)(l=1, \cdots, L-1)}\end{array}
$$
where $a_0\in\mathbb R^d$ is the input of the deep neural network where $n_0$ is the number of feature dimensions, and $y$ is a predefined label vector. $R\left(z_{L} ; y\right)$ is a risk function for the L-th layer, which is convex, continuous and proper, and $\Omega_{l}\left(W_{l}\right)$ is a regularization term for the l-th layer, which is also convex, continuous,
and proper.

Rather than solving Problem 1 directly, we can relax Problem 1 by adding an $\ell_2$ penalty to address Problem 2 as follows:

$$
\begin{array}{l}{\mathrm{ Problem 2 }} \\
 {\begin{aligned} \min _{W_{l}, b_{l}, z_{l}, a_{l}} F(\boldsymbol{W}, \boldsymbol{b}, z, \boldsymbol{a})=& R\left(z_{L} ; y\right)+\sum_{l=1}^{L} \Omega_{l}\left(W_{l}\right) \\
 +\underbrace{(v / 2) \sum_{l=1}^{L-1}\left(\left\|z_{l}-W_{l} a_{l-1}-b_{l}\right\|_{2}^{2}+\left\|a_{l}-f_{l}\left(z_{l}\right)\right\|_{2}^{2}\right)}_{\text{$\ell_2$ penalty} } \\
  \text { s.t. } z_{L}=W_{L} a_{L-1}+b_{L}
 \end{aligned}}\end{array}
$$

where $\mathbf{W}=\left\{W_{l}\right\}_{l=1}^{L}, \mathbf{b}=\left\{b_{l}\right\}_{l=1}^{L}, \mathbf{z}=\left\{z_{l}\right\}_{l=1}^{L}, \mathbf{a}=\left\{a_{l}\right\}_{l=1}^{L-1}$ and $\nu >0$
is a tuning parameter.

Compared with Problem 1, Problem 2 has only a linear constraint $z_L = W_La_L−1 + b_L$ and hence is easier
to solve

<img src="http://5b0988e595225.cdn.sohucs.com/images/20190829/b0776073d50048fabfdc89d90bb65258.png" width="70%" />

- [dlADMM: Deep Learning Optimization via Alternating Direction Method of Multipliers](https://github.com/xianggebenben/dlADMM)
- [Deep Learning Optimization via Alternating Direction Method of Multiplier](https://arxiv.org/abs/1905.13611)

***
* http://maths.nju.edu.cn/~hebma/
* http://stanford.edu/~boyd/admm.html
* [A General Analysis of the Convergence of ADMM](https://arxiv.org/pdf/1502.02009.pdf)
* [用ADMM实现统计学习问题的分布式计算](http://shijun.wang/2016/01/19/admm-for-distributed-statistical-learning/)
* https://www.wikiwand.com/en/Augmented_Lagrangian_method
* [凸优化：ADMM(Alternating Direction Method of Multipliers)交替方向乘子算法](https://blog.csdn.net/shanglianlm/article/details/45919679)

### Monotone Operator Splitting Methods for Optimization

Monotone operator splitting methods, which originated in the late 1970’s in the context of partial differential equations, have started to be highly effective for modeling and solving a wide range of data analysis and processing problems, in particular high-dimensional statistical data analysis.

Operator splitting is to decompose one complicated operator(procedure) into some simple operators (procedures).
For example, ADMM splits the maxmin operator of the augmented Lagrangian into 3 opertors:
$$
\arg\min_{x,y}\max_{\lambda} L_{\beta}(x,y\mid \lambda)
$$
to
$$
\arg\min_{x}L_{\beta}(x,y\mid \lambda) \circ
\\ \,\arg\min_{y}L_{\beta}(x,y\mid \lambda) \circ
\arg\max_{\lambda} L_{\beta}(x,y,\mid \lambda).
$$

<img src="https://simonsfoundation.imgix.net/wp-content/uploads/2018/12/04120318/OSFigure2-e1543943390750.png?auto=format&w=695&q=90" width="70%" />

They are really some block relaxation techniques.

+ https://web.stanford.edu/class/ee364b/lectures/monotone_split_slides.pdf
+ [Operator Splitting by Professor Udell @ORIE 6326: Convex Optimization](https://people.orie.cornell.edu/mru8/orie6326/lectures/splitting.pdf)
+ [A note on the equivalence of operator splitting methods](https://arxiv.org/pdf/1806.03353.pdf)
+ [Splitting methods for monotone operators with applications to parallel optimization](https://dspace.mit.edu/handle/1721.1/14356)
+ [Operator Splitting Methods for Fast MPC](http://www.syscop.de/files/2015ss/numopt/splitting.pdf)
+ https://staffportal.curtin.edu.au/staff/profile/view/Jie.Sun/
+ [Operator Splitting Methods in Data Analysis](https://www.simonsfoundation.org/event/operator-splitting-methods-in-data-analysis/)
+ https://www.samsi.info/programs-and-activities/research-workshops/operator-splitting-methods-data-analysis/
+ http://idda.cuhk.edu.cn/zh-hans/page/10297
+ [Random monotone operators and application to Stochastic Optimization](https://pastel.archives-ouvertes.fr/tel-01960496/document)
* [Splitting methods and ADMM, Thibaut Lienart](https://tlienart.github.io/pub/csml/cvxopt/split.html)
* [Operator Splitting Performance Estimation: Tight contraction factors and optimal parameter selection](https://arxiv.org/pdf/1812.00146.pdf)
* [Split Bregman](https://www.ece.rice.edu/~tag7/Tom_Goldstein/Split_Bregman.html)
* [Accelerated Bregman operator splitting with backtracking](https://www.aimsciences.org/article/doi/10.3934/ipi.2017048)
* [Splitting Algorithms, Modern Operator Theory, and Applications (17w5030)](https://www.birs.ca/cmo-workshops/2017/17w5030/files/)
* [17w5030 Workshop on Splitting Algorithms, Modern Operator Theory, and Applications](https://www.birs.ca/cmo-workshops/2017/17w5030/report17w5030.pdf)
* https://idda.cuhk.edu.cn/zh-hans/news/11729
* [MIngyi Hong](http://people.ece.umn.edu/~mhong/Publications_area.html)

### PDFP and PDHG

#### Primal-dual Fixed Point Algorithm

[We demonstrate how different algorithms can be obtained by splitting the problems in different ways through the classic example of sparsity regularized least square model with constraint. In particular, for a class of linearly constrained problems, which are of great interest in the context of multi-block ADMM, can be also solved by PDFP with a guarantee of convergence. Finally, some experiments are provided to illustrate the performance of several schemes derived by the PDFP algorithm.](http://math.sjtu.edu.cn/faculty/xqzhang/Publications/PDFPM_JCM.pdf)


* [A primal-dual fixed point algorithm for multi-block convex minimization](http://math.sjtu.edu.cn/faculty/xqzhang/Publications/PDFPM_JCM.pdf)
* [A primal–dual fixed point algorithm for convex separable minimization](http://math.sjtu.edu.cn/faculty/xqzhang/publications/CHZ_IP.pdf)
* [A Unified Primal-Dual Algorithm Framework Based on Bregman Iteration](https://link.springer.com/content/pdf/10.1007%2Fs10915-010-9408-8.pdf)
* [Proximal ADMM](https://www.birs.ca/cmo-workshops/2017/17w5030/files/ADMM%20for%20monotone%20operators%20convergence%20analysis%20and%20rates.pdf)
* [The Complexity of Primal-Dual Fixed Point Methods for Ridge Regression ](https://www.maths.ed.ac.uk/~prichtar/papers/pdfixedpoint.pdf),
* [A primal–dual fixed point algorithm for convex separable minimization with applications to image restoration](http://math.sjtu.edu.cn/faculty/xqzhang/publications/CHZ_IP.pdf)

#### Primary Dual Hybrid Gradient

The Primal-Dual Hybrid Gradient (PDHG) method, also known as the `Chambolle-Pock` method, is a powerful splitting method that can solve a wide range of constrained and non-differentiable optimization problems. Unlike the popular ADMM method, the PDHG approach usually does not require expensive minimization sub-steps.

PDHG solves general saddle-point problems of the form
$$\min_{x\in X}\max_{y\in Y} f(x)+\left<Ax, y\right>-g(y)$$
where  $f$ and  $g$ are convex functions, and $A$ is a linear operator.

___
> * $\hat x^{k+1}=x^k -\tau_k A^Ty^k$;
> * $x^{k+1}=\arg\min_{x\in X}f(x)+\frac{1}{2\tau_k}{\|x-\hat x^{k+1}\|}^2$;
> * $\tilde x^{k+1}=x^{k+1}+(x^{k+1}-x^{k})$;
> * $\hat y^{k+1}=y^k +\sigma_k A\tilde x^{k+1}$;
> * $y^{k+1}=\arg\min_{y\in Y} g(x)+\frac{1}{2\sigma_k}{\|y-\hat y^{k+1}\|}^2$.


* [Primary-dual hybrid gradient](https://www.cs.umd.edu/~tomg/projects/pdhg/)
* [Adaptive Primal-Dual Hybrid Gradient Methods for Saddle-Point Problems](https://arxiv.org/abs/1305.0546)
* [Convergence Analysis of Primal-Dual Algorithms for a Saddle-Point Problem: From Contraction Perspective](http://maths.nju.edu.cn/~hebma/paper/C-PPA/2012-SIAM-IM-HY.pdf)
* [An Algorithmic Framework of Generalized Primal-Dual Hybrid Gradient Methods for Saddle Point Problems](http://www.optimization-online.org/DB_HTML/2016/02/5315.html)
* [Stochastic Primal-Dual Hybrid Gradient Algorithm with Arbitrary Sampling and Imaging Applications](https://arxiv.org/abs/1706.04957)
* [A prediction-correction primal-dual hybrid gradient method for convex programming with linear constraints](http://www.scienceasia.org/2018.44.n1/scias44_34.pdf)
* [On the convergence of primal–dual hybrid gradient algorithms for total variation image restoration](http://www.unife.it/prin/pubblications/bonettini_JMIVrev_b.pdf)
* [A primal-dual algorithm framework for convex saddle-point optimization](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5656743/)
* http://www.cs.utah.edu/~ssingla/
* http://maths.nju.edu.cn/~hebma/paper/C-PPA/VI-PPA.pdf
* https://odlgroup.github.io/odl/index.html

## Linear Programming

A linear program (LP) is an optimization problem in which the objective function is linear in the unknowns and the constraints consist of linear equalities and linear inequalities.

The  standard problem is
$$\min_{x}\mathbf{c^Tx} \\ s.t.\quad \mathbf{Ax=b, x\geq 0}.$$

Here $\mathbf x$ is an n-dimensional column vector, $\mathbf{c^T}$ is an n-dimensional row vector, $\mathbf{A}$ is an $m \times n$ matrix, and $\mathbf b$ is an m-dimensional column vector. The vector inequality $\mathbf{x\geq 0}$ means that each component of $\mathbf x$ is nonnegative.

> Fundamental Theorem of Linear Programming. Given a linear program in standard form where $\mathbf A$ is an $m × n$ matrix of rank $m$,
> 1) if there is a feasible solution, there is a basic feasible solution;
> 2) if there is an optimal feasible solution, there is an optimal basic feasible solution.


Linear programming is constrainted convex optimization problem. It is the simplest case of constrainted optimization problem in theory. However, it is useful in many cases.

- [Application of Linear Programming (With Diagram)](http://www.economicsdiscussion.net/linear-programming/application/application-of-linear-programming-with-diagram/18783)
- [Applications of Linear Programming](https://homepages.rpi.edu/~mitchj/handouts/lp/lp.pdf)

If there is no constraints, the linear objectve function is unbounded.

- [EE236A - Linear Programming (Fall Quarter 2013-14)](http://www.seas.ucla.edu/~vandenbe/ee236a/ee236a.html)


## Fixed Point Iteration Methods

The fixed point algorithm is initially to find approximate solutions of the equation

$$f(x)=0\tag 1$$
where $f:\mathbb{R}\to\mathbb{R}$.

In this method, we first rewrite the question(1) in the following form
$$x=g(x)\tag 2$$

in such a way that any solution of the equation (2), which is a fixed point of the function ${g}$, is a solution of
equation (1). For example, we can set $g(x)=f(x)+x, g(x)=x-f(x)$.
Then consider the following algorithm.

> 1. Give the initial point $x^{0}$;
> 2. Compute the recursive procedure $x^{n+1}=g(x^n), n=1,2,\ldots$

So that finally we obtain an sequence $\{x^0, x^1, \cdots, x^{n},\cdots\}$. There are many methods to test whether this sequence is convergent or not as learnt in calculus.

To introduce the acceleration schemes, we define some order of convergence.

>  The order of convergence is defined as the constant $p$ such that $\lim_{n\to \infty}\frac{\| x^{n+1}-x^{\ast}\|}{\| x^{n}-x^{\ast}\|^p}=C$ if $\lim_{n\to\infty}x^{n}=x^{\ast}$, denoted as $O(\frac{1}{n^p})$.

name| the order of convergence
---|----
sublinear | $p<1$
linear | $p = 1$ and $C < 1,$
superlinear | $p >1$
quadratic| $p = 2$

### Solving nonlinear equations

Solving nonlinear equation
$$f(x)=0,$$
means to find such points $x^{\ast}$ such that $f(x^{\ast})=0$, where $f:\mathbb R\mapsto \mathbb R$ is nonlinear.

There are many methods for determining the value of $x^{\ast}$ by successive approximation.
With any such method we begin by choosing one or more values $x_0, x_1, \cdots, x_r$, more or less arbitrarily, and then successively obtain new values $x_n$, as certain functions of the previously obtained $x_0, x_1, x_2,\cdots, x_{n-1}$ and possibly those of the derivatives $f^{\prime}(x_0), \cdots, f^{\prime}(x_{n-1})$ or higher derivative information.

- [Numerical methods: Solving nonlinear equations](http://www.fyzikazeme.sk/mainpage/stud_mat/nm/lecture2.pdf)
- [Newton, Chebyshev, and Halley Basins of Attraction; A Complete Geometric Approach by Bart D. Stewart, Department of Mathematics, United States Military Academy](https://www.mi.sanu.ac.rs/vismath/stewart/index.html)
- [Solutions of Equations in One Variable
Fixed-Point Iteration II](http://www.math.ust.hk/~mamu/courses/231/Slides/ch02_2b.pdf)


**Bisection method**

Find a midpoint of interval $(a^k, b^k )$ and designate it $x^{k+1}=\frac{a^k+b^k}{2}$.

$$
\left(a_{k+1}, b_{k+1}\right)
=\left\{\begin{array}{ll}{\left(a_{k}, x_{k+1}\right),} & {\text { if } \quad f\left(a_{k}\right)  f\left(x_{k+1}\right)<0} \\
{\left(x_{k+1}, b_{k}\right),} & {\text { if } \quad f\left(a_{k}\right) f\left(x_{k+1}\right)>0}\end{array}\right.
$$

**Regula falsi (false position) method**

$$f(x^{k+1})=f(x^k)-\alpha \frac{b^k -a^{k}}{f(b^k) - f(a^{k})}f(x^k)$$

and

$$
\left(a_{k+1}, b_{k+1}\right)
=\left\{\begin{array}{ll}{\left(a_{k}, x_{k+1}\right),} & {\text { if } \quad f\left(a_{k}\right) f\left(x_{k+1}\right)<0} \\ {\left(x_{k+1}, b_{k}\right),} & {\text { if } \quad f\left(a_{k}\right) f\left(x_{k+1}\right)>0}\end{array}\right.
$$

**Secant method**

The $k$-th approximation of root is obtained by
$$f(x^{k+1})=f(x^k)-\alpha \frac{x^k -x^{k-1}}{f(x^k) - f(x^{k-1})}f(x^k)$$

- [Solutions of Equations in One Variable Secant & Regula Falsi Methods](http://www.math.ust.hk/~mamu/courses/231/Slides/ch02_3b.pdf)

**Newton’s method**

$$f(x^{k+1})=f(x^k)-\alpha {(f^{\prime})}^{-1}f(x^k)$$

![](https://www.mi.sanu.ac.rs/vismath/stewart/fig8.gif)

**Steffensen’s Method**

Steffensen’s method is modified Newton’s method

$$f(x^{k+1})=f(x^k)-\alpha {(\frac{f(x^k+f(x^k))-f(x^k)}{f(x^k)})}^{-1}f(x^k)$$

- http://www.fyzikazeme.sk/mainpage/stud_mat/nm/lecture2.pdf
- [On the development of Steffensen’s method and applications to Hammerstein equations](http://ceur-ws.org/Vol-1894/num3.pdf)
- [An improvement of Steffensen's method for solving nonlinear equations by N. Gattal1, and A. S. Chibi](https://www.ripublication.com/gjpam16/gjpamv12n1_80.pdf)

**Muller's Method**

Muller’s method is a generalization of the secant method, in the sense that it does
not require the derivative of the function.

It is an iterative method that requires three starting points $(p_0, f (p_0)), (p_1, f (p_1)),$ and $(p_2, f (p_2))$.
A parabola is constructed that passes through the three points; then the quadratic formula is used to find a root of the quadratic for the next approximation.

![](https://vignette.wikia.nocookie.net/mullersmethod/images/9/94/Mullers.jpg/)

- http://mathworld.wolfram.com/MullersMethod.html
- http://mathfaculty.fullerton.edu/mathews/n2003/mullersmethod/MullersMethodProof.pdf
- [Muller's Method Wiki](https://mullersmethod.fandom.com/wiki/Muller%27s_Method)

**Chebyshev Method**

![](https://www.mi.sanu.ac.rs/vismath/stewart/fig9.gif)

- http://mathworld.wolfram.com/ChebyshevIteration.html
- http://www.sam.math.ethz.ch/~mhg/pub/Cheby-02ParComp.pdf

**Halley's Methods**

![](https://www.mi.sanu.ac.rs/vismath/stewart/fig10.gif)

- [Newton's versus Halley's Method： A Dynamical Systems Approach](http://mathcs.holycross.edu/~groberts/Papers/nwt-hly.pdf)
- [MODIFIED HALLEY’S METHOD FOR SOLVING NONLINEAR FUNCTIONS WITH CONVERGENCE OF ORDER SIX AND EFFICIENCY INDEX 1.8171](https://ijpam.eu/contents/2016-111-1/6/6.pdf)
- http://benisrael.net/NEWTON-MONTHLY.pdf

**Cauchy's Methods**

- [Some New Variants of Cauchy's Methods for Solving Nonlinear Equations](https://www.hindawi.com/journals/jam/2012/927450/)
- [Some variants of Cauchy’s method with accelerated fourth-order convergence](https://core.ac.uk/download/pdf/82046338.pdf)

**Aitken’s $\Delta^2$ method**

- Let $\left\{p_{n}\right\}$ be generated by a method which has a linear convergence,
- Having $p_{0}, p_{1}$ and $p_{2}$ compute $\hat{p}_{0}=p_{0}-\frac{\left(p_{1}-p_{0}\right)^{2}}{p_{2}-2 p_{1}+p_{0}}, n=1,2, \ldots$
  1. compute $p_{n+2}$
  2. compute $\hat{p}_{n}=p_{n}-\frac{\left(p_{n+1}-p_{n}\right)^{2}}{p_{n+2}-2 p_{n+1}+p_{n}};$ and
  3. the algorithm terminates $p \approx \hat{p}_{n}$ if $\left|\hat{p}_{n}-\hat{p}_{n-1}\right|<\epsilon$.


* http://mathfaculty.fullerton.edu/mathews/n2003/AitkenSteffensenMod.html
* [2.6 - Accelerating Convergence Aitken’s Delta squared Method](http://macs.citadel.edu/chenm/343.dir/11.dir/lect2_6.pdf)
* [Aitken’s $\Delta^2$ method extended](https://www.tandfonline.com/doi/pdf/10.1080/23311835.2017.1308622)
* [Higher Order Aitken Extrapolation with Application to
Converging and Diverging Gauss-Seidel Iterations
](https://arxiv.org/abs/1310.4288)
* [AN ACCELERATION TECHNIQUE FOR SLOWLY CONVERGENT FIXED POINT ITERATIVE METHODS](http://users.jyu.fi/~oljumali/teaching/TIES594/14/fixedpoint.pdf)
* https://www.math.usm.edu/lambers/mat460/fall09/lecture13.pdf
* [Fixed-Point Iteration](https://www.csm.ornl.gov/workshops/applmath11/documents/posters/Walker_poster.pdf)
* [Lecture 8 : Fixed Point Iteration Method, Newton’s Method](http://home.iitk.ac.in/~psraj/mth101/lecture_notes/lecture8.pdf)
* [2.2 Fixed-Point Iteration](https://www3.nd.edu/~zxu2/acms40390F12/Lec-2.2.pdf)

**Homotopy Continuation Methods**

`Homotopy Methods` transform a hard problem into a simpler one whit easily calculated zeros and then gradually deform this simpler problem into the original one computing the zeros of the intervening problems and eventually ending with a zero of the original problem.

The homotopy method (continuation method, successive loading method) can be used to generate a good `starting value`.

Suppose one wishes to obtain a solution to a system of $N$ nonlinear equations in $N$ variables, say
$$F(x)=0$$
where $F : \mathbb R^N \to \mathbb R^N$ is a mapping which, for purposes of beginning our discussion we will assume is smooth.

Since we assume that such a priori knowledge is not available, the iteration will often fail, because poor starting values are likely to be chosen.

We construct a parameter depending function
$$H(x, s) = sF(x) + (1 − s)F_0(x), s\in [0,1]$$
and note, that $H(x, 0) = 0$ is the problem with known solution and
$H(x, 1) = 0$ is the original problem $F(x) = 0$.
As the solution of $H(x, s) = 0$ depends on s we denote it by $x^{\ast}(s)$.
We discretize now the intervall into $0 = s_0 < s_1 < \cdots < s_n = 1$ and solve a sequence of nonlinear systems with Newton’s method
$$H(x, s_i) = 0$$

- http://homepages.math.uic.edu/~jan/
- http://people.bu.edu/fdc/H-topy.pdf
- [3.10: Homotopy Method](http://www.maths.lth.se/na/courses/FMN081/FMN081-06/lecture8.pdf)
- [CHAPTER 2: Numerical homotopy continuation](http://people.math.gatech.edu/~aleykin3/math4803spr13/BOOK/chapter2.pdf)
- https://blog.csdn.net/jbb0523/article/details/52460408
- [Homotopy Continuation, Max Buot (CMU)&Donald Richards (PSU)](https://astrostatistics.psu.edu/su05/max_homotopy061605.pdf)
- [On the Link Between Gaussian Homotopy Continuation and Convex Envelopes](http://people.csail.mit.edu/hmobahi/pubs/gaussian_convenv_2015.pdf)
- [HOPE: A Homotopy Optimization Method for Protein Structure Prediction](http://www.cs.umd.edu/~oleary/reprints/j73.pdf)
- [Solving inequality Constrained Optimization Problems by Differential Homotopy Continuation Methods ](https://core.ac.uk/download/pdf/82350656.pdf)
- [PHCpack: a general-purpose solver
for polynomial systems by homotopy continuation](http://homepages.math.uic.edu/~jan/PHCpack/phcpack.html)
- [APPLICATION OF HOMOTOPY ANALYSIS METHOD FOR SOLVING NONLINEAR CAUCHY PROBLEM](http://www.utgjiu.ro/math/sma/v07/a08.html)

---

In high dimensional space, it is a little different. Fixed point iteration as well as the fixed point itself arises in many cases such as [https://arxiv.org/pdf/1511.06393.pdf].

The contracting mapping ${F}:\mathbb{R}^{d}\to\mathbb{R}^{d}$ is defined as
$$\|F(x)-F(y)\|\leq \alpha\|x-y\|, \forall x,y \in\mathbb{R},\alpha\in[0,1).$$
Thus $\lim_{\|x-y\|\to 0}\frac{\|F(x)-F(y)\|}{\|x-y\|}\leq \alpha\in[0,1)$.

Now we rewrite  the necessary condition of unconstrainted optimization problems $\nabla f(x) = 0$ to the fixed point equation:

$$
\begin{align}
\nabla f(x) = 0 \Rightarrow & x - \alpha\nabla f(x) = x \\
\nabla f(x) = 0 \Rightarrow  g(x) - \alpha\nabla f(x) = g(x) \Rightarrow & x - g^{-1}(\alpha\nabla f(x)) = x\\
\nabla f(x) = 0\Rightarrow H(x)x- \alpha\nabla f(x) = H(x)x
\Rightarrow & x - \alpha H(x)^{-1} \nabla f(x) = x \\
\nabla f(x) = 0 \Rightarrow M(x)\nabla f(x) = 0 \Rightarrow & x -\alpha M(x)\nabla f(x) = x
\end{align}
$$

where $H(x)$ is the lambda-matrix.

These correspond to gradient descent, mirror gradient methods, Newton's methods and quasi-Newton's methods, respectively.


And the projected (sub)gradient methods are in the fixed-point iterative form:
$$
x = Proj_{\mathbb{S}}(x-\alpha \nabla f(x))
$$

as well as the mirror gradient and proximal gradient methods different from the projection operator.

Expectation maximization is also an accelerated [fixed point iteration](https://www.csm.ornl.gov/workshops/applmath11/documents/posters/Walker_poster.pdf) as well as Markov chain.

![http://www.drkhamsi.com/fpt/books.html](http://www.drkhamsi.com/fpt/fix2small.gif)

The following figures in the table is form [Formulations to overcome the divergence of iterative method
of fixed-point in nonlinear equations solution](http://www.scielo.org.co/pdf/tecn/v19n44/v19n44a15.pdf)

|Fixed Point Iterations||
|---|---|
|<img src="http://www.scielo.org.co/img/revistas/tecn/v19n44/v19n44a15f1.jpg"  width="70%" />|<img src="http://www.scielo.org.co/img/revistas/tecn/v19n44/v19n44a15f2.jpg" width="70%"/>|
|<img src="http://www.scielo.org.co/img/revistas/tecn/v19n44/v19n44a15f3.jpg"  width="70%" />|<img src="http://www.scielo.org.co/img/revistas/tecn/v19n44/v19n44a15f4.jpg"  width="70%" />|

* https://www.wikiwand.com/en/Fixed-point_theorem
* [FixedPoint: A suite of acceleration algorithms with Application](https://cran.r-project.org/web/packages/FixedPoint/vignettes/FixedPoint.pdf)
* [Books on Fixed Point Theory](http://www.drkhamsi.com/fpt/books.html)
* [Recent Advances in Convex Optimization and Fixed Point Algorithms by Jean-Christophe Pesquet](https://www.i2m.univ-amu.fr/seminaires_signal_apprentissage/Slides/2015_04_28_Pesquet_course_main.pdf)
* [Acceleration methods TIES594 PDE-solvers, Lecture 14, 6.5.2015, Olli Mali](http://users.jyu.fi/~oljumali/teaching/TIES594/14/fixedpoint.pdf)
* [The SUNNonlinearSolver_FixedPoint implementation¶](http://runge.math.smu.edu/arkode_dev/doc/guide/build/html/sunnonlinsol/SUNNonlinSol_FixedPoint.html)

###  Gaussian-Seidel Method and Component Solution Methods

The mapping $\hat T_i :X\mapsto X$, corresponding to an update of the ith block-component only, is given by
$$\hat T_i(x)=\hat T_i(x_1, \dots, x_m)=(x_1,\dots, x_{i-1}, T_i(x), x_{i+1}, \dots, x_m).$$

Updating all block-components of $x$, one at a time in increasing order, is equivalent to applying the mapping $S: X\mapsto X$, defining by
$$S=\hat{T}_m\circ \hat{T}_{m-1}\circ\cdots \circ \hat{T}_1,$$
where $\circ$ denotes composition.  
An equivalent definition of $S$ is given by the equation
$$S_i=T_{i}(S_1(x), \dots, S_{i-1}(x), x_i, \dots, x_m)$$
where $S_i:X\mapsto X_i$ is the $i$th block-component of $S$.
The mapping $S$ will be called the `Gaussian-Seidel mapping` based on the mapping $T$ and the iteration $x(t+1)=S(x(t))$ will be called `Gaussian-Seidel algorithm` based on the mapping $T$.

The system $x=T(x)$ can be decomposed into m smaller system of equations of the form
$$x_i=T_i(x_1,\dots, x_m), \quad i=1,\dots,m,$$
which have to be solved simultaneously. We will consider an algorithm that solves at iteration the $i$th equation in the system for $x_i$, while keeping the other component fixed.

Given a vector $x(t)\in X$, the $i$th block-component $x_i(t+1)$ of the next vector is chosen to be a solution of the $i$th equation in the system, that is,
$$x_i(t+1)\in\{y_i\in X_i\mid y_i=T_i(x_1,\dots,x_{i-1}, y_i, x_{i+1},\dots, x_m)\}.$$

### ISTA and FASTA

The $\ell_1$ regularization is to solve the ill-conditioned equations such as
$$\min_{x}{\|Ax-b\|}_2^2+\lambda{\|x\|}_1.$$

It is less sensitive to outliers and  obtain much more sparse solutions (as opposed to $\ell_2$ regularization).
Its application includes and is not restricted in *LASSO*, *compressed sensing* and *sparse approximation  of signals*.

It is clear that the absolute value function is not smooth or differentiable  everywhere even the objective function ${\|Ax-b\|}_2^2+\lambda{\|x\|}_1$ is convex.
It is not best to solve this problem by gradient-based methods.

**Iterative Shrinkage-Threshold Algorithms(ISTA)** for $\ell_1$ regularization is
$$x^{k+1}=\mathbf{T}_{\lambda t}(x^{k}-tA ^{T}(Ax-b))$$
where $t> 0$ is a step size and $\mathbf{T}_{\alpha}$ is the shrinkage operator defined by
$${\mathbf{T}_{\alpha}(x)}_{i}={(x_i-\alpha)}_{+}sgn(x_{i})$$
where $x_i$ is the $i$ th component of $x\in\mathbb{R}^{n}$.

**FISTA with constant stepsize**

> * $x^{k}= p_{L}(y^k)$ computed as ISTA;
> * $t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2}$;
> * $y^{k+1}=x^k+\frac{t_k -1}{t_{k+1}}(x^k-x^{k-1})$.

* [A Fast Iterative Shrinkage Algorithm for Convex Regularized Linear Inverse Problems](https://www.polyu.edu.hk/~ama/events/conference/NPA2008/Keynote_Speakers/teboulle_NPA_2008.pdf)
* https://pylops.readthedocs.io/en/latest/gallery/plot_ista.html
* [ORF523: ISTA and FISTA](https://blogs.princeton.edu/imabandit/2013/04/11/orf523-ista-and-fista/)

This will lead to the operator splitting methods analysesed by [Wotao Yin](http://www.math.ucla.edu/~wotaoyin/index.html) and others.


* [ORIE 6326: Convex Optimization Operator Splitting](https://people.orie.cornell.edu/mru8/orie6326/lectures/splitting.pdf)
* [Monotone Operator Splitting Methods](https://web.stanford.edu/class/ee364b/lectures/monotone_split_slides.pdf)
* [A Course on First-Order, Operator Splitting, and Coordinate Update Methods for Optimization](http://www.math.ucla.edu/~wotaoyin/summer2016/)
* [Operator Splitting Methods for Convex Optimization Analysis and Implementation](http://people.ee.ethz.ch/~gbanjac/pdfs/banjac_thesis.pdf)
* [Some Operator Splitting Methods for Convex Optimization](https://repository.hkbu.edu.hk/cgi/viewcontent.cgi?article=1042&context=etd_oa)
* [FAST ALTERNATING DIRECTION OPTIMIZATION METHODS](https://www.mia.uni-saarland.de/Publications/goldstein-cam12-35.pdf)
* https://www.math.ucla.edu/~wotaoyin/math285j.18f/
* [Fixed-Point Continuation for $\ell_1$-Minimization: Methodology and Convergence@https://epubs.siam.org/doi/abs/10.1137/070698920](https://epubs.siam.org/doi/abs/10.1137/070698920)


### Generic Acceleration Framework

+ Given
  - existing optimization procedure $M(f, x)$
  - previous iterates $x^{1},x^{2}, \cdots ,x^{k}$ and
  - new proposed guess $x^P = M(f, x^{k})$.
+ Find step $x^{k+1}$ using information at $x^{1},x^{2}, \cdots ,x^{k}, x^P$.

* [Discover acceleration](https://ee227c.github.io/notes/ee227c-lecture06.pdf)
* [The zen of gradient descent](http://blog.mrtz.org/2013/09/07/the-zen-of-gradient-descent.html)
* [Introduction to Optimization Theory MS&E213 / CS269O - Spring 2017 Chapter 4 Acceleration](http://www.aaronsidford.com/chap_4_acceleration_v2.pdf)
* [Algorithms, Nature, and Society](https://nisheethvishnoi.wordpress.com/)
* https://damienscieur.com/sections/paper.html
* [Generalized Framework for Nonlinear Acceleration](https://arxiv.org/abs/1903.08764v1)
* [Nonlinear Acceleration of Stochastic Algorithms](https://papers.nips.cc/paper/6987-nonlinear-acceleration-of-stochastic-algorithms.pdf)
* [Nonlinear Acceleration of Constrained Optimization Algorithms](https://ieeexplore.ieee.org/document/8682962)
* [Cheat Sheet: Acceleration from First Principles](http://www.pokutta.com/blog/research/2019/06/10/cheatsheet-acceleration-first-principles.html)

**Relaxation and inertia**

We will focus here on
- unit memory,
- accelerations using past operation outputs OR iterates.

Given an fixed point iteration $x^{k+1}={T}(x^k)$, there are two simple acceleration schemes.

$$\begin{aligned}
x^{k+1}&=T_{1}\left(x^{k}+\nu^{k}\left(x^{k}-x^{k-1}\right)\right) \\
x^{k+2}&=T_{2}\left(x^{k+1}+\nu^{k+1}\left(x^{k+1}-x^{k}\right)\right)
\end{aligned}$$

Relaxation | Inertia | Alternated Inertia
---|---|---
$T_1=T, T_2=I$|$T_1=T, T_2=T$|$T_1=T, T_2=I$
$\nu^{k}=0, \nu^{k+1}=\eta^{k / 2}-1$ | $\nu^{k}=\gamma^{k}, \nu^{k+1}=\gamma^{k+1}$ | $\nu^{k}=0, \nu^{k+1}=\gamma^{k+1}$
${x^{k+1}=T\left(x^{k}\right)}$|$x^{k+1}=T(x^{k}+\gamma^{k}\left(x^{k}-x^{k-1}\right))$|${x^{k+1}=T\left(x^{k}\right)}$
$x^{k+2}=x^{k+1}+\nu^{k+1}\left(x^{k+1}-x^{k}\right)$|  $x^{k+2}=T(x^{k+1}+\gamma^{k+1}\left(x^{k+1}-x^{k-1}\right))$ | $x^{k+2}=T(x^{k+1}+\gamma^{k+1}\left(x^{k+1}-x^{k-1}\right))$

* [A Generic online acceleration scheme for Optimization algorithms via Relaxation and Inertia](https://arxiv.org/abs/1603.05398)
* [RELAXATION AND INERTIA IN FIXED-POINT ITERATIONS WITH APPLICATIONS](http://bipop.inrialpes.fr/people/malick/Docs/15-titan-iutzeler.pdf)
* [Weak Convergence of a Relaxed and Inertial Hybrid Projection-Proximal Point Algorithm for Maximal Monotone Operators in Hilbert Space](https://epubs.siam.org/doi/10.1137/S1052623403427859?mobileUi=0)
* [FIRE: Fast Inertial Relaxation Engine for Optimization on All Scales](http://users.jyu.fi/~pekkosk/resources/pdf/FIRE.pdf)
* [Structural Relaxation Made Simple](https://www.math.uni-bielefeld.de/~gaehler/papers/fire.pdf)
* [Monotonicity, Acceleration, Inertia, and the Proximal Gradient algorithm](http://www.iutzeler.org/pres/osl2017.pdf)
* [Online Relaxation Method for Improving Linear Convergence Rates of the ADMM](http://beneluxmeeting.eu/2015/uploads/papers/bmsc15_final_478.pdf)
- http://www.iutzeler.org/
- https://www.math.uni-bielefeld.de/~gaehler/
- https://www.researchgate.net/profile/Damien_Scieur

### Anderson Acceleration

Let $H$ be a Hilbert space equipped with a symmetric inner product $\left<\cdot, \cdot\right>: H \times H \to R$. Let
$T : H \to H$ be a `nonexpansive` mapping and consider for fixed $x_0 \in H$ the `Halpern-Iteration` (named after Benjamin Halpern, who introduced it):
$$x^{k+1}= (1-{\alpha}_k)x^0+ {\alpha}_k T(x^k), {\alpha}_k \in (0,1)$$

with ${\alpha}_k = \frac{k+1}{k+2}$ for approximating a fixed point of $T$.

It is proved that $\frac{1}{2}{\| x^k -T(x^k)\|}\leq\frac{\|x^0-x^{\ast} \|}{k+1}$.

* [On the Convergence Rate of the Halpern-Iteration](http://www.optimization-online.org/DB_FILE/2017/11/6336.pdf)

`Krasnosel'skii-Mann(KM, or averaged) iterations` update $x^k$
in iteration ${k}$ to

$$
x^{k+1}=(1-\alpha_k)x^k+ {\alpha}_k T(x^k), {\alpha}_k \in (0,1)
$$

for the fixed point problem (2). Specially, $T(x^k)= x^k-\alpha_k \nabla f(x^k)$ for the convex optimization problems.
If the operator $T$ is non-expensive, then the sequence $\{x^{k}\mid k=0,1,2,\dots\}$ is convergent by `Krasnosel'skii-Mann Theorem`.

In 1953, [Mann](http://www.ams.org/journals/proc/1953-004-03/S0002-9939-1953-0054846-3/S0002-9939-1953-0054846-3.pdf) defined an iterative method:
$$
x^{k+1}=M(x^k, \alpha_k , T)=(1-\alpha_k)x^k+ {\alpha}_k T(x^k),\\
\alpha_k \in [0,1)\,\,\text{satisfying}\, \sum_{k=1}^{\infty}{\alpha}_k = \infty.
$$

The sequence $\{x^{k}\}$ defined by

$$
x^{k+1}= (1-\alpha_k)x^k+ {\alpha}_k T(x^k), 0 < a\leq {\alpha}_k \leq b < 1
$$
(additionally $\sum_{k=1}^{\infty}{\alpha}_k = \infty$ )is called a `modied Mann iteration`.
Take $\{\alpha_k, \beta_k\}$ two sequences in $[0, 1]$ satisfying
$$
\sum_{k=1}^{\infty}{\alpha}_k {\beta}_k = \infty, \lim_{k\infty}\beta_k =0, 0 \leq {\alpha}_k \leq \beta_k \leq 1.
$$

Then the sequence $\{x^{k}\}$ defined by

$$
x^{k+1}= (1-\alpha_k)x^k+ {\alpha}_k T(y^k), \\
y^{k} = (1-\beta_k)x^k + {\beta}_k T(x^k),
$$

is called the `Ishikawa iteration`.

[Hojjat Afshari and Hassen Aydi](https://www.isr-publications.com/jnsa/articles-2534-some-results-about-krasnoselskii-mann-iteration-process) proposes another Mann type iteration:
$$
x^{k+1} = {\alpha}_k x^k + {\beta}_k T(x^k) + {\gamma}_k T^{2}(x^k)
$$

where ${\alpha}_k + {\beta}_k + {\gamma}_k = 1, {\alpha}_k, {\beta}_k, {\gamma}_k \in [0,1)$ for all $k\geq 1$, $\sum_{k}^{\infty}(1-\alpha_k)=\infty$, $\sum_{k=1}^{\infty}\gamma_k < \infty$.

If the Mann type iteration $\{x^k\}$ converges strongly to a point $p$, then $p$ is a fixed point of $T$.

* [Krasnoselskii-Mann method for non-self mappings](https://fixedpointtheoryandapplications.springeropen.com/track/pdf/10.1186/s13663-015-0287-4)
* http://www.krasnoselskii.iitp.ru/papereng.pdf
* [Strong convergence of the modified Mann iterative method for strict pseudo-contractions](https://www.sciencedirect.com/science/article/pii/S0898122108006330)
* [Some results on a modified Mann iterative scheme in a reflexive Banach space](https://fixedpointtheoryandapplications.springeropen.com/articles/10.1186/1687-1812-2013-227)
* [Some results about Krasnosel'skiĭ-Mann iteration process](https://www.isr-publications.com/jnsa/articles-2534-some-results-about-krasnoselskii-mann-iteration-process)
* [Convergence theorems for inertial KM-type algorithms](https://www.sciencedirect.com/science/article/pii/S0377042707003901)
* [Modified inertial Mann algorithm and inertial CQ-algorithm for nonexpansive mappings](https://scinapse.io/papers/2562808805)

There is an acceleration framework of fixed point iterations for the problem (2) called `Anderson Acceleration` or `regularized nonlinear acceleration`：

> 1. $F_k = (h_{k-m_k}, \dots, h_k)$ where $h_i=g(x_i)-x_i$;
> 2. Determine $\alpha^{k}=(\alpha_0^{k},\dots, \alpha_{m_k}^{k})^{T}$ that solves $\min_{\alpha^{k}}{\|F_k\alpha^k\|}_2^2$ s.t. $\sum_{i=0}^{m_k}\alpha_i^{k}=1$;
> 3. Set $x_{k+1}=\sum_{i=0}^{m_k} \alpha_i^{k} g(x_{k - m_k + i})$.

It is maybe interesting to introduce some Bregman divergence $D_{f}(\cdot, \cdot)$ instead of the squared $\ell_2$ norm when choosing $\alpha^{k}$ so that
$$\alpha^{k}=\arg\min_{\alpha^k}\{D_{f}(F_k \alpha^k)\mid \sum_{i=0}^{m_k}\alpha_i^{k}=1\}.$$
Thus we would use mirror gradient methods to solve this problem.

[It is proved that the Anderson acceleration converges if the fixed point mapping is cotractive.](https://www.casl.gov/sites/default/files/docs/CASL-U-2014-0226-000.pdf)


* [Anderson acceleration for fixed point iterations](https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf)
* [Anderson Acceleration](https://nickhigham.wordpress.com/2015/08/05/anderson-acceleration/)
* [MS142 Anderson Acceleration and Applications](http://meetings.siam.org/sess/dsp_programsess.cfm?SESSIONCODE=19874)
* [Convergence Analysis For Anderson Acceleration](https://www.casl.gov/sites/default/files/docs/CASL-U-2014-0226-000.pdf)
* [Comments on "Anderson Acceleration, Mixing and Extrapolation"](https://dash.harvard.edu/handle/1/34773632)
* [Globally Convergent Type-I Anderson Acceleration for Non-Smooth Fixed-Point Iterations](http://59.80.44.49/web.stanford.edu/~boyd/papers/pdf/scs_2.0_v_global.pdf)
* [A proof that Anderson acceleration improves the convergence rate in linearly converging fixed point methods (but not in those converging quadratically)](https://arxiv.org/abs/1810.08455)

[The Baillon-Haddad Theorem provides an important link between convex
optimization and fixed-point iteration,](http://faculty.uml.edu/cbyrne/BHSeminar2015.pdf) which proves that if the gradient of a convex and continuously differentiable function is non-expansive, then it is actually `firmly non-expansive`.

* [The Baillon-Haddad Theorem Revisited](https://people.ok.ubc.ca/bauschke/Research/60.pdf)
* [A Generic online acceleration scheme for Optimization algorithms via Relaxation and Inertia](https://arxiv.org/abs/1603.05398)
* [RELAXATION AND INERTIA IN FIXED-POINT ITERATIONS WITH APPLICATIOn](http://bipop.inrialpes.fr/people/malick/Docs/15-titan-iutzeler.pdf)
* [Monotonicity, Acceleration, Inertia, and the Proximal Gradient algorithm](http://www.iutzeler.org/pres/osl2017.pdf)
* [Iterative Convex Optimization Algorithms; Part One: Using the Baillon–Haddad Theorem](http://faculty.uml.edu/cbyrne/BHSeminar2015.pdf)
* [Recent Advances in Convex Optimization and Fixed Point Algorithms by Jean-Christophe Pesquet](https://www.i2m.univ-amu.fr/seminaires_signal_apprentissage/Slides/2015_04_28_Pesquet_course_main.pdf)
* [A FIXED-POINT OF VIEW ON GRADIENT METHODS FOR BIG DATA](https://arxiv.org/pdf/1706.09880.pdf)

#### Anderson Acceleration of the Alternating Projections Method for Computing the Nearest Correlation Matrix

A correlation matrix is symmetric, has unit diagonal, and is positive semidefinite. Frequently, asynchronous or missing observations lead to the obtained matrix being indefinite.

A standard way to correct an invalid correlation matrix, by which we mean a real, symmetric indefinite matrix with unit diagonal, is to replace it by the nearest correlation matrix in the Frobenius norm, that is, by the solution of the problem
$$\min\{ {\|A − X\|}_F : \text{X is a correlation matrix} \},$$
where ${\|A\|}_F^2=\sum_{ij}a_{ij}^2$.

* [The Nearest Correlation Matrix](https://nickhigham.wordpress.com/2013/02/13/the-nearest-correlation-matrix/)
* [Anderson Acceleration of the Alternating Projections Method for Computing the Nearest Correlation Matrix](http://eprints.maths.manchester.ac.uk/2490/1/hist16.pdf)
* https://github.com/higham/anderson-accel-ncm
* [Preconditioned alternating projection algorithm for solving the penalized-likelihood SPECT reconstruction problem.](https://www.ncbi.nlm.nih.gov/pubmed/28610694)

#### DAAREM

- [Damped Anderson acceleration with restarts and monotonicity control for accelerating EM and EM-like algorithms Talk](http://nhenderstat.com/wp-content/uploads/2018/11/AA_presentation_IMS.pdf)
- [Damped Anderson acceleration with restarts and monotonicity control for accelerating EM and EM-like algorithms](https://arxiv.org/abs/1803.06673)
- [Accelerating the EM Algorithm for Mixture-density Estimation](https://icerm.brown.edu/materials/Slides/tw-15-5/Accelerating_the_EM_algorithm_for_mixture_density_estimation_]_Homer_Walker,_WPI_and_ICERM.pdf)
- http://nhenderstat.com/research/

#### Anderson Accelerated Douglas-Rachford Splitting


* [Anderson Accelerated Douglas-Rachford Splitting](http://stanford.edu/~boyd/papers/a2dr.html)
* https://ctk.math.ncsu.edu/TALKS/Anderson.pdf
* [Using Anderson Acceleration to Accelerate the Convergence of Neutron Transport Calculations with Anisotropic Scattering](http://www.ans.org/pubs/journals/nse/a_37652)

If the gradient equals to 0s, i.e., $\nabla f(x)=0$, it is possible to find some `saddle points`. It means that the fixed point iteration matters.
It seems that ADMM (or generally Langragian multiplier method) is not the fixed point iteration.
The fixed point iteration is in  the form $x^{k}=f(x^k)$ where $f$ is usually explicitly  expressed.

However, this form is easy to generalize any mapping or operators such as in functional analysis.
In this sense, `ADMM` is really fixed point iteration: $(x^{k+1}, y^{k+1}, \lambda^{k+1})=ADMM(x^k, y^k, \lambda^k)$.


* [Plug-and-Play ADMM for Image Restoration: Fixed Point Convergence and Applications](https://arxiv.org/abs/1605.01710)
* [A reconstruction algorithm for compressive quantum tomography using various measurement sets](https://www.nature.com/articles/srep38497)
* [Accelerating ADMM for Efficient Simulation and Optimization](http://orca.cf.ac.uk/125193/14/AA-ADMM.pdf)
* [A FIXED-POINT PROXIMITY APPROACH TO SOLVING THE SUPPORT VECTOR REGRESSION WITH THE GROUP LASSO REGULARIZATION](http://www.math.ualberta.ca/ijnam/Volume-15-2018/No-1-18/2018-01-09.pdf)
* http://staff.ustc.edu.cn/~juyong/publications.html
* [Optimal parameter selection for the alternating direction method of multipliers (ADMM): quadratic problems](https://arxiv.org/pdf/1306.2454.pdf)
* https://www-users.cs.umn.edu/~baner029/
* [Multiplicative noise removal in imaging: An exp-model and its fixed-point proximity algorithm](https://www.sciencedirect.com/science/article/abs/pii/S106352031500144X)
* [A Proximal-Point Analysis of the Preconditioned Alternating Direction Method of Multipliers](https://imsc.uni-graz.at/mobis/publications/SFB-Report-2015-006_2.pdf)
* [Fixed points of generalized approximate message passing with arbitrary matrices](https://nyuscholars.nyu.edu/en/publications/fixed-points-of-generalized-approximate-message-passing-with-arbi)
* https://www.math.ucla.edu/~wotaoyin/summer2013/slides/
* https://xu-yangyang.github.io/papers/AcceleratedLALM.pdf
* https://engineering.purdue.edu/ChanGroup/project_restoration.html
* https://github.com/KezhiLi/Quantum_FixedPoint
* https://epubs.siam.org/doi/abs/10.1137/15M103580X
* https://xu-yangyang.github.io/

#### Anderson Accelerated Halley's Method



### Approximate Minimal Polynomial Extrapolation

[From linear to nonlinear iterative methods](http://www.dcs.bbk.ac.uk/~gmagoulas/APNUM.PDF) is not always direct and correct.

The following optimziation problem
$$\min f(x)=\frac{1}{2}{\|Ax-b\|}_2^2$$
is equals to solve the linear system $Ax=b$.

Given $A\in\mathbb{R}^{n\times n}$ such that 1 is not an eigenvalue of $A$ and $v\in\mathbb{R}^n$, the `minimal polynomial`
of $A$ with respect to the vector $v$ is the lowest degree polynomial $p(x)$ such that $p(A)v = 0, p(1) = 1$.

- [Efficient implementation of minimal polynomial and reduced rank extrapolation methods ](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19900017300.pdf)
- https://core.ac.uk/download/pdf/82614502.pdf
- [Minimal polynomial and reduced rank extrapolation methods are related](http://www.cs.technion.ac.il/~asidi/Sidi_Journal_Papers/P128_AdvCompMath_MPERRE.pdf)
- http://www.cs.technion.ac.il/~asidi/
- https://www.di.ens.fr/~aspremon/PDF/FOCM17.pdf
- https://simons.berkeley.edu/sites/default/files/docs/8821/alexsimons17.pdf
- [Nonlinear Schwarz iterations with Reduced Rank Extrapolation](https://www.math.temple.edu/~szyld/reports/RRE.Schwarz.report.pdf)
- [A BLOCK RECYCLED GMRES METHOD WITH INVESTIGATIONS INTO ASPECTS OF SOLVER PERFORMANCE](https://www.math.temple.edu/~szyld/reports/block-gcrodr.rev.report.pdf)
- http://dd23.kaist.ac.kr/slides/Martin_Gander_2.pdf

As shown before, the acceleration schemes are based on the linear combination of last iterated sequence.
The question is why it is linear combination?
Why not other `extrapolation` of the last updated values?


- [Steepest Descent Preconditioning for Nonlinear GMRES Optimization by Hans De Sterck](https://arxiv.org/abs/1106.4426)
- [A Fast Anderson-Chebyshev Acceleration for Nonlinear Optimization](https://arxiv.org/abs/1809.02341)


### Nemirovski’s Acceleration

Let $f$ be a 1-smooth function. Denote $x^{+} = x - \nabla f(x)$.
The algorithm simply returns the optimal combination of the conjugate point and the gradient descent point, that is:

$$x_{t+1}=\arg\min_{x\in P_t} f(x)$$
where $P_t=\operatorname{span}\{x_t^{+},\sum_{s=1}^t \lambda_s\nabla f(x^s)\}$.

It seems like a trust region methods where one region is given.

* [Fields CQAM Focus Program on Data Science and Optimization](http://www.fields.utoronto.ca/activities/19-20/data)
* http://www.pokutta.com/
* https://github.com/pokutta/lacg
* [Conditional Gradients and Acceleration](http://www.pokutta.com/blog/research/2019/07/04/LaCG-abstract.html)
* [Nemirovski’s acceleration](https://blogs.princeton.edu/imabandit/2019/01/09/nemirovskis-acceleration/)
* https://sunjackson.github.io/page9/

### Damped Inertial Gradient System

`Damped Inertial Gradient System (DIGS)`

- http://www.dii.uchile.cl/~mbravo/adgo2016/Slides/Peypouquet.pdf
- http://www.dii.uchile.cl/~mbravo/adgo2016/

### Alternating Anderson-Richardson method

- [Alternating Anderson-Richardson method: An efficient alternative to preconditioned Krylov methods for large, sparse linear systems](https://arxiv.org/pdf/1606.08740.pdf)
- [Anderson acceleration of the Jacobi iterative method: An efficient alternative to Krylov methods for large, sparse linear systems](https://www.sciencedirect.com/science/article/pii/S0021999115007585)

### Regularized Nonlinear Acceleration

[We describe a convergence acceleration technique for generic optimization problems. Our scheme computes estimates of the optimum from a nonlinear average of the iterates produced by any optimization method. The weights in this average are computed via a simple linear system, whose solution can be updated online. This acceleration scheme runs in parallel to the base algorithm, providing improved estimates of the solution on the fly, while the original optimization method is running. Numerical experiments are detailed on classical classification problems.](http://www.optimization-online.org/DB_HTML/2016/09/5630.html)

$$
\begin{array}{l}{\text { Input: Sequence }\left\{x_{0}, x_{1}, \ldots, x_{k+1}\right\}, \text { parameter } \lambda>0} \\
{\text { 1: Form } U=\left[x_{1}-x_{0}, \ldots, x_{k+1}-x_{k}\right]} \\
{\text { 2. Solve the linear system }\left(U^{T} U+\lambda I\right) z=1} \\
 {\text { 3. Set } c=z /\left(z^{T} \mathbf{1}\right)} \\
{\text { Output: Return } \sum_{i=0}^{k} c_{i} x_{i}, \text { approximating the optimum } x^{\ast}} \end{array}
$$

* [REGULARIZED NONLINEAR ACCELERATION](https://www.di.ens.fr/~aspremon/PDF/Nacc.pdf)
* [Regularized Nonlinear Acceleration@lids.mit.edu](https://lids.mit.edu/news-and-events/events/regularized-nonlinear-acceleration)
* [Regularized Nonlinear Acceleration@simons.berkeley](https://simons.berkeley.edu/talks/alex-daspremont-11-28-17)
* [Regularized Nonlinear Acceleration by Damien Scieur](https://damienscieur.com/pdf/slides/slidesSPARS2017_regularized.pdf)
* [Regularized nonlinear acceleration, Mathematical Programming](https://link.springer.com/article/10.1007%2Fs10107-018-1319-8)
* http://spars2017.lx.it.pt/index_files/papers/SPARS2017_Paper_16.pdf
* https://github.com/windows7lover/RegularizedNonlinearAcceleration
* https://damienscieur.com/

###  Objective Acceleration

[O-ACCEL (objective acceleration), is novel in that it minimizes an approximation to the objective function on subspaces of $\mathbb{R}^n$. We prove that O-ACCEL reduces to the full orthogonalization method for linear systems when the objective is quadratic, which differentiates our proposed approach from existing acceleration methods. Comparisons with the limited-memory Broyden–Fletcher–Goldfarb–Shanno and nonlinear conjugate gradient methods indicate the competitiveness of O-ACCEL.](https://onlinelibrary.wiley.com/doi/pdf/10.1002/nla.2216)

* [Objective acceleration for unconstrained optimization, Optimization methods and software conference 2017, Havana](https://people.maths.ox.ac.uk/riseth/files/presentation_oms_acceleration_dec17.pdf)
* [Objective acceleration for unconstrained optimization: code](https://github.com/anriseth/objective_accel_code)
* [Objective acceleration for unconstrained optimization by Asbjørn Nilsen Riseth](https://onlinelibrary.wiley.com/doi/pdf/10.1002/nla.2216)
* http://julianlsolvers.github.io/Optim.jl/stable/#algo/ngmres/
* http://spars2017.lx.it.pt/

<img src="https://wol-prod-cdn.literatumonline.com/cms/attachment/84ab4942-37b2-4275-af70-a508d0873ad4/nla2216-gra-0001-m.jpg" width="80%"/>

However, it is best to think from  the necessary condition of optima  in non-convex optimization in my opinion.
Another question is to generalize the fixed point iteration to stochastic gradient methods.

- [PyUNLocBoX: Optimization by Proximal Splitting](https://pyunlocbox.readthedocs.io/en/stable/index.html)
- https://lts2.epfl.ch/

### Direct Nonlinear Acceleration

[Optimization acceleration techniques such as momentum play a key role in state-of-the-art machine learning algorithms. Recently, generic vector sequence extrapolation techniques, such as regularized nonlinear acceleration (RNA) of Scieur et al. (Scieur et al., 2016), were proposed and shown to accelerate fixed point iterations. In contrast to RNA which computes extrapolation coefficients by (approximately) setting the gradient of the objective function to zero at the extrapolated point, we propose a more direct approach, which we call direct nonlinear acceleration (DNA). In DNA, we aim to minimize (an approximation of) the function value at the extrapolated point instead. We adopt a regularized approach with regularizers designed to prevent the model from entering a region in which the functional approximation is less precise. While the computational cost of DNA is comparable to that of RNA, our direct approach significantly outperforms RNA on both synthetic and real-world datasets. While the focus of this paper is on convex problems, we obtain very encouraging results in accelerating the training of neural networks.](https://arxiv.org/abs/1905.11692)

- https://arxiv.org/abs/1905.11692
- https://www.aritradutta.com/



### Proportional–Integral–Derivative Optimizer

The principle of feedback is simple  an input, $x^n$, is given, processed through some function, $f$, and then the output, $y^n$, becomes the next input, $x^{n+1}$, repeatedly. When allowing the ouput to equal the next input, an identity exists so that $x^{n+1}=y^n$. Cobweb diagrams exploit the relationship, map the iterations, and reveal the behaviors of fixed points.

<img src="https://www.mi.sanu.ac.rs/vismath/stewart/image022.gif" />
<img src="https://www.mi.sanu.ac.rs/vismath/stewart/image017.gif" />

- https://www.mi.sanu.ac.rs/vismath/stewart/index.html

A PID controller continuously calculates an error $e(t)$, which is the difference between the desired optimal
output and a measured system output, and applies a correction $u(t)$ to the system based on the proportional $(P)$, integral $(I)$, and derivative $(D)$ terms of $e(t)$. Mathematically, there is:
$$u(t)= K_p e(t) + K_i\int_{0}^{t}e(x)\mathrm d x + K_d\frac{\mathrm d}{\mathrm dt}e(t) $$

where $u$ is the control signal and $e$ is the control error.
The control signal is thus a sum of three terms:
1. the P-term (which is proportional to the error);
2. the I-term (which is proportional to the integral of the error);
3. and the D-term (which is proportional to the derivative of the error).

The controller can also be parameterized as
$$u(t)= K_p \{e(t) + \frac{1}{T_i}\int_{0}^{t}e(x)\mathrm d x + T_d\frac{\mathrm d}{\mathrm dt}e(t)\},\tag{PID}$$

where $T_i$ is called integral time and $T_d$ derivative time.

The proportional part acts on the present value of the error, the integral represent and average of past errors and the derivative can be interpreted as a prediction of future errors based on linear extrapolation.

<img src="http://5b0988e595225.cdn.sohucs.com/images/20180720/2194ca12804944859e77b6f4fc5fd2ac.gif" />

* http://www.scholarpedia.org/article/Optimal_control
* [EE365: Stochastic Control Spring Quarter 2014](https://web.stanford.edu/class/ee365/)
* [PID Theory Explained](https://www.ni.com/en-ie/innovations/white-papers/06/pid-theory-explained.html)
* [Chapter 8: PID Control](https://www.cds.caltech.edu/~murray/courses/cds101/fa04/caltech/am04_ch8-3nov04.pdf)
* [CDS 101/110 -- Fall 2004 Analysis and Design of Feedback Systems](https://www.cds.caltech.edu/~murray/courses/cds101/fa04/)
* [PID Control Theory ](http://cdn.intechopen.com/pdfs/29826/InTech-Pid_control_theory.pdf)
* [Control system theory](http://students.iitk.ac.in/roboclub/lectures/PID.pdf)

Methods| Recursion | Integration|
----|:---:|:----:|
Gradient Descent|$x^{t+1} = x^t -\alpha_t g(x^t)$ |?|
Momentum Methods|$x^{t+1} = x^t -\alpha_t  g(x^t) + \rho_t(x^t - x^{t-1})$|?|
Nesterov's Gradient Methods|$x^{t+1} =y^t -\alpha_t g(y^t), y^t = x^t + \rho_t(x^t -x^{t -1})$|?|
Newton's Methods|$x^{t+1} = x^t - \alpha_i H_t^{-1}g(x^t)$ |?|
Mirror Gradient Methods |$\nabla h(x^{t+1})-\nabla h(x^t) = x^t - \alpha_t \nabla f(x^t) , x\in \mathbb{S}$|?|

By viewing the gradient $g(x^t)$ as error $e(t)$, and comparing it to PID
controller, one can see that gradient descent only uses the present gradient to update the weights.

We rewrite the fomula $x^{t+1} = x^t -\alpha_t  g(x^t) + \rho_t(x^t - x^{t-1})$ as
$$x^{t+1} = x^t -\alpha_t  g(x^t) + \rho_t\underbrace{(x^t - x^{t-1})}_{-\alpha_{t-1}g(x^{t-1})+\rho_{t-1}(x^{t-1}-x^{t-2})}\\
= x^t -\alpha_t  g(x^t) - \sum_{i=1}^{t-1}[\prod_{j=0}^{i-1}\rho_{t-j}]\alpha_{t-i}g(x^{t-i})+ \rho_1(x^1-x^0)
.$$

One can see that the update of parameters relies on both the present gradient  and the integral of past gradients.
The only difference is that there
is a decay $\prod_{j=0}^{i-1}\rho_{t-j}$ term  in the I term.


<img title="PID Optimizer" src="http://5b0988e595225.cdn.sohucs.com/images/20180720/904ace2258564f6b98e91ad71de6ff91.jpeg" width="60%" />

****
**PID optimizer**

The proposed PID optimizer updates parameter $x$ at iteration $(t +1)$ by:

* $V^{t+1}=\alpha V^t -r g(x^t)$
* $D^{t+1}=\alpha D^t +(1-\alpha)(g(x^t)-g(x^{t-1}))$
* $x^{t+1}=x^t+V^{t+1}+K_d D^{t+1}$

<img src="http://5b0988e595225.cdn.sohucs.com/images/20180720/5250d32155024e079438d7484d082a03.jpeg" width="60%"/>

In a compact form, it is defined as following
$$x^{t+1}=x^t+\alpha V^t -r g(x^t)+K_d [\alpha D^t +(1-\alpha)(g(x^t)-g(x^{t-1}))]\\
= \alpha (V^t + K_d D^t) +x^t -r g(x^t)+ K_d(1-\alpha)[g(x^t)-g(x^{t-1})]
$$

which looks like an ensmeble of `inertia and relaxation techniques`.
[We, for the first time, connect classical control theory with deep network optimization, and improve up to 50% the efficiency over SGD-Momentum!)](http://www4.comp.polyu.edu.hk/~cslzhang/papers.htm)

Now it is still a emprical method without any convergence proof.

As [Linear Coupling: An Ultimate Unification of Gradient and Mirror Descent](https://arxiv.org/pdf/1407.1537.pdf), it is supposed to be converegnt in convex cases with some tuned parameters.

And it is simple to generalize Newton's methods where the gradients are replaced by rescaled gradients.
In another word, the Newton type PID optimizer updates parameter $x$ at iteration $(t +1)$ by:

- $V^{t+1}=\alpha V^t -r H^{-1}(x^t)g(x^t)$
- $D^{t+1}=\alpha D^t +(1-\alpha)(H^{-1}(x^t)g(x^t)-H^{-1}(x^{t-1})g(x^{t-1}))$
- $x^{t+1}=x^t+V^{t+1}+K_d D^{t+1}$.

The problem is that we have no theoretical proof while it inspired us how to ensemlble different ptimization methods or scehemes to accelerate the convergence procedure.

* [CVPR 2018 | 加速模型收敛的新思路（控制理论+深度学习）](http://www.sohu.com/a/242354509_297288)
* [一种用于深度网络随机优化的PID控制器方法](https://blog.csdn.net/weixin_39506322/article/details/82498701)
* [PID Optimizer (Proportional–Integral–Derivative Optimizer)](https://github.com/tensorboy/PIDOptimizer)
* [A PID Controller Approach for Stochastic Optimization of Deep Networks](https://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR18_PID.pdf)
* [Supplemental Materials to “A PID Controller Approach for Stochastic Optimization of Deep Networks”](http://www4.comp.polyu.edu.hk/~cslzhang/papers.htm)
* [Adaptive Restarting for First Order Optimization Methods](https://statweb.stanford.edu/~candes/math301/Lectures/adap_restart_nesterov.pdf)


[To overcome the oscillation problem in the classical momentum-based optimizer, recent work associates it with the proportional-integral (PI) controller, and artificially adds D term producing a PID controller. It suppresses oscillation with the sacrifice of introducing extra hyper-parameter.](https://arxiv.org/abs/1812.11305)

- [SPI-Optimizer: an integral-Separated PI Controller for Stochastic Optimization](https://arxiv.org/abs/1812.11305)

[By further analyzing the underlying constrained optimization problem, we have found that the two camps of distributed optimization can actually be related through the framework of proportional-integral control.  It turns out that consensus methods with constant step-sizes are akin to proportional control and dual-decomposition is akin to integral control.  In much the same way that proportional and integral control can be combined to create a control method with damped response and zero steady state error, the two methods of distributed optimization can be combined to produce a damped response with zero error.](http://gritslab.gatech.edu/home/2013/09/proportional-integral-distributed-optimization/)

- [Proportional Integral Distributed Optimization for Dynamic Network Topologies](https://smartech.gatech.edu/handle/1853/52389)
- [Proportional-Integral Distributed Optimization](http://gritslab.gatech.edu/home/2013/09/proportional-integral-distributed-optimization/)
- [Continuous-time Proportional-Integral Distributed Optimization for Networked Systems](https://arxiv.org/abs/1309.6613)
- [Continuous-time Proportional-Integral Distributed Optimization for
Networked Systems](https://vision.kuee.kyoto-u.ac.jp/~hiroaki/publication/Droge_2015_JCD.pdf)
- [A Control Perspective for Centralized and Distributed Convex Optimization](http://folk.ntnu.no/skoge/prost/proceedings/cdc-ecc-2011/data/papers/2298.pdf)
- [Input-Feedforward-Passivity-Based Distributed Optimization Over Jointly Connected Balanced Digraphs](https://arxiv.org/pdf/1905.03468)
- [Feedback-Feedforward Control Approach to Distributed Optimization](https://ieeexplore.ieee.org/document/8815008)
- [反馈控制理论在优化、机器学习等领域有哪些应用？](https://www.zhihu.com/question/276693700/answer/734826945)

Sample Recurrence Relation | Idea of Successive Approximations
----|----
$x^{k+1}=M(x^k)$ | $x^k=\underbrace{M(M(\cdots M(x^0)))}_{\text{k times}}$

****
* [Accelerated Optimization in the PDE Framework: Formulations for the Manifold of Diffeomorphism](https://repository.kaust.edu.sa/bitstream/handle/10754/627489/1804.02307v1.pdf?sequence=1&isAllowed=y)
* [Integration Methods and Accelerated Optimization Algorithms](https://arxiv.org/abs/1702.06751)
* https://statweb.stanford.edu/~candes/math301/hand.html


### Ensemble Methods of Optimization Algorithms

In machine learning, boosting algorithms can boost the weak base learner to higher accuarcy learner in probability correct algorithm framework.

The linear coupling technique shows that it is possible to combine some slow optimization methods to a faster method.
The interaction of optimization methods and boosting algorithms may bring benifits to both fields.

A unified framework is expected to ensemble different optimization methods to get a faster one.
As shown in above section, there are many acceleration schemes to combine the points generated by the same algorithm.
`Ensemble Methods of Optimization Algorithms` are supposed to combine the points generated by different optimziation methods.
Another theoretical consideration is to find some fractional order optimization methods.
There are fractional differential equations, fractional drivatives but no fractional order optimization methods.
I mean that fractional order optimization methods are some special combination of integer order optimization methods
not only using fractional derivatives.

Note that there is one step of Newton's coorection in [Halley's method](https://ms.yccd.edu/Data/Sites/1/userfiles/facstaff/jthoo/cvandpubs/papers/halley.pdf).
It is the composite of optimization methods beyond our scope of combination of optimization methods.
For example, can we combine Halley's method and Newton's method to obtain a faster optimizatio method like linear coupling?
Is there non-linear coupling of different methods?

- [Linear Coupling: An Ultimate Unification of Gradient and Mirror Descent](https://arxiv.org/abs/1407.1537)
- [A conformable calculus of radial basis functions and its applications](http://ijocta.balikesir.edu.tr/index.php/files/article/view/544)
- [Numerical Study for the Fractional Differential Equations Generated by Optimization Problem Using Chebyshev Collocation Method and FDM](http://www.naturalspublishing.com/files/published/17uj53u7fp9c85.pdf)
- [On Geometry of Halley's Method](https://ms.yccd.edu/Data/Sites/1/userfiles/facstaff/jthoo/cvandpubs/papers/halley.pdf)
- [Coupling of Immune Algorithms and Game Theory in Multiobjective Optimization](https://link.springer.com/chapter/10.1007/978-3-642-13232-2_61)
- [Optimal Approximations of Coupling in Multidisciplinary Models](https://kiwi.oden.utexas.edu/papers/optimal-decoupling-MDO-UQ.pdf)
- [Coupling of Optimization Algorithms Based on Swarm Intelligence: An Application for Control of Heroin Addiction Epidemic](https://www.igi-global.com/chapter/coupling-of-optimization-algorithms-based-on-swarm-intelligence/201806)
- [On the effective coupling of optimization algorithms to solve inverse problems of electromagnetism](https://www.emerald.com/insight/content/doi/10.1108/03321649810203080/full/html)

Before answering the above questions, it is best to clarify how the convergence speed and order of derivative information.

There is a strong connection between convergence speed and order of derivative information in [Higher-Order Gradient Method](https://www.pnas.org/content/pnas/113/47/E7351.full.pdf).

- [A variational perspective on accelerated methods in optimization](https://www.pnas.org/content/pnas/113/47/E7351.full.pdf)
- [Parallel Boosting with Momentum](https://link.springer.com/content/pdf/10.1007/978-3-642-40994-3_2.pdf)
- http://sime.sufe.edu.cn/teacher/show/56

[Alternated Inertia](https://arxiv.org/abs/1801.05589)  ensembles two schemes in fixed point iteration: relaxation and interia.


- [Alternated Inertia](https://arxiv.org/abs/1801.05589)
- [A GENERIC ONLINE ACCELERATION SCHEME FOR OPTIMIZATION ALGORITHMS VIA RELAXATION AND INERTIA](https://arxiv.org/pdf/1603.05398.pdf)
- [iPiano: Inertial Proximal Algorithm for Nonconvex Optimization](http://www.optimization-online.org/DB_HTML/2014/06/4414.html)

An intuitive way is to combine different optimziation methods of different order via Anderson acceleration.

## Dynamical Systems

We will focus on the optimization methods in the form of fixed point iteration and dynamical systems.
It is to minimize the following function
$$
f(x), x\in\mathbb{R}^p, \quad\nabla f(x) = g(x), \quad\nabla^2 f(x)= H(x).
$$

 Iteration | ODE  | Name
---|---|---
$x^{k+1}=x^{k}-\alpha_k g(x^k)$|$\dot{x}(t)=- g(x(t))$| Gradient descent
$x^{k+1}=x^{k}-\alpha_kH_k^{-1} g(x^k)$|$\dot{x}(t) =- H^{-1}(x)g(x(t))$| Newton's method

Like Newton interpolation, more points can compute higher order derivatives.
The dynamics of accelerated gradient methods are expected to correspond to higher order differential equations.

Some acceleration methods are iterations of the corresponding algorithms of `Asymptotic Vanishing Damping` called by [Hedy Attouch](https://arxiv.org/search/math?searchtype=author&query=Attouch%2C+H):

$$
\quad \quad \ddot{x}(t) + \frac{\alpha}{t} \dot{x}(t) +
\nabla \Phi (x(t)) =0.\tag{AVD}
$$

where $\Phi(x(t))$ is dependent on the objective function; $\alpha >0$ is constant in $\mathbb{R}$.

The fast minimization properties of the trajectories of the second-order evolution equation is also studied by Hedy's group in 2016:
$$
\ddot{x}(t) + \frac{\alpha}{t} \dot{x}(t) +
\nabla^2 \Phi (x(t))\dot{x}(t) + \nabla \Phi (x(t)) =0\tag{HDD}
$$

When it comes to numerical solution to differential equations, it is to find the solution of the equations $x(t)$ so that the equations hold; in optimization, the optima is our goal so that the focus is limit order
$$\lim_{t\to t_0} x(t)=x^{\star}$$
if possible where $x^{\star}$ optimizes the cost/objective function $f(x)$ specially $t_0=\infty$.

<img src="https://i1.rgstatic.net/ii/profile.image/291292945895424-1446461058178_Q128/Hedy_Attouch.jpg" width = "40%" />

- <https://perso.math.univ-toulouse.fr/spot/resumes/>
- [Fast convex optimization via inertial dynamics with Hessian driven damping](https://arxiv.org/abs/1601.07113)
- [A proximal-Newton method for monotone inclusions in Hilbert spaces with complexity $O(1/k^2)$](https://www.ljll.math.upmc.fr/~plc/sestri/attouch2014.pdf)
- <https://www.researchgate.net/profile/Hedy_Attouch>
- https://www.ljll.math.upmc.fr/~plc/sestri/
- [Hedy Attouch at https://biography.omicsonline.org](https://biography.omicsonline.org/france/montpellier-2-university/hedy-attouch-662580)
- [Rate of convergence of the Nesterov accelerated gradient method in the subcritical case $\alpha\leq 3$](http://www.birs.ca/events/2017/5-day-workshops/17w5030/videos/watch/201709180902-Attouch.html)
- [A Dynamical Approach to an Inertial Forward-Backward Algorithm for Convex Minimization
](https://epubs.siam.org/doi/pdf/10.1137/130910294)
- [An Inertial Proximal Method for Maximal Monotone Operators via Discretization of a Nonlinear Oscillator with Damping](https://link.springer.com/article/10.1023/A:1011253113155)
- [THE HEAVY BALL WITH FRICTION METHOD, I. THE CONTINUOUS DYNAMICAL SYSTEM: GLOBAL EXPLORATION OF THE LOCAL MINIMA OF A REAL-VALUED FUNCTION BY ASYMPTOTIC ANALYSIS OF A DISSIPATIVE DYNAMICAL SYSTEM](https://www.worldscientific.com/doi/abs/10.1142/S0219199700000025)
- [Viscosity Solutions of Minimization Problems](https://epubs.siam.org/doi/abs/10.1137/S1052623493259616?journalCode=sjope8)
- [Rate of convergence of the Nesterov accelerated gradient method in the subcritical case $\alpha \leq 3$](https://arxiv.org/abs/1706.05671)
- [A dynamic approach to a proximal-Newton method for monotone inclusions in Hilbert spaces, with complexity $O(\frac{1}{n^2})$](http://mtm.ufsc.br/~maicon/pdf/alv.att.sva-new.jca16.pdf)
- [FAST CONVERGENCE OF INERTIAL DYNAMICS AND ALGORITHMS WITH ASYMPTOTIC VANISHING DAMPING](http://www.optimization-online.org/DB_FILE/2015/10/5179.pdf)

<img src="http://awibisono.github.io/images/sqcomp.png" width="80%" />

***
It is difficult to generalize these methods to stochastic cases.

There is a wonderful summary [DYNAMICAL, SYMPLECTIC AND STOCHASTIC PERSPECTIVES ON GRADIENT-BASED OPTIMIZATION](https://people.eecs.berkeley.edu/~jordan/papers/jordan-icm.pdf) given by Micheal I Jordan at ICM 2018.

<img title = "jordan in ICM 2018" src = "http://www.icm2018.org/wp/wp-content/uploads/2018/08/43228449834_f63f8dc154_k-1280x640.jpg" width = 80% />


Some new connections between dynamical systems and optimization is found.

- [Variational and Dynamical Perspectives On Learning and Optimization](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-78.pdf)
- [Continuous and Discrete Dynamics For Online Learning and Convex Optimization](http://walid.krichene.net/papers/thesis-continuous-discrete.pdf)
- [DYNAMICAL, SYMPLECTIC AND STOCHASTIC PERSPECTIVES ON GRADIENT-BASED OPTIMIZATION](https://people.eecs.berkeley.edu/~jordan/papers/jordan-icm.pdf)
- [On Symplectic Optimization](https://arxiv.org/abs/1802.03653)
- [A variational perspective on accelerated methods in optimization](https://www.pnas.org/content/pnas/113/47/E7351.full.pdf)
- [A Dynamical Systems Perspective on Nesterov Acceleration](https://arxiv.org/abs/1905.07436)
- [Generalized Momentum-Based Methods: A Hamiltonian Perspective](https://arxiv.org/abs/1906.00436v1)
- https://people.eecs.berkeley.edu/~jordan/optimization.html
***

[Weijie J. Su](http://stat.wharton.upenn.edu/~suw/) (joint with Bin Shi, Simon Du, and Michael Jordan)  introduced a set of high-resolution differential equations to model, analyze, interpret, and design accelerated optimization methods.

> <img src="http://stat.wharton.upenn.edu/~suw/WeijieSu.jpg" width = "30%" />

$$
\ddot{x}(t) + 2\sqrt{\mu}\dot{x}(t) + \sqrt{s}\nabla^2 f(x) \dot{x}(t) + (1+\sqrt{\mu s})\nabla f(x) = 0\, \tag{Su}
$$

- [A Differential Equation for Modeling Nesterov’s Accelerated Gradient Method: Theory and Insights](http://stat.wharton.upenn.edu/~suw/paper/Nesterov_ODE.pdf)
- [Acceleration via Symplectic Discretization of High-Resolution Differential Equations](http://stat.wharton.upenn.edu/~suw/paper/symplectic_discretization.pdf)
- [Understanding the Acceleration Phenomenon via High-Resolution Differential Equations](http://stat.wharton.upenn.edu/~suw/paper/highODE.pdf)
- [Global Convergence of Langevin Dynamics Based Algorithms for Nonconvex Optimization](https://papers.nips.cc/paper/7575-global-convergence-of-langevin-dynamics-based-algorithms-for-nonconvex-optimization.pdf)
- https://www.researchgate.net/profile/Weijie_Su

***
- [ ] [Analysis of Hamilton-Jacobi Equation: Optimization, Dynamics and Control - Part II of II](https://www.pathlms.com/siam/courses/1825/sections/2464)
- [ ] [Optimization via the Hamilton-Jacobi-Bellman Method: Theory and Applications by Navin Khaneja, Harvard & KITP](http://online.kitp.ucsb.edu/online/qcontrol09/khaneja3/)
- [ ] [GRADIENT FLOW DYNAMICS](http://www-bcf.usc.edu/~mihailo/Keyword/GRADIENT-FLOW-DYNAMICS.html)
- [Sampling as optimization in the space of measures: The Langevin dynamics as a composite optimization problem](http://proceedings.mlr.press/v75/wibisono18a/wibisono18a.pdf)
- [The dynamics of Lagrange and Hamilton](https://nisheethvishnoi.wordpress.com/2018/09/19/the-dynamics-of-lagrange-and-hamilton/)
- [Optimization and Dynamical Systems](http://users.cecs.anu.edu.au/~john/papers/BOOK/B04.PDF)
- [Direct Runge-Kutta Discretization Achieves Acceleration](https://arxiv.org/abs/1805.00521)
- [The Physical systems Behind Optimization Algorithms](https://arxiv.org/abs/1612.02803)
- [Integration Methods and Optimization Algorithms](https://papers.nips.cc/paper/6711-integration-methods-and-optimization-algorithms.pdf)
***
**ADMM and Dynamics**




- [A Dynamical Systems Perspective on Nonsmooth Constrained Optimization](https://arxiv.org/abs/1808.04048)
- https://kgatsis.github.io/learning_for_control_workshop_CDC2018/assets/slides/Vidal_CDC18.pdf
- [ADMM and Accelerated ADMM as Continuous Dynamical Systems](http://proceedings.mlr.press/v80/franca18a/franca18a.pdf)
- [ADMM, Accelerated-ADMM, and Continuous Dynamical Systems, Talk @DIMACS](http://dimacs.rutgers.edu/events/details?eID=591)
- [Relax, and Accelerate: A Continuous Perspective on ADMM](https://pdfs.semanticscholar.org/0814/423300a6d7e69ed61f10060de5f3b84d7527.pdf)
- http://people.ee.duke.edu/~lcarin/Xuejun12.11.2015.pdf

The last but not least important question is how to rewrite the fixed point iteration as the discrete form of some differential equation. What is more, it is the interaction and connnetion between numerical solution to differential equations and  optimization methods in form of fixed ponit iteration that matters.

- [Glossary of Dynamical Systems Terms](https://lbm.niddk.nih.gov/glossary/glossary.html)
- [Bifurcations Involving Fixed Points and Limit Cycles in Biological Systems](http://www.medicine.mcgill.ca/physio/guevaralab/CNDSUM-bifurcations.pdf)
- https://elmer.unibas.ch/pendulum/nldyn.htm
***
* [ESAIM: Control, Optimization and Calculus of Variations (ESAIM: COCV)](https://www.esaim-cocv.org/)
* [MCT'03  Louisiana Conference on Mathematical Control Theory](https://www.math.lsu.edu/~malisoff/LCMCT/)
* [International Workshop “Variational Analysis and Applications”, ERICE, August 28 – September 5, 2018](http://varana.org/2018/)
* [Games, Dynamics and Optimization, 2019](http://gdo2019.com/programme/)
* [System dynamics & optimization](https://www.b-tu.de/en/fg-ingenieurmathematik-optimierung/forschung/projects/system-dynamics-optimization)
* [Introduction to Dynamical Systems by John K. Hunter, Department of Mathematics, University of California at Davis](https://www.math.ucdavis.edu/~hunter/m207/m207.pdf)
+ [Special Focus on Bridging Continuous and Discrete Optimization](http://dimacs.rutgers.edu/programs/sf/sf-optimization/)
+ [Part 2 Dynamical Systems](http://www.staff.city.ac.uk/g.bowtell/X2DynSyst07/)

[GRADIENTS AND FLOWS: CONTINUOUS OPTIMIZATION APPROACHES TO THE MAXIMUM FLOW PROBLEM](https://eta.impa.br/dl/028.pdf)



## Stochastic Approximation

[`Stochastic approximation` methods are a family of iterative methods typically used for *root-finding* problems or for *optimization* problems. The recursive update rules of stochastic approximation methods can be used, among other things, for solving linear systems when the collected data is corrupted by noise, or for approximating extreme values of functions which cannot be computed directly, but only estimated via noisy observations.](https://www.wikiwand.com/en/Stochastic_approximation)

- [Kiefer-Wolfowitz Algorithm](https://link.springer.com/chapter/10.1007/978-1-4471-4285-0_4)
- [Stochastic Process and Application](http://www.math.wayne.edu/~gyin/conf_web/index.html)


**Robbins–Monro algorithm** introduced in 1951 by Herbert Robbins and Sutton Monro, presented a methodology for solving a root finding problem, where the function is represented as an expected value.

Assume that we have a function ${\textstyle M(\theta )}:\mathbb{R}\mapsto\mathbb{R}$, and a constant ${\textstyle \alpha \in\mathbb R}$, such that the equation ${\textstyle M(\theta )=\alpha }$ has a unique root at ${\textstyle \theta ^{\ast}}$.

It is assumed that while we cannot directly observe the function ${\textstyle M(\theta )}$, we can instead obtain measurements of the random variable ${\textstyle N(\theta )}$ where ${\textstyle \mathbb{E} [N(\theta )]=M(\theta )}$. The structure of the algorithm is to then generate iterates of the form:
$${\displaystyle {\theta}_{n+1}= {\theta}_{n} - a_{n}(N({\theta}_{n})-\alpha )}\tag{R-M}$$
where $a_{1},a_{2},\dots$  is a sequence of positive step sizes.

This process can be considered as `fixed point iteration with random noise`.
Different from the determinant methods, the sequences they generated are also on random.



`Robbins and Monro` proved , Theorem 2 that $\theta_n$ converges in $L^{2}$ (and hence also in probability) to $\theta$ , and Blum later proved the convergence is actually with probability one, provided that:

${\textstyle N(\theta )}$ is uniformly bounded,
${\textstyle M(\theta )}$ is nondecreasing,
${\textstyle M'(\theta ^{\ast})}$ exists and is positive, and
The sequence ${\textstyle a_{n}}$ satisfies the following requirements:
$$\sum_{n=0}^{\infty} a_{n}=\infty \quad \fbox{ and }\quad \sum_{n=0}^{\infty} a_{n}^{2} < \infty \quad$$
A particular sequence of steps which satisfy these conditions, and was suggested by Robbins–Monro, have the form: ${\textstyle a_{n}=a/n}$, for ${\textstyle a>0}$. Other series are possible but in order to average out the noise in ${\textstyle N(\theta )}$, the above condition must be met.

The basic ideas of the Robbins–Monro scheme
can be readily modified to provide successive approximations for the minimum
(or maximum) of a unimodal regression function, as was shown by `Kiefer and
Wolfowitz (1952)` who introduced a recursive scheme of the form

$$\theta_{n+1}= {\theta}_{n} - a_{n}\Delta(x_{n})\tag{K-W}$$

to find the minimum $\theta$ of $M$ (or, equivalently, the solution of $dM/dx = 0$).

During the nth stage of the Kiefer–Wolfowitz scheme, observations $y_n^{(1)}$ and $y_n^{(2)}$ are taken at the design levels $x_n^{(1)} = x_n + c_n$ and $x_n^{(2)} = x_n − c_n$, respectively. In (KW), $\Delta(x_n)=\frac{y_n^{(1)}-y_n^{(2)}}{2c_n}$ an and $c_n$ are positive constants such that $c_n \to 0$, $\sum (c_n/a_n)^2<\infty$ and $\sum a_n=\infty$.


- [5 The Stochastic Approximation Algorithm](http://webee.technion.ac.il/people/shimkin/LCS11/ch5_SA.pdf)
- [Chapter 15: Introduction to Stochastic Approximation Algorithms](http://www.professeurs.polymtl.ca/jerome.le-ny/teaching/DP_fall09/notes/lec11_SA.pdf)
- [Dynamics of Stochastic Approximation](http://members.unine.ch/michel.benaim/perso/SPS99.pdf)
- [Stochastic Approximation by Tze Leung Lai](https://statistics.stanford.edu/sites/g/files/sbiybj6031/f/2002-31.pdf)
- [A Multivariate Stochastic Approximation Procedure](https://statistics.stanford.edu/research/multivariate-stochastic-approximation-procedure)
- [The Robbins–Monro Stochastic Approximation Approach to a Discrimination Problem](https://statistics.stanford.edu/sites/g/files/sbiybj6031/f/JOH%20PHS%2008.pdf)
- [Stochastic approximation: invited paper](https://projecteuclid.org/download/pdf_1/euclid.aos/1051027873)
- [Stochastic Approximation and Recursive Algorithms and Applications](https://link.springer.com/book/10.1007/b97441)
- [Stochastic Approximations, Diffusion Limit and Small Random Perturbations of Dynamical Systems](http://web.mst.edu/~huwen/slides_stochastic_approximation_perturbation.pdf)
- [Regularized iterative stochastic approximation methods for stochastic variational inequality problems](https://ieeexplore.ieee.org/document/6286992/)

### Stochastic Gradient Descent

`Stochastic gradient descent` is classified to stochastic optimization which is considered as the generalization of `gradient descent`.

Stochastic gradient descent takes advantages of stochastic or estimated gradient to replace the true gradient in gradient descent.
It is **stochastic gradient** but may not be **descent**.
The name **stochastic gradient methods**  may be more appropriate to call the methods with stochastic gradient.
It can date back to **stochastic approximation** in statistics.

It is aimed to solve the problem with finite sum optimization problem, i.e.
$$
\arg\min_{\theta}\frac{1}{n}\sum_{i=1}^{n}f(\theta|x_i)
$$
where $n<\infty$ and $\{f(\theta|x_i)\}_{i=1}^{n}$ are in the same function family and $\{x_i\}_{i=1}^{n}\subset \mathbb{R}^{d}$ are constants  while $\theta\in\mathbb{R}^{p}$ is the variable vector.

The difficulty is $p$, that the dimension of $\theta$, is tremendous. In other words, the model is **overparameterized**. And the number $n$ is far larger than $p$ generally, i.e. $n \gg  p\gg d$.
What is worse, the functions  $\{f(\theta|x_i)\}_{i=1}^{n}$ are not convex in most case.

***

The stochastic gradient method is defined as
$$
\theta^{k+1}=\theta^{k}-\alpha_{k}\frac{1}{m}\sum_{j=1}^{m}\nabla f(\theta^{k}| x_{j}^{\prime})
$$
where $x_{j}^{\prime}$ is draw from $\{x_i\}_{i=1}^{n}$ and $m\ll n$ is on random.

It is the fact $m\ll n$ that makes it possible to compute the gradient of finite sum objective function and its side effect is that the objective function is not always descent. Thus it is also called as `mini-batch` gradient descent.

There is fluctuations in the total objective function as gradient steps with respect to mini-batches are taken.

******************************************************************

|The fluctuations in the objective function as gradient steps with respect to mini-batches are taken|
|:------------------------------------:|
|<img src="https://upload.wikimedia.org/wikipedia/commons/f/f3/Stogra.png" width="60%" />|

***

An heuristic proposal for avoiding the choice and for modifying the learning rate while the learning task runs is the **bold driver (BD) method**[^14].
The learning rate increases *exponentially* if successive steps reduce the objective function $f$, and decreases rapidly if an “accident” is encountered (if objective function $f$ increases), until a suitable value is found.
After starting with a small learning rate, its modifications are described by the following equation:
$$
  \alpha_{k+1}=
   \begin{cases}
       \rho \alpha_{k}, & {f(\theta^{k+1})< f(\theta^{k})}; \\
       \eta^n \alpha_{k}, & {f(\theta^{k+1})> f(\theta^{k})} \text{using ${\alpha}_k$},
   \end{cases}
$$

where $\rho$ is close to 1 such as $\rho=1.1$  in order to avoid frequent “accidents” because the
objective function computation is wasted in these cases, $\eta$ is chosen to provide a rapid reduction
($\eta = 0.5$), and $n$ is the minimum integer such that the reduced rate $\eta^n$ succeeds in diminishing the objective function.[^13]

Except the `learning rate or step length`, there is yet another hyperparameter - the batch size $m$  in each iteration.

******

The fact that the sample size is far larger than the dimension of parameter, $n\gg p$,  that makes it expensive to compute total objective function $f(\theta)=\sum_{i=1}^{n}f(\theta|{x}_i)$.
Thus it is not clever to determine the learning rate $\alpha_k$ by line search.
And most stochastic gradient methods are to find  proper step length $\alpha_{k}$ to make it converge at least in convex optimization.
The variants of gradient descent such as momentum methods or mirror gradient methods have their stochastic counterparts.

* It is simplest to set the step length a constant, such as ${\alpha}_k=3\times 10^{-3}\, \forall k$.
* There are decay schemes, i.e. the step length ${\alpha}_k$ diminishes such as ${\alpha}_k=\frac{\alpha}{k}$, where $\alpha$ is constant.
* And another strategy is to tune the step length adaptively such as *AdaGrad, ADAM*.

$\color{lime}{PS}$: the step length  $\alpha_k$ is called **learning rate** in machine learning. Additionally, stochastic gradient descent is also named as [increment gradient methods](http://www.mit.edu/~dimitrib/Incremental_Survey_LIDS.pdf) in some case.

|The Differences of Gradient Descent and Stochastic Gradient Descent|
|:-----------------------------------------------------------------:|
|<img src="https://wikidocs.net/images/page/3413/sgd.png" width = "60%" />|

We can see some examples to see the advantages of incremental method such as the estimation of mean.
Given $x_1, x_2, \dots, x_n$ the mean is estimated as $\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}$. If now we observed more data $y_1, y_2, \dots, y_m$ from the population, the mean could be estimated by $\frac{n\bar{x}}{m+n} + \frac{\sum_{j=1}^{m} y_j }{m+n} =\frac{n\bar{x}+\sum_{j=1}^{m} y_j}{m+n}$. It is not necessary to summarize ${x_1, \dots, x_n}$.

Another example, it is `Newton interpolation formula` in numerical analysis. The task is to fit the function via polynomials given some point in th function $(x_i , f(x_i)), i = 1,2, \dots, n$.
[The Newton form of the interpolating polynomial  is given by](https://nptel.ac.in/courses/122104019/numerical-analysis/Rathish-kumar/rathish-oct31/fratnode5.html)

 $$
   P_n(x) = a_0 + a_1 (x-x_1) + a_2 (x-x_1)(x-x_2) + \cdots \\
         + a_n(x-x_1)(x-x_2)(x-x_3)\cdots (x-x_n).
 $$

This form is incremental and if another points $(x_{n+1}, f(x_{n+1}))$ is observed we will fit the function $f$ more precisely just by adding another term $a_{n+1}(x-x_1)(x-x_2)\cdots (x-x_n)(x-x_{n+1})$
where the coefficients $a_0, a_1,\cdots, a_n$ are determined by $f(x_1), f(x_2), \cdots, f(x_n)$.


See the following links for more information on *stochastic gradient descent*.

* https://www.wikiwand.com/en/Stochastic_gradient_descent
* http://ruder.io/optimizing-gradient-descent/
* https://zhuanlan.zhihu.com/p/22252270
* http://fa.bianp.net/teaching/2018/eecs227at/stochastic_gradient.html
* [A Brief (and Comprehensive) Guide to Stochastic Gradient Descent Algorithms](https://www.bonaccorso.eu/2017/10/03/a-brief-and-comprehensive-guide-to-stochastic-gradient-descent-algorithms/)
* [A look at SGD from a physicist's perspective - Part 1](https://henripal.github.io/blog/stochasticdynamics)
* [A look at SGD from a physicists's perspective - Part 2, Bayesian Deep Learning](https://henripal.github.io/blog/nealbayesian)
* [A look at SGD from a physicists's perspective - Part 3, Langevin Dynamics and Applications](https://henripal.github.io/blog/langevin)

#### Convergence Analysis of Stochastic Gradient Methods

[Progress in machine learning (ML) is happening so rapidly, that it can sometimes feel like any idea or algorithm more than 2 years old is already outdated or superseded by something better. However, old ideas sometimes remain relevant even when a large fraction of the scientific community has turned away from them. This is often a question of context: an idea which may seem to be a dead end in a particular context may become wildly successful in a different one. In the specific case of deep learning (DL), the growth of both the availability of data and computing power renewed interest in the area and significantly influenced research directions.](https://ai.googleblog.com/2018/12/the-neurips-2018-test-of-time-award.html)



![Leon Bottou](https://istcolloq.gsfc.nasa.gov/sites/isat/files/bottou.jpg)

* http://blavatnikawards.org/honorees/profile/leon-bottou/
* https://leon.bottou.org/projects/sgd
* https://leon.bottou.org/research/stochastic
* https://leon.bottou.org/papers/bottou-bousquet-2008
* [Large-Scale Machine Learning with Stochastic Gradient Descent](https://datajobs.com/data-science-repo/Stochastic-Gradient-Descent-[Leon-Bottou].pdf)
* [Trade off of Machine Learning](https://ai.googleblog.com/2018/12/the-neurips-2018-test-of-time-award.html)
* [The Tradeoffs of Large-scale Learning](https://leon.bottou.org/talks/largescale)
* https://sites.google.com/view/panos-toulis/implicit-sgd
* http://dustintran.com/blog/on-asymptotic-convergence-of-averaged-sgd
* https://github.com/ptoulis/implicit-sgd

The stochastic gradient methods generates  random/stochastic sequences, which are so different from the classic methods. The convergence of stochastic methods will bring another problem whether the random sequence is convergent to a optimal point  in some sense.
> Let $\{X_1,\cdots, X_n,\cdots\}$ be a random sequence and $X$ be a random variable.
>
> Convergence in probability/measure:
$$\lim_{n\to\infty}P(|X_n-X|\leq \epsilon
)=1\forall \epsilon>0$$
>
> Convergence in mean square:
$$\lim_{n\to\infty}\mathbb{E}({\|X-X_n\|}_2^2)=0$$
>
> Convergence with probability 1(almost surely):
$$P(\lim_{n\to\infty}X_n=X)=1$$
> Convergence in distribution:
$$\lim_{n\to\infty}F_{X_n}(x)=F_X(x)$$
> for every $x$ at which $F_X(x)$ is continuous.

____
- [Lecture Notes: Weak convergence of stochastic processes, Thomas Mikosch1
(2005)](http://web.math.ku.dk/~erhansen/web/stat1/mikosch1.pdf)
- [EE178: Probabilistic Systems Analysis](http://isl.stanford.edu/~abbas/aeglect178.php)
- [Convergence of Stochastic Processes](http://repository.upenn.edu/cgi/viewcontent.cgi?article=1487&context=cis_reports)
- https://www.eng.tau.ac.il/~liptser/lectures/lect_new2.pdf
- http://isl.stanford.edu/~abbas/ee278/lect05.pdf
- https://www2.math.ethz.ch/education/bachelor/lectures/fs2014/math/bmsc/weak-conv.pdf
- [Convergence of Probability Measure](http://cermics.enpc.fr/~monneau/Billingsley-2eme-edition.pdf)



+ [Convergence Analysis of Gradient Descent Stochastic Algorithms](https://www2.isye.gatech.edu/~ashapiro/JOTA96[1].pdf)
+ [Stochastic Gradient Descent with Exponential Convergence Rates of Expected Classification Errors](http://proceedings.mlr.press/v89/nitanda19a/nitanda19a.pdf)
+ [Stochastic Approximations, Diffusion Limit and Small Random Perturbations of Dynamical Systems](http://web.mst.edu/~huwen/slides_stochastic_approximation_perturbation.pdf)
+ [Incremental Gradient, Subgradient, and Proximal Methods for Convex Optimization: A Survey ](http://www.mit.edu/~dimitrib/Incremental_Survey_LIDS.pdf)
+ [Convex Relaxations of Convolutional Neural Nets](https://arxiv.org/abs/1901.00035)
+ [The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent](https://arxiv.org/pdf/1904.06963v2.pdf)
+ [Quasi-potential as an implicit regularizer for the loss function in the stochastic gradient descent](https://arxiv.org/abs/1901.06054)
+ [The Multiplicative Noise in Stochastic Gradient Descent: Data-Dependent Regularization, Continuous and Discrete Approximation](https://arxiv.org/abs/1906.07405)
+ [On the Convergence of Perturbed Distributed Asynchronous Stochastic Gradient Descent to Second Order Stationary Points in Non-convex Optimization](https://arxiv.org/abs/1910.06000v1)
+ [On the Convergence of Stochastic Gradient Descent with Adaptive Stepsizes](https://arxiv.org/abs/1805.08114v3)
+ [A Convergence Theory for Deep Learningvia Over-Parameterization](https://arxiv.org/pdf/1811.03962.pdf)
+ [Analysis of the Gradient Descent Algorithm for a Deep Neural Network Model with Skip-connections](https://arxiv.org/abs/1904.05263)
+ [Gradient Descent Finds Global Minima of Deep Neural Networks](https://arxiv.org/abs/1811.03804)
+ https://zhuanlan.zhihu.com/p/28819506
+ https://zhuanlan.zhihu.com/p/73441350
+ [The Loss Surfaces of Multilayer Networks](https://arxiv.org/abs/1412.0233)
+ [为什么说随机最速下降法(SGD)是一个很好的方法？](https://zhuanlan.zhihu.com/p/27609238)
+ [Gradient Descent with Random Initialization: Fast Global Convergence for Nonconvex Phase Retrieval](https://arxiv.org/pdf/1803.07726.pdf)

<img src="http://web.cs.ucla.edu/~qgu/image/qgu.jpg" width="40%"/>

+ https://arxiv.org/abs/1806.06763v1
+ http://web.cs.ucla.edu/~qgu/research.html
+ https://users.ece.cmu.edu/~yuejiec/publications.html

### ADAM and More

- [Robust Learning Rate Selection for Stochastic Optimization via Splitting Diagnostic](https://www.arxiv-vanity.com/papers/1910.08597/)
- [A Survey of Optimization Methods from a Machine Learning Perspective](https://arxiv.org/abs/1906.06821v2)
- [Understanding the Role of Momentum in Stochastic Gradient Methods](https://arxiv.org/abs/1910.13962v1)
- [Accelerated Linear Convergence of Stochastic Momentum Methods in Wasserstein Distances](https://arxiv.org/abs/1901.07445v2)

#### YellowFin

[We revisit SGD with Polyak's momentum, study some of its robustness properties and extract the design principles for a tuner, YellowFin. YellowFin automatically tunes a single learning rate and momentum value for SGD.](https://cs.stanford.edu/~zjian/project/YellowFin/)

- [YellowFin and the Art of Momentum Tuning](https://cs.stanford.edu/~zjian/project/YellowFin/)
- https://arxiv.org/abs/1706.03471
- [YellowFin: Adaptive optimization for (A)synchronous systems](https://systemsandml.org/Conferences/2019/doc/2018/188.pdf)
- https://systemsandml.org/
- [YellowFin: An automatic tuner for momentum SGD](https://dawn.cs.stanford.edu//2017/07/05/yellowfin/)
- http://mitliagkas.github.io/
- https://github.com/JianGoForIt/YellowFin
- https://cs.stanford.edu/~zjian/


#### Adam-type algorithms

`ADAM` composes of adaptive step strategies and momentum methods in some sense. It is widely used in deep learning training.

Formally, an Adam-type algorithm is of the following form:

<img src="https://www.ibm.com/blogs/research/wp-content/uploads/2019/04/Algorithm-768x365.png" width="80%"/>

where $x$ is the optimization variable, gt is a stochastic gradient at step $t$, $\beta_1,t$ is a non-increasing sequence, and $h_t$ is an arbitrary function that outputs a vector having the same dimension as $x$.

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [On the convergence of Adam and Beyond](https://www.satyenkale.com/pubs/on-the-convergence-of-adam-and-beyond/)
- [Will Adam Algorithms Work for Me?](https://www.ibm.com/blogs/research/2019/05/adam-algorithms/)
- [On the Convergence of A Class of Adam-Type Algorithms for Non-Convex Optimization](https://openreview.net/forum?id=H1x-x309tm)
- [Gentle Introduction to the Adam Optimization Algorithm for Deep Learning](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
- [A Survey on Proposed Methods to Address Adam Optimizer Deficiencies](http://www.cs.toronto.edu/~sajadn/sajad_norouzi/ECE1505.pdf)
- https://2018.ieeeglobalsip.org/sym/18/AML
- https://github.com/LiyuanLucasLiu/RAdam
- https://github.com/CyberZHG/keras-radam
- http://ruder.io/deep-learning-optimization-2017/
- http://people.ece.umn.edu/~mhong/Activities.html

|The Differences of Stochastic Gradient Descent and its Variants|
|:-------------------------------------------------------------:|
|<img src="http://beamandrew.github.io//images/deep_learning_101/sgd.gif" width = "60%" />|


#### Natasha, Katyusha and Beyond

$\color{green}{PS}$: [Zeyuan Allen-Zhu](http://www.arxiv-sanity.com/search?q=Zeyuan+Allen-Zhu) and others published much work on acceleration of stochastic gradient descent.


- http://www.arxiv-sanity.com/search?q=Zeyuan+Allen-Zhu
- [Natasha: Faster Non-Convex Stochastic Optimization Via Strongly Non-Convex Parameter](https://arxiv.org/abs/1702.00763v5)
- [Natasha 2: Faster Non-Convex Optimization Than SGD](https://arxiv.org/abs/1708.08694v4)
- [Katyusha: The First Direct Acceleration of Stochastic Gradient Methods](https://arxiv.org/abs/1603.05953v6)
- [Katyusha X: Practical Momentum Method for Stochastic Sum-of-Nonconvex Optimization](https://arxiv.org/abs/1802.03866v1)
- [Katyusha Acceleration for Convex Finite-Sum Compositional Optimization](https://arxiv.org/abs/1910.11217v1)
- [Dissipativity Theory for Accelerating Stochastic Variance Reduction: A Unified Analysis of SVRG and Katyusha Using Semidefinite Programs](https://arxiv.org/abs/1806.03677v1)
- [Don't Jump Through Hoops and Remove Those Loops: SVRG and Katyusha are Better Without the Outer Loop](https://arxiv.org/abs/1901.08689v2)
- [A Universal Catalyst for First-Order Optimization](https://arxiv.org/abs/1506.02186)

#### Variance Reduction Stochastic Gradient Methods

For general convex optimization, stochastic gradient descent methods can obtain an $O(1/\sqrt{T})$ convergence rate in expectation.

A more general version of SGD is the following
$$\omega^{(t)}=\omega^{(t−1)}- g_t(\omega(t−1), \epsilon_t)$$

where $\epsilon_t$ is a random variable that may depend on $\omega^{(t−1)}$. And it is usually assumed that $\mathbb E(g_t(\omega^{(t−1)}, \epsilon_t)\mid \omega^{(t−1)} = \nabla f(\omega^{(t−1)})$.

Randomness introduces large variance if $g_t(\omega^{(t−1)}, \epsilon_t)$ is very large, it will slow down the convergence.

___

Procedure SVRG
+ input: update frequency $m$ and learning rate $\eta$
+ initialization: $\tilde{\omega}_0$
+ **for $s=1,2,\cdots$ do**
  + $\tilde w=\tilde w_{s-1}$
  + $\tilde{ \mu}=\nabla f(\tilde w)=\frac{1}{n}\sum_{i=1}^{n}\nabla f_i(\tilde{w})$
  + $\omega_0=\tilde{\omega}$
  + Randomly pick $i_t \in \{1, ..., n\}$ and update weight, repeat $m$ times $\omega^{(t)}=\omega^{(t−1)}- \eta_t [g_t(\omega(t−1), \epsilon_t)-g_t(\tilde\omega, \epsilon_t) - \tilde{ \mu}]$
  + option I: set $\tilde{\omega}_s={\omega}_m$
  + option II: set $\tilde{\omega}_s={\omega}_t$ for randomly chosen $t \in \{1, ..., n-1\}
+ **end for**


***
+ [Stochastic Gradient Descent with Variance Reduction](http://ranger.uta.edu/~heng/CSE6389_15_slides/SGD2.pdf)
+ [Variance reduction for stochastic gradient methods](http://www.princeton.edu/~yc5/ele522_optimization/lectures/variance_reduction.pdf)
+ https://www.di.ens.fr/~fbach/fbach_tutorial_siopt_2017.pdf
+ [Variance-Reduced Stochastic Learning by Networked Agents under Random Reshuffling](https://arxiv.org/pdf/1708.01384.pdf)
+ [Variance Reduction in Stochastic Gradient Langevin Dynamics](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5508544/)
+ https://zhizeli.github.io/
+ http://bigml.cs.tsinghua.edu.cn/~jianfei/
+ https://richtarik.org/i_papers.html
+ [VR-SGD](https://arxiv.org/pdf/1802.09932.pdf)
+ http://ranger.uta.edu/~heng/CSE6389_15_slides/SGD2.pdf
+ [Variance reduction techniques for stochastic optimization](http://cbl.eng.cam.ac.uk/pub/Intranet/MLG/ReadingGroup/VarianceReductionTechniquesForStochasticOptimization.pdf)
+ [Stochastic Variance-Reduced Optimization for Machine Learning Part I](https://www.di.ens.fr/~fbach/fbach_tutorial_siopt_2017.pdf)
+ [Fast Variance Reduction Method with Stochastic Batch Size](https://arxiv.org/abs/1808.02169)
+ [Laplacian Smoothing Gradient Descent](https://www.simai.eu/wp-content/uploads/2018/07/Slides_WNLL_LSGD.pdf)
+ [Entropy SGD](http://59.80.44.48/www.columbia.edu/~aec2163/NonFlash/Papers/Entropy-SGD.pdf)
- [Acceleration of SVRG and Katyusha X by Inexact Preconditioning](http://www.optimization-online.org/DB_HTML/2019/05/7225.html)
- [Stabilized SVRG: Simple Variance Reduction for Nonconvex Optimization](https://arxiv.org/abs/1905.00529)
- [A unified variance-reduced accelerated gradient method for convex optimization](https://arxiv.org/abs/1905.12412)
+ https://github.com/tdozat/Optimization
+ https://zhuanlan.zhihu.com/p/25473305
+ http://ranger.uta.edu/~heng/CSE6389_15_slides/
+ https://caoxiaoqing.github.io/2018/05/11/SVRG%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/
+ https://json0071.gitbooks.io/svm/content/sag.html
+ http://www.cs.toronto.edu/~jmartens/research.html
* https://homepage.cs.uiowa.edu/~tyng/
* http://tongzhang-ml.org/publication.html
* [Fast Stochastic Variance Reduced Gradient Method with Momentum Acceleration for Machine Learning](https://arxiv.org/abs/1703.07948v2)


#### Escape Saddle Points


Saddle points is considered as the fundamnetal problem in high dimensional space when training a deep learning model.

[Noisy gradient descent can find a local minimum of strict saddle functions in polynomial time.](http://www.offconvex.org/2016/03/22/saddlepoints/)

It is  the inherent nature of stochastic gardient methods to escape saddle points.

- http://opt-ml.org/papers/OPT2017_paper_44.pdf
- [Escaping From Saddle Points --- Online Stochastic Gradient for Tensor Decomposition](https://arxiv.org/abs/1503.02101)
- [How to Escape Saddle Points Efficiently](http://www.offconvex.org/2017/07/19/saddle-efficiency/)
- [Escaping from Saddle Points](http://www.offconvex.org/2016/03/22/saddlepoints/)
- [First-order Methods Almost Always Avoid Saddle Points](https://arxiv.org/abs/1710.07406)
- [Revisiting Normalized Gradient Descent: Fast Evasion of Saddle Points](https://export.arxiv.org/pdf/1711.05224)
- [Escaping Saddle Points: from Agent-based Models to Stochastic Gradient Descent](http://www-personal.umich.edu/~fayu/papers/Voter-Model2.pdf)
- [Heavy-ball Algorithms Always Escape Saddle Points](https://www.ijcai.org/proceedings/2019/488)
- https://users.cs.duke.edu/~rongge/
- http://sites.utexas.edu/mokhtari/
- [On Nonconvex Optimization for Machine Learning: Gradients, Stochasticity, and Saddle Points](https://arxiv.org/pdf/1902.04811.pdf)
- [Escaping Undesired Stationary Points in Local Saddle Point Optimization A Curvature Exploitation Approach](http://leox1v.com/poster/local_saddle_opt.pdf)
- [SSRGD: Simple Stochastic Recursive Gradient Descent for Escaping Saddle Points](https://arxiv.org/abs/1904.09265)


#### Stochastic Proximal Point Methods


- [Stochastic (Approximate) Proximal Point Methods: Convergence, Optimality, and Adaptivity](https://arxiv.org/abs/1810.05633)
- [Nonasymptotic convergence of stochastic proximal point methods for constrained convex optimization](http://jmlr.csail.mit.edu/papers/volume18/17-347/17-347.pdf)
- [A Stochastic Proximal Point Algorithm for Saddle-Point Problems](https://deepai.org/publication/a-stochastic-proximal-point-algorithm-for-saddle-point-problems)
- [Stochastic Proximal Gradient Algorithm and it’s application to sum of least squares](http://www-personal.umich.edu/~aniketde/processed_md/Stats607_Aniketde.pdf)
- [Proximal Stochastic Gradient Method with
Variance Reduction](http://helper.ipam.ucla.edu/publications/sgm2014/sgm2014_11808.pdf)  
- https://emtiyaz.github.io/papers/uai2016.pdf
- https://mwang.princeton.edu/
- https://ajwagen.github.io/adsi_learning_and_control/
- https://homepage.cs.uiowa.edu/~yxu71/

#### Stochastic Coordinate Fixed-point Iteration

There are examples such as Forward-Backward, Douglas-Rachford,... for finding a zero of a sum of maximally monotone operators or for minimizing a sum of convex functions.

##### Random block-coordinate Krasnoselskiı–Mann iteration

+ for $n=0,\cdots$
    + for $i=1, \cdots, m$
        + $x_{i, n+1}=x_{i, n}+\epsilon_{i, n}\lambda_n(\mathrm T_i(x_{1,n},\cdots, x_{m, n})+a_{i, n}-x_{i, n})$

where
* $x_0, a_n\in \mathbb H$ and $\mathbb H$ is separable real Hilbert space,
* $\epsilon_{n}$ is random variable in $\{0,1\}^m\setminus \mathbf{0}$,
* $\lambda_n\in (0, 1)$ and $\liminf \lambda_n>0$ and $\limsup \lambda_n<1$,
* the mapping $\mathrm T:\mathbb H\to \mathbb H$ i.e. $x\mapsto (T_1x, \cdots, T_i x, \cdots, T_m x)$ is  nonexpansive operator.


##### Double-layer random block-coordinate algorithms
+ for $n=0, 1, \cdots$
    + $y_n =\mathrm R_n x_n + b_n$
    + for $i=1, \cdots, m$
        + $x_{i, n+1}=x_{i, n}+\epsilon_{i, n}\lambda_n(\mathrm T_{i, n}(y_n)+a_{i, n}-x_{i, n})$

##### Random block-coordinate Douglas-Rachford splitting

* https://www.ljll.math.upmc.fr/~plc/sestri/pesquet2014.pdf
* https://arxiv.org/abs/1404.7536
* https://arxiv.org/abs/1406.6404
* https://arxiv.org/abs/1406.5429
* https://arxiv.org/abs/1706.00088

##### Random block-coordinate forward-backward splitting

- [Stochastic Block-Coordinate Fixed Point Iterations with Applications to Splitting](https://www.ljll.math.upmc.fr/~plc/sestri/pesquet2014.pdf)
- [Stochastic Quasi-Fejer Block-Coordinate Fixed Point Iterations with Random Sweeping](https://core.ac.uk/download/pdf/47081501.pdf)
- [LINEAR CONVERGENCE OF STOCHASTIC BLOCK-COORDINATE FIXED POINT ALGORITHMS](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2018/papers/1570436777.pdf)

##### Asynchronous coordinate fixed-point iteration


- [ARock: An Algorithmic Framework for Asynchronous Parallel Coordinate Update](https://epubs.siam.org/doi/10.1137/15M1024950)
- [Asynchronous Stochastic Coordinate Descent: Parallelism and Convergence Properties](https://epubs.siam.org/doi/abs/10.1137/140961134)

<img src="https://www.msra.cn/wp-content/uploads/2018/11/book-recommendation-distributed-machine-learning-5.jpg" width="80%">

- [Asynchronous Parallel Greedy Coordinate Descent](https://bigdata.oden.utexas.edu/publication/asynchronous-parallel-greedy-coordinate-descent/)

### Stochastic Optimization

Stochastic optimization problems are so diverse that the field has become fragmented into a Balkanized set of communities with competing algorithmic strategies and modeling styles. It is easy to spend a significant part of a career mastering the subtleties of each perspective on stochastic optimization, without recognizing the common themes.

We have developed a unified framework that covers all of these books.
We break all `sequential decision problems` into five components: `state variables`, `decision variables`, `exogenous information variables`, the `transition function` and the `objective function`, where we search over policies (functions) for making decisions.
We have identified two core strategies for designing policies (policy search, and policies based on lookahead approximations), each of which further divides into two classes, creating four classes that cover all of the solution approaches.

- [Clearing the Jungle of Stochastic Optimization](https://castlelab.princeton.edu/jungle/)

<img src = https://castlelab.princeton.edu/html/images/detvsstoch.jpg width=80% />

## Distributed Optimization Methods

[Beginning around ten years ago, the single-threaded CPU performance stopped improving significantly, due to physical limitations; it is the numbers of cores in each machine that continue to arise. Today we can buy 8-core phones, 64-core workstations, and 2.5k-core GPUs at affordable prices. On the other hand, most of our algorithms are still single-threaded, and because so, their running time is about the same as it was to ten years ago and will stay so in the future, say, ten or twenty years. To develop faster algorithms, especially for those large-scale problems that arise in signal processing, medical imaging, and machine learning, it is inevitable to consider parallel computing.](http://www.math.ucla.edu/~wotaoyin/research.html)
Machine learning especially deep learning requires more powerful `distributed and decentralized` optimization methods.

Large scale supervised machine learning methods, which are based on gradient to improve their performance, need online near-real-time feedback from massive users.

- [ ] [Math 285J: A First Course on Large-Scale Optimization Methods](http://www.math.ucla.edu/~wotaoyin/math285j.18f/)
- [ ] http://bicmr.pku.edu.cn/~wenzw/bigdata2019.html
- [ ] [Data Science meet optimzation](http://ds-o.org/)
- [ ] [Distributed optimization for control and learning](https://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=7605&context=etd)
- [ ] [ME 555: Distributed Optimization](https://sites.duke.edu/me555_07_s2015/)
- [ ] https://www.euro-online.org/websites/dso/
- [ ] https://labs.criteo.com/2014/09/poh-part-3-distributed-optimization/
- [ ] [Convex and Distributed Optimization](https://ljk.imag.fr/membres/Jerome.Malick/CDO.pdf)
- [ ] [(749g) Accelerated Parallel Alternating Method of Multipliers (ADMM) for Distributed Optimization](https://www.aiche.org/conferences/aiche-annual-meeting/2018/proceeding/paper/749g-accelerated-parallel-alternating-method-multipliers-admm-distributed-optimization)
- [ ] [8th IEEE Workshop Parallel / Distributed Computing and Optimization (PDCO 2018)](https://pdco2018.sciencesconf.org/)
- [ ] [Distributed Optimization and Control](https://www.nrel.gov/grid/distributed-optimization-control.html)
- [ ] [ADOPT: Asynchronous Distributed Constraint Optimization with Quality Guarantees](http://teamcore.usc.edu/papers/2005/aij-modi.pdf)
- [On Distributed Optimization in Networked Systems](https://people.kth.se/~kallej/grad_students/johansson_thesis08.pdfs)
- [Distributed Optimization and Control using Operator Splitting Methods](https://infoscience.epfl.ch/record/255661)
- [Foundations of Distributed and Large Scale Computing Optimization](http://www-syscom.univ-mlv.fr/~chouzeno/ECP/index.htm)
- [Distributed Optimization of Large-Scale Complex Networks](https://sites.google.com/site/paolodilorenzohp/research/adaptation-and-learning-over-complex-networks)
- [Walkman: A Communication-Efﬁcient Random-Walk Algorithm for Decentralized Optimization](http://www.math.ucla.edu/~wotaoyin/papers/decentralized_random_walk.html)
- [NOVEL GRADIENT-TYPE OPTIMIZATION ALGORITHMS FOR EXTREMELY LARGE-SCALE NONSMOOTH CONVEX OPTIMIZATION](https://www2.isye.gatech.edu/~nemirovs/Lena.pdf)
- [Projects: Structure Exploitation in Large-Scale Non-Convex Optimisation](https://optimisation.doc.ic.ac.uk/project/structure-exploitation-in-large-scale-non-convex-optimisation/)
- [A Distributed Flexible Delay-tolerant Proximal Gradient Algorithm](https://arxiv.org/abs/1806.09429)
- http://ecee.colorado.edu/marden/files/dist-opt-journal.pdf
- [Hemingway: Modeling Distributed Optimization Algorithms](http://shivaram.org/publications/hemingway-mlsys-2016.pdf)
- http://principlesofoptimaldesign.org/
- https://www.researchgate.net/project/Distributed-Optimization-and-Online-Learning
- [A continuous-time analysis of distributed stochastic gradient](https://arxiv.org/abs/1812.10995v3)


<img src="http://www.math.ucla.edu/~wotaoyin/papers/images/walkman_randomwalk.png" width = "50%" />

### Parallelizing Stochastic Gradient Descent

- [Parallel and Distributed Stochastic Learning -Towards Scalable Learning for Big Data Intelligence](https://cs.nju.edu.cn/lwj/slides/PDSL.pdf)

#### Elastic Stochastic Gradient Descent

It is based on an elastic force which links the parameters they compute with a center variable stored by the parameter server (master). The algorithm enables the local workers to perform more exploration, i.e. the algorithm allows the local variables to fluctuate further from the center variable by reducing the amount of communication between local workers and the master.

The loss function of `Elastic-SGD`
$$x^{\ast}=\arg\min_{x, x^1, x^N}\frac{1}{N}\sum_{n=1}^{N} f(x^n)+\frac{1}{2\rho N}\|x-x^n\|^2.$$

- [Deep learning with Elastic Averaging SGD](http://www.columbia.edu/~aec2163/NonFlash/Papers/EASGD_NIPS2015.pdf)

#### Parle

Parle exploits the phenomenon of wide minima that has been shown to improve generalization performance of deep networks and trains multiple “replicas” of a network that are coupled to each other using attractive potentials. It requires infrequent communication with the parameter server and is well-suited to singlemachine-multi-GPU as well as distributed settings.

The method replace the loss function by a smoother loss called local entropy
$$f_{\gamma}^{\beta}(x)=-\frac{1}{\beta}\log(G_{\gamma/\beta}\times \exp(-\beta f(x)))$$
where $G_{\gamma/\beta}$ is the Gaussian kernel with variance $\gamma/\beta$.

Parle solves for
$$x^{\ast}=\arg\min_{x, x^1, x^N}\sum_{n=1}^{N} f_{\gamma}^{\beta}(x^n)+\frac{1}{2\rho N}\| x - x^n\|^2.$$

- [Parle: parallelizing stochastic gradient descent](https://www.sysml.cc/doc/2018/174.pdf)
- [Unraveling the mysteries of stochastic gradient descent on deep neural networks](http://helper.ipam.ucla.edu/publications/dlt2018/dlt2018_14553.pdf)

#### Asynchronous Stochastic Gradient Descent

[Asynchronous Stochastic Gradient (shared memory)](http://www.stat.ucdavis.edu/~chohsieh/teaching/ECS289G_Fall2015/lecture4.pdf):
* Each thread repeatedly performs the following updates:
  * For $t = 1, 2, \cdots$
    * Randomly pick an index $i$
    * $x\leftarrow x - \nabla f_i(x)$.

Main trick: in shared memory systems, every threads can access the same parameter $x$.

[Asynchronous Accelerated Stochastic Gradient Descent](https://www.microsoft.com/en-us/research/publication/asynchronous-accelerated-stochastic-gradient-descent/):

- [Asynchronous Stochastic Gradient Descent with Delay Compensation](https://arxiv.org/abs/1609.08326)
- [Asynchronous Decentralized Parallel Stochastic Gradient Descent](http://proceedings.mlr.press/v80/lian18a/lian18a.pdf)
- https://github.com/AbduElturki/Asynchronous-Stochastic-Gradient-Descent
- [Stochastic Proximal Langevin Algorithm: Potential Splitting and Nonasymptotic Rates](http://bicmr.pku.edu.cn/conference/opt-2014/)
- [Stochastic Proximal Langevin Algorithm: Potential Splitting and Nonasymptotic Rates](https://arxiv.org/abs/1905.11768)
- [Hybrid Stochastic Gradient Descent Algorithms for Stochastic Nonconvex Optimization](https://arxiv.org/abs/1905.05920v1)

#### Hogwild

Hogwild allows processors access to shared memory with the possibility of overwriting each other's work.

- [Hogwild: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent](http://papers.nips.cc/paper/4390-hogwild-a-lock-free-approach-to-parallelizing-stochastic-gradient-descent)
***

- https://www.podc.org/data/podc2018/podc2018-tutorial-alistarh.pdf
- https://www.math.ucla.edu/~wotaoyin/math285j.18
- http://seba1511.net/dist_blog/

#### Distributed Nesterov-like gradient methods

- [Distributed Nesterov-like Gradient Algorithms](http://users.isr.ist.utl.pt/~jxavier/cdc2012c.pdf)
- [Convergence Rates of Distributed Nesterov-Like Gradient Methods on Random Networks](https://ieeexplore.ieee.org/document/6665045/)
- [Accelerated Distributed Nesterov Gradient Descent for Convex and Smooth Functions](https://nali.seas.harvard.edu/files/nali/files/2017cdc_accelerated_distributed_nesterov_gradient_descent.pdf)
- https://nali.seas.harvard.edu/
- http://users.isr.ist.utl.pt/~jxavier/

###  Parallel Coordinate Methods

- [Synchronized Parallel Coordinate Descent](http://www.stat.ucdavis.edu/~chohsieh/teaching/ECS289G_Fall2015/lecture4.pdf)
- [DISTRIBUTED ASYNCHRONOUS COMPUTATION OF FIXED POINTS](http://www.mit.edu/people/dimitrib/Distr_Comp_Fixed.pdf)
- [An Inertial Parallel and Asynchronous Fixed-Point Iteration for Convex Optimization](https://arxiv.org/abs/1706.00088)

### Distributed Non-convex Optimization Problems

- [Distributed Non-Convex First-Order Optimization and Information
Processing: Lower Complexity Bounds and Rate Optimal Algorithms](https://arxiv.org/abs/1804.02729)

### Stochastic ADMM

Linearly constrained stochastic convex optimization is given by
$$
\min_{x,y}\mathbb{E}_{\vartheta}[F(x,\vartheta)]+h(y),\\ s.t. \, Ax+By = b, x\in\mathbb{X}, y\in\mathbb{Y}.
$$
where typically the expectation $\mathbb{E}_{\vartheta}[F(x,\vartheta)]$ is some loss function and ${h}$ is regularizer to prevent from over-fitting.

The first problem is that the distribution of $\vartheta$ is unknown as well as the expectation $\mathbb{E}_{\vartheta}[F(x,\vartheta)]$ in the objective function.

**Modified Augmented Lagrangian**

Linearize $f(x)$ at $x_k$ and add a proximal term：

$$
L_{\beta}^{k}(x,y,\lambda):= f(x_k)+\left<x_k, g_k\right>+h(y)-\left< \lambda, Ax+By-b\right>+\frac{\beta}{2}{\|Ax + By-b\|}_2^2 \\+\frac{1}{2\eta_k}\|x-x_{k}\|^2
$$

where $g_k$ is  a stochastic (sub)gradient of ${f}$.

> 1. $x^{k+1}=\arg\min_{x\in\mathbf{X}}L_{\beta}^{k}(x,y^{\color{aqua}{k}},\lambda^{\color{aqua}{k}});$
> 2. $y^{k+1}=\arg\min_{y\in\mathbf{Y}} L_{\beta}^{k}(x^{\color{red}{k+1}}, y, \lambda^{\color{aqua}{k}});$
> 3. $\lambda^{k+1} = \lambda^{k} - \beta (Ax^{\color{red}{k+1}} + By^{\color{red}{k+1}}-b).$

- [Stochastic ADMM](http://proceedings.mlr.press/v28/ouyang13.pdf)
- [Accelerated Variance Reduced Stochastic ADMM](https://arxiv.org/abs/1707.03190)
- [Towards optimal stochastic ADMM](https://people.eecs.berkeley.edu/~sazadi/icml_2014.pdf) or [the talk in ICML](https://people.eecs.berkeley.edu/~sazadi/icml_2014_presentation.pdf)
- [V-Cycle or Double Sweep ADMM](http://cermics.enpc.fr/~parmenta/frejus/2018Summer04.pdf)
- https://arxiv.org/abs/1903.01786

### Distributed ADMM

Let us assume that the function $F$ has the following decomposition where each $x_i$ has its own dimension.
$$
\min_{x} F(x)\{=\sum_{i=1}^{n}f_i(x_i)\}, \\
s.t. \quad x\in\mathrm{C}
$$

We can reformulate the problem  to get
$$
\min_{x} F(x) + g(y), \\
s.t.\quad x=y
$$

where $g(z)$ is the indictor function of set $\mathrm{C}$.

We can dene an augmented Lagrangian
$$
L_{\beta}(x, y, \lambda)
= F(x)+g(y) - \lambda^{T}(x-y) + \frac{\beta}{2}{\|x-y\|}_{2}^{2} \\
= \sum_{i=1}^{n} [f_i(x_i) - {\lambda}_i(x_i -y_i)+\frac{\beta}{2}(x_i - y_i)^2] + g(y)\\
= \sum_{i=1}^{n}L_{(\beta,i)}(x_i, y, \lambda_i).
$$

We can split the optimization over $x_i$:

> 1. $x_i^{k+1} =\arg\min_{x_i} L_{(\beta,i)}(x_i, y^{k}, \lambda_i^{k})\quad i=1,2,\cdots, n;$
> 2. $y^{k+1} =\arg\min_{x_i} L_{(\beta,i)}(x_i^{\color{green}{k+1}}, y, \lambda_i^k);$
> 3. $\lambda^{k+1}=\lambda^k+\lambda (x^{\color{green}{k+1}} - y^{\color{green}{k+1}}).$

Then optimization over $x_i$ is done in parallel. Subsequently, the results are communicated
back to a master node which performs the $y$ update (usually just a projection as in the example
above) and returns back the result to other worker nodes. The update over $\lambda_i$ is again done
independently.

* http://users.isr.ist.utl.pt/~jmota/DADMM/
* http://repository.ust.hk/ir/Record/1783.1-66353
* https://ieeexplore.ieee.org/document/7039496
* [DISTRIBUTED ASYNCHRONOUS OPTIMIZATION WITH THE ALTERNATING DIRECTION METHOD OF MULTIPLIERS](http://www.iutzeler.org/pres/sem_louvain.pdf)
* [Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers by S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html)
* [Asynchronous Distributed ADMM for Consensus Optimization](http://proceedings.mlr.press/v32/zhange14.pdf)
* [Notes on Distributed ADMM](https://mojmirmutny.weebly.com/uploads/4/1/2/3/41232833/notes.pdf)


#### Distributed Optimization: Analysis and Synthesis via Circuits

+ [Distributed Optimization: Analysis and Synthesis via Circuits](http://web.stanford.edu/class/ee364b/lectures/decomp_ckt.pdf)

General form:
$$\sum_{i=1}^{n}f_i(x_i),\\
s.t. x_i= E_i y$$


#### Resource on Distributed Optimization Methods

+ [DIMACS 2012-2017 Special Focus on Information Sharing and Dynamic Data Analysis](http://dimacs.rutgers.edu/archive/SpecialYears/2012_Data/)
+ [DIMACS Workshop on Distributed Optimization, Information Processing, and Learning](http://archive.dimacs.rutgers.edu/Workshops/Learning/)
+ https://mwang.princeton.edu/
+ https://hkumath.hku.hk/~zhengqu/
+ https://zhangzk.net/
+ http://www.lx.it.pt/~bioucas/
+ https://jliang993.github.io/index.html
+ http://sites.utexas.edu/mokhtari/
- [ORQUESTRA - Distributed Optimization and Control of Large Scale Water Delivery Systems](http://is4.tecnico.ulisboa.pt/~is4.daemon/tasks/distributed-optimization/)
- [Ray is a fast and simple framework for building and running distributed applications.](https://ray.readthedocs.io/en/latest/)
- [CoCoA: A General Framework for Communication-Efficient Distributed Optimization](https://arxiv.org/abs/1611.02189)
- [Federated Optimization: Distributed Machine Learning for On-Device Intelligence](https://arxiv.org/pdf/1610.02527.pdf)
- http://web.stanford.edu/class/ee364b/lectures.html
- http://mjiit.utm.my/bio-ist/optimal-control-optimization/

## Surrogate Optimization

It is a unified principle that we optimize an objective function via sequentially optimizing surrogate functions such as **EM, ADMM**.
In another word, these methods do not dircetly optimze the objective function.

It is obvious that the choice of optimization methods relies on the objective function. Surrogate loss transform the original problem $\min_{x} f(x)$ into successive  trackable subproblems:
$$
x^{k+1}=\arg\min_{x} Q_{k}(x).
$$

We will call $Q_k(x)$ surrogate function. Surrogate function is also known as `merit function`.

As a Fabian Pedregosa asserted, a good surrogate function should:

+ Approximate the objective function.
+ Easy to optimize.

This always leads to the `convexification technique` where the surrogate function is convex. Usually, $f(x^{k+1})\leq f(x^k)$ such as EM or MM.
Generaly, it is required that $\lim_{k\to\infty}x^k\in\arg\min_{x}f(x)$.

For example, we can rewrite gradient descent in the following form
$$
x^{k+1}=\arg\min_{x} \{f(x^k)+\left<\nabla f(x^k), x-x^k\right>+\frac{1}{2\alpha_k}{\|x-x^k\|}_2^2\}.
$$

In Newton’s method, we approximate the objective with a quadratic surrogate of the form:
$$
Q_k(x) = f(x^k)+\left<\nabla f(x^k), x-x^k\right>+\frac{1}{2\alpha_k}(x-x^k)^{T}H_k(x-x^k).
$$

Note that the Hessian matrix $H_k$ is supposed to be positive definite. The quasi-Newton methods will approximate the Hessian  matrix with some inverse symmetric matrix. And they can rewrite in the principle of surrogate function, where the surrogate function is convex in the form of linear function + squared functions in some sense.  

Note that momentum methods can be rewrite in the surrogate function form:
$$
x^{k+1}=x^{k}-\alpha_{k}\nabla_{x}f(x^k)+\rho_{k}(x^k-x^{k-1}) \\
= \arg\min_{x}\{f(x^k) + \left<\nabla f(x^k), x-x^k\right> + \frac{1}{2\alpha_k} {\|x-x^k-\rho_k(x^k-x^{k-1})\|}_2^2\}.
$$

$\color{aqua}{PS}$: How we can extend it to Nesterov gradient methods or stochastic gradient methods?

* [Discover acceleration](https://ee227c.github.io/notes/ee227c-lecture06.pdf)

It is natural to replace the squared function with some non-negative function
such as mirror gradient methods
$$
x^{k+1} = \arg\min_{x} \{ f(x^k) + \left<\nabla f(x^k), x-x^k\right> + \frac{1}{\alpha_k} B(x,x^k)\}.
$$

More generally, auxiliary function $g_k(x)$ may replace the Bregman divergence $B(x, x^k)$, where the auxiliary functions $g_k(x)$ have the properties $g_k(x)\geq 0$ and $g_k(x^{k-1})=0$.

+ http://fa.bianp.net/teaching/2018/eecs227at/
+ http://fa.bianp.net/teaching/2018/eecs227at/newton.html
+ http://faculty.uml.edu/cbyrne/NBHSeminar2015.pdf

The parameters $\alpha_k$ is chosen so that the surrogate function is convex by observing that the condition number of  Hessian matrix is determined by the factor $\frac{1}{\alpha_k}$. And gradient are used in these methods to construct the surrogate function. There are still two problems in unconstrained optimization:

1. If the cost function is not smooth or differential such as absolute value function, the gradient is not available so that it is a problem to construct a convex surrogate function without gradient;
2. In another hand, there is no unified principle to construct a convex surrogate function if we know more information of higher order derivatives.

*****

Practically, we rarely meet pure black box models; rather, we know
something about structure of underlying problems
One possible strategy is:

1. approximate nonsmooth objective function by a smooth function
2. optimize smooth approximation instead (using, e.g., Nesterov’s accelerated method).

A convex function $f$ is called $(\alpha, \beta)$-smoothable if, for any $\mu > 0, \exists$ convex function $f_{\mu}$ s.t.
* $f_{\mu}(x) \leq f(x) \leq f_{\mu}(x) + \beta\mu, \forall x$
* $f_{\mu}(x)$ is $\frac{\alpha}{\mu}$ smooth.

`Moreau envelope (or Moreau-Yosida regularization)` of a convex
function $f$ with parameter $\mu > 0$ is defined as
$$
M_{\mu f}(x)=\inf_{z}\{f(z) + \frac{1}{2\mu}{\|z-x\|}_2^2\}.
$$

Minimizing $f$ and $M_f$ are equivalent.
$prox_{\mu f} (x)$ is unique point that achieves the infimum that defines
$M_{\mu f}$ , i.e.

$$
M_{\mu f}(x)=f(prox_{\mu f} (x)) + \frac{1}{2}{\|prox_{\mu f} (x)-x\|}_2^2\}.
$$

`Moreau envelope` $M_f$ is continuously differentiable with gradients
$$
\nabla M_{\mu f} = \frac{1}{\mu}(prox_{\mu f} (x)-x).
$$

This means
$$
prox_{\mu f} (x)=x-\mu \nabla M_{\mu f}
$$

i.e., $prox_{\mu f} (x)$ is gradient step for minimizing $M_{\mu f}$.

`Fenchel Conjugate` of a function ${h}$ is the function $h^{\star}$ defined by

$$
h^{\star}(x)=\sup_{z} \{\left<z, x\right>-h(z)\}.
$$


+ [Smoothing for non-smooth optimization, ELE522: Large-Scale Optimization for Data Science](http://www.princeton.edu/~yc5/ele522_optimization/lectures/smoothing.pdf)
+ [Smoothing, EE236C (Spring 2013-14)](http://www.seas.ucla.edu/~vandenbe/236C/lectures/smoothing.pdf)
+ [Smoothing and First Order Methods: A Unified Framework](https://epubs.siam.org/doi/abs/10.1137/100818327)
+ [Smooth minimization of non-smooth functions](https://link.springer.com/article/10.1007/s10107-004-0552-5)
+ https://www.zhihu.com/question/312027177/answer/893936302
+ [A unifying principle: surrogate minimization](http://fa.bianp.net/teaching/2018/eecs227at/surrogate.html)
+ [Optimization with First-Order Surrogate Functions](https://arxiv.org/pdf/1305.3120.pdf)
+ [Optimization Transfer Using Surrogate Objective Functions](https://www.semanticscholar.org/paper/Optimization-Transfer-Using-Surrogate-Objective-Lange-Hunter/a9e0444b694a804a9088a622a6123e10a04430ae)

****

[Charles Byrne](http://faculty.uml.edu/cbyrne/cbyrne.html) gives a unified treatment of some iterative optimization algorithms such as auxiliary function methods.

Young | Recent |Now
------|--------|---
<img src="http://faculty.uml.edu/cbyrne/CharlieByrneBookImage.jpg" width = "150%" />|<img src ="https://i1.rgstatic.net/ii/profile.image/551299453919233-1508451440729_Q128/Charles_Byrne.jpg" width = "400%" />|<img src="https://www.crcpress.com/authors/images/profile/author/i11230v1-charles-byrne-557af86baa1a6.jpg" width = "50%" />

[He is a featured author of CRC press ](https://www.crcpress.com/authors/i11230-charles-byrne) and [professor in UML](https://www.uml.edu/umass-BMEBT/faculty/Byrne-Charles.aspx)

`Auxiliary-Function Methods` minimize the function
$$
G_k(x) = f(x) + g_k(x)
$$
over $x\in \mathbb{S}$ for $k=1,2,\cdots$ if $f(x):\mathbb{S}\mapsto\mathbb{R}$ and $g_k(x)$ is auxiliary function.
We do not linearize the cost function as the previous methods.

Auxiliary-Function Methods(AF methods) include
* Barrier- and Penalty-Function Algorithms such as sequential unconstrained minimization (SUM) methods, interior-point methods exterior-point methods;
* Proximal Minimization Algorithms such as majorization minimization.

And an AF method is said to be in the SUMMA class if the SUMMA Inequality holds:

$$
G_k(x) - G_k(x^k)\geq g_{k+1}(x), \forall x\in \mathbb{S}.
$$

Proximal minimization algorithms using Bregman distances, called here `PMAB`,  minimize the function

$$
G_k(x) = f(x) + B(x, x^{k-1}),\forall x\in \mathbb{S}.
$$

The `forward-backward splitting (FBS)` methods is to minimize the function
$$f_1(x) + f_2(x)$$
 where both functions
are convex and $f_2(x)$ is differentiable with its gradient L-Lipschitz continuous in the Euclidean norm.  The iterative step of the FBS algorithm is
$$
x^{k} = \operatorname{prox}_{\gamma f_1}(x^{k-1}-\gamma \nabla f_2(x^{k-1})).
$$

It is equivalent to minimize
$$
G_k(x) = f(x) + \frac{1}{2\gamma} {\|x-x^{k-1}\|}_2^2-B(x, x^{k-1}),
$$

where $B(x,x^{k-1})=f_1(x)-f_1(x^{k-1})-\left<\nabla f_1(x^{k-1}),x-x^{k-1}\right>,\, 0 < \gamma\leq \frac{1}{L}$.
`Alternating Minimization (AM)` can be regarded as coordinate optimization methods with 2 blocks:

> * $p^{n+1}=\arg\min_{p\in P} f(p,q^n)$,
> * $q^{n+1}=\arg\min_{p\in Q} f(p^{n+1},q)$,

where $f(p, q), p\in  P, q\in Q$ is the objective function. It is proved that the sequence $f(p^n, q^n)$ converge tothe minimizer  if the `Five-Point Property` hold:
$$
f(p, q) + f(p, q^{n-1}) \geq f(p, q^{n}) + f(p^n, q^{n-1}).
$$

For each p in the set P, define $q(p)$ in Q as a member of Q for which $f(p; q(p)) \leq f(p; q)$, for all $q \in P$. Let $\hat{f}(p) = f(p; q(p))$.
At the nth step of AM we minimize
$$G_n(p)=f(p; q(p))+[f(p; q^{n-1})-f(p; q(p))]$$
to get $p^n$, where $g_n(p)=f(p; q^{n-1})-f(p; q(p))$ is the auxiliary function. So we can write that $G_n(p)= f(p;q(p))+g_n(p)$.

See [Iterative Convex Optimization Algorithms; Part Two: Without the Baillon–Haddad Theorem](http://faculty.uml.edu/cbyrne/NBHSeminar2015.pdf)
or [Alternating Minimization, Proximal Minimization and Optimization Transfer Are Equivalent](https://arxiv.org/abs/1512.03034) for more information on AF methods.

Expectation-Maximization can be classified to AF method.

> * $Q(\theta|\theta^{(t)})=\mathbb{E}(\ell(\theta|Y_{obs}, Z)|Y_{obs},\theta^{(t)})$;
> * $\theta^{(t+1)}=\arg\min_{\theta} Q(\theta|\theta^{(t)})$.

The $Q(\theta|\theta^{(t)})$ function is  log-likelihood function of complete data $(Y_{os}, Z)$ given $(Y_{obs}, \theta^{(t)})$.

+ [Iterative Convex Optimization Algorithms; Part Two: Without the Baillon–Haddad Theorem](http://faculty.uml.edu/cbyrne/NBHSeminar2015.pdf)
+ [Alternating Minimization, Proximal Minimization and Optimization Transfer Are Equivalent](https://arxiv.org/abs/1512.03034)
+ [A unified treatment of some iterative algorithms in signal processing and image reconstruction](http://adsabs.harvard.edu/abs/2004InvPr..20..103B)
+ [Convergence Rate of Expectation-Maximization](http://opt-ml.org/papers/OPT2017_paper_42.pdf)

*****

[Optimization using surrogate models](https://team.inria.fr/acumes/files/2015/05/cours_meta.pdf) applied to Gaussian Process models (Kriging).



+ [Optimization using surrogate models](https://team.inria.fr/acumes/files/2015/05/cours_meta.pdf)
* [Surrogate loss function](http://www.cs.huji.ac.il/~daphna/theses/Alon_Cohen_2014.pdf)
* [Divergences, surrogate loss functions and experimental design](https://people.eecs.berkeley.edu/~jordan/papers/NguWaiJor_nips05.pdf)
* [Surrogate Regret Bounds for Proper Losses](http://mark.reid.name/bits/pubs/icml09.pdf)
* [Bregman Divergences and Surrogates for Learning](https://www.computer.org/csdl/trans/tp/2009/11/ttp2009112048-abs.html)
* [Meet the Bregman Divergences](http://mark.reid.name/blog/meet-the-bregman-divergences.html)
* [Some Theoretical Properties of an Augmented Lagrangian Merit Function](http://www.ccom.ucsd.edu/~peg/papers/merit.pdf)
* https://people.eecs.berkeley.edu/~wainwrig/stat241b/lec11.pdf
* http://fa.bianp.net/blog/2014/surrogate-loss-functions-in-machine-learning/

### Relaxation and Convexification

- https://www.di.ens.fr/~aspremon/PDF/Oxford14.pdf

The methods discussed in the book `Block Relaxation Methods in Statistics` are special cases of what we shall call block-relaxation methods, although other names such as decomposition or nonlinear Gauss-Seidel or ping-pong or seesaw methods have also been used.

![block relaxation methods](https://bookdown.org/jandeleeuw6/bras/graphics/bookfig1.png)

In a block relaxation method we minimize a real-valued function of several variables by partitioning the variables into blocks. We choose initial values for all blocks, and then minimize over one of the blocks, while keeping all other blocks fixed at their current values. We then replace the values of the active block by the minimizer, and proceed by choosing another block to become active. An iteration of the algorithm steps through all blocks in turn, each time keeping the non-active blocks fixed at current values, and each time replacing the active blocks by solving the minimization subproblems. If there are more than two blocks there are different ways to cycle through the blocks. If we use the same sequence of active blocks in each iteration then the block method is called cyclic.

In the special case in which blocks consist of only one coordinate we speak of the coordinate relaxation method or the coordinate descent (or CD) method. If we are maximizing then it is coordinate ascent (or CA). The cyclic versions are CCD and CCA.

**Augmentation Methods**

[Augmentation and Decomposition Methods](https://bookdown.org/jandeleeuw6/bras/augmentation-and-decomposition-methods.html)
Note: augmentation duality.
$$ h(y)=\min_{x\in\mathcal{X}} g(x,y) $$
then
$$ \min_{x\in\mathcal{X}}f(x)=\min_{x\in\mathcal{X}}\min_{y\in\mathcal{Y}}g(x,y)=\min_{y\in\mathcal{Y}}\min_{x\in\mathcal{X}}g(x,y)=\min_{y\in\mathcal{Y}}h(y). $$

**Alternating Conditional Expectations**

[The alternating descent conditional gradient method](https://www.stat.berkeley.edu/~nickboyd/adcg/)

A ubiquitous prior in modern statistical signal processing asserts that an observed signal is the noisy measurement of a few weighted sources. In other words, compared to the entire dictionary of possible sources, the set of sources actually present is sparse. In many cases of practical interest the sources are parameterized and the measurement of multiple weighted sources is linear in their individual measurements.

As a concrete example, consider the idealized task of identifying the aircraft that lead to an observed radar signal. The sources are the aircraft themselves, and each is parameterized by, perhaps, its position and velocity relative to the radar detectors. The sparse inverse problem is to recover the number of aircraft present, along with each of their parameters.

**Convex Relaxations**

[Convex relaxations are one of the most powerful techniques for designing polynomial time approximation algorithms for NP-hard optimization problems such as
Chromatic Number, MAX-CUT, Minimum Vertex Cover etc. Approximation algorithms for these problems are developed by formulating the problem at hand as an
integer program.](https://ttic.uchicago.edu/~madhurt/Papers/sdpchapter.pdf)

**Quadratic Majorization**

A quadratic  $g$  majorizes  $f$  at  $y$  on  $\mathbb{R}^n$  if  $g(y)=f(y)$  and  $g(x)\geq f(x)$  for all  $x$. If we write it in the form
$$g(x)=f(y)+(x-y)'b+\frac12 (x-y)'A(x-y)$$

<img  title = "cali" src = "https://bookdown.org/jandeleeuw6/bras/graphics/cali.png" width = 60% />

**Majorization**

+ http://www.cs.cmu.edu/afs/cs/user/dwoodruf/www/w10b.pdf
+ [Quadratic Majorization](https://bookdown.org/jandeleeuw6/bras/quadratic-majorization.html)
+ [Majorization Methods](https://bookdown.org/jandeleeuw6/bras/majorization-methods.html)
+ [Tangential Majorization](https://bookdown.org/jandeleeuw6/bras/tangential-majorization.html)
+ [Quadratic Majorization](https://bookdown.org/jandeleeuw6/bras/quadratic-majorization.html)
+ [Sharp Majorization](https://bookdown.org/jandeleeuw6/bras/sharp-majorization.html)
+ [Using Higher Derivatives](https://bookdown.org/jandeleeuw6/bras/using-higher-derivatives.html)


## Gradient Free Optimization Methods

As shown in `Principle of Optimal Design`, `non-gradient methods` are classified into 3 classes:

[We organize the discussion of non-gradient methods in three parts, direct search methods, heuristic methods, and black-box methods. Direct search methods rely on ranking the objective function values rather than using the objective values themselves. Heuristic methods use some random process to generate new candidate solutions, and so the iteration paths and even the solutions obtained can change each time we perform a search. Black-box methods deal with problems that have no known model function structure that we can exploit. For example, functions generated by simulations have no discernible mathematical properties (like convexity), and so we refer to them as black-box functions. In this sense, all nongradient methods can be used for black-box problems. The two black-box methods described in this chapter were created to address design problems with expensive simulations, and so their main goal is to find an optimum quickly with few function evaluations.](http://principlesofoptimaldesign.org/)

> Derivative-free optimization (DFO) algorithms differ in the way they use the sampled function values to determine the new iterate. One class of methods constructs a linear
or quadratic model of the objective function and defines the next iterate by seeking to minimize this model inside a trust region. We pay particular attention to these model-based
approaches because they are related to the unconstrained minimization methods described in earlier chapters. Other widely used DFO methods include the simplex-reflection method
of Nelder and Mead, pattern-search methods, conjugate-direction methods, and simulated annealing.

`Heuristic methods` will be introduced in computational intelligence as well as `Bayesian Optimization`.

Let us start with the example and suppose that we want to
$$
\arg\min_{x}f(x)=x^{2}+4\sin(2x).\tag{1}
$$

The objective function $f(x)$, a non-convex function, has many local minimizer or extrema.
The function (1) is upper bounded by $x^2+4$ and lower bounded by $x^2 - 4$.

![pluskid](http://freemind.pluskid.org/att/2016/03/nonconvex.svg)

Another insightful example is to minimize the following cost function:
$$
x^{2}+4\sin(2x) - 1001 \color{red}{\mathbb{I}_{\sqrt{2}}}(x) \tag{2}
$$
where the last part $\color{red}{\mathbb{I}_{\sqrt{2}}}(x)$ is equal to 1 when $x=\sqrt{2}$ and 0 otherwise, a Dirac function.
It is almost equal to the first function (1) except at the point $x=\sqrt{2}$.
The minimal value of the above function is  $2+4\sin(2\sqrt{2})-1001$ when $x=\sqrt{2}$.
And these two functions are two different kinds of non-convex functions.


[Optimization and Assumptions @ Freemind](http://freemind.pluskid.org/misc/optimization-and-assumptions/)|[Test functions for optimization](https://www.wikiwand.com/en/Test_functions_for_optimization)

- [Book: Introduction to Derivative-Free Optimization](http://www.mat.uc.pt/~lnv/idfo/)
- [Derivative-free optimization methods](http://www.optimization-online.org/DB_FILE/2019/04/7153.pdf)

#### Zeroth-Order Oracle

[Zeroth-Order optimization is increasingly embraced for solving big data and machine learning problems when explicit expressions of the gradients are difficult or infeasible to obtain. It achieves gradient-free optimization by
approximating the full gradient via efficient gradient estimators.](http://www.comp.hkbu.edu.hk/~cib/2018/Dec/article4/iib_vol19no2_article4.pdf)

- [Learning to Learn by Zeroth-Order Oracle](https://www.groundai.com/project/learning-to-learn-by-zeroth-order-oracle/)
- https://openreview.net/forum?id=BJe-DsC5Fm
- [ZOO: Zeroth Order Optimization based Black-box Attacks to Deep Neural Networks without Training Substitute Models](https://arxiv.org/abs/1708.03999)
- [Recent Advances of Zeroth-Order Optimization with Applications in Adversarial ML](https://2018.ieeeglobalsip.org/sym/18/AML)
- [Adversarial Learning and Zeroth Order Optimization for Machine Learning and Data Mining](https://www.ibm.com/blogs/research/2019/08/adversarial-learning/)
- https://sites.google.com/view/adv-robustness-zoopt
- [Zeroth-Order Optimization and Its Application to Adversarial Machine Learning](http://www.comp.hkbu.edu.hk/~cib/2018/Dec/article4/iib_vol19no2_article4.pdf)
- [Zeroth-Order Stochastic Variance Reduction for Nonconvex Optimization](https://arxiv.org/abs/1805.10367v1)
- https://sgo-workshop.github.io/index_2018.html
- [New Advances in Sparse Learning, Deep Networks, and Adversarial Learning: Theory and Applications](http://reports-archive.adm.cs.cmu.edu/anon/ml2019/CMU-ML-19-103.pdf)
- https://www.researchgate.net/profile/Sijia_Liu2
- http://web.cs.ucla.edu/~chohsieh/publications.html
- https://deepai.org/machine-learning/researcher/sijia-liu
- [Zeroth-order (Non)-convex stochastic optimization via conditional gradient and gradient updates](https://dl.acm.org/citation.cfm?id=3327264)


#### Block Coordinate Descent

The methods such as ADMM, proximal gradient methods do not optimize the cost function directly.
For example, we want to minimize the following cost function
$$
f(x,y) = g(x) + h(y)
$$

with or without constraints.
Specially, if the cost function is additionally separable, $f(x) = f_1(x_1) + f_2(x_2) + \cdots + f_n(x_n)$, we would like to minimize the sub-function or component function $f_i(x_i), i=1,2,\cdots, n$ rather than the cost function itself

$$
\min_{x} \sum_{i} f_i(x_i) \leq \sum_{i}\min_{x_i}{f_i(x_i)}.
$$

And ADMM or proximal gradient methods are to split the cost function to 2 blocks, of which one is differentiable and smooth while the other may not be differentiable. In another word, we can use them to solve some non-smooth optimization problem.
However, what if there is no constraints application to the optimization problem?

**Coordinate descent** is aimed to minimize the following cost function
$$
f(x) = g(x) +\sum_{i}^{n} h_i(x_i)
$$

where $g(x)$ is convex, differentiable and each $h_i(x)$ is convex.
We can use coordinate descent to find a minimizer: start with some initial guess $x^0$, and repeat for $k = 1, 2, 3, \dots$:

***

> 1. $x_{1}^{k} \in\arg\min_{x_1}f(x_1, x_2^{k-1}, x_3^{k-1}, \dots, x_n^{k-1});$
> 2. $x_{2}^{k} \in\arg\min_{x_1}f(x_1^{\color{red}{k}}, x_2,x_3^{k-1},\dots, x_n^{k-1});$
> 3. $x_{3}^{k} \in\arg\min_{x_1}f(x_1^{\color{red}{k}}, x_2^{\color{red}{k}},x_3,\dots, x_n^{k-1});$
> 4. $\vdots$
> 5. $x_{n}^{k} \in\arg\min_{x_1}f(x_1^{\color{red}{k}}, x_2^{\color{red}{k}},x_3^{\color{red}{k}},\dots, x_n).$

It can extended to block coordinate descent(`BCD`) if the variables ${x_1, x_2, \dots, x_n}$ are separable in some blocks.

***
- http://bicmr.pku.edu.cn/conference/opt-2014/index.html
- https://calculus.subwiki.org/wiki/Additively_separable_function
- https://www.cs.cmu.edu/~ggordon/10725-F12/slides/25-coord-desc.pdf
- http://bicmr.pku.edu.cn/~wenzw/opt2015/multiconvex_BCD.pdf
- http://pages.cs.wisc.edu/~swright/LPS/sjw-abcr-v3.pdf
- [MS72: Recent Progress in Coordinate-wise Descent Methods](https://meetings.siam.org/sess/dsp_programsess.cfm?SESSIONCODE=66077)
* [On Nesterov’s Random Coordinate Descent Algorithms](http://ranger.uta.edu/~heng/CSE6389_15_slides/nesterov10efficiency.pdf)
* [On Nesterov’s Random Coordinate Descent Algorithms - Continued](http://ranger.uta.edu/~heng/CSE6389_15_slides/nesterov10efficiency2.pdf)
* [Fast Coordinate Descent Methods with Variable Selection for NMF](http://ranger.uta.edu/~heng/CSE6389_15_slides/Fast%20Coordinate%20Descent%20Methods%20with%20Variable%20Selection%20for.pdf)
* https://ryancorywright.github.io/
* http://www.optimization-online.org/DB_FILE/2014/12/4679.pdf
* https://www.math.ucdavis.edu/~sqma/MAT258A_Files/Beck-CD-2013.pdf
#### Block Splitting Methods

- [Block Splitting for Distributed Optimization by N. Parikh and S. Boyd](https://web.stanford.edu/~boyd/papers/block_splitting.html)



***

+ https://pratikac.github.io/
+ [Block Relaxation Methods in Statistics by Jan de Leeuw](https://bookdown.org/jandeleeuw6/bras/)
+ [Deep Relaxation: partial differential equations for optimizing deep neural networks](https://arxiv.org/abs/1704.04932)
+ [Deep Relaxation tutorials](http://www.adamoberman.net/uploads/6/2/4/2/62426505/2017_08_30_ipam.pdf)
+ [CS 369S: Hierarchies of Integer Programming Relaxations](https://web.stanford.edu/class/cs369h/)
+ [Convex Relaxations and Integrality Gaps](https://ttic.uchicago.edu/~madhurt/Papers/sdpchapter.pdf)
+ [LP/SDP Hierarchies Reading Group](https://www.win.tue.nl/~nikhil/hierarchies/index.html)
+ [Proofs, beliefs, and algorithms through the lens of sum-of-squares](https://www.sumofsquares.org/public/index.html)
+ [Iterative Convex Optimization Algorithms; Part Two: Without the Baillon–Haddad Theorem](http://faculty.uml.edu/cbyrne/NBHSeminar2015.pdf)
+ [Relaxation and Decomposition Methods for Mixed Integer Nonlinear Programming](https://www.springer.com/gp/book/9783764372385)

*****

![nonconvex](https://www.math.hu-berlin.de/~stefan/B19/nonconvex.gif)

In order for primal-dual methods to be applicable to a constrained minimization problem, it is necessary that restrictive convexity conditions are satisfied.
A nonconvex problem can be convexified and transformed into one which can be solved with the aid of primal-dual methods.

+ [Convexification and Global Optimization in Continuous and Mixed-Integer Nonlinear Programming: Theory, Algorithms, Software, and Applications](https://b-ok.cc/book/2099773/6478de)
+ [Convexification and Global Optimization of Nonlinear Programs](https://www-user.tu-chemnitz.de/~helmberg/workshop04/tawarmalani.pdf)
+ [Convexification Procedure and Decomposition Methods for Nonconvex Optimization Problem](http://59.80.44.100/web.mit.edu/dimitrib/www/Convexification_Mult.pdf)
+ [Conic Optimization Theory: Convexification Techniques and Numerical Algorithms](https://arxiv.org/abs/1709.08841)
+ [Convexification of polynomial optimization problems by means of monomial patterns](http://www.optimization-online.org/DB_FILE/2019/01/7034.pdf)
+ [On convexification/optimization of functionals including an $\ell^2$-misfit term](http://www.maths.lth.se/matematiklu/personal/mc/On%20convexification%20MP%20version%202.pdf)
+ [Lossless Convexification of Nonconvex Control Bound and Pointing Constraints of the Soft Landing Optimal Control Problem](http://larsblackmore.com/iee_tcst13.pdf)
+ [A General Class of Convexification Transformation for the Noninferior Frontier of a Multiobjective Program](http://file.scirp.org/Html/8-1040011_31681.htm)
+ [Implementation of a Convexification Technique for Signomial Functions](http://www.users.abo.fi/alundell/files/Escape19.pdf)
+ [Sequential quadratic programming](https://web.cse.ohio-state.edu/~parent.1/classes/788/Au10/OptimizationPapers/SQP/actaSqp.pdf)
+ [A method to convexify functions via curve evolution](https://www.tandfonline.com/doi/abs/10.1080/03605309908821476)

### Surrogate-based Optimization

- [Surrogate-based methods for black-box optimization](https://www.lix.polytechnique.fr/~liberti/itor16.pdf)
- https://rdrr.io/cran/suropt/
- http://cdn.intechopen.com/pdfs/30305/InTech-Surrogate_based_optimization.pdf
- https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20050186653.pdf
- [Surrogate-Based Optimization](https://link.springer.com/chapter/10.1007/978-3-319-04367-8_3)
- [Surrogate-based Optimization using Mutual Information for Computer Experiments (optim-MICE)](https://arxiv.org/abs/1909.04600)



### Graduated Optimization

Another related method is `graduated optimization`, which [is a global optimization technique that attempts to solve a difficult optimization problem by initially solving a greatly simplified problem, and progressively transforming that problem (while optimizing) until it is equivalent to the difficult optimization problem.](https://www.wikiwand.com/en/Graduated_optimization)Further, when certain conditions exist, it can be shown to find an optimal solution to the final problem in the sequence. These conditions are:

+ The first optimization problem in the sequence can be solved given the initial starting point.
+ The locally convex region around the global optimum of each problem in the sequence includes the point that corresponds to the global optimum of the previous problem in the sequence.

|Graduated Optimization|
|:---:|
|![Graduated opt](https://upload.wikimedia.org/wikipedia/commons/b/be/Graduated_optimization.svg)|

+ https://www.wikiwand.com/en/Numerical_continuation
+ [Multi-Resolution Methods and Graduated Non-Convexity](http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/BMVA96Tut/node29.html)

### Kiefer-Wolfowitz Algorithm

In stochastic gradient descent, the estimated gradient is a partial sum of the population gradient so that it is necessary to compute the gradient of `sample` function.
`Kiefer-Wolfowitz Algorithm` is the gradient-free version of stochastic gradient descent.
It is a recursive scheme to approximate the minimum or maximum  of the form
$$
x^{k+1} =x^{k} - a_n \Delta(x^k)
$$

During the n-th stage, observations $y^{\prime\prime}$ and $y^{\prime}$ are taken at the design levels $x^{\prime\prime}=x^k+c_k$ and $x^{\prime}=x^k - c_k$, respectively. And $\Delta(x^k)=\frac{y^{\prime\prime} - y^{\prime}}{2c_n}$, $a_n$ and $c_n$ are positive constants so that $c_n\to 0, \sum_{i=0}^{\infty}(a_n/c_n)^2< \infty, \sum a_n =\infty$.


<img title="function with noise" src="http://pawel.sawicz.eu/wp-content/uploads/2014/08/example-of-function.png" width="70%" />

+ http://pawel.sawicz.eu/tag/kiefer-wolfowitz/
+ [A compansion to Kiefer-Wolfowit algorithm](https://projecteuclid.org/euclid.aos/1188405629)
+ [Archive for Kiefer-Wolfowitz algorithm](https://xianblog.wordpress.com/tag/kiefer-wolfowitz-algorithm/)
+ [Strong Convergence of a Stochastic Approximation Algorithm](https://projecteuclid.org/euclid.aos/1176344212)
+ [Almost Sure Approximations to the Robbins-Monro and Kiefer-Wolfowitz Processes with Dependent Noise](https://projecteuclid.org/euclid.aop/1176993921)
+ [A Kiefer–Wolfowitz Algorithm with Randomized Differences](http://lsc.amss.ac.cn/paper-pdf/hfchen-1.pdf)
+ [Stochastic Approximation by Tze Leung Lai](https://statistics.stanford.edu/sites/g/files/sbiybj6031/f/2002-31.pdf)


### Multi-Level Optimization

[Multi-Level Optimization](https://www.cs.ubc.ca/labs/lci/mlrg/slides/mirrorMultiLevel.pdf) is to optimize a related cheap function $\hat{f}$ when the objective function $f$  is very expensive to evaluate.
> Multi-level optimization methods repeat three steps:
> 1. Cheap minimization of modified $\hat{f}$:
  $$y^{k}=\arg\min_{x\in \mathbb{R}^p} + \left<v_k, x\right>.$$
> 2. Use $y^{k}$ to give descent direction,
  $$x^{k+1} = x^k -a_k(x^k - y^k) .$$
> 3. Set $v_k$ to satisfy first-order coherence
$$v_{k+1}=\frac{L_f}{L_F} F^{\prime}(x^{k+1}) - f^{\prime}(x^{k+1}).$$

+ [Panos Parpas](https://www.imperial.ac.uk/people/panos.parpas)
+ [Panos Parpas](http://www.doc.ic.ac.uk/~pp500/)
+ [EMERITUS PROFESSORBERCRUSTEM](https://www.imperial.ac.uk/people/b.rustem/publications.html)
+ [Multilevel Optimization Methods: Convergence and Problem Structure](http://www.optimization-online.org/DB_HTML/2016/11/5701.html)
+ [A Multilevel Proximal Algorithm for Large Scale Composite Convex Optimization](https://www.kcl.ac.uk/nms/depts/mathematics/news/)
+ [Multiresolution Algorithms for Faster
Optimization in Machine Learning](https://gateway.newton.ac.uk/sites/default/files/asset/doc/1805/Parpas%20ML-Workshop-May-2018.pdf)

![The basic scheme of multilevel optimization](http://www.iosotech.com/img/text/ml_sch.gif)

The simplified scheme of work for the `multilevel optimization` procedure can be represented as follows.

1. Solving the optimization problem using a surrogate model. For this purpose, the method of indirect optimization based on the self-organization (IOSO) is used. This method allows finding the single solution for single-objective optimization or the Pareto-optimal set of solutions for multi-objective problems.
2. For the obtained solution the indicators of efficiency are updated using the high-fidelity analysis tools.
3. The adjustment of a current search region is performed.
4. The adjustment of the surrogate model is performed. Depending upon the particular features of the applied mathematical simulation, the adjustment procedure can performed using several approaches. One such approach involves constructing non-linear corrective dependencies. This includes evaluation of the results obtained with different fidelity analysis tools. The other possible approach is application of nonlinear estimation of surrogate model internal parameters.
5. Replacement of the surrogate model by the adjusted one and the return to step (1).


+ [Multilevel Optimization: Convergence Theory, Algorithms and Application to Derivative-Free Optimization](http://www.mtm.ufsc.br/~melissa/arquivos/thesis_melissa.pdf)
+ [multilevel optimization iosotech](http://www.iosotech.com/multilevel.htm)
+ [OptCom: A Multi-Level Optimization Framework for the Metabolic Modeling and Analysis of Microbial Communities](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3271020/)

### Reactive Search Optizmiation

Roberto Battiti and Mauro Brunato explains that how `reactive search optimization, reactive affine shaker(RAS),  memetic algorithm` works for optimization problem without gradient in  [Machine Learning plus Intelligent Optimization](https://intelligent-optimization.org/LIONbook/).

* [Curriculum-based course timetabling solver; uses Tabu Search or Simulated Annealing](https://github.com/stBecker/CB-CTT_Solver)
* [Machine Learning plus Intelligent Optimization by Roberto Battiti and Mauro Brunato](https://intelligent-optimization.org/LIONbook/)
* [Reactive Search and Intelligent Optimization](https://www.springer.com/gp/book/9780387096230)

****
- [Zeroth-Order Method for Distributed Optimization With Approximate Projections](http://or.nsfc.gov.cn/bitstream/00001903-5/487435/1/1000014935638.pdf)
- [Derivative-Free Optimization (DFO)](https://www.gerad.ca/Sebastien.Le.Digabel/MTH8418/)
- [Derivative Free Optimization / Optimisation sans Gradient](http://dumas.perso.math.cnrs.fr/V04.html)
- [Delaunay-based derivative-free optimization via global surrogates, part I: linear constraints](http://fccr.ucsd.edu/pubs/BCB16.pdf)
- [Delaunay-based derivative-free optimization via global surrogates, part II: convex constraints](http://fccr.ucsd.edu/pubs/BB16a.pdf)
- https://projects.coin-or.org/Dfo
- https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
- http://adl.stanford.edu/aa222/Lecture_Notes_files/chapter6_gradfree.pdf
- https://code.fb.com/ai-research/nevergrad/
- https://github.com/facebookresearch/nevergrad
- https://www.kthohr.com/optimlib.html
- https://www.infoq.com/presentations/black-box-optimization

![never grad](https://code.fb.com/wp-content/uploads/2018/12/nevergrad_hero_v1.gif)
***
And there are more topics on optimization such as [this site](http://mat.uab.cat/~alseda/MasterOpt/IntroHO.pdf).
And more courses on optimization:

+ [凸优化 (2018年秋季) by 文在文](http://bicmr.pku.edu.cn/~wenzw/opt-2018-fall.html)
+ [2010- Shanghai Jiao Tong University 张小群](http://math.sjtu.edu.cn/faculty/xqzhang/html/teaching.html)
+ [Optimisation: Master's degree in Modelling for Science and Engineering](http://mat.uab.cat/~alseda/MasterOpt/)
+ [A Course on First-Order, Operator Splitting, and Coordinate Update Methods for Optimization by Yinwo Tao](http://www.math.ucla.edu/~wotaoyin/summer2016/)
+ [Math 273C: Numerical Optimizatoin by Yinwo Tao](http://www.math.ucla.edu/~wotaoyin/math273c/)
+ [Math 273, Section 1, Fall 2009: Optimization, Calculus of Variations, and Control Theory](http://www.math.ucla.edu/~lvese/273.1.10f/)
+ [ECE236B - Convex Optimization (Winter Quarter 2018-19) by Prof. L. Vandenberghe, UCLA](http://www.seas.ucla.edu/~vandenbe/ee236b/ee236b.html)
+ [EECS 127 / 227AT: Optimization Models and Applications  —  Fall 2018
Instructors: A. Bayen, L. El Ghaoui.](https://people.eecs.berkeley.edu/~elghaoui/Teaching/EECS127/index.html)
+ [IEOR 262B: Mathematical Programming II by Professor Javad Lavaei, UC Berkeley](https://lavaei.ieor.berkeley.edu/Course_IEOR262B_Spring_2019.html)
+ https://web.stanford.edu/~boyd/teaching.html
+ [EE364b - Convex Optimization II](http://web.stanford.edu/class/ee364b/lectures.html)
+ [Convex Optimization: Fall 2018
Machine Learning 10-725](http://www.stat.cmu.edu/~ryantibs/convexopt/)
+ [Algorithms for Convex Optimization](https://nisheethvishnoi.wordpress.com/convex-optimization/)
+ [Optimization by Vector Space Methods](https://courses.engr.illinois.edu/ECE580/sp2019/)
+ [Algorithms for Convex Optimization](https://nisheethvishnoi.wordpress.com/convex-optimization/)
+ [EE 227C (Spring 2018), Convex Optimization and Approximation](https://ee227c.github.io/)
+ https://ocw.mit.edu/courses/sloan-school-of-management/15-093j-optimization-methods-fall-2009/

***
* http://niaohe.ise.illinois.edu/IE598_2016/index.html
- [ ] http://www.optimization-online.org/
- [ ] http://convexoptimization.com/
- [ ] More [Optimization Online Links](http://www.optimization-online.org/links.html)
- [ ] **TUTORIALS AND BOOKS** at <http://plato.asu.edu/sub/tutorials.html>.
- [ ] [Provable Nonconvex Methods/Algorithms](http://sunju.org/research/nonconvex/)
- [ ] https://optimization.mccormick.northwestern.edu/index.php/Main_Page
- [ ] [Non-convex Optimization for Machine Learning](https://arxiv.org/abs/1712.07897)
- [ ] [GLOBAL OPTIMALITY CONDITIONS FOR DEEP NEURAL NETWORKS](https://arxiv.org/pdf/1707.02444.pdf)
- [ ] http://www.vision.jhu.edu/assets/HaeffeleCVPR17.pdf
- [ ] https://www.springer.com/us/book/9783319314822
- [ ] https://core.ac.uk/display/83849901
- [ ] https://core.ac.uk/display/73408878
- [ ] https://zhuanlan.zhihu.com/p/51514687
- [ ] http://math.cmu.edu/~hschaeff/research.html
- [ ] https://people.eecs.berkeley.edu/~elghaoui/Teaching/EECS127/index.html
- [ ] https://blogs.princeton.edu/imabandit/
- [ ] http://www.probabilistic-numerics.org/research/index.html
- [ ] https://people.eecs.berkeley.edu/~brecht/eecs227c.html
- [ ] https://neos-guide.org/content/optimization-under-uncertainty
- [ ] [Optimization and Gradient Descent on Riemannian Manifolds](https://wiseodd.github.io/techblog/2019/02/22/optimization-riemannian-manifolds/)
- [ ] [Approximation Theory & Convex Optimization](https://homepages.laas.fr/lasserre/drupal/content/approximation-theory-convex-optimization)
- https://people.maths.ox.ac.uk/riseth/
- https://www.aritradutta.com/
- http://sites.utexas.edu/mokhtari/
- https://www.di.ens.fr/~aspremon/
