## Stochastic Processes

For more information see the [Wikipedia page](https://www.wikiwand.com/en/Stochastic_process).
> To every outcome $\omega$ we now assign – according to a certain rule – a
function of time $\xi(\omega, t)$, real or complex. We have thus created a family
of functions, one for each $\omega$. This family is called a stochastic process
(or a random function). from [the little book](https://www.math.uwaterloo.ca/~mscott/Little_Notes.pdf)

For example, if $\xi(t)$
is a real stochastic process, then its cumulative distribution function is given by
$$F(x;t)=P(\xi(t)\leq x).$$

| Stochastic Process and Management |
|:---------------------------------:|
| <img titile="www.azquotes.com" src="https://www.azquotes.com/picture-quotes/quote-a-stochastic-process-is-about-the-results-of-convolving-probabilities-which-is-just-anthony-stafford-beer-111-68-26.jpg" width = "80%" />

- http://wwwf.imperial.ac.uk/~ejm/M3S4/INTRO.PDF


### Poisson Process

Poisson process is named after the statistician [**Siméon Denis Poisson**](https://www.wikiwand.com/en/Sim%C3%A9on_Denis_Poisson).
> *Definition*:The homogeneous Poisson point process, when considered on the positive half-line, can be defined as a counting process, a type of stochastic process, which can be denoted as  $\{N(t),t\geq 0\}$. A counting process represents the total number of occurrences or events that have happened up to and including time $t$. A counting process is a Poisson counting process with rate $\lambda >0$ if it has the following three properties:
> 
* $N(0)=0$;
* has independent increments, i.e. $N(t)-N(t-\tau)$ and $N(t+\tau)-N(t)$ are independent for $0\leq\tau\leq{t}$; and
* the number of events (or points) in any interval of length $t$ is a Poisson random variable with parameter (or mean) $\lambda t$.

|Siméon Denis Poisson|
|:------------------:|
|![Siméon Denis Poisson](http://www.nndb.com/people/857/000093578/poisson-2-sized.jpg)|

##### Poisson distribution

It is a discrete distribution with the pmf:
$$
P(X=k)=e^{-\lambda}\frac{\lambda^k}{k!}, k\in\{0,1,\cdots\},
$$
where  $k!$ is the factorial of $k$, i.e. $k!=k\times{(k-1)}\times{(k-2)}\cdots\times{2}\times{1}$.
See more in [Wikipedia page](https://www.wikiwand.com/en/Poisson_distribution).

### Markov Chain

Markov chain is a simple stochastic process in honor of **Andrey (Andrei) Andreyevich Marko**.
It is the fundamental of Markov Chain Montel Carlo(**MCMC**).
> Definition: The process $X$ is a **Markov**  chain if it satisfies the *Markov condition*:
> $$P(X_{n}=x_{n}|X_{1}=x_{1},\cdots,X_{n-1}=x_{n-1})=P(X_{n}=x_{n}|X_{n-1}=x_{n-1})$$
>for all $n\geq 1$.

People introduced to Markov chains through a typical course on stochastic processes have
usually only seen examples where the state space is finite or countable. If the state space
is finite, written $\{x_1,\cdots,x_n\}$, then the initial distribution can be associated with a vector $\lambda =(\lambda_1,\cdots, \lambda_n)$ defined by

$$
P(X=x_i)=\lambda_i\quad i=1,\cdots, n,
$$

and the **transition probabilities** can be associated with a matrix $P$ having elements $p_{i j}$
defined by

$$
P(X_{n+1}=x_i|X_n=x_j)=p_{i j}\quad i=1,\cdots, n,\text{and}\, j=1,\cdots,n.
$$

When the state space is countably infinite, we can think of an infinite vector and matrix. And $\sum_{j=1}^{n}p_{ij}=1$.
![](http://iacs-courses.seas.harvard.edu/courses/am207/blog/files/images/Markov_ex1.png)

But most Markov chains of interest in **MCMC** have uncountable state space, and then we
cannot think of the initial distribution as a vector or the transition probability distribution
as a matrix. We must think of them as an unconditional probability distribution and a
conditional probability distribution.

![markov](http://www.statslab.cam.ac.uk/~rrw1/markov/chipfiring.png)

#### Stationarity

A stochastic process is stationary if for every positive integer $k$ the
distribution of the k-tuple
$$(X_{n+1},X_{n+2}, \cdots , X_{n+k})$$
does not depend on $n$. A Markov chain is stationary if it is a stationary stochastic process.
In a Markov chain, the conditional distribution of $(X_{n+2}, \cdots , X_{n+k})$ given $X_{n+1}$ does not depend
on $n$. It follows that a Markov chain is stationary if and only if the marginal distribution of $X_n$ does not depend on $n$.

An initial distribution is said to be **stationary** or **invariant** or **equilibrium** for some transition probability distribution if the Markov chain specified by this initial distribution and transition probability distribution is stationary. We also indicate this by saying that the transition probability distribution **preserves** the initial distribution.

Stationarity implies stationary transition probabilities, but not vice versa.

Let ${P}$ be the transition matrix of a Markov chain with state
space ${S}$. A probability distribution $\pi = (\pi_1, \pi_2, \dots)$ on ${S}$ satisfying
$$\pi P=\pi$$
is called a stationary distribution of the chain.

It is the largest eigenvalue problem of the transition matrix ${P}$. It is also the fixed point problem of the linear operator ${P}$ so that the `Anderson acceleration` may speed up the convergence in its iterative computational process.

#### Reversibility

A transition probability distribution is reversible with respect to an initial distribution if, for
the Markov chain $X_1, X_2, \cdots$ they specify, the distribution of pairs $(X_i, X_{i+1})$ is exchangeable.

A Markov chain is reversible if its transition probability is reversible with respect to its initial distribution. Reversibility implies stationarity, but not vice versa. A reversible Markov chain has the same laws running forward or backward in time, that is, for any $i$ and $k$ the distributions of $(X_{i+1}, \cdots , X_{i+k})$ and $(X_{i+k}, \cdots , X_{i+1})$ are the same.
***

|[Andrey (Andrei) Andreyevich Markov,1856-1922](https://www.wikiwand.com/en/Andrey_Markov)|
|:---------------------------------------------------------------------------------------:|
|<img src=https://upload.wikimedia.org/wikipedia/commons/a/a8/Andrei_Markov.jpg width=40% />|
|[his short biography in .pdf file format](https://wayback.archive-it.org/all/20121218173228/https://netfiles.uiuc.edu/meyn/www/spm_files/Markov-Work-and-life.pdf)|
|[Hamiltonian Monte Carlo explained](http://arogozhnikov.github.io/2016/12/19/markov_chain_monte_carlo.html)|

* [Introduction to Markov Chain Monte Carlo by Charles J. Geyer](http://www.mcmchandbook.net/HandbookChapter1.pdf)
* [6 Markov Chains](http://wwwf.imperial.ac.uk/~ejm/M3S4/NOTES3.pdf)

### Random Walk on Graph and Stochastic Process

> A random walk on a directed graph consists of a sequence of vertices generated from
> a start vertex by selecting an edge, traversing the edge to a new vertex, and repeating the process. From [Foundations of Data Science](https://www.microsoft.com/en-us/research/publication/foundations-of-data-science/)

| Random Walk | Markov Chain |
|:-----------:|:------------:|
| graph | stochastic process |
|    vertex   |     state    |
|strongly connected | persistent |
|  aperiodic  |  aperiodic  |
|strongly connected  and aperiodic | ergotic |
| undirected graph |  time reversible |

A random walk in the Markov chain starts at some state. At a given time step, if it is in state ${x}$, the next state ${y}$ is selected
randomly with probability $p_{xy}$. A Markov chain can be represented by a directed graph
with a vertex representing each state and a directed edge with weight $p_{xy}$ from vertex ${x}$
to vertex ${y}$.
We say that the Markov chain is *connected* if the underlying directed graph
is *strongly connected*. The matrix ${P}$ consisting of the $p_{xy}$ is called the transition probability matrix of
the chain.

A state of a Markov chain is `persistent` if it has the property that should the state ever
be reached, the random process will return to it with probability ${1}$.This is equivalent
to the property that the state is in a strongly connected component with no out edges.

A connected Markov Chain is said to be `aperiodic` if the greatest common divisor of the lengths of directed cycles is ${1}$.
It is known that for connected aperiodic chains, the
probability distribution of the random walk converges to a unique stationary distribution.
Aperiodicity is a technical condition needed in this proof.

The `hitting time` $h_{xy}$, sometimes called discovery time, is the expected time of a random walk starting at vertex ${x}$ to reach vertex ${y}$.

In an undirected graph where $\pi_x p_{xy} = \pi_y p_{yx}$,  edges can be assigned weights such that
$$P_{xy} = \frac{w_{xy}}{\sum_y w_{xy}}.$$

Thus the Metropolis-Hasting algorithm and Gibbs
sampling both involve random walks on edge-weighted undirected graphs.

****

**Definition**: Let ${X(t), t \geq 0}$ be a collection of discrete random
variables taking values in some set ${S}$ and that evolves in time as follows:

* (a) If the current state is ${i}$, the time until the state is changed has an exponential distribution with parameter $\lambda(i)$.
* (b) When state ${i}$ is left, a new state  $j\neq i$ is chosen according to the transition probabilities of a discrete-time Markov chain.

Then $\{X(t)\}$ is called a continuous-time Markov chain.

- Markov property: $P(X(t)=j\mid X(t_1)=i_1,\dots, X(t_n)=i_n)=P(X(t)=j\mid X(t_n)=i_n)$ for any $n>1$ and $t_1<t_2<\cdots<t_n<t$.
- Time Homogeneity: $P(X(t)=j\mid X(s)=i) = P(X(t-1)=j\mid X(0)=i)$ for $0<s<t$.

And discrete stochastic process is matrix-based computation while the stochastic calculus is the blend of differential equation and statistical analysis.

Define $p_{ij}(s, t+s)= P(X(t+s)=j\mid X(s)=i), p_{ij}(0, t)=P(X(t)=j\mid X(0)=i)$, continuous time analogue of C-K equations we obtain $P(s+t)=P(s)P(t)$, where $P_{ij}(t)=(p_{ij}(t))$.
And the initial probability is defined as follows:
$$
p_{ij}(0)=
\begin{cases}1, & \quad\text{if} \quad i=j; \\
0, & \quad\text{otherwise}.
\end{cases}
$$ 

So that $P(0)$ is the identity matrix.
$p_{ij} (t)$ is  continuous and differentiable so that we could compute the derivative:
$$q_{ij}=\frac{\mathrm{d}}{\mathrm{d} t}p_{ij}(t)\mid_{t=0} =\lim_{h\to 0}\frac{p_{ij}(h)-p_{ij}(0)}{h}.$$

Therefore, by taking first order Taylor expansion of $p_{ij}(h)$ at $t=0$, 
$$
p_{ij}(h)=p_{ij}(0)+q_{ij}h+o(h)=
\begin{cases}
1 + q_{ij}h+o(h) &\text{if $i=j$}\\
\,\,\,\,\quad q_{ij}h+o(h) &\text{otherwise}.
\end{cases}
$$

Let _Q_ be the matrix of $q_{ij}$, i.e. $Q_{ij}=q_{ij}$.
Note that the state must transfer to one of the finite state:
$$\sum_{j}p_{ij}=1 \quad \forall i.$$
so that by taking the derivative of both sides:
$$\frac{\mathrm{d}}{\mathrm{d} t}\sum_{j}p_{ij}=\frac{\mathrm{d}}{\mathrm{d} t}1,\\ \sum_{j}q_{ij}=0,$$
 i.e. rows of _Q_ sum to zero.
A continuous time Markov process may be specified by stating its _Q_ matrix.

**Description of process**

Let $T_i$ be the time spent in state $i$ before moving to another state.
$$P(T_i\leq t)= 1- e^{tq_{ii}}$$

**Embedded Markov Chain**

If the process is observed only at jumps, then a Markov chain is observed with transition matrix
$$
P=\begin{pmatrix}
\ddots  & -\frac{q_{ij}}{q_{ii}}  &-\frac{q_{ij}}{q_{ii} } &\cdots \\
-\frac{q_{ij}}{q_{ii}} & 0  &-\frac{q_{ij}}{q_{ii}} &\cdots \\
\vdots & -\frac{q_{ij}}{q_{ii}} & 0 &\cdots \\
\vdots & \cdots & \cdots &\cdots 
\end{pmatrix}
$$

known as the **Embedded Markov Chain**.

**Forward and Backward Equations**

Given _Q_, how do we get $P(t), t \geq 0$ ?


***
* [MA8109 Stochastic processes and differential equations](https://wiki.math.ntnu.no/ma8109/2015h/start)
* [Martingales and the ItÙ Integral](https://www.math.ntnu.no/emner/MA8109/2013h/notes/HEK2011/MartingalesAndIto2011.pdf)
* [Continuous time Markov processes](http://wwwf.imperial.ac.uk/~ejm/M3S4/NOTEScurrent.PDF)
* [Course information, a blog, discussion and resources for a course of 12 lectures on Markov Chains to second year mathematicians at Cambridge in autumn 2012.](https://www.statslab.cam.ac.uk/~rrw1/markov/)
* [Markov Chains: Engel's probabilistic abacus](http://www.statslab.cam.ac.uk/~rrw1/markov/index2011.html)
* [Reversible Markov Chains and Random Walks on Graphs by Aldous ](https://www.stat.berkeley.edu/~aldous/RWG/book.html)
* [Markov Chains and Mixing Times by David A. Levin, Yuval Peres, Elizabeth L. Wilmer](https://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf)
* [Random walks and electric networks by Peter G. Doyle J. Laurie Snell](https://www.math.dartmouth.edu//~doyle/docs/walks/walks.pdf)
* [The PageRank Citation Ranking: Bringing Order to the Web](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf)
* [能否通俗的介绍一下Hamiltonian MCMC？](https://www.zhihu.com/question/41289973/answer/248935294)
  
### Time Series Analysis

Time series analysis is the statistical counterpart of stochastic process in some sense.
In practice, we may observe a series of random variable but its hidden parameters are not clear before some computation.
It is the independence that makes the parameter estimation of statistical models and  time series analysis, which will cover more practical consideration.

[Time series data often arise when monitoring industrial processes or tracking corporate business metrics. The essential difference between modeling data via time series methods or using the process monitoring methods discussed earlier in this chapter is the following:
Time series analysis accounts for the fact that data points taken over time may have an internal structure (such as autocorrelation, trend or seasonal variation) that should be accounted for.](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm)  

**[A brief history of time series analysis](https://www.statistics.su.se/english/research/time-series-analysis/a-brief-history-of-time-series-analysis-1.259451)**
> 
> The theoretical developments in time series analysis started early with stochastic processes. The first actual application of autoregressive models to data can be brought back to the work of G. U Yule and J. Walker in the 1920s and 1930s.
> 
> During this time the moving average was introduced to remove periodic fluctuations in the time series, for example fluctuations due to seasonality. Herman Wold introduced ARMA (AutoRegressive Moving Average) models for stationary series, but was unable to derive a likelihood function to enable maximum likelihood (ML) estimation of the parameters.
> 
> It took until 1970 before this was accomplished. At that time, the classic book "Time Series Analysis" by G. E. P. Box and G. M. Jenkins came out, containing the full modeling procedure for individual series: specification, estimation, diagnostics and forecasting.
> 
> Nowadays, the so-called Box-Jenkins models are perhaps the most commonly used and many techniques used for forecasting and seasonal adjustment can be traced back to these models.
>
> The first generalization was to accept multivariate ARMA models, among which especially VAR models (Vector AutoRegressive) have become popular. These techniques, however, are only applicable for stationary time series. However, especially economic time series often exhibit a rising trend suggesting non-stationarity, that is, a unit root.
>
> Tests for unit roots developed mainly during the 1980:s. In the multivariate case, it was found that non-stationary time series could have a common unit root. These time series are called cointegrated time series and can be used in so called error-correction models within both long-term relationships and short-term dynamics are estimated.
> 
> **ARCH and GARCH models**
> Another line of development in time series, originating from Box-Jenkins models, are the non-linear generalizations, mainly ARCH (AutoRegressive Conditional Heteroscedasticity) - and GARCH- (G = Generalized) models. These models allow parameterization and prediction of non-constant variance. These models have thus proved very useful for financial time series. The invention of them and the launch of the error correction model gave C. W. J Granger and R. F. Engle the Nobel Memorial Prize in Economic Sciences in 2003.
>
> Other non-linear models impose time-varying parameters or parameters whose values changes when the process switches between different regimes. These models have proved useful for modeling many macroeconomic time series, which are widely considered to exhibit non-linear characteristics.

* [Introduction to Time Series Analysis](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm)
* [Time Series Analysis with R](http://r-statistics.co/Time-Series-Analysis-With-R.html)
* [Time series analysis (FMSN45/MASM17)](http://www.maths.lu.se/kurshemsida/time-series-analysis/)
* [6CCM344A Time Series Analysis, Lecturer: Professor Michael Pitt](https://www.kcl.ac.uk/nms/depts/mathematics/study/current/handbook/progs/modules/6CCM344a)