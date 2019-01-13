## Stochastic Processes

Stochastic processes focus on the independent random variables.
For more information see the [Wikipedia page](https://www.wikiwand.com/en/Stochastic_process).

![](https://www.azquotes.com/picture-quotes/quote-a-stochastic-process-is-about-the-results-of-convolving-probabilities-which-is-just-anthony-stafford-beer-111-68-26.jpg)

### Poisson Process

Poisson process is named after the statistician [**Siméon Denis Poisson**](https://www.wikiwand.com/en/Sim%C3%A9on_Denis_Poisson).
> *Definition*:The homogeneous Poisson point process, when considered on the positive half-line, can be defined as a counting process, a type of stochastic process, which can be denoted as  $\{N(t),t\geq 0\}$. A counting process represents the total number of occurrences or events that have happened up to and including time $t$. A counting process is a Poisson counting process with rate $\lambda >0$ if it has the following three properties:
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
and the transition probabilities can be associated with a matrix $P$ having elements $p_{i j}$
defined by
$$
P(X_{n+1}=x_i|X_n=x_j)=p_{i j}\quad i=1,\cdots, n,\text{and}\, j=1,\cdots,n.
$$
When the state space is countably infinite, we can think of an infinite vector and matrix.
But most Markov chains of interest in **MCMC** have uncountable state space, and then we
cannot think of the initial distribution as a vector or the transition probability distribution
as a matrix. We must think of them as an unconditional probability distribution and a
conditional probability distribution.

#### Stationarity

A stochastic process is stationary if for every positive integer $k$ the
distribution of the k-tuple
$$(X_{n+1},X_{n+2}, \cdots , X_{n+k})$$
does not depend on $n$. A Markov chain is stationary if it is a stationary stochastic process.
In a Markov chain, the conditional distribution of $(X_{n+2}, \cdots , X_{n+k})$ given $X_{n+1}$ does not depend
on $n$. It follows that a Markov chain is stationary if and only if the marginal distribution of $X_n$ does not depend on $n$.

An initial distribution is said to be **stationary** or **invariant** or **equilibrium** for some transition probability distribution if the Markov chain specified by this initial distribution and transition probability distribution is stationary. We also indicate this by saying that the transition probability distribution **preserves** the initial distribution.

Stationarity implies stationary transition probabilities, but not vice versa.

#### Reversibility

A transition probability distribution is reversible with respect to an initial distribution if, for
the Markov chain $X_1, X_2, \cdots$ they specify, the distribution of pairs $(X_i, X_{i+1})$ is exchangeable.

A Markov chain is reversible if its transition probability is reversible with respect to its initial distribution. Reversibility implies stationarity, but not vice versa. A reversible Markov chain has the same laws running forward or backward in time, that is, for any $i$ and $k$ the distributions of $(X_{i+1}, \cdots , X_{i+k})$ and $(X_{i+k}, \cdots , X_{i+1})$ are the same.
***

|[Andrey (Andrei) Andreyevich Markov,1856-1922](https://www.wikiwand.com/en/Andrey_Markov)|
|:---------------------------------------------------------------------------------------:|
|![](https://upload.wikimedia.org/wikipedia/commons/a/a8/Andrei_Markov.jpg)|
|[his short biography in .pdf file format](https://wayback.archive-it.org/all/20121218173228/https://netfiles.uiuc.edu/meyn/www/spm_files/Markov-Work-and-life.pdf)|
|http://arogozhnikov.github.io/2016/12/19/markov_chain_monte_carlo.html|

* http://www.mcmchandbook.net/HandbookChapter1.pdf
* http://wwwf.imperial.ac.uk/~ejm/M3S4/NOTES3.pdf

### Random Walk and Stochastic Process

A random walk on a directed graph consists of a sequence of vertices generated from
a start vertex by selecting an edge, traversing the edge to a new vertex, and repeating the process.

| Random Walk | Markov Chain |
|:-----------:|:------------:|
| graph | stochastic process |
|    vertex   |     state    |
|strongly connected | persistent |
|  aperiodic  |  aperiodic  |
|strongly connected  and aperiodic | ergotic |
| undirected graph |  time reversible |

https://www.zhihu.com/question/41289973/answer/248935294
