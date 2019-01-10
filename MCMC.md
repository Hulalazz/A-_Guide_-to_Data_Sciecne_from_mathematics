## Markov Chain Monte Carlo

The practice of MCMC is simple. Set up a *Markov chain* having the required invariant distribution, and run it on a computer.
And in theory, probability distribution and expected values are the reflection of measure and integration theory. And it is a  begin to the tour of probabilistic programming.
It also may provide some mathematical understanding of the machine learning model.
It is more useful in some simulation problem.

1. http://www.mcmchandbook.net/
2. http://www.cs.princeton.edu/courses/archive/spr06/cos598C/papers/AndrieuFreitasDoucetJordan2003.pdf
3. https://math.uchicago.edu/~shmuel/Network-course-readings/MCMCRev.pdf
4. https://skymind.ai/wiki/markov-chain-monte-carlo
5. https://metacademy.org/graphs/concepts/markov_chain_monte_carlo
6. http://probability.ca/jeff/ftpdir/lannotes.4.pdf
7. https://www.seas.harvard.edu/courses/cs281/papers/andrieu-defreitas-doucet-jordan-2002.pdf
8. https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/


***

* https://www.seas.harvard.edu/courses/cs281/papers/neal-1998.pdf
* https://www.seas.harvard.edu/courses/cs281/papers/roberts-rosenthal-2003.pdf
* https://twiecki.github.io/blog/2015/11/10/mcmc-sampling/
* [PyMC2](https://colcarroll.github.io/hamiltonian_monte_carlo_talk/bayes_talk.html)
* https://cosx.org/2013/01/lda-math-mcmc-and-gibbs-sampling
* https://chi-feng.github.io/mcmc-demo/


### Gibbs Sampling

Gibbs sampling is a conditional sampling  technique. It is known that $P(X_1, \dots, X_n) = P(X_1)\prod_{i=2}^{n}P(X_i|X_1,\dots, X_{i-1})$.

> **Algorithm**: Gibbs Sampling
   * Initialize $\{{z}_i^{(0)}\}_{i=1}^{n}$;
   * For $t=1,\dots,T$:
       + Draw $z_{1}^{(t+1)}$ from $P(z_{1}|z_{2}^{\color{green}{(t)}},z_{3}^{(t)},\dots, z_{n}^{(t)})$;
       + Draw $z_{2}^{(t+1)}$ from $P(z_{2}|z_{1}^{\color{red}{(t+1)}},z_{3}^{(t)},\dots, z_{n}^{(t)})$;
       +  $\vdots$;
       + Draw $z_{j}^{(t+1)}$ from $P(z_{j}|z_{1}^{\color{red}{(t+1)}},\dots, z_{j-1}^{\color{red}{(t+1)}}, z_{j+1}^{(t)}, z_{n}^{(t)})$;
       + $\vdots$;
       + Draw $z_{n}^{(t+1)}$ from $P(z_{n}|z_{1}^{\color{red}{(t+1)}},\dots, z_{j-1}^{\color{red}{(t+1)}}, z_{n-1}^{\color{red}{(t+1)}}$.

* https://metacademy.org/graphs/concepts/gibbs_sampling
* https://metacademy.org/graphs/concepts/gibbs_as_mh

### Importance sampling

Let $\vec{X}$ be a random vector, and we wan to compute the integration or the expectation
$$\mu=\mathbb{E}(f(\vec{X}))=\int_{\mathbb{R}}f(\vec{X})p({X})\mathrm{d}X,$$
where $p({X})$ is the probability density function of $\vec{X}$.
We can rewrite the expectation
$$\mu=\int_{\mathbb{R}}\frac{f(\vec{X})p(X)}{q(X)}q(X)\mathbb{d}X=\mathbb{E}_{q}(\frac{f(\vec{X})p(X)}{q(X)}),$$
where $q(X)$ is another probability density function and $q(X)=0$ implies $f(\vec{X})p(X)=0$.

***
The [algorithm of importance sampling](http://math.arizona.edu/~tgk/mc/book_chap6.pdf) is as following:
 1. Generate samples $\vec{X}_1,\vec{X}_2,\cdots,\vec{X}_n$ from the distribution $q(X)$;
 2. Compute the estimator $\hat{\mu}_{q} =\frac{1}{n}\sum_{i=1}^{n}\frac{f(\vec{X_i})p(\vec{X_i})}{q(\vec{X_i})}$
***

See more in
* [Wikipedia page](https://www.wikiwand.com/en/Importance_sampling).
* [Stanford statweb](https://statweb.stanford.edu/~owen/mc/Ch-var-is.pdf)

### Metropolis–Hastings

The Metropolis–Hastings algorithm involves designing a **Markov process** (by constructing transition probabilities) which fulfills the existence of stationary distribution and uniqueness of stationary distribution conditions, such that its stationary distribution $\pi (x)$ is chosen to be $P(x)$.

The approach is to separate the transition in two sub-steps; the proposal and the acceptance-rejection. The `proposal distribution` ${\displaystyle \displaystyle g(x'|x)}$ is the conditional probability of proposing a state $x'$ given $x$, and the `acceptance ratio` ${\displaystyle \displaystyle A(x',x)}$ the probability to accept the proposed state $x'$.The transition probability can be written as the product of them:
$$P(x'|x)=g(x'|x) A(x', x).$$
Inserting this relation in the previous equation, we have
$$\frac{A(x', x)}{A(x, x')}=\frac{P(x'|x)}{P(x|x')} \frac{g(x|x')}{g(x'|x)}=\frac{P(x')}{P(x)} \frac{g(x|x')}{g(x'|x)}$$
where we infer $\frac{P(x'|x)}{P(x|x')} = \frac{P(x')}{P(x)}$ from the fact that $P(x'|x)P(x)=P(x',x)=P(x|x')P(x')$.
The next step in the derivation is to choose an acceptance that fulfils the condition above. One common choice is the Metropolis choice:
$$A(x', x)=\min(1, \frac{P(x')}{P(x)} \frac{g(x|x')}{g(x'|x)})$$
i.e., we always accept when the acceptance is bigger than 1, and we reject accordingly when the acceptance is smaller than 1.

The Metropolis–Hastings algorithm thus consists in the following:
***
1. Initialise
    + Pick an initial state $x_{0}$;
    + Set $t=0$;
2. Iterate
    + Generate: randomly generate a candidate state $x'$ according to ${\displaystyle g(x'|x_{t})}$;
    + Calculate: calculate the acceptance probability $A(x',x_{t})=\min(1, \frac{P(x')}{P(x)} \frac{g(x|x')}{g(x'|x)})$;
    + Accept or Reject:
         - generate a uniform random number ${\displaystyle u\in [0,1]}$;
         - if ${\displaystyle u\leq A(x',x_{t})}$, accept the new state and set ${\displaystyle x_{t+1}=x'}$;
         - if ${\displaystyle u>A(x',x_{t})}$, reject the new state, and copy the old state forward $x_{t+1}=x_{t}$;
   + Increment: set ${\textstyle t=t+1}$;

* https://www.wikiwand.com/en/Metropolis%E2%80%93Hastings_algorithm
* https://metacademy.org/graphs/concepts/metropolis_hastings

### Slice Sampling

Slice sampling, in its simplest form, samples uniformly from underneath the curve $f(x)$ without the need to reject any points, as follows:

* Choose a starting value $x_0$ for which $f(x_0)>0$.
* Sample a y value uniformly between $0$ and $f(x_0)$.
* Draw a horizontal line across the curve at this $y$ position.
* Sample a point $(x,y)$ from the line segments within the curve.
* Repeat from step 2 using the new $x$ value.

The motivation here is that one way to sample a point uniformly from within an arbitrary curve is first to draw thin uniform-height horizontal slices across the whole curve. Then, we can sample a point within the curve by randomly selecting a slice that falls at or below the curve at the x-position from the previous iteration, then randomly picking an x-position somewhere along the slice. By using the x-position from the previous iteration of the algorithm, in the long run we select slices with probabilities proportional to the lengths of their segments within the curve.

***

* https://projecteuclid.org/euclid.aos/1056562461
* https://www.wikiwand.com/en/Slice_sampling

### Simulated Annealing

**Simulated annealing** is a global optimization method inspired by physical annealing model.
If $f^{\star}=\min_{x}f(x)$, the probability $p(x)\propto \exp(-f(x))$ attains its highest probability at the point $f^{\star}$. In short, the optimization really connect with sampling.


* https://en.wikipedia.org/wiki/Simulated_annealing
* http://mathworld.wolfram.com/SimulatedAnnealing.html

### The Hybrid Monte Carlo Algorithm

*MCMC Using Hamiltonian Dynamics*, Radford M. Neal said
> In 1987, a landmark paper by Duane, Kennedy, Pendleton, and Roweth united the MCMC and molecular dynamics approaches. They called their method “hybrid Monte Carlo,” which abbreviates to “HMC,” but the phrase “Hamiltonian Monte Carlo,” retaining the abbreviation, is more specific and descriptive, and I will use it here.

* The first step is to define a Hamiltonian function in terms of the probability distribution we wish to sample from.
* In addition to the variables we are interested in (the "position" variables), we must introduce auxiliary "momentum" variables, which typically have independent Gaussian distributions.
* The HMC method alternates simple updates for these momentum variables with Metropolis updates in which a new state is proposed by computing a trajectory according to Hamiltonian dynamics, implemented with the leapfrog method.

***

* https://arxiv.org/pdf/1206.1901.pdf
* [Hamiltonian Monte Carlo](http://khalibartan.github.io/MCMC-Hamiltonian-Monte-Carlo-and-No-U-Turn-Sampler/)
* [Roadmap of HMM](https://metacademy.org/graphs/concepts/hamiltonian_monte_carlo)
* https://chi-feng.github.io/mcmc-demo/app.html#HamiltonianMC,banana
* http://slac.stanford.edu/pubs/slacpubs/4500/slac-pub-4587.pdf
* http://www.mcmchandbook.net/HandbookChapter5.pdf
* https://physhik.github.io/HMC/
* http://arogozhnikov.github.io/2016/12/19/markov_chain_monte_carlo.html
* https://theclevermachine.wordpress.com/2012/11/18/mcmc-hamiltonian-monte-carlo-a-k-a-hybrid-monte-carlo/

#### Hamiltonian Dynamics

Hamiltonian dynamics are used to describe how objects move throughout a system. Hamiltonian dynamics is defined in terms of object location $x$ and its momentum $p$ (equivalent to object’s mass $m$ times velocity $\nu$, i.e., $p=m\nu$) at some time $t$. For each location of object there is an associated potential energy $U(x)$ and with momentum there is associated kinetic energy $K(p)$.The total energy of system is **constant** and is called as Hamiltonian $H(x,p)$, defined as the sum of potential energy and kinetic energy:
$$
H(x,p)=U(x)+K(p)
$$
The partial derivatives of the Hamiltonian determines how position $x$ and momentum $p$ change over time $t$, according to Hamiltonian’s equations:
$$
\frac{\mathrm{d}x_i}{\mathrm{d} t}=\frac{\partial H}{\partial p_i}=\frac{\partial K(p)}{\partial p_i}  \\
\frac{\mathrm{d} p_i}{\mathrm{d} t}= -\frac{\partial H}{\partial x_i}= -\frac{\partial U(x)}{\partial x_i}
$$
The above equations operates on a d-dimensional position vector $x$ and a d-dimensional momentum vector $p$, for $i=1,2,\cdots,d$.

#### Hamiltonian and Probability: Canonical Distributions

To relate $H(x,p)$ to target distribution $P(x)$ we use a concept from statistical mechanics known as the canonical distribution. For any energy function $E(q)$, defined over a set of variables $q$, we can find corresponding $P(q)$:
$$
P(q)=\frac{1}{z}e^{-\frac{E(q)}{T}}
$$
where $Z$ is normalizing constant called **Partition function** (so that $\int_{\mathbb{R}}P(q)\mathrm{d}q =1$) and $T$ is temperature of system.
Since, the Hamiltonian is an energy function for the joint state of “position”, $x$ and “momentum”, $p$, so we can define a joint distribution for them as follows:
$$
P(x,p)=\frac{exp(-H(x,p))}{z}=\frac{exp[-U(x)-K(p)]}{z}.
$$
Furthermore we can associate probability distribution with each of the potential and kinetic energy ($P(x)$ with potential energy and $P(p)$, with kinetic energy). Thus, we can write above equation as:
$$
P(x,p)=\frac{P(x)P(p)}{Z^{\prime}}
$$
where $Z^{\prime}$ is new normalizing constant. 

Since joint distribution factorizes over $x$ and $p$, we can conclude that $P(x)$ and $P(p)$ are independent. Because of this independence we can choose any distribution from which we want to sample the momentum variable. A common choice is to use a zero mean and unit variance Normal distribution $N(0,I)$. The target distribution of interest $P(x)$ from which we actually want to sample from is associated with potential energy, i.e.,
$$
U(x) = −\log(P(x)).
$$

#### Hamiltonian Monte Carlo

Given initial state $x_0$, stepsize $\epsilon$, number of steps $L$, log density function $U$, number of samples to be drawn $M$
1. set m=0
2. repeat until m=M
    + set $m\leftarrow m+1$
    + Sample new initial momentum $p_0 \sim N(0,I)$
    + Set $x_m\leftarrow x_{m−1}, x^{\prime}\leftarrow x_{m−1}, p^{\prime}\leftarrow p_0$
    + repeat for $L$ steps

        + Set $x^{\prime}, p^{\prime}\leftarrow Leapfrog(x^{\prime},p^{\prime},\epsilon)$
    + Calculate acceptance probability $α=\min(1,\frac{exp(U(x^{\prime})−(p′.p′)/2)}{exp(U(x_m−1)−(p_0.p_0)/2)})$
    + Draw a random number $u \sim Uniform(0, 1)$
    + if $u\leq \alpha$ then $x_m\leftarrow x^{\prime},p_m\leftarrow p^{\prime}$.

Leapfrog is a function that runs a single iteration of Leapfrog method.

+ http://khalibartan.github.io/MCMC-Hamiltonian-Monte-Carlo-and-No-U-Turn-Sampler/
+ http://khalibartan.github.io/MCMC-Metropolis-Hastings-Algorithm/
+ http://khalibartan.github.io/Introduction-to-Markov-Chains/
+ http://khalibartan.github.io/Monte-Carlo-Methods/
+ https://blog.csdn.net/qy20115549/article/details/54561643