## Markov Chain Monte Carlo

The practice of MCMC is simple. Set up a Markov chain having the required invariant distribution, and run it on a computer.
And in theory, probability distribution and expected values are the reflection of measure and integration theory. ANd it is a great begin to probabilistic programming.

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


### Metropolis

### Gibbs Sampling 

Gibbs sampling is a conditional sampling  technique. It is known that $P(X_1, \dots, X_n) = P(X_1)\prod_{i=1}^{n}P(X_i|X_1,\dots, X_{i-1})$.

* https://metacademy.org/graphs/concepts/gibbs_sampling

### Slice Sampling

* https://projecteuclid.org/euclid.aos/1056562461
* https://www.wikiwand.com/en/Slice_sampling

### The Hybrid Monte Carlo Algorithm

* [Hamiltonian Monte Carlo 1](http://khalibartan.github.io/MCMC-Hamiltonian-Monte-Carlo-and-No-U-Turn-Sampler/)
* [Roadmap of HMM](https://metacademy.org/graphs/concepts/hamiltonian_monte_carlo)
* https://chi-feng.github.io/mcmc-demo/app.html#HamiltonianMC,banana