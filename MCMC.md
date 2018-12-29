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

### Metropolis

### Gibbs Sampling 

Gibbs sampling is a conditional sampling  technique. It is known that $P(X_1, \dots, X_n) = P(X_1)\prod_{i=2}^{n}P(X_i|X_1,\dots, X_{i-1})$.

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

### The Hybrid Monte Carlo Algorithm

*MCMC Using Hamiltonian Dynamics*, Radford M. Neal said 
> In 1987, a landmark paper by Duane, Kennedy, Pendleton, and Roweth united the MCMC and molecular dynamics approaches. They called their method “hybrid Monte Carlo,” which abbreviates to “HMC,” but the phrase “Hamiltonian Monte Carlo,” retaining the abbreviation, is more specific and descriptive, and I will use it here.

* The first step is to define a Hamiltonian function in terms of the probability distribution we wish to sample from.
* In addition to the variables we are interested in (the “position” variables), we must introduce auxiliary “momentum” variables, which typically have independent Gaussian distributions.
* The HMC method alternates simple updates for these momentum variables with Metropolis updates in which a new state is proposed by computing a trajectory according to Hamiltonian dynamics, implemented with the leapfrog method.

***

* [Hamiltonian Monte Carlo 1](http://khalibartan.github.io/MCMC-Hamiltonian-Monte-Carlo-and-No-U-Turn-Sampler/)
* [Roadmap of HMM](https://metacademy.org/graphs/concepts/hamiltonian_monte_carlo)
* https://chi-feng.github.io/mcmc-demo/app.html#HamiltonianMC,banana
* http://slac.stanford.edu/pubs/slacpubs/4500/slac-pub-4587.pdf
* http://www.mcmchandbook.net/HandbookChapter5.pdf