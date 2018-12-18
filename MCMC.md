## Markov Chain Monte Carlo

1. http://www.mcmchandbook.net/
2. http://www.cs.princeton.edu/courses/archive/spr06/cos598C/papers/AndrieuFreitasDoucetJordan2003.pdf
3. https://math.uchicago.edu/~shmuel/Network-course-readings/MCMCRev.pdf
4. https://skymind.ai/wiki/markov-chain-monte-carlo
5. https://chi-feng.github.io/mcmc-demo/app.html#HamiltonianMC,banana
6. http://probability.ca/jeff/ftpdir/lannotes.4.pdf
7. https://www.seas.harvard.edu/courses/cs281/papers/andrieu-defreitas-doucet-jordan-2002.pdf


***

* https://www.seas.harvard.edu/courses/cs281/papers/neal-1998.pdf
* http://www.mcmchandbook.net/HandbookChapter1.pdf
* https://www.seas.harvard.edu/courses/cs281/papers/roberts-rosenthal-2003.pdf
* https://twiecki.github.io/blog/2015/11/10/mcmc-sampling/
* [Hamiltonian Monte Carlo 1](http://khalibartan.github.io/MCMC-Hamiltonian-Monte-Carlo-and-No-U-Turn-Sampler/)
* [Roadmap of HMM](https://metacademy.org/graphs/concepts/hamiltonian_monte_carlo)
* [PyMC2](https://colcarroll.github.io/hamiltonian_monte_carlo_talk/bayes_talk.html)
https://cosx.org/2013/01/lda-math-mcmc-and-gibbs-sampling

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
