# Bayesian Learning and Probabilistic Programming

Bayesian learning can be regarded as the extension of Bayesian statistics. The core topic of Bayesian learning is thought as prior information to explain the uncertainty of parameters.
It is related with `Bayesian statistics, computational statistics, probabilistic programming and machine learning`.

* [[Bayesian] “我是bayesian我怕谁”系列 - Gaussian Process](http://www.cnblogs.com/jesse123/p/7802258.html)
* [Bayesian Learning](http://www.cs.cmu.edu/afs/cs/project/theo-20/www/mlbook/ch6.pdf)
* [Lecture 9: Bayesian Learning](http://www.cogsys.wiai.uni-bamberg.de/teaching/ss05/ml/slides/cogsysII-9.pdf)
* [Bayesian Learning](https://frnsys.com/ai_notes/machine_learning/bayesian_learning.html)
* [Bayesian machine learning](https://metacademy.org/roadmaps/rgrosse/bayesian_machine_learning)
* [Infinite Mixture Models with Nonparametric Bayes and the Dirichlet Process](http://blog.echen.me/2012/03/20/infinite-mixture-models-with-nonparametric-bayes-and-the-dirichlet-process/)
* [Bayesian Learning for Machine Learning: Part I - Introduction to Bayesian Learning](https://wso2.com/blog/research/part-one-introduction-to-bayesian-learning)
* [Bayesian Learning for Machine Learning: Part II - Linear Regression](https://wso2.com/blog/research/part-two-linear-regression)
* [Bayesian Learning by Burr H. Settles, CS-540, UW-Madison, www.cs.wisc.edu/~cs540-1](http://pages.cs.wisc.edu/~bsettles/cs540/lectures/15_bayesian_learning.pdf)
* [Bayesian machine learning @fastML](http://fastml.com/bayesian-machine-learning/)
* [Understanding emprical Bayesian hierarchical learning](http://varianceexplained.org/r/hierarchical_bayes_baseball/)
* [While My MCMC Gently Samples - Bayesian modeling, Data Science, and Python](https://twiecki.io/)
* [Probabilistic Models in the Study of Language](http://idiom.ucsd.edu/~rlevy/pmsl_textbook/text.html)
* [Statistical Rethinking A Bayesian Course with Examples in R and Stan (& PyMC3 & brms too)](http://xcelab.net/rm/statistical-rethinking/)
+ [COS597C: Advanced Methods in Probabilistic Modeling](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/#prerequisites)
+ [COS513: Foundations of Probabilistic Modeling](http://www.cs.princeton.edu/courses/archive/spring09/cos513/)
+ [ CSSS 564: Bayesian Statistics for the Social Sciences (University of Washington, Spring 2018)](https://github.com/UW-CSSS-564/2018/blob/master/notes/links.md)
+ [Learn Bayesian statistics](https://docs.pymc.io/learn.html)
+ http://ryanrossi.com/search.php
+ https://bayesianwatch.wordpress.com/

![](https://frnsys.com/ai_notes/assets/model_based_clustering.svg)

|Bayes Formulae | Inverse Bayes Formulae|
|---|---|
|$f_X(x)=\frac{f_{X, Y}(X, Y)}{f_{Y\mid X}(y\mid x)}=\frac{f_{X\mid Y}(x\mid y)f_Y(y)}{f_{Y\mid X}(y\mid x)}$|$f_X(x) = (\int_{S_y} \frac{ f_{Y\mid X}(y\mid x)}{f_{X\mid Y}(x\mid y)}\mathrm{d}y)^{-1}$|
|$f_X(x)\propto f_{X\mid Y}(x\mid y)f_Y(y)(=f_{X, Y}(X, Y))$|$f_X(x) \propto \frac{f_{X\mid Y}(x\mid y_0)}{f_{Y\mid X}(y_0\mid x)}$|

## Naive Bayes

`Naive Bayes` is to reconstruct the joint distribution of features and labels $Pr(\vec{x}, y)$ given some training dataset/samples $T=\{(\vec{X}_i, y_i)\}_{i=1}^{n}$.
However, the features are usually in high dimensional space in practice so the dimension curse occurs which makes it impossible to compute the joint distribution $Pr(\vec{X}, y)$ via the (emprical) conditional probability $Pr(\vec{X}\mid y)$ and the prior $Pr(y)$.
A `naive` idea is to simplify the computation process by assumption that the features are conditional independence so that
$$
Pr(\vec{X}\mid y) =\prod_{i=1}^{p} Pr(\vec{X}^{(i)}\mid y).\tag{1}
$$  

And the predicted labels will be computed via
$$
Pr(y\mid \vec{X}) = \frac{Pr(y) Pr(\vec{x}\mid y)}{\sum Pr(y)Pr(\vec{x}\mid y)}. \tag{2}
$$

where the conditional probability $Pr(\vec{X}\mid y)$ is simplified by the conditional independence assumption in formula (1).
Thus the naive Bayes classifier is represented as maximum a posteriori (MAP)
$$
y=f(x)=\arg_{y} Pr(y\mid \vec{X}).
$$

The prior probability $Pr(y)$ can be emprical or estimated.

* [Naive Bayesian](https://www.saedsayad.com/naive_bayesian.htm)
* [Ritchie Ng on Machine Learning](https://jrnold.github.io/bayesian_notes/naive-bayes.html)
* [MLE/MAP + Naïve Bayes](https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture5-nb.pdf)


## Gaussian Naive Bayes

* https://www.ritchieng.com/machine-learning-gaussian-naive-bayes/
* [Gaussian Naïve Bayes](https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture6-gnb.pdf)
* [Naive Bayes and Gaussian Bayes Classifier](https://www.cs.toronto.edu/~urtasun/courses/CSC411/tutorial4.pdf)

## Average One-Dependence Estimator (AODE)

* [Not so naive Bayes: Aggregating one-dependence estimators](https://perun.pmf.uns.ac.rs/radovanovic/dmsem/cd/install/Weka/doc/classifiers-papers/bayes/AODE/WebbBoughtonWang04.pdf)
* https://link.springer.com/article/10.1007%2Fs10994-005-4258-6


## Hierarchical Bayesian Regression

`Hierarchical Bayesian Regression` extends the Bayesian models by setting  the uncertainty  of the uncertainty such as
$$
y\sim P(\phi(x)\mid \theta)\\
P(\phi(x)\mid \theta) = \frac{P(\theta\mid\phi(x))P(\phi(x))}{P(\theta)}\\
P(\phi(x))= P(\phi(x)\mid \eta)P(\eta)\\
\vdots
$$

![https://www.cnblogs.com/huangxiao2015/p/5667941.html](https://images2015.cnblogs.com/blog/941880/201607/941880-20160713184714686-1547363080.png)

![HierarchicalDiagram](http://www.indiana.edu/~kruschke/BMLR/HierarchicalDiagram.jpg)

We can take any factor into consideration in this hierarchical Bayesian model. And it is a graphical probability model, which consists of the connections and probability.

- [ ] https://twiecki.io/
- [ ] https://twiecki.io/blog/2014/03/17/bayesian-glms-3/
- [ ] https://twiecki.io/blog/2018/08/13/hierarchical_bayesian_neural_network/
- [ ] [BAYESIAN HIERARCHICAL MODELS](https://www.stat.ubc.ca/~gavin/STEPIBookNewStyle/course_clapem.html)
- [ ] https://www.cnblogs.com/huangxiao2015/p/5667941.html
- [ ] https://www.cnblogs.com/huangxiao2015/p/5668140.html
- [ ] [Chapter 4: Regression and Hierarchical Models](http://www.est.uc3m.es/BayesUC3M/Master_course/Chapter4.pdf)
- [ ] https://docs.pymc.io/notebooks/GLM-hierarchical.html
- [ ] http://varianceexplained.org/r/hierarchical_bayes_baseball/
- [ ] http://idiom.ucsd.edu/~rlevy/pmsl_textbook/chapters/pmsl_8.pdf
- [ ] https://www.wikiwand.com/en/Bayesian_hierarchical_modeling

***
- [ ] [SAS/STAT Examples Bayesian Hierarchical Poisson Regression Model for Overdispersed Count Data](https://support.sas.com/rnd/app/stat/examples/BayesSalm/new_example/index.html)
- [ ] [Hierarchical Regression and Spatial models ](http://web2.uconn.edu/cyberinfra/module3/Downloads/Day%206%20-%20Hierarchical%20Bayes.pdf)
- [ ] [Short Course for ENAR 2009 - Sunday, March 15, 2009: Hierarchical Modeling and Analysis of Spatial-Temporal Data: Emphasis in Forestry, Ecology, and Environmental Sciences](http://blue.for.msu.edu/ENAR_09/SC/slides/BayesianLinearModels.pdf)
- [ ] https://sites.google.com/site/doingbayesiandataanalysis/
- [ ] [Bayesian methods for combining multiple Individual and Aggregate data Sources in observational studies.](http://www.bias-project.org.uk/WB2011Man/BHM-2011-slides.pdf)
- [ ] https://www.stat.berkeley.edu/~census/goldbug.pdf
- [ ] http://www.biostat.umn.edu/~ph7440/pubh7440/Lecture5.pdf
- [ ] [CS&SS/STAT 564: Bayesian Statistics for the Social Sciences, University of Washington, Spring 2018](https://uw-csss-564.github.io/2018/)
- [ ] https://jrnold.github.io/bayesian_notes/
- http://doingbayesiandataanalysis.blogspot.com/

## Optimal Learning

The Bayesian perspective casts a different interpretation
on the statistics we compute, which is particularly useful in the context of optimal learning.
In the frequentist perspective, we do not start with any knowledge about the system before we have collected any data.
By contrast, in the Bayesian perspective we assume that we begin with a prior distribution of belief about the unknown parameters.

Everyday decisions are made without the benefit of accurate information. Optimal Learning develops the needed principles for gathering information to make decisions, especially when collecting information is time-consuming and expensive.
Optimal learning addresses the problem of efficiently collecting information with which to make decisions.
Optimal learning is an issue primarily in applications where observations
or measurements are expensive.


It is possible to approach the learning problem using classical and familiar ideas from
optimization. The operations research community is very familiar with the use of gradients to
minimize or maximize functions. Dual variables in linear programs are a form of gradient, and
these are what guide the simplex algorithm. Gradients capture the value of an incremental
change in some input such as a price, fleet size or the size of buffers in a manufacturing system.
We can apply this same idea to learning.


There is [a list of optimal learning problems](https://people.orie.cornell.edu/pfrazier/info_collection_examples.pdf).

* http://yelp.github.io/MOE/
* [Peter I. Frazier@cornell](https://people.orie.cornell.edu/pfrazier/)
* [Optimal Learning book](http://optimallearning.princeton.edu/)
* [Optimal Learning Course](http://optimallearning.princeton.edu/#course)
* https://onlinelibrary.wiley.com/doi/book/10.1002/9781118309858
* [有没有依靠『小数据』学习的机器学习分支? - 覃含章的回答 - 知乎](https://www.zhihu.com/question/275605862/answer/381374728)

## Bayesian Optimization

Bayesian optimization has been successful at global optimization of expensive-to-evaluate multimodal objective functions. However, unlike most optimization methods, Bayesian optimization typically does not use derivative information.

As `response surface methods`, they date back to Box and Wilson in 1951.
Bayesian optimization usually uses `Gaussian process` regression.

![Bayesian Optimization](https://github.com/fmfn/BayesianOptimization/blob/master/examples/func.png)

![BayOpt](https://raw.githubusercontent.com/mlr-org/mlrMBO/master/docs/articles/helpers/animation-.gif)

****

* http://www.sigopt.com/
* https://jmhessel.github.io/Bayesian-Optimization/
* https://mlrmbo.mlr-org.com/index.html
* [Let’s Talk Bayesian Optimization](https://mlconf.com/blog/lets-talk-bayesian-optimization/)
* [Bayesian optimization tutorial slides and article (from INFORMS 2018)](https://people.orie.cornell.edu/pfrazier/Presentations/2018.11.INFORMS.tutorial.pdf)
* [Practical Bayesian optimization in the presence of outliers](http://proceedings.mlr.press/v84/martinez-cantin18a/martinez-cantin18a.pdf)
* [Bayesian Optimization with Gradients](https://arxiv.org/abs/1703.04389)
* https://www.iro.umontreal.ca/~bengioy/cifar/NCAP2014-summerschool/slides/Ryan_adams_140814_bayesopt_ncap.pdf
* [https://haikufactory.com/](https://haikufactory.com/)
* [Bayesian Optimization at Imperial London College](http://wp.doc.ic.ac.uk/sml/project/bayesian-optimization/)
* [PROBABILISTIC-NUMERICS.ORG](http://www.probabilistic-numerics.org/literature/index.html)
* [Bayesian optimization@http://krasserm.github.io/](http://krasserm.github.io/2018/03/21/bayesian-optimization/)
* [Introduction to Bayesian Optimization by Javier Gonz´alez](http://gpss.cc/gpmc17/slides/LancasterMasterclass_1.pdf)
* [A Python implementation of global optimization with gaussian processes](https://github.com/fmfn/BayesianOptimization)
* [Bayesian Optimization using Pyro](https://pyro.ai/examples/bo.html)
* [Taking the Human Out of the Loop:A Review of Bayesian Optimization](https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf)
* [Bayesian Optimization @modAL](https://modal-python.readthedocs.io/en/latest/content/examples/bayesian_optimization.html)
* [RoBO – a Robust Bayesian Optimization framework written in python. ](https://www.automl.org/automl/robo/)
* [Bayesian Optimization@Ployaxon](https://docs.polyaxon.com/references/polyaxon-optimization-engine/bayesian-optimization/)
* [BOAT: Building auto-tuners with structured Bayesian optimization](https://blog.acolyer.org/2017/05/18/boat-building-auto-tuners-with-structured-bayesian-optimization/)
* [The Intuitions Behind Bayesian Optimization](https://www.mindfoundry.ai/learning-hub/the-intuitions-behing-bayesian-optimization)


## Probabilistic Graphical Model

A graphical model or probabilistic graphical model (PGM) or structured probabilistic model is a probabilistic model for which a graph expresses the conditional dependence structure between random variables.
They are commonly used in probability theory, statistics — particularly Bayesian statistics — and machine learning. It is a marriage of graph theory and probability theory.
It is aimed to solve the causal inferences, which is based on principles rather than models.

+ [Probabilistic inference in graphical models by Jordan](http://mlg.eng.cam.ac.uk/zoubin/course04/hbtnn2e-I.pdf)
+ [Foundations of Graphical Models, Fall 2016, Columbia University](http://www.cs.columbia.edu/~blei/fogm/2016F/)
+ [CS 228: Probabilistic Graphical Models, Stanford / Computer Science / Winter 2017-2018](https://cs.stanford.edu/~ermon/cs228/index.html)
+ [Probabilistic Graphical Models 10-708, • Spring 2019 • Carnegie Mellon University](https://sailinglab.github.io/pgm-spring-2019/)
+ [Probabilistic Graphical Models](http://www.cs.cmu.edu/~epxing/Class/10708/lecture.html)

***
+ https://www.wikiwand.com/en/Graphical_model
+ https://blog.applied.ai/probabilistic-graphical-models-for-fraud-detection-part-1/
+ https://blog.applied.ai/probabilistic-graphical-models-for-fraud-detection-part-2/
+ https://blog.applied.ai/probabilistic-graphical-models-for-fraud-detection-part-3/
+ https://frnsys.com/ai_notes/foundations/probabilistic_graphical_models.html

### Bayesian Belief Network(BBN)

### Bayesian Network

Bayesian networks are a type of `Probabilistic Graphical Model` that can be used to build models from data and/or expert opinion.
They are also commonly referred to as `Bayes nets, Belief networks and sometimes Causal networks`.

[Bayesian Network (BN) is an intuitive, graphical representation of a joint probability distribution of a set of random variables with a possible mutual causal relationship.](https://research.csu.edu.au/research-support/data-methods-and-tools/statistics-workshops-and-tools/bayesian-network-workshops)

It is of wide application in many fields such as NLP, medical image analysis.
![BN](https://www.bayesserver.com/docs/images/analytics.png)

* [Bayesian Network Repository](http://www.bnlearn.com/bnrepository/)
* [Bayesian Networks by João Neto](http://www.di.fc.ul.pt/~jpn/r/bayesnets/bayesnets.html)
* [Additive Bayesian Network Modelling in R](http://r-bayesian-networks.org/)
* https://silo.ai/bayesian-networks-for-fast-troubleshooting/
* [Bayesian networks - an introduction](https://www.bayesserver.com/docs/introduction/bayesian-networks)
* [Bayesian Networks: Introductory Examples](http://www.bayesia.com/bayesian-networks-examples)
* [Bayesian Network – Brief Introduction, Characteristics & Examples](https://data-flair.training/blogs/bayesian-network-in-r/)
* [Bayesian Networks(Part I)](https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture22-bayesnet1.pdf)
* [Bayesian Networks(Part II)](https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture23-bayesnet2.pdf)
* [pomegranate is a Python package that implements fast and flexible probabilistic models.](https://pomegranate.readthedocs.io/en/latest/BayesianNetwork.html)
* http://robsonfernandes.net/bnviewer/
* https://www.hugin.com/

### Hidden Markov Models

- [Hidden Markov Model (HMM) Markov Processes and HMM](https://maelfabien.github.io/machinelearning/HMM_2/#decoding-with-viterbi-algorithm)
- https://web.stanford.edu/~jurafsky/slp3/A.pdf
- https://pomegranate.readthedocs.io/en/latest/index.html

## Probabilistic Programming

Probabilistic graphical models provide a formal lingua franca for modeling and a common target for efficient inference algorithms. Their introduction gave rise to an extensive body of work in machine learning, statistics, robotics, vision, biology, neuroscience, artificial intelligence (AI) and cognitive science. However, many of the most innovative and useful probabilistic models published by the AI, machine learning, and statistics community far outstrip the representational capacity of graphical models and associated inference techniques. Models are communicated using a mix of natural language, pseudo code, and mathematical formulae and solved using special purpose, one-off inference methods. Rather than precise specifications suitable for automatic inference, graphical models typically serve as coarse, high-level descriptions, eliding critical aspects such as fine-grained independence, abstraction and recursion.

PROBABILISTIC PROGRAMMING LANGUAGES aim to close this representational gap, unifying general purpose programming with probabilistic modeling; literally, users specify a probabilistic model in its entirety (e.g., by writing code that generates a sample from the joint distribution) and inference follows automatically given the specification. These languages provide the full power of modern programming languages for describing complex distributions, and can enable reuse of libraries of models, support interactive modeling and formal verification, and provide a much-needed abstraction barrier to foster generic, efficient inference in universal model classes.

[We believe that the probabilistic programming language approach within AI has the potential to fundamentally change the way we understand, design, build, test and deploy probabilistic systems. This approach has seen growing interest within AI over the last 10 years, yet the endeavor builds on over 40 years of work in range of diverse fields including mathematical logic, theoretical computer science, formal methods, programming languages, as well as machine learning, computational statistics, systems biology, probabilistic AI.](http://www.probabilistic-programming.org/wiki/Home)

|Graph Nets library|
|--------------------------|
| ![Graph net](https://raw.githubusercontent.com/Hulalazz/graph_nets/master/images/graph-nets-deepmind-shortest-path0.gif) |

* [PROBABILISTIC-PROGRAMMING.org ](http://www.probabilistic-programming.org/wiki/Home)
* [Programming Languages and Logics Fall 2018](https://www.cs.cornell.edu/courses/cs4110/2016fa/lectures/lecture33.html)
* [Graph Nets library](https://github.com/deepmind/graph_nets)
* [Hakaru a simply-typed probabilistic programming language](https://hakaru-dev.github.io/intro/probprog/)
* [PROBPROG 2018 -- The International Conference on Probabilistic Programming](https://probprog.cc/)
* [PyMC: Probabilistic Programming in Python](http://pymc-devs.github.io/pymc3/)
* [Stan is a state-of-the-art platform for statistical modeling and high-performance statistical computation.](http://mc-stan.org/)
* [Welcome to Pyro Examples and Tutorials!](http://pyro.ai/examples/)
* [A quick update: Edward, and some motivations](http://dustintran.com/blog/a-quick-update-edward-and-some-motivations)
* [Monadic probabilistic programming in Scala with Rainier](https://darrenjw.wordpress.com/2018/06/01/monadic-probabilistic-programming-in-scala-with-rainier/)
* [CSCI E-82A Probabilistic Programming and Artificial Intelligence Stephen Elston, PhDPrinciple Consultant, Quantia Analytics LLC](https://www.extension.harvard.edu/course-catalog/courses/probabilistic-programming-and-artificial-intelligence/15757)
* http://probcomp.csail.mit.edu/



+ https://www.stat.cmu.edu/~cshalizi/uADA/17/
+ https://www.cs.cmu.edu/~aarti/Class/10701/lecs.html
