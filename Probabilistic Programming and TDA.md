# Probabilistic Programming and Topological Data Analysis

Probabilistic Programming and Topological Data Analysis may be most theoretical subject in data science.
Probabilistic Programming is designed to solve the uncertain problems via Bayesian statistics.
If $\{A_i\}_{i=1}^{n}$ are disjoint events and ${\cup}_{i=1}^{n}A_i$  is all the results possible to happen, we should keep it in mind:

$$
P(A_i\mid B)=\frac{P(B\mid A_i)P(A_i)}{P(B)}, \\
P(A_i\mid B)\approx P(B\mid A_i)P(A_i).
$$

## Bayesian Learning

Bayesian learning can be regarded as the extension of Bayesian statistics. The core topic of Bayesian learning is thought as prior information to explain the uncertainty of parameters.
It is related with Bayesian statistics, computational statistics, probabilistic programming and machine learning.

* http://www.cnblogs.com/jesse123/p/7802258.html
* https://metacademy.org/roadmaps/rgrosse/bayesian_machine_learning
* http://blog.echen.me/2012/03/20/infinite-mixture-models-with-nonparametric-bayes-and-the-dirichlet-process/
* https://wso2.com/blog/research/part-one-introduction-to-bayesian-learning
* http://pages.cs.wisc.edu/~bsettles/cs540/lectures/15_bayesian_learning.pdf
* http://www.cogsys.wiai.uni-bamberg.de/teaching/ss05/ml/slides/cogsysII-9.pdf
* https://frnsys.com/ai_notes/machine_learning/bayesian_learning.html
* http://fastml.com/bayesian-machine-learning/

## Naive Bayes

Naive Bayes classifier takes the attribute conditional independence assumption.

* https://www.saedsayad.com/naive_bayesian.htm
* https://www.wikiwand.com/en/Naive_Bayes_classifier
* https://www.cnblogs.com/rhyswang/p/8326478.html

## Gaussian Naive Bayes

## Average One-Dependence Estimator (AODE)

* https://www.wikiwand.com/en/Averaged_one-dependence_estimators
* https://link.springer.com/article/10.1007%2Fs10994-005-4258-6

## Bayesian Belief Network(BBN)

## Bayesian Network

It is a probabilistic graphical model.
- https://www.wikiwand.com/en/Bayesian_network
- https://blog.csdn.net/gdp12315_gu/article/details/50002195

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
* https://people.orie.cornell.edu/pfrazier/
* http://optimallearning.princeton.edu/
* http://optimallearning.princeton.edu/#course
* https://onlinelibrary.wiley.com/doi/book/10.1002/9781118309858
* [有没有依靠『小数据』学习的机器学习分支? - 覃含章的回答 - 知乎](https://www.zhihu.com/question/275605862/answer/381374728)

## Bayesian Optimization

Bayesian optimization has been successful at global optimization of expensive-to-evaluate multimodal objective functions. However, unlike most optimization methods, Bayesian optimization typically does not use derivative information.

As `response surface methods`, they date back to Box and Wilson in 1951.

![Bayesian Optimization](https://github.com/fmfn/BayesianOptimization/blob/master/examples/func.png)
****

* http://www.sigopt.com/
* https://mlconf.com/blog/lets-talk-bayesian-optimization/
* https://pubsonline.informs.org/doi/10.1287/
* https://github.com/fmfn/BayesianOptimizationeduc.2018.0188
* http://proceedings.mlr.press/v84/martinez-cantin18a/martinez-cantin18a.pdf
* https://arxiv.org/abs/1703.04389
* https://www.iro.umontreal.ca/~bengioy/cifar/NCAP2014-summerschool/slides/Ryan_adams_140814_bayesopt_ncap.pdf

## Probabilistic Graphical Model

A graphical model or probabilistic graphical model (PGM) or structured probabilistic model is a probabilistic model for which a graph expresses the conditional dependence structure between random variables.
They are commonly used in probability theory, statistics — particularly Bayesian statistics — and machine learning. It is a marriage of graph theory and probability theory.
It is aimed to solve the causal inferences, which is based on principles rather than models.

+ http://mlg.eng.cam.ac.uk/zoubin/course04/hbtnn2e-I.pdf
+ https://www.wikiwand.com/en/Graphical_model
+ http://www.cs.columbia.edu/~blei/fogm/2016F/
+ https://cs.stanford.edu/~ermon/cs228/index.html
+ https://www.stat.cmu.edu/~cshalizi/uADA/17/
+ https://www.cs.cmu.edu/~aarti/Class/10701/lecs.html
+ http://www.cs.cmu.edu/~epxing/Class/10708/lecture.html
+ http://www.cs.princeton.edu/courses/archive/spring09/cos513/
+ https://www.cs.princeton.edu/courses/archive/fall11/cos597C/#prerequisites
+ https://www.wikiwand.com/en/Graphical_model
+ https://blog.applied.ai/probabilistic-graphical-models-for-fraud-detection-part-1/
+ https://blog.applied.ai/probabilistic-graphical-models-for-fraud-detection-part-2/
+ https://blog.applied.ai/probabilistic-graphical-models-for-fraud-detection-part-3/
+ https://darrenjw.wordpress.com/2018/06/01/monadic-probabilistic-programming-in-scala-with-rainier/


## Probabilistic Programming

Probabilistic graphical models provide a formal lingua franca for modeling and a common target for efficient inference algorithms. Their introduction gave rise to an extensive body of work in machine learning, statistics, robotics, vision, biology, neuroscience, artificial intelligence (AI) and cognitive science. However, many of the most innovative and useful probabilistic models published by the AI, machine learning, and statistics community far outstrip the representational capacity of graphical models and associated inference techniques. Models are communicated using a mix of natural language, pseudo code, and mathematical formulae and solved using special purpose, one-off inference methods. Rather than precise specifications suitable for automatic inference, graphical models typically serve as coarse, high-level descriptions, eliding critical aspects such as fine-grained independence, abstraction and recursion.

PROBABILISTIC PROGRAMMING LANGUAGES aim to close this representational gap, unifying general purpose programming with probabilistic modeling; literally, users specify a probabilistic model in its entirety (e.g., by writing code that generates a sample from the joint distribution) and inference follows automatically given the specification. These languages provide the full power of modern programming languages for describing complex distributions, and can enable reuse of libraries of models, support interactive modeling and formal verification, and provide a much-needed abstraction barrier to foster generic, efficient inference in universal model classes.

[We believe that the probabilistic programming language approach within AI has the potential to fundamentally change the way we understand, design, build, test and deploy probabilistic systems. This approach has seen growing interest within AI over the last 10 years, yet the endeavor builds on over 40 years of work in range of diverse fields including mathematical logic, theoretical computer science, formal methods, programming languages, as well as machine learning, computational statistics, systems biology, probabilistic AI.](http://www.probabilistic-programming.org/wiki/Home)

|Graph Nets library|
|--------------------------|
| ![Graph net](https://raw.githubusercontent.com/Hulalazz/graph_nets/master/images/graph-nets-deepmind-shortest-path0.gif) |

* http://www.probabilistic-programming.org/wiki/Home
* https://www.cs.cornell.edu/courses/cs4110/2016fa/lectures/lecture33.html
* https://hakaru-dev.github.io/intro/probprog/
* https://www.extension.harvard.edu/course-catalog/courses/probabilistic-programming-and-artificial-intelligence/15757
* http://probcomp.csail.mit.edu/
* https://probprog.cc/
* http://pymc-devs.github.io/pymc3/
* http://mc-stan.org/
* http://pyro.ai/examples/
* http://dustintran.com/blog/a-quick-update-edward-and-some-motivations
* https://www.wikiwand.com/en/Random_graph

## Hierarchical Bayesian Regression


- [ ] https://twiecki.io/
- [ ] http://www.est.uc3m.es/BayesUC3M/Master_course/Chapter4.pdf
- [ ] https://docs.pymc.io/notebooks/GLM-hierarchical.html
- [ ] http://varianceexplained.org/r/hierarchical_bayes_baseball/
- [ ] http://idiom.ucsd.edu/~rlevy/pmsl_textbook/chapters/pmsl_8.pdf
- [ ] https://twiecki.io/blog/2014/03/17/bayesian-glms-3/

## Topological Data Analysis

Topological data analysis(TDA) is potential to find better representation of the data.
TDA can visualize the high dimensional data and characterize the intrinsic invariants of the data.
It is close to computational geometry, manifold learning and computational topology.
It is one kind of descriptive representation learning.

As [The NIPS 2012 workshop on Algebraic Topology and Machine Learning](https://sites.google.com/site/nips2012topology/) puts:
> Topological methods and machine learning have long enjoyed fruitful interactions as evidenced by popular algorithms
> like ISOMAP, LLE and Laplacian Eigenmaps which have been borne out of studying point cloud data through the lens of geometry.
> More recently several researchers have been attempting to also understand the algebraic topological properties of data.
> Algebraic topology is a branch of mathematics which uses tools from abstract algebra to study and classify topological spaces.
> The machine learning community thus far has focused almost exclusively on clustering as the main tool for unsupervised data analysis.
> Clustering however only scratches the surface, and algebraic topological methods aim at extracting much richer topological information from data.


![TDA](https://pic4.zhimg.com/v2-bca1bc948527745f786d80427fd816f1_1200x500.jpg)
***

+ https://www.wikiwand.com/en/Topological_data_analysis
+ [TDA overview](https://perfectial.com/blog/topological-data-analysis-overview/)
+ [Studying the Shape of Data Using Topology](https://www.ias.edu/ideas/2013/lesnick-topological-data-analysis)
+ [Topological Data Analysis](https://dsweb.siam.org/The-Magazine/Article/topological-data-analysis-1)
+ [Why TDA works?](https://www.ayasdi.com/blog/bigdata/why-topological-data-analysis-works/)
+ [Topology-Based Active Learning](http://www.sci.utah.edu/publications/Mal2014a/UUSCI-2014-001.pdf)
+ https://sites.google.com/site/nips2012topology/
+ http://outlace.com/TDApart1.html
+ https://www.springer.com/cn/book/9783642150135
+ https://jsseely.github.io/notes/TDA/
+ [Extracting insights from the shape of complex data using topology](https://www.nature.com/articles/srep01236).
+ http://appliedtopology.org/
+ http://www.computational-geometry.org/
+ https://www.h-its.org/event/workshop-grg-2018/
+ https://sites.google.com/view/dragon-applied-topology
+ https://icerm.brown.edu/tripods/tri17-1-gtd/
+ https://www.ipam.ucla.edu/programs/long-programs/geometry-and-learning-from-data-in-3d-and-beyond/
+ http://kurlin.org/index.php#group
+ http://chomp.rutgers.edu/
+ http://chomp.rutgers.edu/Projects/Topological_Data_Analysis.html
+ https://www.jstage.jst.go.jp/article/tjsai/32/3/32_D-G72/_pdf
+ http://www.maths.usyd.edu.au/u/tillmann/cats2017/
+ http://www.maths.ox.ac.uk/groups/topology/
+ https://cs.nyu.edu/~yap/classes/
+ http://www.sci.utah.edu/~beiwang/
+ https://www.cs.montana.edu/tda/
+ https://www.csun.edu/~ctoth/Handbook/

### Topology

Topology focuses on the invariants under continuous mapping.
It pays more attention to the geometrical or discrete properties of the objects such as the number of circles or holes.
It is not distance-based.

> **Definition**: Let $X$ be a non-empty set. A set $\tau$ of subsets of $X$ is said to be a **topology** if
> * $X$ and the empty set $\emptyset$  belong to $\tau$;
> * the union of any number of sets in $\tau$ belongs to $\tau$;
> * the intersection of any two sets inn $\tau$ belongs to $\tau$.
> The pair $(X,\tau)$ is called a **topological space**.

As the definition shows the topology may be really not based on the definition of distance or measure. The set can be countable or discountable.e3

> **Definition**: Let $(X,\tau)$ be a topological space. Then the members of $\tau$ (the subsets of $X$) is said to be **open set**. If $X-S$ is open set, we call $S$ as **closed set**.


![klein bottle](https://www.ics.uci.edu/~eppstein/junkyard/nested-klein-bottles.jpg)

+ https://www.wikiwand.com/en/Topology
+ http://www.topologywithouttears.net/
+ https://www.ics.uci.edu/~eppstein/junkyard/topo.html
+ http://brickisland.net/DDGSpring2016/2016/01/20/reading-2-topology/
+ https://www.ayasdi.com/blog/artificial-intelligence/relationships-geometry-artificial-intelligence/

### TDA

Topological data analysis as one data processing method is selected topic for some students on computer science and applied mathematics.
It is not popular for the statisticians, where there is no estimation and test.

Topological data analysis (TDA) refers to statistical methods that find structure in data. As the
name suggests, these methods make use of topological ideas. Often, the term TDA is used narrowly
to describe a particular method called **persistent homology**.

TDA, which originates from mathematical topology, is a discipline that studies shape. It’s concerned with measuring the shape, by means applying math functions to data, and with representing it in forms of topological networks or combinatorial graphs.

There is another field that deals with the topological and geometric structure of data: computational geometry.
The main difference is that in TDA we treat the data as random points,
whereas in computational geometry the data are usually seen as fixed.

![tda](http://brickisland.net/DDGSpring2016/wp-content/uploads/2016/01/tda-300x208.png)

TDA can be applied to manifold estimation, nonlinear dimension reduction, mode estimation, ridge estimation and persistent homology.

***

+ https://datawarrior.wordpress.com/2015/08/03/tda-1-starting-the-journey-of-topological-data-analysis-tda/
+ https://www.annualreviews.org/doi/10.1146/annurev-statistics-031017-100045
+ http://brickisland.net/cs177fa12/
+ https://github.com/prokopevaleksey/TDAforCNN
+ https://github.com/ognis1205/spark-tda
+ https://github.com/stephenhky/PyTDA
+ https://people.clas.ufl.edu/peterbubenik/intro-to-tda/
+ https://www.math.colostate.edu//~adams/research/
+ http://brickisland.net/DDGSpring2016/2016/01/22/reading-3-topological-data-analysis/
+ https://www.math.upenn.edu/~ghrist/research.html
+ http://web.cse.ohio-state.edu/~dey.8/course/CTDA/CTDA.html
+ https://web.stanford.edu/group/bdl/blog/tda-cme-paper/
+ https://www.math.upenn.edu/~ghrist/research.html
+ http://www.stat.cmu.edu/topstat/presentations.html
+ http://www.stat.cmu.edu/research/statistical-theory-methodology/252
+ https://perfectial.com/blog/topological-data-analysis-overview/
+ http://www.sci.utah.edu/~beiwang/teaching/cs6170-spring-2017/
+ http://web.cse.ohio-state.edu/~dey.8/course/CTDA/CTDA.html
+ http://www.sci.utah.edu/~beiwang/tdaphenomics/tdaphenomics.html
+ https://drona.csa.iisc.ac.in/~vijayn/courses/
+ https://dsweb.siam.org/The-Magazine/Article/topological-data-analysis-1
+ https://www.annualreviews.org/doi/10.1146/annurev-statistics-031017-100045
+ https://www.turing.ac.uk/research/research-projects/scalable-topological-data-analysis

### Computational Topology

Computational topology is the mathematical theoretic foundation of topological data analysis. It is different from the deep neural network that origins from the engineering or the simulation to biological neural network.
Topological data analysis is principle-driven and application-inspired in some sense.

[CS 598: Computational Topology Spring 2013](http://jeffe.cs.illinois.edu/teaching/comptop/) put that
> Potential mathematical topics include the topology of ++cell complexes, topological graph theory, homotopy, covering spaces, simplicial homology, persistent homology, discrete Morse theory, discrete differential geometry, and normal surface theory. Potential computing topics include algorithms for computing topological invariants, graphics and geometry processing, mesh generation, curve and surface reconstruction, VLSI routing, motion planning, manifold learning, clustering, image processing, and combinatorial optimization++.

![bugs-topology](http://jeffe.cs.illinois.edu/teaching/comptop/Fig/codex-bugs.png)

+ https://datawarrior.wordpress.com/
+ http://graphics.stanford.edu/courses/cs468-09-fall/
+ https://graphics.stanford.edu/courses/cs468-02-fall/schedule.html
+ http://people.maths.ox.ac.uk/nanda/source/RSVWeb.pdf
+ https://jeremykun.com/tag/computational-topology/
+ http://jeffe.cs.illinois.edu/teaching/comptop/
+ http://www.enseignement.polytechnique.fr/informatique/INF556/
+ https://topology-tool-kit.github.io/
+ https://www.kth.se/student/kurser/kurs/SF2956?l=en
+ https://cs.nyu.edu/~yap/classes/modeling/06f/
+ https://courses.maths.ox.ac.uk/node/161
+ https://www2.cs.duke.edu/courses/fall06/cps296.1/
+ http://www.math.wsu.edu/faculty/bkrishna/CT_Math574_S12.html
