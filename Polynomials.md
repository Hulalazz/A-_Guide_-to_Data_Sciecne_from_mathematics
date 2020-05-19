# Decision Tree, Polynomials Regression and Probabilistic Graphical Model

## Introduction


Decision tree is popular and there are many free and open implementation of tree-based algorithms:
- [scikit-garden](https://scikit-garden.github.io/),
- [Treelite : model compiler for decision tree ensembles](https://github.com/dmlc/treelite),
- [xgboost](https://xgboost.ai/),
- [catBoost](https://catboost.ai/),
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/),
- [thunderGBM](https://github.com/Xtra-Computing/thundergbm).

The discussion [Who invented the decision tree?](https://stats.stackexchange.com/questions/257537/who-invented-the-decision-tree)
gives some clues of the history of decision tree.
Like deep learning, it has a long history.

Decision tree holds some attractive properties:

- hierarchical representation for us to understand its hidden structure;
- inherent adaptive computation; 
- logical expression implementation.

Observe that we train and prune a decision tree according to a specific training data set while we representation decision tree as a collection of logical expression.

[CART in statistics](http://washstat.org/presentations/20150604/loh_slides.pdf) is a historical review on decision tree.

- [Recursive Partitioning and Tree-based Methods](https://ideas.repec.org/p/zbw/caseps/200430.html)
- https://medicine.yale.edu/profile/heping_zhang/
- https://www.stat.berkeley.edu/~breiman/RandomForests/

##  Regression trees and recursive partitioning regression


[Now consider how to determine the structure of the decision tree. ](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
Even for a fixed number of nodes in the tree, the problem of determining the optimal structure 
(including choice of input variable for each split as well as the corresponding thresholds) 
to minimize the sum-of-squares error is usually computationally infeasible 
due to the combinatorially large number of possible solutions. 
Instead, a greedy optimization is generally done by starting with a single root node, 
corresponding to the whole input space, and then growing the tree by adding nodes one at a time. 
At each step there will be some number of candidate regions in input space that can be split,
corresponding to the addition of a pair of leaf nodes to the existing tree. 
For each of these, there is a choice of which of the $D$ input variables to split, 
as well as the value of the threshold. The joint optimization of the choice of region to split, 
and the choice of input variable and threshold, can be done efficiently by exhaustive search noting that, 
for a given choice of split variable and threshold, the optimal choice of predictive variable is given by the local average of the data, as noted earlier. 
This is repeated for all possible choices of variable to be split, 
and the one that gives the smallest residual sum-of-squares error is retained.

Note that $T(x)=\sum_{\ell}^{|T|} c_{\ell}p(x\mid \ell)$
where $p(x\mid \ell)$ is the probability that the sample reach the leaf $\ell$.
The classic decision tree just selects an unique leaf, i.e., only one $p(x\mid \ell)$ is equal to 1 and others are equal to 0.
Now we will consider from the probabilistic perspective.

Note that  $P(x, y)=P(y)P(x\mid y)$, so some probabilistic decision tree extends the $c_{\ell}$ to be a probability distribution in order to overcome the limitation of classic decision tree.



The decision tree is really a discriminant model $p(y\mid x)=\sum_{\ell}^{|T|} p(y, \ell)p(x\mid \ell)$.
And this leads to the probabilistic decision tree.

Note that the tree structure is a special type of directed
acyclic graph structure.
Here we focus on the connection between the decision trees and probabilistic graphical models.


Regression trees and soft decision trees are extensions of
the decision tree induction technique, predicting a numerical output, rather than a discrete class.

For simplicity, we suppose that all variables are numerical and the decision tree can be expressed in a polynomials form:
$$T(x)=\sum_{\ell}^{|T|} c_{\ell}\prod_{i\in P(\ell)}\sigma(\pm(x_i)-\tau_i)$$
where the index $\ell$ is referred to the leaf node;
the constant $|T|$ is the total number of leaf nodes;
the set $P(\ell)$ is the node in the path from the root to the leaf node $\ell$;
the function $\sigma(\cdot)$ is the [unit step function];
the input $x=(x_1, x_2,\cdots, x_d)\in\mathbb{R}^d$ and $x_i\in\mathbb{R}$ is the component of the input $x$;
the constant $c_{\ell}$ is the constant associated with the leaf node.

The product of step functions is the [indicator functions][IndicatorFunctions].
For example, $T(x)=a\sigma(x_1 - 1) + b\sigma(1 - x_1)$, is the decision tree with only the root node and two leaf nodes where $x-1>1$ it returns $a$; otherwise, it return $b$.


For categorical variables $z=(z_1, z_1,\dots, z_n)$, each component $z_i$ for $i=1,2,\cdots, n$ takes finite categorical values.
Embedding techniques will encode these categorical variables into digital codes. In another word, they map each categorical values into unique numerical feature.
For example, the hashing function  can map every string into unique codes.
For categorical variable, the decision trees will perform the `equalsto(==)` test, i.e., if the variable $x_i$ is equal to a given value, it returns 1; otherwise it returns 0.
Based on this observation, we can apply [Interpolation Methods](http://www.gisresources.com/types-interpolation-methods_3/).
Suppose the categorical variable $z_1$ embedded in $\{a_1, a_2, a_3,\cdots, a_n\}$, the [Lagrange Interpolation Formula](https://byjus.com/lagrange-interpolation-formula/) gives $\sum_{i=1}^{n}\prod_{j\not=i}\frac{(z_1-a_j)}{(a_i -a_j)}$, where $\prod_{j\not=i}\frac{(z_1-a_j)}{(a_i -a_j)}$ equals to 1 if $z_1=a_i$ otherwise it is equal to 0.


In [MARS], the recursive partitioning regression, as binary regression tree,  is  viewed in a more conventional light as a stepwise regression procedure:
$$f_M(x)=\sum_{m=1}^{M}a_m\prod_{k=1}^{K_m}H[s_{km}(x_{v(k, m) - t_{km}})]$$
where $H(\cdot)$ is the [unit step function].
The quantity $K_m$, is the number of splits that gave rise to basis function.
The quantities $s_{km}$ take on values k1and indicate the (right/left) sense of the associated step function.
The $v(k, m )$ label the predictor variables and 
the $t_{km}$, represent values on the corresponding variables.
The internal nodes of the binary tree represent the step functions and the terminal nodes represent the final basis functions.

_____

[Delta Function](https://mathworld.wolfram.com/DeltaFunction.html) is the derivative of step function, an example of [generalized function](https://mathworld.wolfram.com/GeneralizedFunction.html);
the unit step function is the derivative of the [Ramp Function](https://mathworld.wolfram.com/RampFunction.html)  defined by
$$R(x)=x\sigma(x)=\max(0,x)=ReLU(x)=(x)^{+}.$$

[Kronecker Delta](https://mathworld.wolfram.com/KroneckerDelta.html) is defined as 
$$\delta_{ij}=\begin{cases}1, \text{if $i=j$}\\
0, \text{otherwise}\end{cases}.$$

The [indicator functions][IndicatorFunctions] is defined as
$$\mathbb{1}_{ \{condition\} }=\begin{cases}1, &\text{if condition is hold }\\
0, &\text{otherwise}.\end{cases}$$

Given a subset $A$ of a larger set, the `characteristic function` $\chi_A$, sometimes also called the **indicator function**, is the function defined to be identically one on $A$, and is zero elsewhere[CharacteristicFunction]. 

A simple function is a finite sum $sum_(i)a_i \chi_(A_i)$, where the functions $\chi_(A_i)$ are characteristic functions on a set $A$. Another description of a simple function is a function that takes on finitely many values in its range[SimpleFunction].


****

The fundamental limitation of decision tree includes: 
1. its lack of `continuity`, 
2. lack `smooth [decision boundary]` or axe-aligned splits,
3. inability to provide good approximations to certain classes of simple often-occurring functions，
4. highly instability with respect to minor perturbations in the training data.

The decision tree is simple function essentially.
Based on above observation, decision tree is considered  to project the raw data into distinct decision region in an adaptive approach 
and select the mode of within-node samples ground truth as their `label`.


We can replace the unit step function with truncated power series as in [MARS] to tackle the discontinuity to invent continuos models with continuous derivatives.
[MARS] is an extension of recursive partitioning regression:
$$f_M(x)=\sum_{m=1}^{M}a_m\prod_{k=1}^{K_m}[s_{km}(x_{v(k, m) - t_{km}})]^{+}.$$
Note that the ramp function $(x)^{+}=\max(0,x)=x\cdot H(x)$ so we can re-express the MARS as 
$$f_M(x)=\sum_{m=1}^{M}a_m(x)\prod_{k=1}^{K_m}H[s_{km}(x_{v(k, m) - t_{km}})]$$
where $a_m(x)=\prod_{k=1}^{K_m}(x_{v(k, m) - t_{km}})$.
[MARS] can approximate the polynomials directly. 
For example, the identity function is $f_M(x)=\sum_{m=1}^{M}x\prod_{k=1}^{K_m}[s_{km}(x_{v(k, m) - t_{km}})]^{+}$.

****

Note that $B_m=\prod_{k=1}^{K_m}H[s_{km}(x_{v(k, m) - t_{km}})]\in \{0,1\}$, it is not continuous and binary.
If $B_m=1$, the sample $x$ will reach the leaf node $m$.
Otherwise the sample will not reach the leaf node.
Each test in the non-terminal nodes are dichotomous and the result is also dichotomous.

There is no transformation of raw input. The decision tree is just to guide the input to a proper terminal nodes associated with some specific tests.
It is difficult to imply some properties of the input according to the test results.
To classify a new instance, one starts at the top node and applies sequentially the dichotomous tests encountered to select the
appropriate successor. 
Finally, a `unique path` is followed, a `unique terminal` node is reached and the output estimation stored there is assigned to this instance.

## Soft decision trees


A  natural modification is to make the step function continuous such as [MARS].
However, [MARS] is still of axis-aligned type basis expansion.


[Competitive Neural Trees for Pattern Classification](https://www.ais.uni-bonn.de/behnke/papers/tnn98.pdf) contains m-ary nodes and
grows during learning by using inheritance to initialize new
nodes.

The [fuzzy decision tree] introduce membership function to replace the unit step function.
In another word, the basis function is $B_m=\prod_{k=1}^{K_m}\sigma[s_{km}(x_{v(k, m) - t_{km}})]\in [0,1]$ where $\sigma(x)\in[0,1]$.
And the given instance reaches multiple terminal nodes 
and the output estimations given by all these terminal nodes are aggregated 
in order to obtain the final estimated membership degree to the target class.


In [soft decision trees](http://www.cs.cornell.edu/~oirsoy/softtree.html), 
the unit step function $\sigma(\cdot)$ is replaced by the smooth sigmoid or logistic function $\sigma(x)=\frac{1}{1+\exp(-x)}$.
And the samples may reach multiple terminal node with some probability.


[Probabilistic Boosting-Tree](http://vision.cse.psu.edu/people/chenpingY/paper/tu_z_pbt.pdf) 
is the probabilistic boosting-tree that automatically constructs a tree in which each node combines a number of weak classifiers (evidence, knowledge) into a strong classifier (a conditional posterior probability).


To find smooth [decision boundary] of decision tree inspired methods, another technique is to transform the data at each non-terminal node.


[Adaptive Neural Trees](https://github.com/rtanno21609/AdaptiveNeuralTrees) gain the complementary benefits of neural networks and decision trees.


******

Many multiple adaptive regression methods are specializations of a general multivariate
regression algorithm that builds `hierarchical models` using a set of basis
functions and `stepwise selection`:
$$f_M(x,\theta)=\sum_{m=1}^{M}\theta_m B_m(x)$$
for $x\in\mathbb{R}^n$.

Let us compare  hierarchical models including
decision tree, multiple adaptive regression spline and [Recursive partitioning regression](https://projecteuclid.org/download/pdf_1/euclid.aos/1176347963).


Decision Tree| MARS| Multivariate adaptive polynomial synthesis (MAPS)
-------------|---------|-------
Discontinuity|Smoothness|----
Unit step function|ReLU|

## Decision boundary

The [MARS] overcomes the (1) and (3);
The [fuzzy decision tree] overcomes the last one.
The above methods are only for regression tasks.
From now we turn to the classification tasks.

As shown in [decision boundary], 

> In general, a pattern classifier carves up (or tesselates or partitions) the feature space into volumes called `decision regions`. 
All feature vectors in a decision region are assigned to the same category. The decision regions are often simply connected, but they can be multiply connected as well, consisting of two or more non-touching regions.

> The decision regions are separated by surfaces called the `decision boundaries`. These separating surfaces represent points where there are ties between two or more categories.

[The `Manifold Hypothesis` states that real-world high-dimensional data lie on low-dimensional manifolds embedded within the high-dimensional space](https://deepai.org/machine-learning-glossary-and-terms/manifold-hypothesis), namely natural high dimensional data concentrates close to a low-dimensional manifold.
[Cluster hypothesis in information retrieval is to say that documents in the same cluster behave similarly with respect to relevance to information needs.](https://nlp.stanford.edu/IR-book/html/htmledition/clustering-in-information-retrieval-1.html)
[The key idea behind the `unsupervised learning` of disentangled representations](https://arxiv.org/pdf/1811.12359.pdf) is that real-world data is generated by a few explanatory factors of variation which can be recovered by unsupervised learning algorithms.

- [The Cluster Hypothesis: A Visual/Statistical Analysis](https://ie.technion.ac.il/~kurland/clustHypothesisTutorial.pdf)
- https://ie.technion.ac.il/~kurland/clustHypothesisTutorial.pdf
- https://muse.jhu.edu/article/20058/pdf
- http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
- [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://arxiv.org/pdf/1811.12359.pdf)

The success of decision trees is a strong evidence that both hypotheses are true.
[The goal of deep learning is to learn the manifold structure in data and the probability distribution associated with the manifold.](https://deepai.org/publication/geometric-understanding-of-deep-learning)
From this geometric perspective, the unsupervised  learning is to find certain properties of the manifold structure in data, 
such as the [Intrinsic Dimension](https://eng.uber.com/intrinsic-dimension/).
What is the goal of regression tree from this geometric perspective?
Is the goal of classification tree to `probability distribution` associated with the manifold  from this geometric perspective?

- http://www.mit.edu/people/mitter/publications/C50_sample_complexity.pdf
- http://www.mit.edu/~mitter/publications/121_Testing_Manifold.pdf
- [Geometric Understanding of Deep Learning](https://arxiv.org/abs/1805.10451v1)
- http://web.mit.edu/cocosci/Papers/man_nips.pdf
- [Topology and Data](https://www.ams.org/journals/bull/2009-46-02/S0273-0979-09-01249-X/S0273-0979-09-01249-X.pdf)
- [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/pdf/1802.03426.pdf)
- https://www.krishnaswamylab.org/
- http://www.vision.jhu.edu/assets/ElhamifarNIPS11.pdf
- http://www.cs.virginia.edu/~jdl/bib/manifolds/souvenir05.pdf
- [SemiBoost: Boosting for Semi-supervised Learning](http://dataclustering.cse.msu.edu/papers/semiboost_toappear.pdf)
- https://alexgkendall.com/media/papers/alex_kendall_phd_thesis_compressed.pdf


In some sense, a good classifier can describe the decision boundary well.
And classification is connected with the manifold approximation. 
In practice, the problem is that the attributes are of diverse types.
Although categorical attributes can be embedded into real space, it is always different to deal with categorical values and numerical values. 

The decision boundary of decision trees is `axis-aligned` and therefor not smooth.
In fact, the product $\prod_{k=1}^{K_m}H[s_{km}(x_{v(k, m) - t_{km}})]$ is the characteristic function of axis-aligned set $S=\{x \mid s_{km}(x_{v(k, m) - t_{km}})>0,k=1,2,\cdots, K-m\}$.
This is because of the relation of ramp functions and unit step function.
The decision boundary is depicted via the split values $t_{km}$.

The axis-aligned decision boundary restricts the expressivity of decision tree , which can be solved by data preprocessing and feature transformation such as kernel technique.
Or we can modify the unit step function.


## Template matching

[Template Matching] is a natural approach to pattern classification. This can be done in a couple of equivalent ways:

* Count the number of agreements. Pick the class that has the maximum number of agreements. This is a maximum correlation approach.
* Count the number of disagreements. Pick the class with the minimum number of disagreements. This is a minimum error approach.

Template matching works well when the variations within a class are due to "additive noise." Clearly, it works for this example because there are no other distortions of the characters -- translation, rotation, shearing, warping, expansion, contraction or occlusion. 
It will not work on all problems, but when it is appropriate it is very effective. It can also be generalized in useful ways.

Template matching can easily be expressed mathematically. 
Let $x$ be the feature vector for the unknown input, and let $m_1, m_2, \cdots, m_c$ be templates (i.e., perfect, noise-free feature vectors) for the $c$ classes. 
Then the error in matching $x$ against $m_k$ is given by
 $$\| x - m_k \|.$$
Here $\| u \|$ is called the norm of the vector $u$. 
A minimum-error classifier computes $\| x - m_k \|$ for $k = 1$ to $c$ and chooses the class for which this error is minimum. 
Since $\| x - m_k \|$ is also the distance from $x$ to $m_k$, we call this a `minimum-distance classifier`. 
Clearly, a template matching system is a minimum-distance classifier.


If a simple minimum-distance classifier is satisfactory, there is no reason to use anything more complicated. 
However, it frequently happens that such a classifier makes too many errors. There are several possible reasons for this:

1. The features may be inadequate to distinguish the different classes
2. The features may be highly correlated
3. The decision boundary may have to be curved
4. There may be distinct subclasses in the data
5. The feature space may simply be too complex


`Linear Discriminant Analysis (LDA)` is most commonly used as dimensionality reduction technique in the pre-processing step for pattern-classification and machine learning applications. 
The goal is to project a dataset onto a lower-dimensional space with good class-separability in order avoid overfitting (“curse of dimensionality”) and also reduce computational costs.

`LDA` assumes that the observations within each class are drawn from a `multivariate Gaussian distribution` and the covariance of the predictor variables are common across all $k$ levels of the response variable $Y$.

Like LDA, the `QDA` classifier assumes that the observations from each class of Y are drawn from a Gaussian distribution. However, unlike LDA, QDA assumes that each class has its own covariance matrix. 

- https://sebastianraschka.com/Articles/2014_python_lda.html
- http://washstat.org/presentations/20150604/loh_slides.pdf
- http://uc-r.github.io/discriminant_analysis
- https://pubmed.ncbi.nlm.nih.gov/12365038/


## Decision manifold

[The boundaries can often be approximated by linear ones connected
by a low-dimensional nonlinear manifold.](http://www.ifs.tuwien.ac.at/~lidy/pub/poe_lidy_wsom07.pdf) 
while it is difficult to determine the number of linear approximations and their `convergence region'.   
For example, we can use the multiple linear combination of features to substitute the univariate function: $\max\{0, \left<w,x\right>\}$.
In this case the [decision boundary] is piece-wise linear.



We will generalize the decision boundary into [decision manifold].

> As decision tree consists of several leaves where k-nearest neighbor
 classifier perform the actual classification task by a voting scheme,
model selection is related to tree technique.

The goal of [decision manifold] is to approximate the decision boundaries rather than the data points  using similar techniques like local approximation as manifold learning.

This algorithm is based upon Self-Organizing Maps and linear classification.
Mathematically, a decision surface is a hyper-surface (i.e. of
dimension $D − 1$, and of arbitrary shape). 
The decision boundary is assumed to consist of a finite number of topological manifolds, thus the possibly non-contiguous hypersurface can be decomposed into contiguous subsets.

The `Decision Manifold` consists of $M$ local classifiers, each specified by
a pair of a representative $c_j$ 
and a classification vector $v_j$ that is orthogonal to the decision hyperplane of the local classifier.
The goal of the training algorithm is to put the representatives $c_j$ in the adequate positions 
and to let $v_j$ point in the correct direction for classification.

The representatives $c_j$ are not true prototype vectors placed where data density is high, 
but rather in positions where there is a transition between two neighboring areas of different classes.


1. The data samples are assigned to the closest local classifier representative
as the [Template Matching]. And compute  data centroid $n_j$ of the partition $j$. 
2. Once all the samples have been assigned, it is to test if samples of both classes are present at certain partition. 
   - *  If true, a linear classifier is trained for the partition $j$ to obtain a separating hyperplane $\tilde{w}_j$.
   - *  compute a new preliminary representative and store the information for classification in the normalized vector:
   $$c_j^{\prime}=\pi_{\tilde{w}_j}(n_J), v_j^{\prime}=\frac{\tilde{w}_j}{\|\tilde{w}_j\|}$$
   where $\pi_{\tilde{w}_j}(\cdot)$ denotes orthogonal projection onto the hyperplane specified by $\tilde{w}_j$
3.  Updated local classifiers subjected to the neighborhood kernel smoothing and weighted according to topological distance.

In its most simple form, classification is performed by assigning a data sample to its closest linear classifier in the same way as during training, and
then classifying it according to its position relative to the hyperplane, and is defined as
$$\tilde{y}=\operatorname{sgn}(\left<x-c_{I(x)}, v_{I(x)}\right>)$$
where $I(x)=\arg\min_{j}\|x-c_j\|$.

we recapitulate the properties of this method:
*  The algorithm is a stochastic supervised learning method for two-class problems.
*  Computation is very efficient.
*  The topology induced by adjacency matrix $A$ defines the ordering and alignment of the local classifiers; it is also exploited for optimizing classification accuracy by a weighted voting scheme.
*  As the topology of the decision hyper-surface is unknown, we apply a heuristic model selection that trains several classifiers with different topologies.
*  The classifier performs well in case of multiple noncontiguous decision surfaces and non-linear classification problems such as XOR.

It is not a recursive partitioning method 
because the number of local classifiers are set as a hyperparameter determined before the training.

- http://www.ifs.tuwien.ac.at/~andi/publications/pdf/dit_ijcnn05.pdf
- http://www.ifs.tuwien.ac.at/~poelzlbauer/publications/Poe05IJCNN.pdf
- https://www.sba-research.org/team/andreas-rauber/

## Conditional framework

Decision trees apply the ancient wisdom "因地制宜""因材施教" to learn from data.
The problem is that we should know the the training data set well and we should evaluate how well our model fit the training data set correctly.
Usually the statistical prediction  methods are discriminant  or generative.
Although the decision trees are unified in conditional framework such as 
[Torsten Hothorn, Kurt Hornik & Achim Zeileis](https://med.stanford.edu/content/dam/sm/dbds/documents/biostats-workshop/CTREE-2006-.pdf), 
it is also possible to apply tree related methods to estimate the density of the data.


- [Naive Bayes, Discriminant Analysis and Generative Methods](https://liyanxu.blog/2018/10/28/naive-bayes-discriminant-analysis/)
- [Comparison of Discriminant Analysis and Decision Trees for the Detection of Subclinical Keratoconus](https://pubmed.ncbi.nlm.nih.gov/28810283/)


It is natural to extend the decision boundary to piece-wise polynomial.
[B-spline] and [Bézier Curve] can be used to describe the decision boundary.
Observe that $xH(x)=\max\{0,x\}$, $\max(0, \operatorname{sgn}x^2)=x^2H(x)$ 
where $H(x)$ is the unit step function while $\max(0, x^2+x^3)\not= (x^2+x^3)H(x)$.
The choice of activation function is based on the belief of decision boundaries.


- [http://blog.pluskid.org/?p=533](http://blog.pluskid.org/?p=533)
- [Explaining The Success Of Nearest Neighbor Methods In Prediction](https://devavrat.mit.edu/publications/explaining-the-success-of-nearest-neighbor-methods-in-prediction/)
- [Think Globally, Fit Locally: Unsupervised Learning of Low Dimensional Manifolds](https://cs.nyu.edu/~roweis/papers/llejmlr.pdf)


The ideal model is to learn the manifold structure of data automatically.


----


- https://eeecon.uibk.ac.at/~zeileis/blog/
- https://arxiv.org/pdf/1909.00900.pdf
- [The power of unbiased recursive partitioning](https://eeecon.uibk.ac.at/~zeileis/news/power_partitioning/)
- [Unbiased Recursive Partitioning I: A Non-parametric Conditional Inference Framework](https://eeecon.uibk.ac.at/~zeileis/papers/StatComp-2005a.pdf)
- http://www.saedsayad.com/docs/gbm2.pdf
- http://www.saedsayad.com/author.htm
- https://epub.ub.uni-muenchen.de/


*****

[We carry the integration of parametric models into trees](https://eeecon.uibk.ac.at/~zeileis/papers/Zeileis+Hothorn+Hornik-2008.pdf)
one step further and provide a rigorous theoretical foundation by introducing a new unified framework that embeds
recursive partitioning into statistical model estimation and variable selection.

Generalized linear mixed-effects model tree (GLMM tree) algorithm  allows for the detection of treatment-subgroup interactions 
while accounting for the clustered structure of a dataset:

$$\sum_{i}g_i\mathbb{I}(x\in S_i)$$

where $g_i$ is a generalized linear model for the data set $S_i$.

- [Model-based Recursive Partitioning](https://eeecon.uibk.ac.at/~zeileis/papers/Zeileis+Hothorn+Hornik-2008.pdf)
- [MPT trees published in BRM](https://eeecon.uibk.ac.at/~zeileis/news/mpttree/)
- [Spatial lag model trees](https://eeecon.uibk.ac.at/~zeileis/news/lagsarlmtree/)
- [Generalised Linear Model Trees with Global Additive Effects](https://eeecon.uibk.ac.at/~zeileis/papers/Seibold+Hothorn+Zeileis-2019.pdf)
- [Partially additive (generalized) linear model trees](https://eeecon.uibk.ac.at/~zeileis/news/palmtree/)
- [party: A Laboratory for Recursive Partytioning](https://cran.r-project.org/web/packages/party/vignettes/party.pdf)
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1906005
- [Distributional regression forests on arXiv](https://eeecon.uibk.ac.at/~zeileis/news/distforest/)
- [Network model trees](https://eeecon.uibk.ac.at/~zeileis/news/networktree/)
- [Network Model Trees project](https://osf.io/ykq2a/)
- [Generalized structured additive regression based on Bayesian P-splines](https://epub.ub.uni-muenchen.de/1702/1/paper_321.pdf)
- [Glm tree](https://eeecon.uibk.ac.at/~zeileis/news/glmertree/)
- https://eeecon.uibk.ac.at/~zeileis/news/bamlss/

****

In representation learning it is often assumed that real-world observations $x$ (e.g., images or videos) are generated by a two-step generative process. 
First, a multivariate latent random variable $z$ is sampled from a distribution $P(z)$. 
Intuitively, $z$ corresponds to semantically meaningful factors of variation of the observations (e.g., content + position of
objects in an image). 
Then, in a second step, the observation $x$ is sampled from the conditional distribution $P(x\mid z)$.

The key idea behind this model is that the high-dimensional data
$x$ can be explained by the substantially lower dimensional and semantically meaningful latent variable $z$ which is mapped to the higher-dimensional space of observations $x$. 
Informally, the goal of representation learning is to find useful transformations $r(x)$ of $x$ that “make it easier to extract useful information when building classifiers or other predictors”.



## Model Interpretation and Mixing

An instance starts a path from the root node all the way down to a leaf node according to its real feature value in a single decision tree. 
All the instances in the training data will fall into several nodes 
and different nodes have quite different label distributions of the instances in them in the additive tree models. 
Every step after passing a node, the probability of being the positive class changes with the label distributions. 
All the features along the path contribute to the final prediction of a single tree.


There is a trend to find a balance between the accuracy and explanation by combining decision trees and deep neural networks.

The main difference from the soft decision trees are that they are designed to learn  the representation of the input samples.

[Instead of combining neural networks and decision trees explicitly, several
works borrow ideas from decision trees for neural networks and vice versa.](https://arxiv.org/pdf/2004.00221.pdf)
These models are so-called neural decision tree in this context.

Generally speaking, tree means that the instances would take different processing methods according to their  feature values. 
One drawback of decision tree is the lack of `feature transformation`.


### Neural-Backed Decision Trees

`Neural-Backed Decision Trees` (NBDT) are the first to combine interpretability of a decision tree with accuracy of a neural network.

Training an NBDT occurs in 2 phases: First, construct the hierarchy for the decision tree. 
Second, train the neural network with a special loss term. 
To run inference, pass the sample through the neural network backbone. 
Finally, run the final fully-connected layer as a sequence of decision rules.

1. Construct a hierarchy for the decision tree, called the Induced Hierarchy.
2. This hierarchy yields a particular loss function, which we call the Tree Supervision Loss.
3. Start inference by passing the sample through the neural network backbone. The backbone is all neural network layers before the final fully-connected layer.
4. Finish inference by running the final fully-connected layer as a sequence of decision rules, which we call `Embedded Decision Rules`. These decisions culminate in the final prediction.

- http://nbdt.alvinwan.com/
- https://arxiv.org/pdf/2004.00221.pdf
- [Making decision trees competitive with neural networks on CIFAR10, CIFAR100, TinyImagenet200, Imagenet](https://github.com/alvinwan/neural-backed-decision-trees)
- [Making Decision Trees Accurate Again: Explaining What Explainable AI Did Not](https://bair.berkeley.edu/blog/2020/04/23/decisions/)

### Deep Neural Decision Trees


Deep Neural Decision Trees (DNDT) are tree models which are realised by neural networks.
A DNDT is intrinsically interpretable,
as it is a tree. Yet as it is also a neural network (NN), 
it can be easily implemented in NN toolkits, and trained with gradient descent rather than greedy splitting.

A soft binning function is used to make the split decisions in DNDT.
Given our binning function, the key idea is to construct the
decision tree via Kronecker product.

- https://github.com/wOOL/DNDT
- https://arxiv.org/abs/1806.06988
- https://github.com/Nicholasli1995/VisualizingNDF
- https://sites.google.com/view/whi2018/home
- http://www.deeplearningpatterns.com/doku.php?id=deep_neural_decision_tree

### Adaptive Neural Trees

Deep neural networks and decision trees operate on largely separate paradigms; typically, the former performs representation learning with pre-specified architectures, while the latter is characterised by learning hierarchies over pre-specified features with data-driven architectures. 
[We unite the two via adaptive neural trees (ANTs)](https://www.microsoft.com/en-us/research/publication/adaptive-neural-trees/), a model that incorporates representation learning into edges, routing functions and leaf nodes of a decision tree, along with a backpropagation-based training algorithm that adaptively grows the architecture from primitive modules (eg, convolutional layers). 


- https://topos-theory.github.io/deep-neural-decision-forests/
- http://www.robots.ox.ac.uk/~tvg/publications/talks/deepNeuralDecisionForests.pdf
- https://www.microsoft.com/en-us/research/publication/adaptive-neural-trees/
- [Deep Neural Decision Forests](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf)
- https://github.com/Microsoft/EdgeML/wiki/Bonsai
- https://www.microsoft.com/en-us/research/publication/resource-efficient-machine-learning-2-kb-ram-internet-things/


### DeepGBM

`DeepGBM`  integrates the advantages of the both NN and GBDT by using two corresponding NN components: 
(1) CatNN, focusing on handling sparse categorical features. 
(2) GBDT2NN, focusing on dense numerical features with distilled knowledge from GBDT. 

<img src="https://www.msra.cn/wp-content/uploads/2019/08/kdd-2019-2.png" width="40%" />
<img src="https://www.msra.cn/wp-content/uploads/2019/08/kdd-2019-3.png" width="60%"/> 

- [Implementation for the paper "DeepGBM: A Deep Learning Framework Distilled by GBDT for Online Prediction Tasks"](https://github.com/motefly/DeepGBM)
- https://tracholar.github.io/wiki/machine-learning/deep-gbm.html
- http://www.arvinzyy.cn/2019/08/13/kdd-2019-paper-notes/
- [DeepGBM: A Deep Learning Framework Distilled by GBDT for Online Prediction Tasks](https://dl.acm.org/doi/pdf/10.1145/3292500.3330858)
- https://www.msra.cn/zh-cn/news/features/kdd-2019
- [Unpack Local Model Interpretation for GBDT](https://arxiv.org/pdf/2004.01358v1.pdf)
- [Optimal Action Extraction for Random Forests and Boosted Trees](https://www.cse.wustl.edu/~ychen/public/OAE.pdf)
- [Interpreting random forest classification models using a feature contribution method](https://arxiv.org/pdf/1312.1121.pdf)
- https://github.com/benedekrozemberczki/awesome-gradient-boosting-papers

### Deep Forest

Deep neural network models usually make an effort to learn a new feature space and employ a multi-label classifier on the top.
`Deep Forest` is an alternative of deep learning. 

<img src="https://academic.oup.com/view-large/figure/165884799/nwy108fig2.jpg" with="60%"/>

`gcForest`  is the first deep learning model
which is NOT based on NNs
and which does NOT rely on BP.

- http://www.lamda.nju.edu.cn/code_gcForest.ashx
- [Talk: Deep Forest Towards An Alternative to Deep Neural Networks](http://tdam-bjkl.bjtu.edu.cn/Slides/Zhouzhihua-MLA2017.pdf)

Different multi-label forests are ensembled in each layer of MLDF. 
From layer $t$, we can obtain the representation $H_t$. 
The part of measure-aware feature reuse will receive the representation $H_t$
and update it by reusing the representation $G_{t−1}$ learned in the layer $t−1$ under the guidance of the performance of different measures. 
Then the new representation $G_t$ will be concatenated together with the raw input features (the red one) and goes into the next layer.

The critical idea of measure-aware feature reuse is to partially reuse the better representation in the previous layer on the current layer 
if the confidence on the current layer is lower than a threshold determined in training. 
Therefore, the challenge lies in defining the confidence of specific multi-label measures on demand. 


- https://explainai.net/
- [Deep Residual Decision Making](https://arxiv.org/pdf/1908.10737v1.pdf)
- [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://arxiv.org/pdf/1811.12359.pdf)
- https://www.compstat.statistik.uni-muenchen.de/publications/
- [Deep Embedding Forest: Forest-based Serving with Deep Embedding Features](https://users.soe.ucsc.edu/~holakou/files/papers/DEF_KDD_2017.pdf)
- [Multi-Label Learning with Deep Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/ecai20mldf.pdf)
- [Talk: An exploration to non-NN deep models based on non-differentiable modules](https://www.aistats.org/aistats2019/0-AISTATS2019-slides-zhi-hua_zhou.pdf)

##  Recursive partitioning and spatial trees


[Recursive partitioning is a very simple idea for clustering. It is the inverse of hierarchical clustering.](https://online.stat.psu.edu/stat555/node/100/)
[This article provides an introduction to recursive partitioning techniques;](https://www.sciencedirect.com/science/article/pii/B9780080448947013142)
that is, methods that predict the value  of a response variable by forming subgroups of subjects 
within which the response is relatively homogeneous on the basis of the values of a set of predictor variables.
[Recursive partitioning are essentially fairly simple nonparametric techniques for prediction and classification.](https://www.sciencedirect.com/topics/mathematics/recursive-partitioning) 
When used in the standard way, they provide trees which display the succession of rules that need to be followed to derive a predicted value or class. 
This simple visual display explains the appeal to decision makers from many disciplines.
[Recursive partitioning methods have become popular and widely used tools for non-parametric regression and classification in many scientific fields.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2927982/)

## Recursive partitioning



It works by building a forest of N binary random projection trees.

In each tree, the set of training points is recursively partitioned into smaller and smaller subsets until a leaf node of at most M points is reached. 
Each partition is based on the cosine of the angle the points make with a randomly drawn hyperplane: points whose angle is smaller than the median angle fall in the left partition, and the remaining points fall in the right partition.

The resulting tree has predictable leaf size (no larger than M) and is approximately balanced because of median splits, leading to consistent tree traversal times.

Querying the model is accomplished by traversing each tree to the query point's leaf node to retrieve ANN candidates from that tree, then merging them and sorting by distance to the query point.

- https://github.com/lyst/rpforest

[We present such a method for multi-dimensional image compression called Compression via Adaptive Recursive Partitioning (CARP). ](https://arxiv.org/abs/1912.05622)
In Nearest Neighbor Search, recursive  partitioning is referred as an approximate methods related with k-d tree as shown in 
[Nearest neighbors and vector models – part 2 – algorithms and data structures](https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces.html).

<img src="https://erikbern.com/assets/2015/09/tree-full-K-1024x793.png" width="80%" />

Nice! We end up with a binary tree that partitions the space. 
The nice thing is `that points that are close to each other in the space 
are more likely to be close to each other in the tree`. 
In other words, if two points are close to each other in the space, 
it's unlikely that any hyperplane will cut them apart.

To search for any point in this space, 
we can traverse the binary tree from the root. 
Every intermediate node (the small squares in the tree above) defines a hyperplane, 
so we can figure out what side of the hyperplane 
we need to go on and that defines 
if we go down to the left or right child node. 
Searching for a point can be done in logarithmic time 
since that is the height of the tree.

- [k-d Tree Nearest Neighbor Search](http://andrewd.ces.clemson.edu/courses/cpsc805/references/nearest_search.pdf)
- [14.2 - Recursive Partitioning](https://online.stat.psu.edu/stat555/node/100/)
- [Recursive Partitioning](https://www.sciencedirect.com/science/article/pii/B9780080448947013142)
- http://meng.rice.edu/research/
- https://core.ac.uk/display/6786050
- [Metric Indexes based on Recursive Voronoi Partitioning](http://disa.fi.muni.cz/wp-content/uploads/disa-voronoi-indexes.pdf)

### Spatial trees 

[Spatial Trees are a recursive space partitioning data structure that can help organize high-dimensional data.](http://cseweb.ucsd.edu/~naverma/SpatialTrees/index.html)
They can assist in analyzing the underlying `data density`, 
perform fast `nearest-neighbor searches`, 
and do high quality `vector-quantization`.
There are several instantiations of spatial trees. 
The most popular include KD trees, RP trees (random projection trees), PD trees (principal direction trees). 


- https://github.com/rpav/spatial-trees
- https://www.cliki.net/spatial-trees
- http://www.sai.msu.su/~megera/
- [Random projection trees and low dimensional manifolds](http://cseweb.ucsd.edu/~dasgupta/papers/rptree-stoc.pdf)
- [Random Projections of Smooth Manifolds](http://inside.mines.edu/~mwakin/papers/randProjManifolds-19sept2007.pdf)
- https://github.com/vioshyvo/mrpt
- https://github.com/nmslib/nmslib
- [Fast Nearest Neighbor Search through
Sparse Random Projections and Voting](https://www.cs.helsinki.fi/u/ttonteri/pub/bigdata2016.pdf)
- https://github.com/lmcinnes/pynndescent
- https://github.com/yahoojapan/NGT
- http://ann-benchmarks.com/
- https://zilliz.com/cn/
- https://www.milvus.io/

## Application of Decision Trees

With the availability of large databases and recent improvements in deep learning methodology, the performance of AI systems is reaching, or even exceeding, the human level on an increasing number of complex tasks. 
Impressive examples of this development can be found in domains such as `image classification`, `sentiment analysis`, `speech understanding` or `strategic game playing`. 
However, because of their `nested non-linear structure`, these highly successful machine learning and artificial intelligence models are usually applied in a black-box manner, i.e. [no information is provided about what exactly makes them arrive at their predictions.](https://www.itu.int/en/journal/001/Documents/itu2017-5.pdf)
Since this lack of transparency can be a major drawback, e.g. in medical applications, the development of methods for `visualizing, explaining and interpreting deep learning` models has recently attracted increasing
attention.


The decision boundary of decision trees are clear and parameterized.
The visualization, explaining and interpreting decision tree models seems muck easier than deep models.

Like geometry, there are diverse forms of decision trees:
1. Graphical form
2. Logical  form
3. Analytic form

And analytic form of decision trees are close to neural network.
Neural decision trees,

- https://github.com/wOOL/DNDT
- [Interpretable Decision Sets: A Joint Framework for
Description and Prediction](https://cs.stanford.edu/people/jure/pubs/interpretable-kdd16.pdf)
- http://yosinski.com/deepvis
- https://sites.google.com/corp/view/whi2018
- http://airesearch.com/
- https://www.itu.int/en/journal/001/Documents/itu2017-5.pdf
- [Interpretable Machine Learning](https://deepai.org/publication/techniques-for-interpretable-machine-learning)
- https://christophm.github.io/interpretable-ml-book/
- https://beenkim.github.io/
- https://zhuanlan.zhihu.com/p/78822770
- https://zhuanlan.zhihu.com/p/74542589
- https://www.jiqizhixin.com/articles/0211https://www.jiqizhixin.com/articles/0211https://www.jiqizhixin.com/articles/0211
- https://www.unite.ai/what-is-a-decision-tree/
- https://github.com/conan7882/CNN-Visualization

## Sum-product network

The [sum-product] network is the general form in [Tractable Deep Learning].

Essentially all tractable graphical models can be cast as SPNs,
but SPNs are also strictly more general. 
We then propose learning algorithms for SPNs, based on backpropagation and EM.

The compactness of graphical models can often be greatly
increased by postulating the existence of hidden variables
$y$: $P(X=x)=\frac{1}{Z}\sum_{y}\prod_{k}\phi_k((x, y)_{\{k\}})$
where $Z$ is the normalized constant.
Note that this is similar to the decision tree $T(x)=\sum_{\ell}^{|T|} c_{\ell}\prod_{i\in P(\ell)}\sigma(\pm(x_i)-\tau_i)$.

[The SPN is defined as following](http://ce.sharif.edu/courses/98-99/1/ce719-1/resources/root/Slides/Lect-25.pdf)
> A SPN is rooted DAG whose leaves are $x_1, \cdots , x_n$ and $\bar{x}_1, \cdots, \bar{x}_n$ with
internal sum and product nodes, where each edge $(i, j)$ emanating
from sum node $i$ has a weight $w_{i} \geq  0$.

Advantages of SPNs:
* Unlike graphical models, SPNs are tractable over high treewidth models
* SPNs are a deep architecture with full probabilistic semantics
* SPNs can incorporate features into an expressive model without requiring approximate inference.

- http://spn.cs.washington.edu/
- https://github.com/arranger1044/awesome-spn

## Statistical relational learning

Decision tree is flexible with categorical and numerical variable.
Besides the [dummy variables](https://www.statisticshowto.datasciencecentral.com/dummy-variables/),
another generalization of decision tree is the statistical relational learning based on sum-product network.

[`Statistical relational learning (SRL)` is revolutionizing the field of automated learning and discovery by moving beyond the conventional analysis of entities in isolation to analyze networks of interconnected entities. In relational domains such as bioinformatics, citation analysis, epidemiology, fraud detection, intelligence analysis, and web analytics, there is often limited information about any one entity in isolation, instead it is the connections among entities that are of crucial importance to pattern discovery. ](https://www.cs.purdue.edu/homes/neville/courses/CS590N.html)

Conventional machine learning techniques have two primary assumptions that limit their application in relational domains.
First, algorithms for propositional data assume that data instances are recorded in `homogeneous structures` (i.e., a fixed number of attributes for each entity) but relational data instances are usually more varied and complex (e.g., molecules have different numbers of atoms and bonds). 
Second, the algorithms assume that data instances are independent but relational data often violate this assumption---`dependencies` may occur either as a result of direct relations or through chaining multiple relations together. 
For example, scientific papers have dependencies through both citations (direct) and authors (indirect).


- [Introduction to Statistical Relational Learning](http://www.cs.umd.edu/srl-book/)
- https://data-science-blog.com/blog/2016/08/17/statistical-relational-learning/
- https://martint.blog/
- [ Robert Peharz publication](http://www3.eng.cam.ac.uk/research_db/publications/rp587)
- [Talk: Tractable Models and Deep Learning: A Love Marriage (Robert Peharz)](https://dtai.cs.kuleuven.be/seminars/tractable-models-and-deep-learning-love-marriage-robert-peharz)



## Pairwise Learning

[Pairwise Learning](https://www.ijcai.org/Proceedings/2018/0329.pdf)  refers to learning tasks with the associated loss functions depending on pairs of examples.
[Circle Loss](https://arxiv.org/pdf/2002.10857.pdf) provides a pair similarity optimization viewpoint on deep feature learning, aiming to maximize the within-class similarity $s_p$ and minimize the between-class similarity $s_n$.

Optimizing $(s_n−s_p)$ usually leads to a decision boundary of $s_p − s_n = m$ ($m$ is the margin).
Given a single sample $x$ in the feature space, let us assume that there are $K$ within-class similarity scores $\{s_p^i\mid i=1,2,\cdots,K\}$ and $L$ between-class similarity scores $\{s_n^j\mid j=1,2,\cdots, L\}$ associated with $x$.
A  `unified loss` function is defined as 
$$\mathcal{L}_{uni}(x)=\log[1+\exp(\sum_{i}\sum_{j}\gamma((s_n^j − s_p^i+m)))]$$
in which $\gamma$ is a scale factor and $m$ is a margin for better
similarity separation.

We consider to enhance the optimization flexibility by allowing each similarity score to learn at its own pace, depending on its current optimization status.
The unified loss function is transferred into the proposed `Circle loss` by
$$\mathcal{L}_{circle}(x)=\log[1+\exp(\sum_{i}\sum_{j}\gamma((\alpha_p^i s_n^j − \alpha_p^i s_p^i)))]$$
in which $\alpha_p^i$ and $\alpha_n^j$  are non-negative weighting factors.


- [Learning pairwise image similarities for multi-classification using Kernel Regression Trees](http://www.brunel.ac.uk/~csstyyl/papers/pr2012.pdf)
- https://arxiv.org/pdf/1612.02295.pdf
- https://arxiv.org/pdf/1703.07464.pdf
- https://www.salford-systems.com/products/mars
- [MARS](http://www.stat.yale.edu/~lc436/08Spring665/Mars_Friedman_91.pdf)
- [Discussion](http://www.stat.yale.edu/~arb4/publications_files/DiscussionMultivariateAdaptiveRegressionSplines.pdf)
- [Tree-classifier via Nearest Neighbor Graph Partitioning](http://papers.www2017.com.au.s3-website-ap-southeast-2.amazonaws.com/companion/p845.pdf)
- [Recursive Partitioning for Personalization using Observational Data](https://arxiv.org/pdf/1608.08925.pdf)
- https://statmodeling.stat.columbia.edu/2019/11/26/machine-learning-under-a-modern-optimization-lens-under-a-bayesian-lens/
- https://arxiv.org/abs/1908.01755
- https://arxiv.org/abs/1904.12847
- http://akyrillidis.github.io/


————————

- https://github.com/jwasham/coding-interview-university
- https://github.com/jackfrued/Python-100-Days

*****

[unit step function]: https://mathworld.wolfram.com/HeavisideStepFunction.html
[IndicatorFunctions]: https://www.statlect.com/fundamentals-of-probability/indicator-functions
[CharacteristicFunction]:https://mathworld.wolfram.com/CharacteristicFunction.html
[SimpleFunction]: https://mathworld.wolfram.com/SimpleFunction.html
[MARS]: https://projecteuclid.org/download/pdf_1/euclid.aos/1176347963
[decision boundary]: https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/PR_simp/bndrys.htm
[B-spline]: https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html
[Bézier Curve]: https://mathworld.wolfram.com/BezierCurve.html
[sum-product]: http://spn.cs.washington.edu/
[Tractable Deep Learning]: https://www.cs.washington.edu/research/tractable-deep-learning
[fuzzy decision tree]: http://www.montefiore.ulg.ac.be/services/stochastic/pubs/2003/OW03/OW03.pdf
[decision manifold]: http://www.ifs.tuwien.ac.at/~lidy/pub/poe_lidy_wsom07.pdf
[Template Matching]: https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/PR_simp/template.htm

### Contrastive Learning

Clusters of points belonging to the same class are pulled together in embedding space,
while simultaneously pushing apart clusters of samples from different classes

- https://paperswithcode.com/
- https://paperswithcode.com/sota
- https://github.com/HobbitLong/SupContrast
- [Supervised Contrastive Learning](https://arxiv.org/pdf/2004.11362v1.pdf)
- https://ankeshanand.com/blog/2020/01/26/contrative-self-supervised-learning.html


# Computational Graph and Graph Learning

## Computational Graph

[A computational graph is defined as](https://www.tutorialspoint.com/python_deep_learning/python_deep_learning_computational_graphs.htm)
 a directed graph where the `nodes` correspond to mathematical operations. Computational graphs are a way of expressing and evaluating a mathematical expression.

- https://deepnotes.io/tensorflow
- https://sites.google.com/a/uclab.re.kr/lhparkdeep/feeding-habits
- https://kharshit.github.io/blog/
- http://rll.berkeley.edu/cgt/
- http://www.charuaggarwal.net/
- http://colah.github.io/posts/2015-08-Backprop/

One application of computational graph is to explain the `automatic differentiation` or `auto-diff`.

- http://www.autodiff.org/
- https://github.com/apache/incubator-tvm
- https://tvm.apache.org/
- https://github.com/google/jax

## Graph Learning 

Graph Learning is to learn from the graph data:
[Simply said, it’s the application of machine learning techniques on graph-like data. ](https://graphsandnetworks.com/graph-learning/)
Deep learning techniques (neural networks) can, in particular,  be applied and yield new opportunities which classic algorithms cannot deliver.

- https://snap-stanford.github.io/cs224w-notes/machine-learning-with-networks/graph-neural-networks
- https://graphsandnetworks.com/blog/
- http://web.stanford.edu/class/cs224w/
- https://networkdatascience.ceu.edu/
- http://tensorlab.cms.caltech.edu/users/anima/
- https://graphvite.io/
- https://github.com/alibaba/graph-learn
- https://www.mindspore.cn/install

### Graph

There are four key network properties to characterize a graph: `degree distribution`, `path length`, `clustering coefficient`, and `connected components`. 

The connectivity of a graph measures the `size` of the largest connected component.
 The largest connected component is the largest set where any two vertices can be joined by a path.

The clustering coefficient (for undirected graphs) measures what proportion of node $i$’s neighbors are connected. 

- https://github.com/GraphBLAS/graphblas-pointers
- https://www.cs.yale.edu/homes/spielman/
- https://www.graphengine.io/
- https://github.com/alibaba/euler
- https://ci.apache.org/projects/flink/flink-docs-stable/dev/libs/gelly/
- http://spark.apache.org/graphx/

****

[Transformers are Graph Neural Networks](https://graphdeeplearning.github.io/post/transformers-are-gnns/)

>  At a high level, all neural network architectures build representations of input data as vectors/embeddings, 
which encode useful statistical and semantic information about the data. These latent or hidden representations can then be used for performing something useful, 
such as classifying an image or translating a sentence. 

We update the  hidden feature  of the $j$th word in a sentence   from layer $\ell$ to layer $\ell+1$ as follows:
$$\ h_{i}^{\ell+1}=\operatorname{Attention}(h_{i}^{\ell}, h_{j}^{\ell}) = \sum_{j \in \mathcal{S}} w_{ij} \left( V^{\ell} h_{j}^{\ell} \right),$$

$$\text{where} \ w_{ij} = \text{softmax}_j \left( Q^{\ell} h_{i}^{\ell} \cdot  K^{\ell} h_{j}^{\ell} \right),$$
where $Q^{\ell}, K^{\ell}, V^{\ell}$ are learnable linear weights (denoting the Query, Key and Value for the attention computation, respectively). The attention mechanism is performed parallelly for each word in the sentence to obtain their updated features in one shot–another plus point for Transformers over RNNs,
 which update features word-by-word.

- [Stacked Capsule Auto-encoders](https://arxiv.org/pdf/1906.06818.pdf)
- [Model-based Recursive Partitioning](https://eeecon.uibk.ac.at/~zeileis/papers/Zeileis+Hothorn+Hornik-2008.pdf)
