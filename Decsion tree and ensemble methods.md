## Tree-based Learning Algorithms

[高效决策树算法系列笔记](https://github.com/wepe/efficient-decision-tree-notes)

The [simple function](https://proofwiki.org/wiki/Definition:Simple_Function) is a real-valued  function $f: \mathrm{X}\to \mathbb{R}$ if and only if it is a finite linear combination of characteristic functions:
$$f=\sum_{i=1}^{n}a_k {\chi}_{S_{k}}$$
where $a_k\in\mathbb{R}$ and the characteristic function is defined as follow
$${\chi}_{S_{k}}=\begin{cases}1, &\text{if $x \in S_{k}$}\\
0, &\text{otherwise}\end{cases}.$$

The tree-based learning algorithms take advantages of these [universal approximators](http://mathonline.wikidot.com/the-simple-function-approximation-theorem) to fit the decision function.

<img title="https://cdn.stocksnap.io/" src="https://cdn.stocksnap.io/img-thumbs/960w/TIHPAM0QFG.jpg" width="80%" />

The core problem is to find the optimal parameters $a_k\in\mathbb{R}$ and $S_k\in\mathbb{R}^p$ when only some finite sample or training data $\{(\mathrm{x}_i, y_i)\mid i=1, 2, \dots, n\}$ is accessible or available where $\mathrm{x}_i\in\mathbb{R}^p$ and $y_i\in\mathbb{R}$ or some categorical domain.

### Decision Tree

In brief, decision tree is to find the optimal parameters by **coordinate-wise** splitting methods.

> A decision tree is the function $T :\mathbb{R}^d \to \mathbb{R}$ resulting from a learning algorithm applied on training data lying in input space $\mathbb{R}^d$ , which always has the following form:
> $$
> T(x) = \sum_{i\in\text{leaves}} g_i(x)\mathbb{I}(x\in R_i) = \sum_{i\in \,\text{leaves}} g_i(x) \prod_{a\in\,\text{ancestors(i)}} \mathbb{I}(S_{a (x)}=c_{a,i})
> $$
> where $R_i \subset \mathbb{R}^d$ is the region associated with leaf ${i}$ of the tree, $\text{ancestors(i)}$ is the set of ancestors of leaf node i, $c_{a,i}$ is the child of node a on the path from a to leaf i, and $S_a$ is the n-array split function at node a.
> $g_i(\cdot)$ is the decision function associated with leaf i and
> is learned only from training examples in $R_i$.

The $g_{i}(x)$ can be a constant in $\mathbb{R}$ or some mathematical expression such as logistic regression. When $g_i(x)$ is constant, the decision tree is actually piecewise constant, a concrete example of simple function.

* [Decision Trees (for Classification) by Willkommen auf meinen Webseiten.](http://christianherta.de/lehre/dataScience/machineLearning/decision-trees.php)
* [DECISION TREES DO NOT GENERALIZE TO NEW VARIATIONS](https://www.iro.umontreal.ca/~lisa/pointeurs/bengio+al-decisiontrees-2010.pdf)
* [On the Boosting Ability of Top-Down Decision Tree Learning Algorithms](http://www.columbia.edu/~aec2163/NonFlash/Papers/Boosting2016.pdf)

A decision tree is a set of questions(i.e. if-then sentence) organized in a **hierarchical** manner and represented graphically as a tree.
It use 'divide-and-conquer' strategy recursively. It is easy to scale up to massive data set. The models are obtained by recursively partitioning
the data space and fitting a simple prediction model within each partition. As a
result, the partitioning can be represented graphically as a decision tree.
[Visual introduction to machine learning](https://explained.ai/decision-tree-viz/index.html) show an visual introduction to decision tree.

***
**Algorithm**  Pseudocode for tree construction by exhaustive search

1. Start at root node.
2. For each node ${X}$, find the set ${S}$ that **minimizes** the sum of the node $\fbox{impurities}$ in the two child nodes and choose the split $\{X^{\ast}\in S^{\ast}\}$ that gives the minimum overall $X$ and $S$.
3. If a stopping criterion is reached, exit. Otherwise, apply step 2 to each child node in turn.

***

Creating a binary decision tree is actually a process of dividing up the input space according to the sum of **impurities**.

This learning process is to minimize the impurities.
C4.5 and CART6 are two later classification
tree algorithms that follow this approach. C4.5 uses `entropy` for its impurity function,
whereas CART uses a generalization of the binomial variance called the `Gini index`.

If the training set $D$ is divided into subsets $D_1,\dots,D_k$, the entropy may be
reduced, and the amount of the reduction is the information gain,

$$
G(D; D_1, \dots, D_k)=Ent(D)-\sum_{i=1}^{k}\frac{|D_k|}{|D|}Ent(D_k)
$$

where $Ent(D)$, the entropy of $D$, is defined as

$$
Ent(D)=\sum_{y \in Y} P(y|D)\log(\frac{1}{P(y | D)}).
$$


The information gain ratio of features $A$ with respect of data set $D$  is defined as

$$
g_{R}(D,A)=\frac{G(D,A)}{Ent(D)}.
$$
And another option of impurity is Gini index of probability $p$:

$$
Gini(p)=\sum_{y}p_y (1-p_y)=1-\sum_{y}p_y^2.
$$

Algorithm | Impurity
---|---
[ID3](https://www.wikiwand.com/en/ID3_algorithm)| Informatio
[C4.5](https://www.wikiwand.com/en/C4.5_algorithm)| Normalized information gain ratio
CART|Gini Index

$\color{red}{\text{PS: all above impurities}}$  are based on the probability $\fbox{distribuion}$  of data.
***

* [Data Mining Tools See5 and C5.0](https://www.rulequest.com/see5-info.html)
* [A useful view of decision trees](https://www.benkuhn.net/tree-imp)
* https://www.wikiwand.com/en/Decision_tree_learning
* https://www.wikiwand.com/en/Decision_tree
* https://www.wikiwand.com/en/Recursive_partitioning
* [TimeSleuth is an open source software tool for generating temporal rules from sequential data](http://timesleuth-rule.sourceforge.net/)
***

Like other supervised algorithms, decision tree makes a trade-off between over-fitting and under-fitting and how to choose the hyper-parameters of decision tree such as the max depth?
The regularization techniques in regression may not suit the tree algorithms such as LASSO.

**Pruning** is a regularization technique for tree-based algorithm. In arboriculture, the reason to prune tree is [because each cut has the potential to change the growth of the tree, no branch should be removed without a reason. Common reasons for pruning are to remove dead branches, to improve form, and to reduce risk. Trees may also be pruned to increase light and air penetration to the inside of the tree’s crown or to the landscape below. ](https://www.treesaregood.org/treeowner/pruningyourtrees)

<img title = "pruning" src="https://www.treesaregood.org/portals/0/images/treeowner/pruning1.jpg" width="40%" />

In machine learning, we prune the decision tree to make a balance between over-fitting and under-fitting. The important step of tree pruning is to define a criterion be used to determine the correct final tree size using one of the following methods:

1. Use a distinct dataset from the training set (called validation set), to evaluate the effect of post-pruning nodes from the tree.
2. Build the tree by using the training set, then apply a statistical test to estimate whether pruning or expanding a particular node is likely to produce an improvement beyond the training set.
    * Error estimation
    * Significance testing (e.g., Chi-square test)
3. Minimum Description Length principle : Use an explicit measure of the complexity for encoding the training set and the decision tree, stopping growth of the tree when this encoding size (size(tree) + size(misclassifications(tree)) is minimized.

* [Decision Tree - Overfitting saedsayad](https://www.saedsayad.com/decision_tree_overfitting.htm)
* [Decision Tree Pruning based on Confidence Intervals (as in C4.5)](http://www.cs.bc.edu/~alvarez/ML/statPruning.html)

***

When the height of a decision tree is limited to 1, i.e., it takes only one
test to make every prediction, the tree is called a decision stump. While decision trees are nonlinear classifiers in general, decision stumps are a kind
of linear classifiers.

[Fifty Years of Classification and
Regression Trees](http://www.stat.wisc.edu/~loh/treeprogs/guide/LohISI14.pdf) and [the website of Wei-Yin Loh](http://www.stat.wisc.edu/~loh/guide.html) helps much understand the decision tree.
Multivariate Adaptive Regression
Splines(MARS) is the boosting ensemble methods for decision tree algorithms.
`Recursive partition` is a recursive  way to construct decision tree.

***
* [Treelite : model compiler for decision tree ensembles](https://treelite.readthedocs.io/en/latest/)
* [Tutorial on Regression Tree Methods for Precision Medicine and Tutorial on Medical Product Safety: Biological Models and Statistical Methods](http://ims.nus.edu.sg/events/2017/quan/tut.php)
* [An Introduction to Recursive Partitioning: Rationale, Application and Characteristics of Classification and Regression Trees, Bagging and Random Forests](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2927982/)
* [ADAPTIVE CONCENTRATION OF REGRESSION TREES, WITH APPLICATION TO RANDOM FORESTS](https://arxiv.org/pdf/1503.06388.pdf)
* [GUIDE Classification and Regression Trees and Forests (version 31.0)](http://www.stat.wisc.edu/~loh/guide.html)
* [How to visualize decision trees by Terence Parr and Prince Grover](https://explained.ai/decision-tree-viz/index.html)
* [CART](https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/)
* [A visual introduction to machine learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
* [Interpretable Machine Learning: Decision Tree](https://christophm.github.io/interpretable-ml-book/tree.html)
* [Tree-based Models](https://dinh-hung-tu.github.io/tree-based-models/)
* http://ai-depot.com/Tutorial/DecisionTrees-Partitioning.html
* https://www.ncbi.nlm.nih.gov/pubmed/16149128
* http://www.cnblogs.com/en-heng/p/5035945.html

### Random Forest

[Decision Trees do not generalize to new variations](https://www.iro.umontreal.ca/~lisa/pointeurs/bengio+al-decisiontrees-2010.pdf) demonstrates some theoretical limitations of decision trees. And they can be seriously hurt by the curse of dimensionality in a sense that is a bit different
from other nonparametric statistical methods, but most importantly, that they cannot generalize to variations not
seen in the training set. This is because a decision tree creates a partition of the input space and needs at least
one example in each of the regions associated with a leaf to make a sensible prediction in that region. A better
understanding of the fundamental reasons for this limitation suggests that one should use forests or even deeper
architectures instead of trees, which provide a form of distributed representation and can generalize to variations
not encountered in the training data.

Random forests (Breiman, 2001) is a substantial modification of bagging
that builds a large collection of de-correlated trees, and then averages them.


On many problems the performance of random forests is very similar to boosting, and they are simpler to train and tune.

***

* For $t=1, 2, \dots, T$:
   + Draw a bootstrap sample $Z^{\ast}$ of size $N$ from the training data.
   + Grow a random-forest tree $T_t$ to the bootstrapped data, by recursively repeating the following steps for each terminal node of the tree, until the minimum node size $n_{min}$ is reached.
     - Select $m$ variables at random from the $p$ variables.
     - Pick the best variable/split-point among the $m$.
     - Split the node into two daughter nodes.
* Vote for classification and average for regression.

<img src="https://dimensionless.in/wp-content/uploads/RandomForest_blog_files/figure-html/voting.png" width="80%" />

***

* [randomForestExplainer](https://mi2datalab.github.io/randomForestExplainer/index.html)
* [Awesome Random Forest](https://github.com/kjw0612/awesome-random-forest)
* [Interpreting random forests](https://blog.datadive.net/interpreting-random-forests/)
* [Random Forests by Leo Breiman and Adele Cutler](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)
* https://dimensionless.in/author/raghav/
* http://www.rhaensch.de/vrf.html
* https://www.wikiwand.com/en/Random_forest
* https://sktbrain.github.io/awesome-recruit-en.v2/
* https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
* https://dimensionless.in/introduction-to-random-forest/
* https://www.elderresearch.com/blog/modeling-with-random-forests

## Ensemble methods

There are many competing techniques for solving the problem, and each technique is characterized
by choices and meta-parameters: when this flexibility is taken into account, one easily
ends up with a very large number of possible models for a given task.

* [Computer Science 598A: Boosting: Foundations & Algorithms](http://www.cs.princeton.edu/courses/archive/spring12/cos598A/)
* [Zhou Zhihua's publication on ensemble methods](http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/publication_toc.htm#Ensemble%20Learning)
* [Ensemble Learning  literature review](http://www.machine-learning.martinsewell.com/ensembles/)
* [KAGGLE ENSEMBLING GUIDE](https://mlwave.com/kaggle-ensembling-guide/)
* [Ensemble Machine Learning: Methods and Applications](https://www.springer.com/us/book/9781441993250)
* [MAJORITY VOTE CLASSIFIERS: THEORY AND APPLICATION](https://web.stanford.edu/~hastie/THESES/gareth_james.pdf)
* [Neural Random Forests](https://arxiv.org/abs/1604.07143)
* [Generalized Random Forests](https://arxiv.org/abs/1610.01271)
* [Additive Models, Boosting, and Inference for Generalized Divergences ](https://www.stat.berkeley.edu/~binyu/summer08/colin.bregman.pdf)
* [Boosting as Entropy Projection](https://users.soe.ucsc.edu/~manfred/pubs/C51.pdf)
* [Weak Learning, Boosting, and the AdaBoost algorithm](https://jeremykun.com/2015/05/18/boosting-census/)

### Bagging

Bagging, short for 'bootstrap aggregating', is a simple but highly effective ensemble method that creates diverse models on different random bootstrap samples of the original data set.
[Random forest](https://www.wikiwand.com/en/Random_forest) is the application of bagging to decision tree algorithms.

The basic motivation of parallel ensemble methods is to exploit the independence between the
base learners, since the error can be reduced dramatically by combining independent base learners.
Bagging adopts the most popular strategies for aggregating the outputs of
the base learners, that is, voting for classification and averaging for regression.

* Draw `bootstrap samples` $B_1, B_2, \dots, B_n$ independently from the original training data set for base learners;
* Train the $i$th base learner $F_i$ at the ${B}_{i}$;
* Vote for classification and average for regression.

<img title = bootstrap-sample src=https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2016/10/bootstrap-sample.png width=80%/>

It is a sample-based ensemble method.

There is an alternative of bagging called combining ensemble method. It trains a linear combination of learner:
$$F = \sum_{i=1}^{n} w_i F_i$$
where the weights $w_i\geq 0, \sum_{i=1}^{n} w_i =1$. The weights $w=\{w_i\}_{i=1}^{n}$ are solved by minimizing the ensemble error
$$
w = \arg\min_{w}\sum_{k}^{K}(F(x_k)-y_k)^{2}
$$
if the training data set $\{x_k, y_k\}_{k=1}^{K}$ is given.


<img title = weighted-unweighted src=https://blogs.sas.com/content/subconsciousmusings/files/2017/05/weighted-unweighted.png width=80%/>

***

* http://www.machine-learning.martinsewell.com/ensembles/bagging/
* https://www.cnblogs.com/earendil/p/8872001.html
* https://www.wikiwand.com/en/Bootstrap_aggregating
* [Bagging Regularizes](http://dspace.mit.edu/bitstream/handle/1721.1/7268/AIM-2002-003.pdf?sequence=2)
* [Bootstrap Inspired Techniques in Computational Intelligence](http://users.rowan.edu/~polikar/RESEARCH/PUBLICATIONS/spm2007.pdf)

#### Random Subspace Methods

[Abstract: "Much of previous attention on decision trees focuses on the splitting criteria and optimization of tree sizes. The dilemma between overfitting and achieving maximum accuracy is seldom resolved. A method to construct a decision tree based classifier is proposed that maintains highest accuracy on training data and improves on generalization accuracy as it grows in complexity. The classifier consists of multiple trees constructed systematically by pseudorandomly selecting subsets of components of the feature vector, that is, trees constructed in randomly chosen subspaces. The subspace method is compared to single-tree classifiers and other forest construction methods by experiments on publicly available datasets, where the method's superiority is demonstrated. We also discuss independence between trees in a forest and relate that to the combined classification accuracy."](http://www.machine-learning.martinsewell.com/ensembles/rsm/Ho1998.pdf)

+ http://www.machine-learning.martinsewell.com/ensembles/rsm/


### Boosting

* http://rob.schapire.net/papers
* https://cseweb.ucsd.edu/~yfreund/papers
* http://www.boosting.org/

The term boosting refers to a family of algorithms that are able to convert weak learners to strong learners.
It is kind of similar to the "trial and error" scheme: if we know that the learners perform worse at some given data set $S$,
the learner may pay more attention to the data drawn from $S$.
For the regression problem, of which the output results are continuous, it  progressively reduce the error by trial.
In another word, we will reduce the error at each iteration.

<img title = Chinese_herb_clinic src = http://www.stat.ucla.edu/~sczhu/Vision_photo/Chinese_herb_clinic.jpg width=50% />


* https://mlcourse.ai/articles/topic10-boosting/
* [Reweighting with Boosted Decision Trees](https://arogozhnikov.github.io/2015/10/09/gradient-boosted-reweighter.html)
* https://betterexplained.com/articles/adept-method/
* [BOOSTING ALGORITHMS: REGULARIZATION, PREDICTION AND MODEL FITTING](https://web.stanford.edu/~hastie/Papers/buehlmann.pdf)
* [What is the difference between Bagging and Boosting?](https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/)
* [Boosting](http://www.machine-learning.martinsewell.com/ensembles/boosting/) and [Ensemble Learning](http://www.machine-learning.martinsewell.com/ensembles/)
* [Boosting at Wikipedia](https://www.wikiwand.com/en/Boosting_(machine_learning))

#### AdaBoost

`AdaBoost` is a boosting methods for supervised classification algorithms, so that the labeled data set is given in the form $D=\{ (x_i, \mathrm{d}_i)\}_{i=1}^{N}$.
AdaBoost is to change the distribution of training data and learn from the shuffled data.
It is an iterative trial-and-error in some sense.


***

* Input  $D=\{ (x_i, \mathrm{d}_i)\}_{i=1}^{N}$ where $x\in \mathcal X$ and $\mathrm{d}\in \{+1, -1\}$.
* Initialize the observation weights ${w}_i=\frac{1}{N}, i=1, 2, \dots, N$.
* For $t = 1, 2, \dots, T$:
   +  Fit a classifier $G_m(x)$ to the training data using weights $w_i$.
   +  Compute
      $$err_{t}=\frac{\sum_{i=1}^{N}w_i \mathbb{I}(G_t(x_i) \not= \mathrm{d}_i)}{\sum_{i=1}^{N} w_i}.$$
   +  Compute $\alpha_t = \log(\frac{1-err_t}{err_t})$.
   +  Set $w_i\leftarrow w_i\exp[\alpha_t(\mathrm{d}_i G_t(x_i))], i=1,2,\dots, N$.
* Output $G(x)=sign[\sum_{t=1}^{T}\alpha_{t}G_t(x)]$.

The indicator function $\mathbb{I}(x\neq y)$ is defined as
$$
\mathbb{I}(x\neq y)=
  \begin{cases}
    1, \text{if $x\neq y$} \\
    0, \text{otherwise}.
  \end{cases}
$$

![1-reweighting](https://arogozhnikov.github.io/images/reweighter/1-reweighting.png)
***

* [AdaBoost at Wikipedia](https://www.wikiwand.com/en/AdaBoost)
* [BrownBoost at Wikipedia](https://www.wikiwand.com/en/BrownBoost)
* [CoBoosting at Wikipedia ](https://www.wikiwand.com/en/CoBoosting)
* [CSDN blog: Adaboost 算法的原理与推导](https://blog.csdn.net/v_july_v/article/details/40718799)

<img src ="https://cseweb.ucsd.edu/~yfreund/portraitsmall.jpg" width="30%" />
<img src=https://www.microsoft.com/en-us/research/wp-content/uploads/2017/09/avatar_user_33549_1504711750-180x180.jpg width=40% />

**Real AdaBoost**

In AdaBoost, the error is binary- it is 0 if the classification is right otherwise it is 1. It is not precise for some setting. The output of decision trees is a class probability estimate $p(x) = P(y=1 | x)$, the probability that ${x}$ is in the positive class

* Input  $D=\{ (x_i, \mathrm{y}_i)\}_{i=1}^{N}$ where $x\in \mathcal X$ and $y\in \{+1, -1\}$.
* Initialize the observation weights ${w}_i=\frac{1}{N}, i=1, 2, \dots, N$;
* For $m = 1, 2, \dots, M$:
   +  Fit a classifier $G_m(x)$ to obtain a class probability estimate $p_m(x)=\hat{P}_{w}(y=1\mid x)\in [0, 1]$, using weights $w_i$.
   +  Compute $f_m(x)\leftarrow \frac{1}{2}\log{p_m(x)/(1-p_m(x))}\in\mathbb{R}$.
   +  Set $w_i\leftarrow w_i\exp[-y_if_m(x_i)], i=1,2,\dots, N$ and renormalize so that $\sum_{i=1}w_i =1$.
* Output $G(x)=sign[\sum_{t=1}^{M}\alpha_{t}G_m(x)]$.

_______

* [Additive logistic regression: a statistical view of boosting](https://web.stanford.edu/~hastie/Papers/AdditiveLogisticRegression/alr.pdf)

**Gentle AdaBoost**

* Input  $D=\{ (x_i, \mathrm{y}_i)\}_{i=1}^{N}$ where $x\in \mathcal X$ and $y\in \{+1, -1\}$.
* Initialize the observation weights ${w}_i=\frac{1}{N}, i=1, 2, \dots, N, F(x)=0$;
* For $m = 1, 2, \dots, M$:
   + Fit a classifier $f_m(x)$ by weighted least-squares of $\mathrm{y}_i$ to $x_i$ with weights $w_i$.
   + Update $F(x)\leftarrow F(x)+f_m (x)$.
   + UPdate $w_i \leftarrow w_i \exp(-\mathrm{y}_i f_m (x_i))$ and renormalize.
* Output the classifier  $sign[F(x)]=sign[\sum_{t=1}^{M}\alpha_{t}f_m(x)]$.


#### LogitBoost

Given a training data set ${\{\mathrm{X}_i, y_i\}}_{i=1}^{N}$, where $\mathrm{X}_i\in\mathbb{R}^p$ is the feature and $y_i\in\{1, 2, \dots, K\}$ is the desired categorical label.
The classifier $F$ learned from data is a function
$$
F:\mathbb{R}^P\to y \\
\quad X_i \mapsto y_i.
$$
 And the function $F$ is usually in the additive model $F(x)=\sum_{m=1}^{M}h(x\mid {\theta}_m)$.

<img title ="logitBoost" src="https://img-blog.csdn.net/20151028220708460" width="80%" />

where $r_{i, k}=1$ if $y_i =k$ otherwise 0.

<img title ="robust logitBoost" src="https://img-blog.csdn.net/20151029111043502" width="80%" />

+ LogitBoost used first and second derivatives to construct the trees;
+ LogitBoost was believed to have numerical instability problems.

- [ ] [Fundamental Techniques in Big Data Data Streams, Trees, Learning, and Search by Li Ping](https://www.stat.rutgers.edu/home/pingli/doc/PingLiTutorial.pdf)
- [ ] [LogitBoost python package](https://logitboost.readthedocs.io/)
- [ ] [ABC-LogitBoost for Multi-Class Classification](http://www.datascienceassn.org/sites/default/files/LogitBoost%20Algorithm.pdf)
- [ ] [LogitBoost学习](https://blog.csdn.net/u014568921/article/details/49474293)
- [ ] [几种Boost算法的比较](https://www.cnblogs.com/jcchen1987/p/4581651.html)
- [ ] [Robust LogitBoost and Adaptive Base Class (ABC) LogitBoost](https://arxiv.org/ftp/arxiv/papers/1203/1203.3491.pdf)

***
- https://arxiv.org/abs/1901.04055
- https://arxiv.org/abs/1901.04065

#### Gradient Boosting Decision Tree

One of the frequently asked questions is `What's the basic idea behind gradient boosting?` and the answer from [https://explained.ai/gradient-boosting/faq.html] is the best one I know:
> Instead of creating a single powerful model, boosting combines multiple simple models into a single **composite model**. The idea is that, as we introduce more and more simple models, the overall model becomes stronger and stronger. In boosting terminology, the simple models are called weak models or weak learners.
> To improve its predictions, gradient boosting looks at the difference between its current approximation,$\hat{y}$ , and the known correct target vector ${y}$, which is called the residual, $y-\hat{y}$. It then trains a weak model that maps feature vector ${x}$  to that residual vector. Adding a residual predicted by a weak model to an existing model's approximation nudges the model towards the correct target. Adding lots of these nudges, improves the overall models approximation.

|Gradient Boosting|
|:---------------:|
|<img title =golf src=https://explained.ai/gradient-boosting/images/golf-MSE.png width=60% />|

It is the first solution to the question that if weak learner is equivalent to strong learner.

***

We may consider the generalized additive model, i.e.,

$$
\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)
$$

where $\{f_k\}_{k=1}^{K}$ is regression decision tree rather than polynomial.
The objective function is given by

$$
obj = \underbrace{\sum_{i=1}^{n} L(y_i,\hat{y}_i)}_{\text{error term} } + \underbrace{\sum_{k=1}^{K} \Omega(f_k)}_{regularazation}
$$

where $\sum_{k=1}^{K} \Omega(f_k)$ is the regular term.

The additive training is to train the regression tree sequentially.
The objective function of the $t$th regression tree is defined as

$$
obj^{(t)} = \sum_{i=1}^{n} L(y_i,\hat{y}^{(t)}_i) + \sum_{k=1}^{t} \Omega(f_k) \\
=  \sum_{i=1}^{n} L(y_i,\hat{y}^{(t-1)}_i + f_t(x_i)) + \Omega(f_t) + C
$$

where C is constant and $C=\sum_{k=1}^{t-1} \Omega(f_k)$.
Particularly, we take $L(x,y)=(x-y)^2$, and the objective function is given by

$$
obj^{(t)}
=  \sum_{i=1}^{n} [y_i - (\hat{y}^{(t-1)}_i + f_t(x_i))]^2 + \Omega(f_t) + C \\
= \sum_{i=1}^{n} [2(y_i - \hat{y}^{(t-1)}_i) f_t(x_i) +  f_t(x_i)^2 ] + \Omega(f_t) + C^{\prime}
$$

where $C^{\prime}=\sum_{i=1}^{n} (y_i - \hat{y}^{(t-1)}_i)^2 + \sum_{k=1}^{t-1} \Omega(f_k)$.

If there is no regular term $\sum_{k=1}^{t} \Omega(f_k)$, the problem is simplified to
$$\arg\min_{f_{t}}\sum_{i=1}^{n} [2(y_i - \hat{y}^{(t-1)}_i) f_t(x_i) +  f_t(x_i)^2 ]\implies f_t(x_i) = -(y_i - \hat{y}^{(t-1)}_i) $$
where $i\in \{1,\cdots, n\}$ and $-(y_i - \hat{y}^{(t-1)}_i)=-\frac{1}{2} {[\frac{\partial L(\mathrm{y}_i, f(x_i))}{\partial f(x_i)}]}_{f=f_{t-1}}$.

***
* Input training data set $\{(x_i, \mathrm{y}_i)\mid i=1, \cdots, n\}$.
* Initialize $f_0(x)=\arg\min_{\gamma} L(\mathrm{y}_i,\gamma)$.
* For $t = 1, 2, \dots, T$:
   +   For $i = 1, 2,\dots , n$ compute
    $$r_{i,t}=-{[\frac{\partial L(\mathrm{y}_i, f(x_i))}{\partial f(x_i)}]}_{f=f_{t-1}}.$$
   +  Fit a regression tree to the targets $r_{i,t}$   giving **terminal regions**
   $$R_{j,m}, j = 1, 2,\dots , J_m. $$
   +  For $j = 1, 2,\dots , J_m$ compute
      $$\gamma_{j,t}=\arg\min_{\gamma}\sum_{x_i\in R_{j,m}}{L(\mathrm{d}_i, f_{t-1}(x_i)+\gamma)}. $$
  +  Update $f_t = f_{t-1}+ \nu{\sum}_{j=1}^{J_m}{\gamma}_{j, t} \mathbb{I}(x\in R_{j, m})$.
* Output $f_T(x)$.

***

An important part of gradient boosting method is regularization by shrinkage which consists in modifying the update rule as follows:
$$
f_{t}
=f_{t-1}+\nu \underbrace{ \sum_{j = 1}^{J_{m}} \gamma_{j, t} \mathbb{I}(x\in R_{j, m}) }_{ \text{ to fit the gradient} }, \\
\approx f_{t-1} + \nu \underbrace{ {\sum}_{i=1}^{n} -{[\frac{\partial L(\mathrm{y}_i, f(x_i))}{\partial f(x_i)}]}_{f=f_{t-1}} }_{ \text{fitted by a regression tree} },
 \nu\in(0,1).
$$

Note that the incremental tree is approximate to the negative gradient of the loss function, i.e.,
$$\fbox{ $\sum_{j=1}^{J_m} \gamma_{j, t} \mathbb{I}(x\in R_{j, m}) \approx {\sum}_{i=1}^{n} -{[\frac{\partial L(\mathrm{y}_i, f(x_i))}{\partial f(x_i)}]}_{f=f_{t-1}}$ }$$
where $J_m$ is the number of the terminal regions and ${n}$ is the number of training samples/data.

* [Greedy Function Approximation: A Gradient Boosting Machine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)
* [Gradient Boosting at Wikipedia](https://www.wikiwand.com/en/Gradient_boosting)
* [Gradient Boosting Explained](https://arogozhnikov.github.io/2016/06/24/gradient_boosting_explained.html)
* [Gradient Boosting Interactive Playground](https://arogozhnikov.github.io/2016/07/05/gradient_boosting_playground.html)
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3885826/
* https://explained.ai/gradient-boosting/index.html
* https://explained.ai/gradient-boosting/L2-loss.html
* https://explained.ai/gradient-boosting/L1-loss.html
* https://explained.ai/gradient-boosting/descent.html
* https://explained.ai/gradient-boosting/faq.html
* [GBDT算法原理 - 飞奔的猫熊的文章 - 知乎](https://zhuanlan.zhihu.com/p/50176849)
* https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf
* [Complete Machine Learning Guide to Parameter Tuning in Gradient Boosting (GBM) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)
* [Gradient Boosting Algorithm – Working and Improvements](https://data-flair.training/blogs/gradient-boosting-algorithm/)


_______________

A general gradient descent “boosting” paradigm is developed for additive expansions based on any fitting criterion. It is not only for the decision tree.

* [Gradient Boosting Machines](http://uc-r.github.io/gbm_regression)
* [Start With Gradient Boosting, Results from Comparing 13 Algorithms on 165 Datasets](https://machinelearningmastery.com/start-with-gradient-boosting/)
* [A Gentle Introduction to the Gradient Boosting Algorithm for Machine Learning](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)

### Bonsai Boosted Decision Tree

AdaBoost is related with so-called expentontial loss $\exp(-{y_i}p(x_i))$ where $x_i\in\mathbb{R}^p, y_i\in\{-1, +1\}, p(\cdot)$ is the input features, labels and prediction function, respectively.
And the weight is update via the following formula:
$$w_i\leftarrow w_i\exp[-y_ip(x_i)], i=1,2,\dots, N.$$

When the lables are multivariate, [Alex Rogozhnikova et al](https://arxiv.org/abs/1410.4140) define a more
general expression of the AdaBoost criteria
$$w_i\leftarrow w_i\exp[-y_i\sum_{j}a_{ij}p(x_j)], i=1,2,\dots, N,$$

where $a_{ij}$ are the elements of some square matrix ${A}$. For the case where A is the identity matrix,
the AdaBoost weighting procedure is recovered.

**bonsai BDT (BBDT)**:

1. $\fbox{discretizes}$ input variables before training which ensures a fast
and robust implementation
2. converts decision trees to n-dimentional table to store
3. prediction operation takes one reading from this table

The first step of preprocessing data limits where the splits of the data can be made and, in effect, permits the grower of the tree to
control and shape its growth; thus, we are calling this a bonsai BDT (BBDT).
The discretization works by enforcing that the smallest keep interval that can be created when training the BBDT is:
$$\Delta x_{min} > \delta_x \forall x\,\,\text{on all leaves}$$
where $\delta_x=\min \{|x_i-x_j|: x_i, x_j\in x_{discrete}\}$.

***
* [Programmable Decision Tree Framework](https://github.com/yubin-park/bonsai-dt)
* [bonsai-dt: Programmable Decision Tree Framework](https://yubin-park.github.io/bonsai-dt/)
* [New approaches for boosting to uniformity](https://arxiv.org/abs/1410.4140)
* [Efficient, reliable and fast high-level triggering using a bonsai boosted decision tree](https://arxiv.org/abs/1210.6861)
* [Boosting bonsai trees for efficient features combination : Application to speaker role identification](https://www.researchgate.net/publication/278798264_Boosting_bonsai_trees_for_efficient_features_combination_Application_to_speaker_role_identification)
* [Bonsai Trees in Your Head: How the Pavlovian System Sculpts Goal-Directed Choices by Pruning Decision Trees](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3297555/)
* [HEM meets machine learning](https://higgsml.lal.in2p3.fr/prizes-and-award/award/)

#### xGBoost

In Gradient Boost, we compute and fit a regression a tree to
$$
r_{i,t}=-{ [\frac{\partial L(\mathrm{d}_i, f(x_i))}{\partial f(x_i)}] }_{f=f_{t-1}}.
$$
Why not the error $L(\mathrm{d}_i, f(x_i))$ itself?
Recall the Taylor expansion
$f(x+h) = f(x)+f^{\prime}(x)h + f^{(2)}(x)h^{2}/2!+ \cdots +f^{(n)}(x)h^{(n)}/n!+\cdots$ so that the non-convex error function can be expressed as a polynomial in terms of $h$,
which is easier to fit than a general common non-convex function.
So that we can implement additive training to boost the supervised algorithm.


In general, we can expand the objective function at $x^{t-1}$ up to  the second order

$$
obj^{(t)}
=  \sum_{i=1}^{n} L[y_i,\hat{y}^{(t-1)}_i + f_t(x_i)] + \Omega(f_t) + C \\
\simeq \sum_{i=1}^{n} \underbrace{ [L(y_i,\hat{y}^{(t-1)}_i) + g_i f_t(x_i) + \frac{h_i f_t^2(x_i)}{2}] }_{\text{2ed Order Taylor Expansion}} + \Omega(f_t) + C^{\prime}
$$

where $\color{red}{ g_i=\partial_{\hat{y}_{i}^{(t-1)}} L(y_i, \hat{y}_{i}^{(t-1)}) }$, $\color{red}{h_i=\partial^2_{\hat{y}_{i}^{(t-1)}} L(y_i, \hat{y}_{i}^{(t-1)})}$.

After we remove all the constants, the specific objective at step ${t}$ becomes
$$
obj^{(t)}\approx \sum_{i=1}^{n} [L(y_i,\hat{y}^{(t-1)}_i) + g_i f_t(x_i) + \frac{h_i f_t^2(x_i)}{2}] + \Omega(f_t)
$$

One important advantage of this definition is that the value of the objective function only depends on $g_i$ and $h_i$. This is how XGBoost supports custom loss functions.

 In order to define the complexity of the tree $\Omega(f)$, let us first refine the definition of the tree $f(x)$ as
$$
f_t(x)= w_{q(x)}={\sum}_{i=1}^{T} w_{i}\mathbb{I} ({q(x)=i}),\\
 w\in\mathbb{R}^{T}, q:\mathbb{R}^d\Rightarrow \{1,2,\dots, T\}.
$$

Here ${w}$ is the vector of scores on leaves, ${q}$ is a function assigning each data point to the corresponding leaf, and ${T}$ is the number of leaves. In XGBoost, we define the complexity as
$$
\Omega(f)=\gamma T + \frac{1}{2}\lambda \sum_{i=1}^{T} {w}_i^2.
$$

After re-formulating the tree model, we can write the objective value with the ${t}$-th tree as:
$$
obj^{(t)} = \sum_{i=1}^{n}[g_i w_{q(x_i)}+\frac{1}{2} h_i w_{q(x_i)}^2 + \gamma T+\frac{1}{2}\lambda \sum_{i=1}^{n}w_i^2]
\\=\sum_{j=1}^{T}[(\sum_{i\in I_{j}}g_i)w_j+\frac{1}{2}(\sum_{i\in I_{j}}h_i + \lambda)w_j^2]+\gamma T
$$
where $I_j=\{i\mid q(x_i)=j\}$ is the set of indices of data points assigned to the $j$-th leaf.
We could further compress the expression by defining $G_j=\sum_{i\in I_j} g_i$ and $H_j=\sum_{i\in I_j} h_i$:
$$
obj^{(t)} = \sum_{j=1}^{T}[(G_j w_j+\frac{1}{2}(H_j +\lambda)w_j^2]+\gamma T.
$$

In this equation, $w_j$ are independent with respect to each other, the form $G_j w_j + \frac{1}{2}(H_j+\lambda)w^2_j$ is quadratic and the best $w_j$ for a given structure $q(x)$ and the best objective reduction we can get is:

$$
w_j^{\ast} =-(H_j+\lambda )^{-1}G_j,\\
obj^{\ast} =\frac{1}{2}\sum_{j=1}^{T}
-(H_j+\lambda )^{-1}G_j^2+\gamma T.
$$

Another key of  xGBoost is how to a construct a tree fast and painlessly.
We will try to optimize _`one level`_ of the tree at a time. Specifically we try to split a leaf into two leaves, and the score it gains is
$$
Gain = \frac{1}{2} \left[\underbrace{\frac{G_L^2}{H_L+\lambda}}_{\text{from left leaf}} + \underbrace{\frac{G_R^2}{H_R+\lambda}}_{\text{from the right leaf}}-\underbrace{\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}}_{\text{from the original leaf} } \right] - \gamma
$$

This formula can be decomposed as 1) the score on the new left leaf 2) the score on the new right leaf 3) The score on the original leaf 4) regularization on the additional leaf. We can see an important fact here: if the gain is smaller than $\gamma$, we would do better not to add that branch. This is exactly the **pruning techniques** in tree based models! By using the principles of supervised learning, we can naturally come up with the reason these techniques work.

<img src="https://pic3.zhimg.com/80/v2-46792243acd6570c3416df14a8d0bb1e_hd.jpg" width="80%" />
<img src="https://pic3.zhimg.com/80/v2-6cd871031772e6ab3005b3166731bae2_hd.jpg" width="80%" />
 Other features include:

 * row sample;
 * column sample;
 * shrinkages.

***
<img title = "Tianqi Chen" src="https://tqchen.com/data/img/tqchen-new.jpg" width="30%" />

* https://xgboost.readthedocs.io/en/latest/tutorials/model.html
* https://xgboost.ai/
* [A Kaggle Master Explains Gradient Boosting](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/)
* [Extreme Gradient Boosting with R](https://datascienceplus.com/extreme-gradient-boosting-with-r/)
* [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
* [xgboost的原理没你想像的那么难](https://www.jianshu.com/p/7467e616f227)
* [一步一步理解GB、GBDT、xgboost](https://www.cnblogs.com/wxquare/p/5541414.html)
* [How to Visualize Gradient Boosting Decision Trees With XGBoost in Python](https://machinelearningmastery.com/visualize-gradient-boosting-decision-trees-xgboost-python/)
* [Awesome XGBoost](https://github.com/dmlc/xgboost/blob/master/demo/README.md#machine-learning-challenge-winning-solutions)
* [Stroy and lessons from xGBoost](https://homes.cs.washington.edu/~tqchen/2016/03/10/story-and-lessons-behind-the-evolution-of-xgboost.html)

<img src=https://pic2.zhimg.com/50/v2-d8191a1191979eadbd4df191b391f917_hd.jpg />

#### LightGBM

LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:

* Faster training speed and higher efficiency.
* Lower memory usage.
* Better accuracy.
* Support of parallel and GPU learning.
* Capable of handling large-scale data.


![lightGBM](http://zhoutao822.coding.me/2019/01/13/LightGBM/2.png)

A major reason is that for each feature, they need to scan all the data instances to estimate the information gain of all
possible split points, which is very time consuming. To tackle this problem, the authors of lightGBM propose two novel techniques: Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB).
<img title="GOSS" src="https://pic3.zhimg.com/80/v2-1608d8b79f3d605e111878b715254d3e_hd.jpg" width="69%" />
<img title="EFB" src="https://pic1.zhimg.com/80/v2-4d7875a088e184c694474be2bec26698_hd.jpg" width="90%" />


`Leaf-wise` learning is to split a leaf(with max delta loss) into 2 leaves rather than split leaves in one level into 2 leaves.

<img src=http://zhoutao822.coding.me/2019/01/13/LightGBM/1.png width=80% />

And it limits the depth of the tree in order to avoid over-fitting.

<img src=http://zhoutao822.coding.me/2019/01/13/LightGBM/7.png width=80% />

`Histogram` is an un-normalized empirical cumulative distribution function, where the continuous features (in flow point data structure) is split into ${k}$ buckets by threahold values such as if $x\in [0, 2)$ then ${x}$ will be split into bucket 1. It really reduces the complexity to store the data and compute the impurities based on the distribution of features.

<img src="http://zhoutao822.coding.me/2019/01/13/LightGBM/5.png" width="100%" />

**Optimization in parallel learning**

[A Communication-Efficient Parallel Algorithm for Decision Tree](https://arxiv.org/abs/1611.01276)

- [LightGBM, Light Gradient Boosting Machine](https://github.com/Microsoft/LightGBM/)
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)
- [Python3机器学习实践：集成学习之LightGBM - AnFany的文章 - 知乎](https://zhuanlan.zhihu.com/p/53583034)
- https://lightgbm.readthedocs.io/en/latest/
- https://www.msra.cn/zh-cn/news/features/lightgbm-20170105
- [LightGBM](http://zhoutao822.coding.me/2019/01/13/LightGBM/)
- [Reference papers of lightGBM](https://lightgbm.readthedocs.io/en/latest/Features.html#references)
- https://lightgbm.readthedocs.io/en/latest/Features.html


#### CatBoost

`CatBoost` is an algorithm for gradient boosting on decision trees. [Developed by Yandex researchers and engineers, it is the successor of the `MatrixNet` algorithm that is widely used within the company for ranking tasks, forecasting and making recommendations. It is universal and can be applied across a wide range of areas and to a variety of problems](https://betapage.co/startup/catboost) such as search, recommendation systems, personal assistant, self-driving cars, weather prediction and many other tasks. It is in open-source and can be used by anyone now.

`CatBoost` is based on gradient boosted decision trees. During training, a set of decision trees is built consecutively. Each successive tree is built with reduced loss compared to the previous trees.

The number of trees is controlled by the starting parameters. To prevent over-fitting, use the over-fitting detector. When it is triggered, trees stop being built.

Before learning, the possible values of objects are divided into disjoint ranges ($\color{red}{\fbox{buckets}}$) delimited by the threshold values ($\color{red}{\fbox{splits}}$). The size of the quantization (the number of splits) is determined by the starting parameters (separately for numerical features and numbers obtained as a result of converting categorical features into numerical features).

Quantization is also used to split the label values when working with categorical features. А random subset of the dataset is used for this purpose on large datasets.

Two critical algorithmic advances introduced in CatBoost are the implementation
of `ordered boosting`, a permutation-driven alternative to the classic algorithm, and
an innovative algorithm for processing `categorical features`. Both techniques were
created to fight a prediction shift caused by a special kind of target leakage present
in all currently existing implementations of gradient boosting algorithms.


|THREE|
|:---:|
|<img src = "https://s2.51cto.com/oss/201808/30/108338cbd6df1a13dd9ed6d14c9da35d.png" width="100%" />|
|<img src = "https://s1.51cto.com/oss/201808/30/e0ac1ddc9b9c0e513e2669f56151edc7.png" width="100%" />|
|<img src = "https://s1.51cto.com/oss/201808/30/4183b4bba0529b55e5f4f1bef8072ab5.png" width="100%" />|


***

![Andrey Gulin](https://hsto.org/webt/59/d5/de/59d5decccf61b358323576.jpeg)

- https://tech.yandex.com/catboost/
- [How is CatBoost? Interviews with developers](https://weekly-geekly.github.io/articles/339384/index.html)
- [Reference papers of CatBoost](https://catboost.ai/docs/concepts/educational-materials-papers.html)
- [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516)
- [Efficient Gradient Boosted Decision Tree Training on GPUs](https://www.comp.nus.edu.sg/~hebs/pub/IPDPS18-GPUGBDT.pdf)
- [CatBoost：比XGBoost更优秀的GBDT算法](http://ai.51cto.com/art/201808/582487.htm)


### More: TencentBoost, ThunderGBM and Beyond

There are more gradient boost tree algorithms such as ThubderGBM, TencentBoost, GBDT on angle and H2o.

[Gradient boosting tree (GBT), a widely used machine learning algorithm, achieves state-of-the-art performance in academia, industry, and data analytics competitions. Although existing scalable systems which implement GBT, such as XGBoost and MLlib, perform well for data sets with medium-dimensional features, they can suffer performance degradation for many industrial applications where the trained data sets contain high dimensional features. The performance degradation derives from their inefficient mechanisms for model aggregation-either map-reduce or all-reduce. To address this high-dimensional problem, we propose a scalable execution plan using the parameter server architecture to facilitate the model aggregation. Further, we introduce a sparse-pull method and an efficient index structure to increase the processing speed. We implement a GBT system, namely TencentBoost, in the production cluster of Tencent Inc. The empirical results show that our system is 2-20× faster than existing platforms.](https://ieeexplore.ieee.org/abstract/document/7929984)

- [ThunderGBM: Fast GBDTs and Random Forests on GPUs](https://github.com/Xtra-Computing/thundergbm)
- [TencentBoost: A Gradient Boosting Tree System with Parameter Server](https://ieeexplore.ieee.org/abstract/document/7929984)
- [GBDT on Angel](https://github.com/Angel-ML/angel/blob/master/docs/algo/gbdt_on_angel.md)
- [Gradient Boosted Categorical Embedding and Numerical Trees](http://www.hongliangjie.com/talks/GB-CENT_MLIS_2017-06-06.pdf)
- [从结构到性能，一文概述XGBoost、Light GBM和CatBoost的同与不同](https://zhuanlan.zhihu.com/p/34698733)
- [从决策树、GBDT到XGBoost/lightGBM/CatBoost](https://zhuanlan.zhihu.com/p/59419786)
- [ThunderGBM：快成一道闪电的梯度提升决策树](https://zhuanlan.zhihu.com/p/58626955)
***

**Optimization and Boosting**


How to combine `ADMM` as an operator splitting methods for numerical optimization and `Boosting` such as gradient boosting/extreme gradient boosting?
Can we do leaves splitting and optimization in the same stage?

The core transfer is how to change the original optimization to one linearly constrained  convex optimization  so that it adjusts to ADMM:  

$$
\arg\min_{f_{t}\in\mathcal F}\sum_{i=1}^{n} L[y_i,\hat{y}^{(t-1)}_i + f_t(x_i)] + \gamma T +\frac{\lambda}{2}{\sum}_{i=1}^{T}w_i^2 \iff \fbox{???} \quad ?
$$
where $f_t(x)={\sum}_{i=1}^{T}w_i\mathbb{I}(q(x)=i)$.

It seems attractive to me to understand the analogy between
$\fbox{operator splitting in ADMM}$ and $\fbox{leaves splitting in Decision Tree}$.

To be more general, how to connect the numerical optimization methods such as fixed pointed iteration methods and the boosting algorithms?
Is it possible to combine $\fbox{Anderson Acceleration}$ and $\fbox{Gradinet Boosting}$ ?  



* [OPTIMIZATION BY GRADIENT BOOSTING](http://www.lsta.upmc.fr/BIAU/bc2.pdf)
* [boosting as optimization](https://metacademy.org/graphs/concepts/boosting_as_optimization)
* [Boosting, Convex Optimization, and Information Geometry](https://ieeexplore.ieee.org/document/6282239?arnumber=6282239)
* [Generalized Boosting Algorithms for Convex Optimization](https://www.ri.cmu.edu/publications/generalized-boosting-algorithms-for-convex-optimization/)
* [Survey of Boosting from an Optimization Perspective](https://users.soe.ucsc.edu/~manfred/pubs/tut/icml2009/ws.pdf)
* [Accelerated Gradient Boosting](https://arxiv.org/abs/1803.02042)

______________
Boosting | Optimziation
---|---
AdaBoost | Coordinate-wise Descent
Stochastic Gradient Boost | Stochastic Gradient Descent
Gradient Boost |  Gradient Descent
Accelerated Gradient Boosting | Accelerated Gradient Descent

**Accelerated Gradient Boosting**

The difficulty in accelerating GBM lies in the fact that weak (inexact) learners are commonly used, and therefore the errors can accumulate in the momentum term. To overcome it, we design a "corrected pseudo residual" and fit best weak learner to this corrected pseudo residual, in order to perform the z-update. Thus, we are able to derive novel computational guarantees for AGBM. This is the first GBM type of algorithm with theoretically-justified accelerated convergence rate.

* Initialize $f_0(x)=g_0(x)=0$;
* For $m = 1, 2, \dots, M$:
   +  Compute a linear combination of ${f}$ and ${h}$: $g^{m}(x)=(1-{\theta}_m) f^m(x) + {\theta}_m h^m(x)$ and ${\theta}_m=\frac{2}{m+2}$
   +  For $i = 1, 2,\dots , n$ compute
    $$r_{i, m}=-{[\frac{\partial L(\mathrm{y}_i, g^m(x_i))}{\partial g^m(x_i)}]}.$$
   +  Find the best weak-learner for pseudo residual:
   $${\tau}_{m,1}=\arg\min_{\tau\in \mathcal T}{\sum}_{i=1}^{n}(r_{i,m}-b_{\tau}(x_i))^2$$
  +  Update the model: $f^{m+1}(x)= g^{m}(x) + \eta b_{\tau_{m,1}}$.
  +  Update the corrected residual:
  $$c_{i,m}=\begin{cases} r_{i, m} & \text{if m=0},\\ r_{i, m}+\frac{m+1}{m+2}(c_{i, m-1}-b_{\tau_{m,2}}) & \text{otherwise}.\end{cases}$$
  +  Find the best weak-learner for the corrected residual: $b_{\tau_{m,2}}=\arg\min_{\tau\in \mathcal T}{\sum}_{i=1}^{n}(c_{i,m}-b_{\tau}(x_i))^2$.
  +  Update the momentum model: $h^{m+1} = h^{m} + \frac{\gamma\eta}{\theta_m} b_{\tau_{m,2}}(x)$.
* Output $f^{M}(x)$.


#### Application

* [拍拍贷教你如何用GBDT做评分卡](http://www.sfinst.com/?p=1389)
* [LambdaMART 不太简短之介绍](https://liam.page/2016/07/10/a-not-so-simple-introduction-to-lambdamart/)
* https://catboost.ai/news
* [Finding Influential Training Samples for Gradient Boosted Decision Trees](https://research.yandex.com/publications/151)
+ [Efficient, reliable and fast high-level triggering using a bonsai boosted decision tree](http://inspirehep.net/record/1193348)
+ [CERN boosts its search for antimatter with Yandex’s MatrixNet search engine tech](https://www.extremetech.com/extreme/147320-cern-boosts-its-search-for-antimatter-with-yandexs-matrixnet-search-engine-tech)
+ [MatrixNet as a specific Boosted Decision Tree algorithm which is available as a service](https://github.com/yandex/rep/blob/master/rep/estimators/matrixnet.py)

### Stacking

Stacked generalization (or stacking)  is a different way of combining multiple models, that introduces the concept of a meta learner. Although an attractive idea, it is less widely used than bagging and boosting. Unlike bagging and boosting, stacking may be (and normally is) used to combine models of different types.

The procedure is as follows:

1. Split the training set into two disjoint sets.
2. Train several base learners on the first part.
3. Test the base learners on the second part.
4. Using the predictions from 3) as the inputs, and the correct responses as the outputs, train a higher level learner.

[Note that steps 1) to 3) are the same as cross-validation, but instead of using a winner-takes-all approach, we train a meta-learner to combine the base learners, possibly non-linearly.](http://www.machine-learning.martinsewell.com/ensembles/stacking/) It is a little similar with **composition** of functions in mathematics.

<img src="https://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier_files/stackingclassification_overview.png" width="65%" />

[Stacking, Blending and and Stacked Generalization are all the same thing with different names. It is a kind of ensemble learning.](http://www.chioka.in/stacking-blending-and-stacked-generalization/)

* http://www.machine-learning.martinsewell.com/ensembles/stacking/
* https://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/
* [Stacked Generalization (Stacking)](http://www.machine-learning.martinsewell.com/ensembles/stacking/)
* [Stacking与神经网络 - 微调的文章 - 知乎](https://zhuanlan.zhihu.com/p/32896968)
* http://www.chioka.in/stacking-blending-and-stacked-generalization/
* https://blog.csdn.net/willduan1/article/details/73618677
* [今我来思，堆栈泛化(Stacked Generalization)](https://www.jianshu.com/p/46ccf40222d6)
* [我爱机器学习:集成学习（一）模型融合与Bagging](https://www.hrwhisper.me/machine-learning-model-ensemble-and-bagging/)

In the sense of stacking, deep neural network is thought as the stacked `logistic regression`. And `Boltzman machine` can be stacked in order to construct more expressive model for discrete random variables.

**Deep Forest**

![Deep Forest](https://raw.githubusercontent.com/DataXujing/Cos_pic/master/pic2.png)

* [Deep forest](http://lamda.nju.edu.cn/code_gcForest.ashx?AspxAutoDetectCookieSupport=1)
* https://github.com/kingfengji/gcForest
* https://zhuanlan.zhihu.com/p/37492203
* https://arxiv.org/abs/1806.00007
* https://arxiv.org/abs/1502.00712
* http://adsabs.harvard.edu/abs/2015arXiv150200712P
* https://cosx.org/2018/10/python-and-r-implementation-of-gcforest/



**Deep Learning Ensemble**

Deep learning and ensemble learning share some similar guide line.

![snapshot](http://ruder.io/content/images/2017/11/snapshot_ensembles.png)


- [ ] [Neural Network Ensembles](https://www.computer.org/csdl/journal/tp/1990/10/i0993/13rRUyv53Gg)
- [ ] [Ensemble Learning Methods for Deep Learning Neural Networks](https://machinelearningmastery.com/ensemble-methods-for-deep-learning-neural-networks/)
- [ ] http://ruder.io/deep-learning-optimization-2017/
- [ ] https://arxiv.org/abs/1704.00109v1
- [ ] http://jtleek.com/advdatasci/17-blending.html
- [ ] https://arxiv.org/abs/1708.03704
- [ ] [Ensemble Deep Learning for Speech Recognition](https://www.isca-speech.org/archive/archive_papers/interspeech_2014/i14_1915.pdf)


****

Other ensemble methods include clustering methods ensemble, dimensionality reduction ensemble, regression ensemble, ranking ensemble.

*****
* https://www.wikiwand.com/en/Ensemble_learning
* https://www.toptal.com/machine-learning/ensemble-methods-machine-learning
* https://machinelearningmastery.com/products/
* https://blog.csdn.net/willduan1/article/details/73618677#
* http://www.scholarpedia.org/article/Ensemble_learning
* https://arxiv.org/abs/1505.01866
