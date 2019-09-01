## Tree-based Learning Algorithms

The [simple function](https://proofwiki.org/wiki/Definition:Simple_Function) is a real-valued  function $f: \mathrm{X}\to \mathbb{R}$ if and only if it is a finite linear combination of characteristic functions:
$$f=\sum_{i=1}^{n}a_k {\chi}_{S_{k}}$$
where $a_k\in\mathbb{R}$ and the characteristic function is defined as follow
$${\chi}_{S_{k}}=\begin{cases}1, &\text{if $x \in S_{k}$}\\
0, &\text{otherwise}\end{cases}.$$

* [The Simple Function Approximation Lemma](http://mathonline.wikidot.com/the-simple-function-approximation-lemma)

The tree-based learning algorithms take advantages of these [universal approximators](http://mathonline.wikidot.com/the-simple-function-approximation-theorem) to fit the decision function.

<img title="https://cdn.stocksnap.io/" src="https://cdn.stocksnap.io/img-thumbs/960w/TIHPAM0QFG.jpg" width="80%" />

The core problem is to find the optimal parameters $a_k\in\mathbb{R}$ and the region $S_k\in\mathbb{R}^p$  when only some finite sample or training data $\{(\mathrm{x}_i, y_i)\mid i=1, 2, \dots, n\}$ is accessible or available where $\mathrm{x}_i\in\mathbb{R}^p$ and $y_i\in\mathbb{R}$ or some categorical domain and the number of regions also depends on the training data set.

### Decision Tree

A decision tree is a set of questions(i.e. if-then sentence) organized in a **hierarchical** manner and represented graphically as a tree.
It use `divide-and-conquer` strategy recursively as similar as the `binary search` in the sorting problem. It is easy to scale up to massive data set. The models are obtained by recursively partitioning
the data space and fitting a simple prediction model within each partition. As a result, the partitioning can be represented graphically as a decision tree.
[Visual introduction to machine learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/) show an visual introduction to decision tree.

In brief, A decision tree is a classifier expressed as a recursive partition of the instance space as a nonparametric statistical method.

[Fifty Years of Classification and Regression Trees](http://www.stat.wisc.edu/~loh/treeprogs/guide/LohISI14.pdf) and [the website of Wei-Yin Loh](http://www.stat.wisc.edu/~loh/guide.html) helps much understand the development of  decision tree methods.
Multivariate Adaptive Regression
Splines(MARS) is the boosting ensemble methods for decision tree algorithms.
`Recursive partition` is a recursive  way to construct decision tree.


***

* [An Introduction to Recursive Partitioning: Rationale, Application and Characteristics of Classification and Regression Trees, Bagging and Random Forests](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2927982/)
* [GUIDE Classification and Regression Trees and Forests (version 31.0)](http://www.stat.wisc.edu/~loh/guide.html)
* [Interpretable Machine Learning: Decision Tree](https://christophm.github.io/interpretable-ml-book/tree.html)
* [Tree-based Models](https://dinh-hung-tu.github.io/tree-based-models/)
* [Decision Trees and Evolutionary Programming](http://ai-depot.com/Tutorial/DecisionTrees-Partitioning.html)
* [Repeated split sample validation to assess logistic regression and recursive partitioning: an application to the prediction of cognitive impairment.](https://www.ncbi.nlm.nih.gov/pubmed/16149128)
* [A comparison of regression trees, logistic regression, generalized additive models, and multivariate adaptive regression splines for predicting AMI mortality.](https://www.ncbi.nlm.nih.gov/pubmed/17186501)
* http://www.cnblogs.com/en-heng/p/5035945.html
* [高效决策树算法系列笔记](https://github.com/wepe/efficient-decision-tree-notes)
* https://scikit-learn.org/stable/modules/tree.html
* https://github.com/SilverDecisions/SilverDecisions

#### A Visual and Interactive Guide

Decision tree is represented graphically as a tree as the following.

<img src="https://www.dataversity.net/wp-content/uploads/2015/07/3049155-inline-i-1-machine-learning-is-just-a-big-game-of-plinko.gif" width="60%" />

As shown above, there are differences between the length from root to  the terminal nodes, which the inputs arrive at. In another  word, some inputs take  more tests(pass more nodes) than others.

<img src="https://computing.llnl.gov/projects/sapphire/dtrees/pol.a.gif" width="40%"/>

Divisive Hierarchical Clustering | Decision Tree
----|----
Unsupervised | Supervised
Clustering | Classification and Regression



* https://flowingdata.com/
* https://github.com/parrt/dtreeviz
* https://narrative-flow.github.io/exploratory-study-2/
* https://modeloriented.github.io/randomForestExplainer/
* [A Visual Introduction to Machine Learning](https://www.dataversity.net/a-visual-introduction-to-machine-learning/)
* [How to visualize decision trees by Terence Parr and Prince Grover](https://explained.ai/decision-tree-viz/index.html)
* [A visual introduction to machine learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
* [Interactive demonstrations for ML courses, Apr 28, 2016 by  Alex Rogozhnikov](https://arogozhnikov.github.io/2016/04/28/demonstrations-for-ml-courses.html)
* [Can Gradient Boosting Learn Simple Arithmetic?](http://mariofilho.com/can-gradient-boosting-learn-simple-arithmetic/)
* [Viusal Random Forest](http://www.rhaensch.de/vrf.html)

#### Tree Construction

> A decision tree is the function $T :\mathbb{R}^d \to \mathbb{R}$ resulting from a learning algorithm applied on training data lying in input space $\mathbb{R}^d$ , which always has the following form:
 $$
 T(x) = \sum_{i\in\text{leaves}} g_i(x)\mathbb{I}(x\in R_i) = \sum_{i\in \,\text{leaves}} g_i(x) \prod_{a\in\,\text{ancestors(i)}} \mathbb{I}(S_{a (x)}=c_{a,i})
 $$
> where $R_i \subset \mathbb{R}^d$ is the region associated with leaf ${i}$ of the tree, $\text{ancestors(i)}$ is the set of ancestors of leaf node $i$, $c_{a,i}$ is the child of node a on the path from $a$ to leaf $i$, and **$S_a$ is the n-array split function at node $a$**.
> $g_i(\cdot)$ is the decision function associated with leaf $i$ and
> is learned only from training examples in $R_i$.

The $g_{i}(x)$ can be a constant in $\mathbb{R}$ or some mathematical expression such as logistic regression. When $g_i(x)$ is constant, the decision tree is actually piecewise constant, a concrete example of simple function.

The interpretation is simple: Starting from the root node, you go to the next nodes and the edges tell you which subsets you are looking at. Once you reach the leaf node, the node tells you the predicted outcome. All the edges are connected by ‘AND’.

`Template: If feature x is [smaller/bigger] than threshold c AND … then the predicted outcome is the mean value of y of the instances in that node.`

* [Decision Trees (for Classification) by Willkommen auf meinen Webseiten.](http://christianherta.de/lehre/dataScience/machineLearning/decision-trees.php)
* [DECISION TREES DO NOT GENERALIZE TO NEW VARIATIONS](https://www.iro.umontreal.ca/~lisa/pointeurs/bengio+al-decisiontrees-2010.pdf)
* [On the Boosting Ability of Top-Down Decision Tree Learning Algorithms](http://www.columbia.edu/~aec2163/NonFlash/Papers/Boosting2016.pdf)
* [Improving Stability of Decision Trees ](http://www.cs.cmu.edu/~einat/Stability.pdf)
* [ADAPTIVE CONCENTRATION OF REGRESSION TREES, WITH APPLICATION TO RANDOM FORESTS](https://arxiv.org/pdf/1503.06388.pdf)

What is the parameters to learn when constructing a decision tree?
The value of leaves $g_i(\cdot)$ and the spliiting function $S_i(\cdot)$.
Another approach to depict a decsion tree $T_h = (N_h;L_h)$ is given the set of  internal nodes, $N_h$, and  the set of leaves, $L_h$.

***
**Algorithm**  Pseudocode for tree construction by exhaustive search

1. Start at root node.
2. For each node ${X}$, find the set ${S}$ that **minimizes** the sum of the node $\fbox{impurities}$ in the two child nodes and choose the split $\{X^{\ast}\in S^{\ast}\}$ that gives the minimum overall $X$ and $S$.
3. If a stopping criterion is reached, exit. Otherwise, apply step 2 to each child node in turn.

***

Creating a binary decision tree is actually a process of dividing up the input space according to the sum of **impurities**, which is different from other learning mthods such as support vector machine.

This learning process is to minimize the impurities.
C4.5 and CART6 are two later classification tree algorithms that follow this approach. C4.5 uses `entropy` for its impurity function,
whereas CART uses a generalization of the binomial variance called the `Gini index`.

If the training set $D$ is divided into subsets $D_1,\dots,D_k$, the entropy may be
reduced, and the amount of the reduction is the information gain,

$$
G(D; D_1, \dots, D_k)=Ent(D)-\sum_{i=1}^{k}\frac{|D_k|}{|D|}Ent(D_k)
$$

where $Ent(D)$, the entropy of $D$, is defined as

$$
Ent(D)=\sum_{y \in Y} P(y|D)\log(\frac{1}{P(y | D)}),
$$
where $y\in Y$ is the class index.

The information gain ratio of features $A$ with respect of data set $D$  is defined as

$$
g_{R}(D,A)=\frac{G(D,A)}{Ent(D)}.
$$
And another option of impurity is Gini index of probability $p$:

$$
Gini(p)=\sum_{y}p_y (1-p_y)=1-\sum_{y}p_y^2.
$$

$\color{red}{\text{PS: all above impurities}}$  are based on the probability $\fbox{distribuion}$  of data.
So that it is necessary to estimate the probability distribution of each attribute.

#### Pruning and Regularization

Like other supervised algorithms, decision tree makes a trade-off between over-fitting and under-fitting and how to choose the hyper-parameters of decision tree such as the max depth?
The regularization techniques in regression may not suit the tree algorithms such as LASSO.

**Pruning** is a regularization technique for tree-based algorithm. In arboriculture, the reason to prune tree is [because each cut has the potential to change the growth of the tree, no branch should be removed without a reason. Common reasons for pruning are to remove dead branches, to improve form, and to reduce risk. Trees may also be pruned to increase light and air penetration to the inside of the tree’s crown or to the landscape below. ](https://www.treesaregood.org/treeowner/pruningyourtrees)

<img title = "pruning" src="https://www.treesaregood.org/portals/0/images/treeowner/pruning1.jpg" width="40%" />

In machine learning, it is to avoid the overfitting, to make a balance between over-fitting and under-fitting and boost the generalization ability. The important step of tree pruning is to define a criterion be used to determine the correct final tree size using one of the following methods:

1. Use a distinct dataset from the training set (called validation set), to evaluate the effect of post-pruning nodes from the tree.
2. Build the tree by using the training set, then apply a statistical test to estimate whether pruning or expanding a particular node is likely to produce an improvement beyond the training set.
    * Error estimation
    * Significance testing (e.g., Chi-square test)
3. Minimum Description Length principle : Use an explicit measure of the complexity for encoding the training set and the decision tree, stopping growth of the tree when this encoding size (size(tree) + size(misclassifications(tree)) is minimized.

* [Decision Tree - Overfitting saedsayad](https://www.saedsayad.com/decision_tree_overfitting.htm)
* [Decision Tree Pruning based on Confidence Intervals (as in C4.5)](http://www.cs.bc.edu/~alvarez/ML/statPruning.html)

***

When the height of a decision tree is limited to 1, i.e., it takes only one
test to make every prediction, the tree is called a decision stump.
While decision trees are nonlinear classifiers in general, decision stumps are a kind of linear classifiers.

It is also useful to restrict the number of terminal nodes, the height/depth of the decision tree in order to avoid overfitting.



#### Missing values processing

[Assuming the features are missing completely at random, there are a number of ways of handling missing data](https://koalaverse.github.io/machine-learning-in-R/):

1. Discard observations with any missing values.
2. Rely on the learning algorithm to deal with missing values in its training phase.
3. Impute all missing values before training.

For most learning methods, the imputation approach (3) is necessary. The simplest tactic is to impute the missing value with the mean or median of the nonmissing values for that feature. If the features have at least some moderate degree of dependence, one can do better by estimating a predictive model for each feature given the other features and then imputing each missing value by its prediction from the model.

Some software packages handle missing data automatically, although many don’t, so it’s important to be aware if any pre-processing is required by the user.

- [Missing values processing in CatBoost Packages](https://catboost.ai/docs/concepts/algorithm-missing-values-processing.html)
- [Decision Tree: Review of Techniques for Missing Values at Training, Testing and Compatibility](http://uksim.info/aims2015/CD/data/8675a122.pdf)
- http://oro.open.ac.uk/22531/1/decision_trees.pdf
- https://courses.cs.washington.edu/courses/cse416/18sp/slides/S6_missing-data-annotated.pdf
- [Handling Missing Values when Applying Classification Models](http://pages.stern.nyu.edu/~fprovost/Papers/missing.pdf)
- [CLASSIFICATION AND REGRESSION TREES AND FORESTS FOR INCOMPLETE DATA FROM SAMPLE SURVEYS](http://pages.stat.wisc.edu/~loh/treeprogs/guide/LECL19.pdf)

#### Regression Trees

Starting with all of the data, consider a splitting variable $j$ and
split point $s$, and define the pair of half-planes
$$R_1(j, s)=\{X\mid X_j\leq s\}, R_2(j, s)=\{X\mid X_j> s\}.$$

Then we seek the splitting variable $j$ and split point $s$ that solve
$$\min_{j, s}[\min_{c_1}\sum_{x_i\in R_1}(y_i-c_1)^2+\min_{c_2}\sum_{x_i\in R_2}(y_i-c_2)^2].$$

For any choice $j$ and $s$, the inner minimization is solved by
$$\hat{c}_{1}=\operatorname{ave}\left(y_{i} | x_{i} \in R_{1}(j, s)\right) \text { and } \hat{c}_{2}=\operatorname{ave}\left(y_{i} | x_{i} \in R_{2}(j, s)\right).$$
For each splitting variable, the determination of the split point $s$ can
be done very quickly and hence by scanning through all of the inputs, determination of the best pair $\left(j, s\right)$ is feasible.
Having found the best split, we partition the data into the two resulting regions and repeat the splitting process on each of the two regions.
Then this process is repeated on all of the resulting regions.

Tree size is a tuning parameter governing the model’s complexity, and the optimal tree size should be adaptively chosen from the data.
One approach would be to split tree nodes only if the decrease in sum-of-squares due to the split exceeds some threshold.
This strategy is too short-sighted, however, since a seemingly worthless split might lead to a very good split below it.

The preferred strategy is to grow a large tree $T_0$ , stopping the splitting process only when some minimum node size (say 5) is reached.
Then this large tree is pruned using cost-complexity pruning.

we define the cost complexity criterion
$$C_{\alpha}(T)=\sum_{m=1}^{|T|}N_m Q_m (T) + \alpha |T|.$$

The idea is to find, for each $\alpha$, the subtree $T_{\alpha}\subset T_0$ to minimize $C_{\alpha}(T)$.
The tuning parameter $\alpha \geq 0$ governs the tradeoff between tree size and its goodness of fit to the data.
Large values of $\alpha$ result in smaller trees $T_{\alpha}$, and conversely for smaller values of $\alpha$.
As the notation suggests, with $\alpha = 0$ the
solution is the full tree $T_0$.

* [Tutorial on Regression Tree Methods for Precision Medicine and Tutorial on Medical Product Safety: Biological Models and Statistical Methods](http://ims.nus.edu.sg/events/2017/quan/tut.php)
* [ADAPTIVE CONCENTRATION OF REGRESSION TREES, WITH APPLICATION TO RANDOM FORESTS](https://arxiv.org/pdf/1503.06388.pdf)
* [REGRESSION TREES FOR LONGITUDINAL AND MULTIRESPONSE DATA](http://pages.stat.wisc.edu/~loh/treeprogs/guide/AOAS596.pdf)
* [REGRESSION TREE MODELS FOR DESIGNED EXPERIMENTS](http://pages.stat.wisc.edu/~loh/treeprogs/guide/dox.pdf)


#### Classification Trees

If the target is a classification outcome taking values $1,2,\dots,K$, the only
changes needed in the tree algorithm pertain to `the criteria for splitting` nodes and pruning the tree.

[It turns out that most popular splitting criteria can be derived from thinking of decision trees as greedily learning a **piecewise-constant, expected-loss-minimizing approximation** to the function $f(X)=P(Y=1|X)$. For instance, the split that maximizes information gain is also the split that produces the piecewise-constant model that maximizes the expected log-likelihood of the data. Similarly, the split that minimizes the Gini node impurity criterion is the one that minimizes the Brier score of the resulting model. Variance reduction corresponds to the model that minimizes mean squared error.](https://www.benkuhn.net/tree-imp)

Algorithm | Splitting Criteria | Loss Function
---|---|---
ID3| Information Gain
C4.5| Normalized information gain ratio
CART| Gini Index

If your splitting criterion is information gain, this corresponds to a log-likelihood loss function. This works as follows.

If you have a constant approximation $\hat{f}$ to $f$ on some regions $S$, then the approximation that maximizes the expected log-likelihood of the data (that is, the probability of seeing the data if your approximation is correct)is
$$L(\hat f) = E(\log P(Y=Y_{observed} | X, f = \hat f)) = \sum_{X_i} Y_i \log \hat f(X_i) + (1 - Y_i) \log (1 - \hat f(X_i))$$
for a binary classification problem where $Y_i \in \{0, 1\}$ is  its classification.
Here Suppose $Y$ is determined from $X$ by some function $f(X)=P(Y=1|X)$.
First we need to find the constant value $\hat f(X)=f$ that maximizes this value.

Suppose that you have $n$ total instances, $p$ of them positive ($Y=1$) and the rest negative. Suppose that you predict some arbitrary probability $f$ – we’ll solve for the one that maximizes expected log-likelihood. So we take the expected log-likelihood $\mathbb E_X(logP(Y=Y_{observed}|X))$, and break up the expectation by the value of $Y_{observed}$:

$$L(\hat f)
= (\log P(Y=Y_{observed} | X, Y_{observed}=1)) P(Y_{observed} = 1) \\
+ (\log P(Y=Y_{observed} | X, Y_{observed}=0)) P(Y_{observed} = 0)$$

Substituting in some variables gives
$$L(\hat f)
= \frac{p}{n} \log f
+ \frac{n - p}{n} \log (1 - f).$$

And  $\arg\max_{f}L(\hat f)=\frac{p}{n}$ by setting its derivative 0.
Let’s substitute this back into the likelihood formula and shuffle some variables around:
$$L(\hat f) = \left(f \log f + (1 - f) \log (1 - f) \right).$$
***
A similar derivation shows that Gini impurity corresponds to a Brier score loss function. The Brier score for a candidate model
$$B(\hat f) = E((Y - \hat f(X))^2).$$

Like log-likelihood, the predictions that f^ should make to minimize the Brier score are simply $f=p/n$.

Now let’s take the expected Brier score and break up by $Y$, like we did before:
$$B(\hat f)
= (1 - \hat f(X))^2 P(Y_{observed} = 1)
+ \hat f(X)^2 P(Y_{observed} = 0)$$
Plugging in some values:
$$B(\hat f) = (1 - f)^2 f + f^2(1 - f) = f(1-f)$$
which is exactly (proportional to) the Gini impurity in the 2-class setting. (A similar result holds for multiclass learning as well.)
***
---|---
---|---
QUEST|?
Oblivious Decision Trees|?
Online Adaptive Decision Trees|?
Lazy Tree|?
Option Tree|?
Oblique Decision Trees|?
MARS|?


* [Building Classification Models: id3-c45](https://cis.temple.edu/~giorgio/cis587/readings/id3-c45.html)
* [Data Mining Tools See5 and C5.0](https://www.rulequest.com/see5-info.html)
* [A useful view of decision trees](https://www.benkuhn.net/tree-imp)
* https://www.wikiwand.com/en/Decision_tree_learning
* https://www.wikiwand.com/en/Decision_tree
* https://www.wikiwand.com/en/Recursive_partitioning
* [TimeSleuth is an open source software tool for generating temporal rules from sequential data](http://timesleuth-rule.sourceforge.net/)

##### Oblique Decision Trees

In this paper, we consider that the instances take the form $(x_1, x_2, \cdots, x_d, c_j)$ , where the $x_i$ are real-valued attributes, and the $c_j$ is a discrete value that represents the class label of the instance. Most tree inducers consider tests of the form $x_i > k$ that are equivalent to axis-parallel hyperplanes in the attribute space. The task of the inducer is to find appropriate values for $i$ and $k$. Oblique decision trees consider more general tests of the form
$$\sum_{i=1}^d \alpha_i x_i +\alpha_{d+1}>0$$
where the $\alpha_{d+1}$ are real-valued coefficients

In a compact way, the general linear test can be rewriiten as
$$\left<\alpha, x\right>+b>0$$
where $\alpha=(\alpha_1,\cdots, \alpha_d)^T$ and $x=(x_1, x_2, \cdots, x_d)$.

<img src="https://computing.llnl.gov/projects/sapphire/dtrees/pol.o.gif" width="50%"/>

- https://computing.llnl.gov/projects/sapphire/dtrees/oc1.html
- [Decision Forests with Oblique Decision Trees](http://users.monash.edu/~dld/Publications/2006/TanDoweMICAI2006_final.pdf)
- [Global Induction of Oblique Decision Trees: An Evolutionary Approach](https://www.cs.kent.ac.uk/people/staff/mg483/documents/kr05iis.pdf)
- [On Oblique Random Forests](http://people.csail.mit.edu/menze/papers/menze_11_oblique.pdf)

It is natural to generalized to nonlinear test, which can be seen as feature engineering of the input data.


#### Classification and Regression Tree

[Classification and regression trees (CART) are a non-parametric decision tree learning technique that produces either classification or regression trees, depending on whether the dependent variable is categorical or numeric, respectively. CART is both a generic term to describe tree algorithms and also a specific name for Breiman’s original algorithm for constructing classification and regression trees.](https://koalaverse.github.io/machine-learning-in-R/decision-trees.html)

* [Classification and Regression Tree Methods(In Encyclopedia of Statistics in Quality and Reliability)](http://pages.stat.wisc.edu/~loh/treeprogs/guide/eqr.pdf)
* [Classification And Regression Trees for Machine Learning](https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/)
* [Classification and regression trees](http://pages.stat.wisc.edu/~loh/treeprogs/guide/wires11.pdf)
* http://homepages.uc.edu/~lis6/Teaching/ML19Spring/Lab/lab8_tree.html
* [CLASSIFICATION AND REGRESSION TREES AND FORESTS FOR INCOMPLETE DATA FROM SAMPLE SURVEYS](http://pages.stat.wisc.edu/~loh/treeprogs/guide/LECL19.pdf)
* [Classification and Regression Tree Approach for Prediction of Potential Hazards of Urban Airborne Bacteria during Asian Dust Events](https://www.nature.com/articles/s41598-018-29796-7)

### VFDT and Beyond

[`Hoeffding Tree` or `VFDT` is the standard decision tree algorithm for data stream classification. VFDT uses the Hoeffding bound to decide the minimum number of arriving instances to achieve certain level of confidence in splitting the node. This confidence level determines how close the statistics between the attribute chosen by VFDT and the attribute chosen by decision tree for batch learning.](https://samoa.incubator.apache.org/documentation/Vertical-Hoeffding-Tree-Classifier.html)

[`Hoeffding Anytime Tree` produces the asymptotic batch tree in the limit, is naturally resilient to concept drift, and can be used as a higher accuracy replacement for Hoeffding Tree in most scenarios, at a small additional computational cost.](https://arxiv.org/pdf/1802.08780.pdf)

[Although exceedingly simple conceptually, most implementations of tree-based models do not efficiently utilize `modern superscalar processors`. By laying out data structures in memory in a more cache-conscious fashion, removing branches from the execution flow using a technique called predication, and micro-batching predictions using a technique called vectorization, we are able to better exploit modern processor architectures. ](https://cs.uwaterloo.ca/~jimmylin/publications/Asadi_etal_TKDE2014.pdf)

* [Very Fast Decision Tree (VFDT) classifier](https://samoa.incubator.apache.org/documentation/Vertical-Hoeffding-Tree-Classifier.html)
* [Runtime Optimizations for Tree-based Machine Learning Models](https://cs.uwaterloo.ca/~jimmylin/publications/Asadi_etal_TKDE2014.pdf)
* [Extremely Fast Decision Tree](https://arxiv.org/abs/1802.08780)
* [Optimized very fast decision tree with balanced classification accuracy and compact tree size](https://ieeexplore.ieee.org/abstract/document/6108399)
* [Distributed Decision Trees with Heterogeneous Parallelism](https://raypeng.github.io/DGBDT/)
* [基于特征预排序的算法SLIQ](https://github.com/wepe/efficient-decision-tree-notes/blob/master/SLIQ.md)
* [基于特征预排序的算法SPRINT](https://github.com/wepe/efficient-decision-tree-notes/blob/master/SPRINT.md)
* [基于特征离散化的算法ClOUDS](https://github.com/wepe/efficient-decision-tree-notes/blob/master/ClOUDS.md)

### Decision Stream

[Decision stream is a statistic-based supervised learning technique that generates a deep directed acyclic graph of decision rules to solve classification and regression tasks. This decision tree based method avoids the problem of data exhaustion in terminal nodes by merging of leaves from the same/different levels of predictive model.](https://metacademy.org/roadmaps/Prof.Kee/Decision_Stream)

Unlike the classical decision tree approach, this method builds a predictive model with high degree of connectivity by merging statistically indistinguishable nodes at each iteration. The key advantage of decision stream is an efficient usage of every node, taking into account all fruitful feature splits. With the same quantity of nodes, it provides higher depth than decision tree, splitting and merging the data multiple times with different features. The predictive model is growing till no improvements are achievable, considering different data recombinations, and resulting in deep directed acyclic graph, where decision branches are loosely split and merged like natural streams of a waterfall. Decision stream supports generation of extremely deep graph that can consist of hundreds of levels.

- https://metacademy.org/roadmaps/Prof.Kee/Decision_Stream
- https://arxiv.org/pdf/1704.07657.pdf

##### Oblivious Decision Trees

- https://www.ijcai.org/Proceedings/95-2/Papers/008.pdf
- http://www.aaai.org/Papers/Workshops/1994/WS-94-01/WS94-01-020.pdf

#### Decision Graph

- [Decision Graphs : An Extension of Decision Trees](https://pdfs.semanticscholar.org/73f1/d17df0e1232da9e2331878a802a941f351c6.pdf)

### Random Forest

[Decision Trees do not generalize to new variations](https://www.iro.umontreal.ca/~lisa/pointeurs/bengio+al-decisiontrees-2010.pdf) demonstrates some theoretical limitations of decision trees. And they can be seriously hurt by the curse of dimensionality in a sense that is a bit different
from other nonparametric statistical methods, but most importantly, that they cannot generalize to variations not seen in the training set.
This is because a decision tree creates a partition of the input space and needs at least one example in each of the regions associated with a leaf to make a sensible prediction in that region.
A better understanding of the fundamental reasons for this limitation suggests that one should use forests or even deeper architectures instead of trees,
which provide a form of distributed representation and can generalize to variations not encountered in the training data.

[Random forests (Breiman, 2001)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) is a substantial modification of bagging
that builds a large collection of de-correlated trees, and then averages them.


On many problems the performance of random forests is very similar to boosting, and they are simpler to train and tune.

- [ RANDOM FORESTS by Leo Breiman](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)

***

* For $t=1, 2, \dots, T$:
   + Draw a bootstrap sample $Z^{\ast}$ of size $N$ from the training data.
   + Grow a random-forest tree $T_t$ to the bootstrapped data, by recursively repeating the following steps for each terminal node of the tree, until the minimum node size $n_{min}$ is reached.
     - Select $m$ variables at random from the $p$ variables.
     - Pick the best variable/split-point among the $m$.
     - Split the node into two daughter nodes.
* Vote for classification and average for regression.

<img src="https://dimensionless.in/wp-content/uploads/RandomForest_blog_files/figure-html/voting.png" width="80%" />

|[properties of random forest](https://www.elderresearch.com/blog/modeling-with-random-forests)|
|:-------:|
|Robustness to Outliers|
|Scale Tolerance|
|Ability to Handle Missing Data |
|Ability to Select Features|
|Ability to Rank Features|

***

* [randomForestExplainer](https://github.com/ModelOriented/randomForestExplainer)
* https://modeloriented.github.io/randomForestExplainer/
* [Awesome Random Forest](https://github.com/kjw0612/awesome-random-forest)
* [Interpreting random forests](https://blog.datadive.net/interpreting-random-forests/)
* [Random Forests by Leo Breiman and Adele Cutler](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)
* https://dimensionless.in/author/raghav/
* https://koalaverse.github.io/machine-learning-in-R/random-forest.html
* https://www.wikiwand.com/en/Random_forest
* https://sktbrain.github.io/awesome-recruit-en.v2/
* [Introduction to Random forest by Raghav Aggiwal](https://dimensionless.in/introduction-to-random-forest/)
* [Jump Start your Modeling with Random Forests by Evan Elg](https://www.elderresearch.com/blog/modeling-with-random-forests)
* [Complete Analysis of a Random Forest Model](https://pdfs.semanticscholar.org/82ac/827885f0941723878aff5df27a3207748983.pdf?_ga=2.167570878.1288016698.1567172049-21308644.1555689715)
* [Analysis of a Random Forests Model](https://arxiv.org/abs/1005.0208)
* [Narrowing the Gap: Random Forests In Theory and In Practice](https://arxiv.org/abs/1310.1415)
* [Random Forest:  A Classification and Regression Tool for Compound Classification and QSAR Modeling](https://pubs.acs.org/doi/10.1021/ci034160g)

<img title="Data Mining with Decision Tree" src="https://www.worldscientific.com/na101/home/literatum/publisher/wspc/books/content/smpai/2014/9097/9097/20140827-01/9097.cover.jpg" width= "30%" />

### MARS and Bayesian MARS

#### MARS

[Multivariate adaptive regression splines (MARS) provide a convenient approach to capture the nonlinearity aspect of polynomial regression by assessing cutpoints (knots) similar to step functions. The procedure assesses each data point for each predictor as a knot and creates a linear regression model with the candidate feature(s).](http://uc-r.github.io/mars)

[Multivariate Adaptive Regression Splines (MARS) is a non-parametric regression method that builds multiple linear regression models across the range of predictor values. It does this by `partitioning the data`, and run a `linear regression model` on each different partition.](https://support.bccvl.org.au/support/solutions/articles/6000118097-multivariate-adaptive-regression-splines)

Whereas polynomial functions impose a global non-linear relationship, step functions break the range of x into bins, and fit a different constant for each bin. This amounts to converting a continuous variable into an ordered categorical variable such that our linear regression function is converted to Equation 1：
$$y_i = \beta_0 + \beta_1 C_1(x_i) + \beta_2 C_2(x_i) + \beta_3 C_3(x_i) \dots + \beta_d C_d(x_i) + \epsilon_i, \tag{1}$$

where $C_n(x)$ represents $x$ values ranging from $% <![CDATA[
c_n \leq x < c_{n+1} %]]>$ for $n=1,2,\dots, d$.

The MARS algorithm builds a model in two steps. First, it creates a collection of so-called basis functions (BF). In this procedure, the range of predictor values is partitioned in several groups. For each group, a separate linear regression is modeled, each with its own slope. The connections between the separate regression lines are called knots. The MARS algorithm automatically searches for the best spots to place the knots. Each knot has a pair of basis functions. These basis functions describe the relationship between the environmental variable and the response. The first basis function is ‘max(0, env var - knot), which means that it takes the maximum value out of two options: 0 or the result of the equation ‘environmental variable value – value of the knot’. The second basis function has the opposite form: max(0, knot - env var).

<img src="https://s3.amazonaws.com/cdn.freshdesk.com/data/helpdesk/attachments/production/6018214220/original/MARS.png" width="70%" />

To highlight the progression from recursive partition regression to MARS we start by giving the partition regression model,
$$\hat{f}(x)=\sum_{i=1}^{k}a_iB_i(x)\tag{2}$$
where $x\in D$ and $a_i(i=1,2,\dots, k)$ are the suitably chosen coefficients of the basis functions $B_i$ and $k$ is the number of basis functions in the model.
These basis functions are such that $B_i(x)=\mathbb{I}(x\in R_i)$ where $\mathbb I$ is the indicator function
which is one where the argument is true, zero elsewhere and
the $R_i(i=1, \dots, k)$ form a partition of $D$.

[The usual MARS model is the same as that given in (2) except that the basis functions are different.](https://astro.temple.edu/~msobel/courses_files/mars.pdf) Instead the $B_i$ are given by
$$B_i(X)=\begin{cases} 1, &\text{$i=1$}\\
\Pi_{j=1}^{J_i}[s_{ji}(x_{\nu(ji)}-t_{ji})]_{+}, &\text{$i=2,3,\dots$}
\end{cases}$$
where ${[\cdot]}_{+}=\max(0, \cdot)$; $J_i$ is the degree of the interaction of basis $B_i$, the $s_{ji}$, which we shall call the sign indicators,
equal $\pm 1$, $\nu(ji)$ give the index of the predictor variable
which is being split on the $t_{ji}$ (known as knot points) give
the position of the splits.

* http://uc-r.github.io/mars
* [OVERVIEW OF SDM METHODS IN BCCVL](https://support.bccvl.org.au/support/solutions/articles/6000118097-multivariate-adaptive-regression-splines)
* https://projecteuclid.org/download/pdf_1/euclid.aos/1176347963
* [Using multivariate adaptive regression splines to predict the distributions of New Zealand’s freshwater diadromous fish](https://web.stanford.edu/~hastie/Papers/Ecology/fwb_1448.pdf)
* http://www.stat.ucla.edu/~cocteau/stat204/readings/mars.pdf
* [Multivariate Adaptive Regression Splines (MARS)](https://asbates.rbind.io/2019/03/02/multivariate-adaptive-regression-splines/)
* https://en.wikipedia.org/wiki/Multivariate_adaptive_regression_splines
* https://github.com/cesar-rojas/mars
* http://www.milbo.users.sonic.net/earth/
* https://github.com/scikit-learn-contrib/py-earth
* https://bradleyboehmke.github.io/HOML/mars.html
* http://www.cs.rtu.lv/jekabsons/Files/ARESLab.pdf

#### Bayesian MARS

A Bayesian approach to multivariate adaptive regression spline (MARS) fitting (Friedman, 1991) is proposed. This takes the form of a probability distribution over the space of possible MARS models which is explored using reversible jump Markov chain Monte Carlo methods (Green, 1995). The generated sample of MARS models produced is shown to have good predictive power when averaged and allows easy interpretation of the relative importance of predictors to the overall fit.

The BMARS basis function can be written as
$$B(\vec{x})=\beta_{0}+\sum_{k=1}^{\mathrm{K}} \beta_{k} \prod_{l=0}^{\mathrm{I}}\left(x_{l}-t_{k, l}\right)_{+}^{o_{k, l}}\tag{1}$$
where $\vec x$ is a vector of input, $t_{k,l}$ is the knot point in the $l^{th}$ dimension of the $k^{th}$ component, the function ${(y)}_{+}$ evalutes to $y$ if $y>0$, else it is 0, $o$ is the polynomial degree in the $l^{th}$ dimension of the $k^{th}$ component, $\beta_k$ is the coefficient of the $k^{th}$ component, $K$ is the maximum number of components of the basis function, and $I$ is the maximum allowed number of interactions between
the $L$ dimensions of the input space.

<img src="http://www.milbo.users.sonic.net/gallery/plotmo-example1.png" width="70%" />

- [Bayesian MARS](https://dl.acm.org/citation.cfm?id=599231.599292)
- [An Implementation of Bayesian Adaptive Regression Splines (BARS) in C with S and R Wrappers](http://www.stat.cmu.edu/~kass/papers/jss.pdf)
- [Classification with Bayesian MARS](https://www.bdi.ox.ac.uk/publications/104765)
- http://www.drryanmc.com/presentations/BMARS.pdf
- [Bayesian methods for nonlinear classification and regression. (2002). Denison, Holmes, Mallick and Smith: Wiley.](http://www.stat.tamu.edu/~bmallick/wileybook/book_code.html)
- [Gradient Enhanced Bayesian MARS for Regression and Uncertainty Quantification](http://www.drryanmc.com/presentations/ANS2011_striplingMcClarren_gBMARS_pres.pdf)

## Ensemble methods

There are many competing techniques for solving the problem, and each technique is characterized
by choices and meta-parameters: when this flexibility is taken into account, one easily
ends up with a very large number of possible models for a given task.

[Ensemble methods are meta-algorithms that combine several machine learning techniques into one predictive model in order to decrease variance (bagging), bias (boosting), or improve predictions (stacking).](https://blog.statsbot.co/ensemble-learning-d1dcd548e936)

* [Computer Science 598A: Boosting: Foundations & Algorithms](http://www.cs.princeton.edu/courses/archive/spring12/cos598A/)
* [4th Workshop on Ensemble Methods](http://www.raetschlab.org/ensembleWS)
* [Zhou Zhihua's publication on ensemble methods](http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/publication_toc.htm#Ensemble%20Learning)
* [Online Ensemble Learning: An Empirical Study](https://engineering.purdue.edu/~givan/papers/online-mlj.pdf)
* [Ensemble Learning  literature review](http://www.machine-learning.martinsewell.com/ensembles/)
* [KAGGLE ENSEMBLING GUIDE](https://mlwave.com/kaggle-ensembling-guide/)
* [Ensemble Machine Learning: Methods and Applications](https://www.springer.com/us/book/9781441993250)
* [MAJORITY VOTE CLASSIFIERS: THEORY AND APPLICATION](https://web.stanford.edu/~hastie/THESES/gareth_james.pdf)
* [Neural Random Forests](https://arxiv.org/abs/1604.07143)
* [Generalized Random Forests](https://arxiv.org/abs/1610.01271)
* [Selective Ensemble of Decision Trees](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/rsfdgrc03.pdf)
* [An Empirical Comparison of Voting Classification Algorithms: Bagging, Boosting, and Variants](http://ai.stanford.edu/~ronnyk/vote.pdf)
* [DIFFERENT TYPES OF ENSEMBLE METHODS](https://www.datavedas.com/ensemble-methods/)

Bagging |Boosting |Stacking
---|---|----
<img src="https://datavedas.com/wp-content/uploads/2018/04/image015.jpg" /> | <img src="https://www.datavedas.com/wp-content/uploads/2018/05/3.1.1.1.6-ENSEMBLE-METHODS.jpg" /> | <img src="https://datavedas.com/wp-content/uploads/2018/04/image045-2.jpg" />

* [ML-Ensemble: High performance ensemble learning in Python](http://ml-ensemble.com/)
* https://github.com/flennerhag/mlens
* https://mlbox.readthedocs.io/en/latest/
* [Ensemble Systems & Learn++ by Robi Polikar](http://users.rowan.edu/~polikar/ensemble.html)
- [Applications of Supervised and Unsupervised Ensemble Methods](https://b-ok.cc/book/2096655/6dac48)
- [Boosting-Based Face Detection and Adaptation (Synthesis Lectures on Computer Vision #2)](https://b-ok.cc/book/1270766/3ad0b3)
- [Feature Selection and Ensemble Methods for Bioinformatics: Algorithmic Classification and Implementations](https://b-ok.cc/book/1190611/35413c)
- [Outlier Ensembles: An Introduction](https://b-ok.cc/book/2941709/cec2d0)

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

<img title="bootstrap-sample" src="https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2016/10/bootstrap-sample.png" width="70%"/>

It is a sample-based ensemble method.

***

* http://www.machine-learning.martinsewell.com/ensembles/bagging/
* https://www.cnblogs.com/earendil/p/8872001.html
* https://www.wikiwand.com/en/Bootstrap_aggregating
* [Bagging Regularizes](http://dspace.mit.edu/bitstream/handle/1721.1/7268/AIM-2002-003.pdf?sequence=2)
* [Bootstrap Inspired Techniques in Computational Intelligence](http://users.rowan.edu/~polikar/RESEARCH/PUBLICATIONS/spm2007.pdf)
* [ranger: A Fast Implementation of Random Forests for High Dimensional Data in C++ and R](https://arxiv.org/pdf/1508.04409.pdf)

#### Random Subspace Methods

[Abstract: "Much of previous attention on decision trees focuses on the splitting criteria and optimization of tree sizes. The dilemma between overfitting and achieving maximum accuracy is seldom resolved. A method to construct a decision tree based classifier is proposed that maintains highest accuracy on training data and improves on generalization accuracy as it grows in complexity. The classifier consists of multiple trees constructed systematically by pseudo-randomly selecting subsets of components of the feature vector, that is, trees constructed in randomly chosen subspaces. The subspace method is compared to single-tree classifiers and other forest construction methods by experiments on publicly available datasets, where the method's superiority is demonstrated. We also discuss independence between trees in a forest and relate that to the combined classification accuracy."](http://www.machine-learning.martinsewell.com/ensembles/rsm/Ho1998.pdf)

+ http://www.machine-learning.martinsewell.com/ensembles/rsm/


### Boosting

* http://rob.schapire.net/papers
* https://cseweb.ucsd.edu/~yfreund/papers
* http://www.boosting.org/
* [FastForest: Learning Gradient-Boosted Regression Trees for Classiﬁcation, Regression and Ranking](https://claudio-lucchese.github.io/archives/20180517/index.html)
* [Additive Models, Boosting, and Inference for Generalized Divergences ](https://www.stat.berkeley.edu/~binyu/summer08/colin.bregman.pdf)
* [Boosting as Entropy Projection](https://users.soe.ucsc.edu/~manfred/pubs/C51.pdf)
* [Weak Learning, Boosting, and the AdaBoost algorithm](https://jeremykun.com/2015/05/18/boosting-census/)


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
* [Tree, Forest and Ensemble](https://amueller.github.io/COMS4995-s18/slides/aml-10-021918-trees-forests/#45)
* [An Empirical Comparison of Voting Classification Algorithms: Bagging, Boosting, and Variants](http://ai.stanford.edu/~ronnyk/vote.pdf)
* [Online Parallel Boosting ](https://www.aaai.org/Papers/AAAI/2004/AAAI04-059.pdf)

Methods | Overfit-underfitting | Training Type
---|---|---
Bagging | avoid over-fitting | parallel
Boosting | avoid under-fitting| sequential

### AdaBoost

`AdaBoost` is a boosting methods for supervised classification algorithms, so that the labeled data set is given in the form $D=\{ (x_i, \mathrm{y}_i)\}_{i=1}^{N}$.
AdaBoost is to change the distribution of training data and learn from the shuffled data.
It is an iterative trial-and-error in some sense.

***
**Discrete AdaBoost**
* Input  $D=\{ (x_i, \mathrm{y}_i)\}_{i=1}^{N}$ where $x\in \mathcal X$ and $\mathrm{y}\in \{+1, -1\}$.
* Initialize the observation weights ${w}_i=\frac{1}{N}, i=1, 2, \dots, N$.
* For $t = 1, 2, \dots, T$:
   +  Fit a classifier $G_t(x)$ to the training data using weights $w_i$.
   +  Compute
      $$err_{t}=\frac{\sum_{i=1}^{N}w_i \mathbb{I}(G_t(x_i) \not= \mathrm{y}_i)}{\sum_{i=1}^{N} w_i}.$$
   +  Compute $\alpha_t = \log(\frac{1-err_t}{err_t})$.
   +  Set $w_i\leftarrow w_i\exp[\alpha_t\mathbb{I}(G_t(x_i) \not= \mathrm{y}_i)], i=1,2,\dots, N$ and renormalize so that  $\sum_{i}w_i=1$.
* Output $G(x)=sign[\sum_{t=1}^{T}\alpha_{t}G_t(x)]$.

The indicator function $\mathbb{I}(x\neq y)$ is defined as
$$
\mathbb{I}(x\neq y)=
  \begin{cases}
    1, \text{if $x\neq y$} \\
    0, \text{otherwise}.
  \end{cases}
$$

Note that the weight updating $w_i\leftarrow w_i\exp[\alpha_t\mathbb{I}(G_t(x_i) \not= \mathrm{y}_i)]$ so that the weight does not vary if the prediction is correct $\mathbb{I}(G_t(x_i) \not= \mathrm{y}_i)=0$ and the weight does increase if the prediction is wrong $\mathbb{I}(G_t(x_i) \not= \mathrm{y}_i)=1$.

<img title="reweighting" src="https://arogozhnikov.github.io/images/reweighter/1-reweighting.png" width= "80%" />

<img src="https://cdn-images-1.medium.com/max/1600/0*WOo4d8oNmb85y_Eb.png" width="60%">


***

* [AdaBoost at Wikipedia](https://www.wikiwand.com/en/AdaBoost)
* [BrownBoost at Wikipedia](https://www.wikiwand.com/en/BrownBoost)
* [CoBoosting at Wikipedia ](https://www.wikiwand.com/en/CoBoosting)
* [CSDN blog: Adaboost 算法的原理与推导](https://blog.csdn.net/v_july_v/article/details/40718799)
* [On the Convergence Properties of Optimal AdaBoost](https://arxiv.org/abs/1212.1108)
* [Some Open Problems in Optimal AdaBoost and Decision Stumps](https://arxiv.org/abs/1505.06999)
* [Parallelizing AdaBoost by weights dynamics](https://www.sciencedirect.com/science/article/pii/S0167947306003239)

<img src ="https://cseweb.ucsd.edu/~yfreund/portraitsmall.jpg" width="30%" />
<img src=https://www.microsoft.com/en-us/research/wp-content/uploads/2017/09/avatar_user_33549_1504711750-180x180.jpg width=40% />

For a two-class problem, an `additive logistic model` has the form
$$\log\frac{P(y=1\mid x)}{P(y=-1\mid x)}=\sum_{m=1}^{M}f_{m}=F(x)$$

where $P(y=-1\mid x)+P(y=1\mid x)=1$; inverting we obtain
$$p(x) = P(y=1\mid x) = \frac{\exp(F(x))}{1+\exp(F(x))}.$$

Consider the exponential criterion
$$\arg\min_{F(x)}\mathbb{E}[\exp(-yF(x))]\iff F(x)=\frac{1}{2}\log\frac{P(y=1\mid x)}{P(y=-1\mid x)}.$$

> The Discrete AdaBoost algorithm (population version) builds an additive logistic regression model via Newton-like updates for minimizing $\mathbb{E}[\exp(-yF(x))]$.

$$
\mathbb{E}[\exp(-yF(x))]=\exp(F(x))P(y=-1\mid x)+\exp(-F(x))P(y=+1\mid x)
$$

so that
$$
\frac{\partial \mathbb{E}[\exp(-yF(x))]}{\partial F(x)} = - \exp(-F(x))P(y=+1\mid x) + \exp(F(x))P(y=-1\mid x)\tag{1}
$$
where and $\mathbb E$ represents expectation.
Setting the equation (1) to 0, we get
$$F(x)=\frac{1}{2}\log\frac{P(y=1\mid x)}{P(y=-1\mid x)}.$$

So that
$$\operatorname{sign}(H(x))=\operatorname{sign}(\frac{1}{2}\log\frac{P(y=1\mid x)}{P(y=-1\mid x)})
\\=\begin{cases}1,&\text{if $\frac{P(y=1\mid x)}{P(y=-1\mid x)}>1$}\\
-1, &\text{if $\frac{P(y=1\mid x)}{P(y=-1\mid x)}< 1$}
\end{cases}=\arg\max_{x\in\{+1, -1\}}P(f(x)=y\mid x).$$

***
* Input  $D=\{ (x_i, \mathrm{y}_i)\}_{i=1}^{N}$ where $x\in \mathcal X$ and $\mathrm{y}\in \{+1, -1\}$.
* Initialize the observation weights ${w}_i=\frac{1}{N}, i=1, 2, \dots, N$.
* For $t = 1, 2, \dots, T$:
   +  Fit a classifier $G_t(x)$ to the training data using weights $w_i$.
   +  Compute
      $$err_{t}=\frac{\sum_{i=1}^{N}w_i \mathbb{I}(G_t(x_i) \not= \mathrm{y}_i)}{\sum_{i=1}^{N} w_i}.$$
   +  Compute $\alpha_t = \log(\frac{1-err_t}{err_t})$.
   +  Set $w_i\leftarrow w_i\exp[-\alpha_t G_t(x_i)\mathrm{y}_i)], i=1,2,\dots, N$ and renormalize so that  $\sum_{i}w_i=1$.
* Output $G(x)=sign[\sum_{t=1}^{T}\alpha_{t}G_t(x)]$.

#### Real AdaBoost

In `AdaBoost`, the error is binary- it is 0 if the classification is right otherwise it is 1. It is not precise for some setting. The output of decision trees is a class probability estimate $p(x) = P(y=1 | x)$, the probability that ${x}$ is in the positive class

**Real AdaBoost**

* Input  $D=\{ (x_i, \mathrm{y}_i)\}_{i=1}^{N}$ where $x\in \mathcal X$ and $y\in \{+1, -1\}$.
* Initialize the observation weights ${w}_i=\frac{1}{N}, i=1, 2, \dots, N$;
* For $m = 1, 2, \dots, M$:
   +  Fit a classifier $G_m(x)$ to obtain a class probability estimate $p_m(x)=\hat{P}_{w}(y=1\mid x)\in [0, 1]$, using weights $w_i$.
   +  Compute $f_m(x)\leftarrow \frac{1}{2}\log{p_m(x)/(1-p_m(x))}\in\mathbb{R}$.
   +  Set $w_i\leftarrow w_i\exp[-y_if_m(x_i)], i=1,2,\dots, N$ and renormalize so that $\sum_{i=1}w_i =1$.
* Output $G(x)=sign[\sum_{t=1}^{M}\alpha_{t}G_m(x)]$.

_______

> The Real AdaBoost algorithm fits an additive logistic regression model by stagewise and approximate optimization of $\mathbb{E}[\exp(-yF(x))]$.

* [Additive logistic regression: a statistical view of boosting](https://web.stanford.edu/~hastie/Papers/AdditiveLogisticRegression/alr.pdf)

#### Gentle AdaBoost

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

**LogitBoost (two classes)**

* Input  $D=\{ (x_i, \mathrm{y}_i)\}_{i=1}^{N}$ where $x\in \mathcal X$ and $y\in \{+1, -1\}$.
* Initialize the observation weights ${w}_i=\frac{1}{N}, i=1, 2, \dots, N, F(x)=0, p(x_i)=\frac{1}{2}$;
* For $m = 1, 2, \dots, M$:
   + Compute the working response and weights:
  $$z_i = \frac{y_i^{\ast}-p_i}{p_i(1-p_i)},\\
    w_i = p_i(1-p_i).
  $$
   + Fit a classifier $f_m(x)$ by `weighted least-squares` of $\mathrm{y}_i$ to $x_i$ with weights $w_i$.
   + Update $F(x)\leftarrow F(x)+\frac{1}{2}f_m (x)$.
   + UPdate $w_i \leftarrow w_i \exp(-\mathrm{y}_i f_m (x_i))$ and renormalize.
* Output the classifier  $sign[F(x)]=sign[\sum_{t=1}^{M}\alpha_{t}f_m(x)]$.

Here $y^{\ast}$ represents the outcome and $p(y^{\ast}=1)=p(x)=\frac{\exp(F(x))}{\exp(F(x))+\exp(-F(x))}$.

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


#### arc-x4 Algorithm

[Recent work has shown that combining multiple versions of unstable classifiers such as trees or neural nets results in reduced test set error. One of the more effective is bagging (Breiman [1996a]) Here, modified training sets are formed by resampling from the original training set, classifiers constructed using these training sets and then combined by voting. Freund and Schapire [1995,1996] propose an algorithm the basis of which is to adaptively resample and combine (hence the acronym--arcing) so that the weights in the resampling are increased for those cases most often misclassified and the combining is done by weighted voting. Arcing is more successful than bagging in test set error reduction. We explore two arcing algorithms, compare them to each other and to bagging, and try to understand how arcing works. We introduce the definitions of bias and variance for a classifier as components of the test set error. Unstable classifiers can have low bias on a large range of data sets. Their problem is high variance. Combining multiple versions either through bagging or arcing reduces variance significantly.](https://statistics.berkeley.edu/tech-reports/460)

Breiman proposes a boosting algorithm called `arc-x4` to investigate whether the success of AdaBoost roots in its technical details or in the resampling scheme it uses.
The difference between AdaBoost and arc-x4 is twofold.
First, the weight for object $z_j$ at step $k$ is calculated as the proportion of times $z_j$ has been misclassified by the $k - 1$ classifiers built so far.
Second, the final decision is made by plurality voting rather than weighted majority voting.

`arc` represents `adaptively resample and combine`.

<img src="https://cdn.mathpix.com/snip/images/-OxsjINF-1pNoCPvQ-z1OJwSyJ2ref2JyCdqtBD_D0M.original.fullsize.png" width="70%">

- [Combining Pattern Classifiers: Methods and Algorithms](https://b-ok.cc/book/448487/057f55)
- [BIAS, VARIANCE , AND ARCING CLASSIFIERS](http://docs.salford-systems.com/BIAS_VARIANCE_ARCING.pdf)
- [Online Ensemble Learning: An Empirical Study](https://engineering.purdue.edu/~givan/papers/bp.pdf)
- [Arcing Classifiers](https://statistics.berkeley.edu/tech-reports/460)


|Properties of AdaBoost|
|---|
|AdaBoost is inherently sequential.|
|The training classification error has to go down exponentially fast if the weighted errors of the component classifiers are strictly better than chance.|
|A crucial property of AdaBoost is that it almost never overfits the data no matter how many iterations it is run.|



### multiBoost

[Similar to AdaBoost in the two class case, this new algorithm combines weak classifiers and only requires the performance of each weak classifier be better than random guessing (rather than 1/2).](https://www.intlpress.com/site/pub/pages/journals/items/sii/content/vols/0002/0003/a008/)

[SAMME](https://web.stanford.edu/~hastie/Papers/samme.pdf)
____
* Initialize the observation weights ${w}_i=\frac{1}{N}, i=1, 2, \dots, N$.
* For $t = 1, 2, \dots, T$:
  +  Fit a classifier $G_t(x)$ to the training data using weights $w_i$.
  +  Compute
     $$err_{t}=\frac{\sum_{i=1}^{N}w_i \mathbb{I}(G_t(x_i) \not= \mathrm{y}_i)}{\sum_{i=1}^{N} w_i}.$$
  +  Compute $\alpha_t = \log(\frac{1-err_t}{err_t})+\log(K-1)$.
  +  Set $w_i\leftarrow w_i\exp[\alpha_t\mathbb{I}(G_t(x_i) \not= \mathrm{y}_i)], i=1,2,\dots, N$ and renormalize so that  $\sum_{i}w_i=1$.
* Output $G(x)=\arg\max_{k}[\sum_{t=1}^{T}\alpha_{t}\mathbb{I}_{G_t(x)=k}]$.

<img src="https://cdn.mathpix.com/snip/images/aoxWmzifAs8sfUHfXVHlUDeDB_C3XDh6i-P5OtAitCA.original.fullsize.png">

- http://www.multiboost.org/
- https://www.lri.fr/~kegl/research/publications.html
- [MultiBoost: A Multi-purpose Boosting Package](https://www.lri.fr/~kegl/research/PDFs/BBCCK11.pdf)
- https://web.stanford.edu/~hastie/Papers/samme.pdf
- [A theory of multiclass boosting](http://rob.schapire.net/papers/multiboost-journal.pdf)
- [The return of AdaBoost.MH: multi-class Hamming trees](https://arxiv.org/abs/1312.6086)
- [Multi-class AdaBoost](http://users.stat.umn.edu/~zouxx019/Papers/samme.pdf)
- https://github.com/tizfa/sparkboost
- [multiclass boosting: theory and algorithms](https://papers.nips.cc/paper/4450-multiclass-boosting-theory-and-algorithms.pdf)
- [LDA-AdaBoost.MH: Accelerated AdaBoost.MH based on latent Dirichlet allocation for text categorization](https://journals.sagepub.com/doi/abs/10.1177/0165551514551496?journalCode=jisb)

### Bonsai Boosted Decision Tree

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

Discretization means that the data can be thought of as being binned, even though many of the possible bins may not form leaves in the BBDT; thus, there are a finite number, $n^{max}_{keep}$, of possible keep regions that can be defined. If the $n^{max}_{keep}$ BBDT response values can be stored in memory, then the extremely large number of if/else statements that make up a BDT can be converted into a one-dimensional array of response values. One-dimensional array look-up speeds are extremely fast; they take, in essense, zero time.
If there is not enough memory available to store all of the response values, there are a number of simple alternatives that can be used. For example, if the cut value is known then just the list of indices for keep regions could be stored.  

***
* [Efficient, reliable and fast high-level triggering using a bonsai boosted decision tree](https://arxiv.org/abs/1210.6861)
* [Boosting bonsai trees for efficient features combination : Application to speaker role identification](https://www.researchgate.net/publication/278798264_Boosting_bonsai_trees_for_efficient_features_combination_Application_to_speaker_role_identification)
* [Bonsai Trees in Your Head: How the Pavlovian System Sculpts Goal-Directed Choices by Pruning Decision Trees](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3297555/)
* [HEM meets machine learning](https://higgsml.lal.in2p3.fr/prizes-and-award/award/)


### Gradient Boosting Decision Tree

- [Gradient Boosted Feature Selection](https://arxiv.org/abs/1901.04055)
- [Gradient Regularized Budgeted Boosting](https://arxiv.org/abs/1901.04065)
- [Open machine learning course. Theme 10. Gradient boosting](https://weekly-geekly.github.io/articles/327250/index.html)
- [GBM in Machine Learning in R](https://koalaverse.github.io/machine-learning-in-R/gradient-boosting-machines.html)

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

where $C=\sum_{k=1}^{t-1} \Omega(f_k)$.

Particularly, we take $L(x,y)=(x-y)^2$, and the objective function is given by

$$
obj^{(t)}
=  \sum_{i=1}^{n} [y_i - (\hat{y}^{(t-1)}_i + f_t(x_i))]^2 + \Omega(f_t) + C \\
= \sum_{i=1}^{n} [-2(y_i - \hat{y}^{(t-1)}_i) f_t(x_i) +  f_t^2(x_i) ] + \Omega(f_t) + C^{\prime}
$$

where $C^{\prime}=\sum_{i=1}^{n} (y_i - \hat{y}^{(t-1)}_i)^2 + \sum_{k=1}^{t-1} \Omega(f_k)$.

If there is no regular term $\sum_{k=1}^{t} \Omega(f_k)$, the problem is simplified to
$$\arg\min_{f_{t}}\sum_{i=1}^{n} [-2(y_i - \hat{y}^{(t-1)}_i) f_t(x_i) +  f_t^2(x_i) ]\implies f_t(x_i) = (y_i - \hat{y}^{(t-1)}_i) $$
where $i\in \{1,\cdots, n\}$ and $(y_i - \hat{y}^{(t-1)}_i)=- \frac{1}{2}{[\frac{\partial L(\mathrm{y}_i, f(x_i))}{\partial f(x_i)}]}_{f=f_{t-1}}$.

***

**Boosting  for Regression Tree**

* Input training data set $\{(x_i, \mathrm{y}_i)\mid i=1, \cdots, n\}, x_i\in\mathcal x\subset\mathbb{R}^n, y_i\in\mathcal Y\subset\mathbb R$.
* Initialize $f_0(x)=0$.
* For $t = 1, 2, \dots, T$:
   +   For $i = 1, 2,\dots , n$ compute the residuals
    $$r_{i,t}=y_i-f_{t-1}(x_i)=y_i - \hat{y}_i^{t-1}.$$
   +  Fit a regression tree to the targets $r_{i,t}$   giving **terminal regions**
   $$R_{j,m}, j = 1, 2,\dots , J_m. $$
   +  For $j = 1, 2,\dots , J_m$ compute
      $$\fbox{$\gamma_{j,t}=\arg\min_{\gamma}\sum_{x_i\in R_{j,m}}{L(\mathrm{d}_i, f_{t-1}(x_i)+\gamma)} $}. $$
  +  Update $f_t = f_{t-1}+ \nu{\sum}_{j=1}^{J_m}{\gamma}_{j, t} \mathbb{I}(x\in R_{j, m}),\nu\in(0, 1)$.
* Output $f_T(x)$.

***
For general loss function, it is more common that $(y_i - \hat{y}^{(t-1)}_i) \not=-\frac{1}{2} {[\frac{\partial L(\mathrm{y}_i, f(x_i))}{\partial f(x_i)}]}_{f=f_{t-1}}$.

**Gradient Boosting for Regression Tree**

* Input training data set $\{(x_i, \mathrm{y}_i)\mid i=1, \cdots, n\}, x_i\in\mathcal x\subset\mathbb{R}^n, y_i\in\mathcal Y\subset\mathbb R$.
* Initialize $f_0(x)=\arg\min_{\gamma} L(\mathrm{y}_i,\gamma)$.
* For $t = 1, 2, \dots, T$:
   +   For $i = 1, 2,\dots , n$ compute
    $$r_{i,t}=-{[\frac{\partial L(\mathrm{y}_i, f(x_i))}{\partial f(x_i)}]}_{f=f_{t-1}}.$$
   +  Fit a regression tree to the targets $r_{i,t}$   giving **terminal regions**
   $$R_{j,m}, j = 1, 2,\dots , J_m. $$
   +  For $j = 1, 2,\dots , J_m$ compute
      $$\gamma_{j,t}=\arg\min_{\gamma}\sum_{x_i\in R_{j,m}}{L(\mathrm{d}_i, f_{t-1}(x_i)+\gamma)}. $$
  +  Update $f_t = f_{t-1}+ \nu{\sum}_{j=1}^{J_m}{\gamma}_{j, t} \mathbb{I}(x\in R_{j, m}),\nu\in(0, 1)$.
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

Method | Hypothesis space| Update formulea | Loss function
---|---|---|---|---
Gradient Descent | parameter space $\Theta$  | $\theta_t=\theta_{t-1}-\rho_t\underbrace{\nabla_{\theta} L\mid_{\theta=\theta_{t-1}}}_{\text{Computed by Back-Propagation}}$ |$L(f)=\sum_{i}\ell(y_i, f(x_i\mid \theta))$
Gradient Boost   | function space $\mathcal F$ | $F_{t}= F_{t-1}- \rho_t\underbrace{\nabla_{F} L\mid_{F=F_{t-1}}}_{\text{Approximated by Decision Tree}}$ | $L(F)=\sum_{i}\ell(y_i, F(x_i))$

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
* https://github.com/benedekrozemberczki/awesome-gradient-boosting-papers
* https://github.com/parrt/dtreeviz

<img src="https://raw.githubusercontent.com/benedekrozemberczki/awesome-gradient-boosting-papers/master/boosting.gif">

Ensemble Methods| Training Data |Decision Tree Construction |Update Formula
---|---|---|---
AdaBoost| $(x_i, y_i, w_{i, t})$ | Fit a `classifier` $G_t(x)$ to the training data using weights $w_i$| $f_t=f_{t-1}+\alpha_t G_t$
Gradient  Boost | $(x_i, y_i, r_{i, t})$|Fit a `tree` $G_t(x)$ to the targets $r_{i,t}$| $f_t=f_{t-1}+\nu G_t$.

Note that $\alpha_t$ is computed as $\alpha_t=\log(\frac{1-err_t}{err_t})$ while the shrinkage parameter $\nu$ is chosen in $(0, 1)$.
AdaBoost is desined for any classifier while Gradient Boost methods is usually applied to decision tree.

#### Stochastic Gradient Boost

A minor modification was made to gradient boosting to incorporate randomness as an integral part of the procedure.
Specially, at each iteration a subsample of the training data is drawn at random (without replacement) from the full training data set. This randomly selected subsamples is then used, instead of the full sample, to fit the base learner and compute the model update for the current iteration.

**Stochastic Gradient Boosting for Regression Tree**

* Input training data set $\{(x_i, \mathrm{y}_i)\mid i=1, \cdots, n\}, x_i\in\mathcal x\subset\mathbb{R}^n, y_i\in\mathcal Y\subset\mathbb R$. Amd ${\{\pi(i)\}}_1^N$ be a random permutation of their integers $\{1,\dots, n\}$. Then a random sample of size $\hat n< n$ is given by $\{(x_{\pi(i)}, \mathrm{y}_{\pi(i)})\mid i=1, \cdots, \hat n\}$.
* Initialize $f_0(x)=\arg\min_{\gamma} L(\mathrm{y}_i,\gamma)$.
* For $t = 1, 2, \dots, T$:
   +   For $i = 1, 2,\dots , \hat n$ compute
    $$r_{\pi(i),t}=-{[\frac{\partial L(\mathrm{y}_{\pi(i),t}, f(x_{\pi(i),t}))}{\partial f(x_{\pi(i),t})}]}_{f=f_{t-1}}.$$
   +  Fit a regression tree to the targets $r_{\pi(i),t}$   giving **terminal regions**
   $$R_{j,m}, j = 1, 2,\dots , J_m. $$
   +  For $j = 1, 2,\dots , J_m$ compute
      $$\gamma_{j,t}=\arg\min_{\gamma}\sum_{x_{\pi(i)}\in R_{j,m}}{L(\mathrm{d}_i, f_{t-1}(x_{\pi(i)})+\gamma)}. $$
  +  Update $f_t = f_{t-1}+ \nu{\sum}_{j=1}^{J_m}{\gamma}_{j, t} \mathbb{I}(x\in R_{j, m}),\nu\in(0, 1)$.
* Output $f_T(x)$.


- https://statweb.stanford.edu/~jhf/ftp/stobst.pdf
- https://statweb.stanford.edu/~jhf/
- [Stochastic gradient boosted distributed decision trees](https://dl.acm.org/citation.cfm?id=1646301)


### xGBoost

In Gradient Boost, we compute and fit a regression a tree to
$$
r_{i,t}=-{ [\frac{\partial L(\mathrm{d}_i, f(x_i))}{\partial f(x_i)}] }_{f=f_{t-1}}.
$$
Why not the error $L(\mathrm{d}_i, f(x_i))$ itself?
Recall the Taylor expansion as following
$$f(x+h) = f(x)+f^{\prime}(x)h + f^{(2)}(x)h^{2}/2!+ \cdots +f^{(n)}(x)h^{(n)}/n!+\cdots$$
so that the non-convex error function can be expressed as a polynomial in terms of $h$,
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

Here ${w}$ is the vector of scores on leaves, **${q}$ is a function assigning each data point to the corresponding leaf**, and ${T}$ is the number of leaves.
In XGBoost, we define the complexity as
$$
\Omega(f)=\gamma T + \frac{1}{2}\lambda \sum_{i=1}^{T} {w}_i^2.
$$

After re-formulating the tree model, we can write the objective value with the ${t}$-th tree as:
$$
obj^{(t)} = \sum_{i=1}^{n}[g_i w_{q(x_i)}+\frac{1}{2} h_i w_{q(x_i)}^2 + \gamma T+\frac{1}{2}\lambda \sum_{i=1}^{n}w_i^2]
\\=\sum_{j=1}^{T}[(\sum_{i\in I_{j}}g_i)w_j+\frac{1}{2}(\sum_{i\in I_{j}}h_i + \lambda)w_j^2]+\gamma T
$$
where $I_j=\{i\mid q(x_i)=j\}$ is the set of indices of data points assigned to the $j$-th leaf.
The equation hold because only the leaves or terminal nodes output the results.
We could further compress the expression by defining $G_j=\sum_{i\in I_j} g_i$ and $H_j=\sum_{i\in I_j} h_i$:
$$
obj^{(t)} = \sum_{j=1}^{T}[(G_j w_j+\frac{1}{2}(H_j +\lambda)w_j^2]+\gamma T.
$$

In this equation, $w_j$ are independent with respect to each other, the form $G_j w_j + \frac{1}{2}(H_j+\lambda)w^2_j$ is quadratic and the best $w_j$ for a given structure $q(x)$ and the best objective reduction we can get is:

$$
w_j^{\ast} =-\underbrace{(\overbrace{H_j+\lambda}^{\text{Hessian of objective function}} )^{-1}}_{\text{learning rate}}\underbrace{G_j}_{\text{gradient}},\\
obj^{\ast} =\frac{1}{2}\sum_{j=1}^{T}
-(H_j+\lambda )^{-1}G_j^2+\gamma T.
$$


Method | Hypothesis space| Update formulea
---|---|---|---
Newton's Method  | parameter space $\Theta$  | $\theta_t=\theta_{t-1}-\rho_t(\overbrace{\nabla_{\theta}^2 L\mid_{\theta=\theta_{t-1}}}^{\text{Hessian Matrix}})^{\color{purple}{-1}}\underbrace{\nabla_{\theta} L\mid_{\theta=\theta_{t-1}}}_{\text{Gradient}}$
xGBoost   | function space $\mathcal F$ | $F_{t}= F_{t-1}- \rho_t\underbrace{(\overbrace{\nabla_{F}^2 L\mid_{F=F_{t-1}}}^{\text{Hessian}})^{\color{red}{-1}} \overbrace{\nabla_{F} L\mid_{F=F_{t-1}}}^{\text{Gradient}}}_{\text{Approximated by Decision Tree}}$
____
Another key of  xGBoost is how to a construct a tree fast and painlessly.
We will try to optimize _`one level`_ of the tree at a time. Specifically we try to split a leaf into two leaves, and the score it gains is
$$
Gain = \frac{1}{2} \left[\underbrace{\frac{G_L^2}{H_L+\lambda}}_{\text{from left leaf}} + \underbrace{\frac{G_R^2}{H_R+\lambda}}_{\text{from the right leaf}}-\underbrace{\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}}_{\text{from the original leaf} } \right] - \gamma
$$

This formula can be decomposed as 1) the score on the new left leaf 2) the score on the new right leaf 3) The score on the original leaf 4) regularization on the additional leaf.
We can see an important fact here: if the gain is smaller than $\gamma$, we would do better not to add that branch. This is exactly the **pruning techniques** in tree based models! By using the principles of supervised learning, we can naturally come up with the reason these techniques work.
<img src="https://raw.githubusercontent.com/dmlc/web-data/master/xgboost/model/struct_score.png" width="70%">

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
* [How to Visualize Gradient Boosting Decision Trees With XGBoost in Python](https://machinelearningmastery.com/visualize-gradient-boosting-decision-trees-xgboost-python/)
* [Awesome XGBoost](https://github.com/dmlc/xgboost/blob/master/demo/README.md#machine-learning-challenge-winning-solutions)
* [Story and lessons from xGBoost](https://homes.cs.washington.edu/~tqchen/2016/03/10/story-and-lessons-behind-the-evolution-of-xgboost.html)
* [Awesome XGBoost](https://github.com/dmlc/xgboost/tree/master/demo)

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

Instead of one-hot encoding, the optimal solution is to split on a `categorical feature` by partitioning its categories into 2 subsets. If the feature has $k$ categories, there are $2^{(k-1)} - 1$ possible partitions. But there is an efficient solution for regression trees[8]. It needs about $O(k \times log(k))$ to find the optimal partition.

The basic idea is to sort the categories according to the training objective at each split. More specifically, LightGBM sorts the `histogram` (for a categorical feature) according to its `accumulated values` (sum_gradient / sum_hessian) and then finds the best split on the sorted histogram.

`Histogram` is an un-normalized empirical cumulative distribution function, where the continuous features (in flow point data structure) is split into ${k}$ buckets by threahold values such as if $x\in [0, 2)$ then ${x}$ will be split into bucket 1. It really reduces the complexity to store the data and compute the impurities based on the distribution of features.

<img src="http://zhoutao822.coding.me/2019/01/13/LightGBM/5.png" width="80%" />

**Optimization in parallel learning**

[Feature Parallel in LightGBM, Data Parallel in LightGBM](https://lightgbm.readthedocs.io/en/latest/Features.html#optimization-in-network-communication)
<img src="https://zhoutao822.coding.me/2019/01/13/LightGBM/6.png" width="80%">

- [A Communication-Efficient Parallel Algorithm for Decision Tree](https://arxiv.org/abs/1611.01276)
- [LightGBM, Light Gradient Boosting Machine](https://github.com/Microsoft/LightGBM/)
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)
- [Python3机器学习实践：集成学习之LightGBM - AnFany的文章 - 知乎](https://zhuanlan.zhihu.com/p/53583034)
- https://lightgbm.readthedocs.io/en/latest/
- https://www.msra.cn/zh-cn/news/features/lightgbm-20170105
- [LightGBM](http://zhoutao822.coding.me/2019/01/13/LightGBM/)
- [Reference papers of lightGBM](https://lightgbm.readthedocs.io/en/latest/Features.html#references)
- https://lightgbm.readthedocs.io/en/latest/Features.html
- https://aichamp.wordpress.com/tag/lightgbm/

#### CatBoost

`CatBoost` is an algorithm for gradient boosting on decision trees. [Developed by Yandex researchers and engineers, it is the successor of the `MatrixNet` algorithm that is widely used within the company for ranking tasks, forecasting and making recommendations. It is universal and can be applied across a wide range of areas and to a variety of problems](https://betapage.co/startup/catboost) such as search, recommendation systems, personal assistant, self-driving cars, weather prediction and many other tasks. It is in open-source and can be used by anyone now.

Two critical algorithmic advances introduced in CatBoost are the implementation of `ordered boosting`, a permutation-driven alternative to the classic algorithm, and
an innovative algorithm for processing `categorical features`.
Both techniques were created to fight a prediction shift caused by a special kind of target leakage present in all currently existing implementations of gradient boosting algorithms.

The number of trees is controlled by the starting parameters. To prevent over-fitting, use the over-fitting detector. When it is triggered, trees stop being built.
***
Before learning, the possible values of objects are divided into disjoint ranges ($\color{red}{\fbox{buckets}}$) delimited by the threshold values ($\color{red}{\fbox{splits}}$). The size of the quantization (the number of splits) is determined by the starting parameters (separately for numerical features and numbers obtained as a result of converting categorical features into numerical features).

Quantization is also used to split the label values when working with categorical features. А random subset of the dataset is used for this purpose on large datasets.



The most widely used technique which is usually applied to low-cardinality categorical features
is one-hot encoding; another way to deal with categorical features is to compute some statistics using the label values of the examples.
Namely, assume that we are given a dataset of observations $D = \{(\mathrm{X}_i, \mathrm{Y}_i)\mid i=1,2,\cdots, n\}$,
where $\mathrm{X}_i = (x_{i,1}, x_{i, 2}, \cdots, x_{i,m})$ is a vector of ${m}$ features, some numerical, some categorical, and $\mathrm{Y}_i\in\mathbb{R}$ is a label value.
The simplest way is to substitute the category with the _average_ label value on the whole train dataset. So, $x_{i;k}$ is substituted with $\frac{\sum_{j=1}^n [x_{j;k}=x_{i;k}]\cdot \mathrm{Y}_j}{\sum_{j=1}^n [x_{j;k}=x_{i;k}]}$; where $[\cdot]$ denotes Iverson
brackets, i.e., $[x_{j;k} = x_{i;k}]$ equals 1 if $x_{j;k} = x_{i;k}$ and 0 otherwise.
This procedure, obviously, leads to overfitting.

CatBoost uses a more efficient strategy which reduces overfitting and allows to use the whole dataset for training.

[Before each split is selected in the tree (see Choosing the tree structure), `categorical features are transformed to numerical`. This is done using various statistics on combinations of categorical features and combinations of categorical and numerical features.](https://catboost.ai/docs/concepts/algorithm-main-stages_cat-to-numberic.html)

The method of transforming categorical features to numerical generally includes the following stages:
* Permutating the set of input objects in a random order.
* Converting the label value from a floating point to an integer.

Namely, we perform a random permutation of the dataset and
for each example we compute average label value for the example with the same category value placed before the given one in the permutation.
Let $\sigma=(\sigma_1, \cdots, \sigma_n)$ be the permutation, then $x_{\sigma_p;k}$ is substituted with
$$\frac{\sum_{j=1}^{p-1} [x_{\sigma_j; k}=x_{\sigma_p;k}]\cdot \mathrm{Y}_{\sigma_j} + a\cdot P}{\sum_{j=1}^{p-1} [x_{\sigma_j; k}=x_{\sigma_p;k}]}$$

where we also add a prior value ${P}$ and a parameter $a > 0$, which is the weight of the prior.


The method depends on the machine learning problem being solved (which is determined by the selected loss function).

The tree depth and other rules for choosing the structure are set in the starting parameters.
****
How a “feature-split” pair is chosen for a leaf:
* A list is formed of possible candidates (“feature-split pairs”) to be assigned to a leaf as the split.
* A number of penalty functions are calculated for each object (on the condition that all of the candidates obtained from step 1 have been assigned to the leaf).
* The split with the smallest penalty is selected.

The resulting value is assigned to the leaf.

[This procedure is repeated for all following leaves (the number of leaves needs to match the depth of the tree).](https://catboost.ai/docs/concepts/algorithm-main-stages_choose-tree-structure.html)

[CatBoost implements an algorithm that allows to fight usual gradient boosting biases.](https://catboost.ai/docs/concepts/algorithm-main-stages_choose-tree-structure.html)

Assume that we take one random permutation $\sigma$ of the training examples and maintain n different supporting models $M_1, \cdots , M_n$ such that the model $M_i$ is learned using only the `first i examples in the permutation`.
At each step, in order to obtain the residual for $j$-th sample, we use the model $M_{j−1}$.

<img src="https://cdn.mathpix.com/snip/images/kVsAOTih7qFTcp3z-BPJIH8cKdrq5GpvkYivu0XRg-o.original.fullsize.png" witdh= "80%"/>

|Three Steps|
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


#### More: TencentBoost, ThunderGBM and Beyond

There are more gradient boost tree algorithms such as ThubderGBM, TencentBoost, GBDT on angle and H2o.

##### TencentBoost

[Gradient boosting tree (GBT), a widely used machine learning algorithm, achieves state-of-the-art performance in academia, industry, and data analytics competitions. Although existing scalable systems which implement GBT, such as XGBoost and MLlib, perform well for data sets with medium-dimensional features, they can suffer performance degradation for many industrial applications where the trained data sets contain high dimensional features. The performance degradation derives from their inefficient mechanisms for model aggregation-either map-reduce or all-reduce. To address this high-dimensional problem, we propose a scalable execution plan using the parameter server architecture to facilitate the model aggregation. Further, we introduce a sparse-pull method and an efficient index structure to increase the processing speed. We implement a GBT system, namely `TencentBoost`, in the production cluster of Tencent Inc. The empirical results show that our system is 2-20× faster than existing platforms.](http://net.pku.edu.cn/~cuibin/Papers/2017%20ICDE%20boost.pdf)

- [TencentBoost: A Gradient Boosting Tree System with Parameter Server](https://ieeexplore.ieee.org/abstract/document/7929984)
- [GBDT on Angel](https://github.com/Angel-ML/angel/blob/master/docs/algo/gbdt_on_angel.md)
- [The purposes of using parameter server in GBDT](https://github.com/Angel-ML/angel/issues/7)

##### ThunderGBM

[`ThunderGBM` is dedicated to helping users apply GBDTs and Random Forests to solve problems efficiently and easily using GPUs. Key features of ThunderGBM are as follows.](https://github.com/Xtra-Computing/thundergbm/blob/master/docs/index.md)

* Support regression, classification and ranking.
* Use same command line options as XGBoost, and support Python (scikit-learn) interface.
* Supported Operating Systems: Linux and Windows.
* ThunderGBM is often 10 times faster than XGBoost, LightGBM and CatBoost. It has excellent performance on handling high dimensional and sparse problems.

----
Methods | Tree Construction | Update Formula | Training Methods
---|---|---|---
XGBoost| Newton-like
LightGBM | leaf-wise
CatBoost| ordered boosting
TencentBoost|
ThunderGBM|



- [ThunderGBM: Fast GBDTs and Random Forests on GPUs](https://github.com/Xtra-Computing/thundergbm)
- [ThunderGBM：快成一道闪电的梯度提升决策树](https://zhuanlan.zhihu.com/p/58626955)
- [Gradient Boosted Categorical Embedding and Numerical Trees](http://www.hongliangjie.com/talks/GB-CENT_MLIS_2017-06-06.pdf)
- [一步一步理解GB、GBDT、xgboost](https://www.cnblogs.com/wxquare/p/5541414.html)
- [从结构到性能，一文概述XGBoost、Light GBM和CatBoost的同与不同](https://zhuanlan.zhihu.com/p/34698733)
- [从决策树、GBDT到XGBoost/lightGBM/CatBoost](https://zhuanlan.zhihu.com/p/59419786)

* [PLANET: Massively Parallel Learning of Tree Ensembles with MapReduce](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/36296.pdf)
* [Tiny Gradient Boosting Tree](https://github.com/wepe/tgboost)
* [FastForest: Learning Gradient-Boosted Regression Trees for Classiﬁcation, Regression and Ranking](https://claudio-lucchese.github.io/archives/20180517/index.html)
* [Programmable Decision Tree Framework](https://github.com/yubin-park/bonsai-dt)
* [bonsai-dt: Programmable Decision Tree Framework](https://yubin-park.github.io/bonsai-dt/)
* [Treelite : model compiler for decision tree ensembles](https://treelite.readthedocs.io/en/latest/)
* [Block-distributed Gradient Boosted Trees](https://arxiv.org/abs/1904.10522)
* [Distributed decision tree ensemble learning in Scala](https://github.com/stripe/brushfire)
* [Yggdrasil: An Optimized System for Training Deep Decision Trees at Scale](https://cs.stanford.edu/~matei/papers/2016/nips_yggdrasil.pdf)
* [Efficient Distributed Decision Trees for Robust Regression](https://infoscience.epfl.ch/record/218970)
* [TF Boosted Trees: A scalable TensorFlow based framework for gradient boosting](https://arxiv.org/abs/1710.11555)

### Fast Traversal of Large Ensembles of Regression Trees

* [Fast Traversal of Large Ensembles of Regression Trees](https://ercim-news.ercim.eu/en107/special/fast-traversal-of-large-ensembles-of-regression-trees)
* [Parallelizing the Traversal of Large Ensembles of Decision Trees](http://pages.di.unipi.it/rossano/wp-content/uploads/sites/7/2019/03/ITPDS19.pdf)
#### QuickScorer

`QuickScorer` was designed by Lucchese, C., Nardini, F. M., Orlando, S., Perego, R., Tonellotto, N., and Venturini, R. with the support of Tiscali S.p.A.

It adopts a novel bitvector representation of the tree-based ranking model, and performs an interleaved traversal of the ensemble by means of simple logical bitwise operations. The performance of the proposed algorithm are unprecedented, due to its cache-aware approach, both in terms of data layout and access patterns, and to a control ﬂow that entails very low branch mis-prediction rates.


**All the nodes whose Boolean conditions evaluate to _False_ are called false nodes, and true nodes otherwise.**
The scoring of a document represented by a feature vector $\mathrm{x}$  requires the traversing of all the trees in the ensemble, starting at their root nodes.
If a visited node in N is a false one, then the right branch is taken, and the left branch otherwise.
The visit continues recursively until a leaf node is reached, where the value of the prediction is returned.

The building block of this approach is an alternative method for tree traversal based on `bit-vector computations`.
Given a tree and a vector of document features,
this traversal processes all its nodes and produces a bitvector
which encodes the exit leaf for the given document.

Given an input feature vector $\mathrm x$ and a tree $T_h = (N_h;L_h)$, where $N_h$ is a set of  internal nodes and $L_h$ is a set
of leaves,
our tree traversal algorithm processes the internal nodes of
$T_h$ with the goal of identifying a set of candidate exit leaves, denoted by $C_h$ with $C_h \subset L_h$,
which includes the actual exit leaf $e_h$.
Initially $C_h$ contains all the leaves in $L_h$, i.e., $C_h = L_h$.
Then, the algorithm evaluates one after the other in an arbitrary order the test conditions of all the internal nodes of $T_h$.
Considering the result of the test for a certain internal node $n \in N_h$,
the algorithm is able to infer that some leaves cannot be the exit leaf and, thus, it can safely remove them from $C_h$.
**Indeed, if $n$ is a false node (i.e., its test condition is false), the leaves in the left subtree of $n$ cannot be the exit leaf and they can be safely removed from $C_h$.
Similarly, if $n$ is a true node, the leaves in the right subtree of n can be removed from $C_h$.**

The second refinement implements the operations on $C_h$ with fast bit-wise operations.
The idea is to represent $C_h$ with a bitvector ${\text{leaf_index}}_h$, where each bit corresponds to a distinct leaf in $L_h$, i.e., $\text{leaf_index}_h$ is the characteristic vector of $C_h$.
Moreover, every internal node $n$ is associated with a bit mask of the same length encoding
(with 0’s) the set of leaves to be removed from $C_h$ whenever $n$ turns to be a false node.
**In this way, the bitwise `logical AND` between $\text{leaf_index}_h$ and the bit mask of a false
node $n$ corresponds to the removal of the leaves in the left subtree of $n$ from $C_h$.**
Once identified all the false nodes in a tree and performed the associated AND operations over $\text{leaf_index}_h$, the exit leaf of the tree corresponds to the leftmost bit set to 1 in $\text{leaf_index}_h$.

One important result is that `Quick Scorer` computes
$s(x)$ by only identifying the branching nodes whose test evaluates to false, called false nodes.
For each false node detected in $T_h \in T$ , `Quick Scorer` updates a bitvector associated with $T_h$, which stores information that is eventually exploited to identify
the exit leaf of $T_h$ that contributes to the final score $s(x)$.
To this end, `Quick Scorer` maintains for each tree $T_h \in T$ a bitvector *leafidx[h]*, made of $\land$ bits, one per leaf.
Initially, every bit in *leafidx[h]* is set to $\mathrm 1$. Moreover, each branching node is
associated with a bitvector mask, still of $\land$ bits, identifying the set of unreachable
leaves of $T_h$ in case the corresponding test evaluates to false.
Whenever a false node is visited, the set of unreachable leaves *leafidx[h]* is updated through a logical $AND (\land)$ with mask.
Eventually, the leftmost bit set in *leafidx[h]* identifies the leaf corresponding to the score contribution of $T_h$, stored in the lookup table *leafvalues*.
____
ALGORITHM 1: Scoring a feature vector $x$ using a binary decision tree $T_h$

* **Input**:
  * $x$: input feature vector
  * $T_h = (N_h, L_h)$: binary decision tree, with
    *  $N_h = \{n_0, n_1, \cdots\}$: internal nodes of $T_h$
    *  $L_h = \{l_0, l_1, \cdots\}$: leaves of $T_h$
    *  $n.mask$: node bit mask associated with $n\in N_h$
    *  $l_j.val$: score contribution associated with leaf $l_j\in L_h$
* **Output**:
  * tree traversal output value
* **$score(x, T_h)$**:
  *  $\text{leaf_index}_h\leftarrow (1,1,\dots, 1)$
  *  $U\leftarrow FindFalse(x, T_h)$
  *  **foreach node** $n \in U$ **do**
     *  $\text{leaf_index}_h\leftarrow \text{leaf_index}_h\land n.mask$
  *  $j \leftarrow\text{index of leftmost bit set to 1 of leaf_index}_h$
  * **return** $l_j.val$

____
ALGORITHM 2: : The QUICKSCORER Algorithm

* **Input**:
  * $x$: input feature vector
  * $\mathcal T$: ensemble of binary decision trees, with
    *  $\{w_0, w_1, \cdots, w_{|\mathcal{T}|-1}\}$:  weights, one per tree
    *  $thresholds$: sorted sublists of thresholds, one sublist per feature
    *  $treeids$: tree’s ids, one per node/threshold
    *  $nodemasks$: node bit masks, one per node/threshold
    *  $offsets$: offsets of the blocks of triples
    *  $leafindexes$: result bitvectors, one per each tree
    *  $leafvalues$: score contributions, one per each tree leaf
* **Output**:
  * final score of $x$
* **$\text{QUICKSCORER}(x, T_h)$**:
   *  **foreach node** $h \in \{0,1\cdots, |T|-1\}$ **do**
      *  $\text{leaf_index}_h\leftarrow (1,1,\dots, 1)$
   *  **foreach node** $k \in \{0,1\cdots, |\mathcal{F}|-1\}$ **do**
      *  $i\leftarrow offsets[k]$
      *  $end\leftarrow offsetsets[k+1]$
      *  **while** $x[k] > thresholds[i]$ do
         * $h \leftarrow treeids[i]$
         * $\text { leafindexes }[h] \leftarrow \text { leafindexes }[h] \wedge \text { nodemasks }[i]$
         * $i\leftarrow i+1$
         * **if** $i\geq end$ **then**
           * **break**
     *  $score \leftarrow 0$
     * **foreach node** $h \in \{0,1\cdots, |T|-1\}$ **do**
       * $j \leftarrow \text { index of leftmost bit set to 1 of}\,\, {leafindexes }[h]$
       * $l \leftarrow h \cdot| L_{h} |+j$
       * $\text { score } \leftarrow \text { score }+w_{h} \cdot \text { leafvalues }[l]$
  * **return** $score$


- [ ] [QuickScorer: a fast algorithm to rank documents with additive ensembles of regression trees](https://www.cse.cuhk.edu.hk/irwin.king/_media/presentations/sigir15bestpaperslides.pdf)
- [ ] [Official repository of Quickscorer](https://github.com/hpclab/quickscorer)
- [ ] [QuickScorer: Efficient Traversal of Large Ensembles of Decision Trees](http://ecmlpkdd2017.ijs.si/papers/paperID718.pdf)
- [ ] [Fast Ranking with Additive Ensembles of Oblivious and Non-Oblivious Regression Trees](http://pages.di.unipi.it/rossano/wp-content/uploads/sites/7/2017/04/TOIS16.pdf)
- [Tree traversal](https://venus.cs.qc.cuny.edu/~mfried/cs313/tree_traversal.html)
- https://github.com/hpclab/gpu-quickscorer
- https://github.com/hpclab/multithread-quickscorer
- https://github.com/hpclab/vectorized-quickscorer
- https://patents.google.com/patent/WO2016203501A1/en

<img src="https://ercim-news.ercim.eu/images/stories/EN107/perego.png" width="60%" />

#### vQS

[Considering that in most application scenarios the same tree-based model is applied to a multitude of items, we recently introduced further optimisations in QS. In particular, we introduced vQS [3], a parallelised version of QS that exploits the SIMD capabilities of mainstream CPUs to score multiple items in parallel. Streaming SIMD Extensions (SSE) and Advanced Vector Extensions (AVX) are sets of instructions exploiting wide registers of 128 and 256 bits that allow parallel operations to be performed on simple data types. Using SSE and AVX, vQS can process up to eight items in parallel, resulting in a further performance improvement up to a factor of 2.4x over QS. In the same line of research we are finalising the porting of QS to GPUs, which, preliminary tests indicate, allows impressive speedups to be achieved.](https://ercim-news.ercim.eu/en107/special/fast-traversal-of-large-ensembles-of-regression-trees)

- [Exploiting CPU SIMD Extensions to Speed-up Document Scoring with Tree Ensembles](http://pages.di.unipi.it/rossano/wp-content/uploads/sites/7/2016/07/SIGIR16a.pdf)

#### RapidScorer

[`RapidScorer`](http://ai.stanford.edu/~wzou/kdd_rapidscorer.pdf) is a novel framework
for speeding up the scoring process of industry-scale tree ensemble models, without hurting the quality of scoring results.
`RapidScorer` introduces a modified run length encoding called `epitome` to the bitvector representation of the tree nodes.
Epitome can greatly reduce the computation cost to traverse the tree ensemble, and work with several other proposed strategies to maximize the compactness of data units in memory.
The achieved compactness makes it possible to fully utilize data parallelization to improve model scalability.

![RapidScorer](https://cdn.mathpix.com/snip/images/vLj1OrPKymNq_nWn3xY0TLijGP-8K1e3eYtQG_Wm-Cw.original.fullsize.png)

- http://ai.stanford.edu/~wzou/kdd_rapidscorer.pdf

#### AdaQS

This article extends the work of quickscorer by proposing a novel adaptive algorithm (i.e., AdaQS) for sparse data and regression trees with default directions as trained by XGBoost.

For each tree node with default direction going right, we adaptively swap its left child and right child. The swap operation is to ensure every default direction going left, thus the absent features of sparse data lead to no false node.

However, the swap has a side effect that changes the Boolean condition from '<' (less than) operation to '>=' (more than or equal to) operation. To preserve the efficiency of quickscorer's search strategy we transform the regression trees into two separate suites of flat structures. One corresponds to the tree nodes with '>' operation and the other corresponds to the tree nodes with '<=' operation. When a sparse instance queries the score, we search in both the two suites and integrate the results.

<img src="https://pic1.zhimg.com/80/v2-a911464197f0eb281ca742c0ea954e98_hd.jpg" width="80%" />

- https://zhuanlan.zhihu.com/p/54932438
- https://github.com/qf6101/adaqs

### Accelerated Gradient Boosting

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
____________

* [Accelerated Gradient Boosting](https://arxiv.org/abs/1803.02042)


### Gradient Boosting  Machine: Beyond Boost Tree

A general gradient descent “boosting” paradigm is developed for additive expansions based on any fitting criterion. It is not only for the decision tree.

* [Gradient Boosting Machines](http://uc-r.github.io/gbm_regression)
* [Start With Gradient Boosting, Results from Comparing 13 Algorithms on 165 Datasets](https://machinelearningmastery.com/start-with-gradient-boosting/)
* [A Gentle Introduction to the Gradient Boosting Algorithm for Machine Learning](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)
* [Gradient Boosting Algorithm – Working and Improvements](https://data-flair.training/blogs/gradient-boosting-algorithm/)
----
* [Complete Machine Learning Guide to Parameter Tuning in Gradient Boosting (GBM) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)
* https://bradleyboehmke.github.io/HOML/gbm.html
* [Leveraging k-NN for generic classification boosting](https://hal.inria.fr/hal-00664462)
* [Constructing Boosting Algorithms from SVMs: an Application to One-Class Classification](https://pdfs.semanticscholar.org/a724/bb040771307571f3ae1233a115cd62bb52be.pdf)
****

`AdaBoost` is related with so-called exponential loss $\exp(-{y_i}p(x_i))$ where $x_i\in\mathbb{R}^p, y_i\in\{-1, +1\}, p(\cdot)$ is the input features, labels and prediction function, respectively.
And the weight is update via the following formula:
$$w_i\leftarrow w_i\exp[-y_ip(x_i)], i=1,2,\dots, N.$$


The gradient-boosting algorithm is general in that it only requires the analyst specify a `loss function` and its `gradient`.
When the labels are multivariate, [Alex Rogozhnikova et al](https://arxiv.org/abs/1410.4140) define a more
general expression of the AdaBoost criteria
$$w_i\leftarrow w_i\exp[-y_i\sum_{j}a_{ij}p(x_j)], i=1,2,\dots, N,$$

where $a_{ij}$ are the elements of some square matrix ${A}$. For the case where A is the identity matrix,
the AdaBoost weighting procedure is recovered. Other choices of ${A}$ will induce `non-local effects`,
e.g., consider the sparse matrix $A_{knn}$ given by
$$a_{ij}^{knn}=
\begin{cases}
1, & \text{$j \in knn(i)$; events ${i}$ and ${j}$ belong to the same class} \\
0, & \text{otherwise}.
\end{cases}$$


* [New approaches for boosting to uniformity](https://arxiv.org/abs/1410.4140)
* [uBoost: A boosting method for producing uniform selection efficiencies from multivariate classifiers](https://arxiv.org/abs/1305.7248)

Other ensemble methods include clustering methods ensemble, dimensionality reduction ensemble, regression ensemble, ranking ensemble.

### Optimization and Boosting

Gradient descent methods, as `numerical optimization methods`, update the values of the parameters at each iteration while the size of parameter is fixed  so that the complexity of the model is limited.
In the end, these methods output some optimal or sub-optimal parameters of the model $\theta_T$ where
$$\theta_T=\theta_0-\sum_{t=1}^{T-1}\rho_t\nabla_{\theta} L\mid_{\theta=\theta_{t-1}}, \quad\theta_{t}=\theta_{t-1}-\rho_t\nabla_{\theta} L\mid_{\theta=\theta_{t-1}}.$$
The basic idea of gradient descent methods is to find
$$\theta_t=\arg\min_{\alpha\in\mathbb R}L(f(\theta_{t-1} + \alpha\Delta)),\Delta\in span(\nabla_{\theta} L\mid_{\theta=\theta_{t-1}}).$$
So that $L(f(\theta_t))\leq L(f(\theta_{t-1}))$. In some sense, it requires the model is expressive enough to solve the problems or the size of the parameter is large.

Gradient boost methods, as `boost methods`, are in the additive training form and the size of the models increases after each iteration so that the model complexity grows. In the end, these methods output some optimal or sub-optimal models $F_T$ where
$$F_T=f_0+\sum_{t=1}^T \rho_t f_t,\quad f_t\approx - \nabla_{F} L\mid_{F=F_{t-1}}, F_{t-1}=\sum_{i=0}^{t-1}f_i.$$
The basic idea of gradient boosts methods is to find
$$f_t=\arg\min_{f\in\mathcal F}L(F_{t-1}+f).$$
so that $L(F_t)\leq L(F_{t-1})$. In some sense, it requires the model $f$ is easy to construct.


* [OPTIMIZATION BY GRADIENT BOOSTING](http://www.lsta.upmc.fr/BIAU/bc2.pdf)
* [boosting as optimization](https://metacademy.org/graphs/concepts/boosting_as_optimization)
* [Boosting, Convex Optimization, and Information Geometry](https://ieeexplore.ieee.org/document/6282239?arnumber=6282239)
* [Generalized Boosting Algorithms for Convex Optimization](https://www.ri.cmu.edu/publications/generalized-boosting-algorithms-for-convex-optimization/)
* [Survey of Boosting from an Optimization Perspective](https://users.soe.ucsc.edu/~manfred/pubs/tut/icml2009/ws.pdf)



______________
Boosting | Optimization
---|---
Decision Tree | Coordinate-wise Optimization
AdaBoost | ???
[Stochastic Gradient Boost](https://statweb.stanford.edu/~jhf/ftp/stobst.pdf) | Stochastic Gradient Descent
Gradient Boost |  Gradient Descent
Accelerated Gradient Boosting | Accelerated Gradient Descent
xGBoost | Newton's Methods
??? | Mirror Gradient Descent
??? | ADMM


AdaBoost stepwise minimizes a function
$$L(G_t)=\sum_{n=1}^N \exp(-\mathrm{y}_n G_t(x_n))$$
The gradient of $L(G_t)$ gives the example weights used for AdaBoost:
$$\frac{\partial L(G_t)}{\partial G_t(x_n)}=-\mathrm{y}_n\exp(-\mathrm{y}_nG_t(x_n)).$$

Compared with `entropic descent method`, in each iteration of AdaBoost:
$$w_i\leftarrow w_i\exp[-\alpha_t(\mathrm{y}_i G_t(x_i))]>0,  i=1,2,\dots, N, \sum_{n=1}^{N}w_n=1.$$

$\color{red}{Note}$: given the input feature  $x_i$, the label $\mathrm y_i$ is a fixed constant and the model is modified  with the training data set and the distribution $(D, w)$ i.e., $\{(x_i, y_i, w_i)\mid i=1,2,\cdots, N\}$.

* [Boost: Foundations and Algorithms](https://mitpress.mit.edu/sites/default/files/titles/content/boosting_foundations_algorithms/toc.html)
* [机器学习算法中GBDT与Adaboost的区别与联系是什么？](https://www.zhihu.com/question/54626685)
* [Logistic Regression, AdaBoost and Bregman Distances](https://link.springer.com/article/10.1023/A:1013912006537)

#### Translation Optimization Methods to Boost Algorithm

The  following steps are the keys to a constructed a decision tree in gardient boost methods:
+   For $i = 1, 2,\dots , n$ compute
    $$r_{i,t}=-{[\frac{\partial L(\mathrm{y}_i, f(x_i))}{\partial f(x_i)}]}_{f=f_{t-1}}.$$
+  Fit a regression tree to the targets $r_{i,t}$   giving **terminal regions**
   $$R_{j,m}, j = 1, 2,\dots , J_m. $$

Here we compute the gradient of loss function with respective to each prediction $f(x_i)$ and it is why we call it `gradient boost`.
If we  fit a regression tree to a subsample of  the targets $r_{i,t}$ randomly, it is `stochastic gradient boost`.
All variants of gradient boost methods mainly modify $\fbox{the steps to construct a new decision tree}$. And it is trained in additive way.

Mirror gradient descent update formulea can be transferred to be

$$
r_{i,t} = f(x_i)\exp(-\alpha [\frac{\partial L(\mathrm{y}_i, f(x_i))}{\partial f(x_i)}])\mid_{f=f_{t-1}}.
$$

then fit a regression tree to the targets $r_{i,t}$.

----
What is the alternative of gradient descent  in order to combine `ADMM` as an operator splitting methods for numerical optimization and `Boosting` such as gradient boosting/extreme gradient boosting?
Can we do leaves splitting and optimization in the same stage?

The core transfer from ADMM to Boost is how to change the original optimization to one linearly constrained  convex optimization  so that it adjusts to ADMM:  

$$
\arg\min_{f_{t}\in\mathcal F}\sum_{i=1}^{n} \ell[y_i,\hat{y}^{(t-1)}_i + f_t(x_i)] + \gamma T +\frac{\lambda}{2}{\sum}_{i=1}^{T}w_i^2 \iff \fbox{???} \quad ?
$$
where $f_t(x)={\sum}_{i=1}^{T}w_i\mathbb{I}(q(x)=i)$ is a decision tree.

In a compact form, we rewrite the above problem as
$$\arg\min_{f_t\in\mathcal F}L(F_{t-1}+f_t) + Regularier(f_t)$$
where $Regularier(f_t)=\gamma T +\frac{\lambda}{2}{\sum}_{i=1}^{T}w_i^2$.

It is similar to some regularized cost function to minimize:

$$\arg\min_{f_t\in\mathcal F}L(F_{t-1}+f_t)+ \mathcal R(f_t)\approx \\
\arg\min_{f_t\in\mathcal F} L(F_{t-1}+f_t), s.t. \mathcal R(f_t)\leq c.$$

$\color{red}{Note}$ that $F_t=F_{t-1} + f_t=\left<(1,\dots,1),(f_0,\dots, f_t)\right>$,i.e.,it is a linear combination.

If we want to use ADMM, the regular term $\mathcal R(f_t)$ must be written in `linear constraints`.

It seems attractive to me to understand the analogy between
$\fbox{operator splitting in ADMM}$ and $\fbox{leaves splitting in Decision Tree}$.

To be more general, how to connect the numerical optimization methods such as fixed pointed iteration methods and the boosting algorithms?
Is it possible to combine $\fbox{Anderson Acceleration}$ and $\fbox{Gradinet Boosting}$ ?  

Another interesting question is how to boost the composite/multiplicative models rather than the additive model?

### Deep Gradient Boosting

[It](https://arxiv.org/pdf/1907.12608.pdf) shows that each iteration of the backpropagation algorithm can be viewed as fitting a
weak linear regressor to the gradients at the output of each layer, before non-linearity is applied.
We call this approach `deep gradient boosting (DGB)`, since it’s effectively a `layer-wise boosting approach`
where the typical decision trees are replaced by linear regressors. Under this model, SGD naturally emerges as an extreme case where the network weights are highly regularized, in the L2 norm sense.
In addition, DGB takes into account the correlations between training samples (features), just like regular regression would, when calculating the weight updates while SGD does not.
Intuitively, it makes sense to ignore the correlations between training samples considering that the most difficult
test samples would be the ones that show low correlations with the training set.

This work suggests an alternative explanation for why SGD generalizes so well when training neural networks.
We show that each iteration of the backpropagation algorithm can be viewed as fitting a weak linear regressor to the gradients at the output of each layer, before non-linearity is applied.

The classic backpropagation algorithm minimizes an error function $E$ of a multi-layer neural network using gradient descent and the chain rule.
The resulting weight updates at a given layer are
$$\Delta w_{ij} = x_i y_j$$

where $x_i$ is the output from the previous layer at node $i$ while $y_j=\frac{\partial E}{\partial \text{net}_j}$ is the derivative with respect to the input at node $j$ calculated using the chain rule from the last layer to the current one.
We can interpret $y_i$ as a pseudo-residual and infer the weight updates  $\Delta w_{ij}=v_{ij}$ such that $\sum_{i} x_{i} v_{i, j}=y_{j},\forall j$.

For the extreme case of a single sample update we need to use L2 regularization which leads to the following optimization problem:

$$\begin{array}{ll}{\underset{v_{i, j}}{\operatorname{minimize}}} & {\frac{1}{2} \sum_{i} v_{i, j}^{2}} \\
{\text { subject to }} & {\sum_{i} x_{i} v_{i, j}=y_{j}, \quad \forall j}\end{array}
$$


- https://arxiv.org/pdf/1907.12608.pdf


- http://www.drryanmc.com/

### The Generic Leveraging Algorithm

Let us assume the loss function $G(f, D)$ has the following additive form
$$G(f, D)=\sum_{n=1}^{N} g(f(x_n), y_n),$$
and we would like to solve the optimization problem
$$\min_{f\in\mathcal F}G(f, D)=\min_{w}\sum_{n=1}^{N} g(f_w(x_n), y_n).$$
And $g^{\prime}(f_w(x_n), y_n))=\frac{\partial g(f_w(x_n), y_n)}{\partial f_w(x_n)}$ for $n=1,2,\cdots, N$.

[`Leveraging methods` are designed under a subsampling framework, in which one samples a small proportion of the data (subsample) from the full sample, and then performs intended computations for the full sample using the small subsample as a surrogate. The key of the success of the leveraging methods is to construct nonuniform sampling probabilities so that influential data points are sampled with high probabilities](http://homepages.math.uic.edu/~minyang/Big%20Data%20Discussion%20Group/Leveraging%20for%20big%20data%20regression.pdf)

$\fbox{Leveraging = Boosting without PAC Boosting property}.$

***
* Input  $D=\{ (x_i, y_i)\}_{i=1}^{N}$;Loss function $G:\mathbb{R}^n\to\mathbb{R}$ .
* Initialize the observation weights $f_o=0, d_n^1=g^{\prime}(f_0(x_n), y_n), n=1, 2, \dots, N$.
* For $t = 1, 2, \dots, T$:
   +  Train classifier on $\{D, \mathbf d^t\}$ and obtain hypothesis $h_t:\mathbb{X}\to\mathbb{Y}$
   +  Set $\alpha_t=\arg\min_{\alpha\in\mathbb{R}}G[f_t + \alpha h_t]$
   +  Update $f_{t+1} = f_t + {\alpha_t}h_t$ and $d_n^{t+1}=g^{\prime}(f_{t+1}(x_n), y_n), n=1, 2, \dots, N$
* Output $f_T$.

Here $\mathbf d^t=(d^t_1, d^t_2, \cdots, d^t_N)$ for $t=1,2,\cdots, T$.

______

* [An Introduction to Boosting and Leveraging](http://face-rec.org/algorithms/Boosting-Ensemble/8574x0tm63nvjbem.pdf)
* [FACE RECOGNITION HOMEPAGE](http://face-rec.org/algorithms/)
* [Leveraging for big data regression](http://homepages.math.uic.edu/~minyang/Big%20Data%20Discussion%20Group/Leveraging%20for%20big%20data%20regression.pdf)
* [A Statistical Perspective on Algorithmic Leveraging](http://www.jmlr.org/papers/v16/ma15a.html)
* http://homepages.math.uic.edu/~minyang/BD.htm


### Matrix Multiplicative Weight Algorithms

`Matrix Multiplicative Weight` can be considered as an ensemble method of optimization methods.
The name “multiplicative weights” comes from how we implement the last step: if the weight of the chosen object at step $t$ is $w_t$ before the event, and $G$ represents how well the object did in the event, then we’ll update the weight according to the rule:
$$
w_{t+1}=w_{t}(1+G).
$$

> ![Matrix Multiplicative Weight](https://pic3.zhimg.com/80/v2-bb705627cf962661e5eedfc78c3420aa_hd.jpg)

[Jeremy](https://jeremykun.com/) wrote a blog on this topic:

> In general we have some set $X$ of objects and some set $Y$ of “event outcomes” which can be completely independent. If these sets are finite, we can write down a table M whose rows are objects, whose columns are outcomes, and whose $i,j$ entry $M(i,j)$ is the reward produced by object $x_i$ when the outcome is $y_j$. We will also write this as $M(x, y)$ for object $x$ and outcome $y$. The only assumption we’ll make on the rewards is that the values $M(x, y)$ are bounded by some small constant $B$ (by small I mean $B$ should not require exponentially many bits to write down as compared to the size of $X$). In symbols, $M(x,y) \in [0,B]$. There are minor modifications you can make to the algorithm if you want negative rewards, but for simplicity we will leave that out. Note the table $M$ just exists for analysis, and the algorithm does not know its values. Moreover, while the values in $M$ are static, the choice of outcome $y$ for a given round may be nondeterministic.

> The `MWUA` algorithm randomly chooses an object $x \in X$ in every round, observing the outcome $y \in Y$, and collecting the reward $M(x,y)$ (or losing it as a penalty). The guarantee of the MWUA theorem is that the expected sum of rewards/penalties of MWUA is not much worse than if one had picked the best object (in hindsight) every single round.

**Theorem (from [Arora et al](https://www.cs.princeton.edu/~arora/pubs/MWsurvey.pdf)):** The cumulative reward of the MWUA algorithm is, up to constant multiplicative factors, at least the cumulative reward of the best object minus $\log(n)$, where $n$ is the number of objects.

+ [The Reasonable Effectiveness of the Multiplicative Weights Update Algorithm](https://jeremykun.com/tag/multiplicative-weights-update-algorithm/)
+ [Matrix Multiplicative Weight （1）](https://zhuanlan.zhihu.com/p/47423225)
+ [Matrix Multiplicative Weight （2）](https://zhuanlan.zhihu.com/p/47891504)
+ [Matrix Multiplicative Weight （3）](https://zhuanlan.zhihu.com/p/48084069)
+ [The Multiplicative Weights Update framework](https://nisheethvishnoi.files.wordpress.com/2018/05/lecture42.pdf)
+ [The Multiplicative Weights Update Method: a Meta Algorithm and Applications](https://www.cs.princeton.edu/~arora/pubs/MWsurvey.pdf)
+ [Nonnegative matrix factorization with Lee and Seung's multiplicative update rule](https://www.wikiwand.com/en/Non-negative_matrix_factorization)
+ [A Combinatorial, Primal-Dual approach to Semidefinite Programs](http://www.satyenkale.com/papers/mmw.pdf)
+ [Milosh Drezgich, Shankar Sastry. "Matrix Multiplicative Weights and Non-Zero Sum Games".](https://ptolemy.berkeley.edu/projects/chess/pubs/780.html)
+ [The Matrix Multiplicative Weights Algorithm for Domain Adaptation by David Alvarez Melis](https://people.csail.mit.edu/davidam/assets/publications/MS_thesis/MSThesis.pdf)
+ https://lcbb.epfl.ch/algs16/notes/notes-04-14.pdf

#### Application

[News](https://catboost.ai/news) lists some news on CatBoost.
See [XGBoost Resources Page](https://github.com/dmlc/xgboost/blob/master/demo/README.md) for a complete list of use cases of XGBoost, including machine learning challenge winning solutions, data science tutorials and industry adoptions.

* [拍拍贷教你如何用GBDT做评分卡](http://www.sfinst.com/?p=1389)
* [LambdaMART 不太简短之介绍](https://liam.page/2016/07/10/a-not-so-simple-introduction-to-lambdamart/)
* https://catboost.ai/news
* [Finding Influential Training Samples for Gradient Boosted Decision Trees](https://research.yandex.com/publications/151)
* [Parallel Boosted Regression Trees for Web Search Ranking](https://www.cse.wustl.edu/~kunal/resources/Papers/boost.pdf)
+ [Efficient, reliable and fast high-level triggering using a bonsai boosted decision tree](http://inspirehep.net/record/1193348)
+ [CERN boosts its search for antimatter with Yandex’s MatrixNet search engine tech](https://www.extremetech.com/extreme/147320-cern-boosts-its-search-for-antimatter-with-yandexs-matrixnet-search-engine-tech)
+ [MatrixNet as a specific Boosted Decision Tree algorithm which is available as a service](https://github.com/yandex/rep/blob/master/rep/estimators/matrixnet.py)
+ [Bagging and Boosting statistical machine translation systems](http://www.nlplab.com/papers/AI2013-Xiao-Zhu-Liu.pdf)
+ [A Novel, Gradient Boosting Framework for Sentiment Analysis in Languages where NLP Resources Are Not Plentiful: A Case Study for Modern Greek ](https://www.mdpi.com/1999-4893/10/1/34/htm)
+ [EGBMMDA: Extreme Gradient Boosting Machine for MiRNA-Disease Association prediction](https://www.nature.com/articles/s41419-017-0003-x.pdf?origin=ppub)
+ [Awesome gradient boosting](https://github.com/talperetz/awesome-gradient-boosting)

### Selective Ensemble


[Selective ensemble naturally bears two goals simultaneously, i.e., maximizing the generalization performance and minimizing the number of learners. When pushing to the limit, the two goals are conflicting, as overly fewer individual learners lead to poor performance. To achieve both good performance and a small ensemble size, previous selective ensemble approaches solve some objectives that mix the two goals.](https://link.springer.com/chapter/10.1007/978-981-13-5956-9_13)

- [Selective Ensemble of Decision Trees](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/rsfdgrc03.pdf)
- [Growing and Pruning Selective Ensemble Regression over Drifting Data Stream](http://www.auto.shu.edu.cn/info/1125/7181.htm)
- [Selective Ensemble under Regularization Framework](https://link.springer.com/chapter/10.1007/978-3-642-02326-2_30)
- [Selecting a representative decision tree from an ensemble of decision-tree models for fast big data classification](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0186-3)

<img src="https://media.springernature.com/full/springer-static/image/art%3A10.1186%2Fs40537-019-0186-3/MediaObjects/40537_2019_186_Figa_HTML.png" width="70%"/>

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

[When a lot of different models are applied to a data simultaneously then such a method of meta-ensemble modeling is known as Stacking. Here, there is no single function, rather we have meta-level where a function is used to combine the outputs of different functions. Thus the information from various models is combined to come up with a unique model. This is among the most advanced form of data modeling used commonly in data hackathons and other online competitions where maximum accuracy is required. Stacking models can have multiple levels and can be made very complex by using various combinations of features and algorithms. There are many forms of Stacking method and in this blog post, a stacking method known as blending has been explored.](https://www.datavedas.com/ensemble-methods/)

* http://www.machine-learning.martinsewell.com/ensembles/stacking/
* https://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/
* [Stacked Generalization (Stacking)](http://www.machine-learning.martinsewell.com/ensembles/stacking/)
* [Stacking与神经网络 - 微调的文章 - 知乎](https://zhuanlan.zhihu.com/p/32896968)
* [Blending and deep learning](http://jtleek.com/advdatasci/17-blending.html)
* http://www.chioka.in/stacking-blending-and-stacked-generalization/
* https://blog.csdn.net/willduan1/article/details/73618677
* [今我来思，堆栈泛化(Stacked Generalization)](https://www.jianshu.com/p/46ccf40222d6)
* [我爱机器学习:集成学习（一）模型融合与Bagging](https://www.hrwhisper.me/machine-learning-model-ensemble-and-bagging/)
* https://github.com/ikki407/stacking
* [Spatial Pyramids and Two-layer Stacking SVM Classifiers for Image Categorization: A Comparative Study](https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/spatial-pyramid-two-layer.pdf)
* [Cascaded classifiers and stacking methods for classification of pulmonary nodule characteristics](https://www.sciencedirect.com/science/article/pii/S0169260718304413)
* [Python package for stacking (machine learning technique)](https://github.com/vecxoz/vecstack)
* [Signal Processing and Pattern Recognition Laboratory](http://users.rowan.edu/~polikar/spprl.html)
* https://blog.csdn.net/mrlevo520/article/details/78161590

**Issues in Stacked Generalization**

[Stacked generalization is a general method of using a high-level model to combine lower-level models to achieve greater predictive accuracy. In this paper we address two crucial issues which have been considered to be a `black art' in classification tasks ever since the introduction of stacked generalization in 1992 by Wolpert: the type of generalizer that is suitable to derive the higher-level model, and the kind of attributes that should be used as its input. We find that best results are obtained when the higher-level model combines the confidence (and not just the predictions) of the lower-level ones. We demonstrate the effectiveness of stacked generalization for combining three different types of learning algorithms for classification tasks. We also compare the performance of stacked generalization with majority vote and published results of arcing and bagging.](https://arxiv.org/abs/1105.5466)

####  Linear Blending

There is an alternative of bagging called `combining ensemble method`. It trains a linear combination of learner:
$$F = \sum_{i=1}^{n} w_i F_i$$
where the weights $w_i\geq 0, \sum_{i=1}^{n} w_i =1$. The weights $w=\{w_i\}_{i=1}^{n}$ are solved by minimizing the ensemble error
$$
w = \arg\min_{w}\sum_{k}^{K}(F(x_k)-y_k)^{2}
$$
if the training data set $\{x_k, y_k\}_{k=1}^{K}$ is given.


<img title = weighted-unweighted src=https://blogs.sas.com/content/subconsciousmusings/files/2017/05/weighted-unweighted.png width=80%/>



In the sense of stacking, deep neural network is thought as the stacked `logistic regression`. And `Boltzman machine` can be stacked in order to construct more expressive model for discrete random variables.

<img src="http://www.chioka.in/wp-content/uploads/2013/09/stacking.png" width="80%" />

- [Mixture of Experts](http://www.scholarpedia.org/article/Ensemble_learning)
- [Hierarchical Mixture of Experts and the EM Algorithms](https://cs.nyu.edu/~roweis/csc2515-2006/readings/hme.pdf)

$\fbox{partition + stacking}$: Different data activates different algorithms.

#### Deep Forest

[In this paper, we propose gcForest, a decision tree ensemble approach with performance highly competitive to deep neural networks.](https://arxiv.org/abs/1702.08835v2)

<img title="Deep Forest" src="https://raw.githubusercontent.com/DataXujing/Cos_pic/master/pic2.png" width="80%" />

* [Deep forest](http://lamda.nju.edu.cn/code_gcForest.ashx?AspxAutoDetectCookieSupport=1)
* https://github.com/kingfengji/gcForest
* [周志华团队和蚂蚁金服合作：用分布式深度森林算法检测套现欺诈](https://zhuanlan.zhihu.com/p/37492203)
* [Multi-Layered Gradient Boosting Decision Trees](https://arxiv.org/abs/1806.00007)
* [Deep Boosting: Layered Feature Mining for General Image Classification](https://arxiv.org/abs/1502.00712)
* [gcForest 算法原理及 Python 与 R 实现](https://cosx.org/2018/10/python-and-r-implementation-of-gcforest/)

*****
* https://www.wikiwand.com/en/Ensemble_learning
* https://www.toptal.com/machine-learning/ensemble-methods-machine-learning
* https://machinelearningmastery.com/products/
* https://blog.csdn.net/willduan1/article/details/73618677#
* http://www.scholarpedia.org/article/Ensemble_learning
* https://arxiv.org/abs/1505.01866
* http://users.rowan.edu/~polikar/publications.html
* [CAREER: An Ensemble of Classifiers Based Approach for Incremental Learning](https://www.nsf.gov/awardsearch/showAward?AWD_ID=0239090)
