### Decision Tree


A decision tree is a set of questions(i.e. if-then sentence) organized in a **hierarchical** manner and represented graphically as a tree.
It use 'divide-and-conquer' strategy recursively. It is easy to scale up to massive data set. The models are obtained by recursively partitioning
the data space and fitting a simple prediction model within each partition. As a
result, the partitioning can be represented graphically as a decision tree.
[Visual introduction to machine learning](https://explained.ai/decision-tree-viz/index.html) show an visual introduction to decision tree.

***
**Algorithm**  Pseudocode for tree construction by exhaustive search

1. Start at root node.
2. For each node X, find the set $S$ that **minimizes** the sum of the node impurities in the two child nodes and choose the split $\{X^{\star}\in S^{\star}\}$ that gives the minimum overall $X$ and $S$.
3. If a stopping criterion is reached, exit. Otherwise, apply step 2 to each child node in turn.

***

Creating a binary decision tree is actually a process of dividing up the input space according to the sum of **impurities**.

This learning process is to minimize the impurities.
C4.55 and CART6 are two later classification
tree algorithms that follow this approach. C4.5 uses
`entropy` for its impurity function, whereas CART
uses a generalization of the binomial variance called the `Gini index`.

If the training set $D$ is divided into subsets $D_1,\dots,D_k$, the entropy may be
reduced, and the amount of the reduction is the information gain,

$$
G(D;D_1,\dots,D_k)=Ent(D)-\sum_{i=1}^{k}\frac{|D_k|}{|D|}Ent(D_k)
$$

where $Ent(D)$, the entropy of $D$, is defined as

$$
Ent(D)=\sum_{y\in Y}P(y|D)\log(\frac{1}{P(y|D)}).
$$


The information gain ratio of features $A$ with respect of data set $D$  is defined as

$$
g_{R}(D,A)=\frac{G(D,A)}{Ent(D)}.
$$
And another option of impurity is Gini index of probability $p$:

$$
Gini(p)=\sum_{y}p_y (1-p_y)=1-\sum_{y}p_y^2.
$$

***

Like other supervised algorithms, decision tree makes a trade-off between over-fitting and under-fitting and how to choose the hyper-parameters of decision tree such as the max depth?
The regularization techniques in regression may not suit the tree algorithms such as LASSO.

**Pruning** is a regularization technique for tree-based algorithm. In arboriculture, the reason to prune tree is [because each cut has the potential to change the growth of the tree, no branch should be removed without a reason. Common reasons for pruning are to remove dead branches, to improve form, and to reduce risk. Trees may also be pruned to increase light and air penetration to the inside of the tree’s crown or to the landscape below. ](https://www.treesaregood.org/treeowner/pruningyourtrees)

![](https://www.treesaregood.org/portals/0/images/treeowner/pruning1.jpg)

In machine learning, we prune the decision tree to make a balance between overfitting and underfitting. The important step of tree pruning is to define a criterion be used to determine the correct final tree size using one of the following methods:		

1. Use a distinct dataset from the training set (called validation set), to evaluate the effect of post-pruning nodes from the tree.
2. Build the tree by using the training set, then apply a statistical test to estimate whether pruning or expanding a particular node is likely to produce an improvement beyond the training set.
    * Error estimation
    * Significance testing (e.g., Chi-square test)
3. Minimum Description Length principle : Use an explicit measure of the complexity for encoding the training set and the decision tree, stopping growth of the tree when this encoding size (size(tree) + size(misclassifications(tree)) is minimized.

- https://www.saedsayad.com/decision_tree_overfitting.htm
- http://www.cs.bc.edu/~alvarez/ML/statPruning.html

***

When the height of a decision tree is limited to 1, i.e., it takes only one
test to make every prediction, the tree is called a decision stump. While decision trees are nonlinear classifiers in general, decision stumps are a kind
of linear classifiers.
[Fifty Years of Classification and
Regression Trees](http://www.stat.wisc.edu/~loh/treeprogs/guide/LohISI14.pdf) and [the website of Wei-Yin Loh](http://www.stat.wisc.edu/~loh/guide.html) helps much understand the decision tree.
Multivariate Adaptive Regression
Splines(MARS) is the boosting ensemble methods for decision tree algorithms.

***

* https://www.benkuhn.net/tree-imp
* https://www.wikiwand.com/en/Decision_tree_learning
* [An Introduction to Recursive Partitioning: Rationale, Application and Characteristics of Classification and Regression Trees, Bagging and Random Forests](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2927982/)
* https://www.wikiwand.com/en/Decision_tree
* https://www.wikiwand.com/en/Recursive_partitioning
* http://ai-depot.com/Tutorial/DecisionTrees-Partitioning.html
* https://www.ncbi.nlm.nih.gov/pubmed/16149128
* [ADAPTIVE CONCENTRATION OF REGRESSION TREES, WITH APPLICATION TO RANDOM FORESTS](https://arxiv.org/pdf/1503.06388.pdf)
* http://www.stat.wisc.edu/~loh/guide.html
* https://explained.ai/decision-tree-viz/index.html
* http://www.cnblogs.com/en-heng/p/5035945.html
* https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/
* (http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
* https://christophm.github.io/interpretable-ml-book/tree.html
* https://dinh-hung-tu.github.io/tree-based-models/

#### Random Forest

Random forests (Breiman, 2001) is a substantial modification of bagging
that builds a large collection of de-correlated trees, and then averages them.
On many problems the performance of random forests is very similar to boosting, and they are simpler to train and tune.

***

* For $t=1, 2, \dots, T$:
    + Draw a bootstrap sample $Z^{\star}$ of size $N$ from the training data.
    + Grow a random-forest tree $T_t$ to the bootstrapped data, by recursively repeating the following steps for each terminal node of the tree, until the minimum node size $n_{min}$ is reached.
      - Select $m$ variables at random from the $p$ variables.
      - Pick the best variable/split-point among the $m$.
      - Split the node into two daughter nodes.
* Vote for classification and average for regression.

![](https://dimensionless.in/wp-content/uploads/RandomForest_blog_files/figure-html/voting.png)
***

* https://mi2datalab.github.io/randomForestExplainer/index.html
* https://github.com/kjw0612/awesome-random-forest
* https://blog.datadive.net/interpreting-random-forests/
* https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
* https://dimensionless.in/author/raghav/
* http://www.rhaensch.de/vrf.html
* https://www.wikiwand.com/en/Random_forest
* https://sktbrain.github.io/awesome-recruit-en.v2/
* https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
* https://dimensionless.in/introduction-to-random-forest/https://www.elderresearch.com/blog/modeling-with-random-forests

## Ensemble methods

There are many competing techniques for solving the problem, and each technique is characterized
by choices and meta-parameters: when this flexibility is taken into account, one easily
ends up with a very large number of possible models for a given task.


* [Zhou Zhihua's publication on ensemble methods](http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/publication_toc.htm#Ensemble%20Learning)
* https://mlwave.com/kaggle-ensembling-guide/
* https://www.springer.com/us/book/9781441993250
* https://web.stanford.edu/~hastie/THESES/gareth_james.pdf
* https://liam.page/2016/07/10/a-not-so-simple-introduction-to-lambdamart/
* [Neural Random Forests](https://arxiv.org/abs/1604.07143)
* [Generalized Random Forests](https://arxiv.org/abs/1610.01271)
* [Additive Models, Boosting, and Inference for Generalized Divergences ](https://www.stat.berkeley.edu/~binyu/summer08/colin.bregman.pdf)
* [Boosting as Entropy Projection](https://users.soe.ucsc.edu/~manfred/pubs/C51.pdf)

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

![](https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2016/10/bootstrap-sample.png)

It is a sample-based ensemble method.

There is an alternative of bagging called combining ensemble method. It trains a linear combination of learner:
$$F = \sum_{i=1}^{n} w_i F_i$$
where the weights $w_i\geq 0, \sum_{i=1}^{n} w_i =1$. The weights $w=\{w_i\}_{i=1}^{n}$ are solved by minimizing the ensemble error
$$
w = \arg\min_{w}\sum_{k}^{K}(F(x_k)-y_k)^{2}
$$
if the training data set $\{x_k, y_k\}_{k=1}^{K}$ is given.


![](https://blogs.sas.com/content/subconsciousmusings/files/2017/05/weighted-unweighted.png)

***

* http://www.machine-learning.martinsewell.com/ensembles/bagging/
* https://www.cnblogs.com/earendil/p/8872001.html
* https://www.wikiwand.com/en/Bootstrap_aggregating
* [Bagging Regularizes](http://dspace.mit.edu/bitstream/handle/1721.1/7268/AIM-2002-003.pdf?sequence=2)


### Boosting

The term boosting refers to a family of algorithms that are able to convert weak learners to strong learners.
It is kind of similar to the "trial and error" scheme: if we know that the learners perform worse at some given data set $S$,
the learner may pay more attention to the data drawn from $S$.
For the regression problem, of which the output results are continuous, it  progressively reduce the error by trial.
In another word, we will reduce the error at each iteration.

![](http://www.stat.ucla.edu/~sczhu/Vision_photo/Chinese_herb_clinic.jpg)

[Reweighting with Boosted Decision Trees](https://arogozhnikov.github.io/2015/10/09/gradient-boosted-reweighter.html)

* https://betterexplained.com/articles/adept-method/
* https://web.stanford.edu/~hastie/Papers/buehlmann.pdf
* https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/
* http://www.machine-learning.martinsewell.com/ensembles/boosting/
* [Boosting at Wikipedia](https://www.wikiwand.com/en/Boosting_(machine_learning))

#### AdaBoost

AdaBoost is a boosting methods for supervised classification algorithms, so that the labeled data set is given in the form $D=\{ (x_i, \mathrm{d}_i)\}_{i=1}^{N}$.
AdaBoost is to change the distribution of training data and learn from the shuffled data.
It is an iterative trial-and-error in some sense.


***

* Initialize the observation weights ${w}_i=\frac{1}{N}, i=1, 2, \dots, N$;
* For $t = 1, 2, \dots, T$:
   +  Fit a classifier $G_m(x)$ to the training data using weights $w_i$;
   +  Compute
      $$err_{t}=\frac{\sum_{i=1}^{N}\mathbb{I}(G_t(x_i) \not= \mathrm{d}_i)}{\sum_{i=1}^{N} w_i}.$$
   +  Compute $\alpha_t = \log(\frac{1-err_t}{err_t})$.
   +  Set $w_i\leftarrow w_i\exp[\alpha_t\mathbb{I}(\mathrm{d}_i\not=G_t(x_i))], i=1,2,\dots, N$.
* Output $G(x)=sign[\sum_{t=1}^{T}\alpha_{t}G_t(x)]$.

The indicator function $\mathbb{I}(x\neq y)$ is defined as
$$
\mathbb{I}(x\neq y)=
  \begin{cases}
    1, \text{if $x\neq y$} \\
    0, \text{otherwise}.
  \end{cases}
$$

![](https://arogozhnikov.github.io/images/reweighter/1-reweighting.png)
***

* [AdaBoost at Wikipedia](https://www.wikiwand.com/en/AdaBoost)
* [CSDN blog](https://blog.csdn.net/v_july_v/article/details/40718799)

#### Gradient Boosting Decision Tree

One of the frequently asked questions is `What's the basic idea behind gradient boosting?` and the answer from [https://explained.ai/gradient-boosting/faq.html] is the best one I know:
> Instead of creating a single powerful model, boosting combines multiple simple models into a single **composite model**. The idea is that, as we introduce more and more simple models, the overall model becomes stronger and stronger. In boosting terminology, the simple models are called weak models or weak learners.
> To improve its predictions, gradient boosting looks at the difference between its current approximation,$\hat{y}$ , and the known correct target vector, $y$, which is called the residual, $y-\hat{y}$. It then trains a weak model that maps feature vector $x$  to that residual vector. Adding a residual predicted by a weak model to an existing model's approximation nudges the model towards the correct target. Adding lots of these nudges, improves the overall models approximation.

|Gradient Boosting|
|:---:|
|![golf](https://explained.ai/gradient-boosting/images/golf-MSE.png)|

It is the first solution to the question that if weak learner is equivalent to strong learner.

***

We may consider the generalized additive model, i.e.,

$$
\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)
$$

where $\{f_k\}_{k=1}^{K}$ is regression decision tree rather than polynomial.

The objective function is given by

$$
obj = \sum_{i=1}^{n} L(y_i,\hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
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
= \sum_{i=1}^{n} [(y_i - \hat{y}^{(t-1)}_i) f_t(x_i)) +  f_t(x_i)^2 ] + \Omega(f_t) + C^{\prime}
$$

where $C^{\prime}=\sum_{i=1}^{n} (y_i - \hat{y}^{(t-1)}_i)^2 + \sum_{k=1}^{t-1} \Omega(f_k)$.

If there is no regular term $\sum_{k=1}^{t} \Omega(f_k)$, the problem is simplfied to $\arg\min_{f_{t}}\sum_{i=1}^{n} [(y_i - \hat{y}^{(t-1)}_i) f_t(x_i)) +  f_t(x_i)^2 ]$.


***

* Initialize $f_0(x)=\arg\min_{\gamma} L(\mathrm{d}_i,\gamma)$;
* For $t = 1, 2, \dots, T$:
   +  Compute
      $$r_{i,t}=-{[\frac{\partial L(\mathrm{d}_i, f(x_i))}{\partial f(x_i)}]}_{f=f_{t-1}}.$$
   +  Fit a regression tree to the targets $r_{i,t}$   giving terminal regions $R_{j,m}, j = 1, 2,\dots , J_m$.
   +  For $j = 1, 2,\dots , J_m$ compute
      $$\gamma_{j,t}=\arg\min_{\gamma}\sum_{x_i\in R_{j,m}}{L(\mathrm{d}_i,f_{t-1}+\gamma)}$$
* Output $f_T(x)$.

***

An important part of gradient boosting method is regularization by shrinkage which consists in modifying the update rule as follows:
$$
f_{t}=f_{t-1}+\nu\sum_{j = 1}^{J_{m}} \gamma_{j,t} \mathbb{1}_{R_{j,m}}, \nu\in(0,1).
$$

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
* https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
* https://data-flair.training/blogs/gradient-boosting-algorithm/
* https://arxiv.org/abs/1803.02042
* https://statweb.stanford.edu/~jhf/ftp/trebst.pdf
* https://liam.page/2016/07/10/a-not-so-simple-introduction-to-lambdamart/

***

A general gradient descent “boosting” paradigm is
developed for additive expansions based on any fitting criterion. It is not only for the decision tree.

* http://uc-r.github.io/gbm_regression
* https://machinelearningmastery.com/start-with-gradient-boosting/
* https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/

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


In general, we can expand the objective function at up to  the second order

$$
obj^{(t)}
=  \sum_{i=1}^{n} L[y_i,\hat{y}^{(t-1)}_i + f_t(x_i)] + \Omega(f_t) + C \\
\simeq \sum_{i=1}^{n} [L(y_i,\hat{y}^{(t-1)}_i) + g_i f_t(x_i) + \frac{h_i f_t^2(x_i)}{2}] + \Omega(f_t) + C^{\prime}
$$

where $g_i=\partial_{\hat{y}_{i}^{(t-1)}} L(y_i, \hat{y}_{i}^{(t-1)})$, $h_i=\partial^2_{\hat{y}_{i}^{(t-1)}} L(y_i, \hat{y}_{i}^{(t-1)})$.

After we remove all the constants, the specific objective at step ${t}$ becomes
$$
obj^{(t)}\approx \sum_{i=1}^{n} [L(y_i,\hat{y}^{(t-1)}_i) + g_i f_t(x_i) + \frac{h_i f_t^2(x_i)}{2}] + \Omega(f_t)
$$

One important advantage of this definition is that the value of the objective function only depends on $g_i$ and $h_i$. This is how XGBoost supports custom loss functions.

***

* https://xgboost.readthedocs.io/en/latest/tutorials/model.html
* https://xgboost.ai/
* http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/
* https://datascienceplus.com/extreme-gradient-boosting-with-r/
* [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
* [xgboost的原理没你想像的那么难](https://www.jianshu.com/p/7467e616f227)
* https://www.cnblogs.com/wxquare/p/5541414.html
* https://machinelearningmastery.com/visualize-gradient-boosting-decision-trees-xgboost-python/

![](https://pic2.zhimg.com/50/v2-d8191a1191979eadbd4df191b391f917_hd.jpg)

- https://github.com/Microsoft/LightGBM/blob/master/docs/Features.rst
- [Python3机器学习实践：集成学习之LightGBM - AnFany的文章 - 知乎](https://zhuanlan.zhihu.com/p/53583034)
- https://ieeexplore.ieee.org/abstract/document/7929984
- https://tech.yandex.com/catboost/
- https://catboost.ai/
- https://lightgbm.readthedocs.io/en/latest/s


### Stacking

Stacked generalization (or stacking)  is a different way of combining multiple models, that introduces the concept of a meta learner. Although an attractive idea, it is less widely used than bagging and boosting. Unlike bagging and boosting, stacking may be (and normally is) used to combine models of different types.

The procedure is as follows:

1. Split the training set into two disjoint sets.
2. Train several base learners on the first part.
3. Test the base learners on the second part.
4. Using the predictions from 3) as the inputs, and the correct responses as the outputs, train a higher level learner.

Note that steps 1) to 3) are the same as cross-validation, but instead of using a winner-takes-all approach, we train a meta-learner to combine the base learners, possibly non-linearly such as feedforward neural network. It is a little similar with **composition** of functions in mathematics.

![](https://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier_files/stackingclassification_overview.png)

[Stacking, Blending and and Stacked Generalization are all the same thing with different names. It is a kind of ensemble learning.](http://www.chioka.in/stacking-blending-and-stacked-generalization/)

* https://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/
* http://www.machine-learning.martinsewell.com/ensembles/stacking/
* [Stacking与神经网络 - 微调的文章 - 知乎](https://zhuanlan.zhihu.com/p/32896968)
* http://www.chioka.in/stacking-blending-and-stacked-generalization/
* https://blog.csdn.net/willduan1/article/details/73618677
* [今我来思，堆栈泛化(Stacked Generalization)](https://www.jianshu.com/p/46ccf40222d6)

**Deep Forest**

![Deep Forest](https://raw.githubusercontent.com/DataXujing/Cos_pic/master/pic2.png)

* [Deep forest](http://lamda.nju.edu.cn/code_gcForest.ashx?AspxAutoDetectCookieSupport=1)
* https://github.com/kingfengji/gcForest
* https://zhuanlan.zhihu.com/p/37492203
* https://arxiv.org/abs/1806.00007
* https://cosx.org/2018/10/python-and-r-implementation-of-gcforest/

***

* https://www.wikiwand.com/en/Ensemble_learning
* https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/
* https://www.toptal.com/machine-learning/ensemble-methods-machine-learning
* https://machinelearningmastery.com/products/
* https://blog.csdn.net/willduan1/article/details/73618677#
