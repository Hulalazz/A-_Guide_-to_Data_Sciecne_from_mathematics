### Decision Tree

A decision tree is a set of questions(i.e. if-then sentence) organized in a **hierarchical** manner and represented graphically as a tree.
It use 'divide-and-conquer' strategy recursively. It is easy to scale up to massive data set.
[Visual introduction to machine learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/) show an visual introduction to decision tree.

If the training set D is divided into subsets $D_1,\dots,D_k$, the entropy may be
reduced, and the amount of the reduction is the information gain,
$$G(D;D_1,\dots,D_k)=Ent(D)-\sum_{i=1}^{k}\frac{|D_k|}{|D|}Ent(D_k)$$
where $Ent(D)$ is the entropy of $D$ is defined as $Ent(D)=\sum_{y\in Y}P(y|D)\log(\frac{1}{P(y|D)})$.

* https://www.wikiwand.com/en/Decision_tree_learning
* [An Introduction to Recursive Partitioning: Rationale, Application and Characteristics of Classification and Regression Trees, Bagging and Random Forests](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2927982/)
* https://www.wikiwand.com/en/Decision_tree
* https://www.wikiwand.com/en/Recursive_partitioning
* http://ai-depot.com/Tutorial/DecisionTrees-Partitioning.html
* https://www.ncbi.nlm.nih.gov/pubmed/16149128
* [ADAPTIVE CONCENTRATION OF REGRESSION TREES, WITH APPLICATION TO RANDOM FORESTS](https://arxiv.org/pdf/1503.06388.pdf)
* [Neural Random Forests](https://arxiv.org/abs/1604.07143)
* [Generalized Random Forests](https://arxiv.org/abs/1610.01271)
* https://dimensionless.in/author/raghav/
* https://mi2datalab.github.io/randomForestExplainer/index.html
* https://explained.ai/decision-tree-viz/index.html

#### Random Forest

Random forests (Breiman, 2001) is a substantial modification of bagging
that builds a large collection of de-correlated trees, and then averages them.
On many problems the performance of random forests is very similar to
boosting, and they are simpler to train and tune. 

* For $t=1, 2, \dots, T$:
    + Draw a bootstrap sample $Z^{\star}$ of size $N$ from the training data.
    + Grow a random-forest tree $T_t$ to the bootstrapped data, by recursively repeating the following steps for each terminal node of the tree, until the minimum node size $n_{min}$ is reached.
      - Select $m$ variables at random from the $p$ variables.
      - Pick the best variable/split-point among the $m$.
      - Split the node into two daughter nodes.
* Vote for classification and average for regression.

* https://github.com/kjw0612/awesome-random-forest
* https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
* https://dimensionless.in/author/raghav/
* http://www.rhaensch.de/vrf.html
* https://www.wikiwand.com/en/Random_forest
* https://sktbrain.github.io/awesome-recruit-en.v2/
* https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf

## Ensemble methods

* [Zhou Zhihua's publication on ensemble methods](http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/publication_toc.htm#Ensemble%20Learning)
* https://mlwave.com/kaggle-ensembling-guide/
* https://www.springer.com/us/book/9781441993250
* https://web.stanford.edu/~hastie/THESES/gareth_james.pdf
* https://liam.page/2016/07/10/a-not-so-simple-introduction-to-lambdamart/

### Bagging

Bagging, short for 'bootstrap aggregating', is a simple but highly effective ensemble method that creates diverse models on different random bootstrap samples of the original data set.
[Random forest](https://www.wikiwand.com/en/Random_forest) is the application of bagging to decision tree algorithms.

The basic motivation of parallel ensemble methods is to exploit the independence between the
base learners, since the error can be reduced dramatically by combining independent base learners.
Bagging adopts the most popular strategies for aggregating the outputs of
the base learners, that is, voting for classification and averaging for regression.

* Draw Bootstrap samples $B_1, B_2, \dots, B_n$ independently from the original training data set for base learners;
* Train the $i$th base learner $F_i$ at the ${B}_{i}$;
* Voting for classification and average for regression.

![](https://www.statisticshowto.datasciencecentral.com/wp-content/uploads/2016/10/bootstrap-sample.png)

***

* http://www.machine-learning.martinsewell.com/ensembles/bagging/
* https://www.cnblogs.com/earendil/p/8872001.html
* https://www.wikiwand.com/en/Bootstrap_aggregating
* [Bagging Regularizes](http://dspace.mit.edu/bitstream/handle/1721.1/7268/AIM-2002-003.pdf?sequence=2)

### Boosting

[Gradient Boosting Visualized](https://arogozhnikov.github.io/2016/06/24/gradient_boosting_explained.html)

[Reweighting with Boosted Decision Trees](https://arogozhnikov.github.io/2015/10/09/gradient-boosted-reweighter.html)

The term boosting refers to a family of algorithms that are able to convert weak learners to strong learners.
It is kind of similar to the "trial and error" scheme: if we know that the learners perform worse at some given data set $S$,
the learner may pay more attention to the data drawn from $S$.
For the regression problem, of which the output results are continuous, it  progressively reduce the error by trial.
In another word, we will reduce the error at each iteration.


https://betterexplained.com/articles/adept-method/
https://web.stanford.edu/~hastie/Papers/buehlmann.pdf
https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/

* http://www.machine-learning.martinsewell.com/ensembles/boosting/
* [Boosting at Wikipedia](https://www.wikiwand.com/en/Boosting_(machine_learning))

#### AdaBoost

AdaBoost is a boosting methods for supervised classification algorithms, so that the labelled data set is given in the form $D=\{ (x_i, \mathrm{d}_i)\}_{i=1}^{N}$.

***
* Initialize the observation weights ${w}_i=\frac{1}{N}, i=1, 2, \dots, N$;
* For $t = 1, 2, \dots, T$:
   +  Fit a classifier $G_m(x)$ to the training data using weights $w_i$;
   +  Compute
      $$err_{t}=\frac{\sum_{i=1}^{N}\mathbb{I}(G_t(x_i) \not= \mathrm{d}_i)}{\sum_{i=1}^{N} w_i}.$$
   +  Compute $\alpha_t = \log(\frac{1-err_t}{err_t})$.
   +  Set $w_i\leftarrow w_i\exp[\alpha_t\mathbb{I}(\mathrm{d}_i\not=G_t(x_i))], i=1,2,\dots, N$.
* Output $G(x)=sign[\sum_{t=1}^{T}\alpha_{t}G_t(x)]$.

The indictor function $\mathbb{I}(x\not=y)$ is defined as
$$\mathbb{I}(x\not=y)=
\begin{cases}
1, \text{if $x\not= y$} \\
0, \text{otherwise}.
\end{cases}$$
***

* [AdaBoost at Wikipedia](https://www.wikiwand.com/en/AdaBoost)
* [CSDN blog](https://blog.csdn.net/v_july_v/article/details/40718799)

#### Gradient Boosting Decsion Tree

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

* [Gradient Boosting at Wikipedia](https://www.wikiwand.com/en/Gradient_boosting)
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3885826/
* https://explained.ai/gradient-boosting/index.html
* [GBDT算法原理 - 飞奔的猫熊的文章 - 知乎](https://zhuanlan.zhihu.com/p/50176849)
* https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf
* https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
* https://data-flair.training/blogs/gradient-boosting-algorithm/

#### xGBoost

In Gradient Boost, we compute and fit a regression a tree to $r_{i,t}=-{[\frac{\partial L(\mathrm{d}_i, f(x_i))}{\partial f(x_i)}]}_{f=f_{t-1}}$.
Why not the error $L(\mathrm{d}_i, f(x_i))$ itself?
Recall the Taylor expansion $f(x+h) = f(x)+f^{\prime}(x)h + f^{(2)}(x)h^{2}/2!+ \cdots +f^{(n)}(x)h^{(n)}/n!+\cdots$ so that the non-convex error function can be expressed as a polynomial in terms of $h$, which is easier to fit than a general non-convex function.
So that we can implement additive training to boost the supervised algorithm.

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
In general, we can expand the objective function at $2$ed order
$$
obj^{(t)} 
=  \sum_{i=1}^{n} L(y_i,\hat{y}^{(t-1)}_i + f_t(x_i)) + \Omega(f_t) + C \\
\simeq \sum_{i=1}^{n} [L(y_i,\hat{y}^{(t-1)}_i) + g_i f_t(x_i) + \frac{h_i f_t(x_i)^2}{2}] + \Omega(f_t) + C^{\prime}
$$
where $g_i=\partial_{y_{i}^{(t-1)}} L(y_i, y_{i}^{(t-1)})$, $h_i=\partial^2_{y_{i}^{(t-1)}} L(y_i, y_{i}^{(t-1)})$.

***

* https://xgboost.readthedocs.io/en/latest/tutorials/model.html
* https://xgboost.ai/
* http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/
* https://datascienceplus.com/extreme-gradient-boosting-with-r/
* [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
* [xgboost的原理没你想像的那么难](https://www.jianshu.com/p/7467e616f227)
* https://www.cnblogs.com/wxquare/p/5541414.html


### Stacking

Stacked generalization (or stacking)  is a different way of combining multiple models, that introduces the concept of a meta learner. Although an attractive idea, it is less widely used than bagging and boosting. Unlike bagging and boosting, stacking may be (and normally is) used to combine models of different types. The procedure is as follows:

1. Split the training set into two disjoint sets.
2. Train several base learners on the first part.
3. Test the base learners on the second part.
4. Using the predictions from 3) as the inputs, and the correct responses as the outputs, train a higher level learner.

Note that steps 1) to 3) are the same as cross-validation, but instead of using a winner-takes-all approach, we combine the base learners, possibly nonlinearly. It is a little similar with **composition** of functions.

* http://www.machine-learning.martinsewell.com/ensembles/stacking/
* https://www.jianshu.com/p/46ccf40222d6
* [Deep forest](http://lamda.nju.edu.cn/code_gcForest.ashx?AspxAutoDetectCookieSupport=1)
* https://cosx.org/2018/10/python-and-r-implementation-of-gcforest/
* https://blog.csdn.net/willduan1/article/details/73618677

***

* https://www.wikiwand.com/en/Ensemble_learning
* https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/
* https://www.toptal.com/machine-learning/ensemble-methods-machine-learning
* https://machinelearningmastery.com/products/
