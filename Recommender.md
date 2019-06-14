# Recommender System

Recommender Systems (RSs) are software tools and techniques providing suggestions for items to be of use to a user.

RSs are primarily directed towards individuals who lack sufficient personal experience or competence to evaluate the potentially overwhelming number of alternative items that a Web site, for example, may offer.

Xavier Amatriain discusses the traditional definition and its data mining core.

Traditional definition: The **recommender system** is to estimate a utility  function that automatically predicts how a user will like an item.

User Interest is **implicitly** reflected in `Interaction history`, `Demographics` and `Contexts`, which can be regarded as a typical example of data mining. Recommender system should match a context to a collection of information objects. There are some methods called `Deep Matching Models for Recommendation`.
It is an application of machine learning, which is in the *representation + evaluation + optimization* form. And we will focus on the `representation and evaluation`.


- [ ] https://github.com/hongleizhang/RSPapers
- [ ] https://github.com/benfred/implicit
- [ ] https://github.com/YuyangZhangFTD/awesome-RecSys-papers
- [ ] https://github.com/daicoolb/RecommenderSystem-Paper
- [ ] https://github.com/grahamjenson/list_of_recommender_systems
- [ ] https://www.zhihu.com/question/20465266/answer/142867207
- [X] [直接优化物品排序的推荐算法](https://blog.csdn.net/u013166160/article/details/17935193)
- [ ] [推荐系统遇上深度学习](https://www.jianshu.com/c/e12d7195a9ff)
- [ ] [Large-Scale Recommender Systems@UTexas](http://bigdata.ices.utexas.edu/project/large-scale-recommender-systems/)
- [ ] [Alan Said's publication](https://www.alansaid.com/publications.html)
- [ ] [MyMediaLite Recommender System Library](http://www.mymedialite.net/links.html)
- [ ] [Recommender System Algorithms @ deitel.com](http://www.deitel.com/ResourceCenters/Web20/RecommenderSystems/RecommenderSystemAlgorithms/tabid/1317/Default.aspx)
- [ ] [Workshop on Recommender Systems: Algorithms and Evaluation](http://sigir.org/files/forum/F99/Soboroff.html)
- [ ] [Semantic Recommender Systems. Analysis of the state of the topic](https://www.upf.edu/hipertextnet/en/numero-6/recomendacion.html)
- [ ] [Recommender Systems (2019/1)](https://homepages.dcc.ufmg.br/~rodrygo/recsys-2019-1/)
- [ ] [Recommender systems & ranking](https://sites.google.com/view/chohsieh-research/recommender-systems)

**Evaluation of Recommendation System**

The evaluation of machine learning algorithms depends on the tasks.
The evalution of recommendation system can be regarded as some machine learning models such as regression, classification and so on.
We only take the mathematical convenience into consideration in the following methods.
Gini index, covering rate and more realistic factors are not discussed in the following content.
- [Evaluating recommender systems](http://fastml.com/evaluating-recommender-systems/)

## Collaborative Filtering

There are 3 kinds of collaborative filtering: user-based, item-based and model-based collaborative filtering.

The user-based methods are based on the similarities of users. If user ${u}$ and ${v}$ are very similar friends, we may recommend the items which user ${u}$ bought to the user ${v}$ and explains it that your friends also bought it.

The item-based methods are based on the similarity of items. If one person added a brush to shopping-list, it is reasonable to recommend some toothpaste to him or her. And we can explain that you bought item $X$ and the people who bought $X$ also bought $Y$.
And we focus on the model-based collaborative filtering.

- [协同过滤详解](https://www.cnblogs.com/ECJTUACM-873284962/p/8729010.html)
- [深入推荐引擎相关算法 - 协同过滤](https://www.ibm.com/developerworks/cn/web/1103_zhaoct_recommstudy2/index.html)
- http://topgeek.org/blog/2012/02/10/%E6%8E%A2%E7%B4%A2%E6%8E%A8%E8%8D%90%E5%BC%95%E6%93%8E%E5%86%85%E9%83%A8%E7%9A%84%E7%A7%98%E5%AF%86%EF%BC%8C%E7%AC%AC-1-%E9%83%A8%E5%88%86-%E6%8E%A8%E8%8D%90%E5%BC%95%E6%93%8E%E5%88%9D%E6%8E%A2/
- http://topgeek.org/blog/2012/02/13/%E6%8E%A2%E7%B4%A2%E6%8E%A8%E8%8D%90%E5%BC%95%E6%93%8E%E5%86%85%E9%83%A8%E7%9A%84%E7%A7%98%E5%AF%86%EF%BC%8C%E7%AC%AC-2-%E9%83%A8%E5%88%86-%E6%B7%B1%E5%85%A5%E6%8E%A8%E8%8D%90%E5%BC%95%E6%93%8E/


***
Matrix completion is to complete the matrix $X$ with missing elements, such as

$$
\min_{Z} Rank(Z) \\
s.t. \sum_{(i,j):Observed}(Z_{(i,j)}-X_{(i,j)})^2\leq \delta
$$

Note that the rank of a matrix is not easy or robust  to compute.

We can apply [customized PPA](http://maths.nju.edu.cn/~hebma/Talk/Unified_Framework.pdf) to matrix completion problem

$$
\min \{ {\|Z\|}_{\star}\} \\
s.t. Z_{\Omega} = X_{\Omega}
$$

We let ${Y}\in\mathbb{R}^{n\times n}$ be the the Lagrangian multiplier to the constraints $Z_{\Omega} = X_{\Omega}$
and Lagrange function is
$$
L(Z,Y) = {\|Z\|}_{\star} - Y(Z_{\Omega} - X_{\Omega}).
$$

1. Producing $Y^{k+1}$ by
   $$Y^{k+1}=\arg\max_{Y} {L([2Z^k-Z^{k-1}],Y)-\frac{s}{2}\|Y-Y^k\|};$$
2. Producing $Z^{k+1}$ by
    $$Z^{k+1}=\arg\min_{Z} {L(Z,Y^{k+1}) + \frac{r}{2}\|Z-Z^k\|}.$$

![ADMM](https://pic3.zhimg.com/80/dc9a2b89742a05c3cd2f025105ba1c4a_hd.png)


[Rahul Mazumder, Trevor Hastie, Robert Tibshirani](http://www.jmlr.org/papers/v11/mazumder10a.html) reformulate it as the following:

$$
\min f_{\lambda}(Z)=\frac{1}{2}{\|P_{\Omega}(Z-X)\|}_F^2 + \lambda {\|Z\|}_{\star}
$$

where $X$ is the observed matrix, $P_{\Omega}$ is a projector and ${\|\cdot\|}_{\star}$ is the nuclear norm of matrix.


* [A SINGULAR VALUE THRESHOLDING ALGORITHM FOR MATRIX COMPLETION](https://www.zhihu.com/question/47716840/answer/110843844)
* [Matrix and Tensor Decomposition in Recommender Systems](http://delab.csd.auth.gr/papers/RecSys2016s.pdf)
* [Low-Rank Matrix Recovery](http://www.princeton.edu/~yc5/ele538b_sparsity/lectures/matrix_recovery.pdf)
* [ECE 18-898G: Special Topics in Signal Processing: Sparsity, Structure, and Inference Low-rank matrix recovery via nonconvex optimization](https://users.ece.cmu.edu/~yuejiec/ece18898G_notes/ece18898g_nonconvex_lowrank_recovery.pdf)

![](https://pic3.zhimg.com/80/771b16ac7e7aaeb50ffd8a8f5cf4e582_hd.png)

* http://people.eecs.berkeley.edu/~yima/
* [New tools for recovering low-rank matrices from incomplete or corrupted observations by Yi Ma@UCB](http://people.eecs.berkeley.edu/~yima/matrix-rank/home.html)
* [Matrix Completion/Sensing as NonConvex Optimization Problem](http://sunju.org/research/nonconvex/)
* [Exact Matrix Completion via Convex Optimization](http://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf)
* [A SINGULAR VALUE THRESHOLDING ALGORITHM FOR MATRIX COMPLETION](http://statweb.stanford.edu/~candes/papers/SVT.pdf)
* [Customized PPA for convex optimization](http://maths.nju.edu.cn/~hebma/Talk/Unified_Framework.pdf)
* [Matrix Completion.m](http://www.convexoptimization.com/wikimization/index.php/Matrix_Completion.m)

**Maximum Margin Matrix Factorization**

> A  novel approach to collaborative prediction is presented, using low-norm instead of low-rank factorizations. The approach is inspired by, and has strong connections to, large-margin linear discrimination. We show how to learn low-norm factorizations by solving a semi-definite program, and present generalization error bounds based on analyzing the Rademacher complexity of low-norm factorizations.

Consider the soft-margin learning, where we minimize a trade-off between the trace norm of $Z$ and its
hinge-loss relative to $X_O$:
$$
\min_{Z} { \| Z \| }_{\Omega} + c \sum_{(ui)\in O}\max(0, 1 - Z_{ui}X_{ui}).
$$

And it can be rewritten  as  a semi-definite optimization problem (SDP):
$$
\min_{A, B} \frac{1}{2}(tr(A)+tr(B))+c\sum_{(ui)\in O}\xi_{ui}, \\
s.t.  \, \begin{bmatrix} A & X \\ X^T & B \\ \end{bmatrix} \geq 0, Z_{ui}X_{ui}\geq 1- \xi_{ui},
\xi_{ui}>0 \,\forall ui\in O
$$
where $c$ is a trade-off constant.

- [Maximum Margin Matrix Factorization](https://ttic.uchicago.edu/~nati/Publications/MMMFnips04.pdf)
- [Fast Maximum Margin Matrix Factorization for Collaborative Prediction](https://ttic.uchicago.edu/~nati/Publications/RennieSrebroICML05.pdf)
- [Maximum Margin Matrix Factorization by Nathan Srebro](https://ttic.uchicago.edu/~nati/mmmf/)

This technique is also called **nonnegative matrix factorization**.

The data sets we more frequently encounter in collaborative prediction problem are of `ordinal ratings` $X_{ij} \in \{1, 2, \dots, R\}$ such as $\{1, 2, 3, 4, 5\}$.
To relate the real-valued $Z_{ij}$ to the
discrete $X_{ij}$. we use $R − 1$ thresholds $\theta_{1}, \dots, \theta_{R-1}$.

***

If we have collected user ${u}$'s explicit evaluation score to the item ${i}$ ,  $R_{[u][i]}$, and all such data makes up a matrix $R=(R_{[u][i]})$ while the user $u$ cannot evaluate all the item so that the matrix is incomplete and missing much data.
**SVD** is to factorize the matrix into the multiplication of matrices so that
$$
\hat{R} = P^{T}Q.
$$

And we can predict the score $R_{[u][i]}$ via
$$
\hat{R}_{[u][i]} = \hat{r}_{u,i} = \left<P_u,Q_i\right> = \sum_f p_{u,f} q_{i,f}
$$

where $P_u, Q_i$ is the ${u}$-th column of ${P}$ and the ${i}$-th column of ${Q}$, respectively.
And we can define the cost function

$$
C(P,Q) = \sum_{(u,i):Observed}(r_{u,i}-\hat{r}_{u,i})^{2}=\sum_{(u,i):Observed}(r_{u,i}-\sum_f p_{u,f}q_{i,f})^{2}\\
\arg\min_{P_u, Q_i} C(P, Q)
$$

where $\lambda_u$ is always equal to $\lambda_i$.

Additionally, we can add regular term into the cost function to void over-fitting

$$
C(P,Q) = \sum_{(u,i):Observed}(r_{u,i}-\sum_f p_{u,f}q_{i,f})^{2}+\lambda_u\|P_u\|^2+\lambda_i\|Q_i\|^2.
$$

It is called [the regularized singular value decomposition](https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/Regular-Paterek.pdf)  or **Regularized SVD**.


**Funk-SVD** considers the user's preferences or bias.
It predicts the scores by
$$
\hat{r}_{u,i} = \mu + b_u + b_i + \left< P_u, Q_i \right>
$$
where $\mu, b_u, b_i$ is biased mean, biased user, biased item, respectively.
And the cost function is defined as
$$
\min\sum_{(u,i): Observed}(r_{u,i} - \hat{r}_{u,i})^2 + \lambda (\|P_u\|^2+\|Q_i\|^2+\|b_i\|^2+\|b_u\|^2).
$$

**SVD ++** predicts the scores by

$$
\hat{r}_{u,i} = \mu + b_u + b_i + (P_u + |N(u)|^{-0.5}\sum_{i\in N(u)} y_i) Q_i^{T}
$$
where $y_j$ is the implicit  feedback of item ${j}$ and $N(u)$ is user ${u}$'s item set.
And it can decompose into 3 parts:

* $\mu + b_u + b_i$ is the base-line prediction;
* $\left<P_u, Q_i\right>$ is the SVD of rating matrix;
* $\left<|N(u)|^{-0.5}\sum_{i\in N(u)} y_i, Q_i\right>$ is the implicit feedback where $N(u)$ is user ${u}$'s item set, $y_j$ is the implicit feedback of item $j$.

We learn the values of involved parameters by minimizing the regularized squared error function.

* [Biased Regularized Incremental Simultaneous Matrix Factorization@orange3-recommender](https://orange3-recommendation.readthedocs.io/en/latest/scripting/rating.html)
* [SVD++@orange3-recommender](https://orange3-recommendation.readthedocs.io/en/latest/widgets/svdplusplus.html)
* [矩阵分解之SVD和SVD++](https://cloud.tencent.com/developer/article/1107364)
* [SVD++：推荐系统的基于矩阵分解的协同过滤算法的提高](https://www.bbsmax.com/A/KE5Q0M9ZJL/)
* https://zhuanlan.zhihu.com/p/42269534

One possible improvement of this cost function is that we may design more appropriate loss function other than the squared  error function.

![utexas.edu](http://bigdata.ices.utexas.edu/wp-content/uploads/2015/09/IMC.png)

**Inductive Matrix Completion (IMC)** is an algorithm for recommender systems with side-information of users and items. The IMC formulation incorporates features associated with rows (users) and columns (items) in matrix completion, so that it enables predictions for users or items that were not seen during training, and for which only features are known but no dyadic information (such as ratings or linkages).

IMC assumes that the associations matrix is generated by applying feature vectors associated with
its rows as well as columns to a low-rank matrix ${Z}$.
The goal is to recover ${Z}$ using observations from ${P}$.

The  inputs $x_i, y_j$ are feature vectors.
The entry $P_{(i, j)}$ of the matrix is modeled as $P_{(i, j)}=x_i^T Z  y_j$ and ${Z}$ is to recover in the form of $Z=WH^T$.

$$
\min \sum_{(i,j)\in \Omega}\ell(P_{(i,j)}, x_i^T W H^T y_j) + \frac{\lambda}{2}(\| W \|^2+\| H \|^2)
$$
The loss function $\ell$ penalizes the deviation of estimated entries from the observations.
And $\ell$ is diverse such as the squared error $\ell(a,b)=(a-b)^2$, the logistic error $\ell(a,b) = \log(1 + \exp(-ab))$.

* [Inductive Matrix Completion for Recommender Systems with Side-Information](http://bigdata.ices.utexas.edu/software/inductive-matrix-completion/)
* [Inductive Matrix Completion for Predicting Gene-Diseasev Associations](http://www.cs.utexas.edu/users/inderjit/public_papers/imc_bioinformatics14.pdf)

**Probabilistic Matrix Factorization**

|Regularized SVD|
|---------------|
|$C(P,Q) = \sum_{(u,i):Observed}(r_{(u,i)}-\sum_f p_{(u,f)} q_{(i,f)})^{2}+\lambda_u\|P_u\|^2+\lambda_i\|Q_i\|^2$|

|Probabilistic model|
|------------------|
|$r_{u,i}\sim N(\sum_f p_{(u,f)} q_{(i,f)},\sigma^2), P_u\sim N(0,\sigma_u^2 I), Q_i\sim N(0,\sigma_i^2 I)$|

And $\sigma_u^2$ and $\sigma_i^2$ is related with the regular term $\lambda_u$ and $\lambda_u$.

So that we can reformulate the optimization problem as maximum likelihood estimation.

* [Latent Factor Models for Web Recommender Systems](http://www.ideal.ece.utexas.edu/seminar/LatentFactorModels.pdf)
* [Regression-based Latent Factor Models@CS 732 - Spring 2018 - Advanced Machine Learning by Zhi Wei](https://web.njit.edu/~zhiwei/CS732/papers/Regression-basedLatentFactorModels_KDD2009.pdf)

**BellKor's Progamatic Chaos**

Until now, we consider the recommendation task as a regression prediction process, which is really common in machine learning.
The boosting or stacking methods may help us to enhance these methods.

> A key to achieving highly competitive results on the Netflix data is usage of sophisticated blending schemes, which combine the multiple individual predictors into a single final solution. This significant component was managed by our colleagues at the Big Chaos team. Still, we were producing a few blended solutions, which were later incorporated as individual predictors in the final blend. Our blending techniques were applied to three distinct sets of predictors. First is a set of 454 predictors, which represent all predictors of the BellKor’s Pragmatic Chaos team for which we have matching Probe and Qualifying results. Second, is a set of 75 predictors, which the BigChaos team picked out of the 454 predictors by forward selection. Finally, a set of 24 BellKor predictors for which we had matching Probe and Qualifying results. from [Netflix Prize.](https://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf)



* https://www.netflixprize.com/community/topic_1537.html
* https://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf
* https://www.netflixprize.com/assets/GrandPrize2009_BPC_BigChaos.pdf

***
Another advantage of collaborative filtering or matrix completion is that even the element of matrix is binary or implicit information such as

* [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf),
* [Applications of the conjugate gradient method for implicit feedback collaborative filtering](http://rs1.sze.hu/~gtakacs/download/recsys_2011_draft.pdf),
* [Intro to Implicit Matrix Factorization](https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/)
* [a curated list in github.com](https://github.com/benfred/implicit).

**Recommendation with Implict Information**

|Explicit and implicit feedback|
|:---:|
|![](https://www.msra.cn/wp-content/uploads/2018/06/knowledge-graph-in-recommendation-system-i-8.png)|

**WRMF** is simply a modification of this loss function:
$$
C(P,Q)_{WRMF} = \sum_{(u,i):Observed}c_{u,i}(I_{u,i} - \sum_f p_{u,f}q_{i,f})^{2} + \lambda_u\|P_u\|^2 + \lambda_i\|Q_i\|^2.
$$

We make the assumption that if a user has interacted at all with an item, then $I_{u,i} = 1$. Otherwise, $I_{u,i} = 0$.
If we take $d_{u,i}$ to be the number of times a user ${u}$ has clicked on an item ${i}$ on a website, then
$$c_{u,i}=1+\alpha d_{u,i}$$
where $\alpha$ is some hyperparameter determined by cross validation.
The new  term in cost function $C=(c_{u,i})$ is called confidence matrix.

WRMF does not make the assumption that a user who has not interacted with an item does not like the item. WRMF does assume that that user has a negative preference towards that item, but we can choose how confident we are in that assumption through the confidence hyperparameter.

[Alternating least square](http://suo.im/4YCM5f) (**ALS**) can give an analytic solution to this optimization problem by setting the gradients equal to 0s.


* http://nicolas-hug.com/blog/matrix_facto_1
* http://nicolas-hug.com/blog/matrix_facto_2
* http://nicolas-hug.com/blog/matrix_facto_3
* [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf)
* [Alternating Least Squares Method for Collaborative Filtering](https://bugra.github.io/work/notes/2014-04-19/alternating-least-squares-method-for-collaborative-filtering/)
* [Implicit Feedback and Collaborative Filtering](http://datamusing.info/blog/2015/01/07/implicit-feedback-and-collaborative-filtering/)

**Collaborative Less-is-More Filtering**

Sometimes, the information of user we could collect is implicit such as the clicking at some item.

In `CLiMF` [the model parameters are learned by directly maximizing the Mean Reciprocal Rank (MRR).](https://github.com/gamboviol/climf)

Its objective function is
$$
F(U,V)=\sum_{i=1}^{M}\sum_{j=1}^{N} Y_{ij} [\ln g(U_{i}^{T}V_{j})+\sum_{k=1}^{N}\ln (1 - Y_{ij} g(U_{i}^{T}V_{k}-U_{i}^{T}V_{j}))] \\-\frac{\lambda}{2}({\|U\|}^2 + {\|V\|}^2)
$$

where ${M, N}$ is the number of users and items, respectively. Additionally, $\lambda$ denotes the regularization coefficient and $Y_{ij}$ denotes the binary relevance score of item ${j}$ to user ${i}$, i.e.,$Y_{ij} = 1$ if item ${j}$ is relevant to user ${j}$, 0 otherwise. The function $g$ is logistic function $g(x)=\frac{1}{1+\exp(-x)}$.
The vector $U_i$ denotes a d-dimensional latent factor vector for
user ${i}$, and $V_j$ a d-dimensional latent factor vector for item ${i}$.

|Numbers||Factors||Others||
|:------:|---|:---:|---|:---:|---|
|$M$|the number of users|$U_i$|latent factor vector for user ${i}$|$Y_{ij}$|binary relevance score|
|$N$|the number of items|$V_j$|latent factor vector for item ${i}$|$f$|logistic function|


We use stochastic gradient ascent to maximize the objective function.

* [Collaborative Less-is-More Filtering@orange3-recommendation](https://orange3-recommendation.readthedocs.io/en/latest/scripting/ranking.html)
* https://dl.acm.org/citation.cfm?id=2540581
* [Collaborative Less-is-More Filtering python Implementation](https://github.com/gamboviol/climf)
* [CLiMF: Collaborative Less-Is-More Filtering](https://www.ijcai.org/Proceedings/13/Papers/460.pdf)


***

* https://www.cnblogs.com/Xnice/p/4522671.html
* https://blog.csdn.net/turing365/article/details/80544594
* https://en.wikipedia.org/wiki/Collaborative_filtering
* \url{https://www.wikiwand.com/en/Matrix_factorization_(recommender_systems)}
* http://www.cnblogs.com/DjangoBlog/archive/2014/06/05/3770374.html
* https://www.acemap.info/author/page?AuthorID=7E61F31B
* [Fast Python Collaborative Filtering for Implicit Feedback Datasets](https://github.com/benfred/implicit)
* https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/
* https://www.ethanrosenthal.com/2016/11/07/implicit-mf-part-2/
* https://www.benfrederickson.com/matrix-factorization/
* https://www.benfrederickson.com/fast-implicit-matrix-factorization/
* https://www.benfrederickson.com/implicit-matrix-factorization-on-the-gpu/
* [Top-N Recommendations from Implicit Feedback Leveraging Linked Open Data ?](https://core.ac.uk/display/23873231)
****


**Hyperbolic Recommender Systems**

Many well-established recommender systems are based on representation learning in Euclidean space.
In these models, matching functions such as the Euclidean distance or inner product are typically used for computing similarity scores between user and item embeddings. This paper investigates the notion of learning
user and item representations in hyperbolic space.

Given a user ${u}$ and an item ${v}$ that are both lying in the Poincare ball $B^n$,
the distance between two points on *P* is given by
$$d_p(x, y)=cosh^{-1}(1+2\frac{\|(x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}).$$

HyperBPR leverages BPR pairwise learning to minimize the pairwise ranking loss between the positive and negative items.
Given a user ${u}$ and an item ${v}$ that are both lying in Poincare ball $B^n$, we take:
$$\alpha(u, v) = f(d_p(u,v)).$$
The objective function is defined as follows:
$$\arg\min_{\Theta} \sum_{i,j,k} -\ln(\sigma\{\alpha(u_i, v_j) - \alpha(u_i, v_k)\}) + \lambda  {\|\Theta\|}_2^2$$

where $(i, j, k)$ is the triplet that belongs to the set ${D}$ that
contains all pairs of positive and negative items for each
user; $\sigma$ is the logistic sigmoid function; $\Theta$ represents the model parameters; and $\lambda$ is the regularization parameter.

The parameters of our model are learned by using `RSGD`.

* https://arxiv.org/abs/1809.01703
* https://arxiv.org/abs/1902.0864
* https://arxiv.org/abs/1111.5280


## Deep Learning and Recommender System

Deep learning is powerful in processing visual and text information so that it helps to find the interests of users such as
[Deep Interest Network](http://www.cnblogs.com/rongyux/p/8026323.html), [xDeepFM](https://www.jianshu.com/p/b4128bc79df0)  and more.

Deep learning models for recommender system may come from the restricted Boltzman machine.
And deep learning models are powerful information extractors.
Deep learning is really popular in recommender system such as [spotlight](https://github.com/maciejkula/spotlight).

***

**Factorization Machines(FM)**

The matrix completion used in recommender system are linear combination of some features such as regularized SVD.
The model equation for a factorization machine of degree ${d = 2}$ is defined as
$$
\hat{y}
= w_0 + \sum_{i=1}^{n} w_i x_i+\sum_{i=1}^{n}\sum_{j=i+1}^{n}\left<v_i, v_j\right> x_i x_j\\
= w_0  + \left<w, x\right> + \sum_{i=1}^{n}\sum_{j=i+1}^{n}\left<v_i, v_j\right> x_i x_j
$$

where the model parameters that have to be estimated are
$$
w_0 \in \mathbb{R}, w\in\mathbb{R}^n, V\in\mathbb{R}^{n\times k}.
$$

And $\left<\cdot,\cdot\right>$ is the dot (inner) product of two vectors so that $\left<v_i, v_j\right>=\sum_{f=1}^{k}v_{i,f} \cdot v_{j,f}$.
A row $v_i$ within ${V}$ describes the ${i}$-th latent variable with ${k}$ factors for $x_i$.

And the linear regression $w_0 + \sum_{i=1}^{n} w_i x_i$ is called `the first order part`; the pair-wise interactions between features
$\sum_{i=1}^{n}\sum_{j=i+1}^{n}\left<v_i, v_j\right> x_i x_j$ is called the `second order part`.

* https://blog.csdn.net/g11d111/article/details/77430095
* [Factorization Machines for Recommendation Systems](https://getstream.io/blog/factorization-recommendation-systems/)
* http://www.52caml.com/head_first_ml/ml-chapter9-factorization-family/
* https://www.cnblogs.com/pinard/p/6370127.html

**Field-aware Factorization Machine(FFM)**

In FMs, every feature has only one latent vector to learn the latent effect with any other features.
In FFMs, each feature has several latent vectors. Depending on the field of other features, one of them is used to do the inner product.
Mathematically,
$$
\hat{y}=\sum_{j_1=1}^{n}\sum_{j_2=i+1}^{n}\left<v_{j_1,f_2}, v_{j_2,f_1}\right> x_{j_1} x_{j_2}
$$
where $f_1$ and $f_2$ are respectively the fields of $j_1$ and $j_2$.
* https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf
* https://huangzhanpeng.github.io/2018/01/04/Field-aware-Factorization-Machines-for-CTR-Prediction/
* https://blog.csdn.net/mmc2015/article/details/51760681

**Wide & Deep Model**

The output of this model is
$$
P(Y=1|x) = \sigma(W_{wide}^T[x,\phi(x)] + W_{deep}^T \alpha^{(lf)}+b)
$$
where the `wide` part deal with the categorical features such as user demographics and the `deep` part deal with continuous features.


![](https://upload-images.jianshu.io/upload_images/1500965-13fa11d119bb20b7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)

![](http://kubicode.me/img/Take-about-CTR-With-Deep-Learning/fnn_pnn_wdl.png)

* https://arxiv.org/pdf/1606.07792.pdf
* https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html
* https://www.jianshu.com/p/dbaf2d9d8c94
* https://www.sohu.com/a/190148302_115128

<img src = http://kubicode.me/img/Take-about-CTR-With-Deep-Learning/dcn_arch.png width=60%/>

**Deep FM**

DeepFM ensembles FM and DNN and to learn both second order and higher-order feature interactions:
$$\hat{y}=\sigma(y_{FM} + y_{DNN})$$
where $\sigma$ is the sigmoid function so that $\hat{y}\in[0, 1]$ is the predicted CTR, $y_{FM}$ is the output of
FM component, and $y_{DNN}$ is the output of deep component.

<img src = https://pic3.zhimg.com/v2-c0b871f214bdae6284e98989dc8ac99b_1200x500.jpg width=60%/>

The **FM component** is a factorization machine and the output of FM is the summation of
an `Addition` unit and a number of `Inner Product` units:

$$
\hat{y}
= \left<w, x\right>+\sum_{j_1=1}^{n}\sum_{j_2=i+1}^{n}\left<v_i, v_j\right> x_{j_1} x_{j_2}.
$$

The **deep component** is a `feed-forward neural network`, which is used to learn high-order feature interactions. There is a personal guess that the component function in activation function $e^x$ can expand in the polynomials form $e^x=1+x+\frac{x^2}{2!}+\dots,+\frac{x^n}{n!}+\dots$, which include all the order of interactions.

We would like to point out the two interesting features of this network structure:

1) while the lengths of different input field vectors can be different, their embeddings are of the same size $(k)$;
2) the latent feature vectors $(V)$ in FM now server as network weights which are learned and used to compress the input field vectors to the embedding vectors.

It is worth pointing out that FM component and deep component share the same feature embedding, which brings two important benefits:

1) it learns both low- and high-order feature interactions from raw features;
2) there is no need for expertise feature engineering of the input.

![](http://kubicode.me/img/Deep-in-out-Wide-n-Deep-Series/deepfm_arch.png)
* https://zhuanlan.zhihu.com/p/27999355
* https://zhuanlan.zhihu.com/p/25343518
* https://zhuanlan.zhihu.com/p/32127194
* https://arxiv.org/pdf/1703.04247.pdf
* https://blog.csdn.net/John_xyz/article/details/78933253#deep-fm

**Neural Factorization Machines**

$$
\hat{y} = w_0 + \left<w,x\right> + f(x)
$$
where the first and second terms are the linear regression part similar to that for FM, which models global bias of data and weight
of features. The third term $f(x)$ is the core component of NFM
for modelling feature interactions, which is a `multi-layered feedforward neural network`.

`B-Interaction Layer` including `Bi-Interaction Pooling` is an innovation in artificial neural network.

![https://i.ooxx.ooo](https://i.ooxx.ooo/2017/12/27/ab7149f31f904f8f2bd6f15e0b9900c9.png)

* https://www.comp.nus.edu.sg/~xiangnan/papers/sigir17-nfm.pdf
* http://staff.ustc.edu.cn/~hexn/
* https://github.com/hexiangnan/neural_factorization_machine

**Attentional Factorization Machines**

Attentional Factorization Machine (AFM) learns the importance of each feature interaction from data via a neural attention network.

We employ the attention mechanism on feature interactions by performing a weighted sum on the interacted vectors:

$$\sum_{(i,j)} a_{(i,j)}(V_i\odot V_j)x_i x_j$$

where $a_{i,j}$ is the attention score for feature interaction.

![](https://deepctr-doc.readthedocs.io/en/latest/_images/AFM.png)

* https://www.comp.nus.edu.sg/~xiangnan/papers/ijcai17-afm.pdf
* http://blog.leanote.com/post/ryan_fan/Attention-FM%EF%BC%88AFM%EF%BC%89

**xDeepFM**

`Compressed Interaction Network(CIN)`

![](https://www.msra.cn/wp-content/uploads/2018/08/kdd-2018-xdeepfm-5.png)

- [X] [KDD 2018 | 推荐系统特征构建新进展：极深因子分解机模型](https://www.msra.cn/zh-cn/news/features/kdd-2018-xdeepfm)
- [ ] https://arxiv.org/abs/1803.05170
- [ ] http://kubicode.me/2018/09/17/Deep%20Learning/eXtreme-Deep-Factorization-Machine/
- [ ] [推荐系统遇上深度学习(二十二)--DeepFM升级版XDeepFM模型强势来袭！](https://www.jianshu.com/p/b4128bc79df0)

**Restricted Boltzmann Machines for Collaborative Filtering(RBM)**

Let ${V}$ be a $K\times m$ observed binary indicator matrix with $v_i^k = 1$ if the user rated item ${i}$ as ${k}$ and ${0}$ otherwise.
We also let $h_j$, $j = 1, \dots, F,$ be the binary values of hidden (latent) variables, that can be thought of as representing
stochastic binary features that have different values for different users.

We use a conditional multinomial distribution (a “softmax”) for modeling each column of the observed
"visible" binary rating matrix ${V}$ and a conditional
Bernoulli distribution for modeling "hidden" user features *${h}$*:
$$
p(v_i^k = 1 | h) = \frac{\exp(b_i^k+\sum_{j=1}^{F}h_j W_{i,j}^{k})}{\sum_{l=1}^{K}\exp(b_i^k+\sum_{j=1}^{F}h_j W_{i,j}^{l})} \\
p( h_j = 1 | V) = \sigma(b_j + \sum_{i=1}^{m}\sum_{k=1}^{K} v_i^k W_{i,j}^k)
$$
where $\sigma(x)=\frac{1}{1+exp(-x)}$ is the logistic function, $W_{i,j}^{k}$ is is a symmetric interaction parameter between feature
${j}$ and rating ${k}$ of item ${i}$, $b_i^k$ is the bias of rating ${k}$ for item ${i}$, and $b_j$ is the bias of feature $j$.

The marginal distribution over the visible ratings ${V}$ is
$$
p(V) = \sum_{h}\frac{\exp(-E(V,h))}{\sum_{V^{\prime},h^{\prime}} \exp(-E(V^{\prime},h^{\prime}))}
$$
with an "energy" term given by:

$$
E(V,h) = -\sum_{i=1}^{m}\sum_{j=1}^{F}\sum_{k=1}^{K}W_{i,j}^{k} h_j v_i^k - \sum_{i=1}^{m}\sum_{k=1}^{K} v_i^k b_i^k -\sum_{j=1}^{F} h_j b_j.
$$
The items with missing ratings do not make any contribution to the energy function

The parameter updates required to perform gradient ascent in the log-likelihood  over the visible ratings ${V}$ can be obtained
$$
\Delta W_{i,j}^{k} = \epsilon \frac{\partial\log(p(V))}{\partial W_{i,j}^{k}}
$$
where $\epsilon$ is the learning rate.
The authors put a `Contrastive Divergence` to approximate the gradient.

We can also model “hidden” user features h as Gaussian latent variables:
$$
p(v_i^k = 1 | h) = \frac{\exp(b_i^k+\sum_{j=1}^{F}h_j W_{i,j}^{k})}{\sum_{l=1}^{K}\exp(b_i^k+\sum_{j=1}^{F}h_j W_{i,j}^{l})} \\
p( h_j = 1 | V) = \frac{1}{\sqrt{2\pi}\sigma_j} \exp(\frac{(h - b_j -\sigma_j \sum_{i=1}^{m}\sum_{k=1}^{K} v_i^k W_{i,j}^k)^2}{2\sigma_j^2})
$$
where $\sigma_j^2$ is the variance of the hidden unit ${j}$.


* https://www.cnblogs.com/pinard/p/6530523.html
* https://www.cnblogs.com/kemaswill/p/3269138.html
* https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf
* http://www.cs.toronto.edu/~fritz/absps/cdmiguel.pdf
* http://deeplearning.net/tutorial/rbm.html
* [RBM notebook form Microsoft](https://github.com/Microsoft/Recommenders/blob/master/notebooks/00_quick_start/rbm_movielens.ipynb)

**AutoRec**

[AutoRec](http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf) is a novel `autoencoder` framework for collaborative filtering (CF). Empirically, AutoRec’s
compact and efficiently trainable model outperforms state-of-the-art CF techniques (biased matrix factorization, RBMCF and LLORMA) on the Movielens and Netflix datasets.

Formally, the objective function for the Item-based AutoRec (I-AutoRec) model is, for regularisation strength $\lambda > 0$,

$$
\min_{\theta}\sum_{i=1}^{n} {\|r^{i}-h(r^{i}|\theta)\|}_{O}^2 +\frac{1}{2}({\|W\|}_F^{2}+ {\|V\|}_F^{2})
$$

where $\{r^{i}\in\mathbb{R}^{d}, i=1,2,\dots,n\}$ is partially observed vector and ${\| \cdot \|}_{o}^2$ means that we only consider the contribution of observed ratings.
The function $h(r|\theta)$ is  the reconstruction of input $r\in\mathbb{R}^{d}$:

$$
h(r|\theta) = f(W\cdot g(Vr+\mu)+b)
$$

for for activation functions $f, g$ as described in  dimension reduction. Here $\theta = \{W,V,r,b\}$.

* https://blog.csdn.net/studyless/article/details/70880829
* http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf

***
![http://kubicode.me](http://kubicode.me/img/More-Session-Based-Recommendation/repeatnet_arch.png)

****

* https://github.com/hwwang55/DKN
* https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
* http://lipixun.me/2018/02/01/youtube
* https://www.cnblogs.com/pinard/p/6370127.html
* https://www.jianshu.com/p/6f1c2643d31b
* https://blog.csdn.net/John_xyz/article/details/78933253
* http://kubicode.me/2018/02/23/Deep%20Learning/Deep-in-out-Factorization-Machines-Series/
* https://zhuanlan.zhihu.com/p/38613747
* https://www.infosec-wiki.com/?p=394011
* http://danyangliu.me/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9B%B8%E5%85%B3%E6%8E%A8%E8%8D%90%E6%A8%A1%E5%9E%8B/
* https://amundtveit.com/2016/11/20/recommender-systems-with-deep-learning/
* http://kubicode.me/2018/10/25/Deep%20Learning/More-Session-Based-Recommendation/

**Deep Geometric Matrix Completion**

It’s easy to observe how better matrix completions can be achieved by considering the sparse matrix as defined over two different graphs:
a user graph and an item graph. From a signal processing point of view, the matrix ${X}$
can be considered as a bi-dimensional signal defined over two distinct domains.
Instead of recurring to multigraph convolutions realized over the entire matrix ${X}$, two
independent single-graph GCNs (graph convolution networks) can be applied on matrices ${W}$ and ${H}$.

Given the aforementioned multi-graph convolutional layers,
the last step that remains concerns the choice of the architecture to use for reconstructing the missing information.
Every (user, item) pair in the multi-graph approach and every user/item in the separable
one present in this case an independent state, which is updated (at every step) by means of the features produced by
the selected GCN.

* [graph convolution network有什么比较好的应用task？ - superbrother的回答 - 知乎](https://www.zhihu.com/question/305395488/answer/554847680)
* https://arxiv.org/abs/1704.06803
* http://www.ipam.ucla.edu/abstract/?tid=14552&pcode=DLT2018
* http://helper.ipam.ucla.edu/publications/dlt2018/dlt2018_14552.pdf

**Deep Matching Models for Recommendation**

It is essential for the recommender system  to find the item which matches the users' demand. Its difference from web search is that recommender system provides item information even if the users' demands or generally interests are not provided.
It sounds like modern crystal ball to read your mind.

In [A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems](http://sonyis.me/paperpdf/frp1159-songA-www-2015.pdf) the authors propose to extract rich features from user’s browsing
and search histories to model user’s interests. The underlying assumption is that, users’ historical online activities
reflect a lot about user’s background and preference, and
therefore provide a precise insight of what items and topics users might be interested in.

* http://sonyis.me/dnn.html
* https://akmenon.github.io/
* https://sigir.org/sigir2018/program/tutorials/
* https://www.comp.nus.edu.sg/~xiangnan/papers/www18-tutorial-deep-matching.pdf
* http://www.hangli-hl.com/uploads/3/4/4/6/34465961/wsdm_2019_workshop.pdf
* https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1159-songA.pdf
* http://www.wanghao.in/CDL.htm

**Social Recommendation**

We present a novel framework for studying recommendation algorithms in terms of the
‘jumps’ that they make to connect people to artifacts. This approach emphasizes reachability via an algorithm within the `implicit graph structure` underlying a recommender
dataset and allows us to consider questions relating algorithmic parameters to properties of the datasets.

- [ ] http://dmml.asu.edu/smm/slides/
- [ ] http://dmml.asu.edu/smm/slide/SMM-Slides-ch9.pdf
- [ ] https://arxiv.org/pdf/1304.3405.pdf

**Knowledge Graph and Recommender System**

- [ ] https://www.msra.cn/zh-cn/news/features/embedding-knowledge-graph-in-recommendation-system-i
- [ ] https://www.msra.cn/zh-cn/news/features/embedding-knowledge-graph-in-recommendation-system-ii
- [ ] https://www.msra.cn/zh-cn/news/features/explainable-recommender-system-20170914

**Reinforcement Learning and RecSys**

* [Deep Reinforcement Learning for Page-wise Recommendations](https://arxiv.org/abs/1805.02343)
* [Generative Adversarial User Model for Reinforcement Learning Based Recommendation System](https://arxiv.org/abs/1812.10613)
+ [Adversarial Personalized Ranking for Recommendation](http://bio.duxy.me/papers/sigir18-adversarial-ranking.pdf)
+ [Adversarial Training Towards Robust Multimedia Recommender System](https://github.com/duxy-me/AMR)
_____________
|Evolution of the Recommender Problem|
|:---:|
|Rating|
|Ranking|
|Page Optimization|
|Context-aware Recommendations|

- [ ] [Deep Learning Meets Recommendation Systems](https://nycdatascience.com/blog/student-works/deep-learning-meets-recommendation-systems/)
- [ ] [Using Keras' Pretrained Neural Networks for Visual Similarity Recommendations](https://www.ethanrosenthal.com/2016/12/05/recasketch-keras/)
- [ ] https://tech.meituan.com/2018/06/07/searchads-dnn.html
- [ ] [Recommending music on Spotify with deep learning](http://benanne.github.io/2014/08/05/spotify-cnns.html)
_______
|Traditional Approaches | Beyond Traditional Methods|
|---------------------- |--------------------------|
|Collaborative Filtering | Tensor Factorization & Factorization Machines|
|Content-Based Recommendation | Social Recommendations|
|Item-based Recommendation | Learning to rank|
|Hybrid Approaches | MAB Explore/Exploit|

#### Ensemble Methods for RecSys

The RecSys can be considered as some regression or classification tasks, so that we can apply the ensemble methods to these methods as  `BellKor's Progamatic Chaos` used the blended solution to win the prize.
In fact, its essence is bagging or blending, which is one sequential ensemble strategy in order to avoid over-fitting or reduce the variance.

In this section, the boosting is the focus, which is to reduce the error and boost the performance from a weaker learner.

There are two common methods to construct a stronger learner from a weaker learner: (1) rewight the samples and learn from the error: AdaBoosting; (2) retrain another learner and learn to approximate the error: Gradient Boosting. 

- [General Functional Matrix Factorization Using Gradient Boosting](http://w.hangli-hl.com/uploads/3/1/6/8/3168008/icml_2013.pdf)

**BoostFM**

- [BoostFM: Boosted Factorization Machines for Top-N Feature-based Recommendation](http://wnzhang.net/papers/boostfm.pdf)

**Adaptive Boosting Personalized Ranking (AdaBPR)**

`AdaBPR (Adaptive Boosting Personalized Ranking)` is a boosting algorithm for top-N item recommendation using users' implicit feedback.
In this framework, multiple homogeneous component recommenders are linearly combined to achieve more accurate recommendation.
The component recommenders are learned based on a re-weighting strategy that assigns a dynamic weight to each observed user-item interaction.

Here explicit feedback refers to users' ratings to items while implicit feedback is derived
from users' interactions with items, e.g., number of
times a user plays a song.


- [A Boosting Algorithm for Item Recommendation with Implicit Feedback](https://www.ijcai.org/Proceedings/15/Papers/255.pdf)
- [The review @Arivin's blog](http://www.arvinzyy.cn/2017/09/23/A-Boosting-Algorithm-for-Item-Recommendation-with-Implicit-Feedback/)

**Gradient Boosting Factorization Machines**

- [Gradient boosting factorization machines](http://tongzhang-ml.org/papers/recsys14-fm.pdf)
****

- [ ] https://wsdm2019-dapa.github.io/#section-ketnotes
- [ ] https://github.com/robi56/Deep-Learning-for-Recommendation-Systems
- [ ] https://github.com/wzhe06/Reco-papers
- [ ] https://github.com/hongleizhang/RSPapers
- [ ] https://github.com/hongleizhang/RSAlgorithms
- [ ] https://github.com/cheungdaven/DeepRec
- [ ] https://github.com/cyhong549/DeepFM-Keras
- [ ] https://github.com/grahamjenson/list_of_recommender_systems
- [ ] https://zhuanlan.zhihu.com/p/26977788
- [ ] https://zhuanlan.zhihu.com/p/45097523
- [ ] https://www.zhihu.com/question/20830906
- [ ] https://www.zhihu.com/question/56806755/answer/150755503
+ [DLRS 2018 : 3rd Workshop on Deep Learning for Recommender Systems](http://www.wikicfp.com/cfp/servlet/event.showcfp?eventid=76328&copyownerid=87252)
+ [Deep Learning based Recommender System: A Survey and New Perspectives](https://arxiv.org/pdf/1707.07435.pdf)
+ [$5^{th}$ International Workshop on Machine Learning Methods for Recommender Systems](https://doogkong.github.io/2019/)
+ [MoST-Rec 2019: Workshop on Model Selection and Parameter Tuning in Recommender Systems](http://most-rec.gt-arc.com/)
+ [2018 Personalization, Recommendation and Search (PRS) Workshop](https://prs2018.splashthat.com/)
+ [WIDE & DEEP RECOMMENDER SYSTEMS AT PAPI](https://www.papis.io/recommender-systems)
+ [Interdisciplinary Workshop on Recommender Systems](http://www.digitaluses-congress.univ-paris8.fr/Interdisciplinary-Workshop-on-Recommender-Systems)
+ [2nd FATREC Workshop: Responsible Recommendation](https://piret.gitlab.io/fatrec2018/)

### Implementation

- [ ] https://github.com/maciejkula/spotlight
- [ ] http://surpriselib.com/
- [ ] https://github.com/Microsoft/Recommenders
- [ ] https://github.com/cheungdaven/DeepRec
- [ ] https://github.com/alibaba/euler
- [ ] https://github.com/alibaba/x-deeplearning/wiki/
- [ ] https://github.com/lyst/lightfm
- [ ] https://orange3-recommendation.readthedocs.io/en/latest/
- [ ] http://www.mymedialite.net/index.html
- [ ] http://www.mymediaproject.org/
- [Workshop: Building Recommender Systems w/ Apache Spark 2.x](https://qcon.ai/qconai2019/workshop/building-recommender-systems-w-apache-spark-2x)
- [A Leading Java Library for Recommender Systems](https://www.librec.net/)
- [lenskit: Python Tools for Recommender Experiments](https://lenskit.org/)

## Computational Advertising

Online advertising has grown over the past decade to over $26 billion in recorded revenue in 2010. The revenues generated are based on different pricing models that can be fundamentally grouped into two types: cost per (thousand) impressions (CPM) and cost per action (CPA), where an action can be a click, signing up with the advertiser, a sale, or any other measurable outcome. A web publisher generating revenues by selling advertising space on its site can offer either a CPM or CPA contract. We analyze the conditions under which the two parties agree on each contract type, accounting for the relative risk experienced by each party.

The information technology industry relies heavily on the on-line advertising such as [Google，Facebook or Alibaba].
Advertising is nothing except information.

**GBRT+LR**

[Practical Lessons from Predicting Clicks on Ads at
Facebook](https://www.jianshu.com/p/96173f2c2fb4) or the [blog](https://zhuanlan.zhihu.com/p/25043821) use the GBRT to select proper features and LR to map these features into the interval $[0,1]$ as a ratio.
Once we have the right features and the right model (decisions trees plus logistic regression), other factors play small roles (though even small improvements are important at scale).

<img src="https://pic4.zhimg.com/80/v2-fcb223ba88c456ce34c9d912af170e97_hd.png" width = "60%" />

When the feature vector ${x}$ are given, the tree split the features by GBRT then we transform and input the features to the logistic regression.


****

* http://kubicode.me/2018/03/19/Deep%20Learning/Talk-About-CTR-With-Deep-Learning/
* https://github.com/shenweichen/DeepCTR
* https://github.com/wzhe06/Ad-papers
* https://github.com/wnzhang/rtb-papers
* https://github.com/wzhe06/CTRmodel
* https://github.com/cnkuangshi/LightCTR
* http://www.cse.fau.edu/~xqzhu/courses/cap6807.html
* https://www.soe.ucsc.edu/departments/technology-management/research/computational-advertising
* http://alex.smola.org/teaching/ucsc2009/ucsc_1.pdf
* https://deepctr.readthedocs.io/en/latest/models/DeepModels.html
* https://blog.csdn.net/john_xyz/article/details/78933253
* https://people.eecs.berkeley.edu/~jfc/DataMining/SP12/lecs/lec12.pdf
* http://quinonero.net/Publications/predicting-clicks-facebook.pdf
* https://tech.meituan.com/2019/01/17/dianping-search-deeplearning.html
* http://yelp.github.io/MOE/
______________________________________________________

### Labs

- [Data Mining Machine Learning @The University of Texas at Austin](http://www.ideal.ece.utexas.edu/)
- [Center for Big Data Analytics@The University of Texas at Austin](https://bigdata.oden.utexas.edu/)
- [Multimedia Computing Group@tudelft.nl](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/multimedia-computing/)
- [knowledge Lab@Uchicago](https://www.knowledgelab.org/)
- [DIGITAL TECHNOLOGY CENTER@UMN](https://www.dtc.umn.edu/)
- [The Innovation Center for Artificial Intelligence (ICAI)](https://icai.ai/)
- [Data Mining and Machine Learning lab (DMML)@ASU](http://dmml.asu.edu/)
- [Next Generation Personalization Technologies](http://ids.csom.umn.edu/faculty/gedas/NSFcareer/)
- [Recommender systems & ranking](https://sites.google.com/view/chohsieh-research/recommender-systems)
