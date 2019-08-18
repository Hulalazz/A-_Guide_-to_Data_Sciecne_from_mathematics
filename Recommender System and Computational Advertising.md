# Recommender System

<img src= "https://img.dpm.org.cn/Uploads/Picture/dc/27569[1024].jpg" width="50%" />

* [最新！五大顶会2019必读的深度推荐系统与CTR预估相关的论文 - 深度传送门的文章 - 知乎](https://zhuanlan.zhihu.com/p/69050253)
* [深度学习在搜索和推荐系统中的应用](https://blog.csdn.net/malefactor/article/details/52040228)
* [CSE 258: Web Mining and Recommender Systems](http://cseweb.ucsd.edu/classes/fa18/cse258-a/)
* [CSE 291: Trends in Recommender Systems and Human Behavioral Modeling](https://cseweb.ucsd.edu/classes/fa17/cse291-b/)
* [THE AAAI-19 WORKSHOP ON RECOMMENDER SYSTEMS AND NATURAL LANGUAGE PROCESSING (RECNLP)](https://recnlp2019.github.io/)
* [Information Recommendation for Online Scientific Communities, Purdue University, Luo Si, Gerhard Klimeck and Michael McLennan](https://www.cs.purdue.edu/homes/lsi/CI_Recom/CI_Recom.html)
* [Recommendations for all : solving thousands of recommendation problems a day](https://ai.google/research/pubs/pub46822)
* http://staff.ustc.edu.cn/~hexn/
* [Learning Item-Interaction Embeddings for User Recommendations](https://arxiv.org/abs/1812.04407)
* [Summary of RecSys](https://github.com/fuxuemingzhu/Summary-of-Recommender-System-Papers)
* [How Netflix’s Recommendations System Works](https://help.netflix.com/en/node/100639)
* [个性化推荐系统，必须关注的五大研究热点](https://www.msra.cn/zh-cn/news/executivebylines/tech-bylines-personalized-recommendation-system)
* [How Does Spotify Know You So Well?](https://medium.com/s/story/spotifys-discover-weekly-how-machine-learning-finds-your-new-music-19a41ab76efe)
* [推荐系统论文集合](https://daiwk.github.io/posts/links-navigation-recommender-system.html)
* https://hong.xmu.edu.cn/Services___fw/Recommender_System.htm
* https://blog.statsbot.co/recommendation-system-algorithms-ba67f39ac9a3
* https://buildingrecommenders.wordpress.com/
* https://homepages.dcc.ufmg.br/~rodrygo/recsys-2019-1/
* https://developers.google.com/machine-learning/recommendation/
* https://sites.google.com/view/lianghu/home/tutorials/ijcai2019
* https://acmrecsys.github.io/rsss2019/
* https://github.com/alibaba/x-deeplearning/wiki

Recommender Systems (RSs) are software tools and techniques providing suggestions for items to be of use to a user.

RSs are primarily directed towards individuals who lack sufficient personal experience or competence to evaluate the potentially overwhelming number of alternative items that a Web site, for example, may offer.

[Xavier Amatriain discusses the traditional definition and its data mining core.](https://www.kdd.org/exploration_files/V14-02-05-Amatriain.pdf)

Traditional definition: The **recommender system** is to estimate a utility  function that automatically predicts how a user will like an item.

User Interest is **implicitly** reflected in `Interaction history`, `Demographics` and `Contexts`, which can be regarded as a typical example of data mining. Recommender system should match a context to a collection of information objects. There are some methods called `Deep Matching Models for Recommendation`.
It is an application of machine learning, which is in the *representation + evaluation + optimization* form. And we will focus on the `representation and evaluation`.


- [ ] https://github.com/hongleizhang/RSPapers
- [ ] https://github.com/benfred/implicit
- [ ] https://github.com/YuyangZhangFTD/awesome-RecSys-papers
- [ ] https://github.com/daicoolb/RecommenderSystem-Paper
- [ ] https://github.com/grahamjenson/list_of_recommender_systems
- [ ] https://www.zhihu.com/question/20465266/answer/142867207
- [ ] http://www.mbmlbook.com/Recommender.html
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

|Evolution of the Recommender Problem|
|:---:|
|Rating|
|Ranking|
|Page Optimization|
|Context-aware Recommendations|
----
|[Recommendation Strategies](https://datawarrior.wordpress.com/2019/06/19/strategies-of-recommendation-systems/)|
|:---:|
|Collaborative Filtering (CF)|
|Content-Based Filtering (CBF)|
|Demographic Filtering (DF)|
|Knowledge-Based Filtering (KBF)|
|Hybrid Recommendation Systems|

**Evaluation of Recommendation System**

The evaluation of machine learning algorithms depends on the tasks.
The evaluation of recommendation system can be regarded as some machine learning models such as regression, classification and so on.
We only take the mathematical convenience into consideration in the following methods.
`Gini index, covering rate` and more realistic factors are not discussed in the following content.

- [Evaluating recommender systems](http://fastml.com/evaluating-recommender-systems/)
- [Distance Metrics for Fun and Profit](https://www.benfrederickson.com/distance-metrics/)
- [Recsys2018 evaluation: tutorial](https://github.com/jeanigarcia/recsys2018-evaluation-tutorial)



## Collaborative Filtering

There are 3 kinds of collaborative filtering: user-based, item-based and model-based collaborative filtering.

The user-based methods are based on the similarities of users. If user ${u}$ and ${v}$ are very similar friends, we may recommend the items which user ${u}$ bought to the user ${v}$ and explains it that your friends also bought it.

The item-based methods are based on the similarity of items. If one person added a brush to shopping-list, it is reasonable to recommend some toothpaste to him or her. And we can explain that you bought item $X$ and the people who bought $X$ also bought $Y$.
And we focus on the model-based collaborative filtering.

- [协同过滤详解](https://www.cnblogs.com/ECJTUACM-873284962/p/8729010.html)
- [深入推荐引擎相关算法 - 协同过滤](https://www.ibm.com/developerworks/cn/web/1103_zhaoct_recommstudy2/index.html)


### Matrix Completion

Matrix completion is to complete the matrix $X$ with missing elements, such as

$$
\min_{Z} Rank(Z) \\
s.t. \sum_{(i,j):Observed}(Z_{(i,j)}-X_{(i,j)})^2\leq \delta
$$

Note that the rank of a matrix is not easy or robust  to compute.

We can apply [customized PPA](http://maths.nju.edu.cn/~hebma/Talk/Unified_Framework.pdf) to matrix completion problem

$$
\min \{ {\|Z\|}_{\ast}\} \\
s.t. Z_{\Omega} = X_{\Omega}
$$

We let ${Y}\in\mathbb{R}^{n\times n}$ be the the Lagrangian multiplier to the constraints $Z_{\Omega} = X_{\Omega}$
and Lagrange function is
$$
L(Z,Y) = {\|Z\|}_{\ast} - Y(Z_{\Omega} - X_{\Omega}).
$$

1. Producing $Y^{k+1}$ by
   $$Y^{k+1}=\arg\max_{Y} {L([2Z^k-Z^{k-1}],Y)-\frac{s}{2}\|Y-Y^k\|};$$
2. Producing $Z^{k+1}$ by
    $$Z^{k+1}=\arg\min_{Z} {L(Z,Y^{k+1}) + \frac{r}{2}\|Z-Z^k\|}.$$

<img title = "Netflix DataSet" src=https://pic3.zhimg.com/80/dc9a2b89742a05c3cd2f025105ba1c4a_hd.png width = 80% />

[Rahul Mazumder, Trevor Hastie, Robert Tibshirani](http://www.jmlr.org/papers/v11/mazumder10a.html) reformulate it as the following:

$$
\min f_{\lambda}(Z)=\frac{1}{2}{\|P_{\Omega}(Z-X)\|}_F^2 + \lambda {\|Z\|}_{\ast}
$$

where $X$ is the observed matrix, $P_{\Omega}$ is a projector and ${\|\cdot\|}_{\ast}$ is the nuclear norm of matrix.


* [A SINGULAR VALUE THRESHOLDING ALGORITHM FOR MATRIX COMPLETION](https://www.zhihu.com/question/47716840/answer/110843844)
* [Matrix and Tensor Decomposition in Recommender Systems](http://delab.csd.auth.gr/papers/RecSys2016s.pdf)
* [Low-Rank Matrix Recovery](http://www.princeton.edu/~yc5/ele538b_sparsity/lectures/matrix_recovery.pdf)
* [ECE 18-898G: Special Topics in Signal Processing: Sparsity, Structure, and Inference Low-rank matrix recovery via nonconvex optimization](https://users.ece.cmu.edu/~yuejiec/ece18898G_notes/ece18898g_nonconvex_lowrank_recovery.pdf)

<img title = "MMM" src=https://pic3.zhimg.com/80/771b16ac7e7aaeb50ffd8a8f5cf4e582_hd.png width = 80% />


* [Matrix Completion/Sensing as NonConvex Optimization Problem](http://sunju.org/research/nonconvex/)
* [Exact Matrix Completion via Convex Optimization](http://statweb.stanford.edu/~candes/papers/MatrixCompletion.pdf)
* [A SINGULAR VALUE THRESHOLDING ALGORITHM FOR MATRIX COMPLETION](http://statweb.stanford.edu/~candes/papers/SVT.pdf)
* [Customized PPA for convex optimization](http://maths.nju.edu.cn/~hebma/Talk/Unified_Framework.pdf)
* [Matrix Completion.m](http://www.convexoptimization.com/wikimization/index.php/Matrix_Completion.m)

### Maximum Margin Matrix Factorization

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

\(\color{red}{Note:}\) The data sets we more frequently encounter in collaborative prediction problem are of `ordinal ratings` $X_{ij} \in \{1, 2, \dots, R\}$ such as $\{1, 2, 3, 4, 5\}$.
To relate the real-valued $Z_{ij}$ to the
discrete $X_{ij}$. we use $R − 1$ thresholds $\theta_{1}, \dots, \theta_{R-1}$.

### SVD and Beyond

If we have collected user ${u}$'s explicit evaluation score to the item ${i}$ ,  $R_{[u][i]}$, and all such data makes up a matrix $R=(R_{[u][i]})$ while the user $u$ cannot evaluate all the item so that the matrix is incomplete and missing much data.
**SVD** is to factorize the matrix into the multiplication of matrices so that
$$
\hat{R} = P^{T}Q.
$$

And we can predict the score $R_{[u][i]}$ via
$$
\hat{R}_{[u][i]} = \hat{r}_{u,i} = \left<P_u, Q_i\right> = \sum_f p_{u,f} q_{i,f}
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
* [使用SVD++进行协同过滤（算法原理部分主要引用自他人）](https://www.cnblogs.com/Xnice/p/4522671.html)
* [SVD++推荐系统](https://blog.csdn.net/turing365/article/details/80544594)

### Probabilistic Matrix Factorization

In linear regression, the least square methods is equivalent to maximum likelihood estimation of the error in standard normal distribution.  


|Regularized SVD|
|:-------------:|
|$C(P,Q) = \sum_{(u,i):Observed}(r_{(u,i)}-\sum_f p_{(u,f)} q_{(i,f)})^{2}+\lambda_u\|P_u\|^2+\lambda_i\|Q_i\|^2$|

|Probabilistic model|
|:-----------------:|
|$r_{u,i}\sim N(\sum_f p_{(u,f)} q_{(i,f)},\sigma^2), P_u\sim N(0,\sigma_u^2 I), Q_i\sim N(0,\sigma_i^2 I)$|

And $\sigma_u^2$ and $\sigma_i^2$ is related with the regular term $\lambda_u$ and $\lambda_u$.

So that we can reformulate the optimization problem as maximum likelihood estimation.

* [Latent Factor Models for Web Recommender Systems](http://www.ideal.ece.utexas.edu/seminar/LatentFactorModels.pdf)
* [Regression-based Latent Factor Models @CS 732 - Spring 2018 - Advanced Machine Learning by Zhi Wei](https://web.njit.edu/~zhiwei/CS732/papers/Regression-basedLatentFactorModels_KDD2009.pdf)
* [Probabilistic Matrix Factorization](https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf)

### Coupled Poisson Factorization



+ [Coupled Poisson Factorization Integrated with User/Item Metadata for Modeling Popular and Sparse Ratings in Scalable Recommendation](http://www.datasciences.org/)
+ [Coupled Compound Poisson Factorization](https://arxiv.org/pdf/1701.02058.pdf)
+ https://github.com/mehmetbasbug/ccpf

### Collaborative Less-is-More Filtering(CliMF)

Sometimes, the information of user we could collect is implicit such as the clicking at some item.

In `CLiMF` [the model parameters are learned by directly maximizing the Mean Reciprocal Rank (MRR).](https://github.com/gamboviol/climf)

Its objective function is
$$
F(U,V)=\sum_{i=1}^{M}\sum_{j=1}^{N} Y_{ij} [\ln g(U_{i}^{T}V_{j})+\sum_{k=1}^{N}\ln (1 - Y_{ij} g(U_{i}^{T}V_{k}-U_{i}^{T}V_{j}))] \\-\frac{\lambda}{2}({\|U\|}^2 + {\|V\|}^2)
$$

where ${M, N}$ is the number of users and items, respectively. Additionally, $\lambda$ denotes the regularization coefficient and $Y_{ij}$ denotes the binary relevance score of item ${j}$ to user ${i}$, i.e., $Y_{ij} = 1$ if item ${j}$ is relevant to user ${j}$, 0 otherwise. The function $g$ is logistic function $g(x)=\frac{1}{1+\exp(-x)}$.
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



### BellKor's Progamatic Chaos

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

### Recommendation with Implicit Information

|Explicit and implicit feedback|
|:---:|
|![](https://www.msra.cn/wp-content/uploads/2018/06/knowledge-graph-in-recommendation-system-i-8.png)|

[**WRMF**](https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/) is simply a modification of this loss function:

$$
{C(P,Q)}_{WRMF} = \sum_{(u,i):Observed}c_{u,i}(I_{u,i} - \sum_f p_{u,f}q_{i,f})^{2} + \underbrace{\lambda_u\|P_u\|^2 + \lambda_i\|Q_i\|^2}_{\text{regularization terms}}.
$$

We make the assumption that if a user has interacted at all with an item, then $I_{u,i} = 1$. Otherwise, $I_{u,i} = 0$.
If we take $d_{u,i}$ to be the number of times a user ${u}$ has clicked on an item ${i}$ on a website, then
$$c_{u,i}=1+\alpha d_{u,i}$$
where $\alpha$ is some hyperparameter determined by cross validation.
The new  term in cost function $C=(c_{u,i})$ is called confidence matrix.

WRMF does not make the assumption that a user who has not interacted with an item does not like the item. WRMF does assume that that user has a negative preference towards that item, but we can choose how confident we are in that assumption through the confidence hyperparameter.

[Alternating least square](http://suo.im/4YCM5f) (**ALS**) can give an analytic solution to this optimization problem by setting the gradients equal to 0s.

* [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf)
* [Fast Collaborative Filtering from Implicit Feedback with Provable Guarantees](http://proceedings.mlr.press/v63/Dasgupta79.pdf)
* [Intro to Implicit Matrix Factorization: Classic ALS with Sketchfab Models](https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/)
* http://nicolas-hug.com/blog/matrix_facto_1
* http://nicolas-hug.com/blog/matrix_facto_2
* http://nicolas-hug.com/blog/matrix_facto_3
* [A recommender systems development and evaluation package by Mendeley](https://github.com/Mendeley/mrec)
* https://mendeley.github.io/mrec/
* [Fast Python Collaborative Filtering for Implicit Feedback Datasets](https://github.com/benfred/implicit)
* [Alternating Least Squares Method for Collaborative Filtering](https://bugra.github.io/work/notes/2014-04-19/alternating-least-squares-method-for-collaborative-filtering/)
* [Implicit Feedback and Collaborative Filtering](http://datamusing.info/blog/2015/01/07/implicit-feedback-and-collaborative-filtering/)
* [Faster Implicit Matrix Factorization](https://www.benfrederickson.com/fast-implicit-matrix-factorization/)
* [CUDA Tutorial: Implicit Matrix Factorization on the GPU](https://www.benfrederickson.com/implicit-matrix-factorization-on-the-gpu/)



***

* [Matrix factorization for recommender system@Wikiwand](https://www.wikiwand.com/en/Matrix_factorization_(recommender_systems)})
* http://www.cnblogs.com/DjangoBlog/archive/2014/06/05/3770374.html
* [Learning to Rank Sketchfab Models with LightFM](https://www.ethanrosenthal.com/2016/11/07/implicit-mf-part-2/)
* [Finding Similar Music using Matrix Factorization](https://www.benfrederickson.com/matrix-factorization/)
* [Top-N Recommendations from Implicit Feedback Leveraging Linked Open Data ?](https://core.ac.uk/display/23873231)

### Inductive Matrix Completion

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

**More on Matrix Factorization**

- [The Advanced Matrix Factorization Jungle](https://sites.google.com/site/igorcarron2/matrixfactorizations)
- [Non-negative Matrix Factorizations](http://www.ams.org/publicoutreach/feature-column/fc-2019-03)
- http://people.eecs.berkeley.edu/~yima/
- [New tools for recovering low-rank matrices from incomplete or corrupted observations by Yi Ma@UCB](http://people.eecs.berkeley.edu/~yima/matrix-rank/home.html)
- [DiFacto — Distributed Factorization Machines](https://www.cs.cmu.edu/~muli/file/difacto.pdf)
- [Learning with Nonnegative Matrix Factorizations](https://sinews.siam.org/Details-Page/learning-with-nonnegative-matrix-factorizations)
- [Nonconvex Optimization Meets Low-Rank Matrix Factorization: An Overview](http://www.princeton.edu/~yc5/publications/NcxOverview_Arxiv.pdf)
- [Taming Nonconvexity in Information Science, tutorial at ITW 2018.](https://www.princeton.edu/~yc5/slides/itw2018_tutorial.pdf)
- [Nonnegative Matrix Factorization by Optimization on the Stiefel Manifold with SVD Initialization](https://user.eng.umd.edu/~smiran/Allerton16.pdf)

----


### Factorization Machines(FM)

The matrix completion used in recommender system are linear combination of some features such as regularized SVD and they only take the user-user interaction and item-item similarity.
`Factorization Machines(FM)` is inspired from previous factorization models.
It represents each feature an embedding vector, and models the second-order feature interactions:
$$
\hat{y}
= w_0 + \sum_{i=1}^{n} w_i x_i+\sum_{i=1}^{n-1}\sum_{j=i+1}^{n}\left<v_i, v_j\right> x_i x_j\\
= \underbrace{w_0  + \left<w, x\right>}_{\text{First-order: Linear Regression}} + \underbrace{\sum_{i=1}^{n-1}\sum_{j=i+1}^{n}\left<v_i, v_j\right> x_i x_j}_{\text{Second-order: pair-wise interactions between features}}
$$

where the model parameters that have to be estimated are
$$
w_0 \in \mathbb{R}, w\in\mathbb{R}^n, V\in\mathbb{R}^{n\times k}.
$$

And $\left<\cdot,\cdot\right>$ is the dot (inner) product of two vectors so that $\left<v_i, v_j\right>=\sum_{f=1}^{k}v_{i,f} \cdot v_{j,f}$.
A row $v_i$ within ${V}$ describes the ${i}$-th latent variable with ${k}$ factors for $x_i$.

And the linear regression $w_0 + \sum_{i=1}^{n} w_i x_i$ is called `the first order part`; the pair-wise interactions between features
$\sum_{i=1}^{n}\sum_{j=i+1}^{n}\left<v_i, v_j\right> x_i x_j$ is called the `second order part`.

However, why we call it `factorization machine`? Where is the _factorization_?
If ${[W]}_{ij}=w_{ij}= \left<v_i, v_j\right>$, $W=V V^T$.

In order to reduce the computation complexity, the second order part $\sum_{i=1}^{n-1}\sum_{j=i+1}^{n}\left<v_i, v_j\right> x_i x_j$ is rewritten in the following form
$$\frac{1}{2}\sum_{l=1}^{k}\{[\sum_{i=1}^{n}(v_{il}x_i))]^2-\sum_{i=1}^{n}(v_{il}x_i)^2\}.$$

* [FM算法（Factorization Machine）](https://blog.csdn.net/g11d111/article/details/77430095)
* [分解机(Factorization Machines)推荐算法原理 by 刘建平Pinard](https://www.cnblogs.com/pinard/p/6370127.html)
* [Factorization Machines for Recommendation Systems](https://getstream.io/blog/factorization-recommendation-systems/)
* [第09章：深入浅出ML之Factorization家族](http://www.52caml.com/head_first_ml/ml-chapter9-factorization-family/)
* [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
* [TensorFlow implementation of an arbitrary order Factorization Machine](https://github.com/geffy/tffm)

### Field-aware Factorization Machine(FFM)

In FMs, every feature has only one latent vector to learn the latent effect with any other features.
In FFMs, each feature has several latent vectors. Depending on the field of other features, one of them is used to do the inner product.
Mathematically,
$$
\hat{y}=\sum_{j_1=1}^{n}\sum_{j_2=i+1}^{n}\left<v_{j_1,f_2}, v_{j_2,f_1}\right> x_{j_1} x_{j_2}
$$
where $f_1$ and $f_2$ are respectively the fields of $j_1$ and $j_2$.

* [Yuchin Juan at ACEMAP](https://www.acemap.info/author/page?AuthorID=7E61F31B)
* [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)
* https://blog.csdn.net/mmc2015/article/details/51760681

## Deep Learning for Recommender System

Deep learning is powerful in processing visual and text information so that it helps to find the interests of users such as
[Deep Interest Network](http://www.cnblogs.com/rongyux/p/8026323.html), [xDeepFM](https://www.jianshu.com/p/b4128bc79df0)  and more.

Deep learning models for recommender system may come from the restricted Boltzman machine.
And deep learning models are powerful information extractors.
Deep learning is really popular in recommender system such as [spotlight](https://github.com/maciejkula/spotlight).

* [A review on deep learning for recommender systems:
challenges and remedies](https://daiwk.github.io/assets/Batmaz2018_Article_AReviewOnDeepLearningForRecomm.pdf)

### Restricted Boltzmann Machines for Collaborative Filtering(RBM)

Let ${V}$ be a $K\times m$ observed binary indicator matrix with $v_i^k = 1$ if the user rated item ${i}$ as ${k}$ and ${0}$ otherwise.
We also let $h_j$, $j = 1, \dots, F,$ be the binary values of hidden (latent) variables, that can be thought of as representing
stochastic binary features that have different values for different users.

We use a conditional multinomial distribution (a “softmax”) for modeling each column of the observed
"visible" binary rating matrix ${V}$ and a conditional
Bernoulli distribution for modeling "hidden" user features *${h}$*:
$$
p(v_i^k = 1 \mid h) = \frac{\exp(b_i^k + \sum_{j=1}^{F} h_j W_{i,j}^{k})}{\sum_{l=1}^{K}\exp( b_i^k + \sum_{j=1}^{F} h_j W_{i, j}^{l})} \\
p( h_j = 1 \mid V) = \sigma(b_j + \sum_{i=1}^{m}\sum_{k=1}^{K} v_i^k W_{i,j}^k)
$$
where $\sigma(x) = \frac{1}{1 + exp(-x)}$ is the logistic function, $W_{i,j}^{k}$ is is a symmetric interaction parameter between feature
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

We can also model “hidden” user features $h$ as Gaussian latent variables:
$$
p(v_i^k = 1 | h) = \frac{\exp(b_i^k+\sum_{j=1}^{F}h_j W_{i,j}^{k})}{\sum_{l=1}^{K}\exp(b_i^k+\sum_{j=1}^{F}h_j W_{i,j}^{l})} \\
p( h_j = 1 | V) = \frac{1}{\sqrt{2\pi}\sigma_j} \exp(\frac{(h - b_j -\sigma_j \sum_{i=1}^{m}\sum_{k=1}^{K} v_i^k W_{i,j}^k)^2}{2\sigma_j^2})
$$
where $\sigma_j^2$ is the variance of the hidden unit ${j}$.

<img title = "RBM " src ="https://raw.githubusercontent.com/adityashrm21/adityashrm21.github.io/master/_posts/imgs/book_reco/rbm.png" width="30%" />

* https://www.cnblogs.com/pinard/p/6530523.html
* https://www.cnblogs.com/kemaswill/p/3269138.html
* [Restricted Boltzmann Machines for Collaborative Filtering](https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf)
* [Building a Book Recommender System using Restricted Boltzmann Machines](https://adityashrm21.github.io/Book-Recommender-System-RBM/)
* [On Contrastive Divergence Learning](http://www.cs.toronto.edu/~fritz/absps/cdmiguel.pdf)
* http://deeplearning.net/tutorial/rbm.html
* [RBM notebook form Microsoft](https://github.com/Microsoft/Recommenders/blob/master/notebooks/00_quick_start/rbm_movielens.ipynb)

### AutoRec

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

* [《AutoRec: Autoencoders Meet Collaborative Filtering》WWW2015 阅读笔记](https://blog.csdn.net/studyless/article/details/70880829)
* [AutoRec: Autoencoders Meet Collaborative Filtering](http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)

### Wide & Deep Model

The output of this model is
$$
P(Y=1|x) = \sigma(W_{wide}^T[x,\phi(x)] + W_{deep}^T \alpha^{(lf)}+b)
$$
where the `wide` part deal with the categorical features such as user demographics and the `deep` part deal with continuous features.


<img src=https://upload-images.jianshu.io/upload_images/1500965-13fa11d119bb20b7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp width=70%/>
<img src=http://kubicode.me/img/Take-about-CTR-With-Deep-Learning/fnn_pnn_wdl.png width=70%/>

* https://arxiv.org/pdf/1606.07792.pdf
* [Wide & Deep Learning: Better Together with TensorFlow, Wednesday, June 29, 2016](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)
* [Wide & Deep](https://www.jianshu.com/p/dbaf2d9d8c94)
* https://www.sohu.com/a/190148302_115128

<img src = http://kubicode.me/img/Take-about-CTR-With-Deep-Learning/dcn_arch.png width=60%/>

### Deep FM

`DeepFM` ensembles FM and DNN and to learn both second order and higher-order feature interactions:
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

<img src=http://kubicode.me/img/Deep-in-out-Wide-n-Deep-Series/deepfm_arch.png width=80% />

* https://zhuanlan.zhihu.com/p/27999355
* https://zhuanlan.zhihu.com/p/25343518
* https://zhuanlan.zhihu.com/p/32127194
* https://arxiv.org/pdf/1703.04247.pdf
* [CTR预估算法之FM, FFM, DeepFM及实践](https://blog.csdn.net/John_xyz/article/details/78933253#deep-fm)

### Neural Factorization Machines

$$
\hat{y} = w_0 + \left<w, x\right> + f(x)
$$
where the first and second terms are the linear regression part similar to that for FM, which models global bias of data and weight
of features. The third term $f(x)$ is the core component of NFM
for modelling feature interactions, which is a `multi-layered feedforward neural network`.

`B-Interaction Layer` including `Bi-Interaction Pooling` is an innovation in artificial neural network.

<img title="Neu FM" src="https://pic2.zhimg.com/80/v2-c7012d7a76e488643db9911d7588ccbd_hd.jpg" width="70%" />


* http://staff.ustc.edu.cn/~hexn/
* https://github.com/hexiangnan/neural_factorization_machine
* [LibRec 每周算法：NFM (SIGIR'17)](https://www.infosec-wiki.com/?p=394011)

### Attentional Factorization Machines

Attentional Factorization Machine (AFM) learns the importance of each feature interaction from data via a neural attention network.

We employ the attention mechanism on feature interactions by performing a weighted sum on the interacted vectors:

$$\sum_{(i, j)} a_{(i, j)}(V_i \odot V_j) x_i x_j$$

where $a_{i, j}$ is the attention score for feature interaction.

<img src=https://deepctr-doc.readthedocs.io/en/latest/_images/AFM.png width=80% />

* https://www.comp.nus.edu.sg/~xiangnan/papers/ijcai17-afm.pdf
* http://blog.leanote.com/post/ryan_fan/Attention-FM%EF%BC%88AFM%EF%BC%89

### xDeepFM

It mainly consists of 3 parts: `Embedding Layer`, `Compressed Interaction Network(CIN)` and `DNN`.

<img title="xDeepFM" src="https://www.msra.cn/wp-content/uploads/2018/08/kdd-2018-xdeepfm-5.png" width="60%" />


<img src="http://kubicode.me/img/eXtreme-Deep-Factorization-Machine/CIN-Network.png" width="80%" />

- [X] [KDD 2018 | 推荐系统特征构建新进展：极深因子分解机模型](https://www.msra.cn/zh-cn/news/features/kdd-2018-xdeepfm)
- [ ] [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/abs/1803.05170)
- [ ] https://arxiv.org/abs/1803.05170
- [ ] [据说有RNN和CNN结合的xDeepFM](http://kubicode.me/2018/09/17/Deep%20Learning/eXtreme-Deep-Factorization-Machine/)
- [ ] [推荐系统遇上深度学习(二十二)--DeepFM升级版XDeepFM模型强势来袭！](https://www.jianshu.com/p/b4128bc79df0)

### RepeatNet

<img title="Repeat Net" src="http://kubicode.me/img/More-Session-Based-Recommendation/repeatnet_arch.png" width="70%"/>

* https://arxiv.org/pdf/1806.08977.pdf
* https://github.com/PengjieRen/RepeatNet

****

* [Deep Knowledge-aware Network for News Recommendation](https://github.com/hwwang55/DKN)
* https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
* https://www.cnblogs.com/pinard/p/6370127.html
* https://www.jianshu.com/p/6f1c2643d31b
* https://blog.csdn.net/John_xyz/article/details/78933253
* https://zhuanlan.zhihu.com/p/38613747
* [Recommender Systems with Deep Learning](https://amundtveit.com/2016/11/20/recommender-systems-with-deep-learning/)
* [深度学习在序列化推荐中的应用](http://kubicode.me/2018/10/25/Deep%20Learning/More-Session-Based-Recommendation/)
* [深入浅出 Factorization Machine 系列](http://kubicode.me/2018/02/23/Deep%20Learning/Deep-in-out-Factorization-Machines-Series/)
* [论文快读 - Deep Neural Networks for YouTube Recommendations](http://lipixun.me/2018/02/01/youtube)

### Deep Matrix Factorization

* [Deep Matrix Factorization Models for Recommender Systems](https://www.ijcai.org/proceedings/2017/0447.pdf)
* [Deep Matrix Factorization for Recommender Systems with Missing Data not at Random](https://iopscience.iop.org/article/10.1088/1742-6596/1060/1/012001)

### Deep Geometric Matrix Completion

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
* [Deep Geometric Matrix Completion: a Geometric Deep Learning approach to Recommender Systems](http://www.ipam.ucla.edu/abstract/?tid=14552&pcode=DLT2018)
* [Talk: Deep Geometric Matrix Completion](http://helper.ipam.ucla.edu/publications/dlt2018/dlt2018_14552.pdf)

### Collaborative Deep Learning for Recommender Systems

[Collaborative filtering (CF) is a successful approach commonly used by many recommender systems. Conventional CF-based methods use the ratings given to items by users as the sole source of information for learning to make recommendation. However, the ratings are often very sparse in many applications, causing CF-based methods to degrade significantly in their recommendation performance. To address this sparsity problem, auxiliary information such as item content information may be utilized. Collaborative topic regression (CTR) is an appealing recent method taking this approach which tightly couples the two components that learn from two different sources of information. Nevertheless, the latent representation learned by CTR may not be very effective when the auxiliary information is very sparse. To address this problem, we generalize recently advances in deep learning from i.i.d. input to non-i.i.d. (CF-based) input and propose in this paper a hierarchical Bayesian model called collaborative deep learning (CDL), which jointly performs deep representation learning for the content information and collaborative filtering for the ratings (feedback) matrix. Extensive experiments on three real-world datasets from different domains show that CDL can significantly advance the state of the art.](http://www.wanghao.in/CDL.htm)

Given part of the ratings in ${R}$ and the content information $X_c$, the problem is to predict the other ratings in ${R}$,
where row ${j}$ of the content information matrix $X_c$ is the bag-of-words vector $Xc;j{\ast}$ for item ${j}$ based on a vocabulary of size ${S}$.

`Stacked denoising autoencoders(SDAE)` is a feedforward neural network for learning
representations (encoding) of the input data by learning to predict the clean input itself in the output.
Using the Bayesian SDAE as a component, the generative
process of CDL is defined as follows:
1. For each layer ${l}$ of the SDAE network,
    * For each column ${n}$ of the weight matrix $W_l$, draw
    $$W_l;{\ast}n \sim \mathcal{N}(0,\lambda_w^{-1} I_{K_l}).$$
    * Draw the bias vector
    $$b_l \sim \mathcal{N}(0,\lambda_w^{-1} I_{K_l}).$$
    * For each row ${j}$ of $X_l$, draw
    $$X_{l;j\ast}\sim \mathcal{N}(\sigma(X_{l-1;j\ast}W_l b_l), \lambda_s^{-1} I_{K_l}).$$

2. For each item ${j}$,
      * Draw a clean input
        $$X_{c;j\ast}\sim \mathcal{N}(X_{L, j\ast}, \lambda_n^{-1} I_{K_l}).$$
      * Draw a latent item offset vector $\epsilon_j \sim \mathcal{N}(0, \lambda_v^{-1} I_{K_l})$ and then set the latent item vector to be:
        $$v_j=\epsilon_j+X^T_{\frac{L}{2}, j\ast}.$$
3. Draw a latent user vector for each user ${i}$:
     $$u_i \sim \mathcal{N}(0, \lambda_u^{-1} I_{K_l}).$$

4. Draw a rating $R_{ij}$ for each user-item pair $(i; j)$:
  $$R_{ij}\sim \mathcal{N}(u_i^T v_j, C_{ij}^{-1}).$$

Here $\lambda_w, \lambda_s, \lambda_n, \lambda_u$and $\lambda_v$ are hyperparameters and $C_{ij}$ is
a confidence parameter similar to that for CTR ($C_{ij} = a$ if $R_{ij} = 1$ and $C_{ij} = b$ otherwise).

And joint log-likelihood of these parameters is
$$L=-\frac{\lambda_u}{2}\sum_{i} {\|u_i\|}_2^2-\frac{\lambda_w}{2}\sum_{l} [{\|W_l\|}_F+{\|b_l\|}_2^2]\\
-\frac{\lambda_v}{2}\sum_{j} {\|v_j - X^T_{\frac{L}{2},j\ast}\|}_2^2-\frac{\lambda_n}{2}\sum_{l} {\|X_{c;j\ast}-X_{L;j\ast}\|}_2^2 \\
-\frac{\lambda_s}{2}\sum_{l}\sum_{j} {\|\sigma(X_{l-1;j\ast}W_l b_l)-X_{l;j}\|}_2^2 -\sum_{ij} {\|R_{ij}-u_i^Tv_j\|}_2^2
$$

It is not easy to prove that it converges.


* http://www.winsty.net/
* http://www.wanghao.in/
* https://www.cse.ust.hk/~dyyeung/
* [Collaborative Deep Learning for Recommender Systems](http://www.wanghao.in/CDL.htm)
* [Deep Learning for Recommender Systems](https://www.inovex.de/fileadmin/files/Vortraege/2017/deep-learning-for-recommender-systems-pycon-10-2017.pdf)
* https://github.com/robi56/Deep-Learning-for-Recommendation-Systems
* [推荐系统中基于深度学习的混合协同过滤模型](http://www.10tiao.com/html/236/201701/2650688117/2.html)
* [CoupledCF: Learning Explicit and Implicit User-item Couplings in Recommendation for Deep Collaborative Filtering](http://203.170.84.89/~idawis33/DataScienceLab/publication/nonIID-RS-final.pdf)
- [ ] [Deep Learning Meets Recommendation Systems](https://nycdatascience.com/blog/student-works/deep-learning-meets-recommendation-systems/)
- [ ] [Using Keras' Pretrained Neural Networks for Visual Similarity Recommendations](https://www.ethanrosenthal.com/2016/12/05/recasketch-keras/)
- [ ] [Recommending music on Spotify with deep learning](http://benanne.github.io/2014/08/05/spotify-cnns.html)

### Deep Matching Models for Recommendation

It is essential for the recommender system  to find the item which matches the users' demand. Its difference from web search is that recommender system provides item information even if the users' demands or generally interests are not provided.
It sounds like modern crystal ball to read your mind.

In [A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems](http://sonyis.me/paperpdf/frp1159-songA-www-2015.pdf) the authors propose to extract rich features from user’s browsing
and search histories to model user’s interests. The underlying assumption is that, users’ historical online activities
reflect a lot about user’s background and preference, and
therefore provide a precise insight of what items and topics users might be interested in.


Its training data set and the test data is  $\{(\mathrm{X}_i, y_i, r_i)\mid i =1, 2, \cdots, n\}$ and $(\mathrm{X}_{n+1}, y_{n+1})$, respectively.
Matching Model is trained using the training data set: a class of `matching functions’
$\mathcal F= \{f(x, y)\}$ is defined, while the value of the function $r(\mathrm{X}, y)\in \mathcal F$ is a real number  a set of numbers $R$ and the $r_{n+1}$ is predicted as  $r_{n+1} = r(\mathrm{X}_{n+1}, y_{n+1})$.

The data is assumed to be generated according to the distributions $(x, y) \sim P(X,Y)$, $r \sim P(R \mid X,Y)$ . The goal of
the learning task is to select a matching function $f (x, y)$ from the class $F$ based on the observation of the training data.
The learning task, then, becomes the following optimization problem.
$$\arg\min_{r\in \mathcal F}\sum_{i=1}^{n}L(r_i, r(x_i, y_i))+\Omega(r)$$
where $L(\cdot, \cdot)$ denotes a loss function and $\Omega(\cdot)$ denotes regularization.

In fact, the inputs x and y can be instances (IDs), feature vectors, and structured objects, and thus the task can be carried out at instance level, feature level, and structure level.

And $r(x, y)$ is supposed to be non-negative in some cases.

|Framework of Matching|
|:---:|
|Output: MLP|
|Aggregation: Pooling, Concatenation|
|Interaction: Matrix, Tensor|
|Representation: MLP, CNN, LSTM|
|Input: ID Vectors $\mathrm{X}$, Feature Vectors $y$|


Sometimes, matching model and ranking model are combined and trained together with pairwise loss.
Deep Matching models takes the ID vectors and features together as the input to a deep neural network to train the matching scores including **Deep Matrix Factorization, AutoRec, Collaborative Denoising Auto-Encoder, Deep User and Image Feature, Attentive Collaborative Filtering, Collaborative Knowledge Base Embedding**.

`semantic-based matching models`

<img src="https://www.msra.cn/wp-content/uploads/2018/06/knowledge-graph-in-recommendation-system-i-18.png" width="80%"/>

* https://sites.google.com/site/nkxujun/
* http://sonyis.me/dnn.html
* https://akmenon.github.io/
* https://sigir.org/sigir2018/program/tutorials/
* [Learning  to Match](http://www.hangli-hl.com/uploads/3/4/4/6/34465961/learning_to_match.pdf)
* [Deep Learning for Matching in Search and Recommendation](http://staff.ustc.edu.cn/~hexn/papers/sigir18-tutorial-deep-matching.pdf)
* [Facilitating the design, comparison and sharing of deep text matching models.](https://github.com/NTMC-Community/MatchZoo)
* [Framework and Principles of Matching Technologies](http://www.hangli-hl.com/uploads/3/4/4/6/34465961/wsdm_2019_workshop.pdf)
* [A Multi-View Deep Learning Approach for Cross Domain User Modeling in Recommendation Systems](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/frp1159-songA.pdf)
* [Learning to Match using Local and Distributed Representations of Text for Web Search](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf)
* https://github.com/super-zhangchao/learning-to-match


### Hyperbolic Recommender Systems

Many well-established recommender systems are based on representation learning in Euclidean space.
In these models, matching functions such as the Euclidean distance or inner product are typically used for computing similarity scores between user and item embeddings.
`Hyperbolic Recommender Systems` investigate the notion of learning user and item representations in hyperbolic space.

Given a user ${u}$ and an item ${v}$ that are both lying in the Poincare ball $B^n$, the distance between two points on *P* is given by
$$d_p(x, y)=cosh^{-1}(1+2\frac{\|(x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}).$$

`Hyperbolic Bayesian Personalized
Ranking(HyperBPR)` leverages BPR pairwise learning to minimize the pairwise ranking loss between the positive and negative items.
Given a user ${u}$ and an item ${v}$ that are both lying in Poincare ball $B^n$, we take:
$$\alpha(u, v) = f(d_p(u, v))$$
where $f(\cdot)$ is simply preferred as a linear function $f(x) = \beta x + c$ with $\beta\in\mathbb{R}$ and $c\in\mathbb{R}$ are scalar parameters and learned along with the network.
The objective function is defined as follows:
$$\arg\min_{\Theta} \sum_{i, j, k} -\ln(\sigma\{\alpha(u_i, v_j) - \alpha(u_i, v_k)\}) + \lambda  {\|\Theta\|}_2^2$$

where $(i, j, k)$ is the triplet that belongs to the set ${D}$ that
contains all pairs of positive and negative items for each
user; $\sigma$ is the logistic sigmoid function; $\Theta$ represents the model parameters; and $\lambda$ is the regularization parameter.

The parameters of our model are learned by using [`RSGD`](https://arxiv.org/abs/1111.5280).

* [Stochastic gradient descent on Riemannian manifolds](https://arxiv.org/abs/1111.5280)
* [Hyperbolic Recommender Systems](https://arxiv.org/abs/1809.01703)
* [Scalable Hyperbolic Recommender Systems](https://arxiv.org/abs/1902.08648v1)


----

## Ensemble Methods for Recommender System

The RecSys can be considered as some regression or classification tasks, so that we can apply the ensemble methods to these methods as  `BellKor's Progamatic Chaos` used the blended solution to win the prize.
In fact, its essence is bagging or blending, which is one sequential ensemble strategy in order to avoid over-fitting or reduce the variance.

In this section, the boosting is the focus, which is to reduce the error and boost the performance from a weaker learner.

There are two common methods to construct a stronger learner from a weaker learner: (1) reweight the samples and learn from the error: AdaBoosting; (2) retrain another learner and learn to approximate the error: Gradient Boosting.

- [General Functional Matrix Factorization Using Gradient Boosting](http://w.hangli-hl.com/uploads/3/1/6/8/3168008/icml_2013.pdf)

### BoostFM

`BoostFM` integrates boosting into factorization models during the process of item ranking.
Specifically, BoostFM is an adaptive boosting framework that linearly combines multiple homogeneous component recommender system,
which are repeatedly constructed on the basis of the individual FM model by a re-weighting scheme.

**BoostFM**

+ _Input_: The observed context-item interactions or Training Data $S =\{(\mathbf{x}_i, y_i)\}$ parameters E and T.
+ _Output_: The strong recommender $g^{T}$.
+ Initialize $Q_{ci}^{(t)}=1/|S|,g^{(0)}=0, \forall (c, i)\in S$.
+ for $t = 1 \to T$ do
  +  1. Create component recommender $\hat{y}^{(t)}$ with $\bf{Q}^{(t)}$ on $\bf S$,$\forall (c,i) \in \bf S$, , i.e., `Component Recommender Learning Algorithm`;
  +  2. Compute the ranking accuracy $E[\hat{r}(c, i, y^{(t)})], \forall (c,i) \in \bf S$;
  +  3. Compute the coefficient $\beta_t$,
 $$ \beta_t = \ln (\frac{\sum_{(c,i) \in \bf S} \bf{Q}^{(t)}_{ci}\{1 + E[\hat{r}(c, i, y^{(t)})]\}}{\sum_{(c,i) \in \bf S} \bf{Q}^{(t)}_{ci}\{1-  E[\hat{r}(c, i, y^{(t)})]\}})^{\frac{1}{2}} ; $$
  +  4. Create the strong recommender $g^{(t)}$,
  $$ g^{(t)} = \sum_{h=1}^{t} \beta_h \hat{y}^{(t)} ;$$
  +  5. Update weight distribution \(\bf{Q}^{t+1}\),
  $$ \bf{Q}^{t+1}_{ci} = \frac{\exp(E[\hat{r}(c, i, y^{(t)})])}{\sum_{(c,i)\in \bf{S}} E[\hat{r}(c, i, y^{(t)})]} ; $$
+ end for


**Component Recommender**

Naturally, it is feasible to exploit the L2R techniques to optimize Factorization Machines
(FM). There are two major approaches in the field of L2R, namely, pairwise and listwise approaches.
In the following, we demonstrate ranking factorization machines with both pairwise and listwise optimization.

`Weighted Pairwise FM (WPFM)`

`Weighted ‘Listwise’ FM (WLFM)`

- [BoostFM: Boosted Factorization Machines for Top-N Feature-based Recommendation](http://wnzhang.net/papers/boostfm.pdf)
- http://wnzhang.net/
- https://fajieyuan.github.io/
- https://www.librec.net/luckymoon.me/
- [The author’s final accepted version.](http://eprints.gla.ac.uk/135914/7/135914.pdf)

### Gradient Boosting Factorization Machines

`Gradient Boosting Factorization Machine (GBFM)` model is to incorporate feature selection algorithm with Factorization Machines into a unified framework.

**Gradient Boosting Factorization Machine Model**

> + _Input_: Training Data $S =\{(\mathbf{x}_i, y_i)\}$.
> + _Output_: $\hat{y}_S =y_0(x) + {\sum}^S_{s=1}\left<v_{si}, v_{sj}\right>$.
> + Initialize rating prediction function as $\hat{y}_0(x)$
> + for $s = 1 \to S$ do
> +  1. Select interaction feature $C_p$ and $C_q$ from Greedy Feature Selection Algorithm;
> +  2. Estimate latent feature matrices $V_p$ and $V_q$;
> +  3. Update  $\hat{y}_s(\mathrm{x}) = \hat{y}_{s-1}(\mathrm{x}) + {\sum}_{i\in C_p}{\sum}_{j\in C_q} \mathbb{I}[i,j\in \mathrm{x}]\left<V_{p}^{i}, V_{q}^{j}\right>$
> + end for

where s is the iteration step of the learning algorithm. At step s, we greedily select two interaction features $C_p$ and $C_q$
where $\mathbb{I}$ is the indicator function, the value is 1 if the condition holds otherwise 0.

**Greedy Feature Selection Algorithm**

From the view of gradient boosting machine, at each
step s, we would like to search a function ${f}$ in the function
space ${F}$ that minimize the objective function:
$$L=\sum_{i}\ell(\hat{y}_s(\mathrm{x}_i), y_i)+\Omega(f)$$

where $\hat{y}_s(\mathrm{x}) = \hat{y}_{s−1}(\mathrm{x}) + \alpha_s f_s(\mathrm{x})$.

We heuristically assume that the
function ${f}$ has the following form:
$$ f_{\ell}(\mathrm{x})={\prod}_{t=1}^{\ell} q_{C_{i}(t)}(\mathrm{x}) $$
where the function _q_ maps latent feature
vector x to real value domain
$$ q_{C_{i}(t)}(\mathrm{x})=\sum_{j\in C_{i}(t)}\mathbb{I}[j\in \mathrm{x}]w_{t} $$

It is hard for a general convex loss function $\ell$ to search function ${f}$ to optimize the objective function:
$L=\sum_{i}\ell(\hat{y}_s(\mathrm{x}_i), y_i)+\Omega(f)$.

The most common way is to approximate it by least-square
minimization, i.e., $\ell={\| \cdot \|}_2^2$. Like in `xGBoost`, it takes second order Taylor expansion of the loss function $\ell$ and problem isfinalized to find the ${i}$(t)-th feature which:

$$\arg{\min}_{i(t)\in \{0, \dots, m\}} \sum_{i=1}^{n} h_i(\frac{g_i}{h_i}-f_{t-1}(\mathrm{x}_i) q_{C_{i}(t)}(\mathrm{x}_i))^2 + {\|\theta\|}_2^2 $$
where the negativefirst derivative and the second derivative at instance ${i}$ as $g_i$ and $h_i$.

- [Gradient boosting factorization machines](http://tongzhang-ml.org/papers/recsys14-fm.pdf)

#### Gradient Boosted Categorical Embedding and Numerical Trees

`Gradient Boosted Categorical Embedding and Numerical Trees (GB-CSENT)` is to combine Tree-based Models and Matrix-based Embedding Models in order to handle numerical features and large-cardinality categorical features.
A prediction is based on:

* Bias terms from each categorical feature.
* Dot-product of embedding features of two categorical features,e.g., user-side v.s. item-side.
* Per-categorical decision trees based on numerical features ensemble of numerical decision trees where each tree is based on one categorical feature.

In details, it is as following:
$$
\hat{y}(x) = \underbrace{\underbrace{\sum_{i=0}^{k} w_{a_i}}_{bias} + \underbrace{(\sum_{a_i\in U(a)} Q_{a_i})^{T}(\sum_{a_i\in I(a)} Q_{a_i}) }_{factors}}_{CAT-E} + \underbrace{\sum_{i=0}^{k} T_{a_i}(b)}_{CAT-NT}.
$$
And it is decomposed as the following table.
_____
Ingredients| Formulae| Features
---|---|---
Factorization Machines |$\underbrace{\underbrace{\sum_{i=0}^{k} w_{a_i}}_{bias} + \underbrace{(\sum_{a_i\in U(a)} Q_{a_i})^{T}(\sum_{a_i\in I(a)} Q_{a_i}) }_{factors}}_{CAT-E}$ | Categorical Features
GBDT |$\underbrace{\sum_{i=0}^{k} T_{a_i}(b)}_{CAT-NT}$ | Numerical Features
_________
- http://www.hongliangjie.com/talks/GB-CENT_SD_2017-02-22.pdf
- http://www.hongliangjie.com/talks/GB-CENT_SantaClara_2017-03-28.pdf
- http://www.hongliangjie.com/talks/GB-CENT_Lehigh_2017-04-12.pdf
- http://www.hongliangjie.com/talks/GB-CENT_PopUp_2017-06-14.pdf
- http://www.hongliangjie.com/talks/GB-CENT_CAS_2017-06-23.pdf
- http://www.hongliangjie.com/talks/GB-CENT_Boston_2017-09-07.pdf
- [Talk: Gradient Boosted Categorical Embedding and Numerical Trees](http://www.hongliangjie.com/talks/GB-CENT_MLIS_2017-06-06.pdf)
- [Paper: Gradient Boosted Categorical Embedding and Numerical Trees](https://qzhao2018.github.io/zhao/publication/zhao2017www.pdf)
- https://qzhao2018.github.io/



### Adaptive Boosting Personalized Ranking (AdaBPR)

`AdaBPR (Adaptive Boosting Personalized Ranking)` is a boosting algorithm for top-N item recommendation using users' implicit feedback.
In this framework, multiple homogeneous component recommenders are linearly combined to achieve more accurate recommendation.
The component recommenders are learned based on a re-weighting strategy that assigns a dynamic weight to each observed user-item interaction.

Here explicit feedback refers to users' ratings to items while implicit feedback is derived
from users' interactions with items, e.g., number of times a user plays a song.

The primary idea of applying boosting for item recommendation is to learn a set of homogeneous component recommenders and then create an ensemble of the component recommenders to predict users' preferences.

Here, we use a linear combination of component recommenders as the final recommendation model
$$f=\sum_{t=1}^{T}{\alpha}_t f_{t}.$$

In the training process, AdaBPR runs for ${T}$ rounds, and the component recommender $f_t$ is created at t-th round by
$$
\arg\min_{f_t\in\mathbb{H}} \sum_{(u,i)\in\mathbb{O}} {\beta}_{u} \exp\{-E(\pi(u,i,\sum_{n=1}^{t}{\alpha}_n f_{n}))\}.
$$

where the notations are listed as follows:

- $\mathbb{H}$ is the set of possible component recommenders such as collaborative ranking algorithms;
- $E(\pi(u,i,f))$ denotes the ranking accuracy associated with each observed interaction pair;
- $\pi(u,i,f)$ is the rank position of item ${i}$ in the ranked item list of ${u}$, resulted by a learned ranking model ${f}$;
- $\mathbb{O}$ is the set of all observed user-item interactions;
- ${\beta}_{u}$ is defined as reciprocal of the number of user $u$'s  historical items  ${\beta}_{u}=\frac{1}{|V_{u}^{+}|}$ ($V_{u}^{+}$ is the historical items of ${u}$).
***
- [A Boosting Algorithm for Item Recommendation with Implicit Feedback](https://www.ijcai.org/Proceedings/15/Papers/255.pdf)
- [The review @Arivin's blog](http://www.arvinzyy.cn/2017/09/23/A-Boosting-Algorithm-for-Item-Recommendation-with-Implicit-Feedback/)


## Explainable Recommendations

Explainable recommendation and search attempt to develop models or methods that not only generate high-quality recommendation or search results, but also intuitive explanations of the results for users or system designers, which can help improve the system transparency, persuasiveness, trustworthiness, and effectiveness, etc.

Providing personalized explanations for recommendations can help users to understand the underlying insight of the recommendation results, which is helpful to the effectiveness, transparency, persuasiveness and trustworthiness of recommender systems. Current explainable recommendation models mostly generate textual explanations based on pre-defined sentence templates. However, the expressiveness power of template-based explanation sentences are limited to the pre-defined expressions, and manually defining the expressions require significant human efforts

+ [Explainable Recommendation and Search @ rutgers](https://www.cs.rutgers.edu/content/explainable-recommendation-and-search)
+ [Explainable Recommendation: A Survey and New Perspectives](https://www.groundai.com/project/explainable-recommendation-a-survey-and-new-perspectives/)
+ [Explainable Entity-based Recommendations with Knowledge Graphs](http://www.cs.cmu.edu/~wcohen/postscript/recsys-2017-poster.pdf)
+ [2018 Workshop on Explainable Recommendation and Search (EARS 2018)](https://ears2018.github.io/)
+ [EARS 2019](https://sigir.org/sigir2019/program/workshops/ears/)
+ [Explainable Recommendation and Search (EARS)](http://yongfeng.me/projects/)
+ [TEM: Tree-enhanced Embedding Model for Explainable Recommendation](http://staff.ustc.edu.cn/~hexn/slides/www18-tree-embedding-recsys.pdf)
+ https://ears2019.github.io/
+ [Explainable Recommendation for Self-Regulated Learning](http://www.cogsys.org/papers/ACSvol6/posters/Freed.pdf)
+ [Dynamic Explainable Recommendation based on Neural Attentive Models](http://www.yongfeng.me/attach/dynamic-explainable-recommendation.pdf)
+ https://github.com/fridsamt/Explainable-Recommendation
+ [Explainable Recommendation for Event Sequences: A Visual Analytics Approach by Fan Du](https://talks.cs.umd.edu/talks/2028)
+ https://wise.cs.rutgers.edu/code/
+ http://www.cs.cmu.edu/~rkanjira/thesis/rose_proposal.pdf
+ http://jamesmc.com/publications
+ [FIRST INTERNATIONAL WORKSHOP ON  DEEP MATCHING IN PRACTICAL APPLICATIONS ](https://wsdm2019-dapa.github.io/#section-ketnotes)

## Social Recommendation

[We present a novel framework for studying recommendation algorithms in terms of the ‘jumps’ that they make to connect people to artifacts. This approach emphasizes reachability via an algorithm within the `implicit graph structure` underlying a recommender dataset and allows us to consider questions relating algorithmic parameters to properties of the datasets.](http://people.cs.vt.edu/~ramakris/papers/receval.pdf)

User-item/user-user interactions are usually in the form of graph/network structure. What is more, the graph is dynamic, and  we need to apply to new nodes without model retraining.

- [ ] [Accurate and scalable social recommendation using mixed-membership stochastic block models](https://www.pnas.org/content/113/50/14207)
- [ ] [Do Social Explanations Work? Studying and Modeling the
Effects of Social Explanations in Recommender Systems](https://arxiv.org/pdf/1304.3405.pdf)
- [ ] [Existing Methods for Including Social Networks until 2015](http://ajbc.io/projects/slides/chaney_recsys2015.pdf)
- [ ] [Social Recommendation With Evolutionary Opinion Dynamics](https://shiruipan.github.io/pdf/TSMC-18-Xiong.pdf)
- [ ] [Workshop on Responsible Recommendation](https://piret.gitlab.io/fatrec/)
- [ ] https://recsys.acm.org/recsys18/fatrec/
- [ ] [A Probabilistic Model for Using Social Networks in Personalized Item Recommendation](http://ajbc.io/projects/papers/Chaney2015.pdf)
- [ ] [Product Recommendation and Rating Prediction based on Multi-modal Social Networks](http://delab.csd.auth.gr/papers/RecSys2011stm.pdf)
- [ ] [Graph Neural Networks for Social Recommendation](https://paperswithcode.com/paper/graph-neural-networks-for-social)
- [ ] [Studying Recommendation Algorithms by Graph Analysis](http://people.cs.vt.edu/~ramakris/papers/receval.pdf)
- [ ] [Low-rank Linear Cold-Start Recommendation from Social Data](https://akmenon.github.io/papers/loco/loco-paper.pdf)

### SocialMF: MF with social trust propagation

Based on the assumption of trust aware recommender
* users have similar tastes with other users they trust
* the transitivity of trust and propagate trust to indirect neighbors in the social network.

## Knowledge Graph and Recommender System

Items usually correspond to entities in many fields, such as books, movies and music, making it possible for transferring information between them.
These information involving in `recommender system and knowledge graph` are complementary revealing the connectivity among items or between users and items.
In terms of models, the two tasks are both to rank candidates for a target according to either implicit or explicit relations.
For example, KG completion is to find correct movies (e.g., Death Becomes Her) for the person Robert Zemeckis given the explicit relation is Director Of.
Item recommendation aims at recommending movies for a target user satisfying some implicit preference.
Therefore, we are to fill in the gap between `item recommendation` and `KG completion` via a joint model, and systematically investigate how the two tasks impact each other.


- [ ] [推荐算法不够精准？让知识图谱来解决](https://www.msra.cn/zh-cn/news/features/embedding-knowledge-graph-in-recommendation-system-i)
- [ ] [如何将知识图谱特征学习应用到推荐系统？](https://www.msra.cn/zh-cn/news/features/embedding-knowledge-graph-in-recommendation-system-ii)
- [ ] [可解释推荐系统：身怀绝技，一招击中用户心理](https://www.msra.cn/zh-cn/news/features/explainable-recommender-system-20170914)
- [ ] [深度学习与知识图谱在美团搜索广告排序中的应用实践](https://tech.meituan.com/2018/06/07/searchads-dnn.html)
- [ ] [Unifying Knowledge Graph Learning and Recommendation: Towards a Better Understanding of User Preferences](http://staff.ustc.edu.cn/~hexn/papers/www19-KGRec.pdf)
- [ ] [Explainable Reasoning over Knowledge Graphs for Recommendation](https://arxiv.org/pdf/1811.04540.pdf)

## Reinforcement Learning and Recommender System

Services that introduce stores to users on the Internet are increasing in recent years. Each service conducts thorough analyses in order to display stores matching each user's preferences. In the field of recommendation, collaborative filtering performs well when there is sufficient click information from users. Generally, when building a user-item matrix, data sparseness becomes a problem. It is especially difficult to handle new users. When sufficient data cannot be obtained, a multi-armed bandit algorithm is applied. Bandit algorithms advance learning by testing each of a variety of options sufficiently and obtaining rewards (i.e. feedback). It is practically impossible to learn everything when the number of items to be learned periodically increases. The problem of having to collect sufficient data for a new user of a service is the same as the problem that collaborative filtering faces. In order to solve this problem, we propose a recommender system based on deep reinforcement learning. In deep reinforcement learning, a multilayer neural network is used to update the value function.

* [eep reinforcement learning for recommender systems](https://ieeexplore.ieee.org/document/8350761)
* [Deep Reinforcement Learning for Page-wise Recommendations](https://pdfs.semanticscholar.org/5956/c34032126185d8ad19695e4a1a191c08b5a1.pdf)
* [A Reinforcement Learning Framework for Explainable Recommendation](https://www.microsoft.com/en-us/research/uploads/prod/2018/08/main.pdf)
+ [Generative Adversarial User Model for Reinforcement Learning Based Recommendation System](https://arxiv.org/abs/1812.10613)
+ [Adversarial Personalized Ranking for Recommendation](http://bio.duxy.me/papers/sigir18-adversarial-ranking.pdf)
+ [Adversarial Training Towards Robust Multimedia Recommender System](https://github.com/duxy-me/AMR)
+ [Explore, Exploit, and Explain: Personalizing Explainable Recommendations with Bandits](http://jamesmc.com/blog/2018/10/1/explore-exploit-explain)
+ [Learning from logged bandit feedback](https://drive.google.com/file/d/0B2Rxz7LRWLOMX2dycXpWTGxoUE5lNkRnRWZuaDNZUlVRZ1kw/view)
+ [Improving the Quality of Top-N Recommendation](https://drive.google.com/file/d/0B2Rxz7LRWLOMekRtdExZVVpZQmlXNks0Y2FJTnd6ZG90TXdZ/view)

_______
|Traditional Approaches | Beyond Traditional Methods|
|---------------------- |--------------------------|
|Collaborative Filtering | Tensor Factorization & Factorization Machines|
|Content-Based Recommendation | Social Recommendations|
|Item-based Recommendation | Learning to rank|
|Hybrid Approaches | MAB Explore/Exploit|


- [ ] https://github.com/wzhe06/Reco-papers
- [ ] https://github.com/hongleizhang/RSPapers
- [ ] https://github.com/hongleizhang/RSAlgorithms
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
- [ ] [Social Media Mining: An Introduction](http://dmml.asu.edu/smm/slides/)
- [ ] http://dmml.asu.edu/smm/slide/SMM-Slides-ch9.pdf
- [ ] [PRS 2019](https://prs2018.splashthat.com/)

## Implementation

- [ ] https://github.com/gasevi/pyreclab
- [ ] https://github.com/cheungdaven/DeepRec
- [ ] https://github.com/cyhong549/DeepFM-Keras
- [ ] https://github.com/grahamjenson/list_of_recommender_systems
- [ ] https://github.com/maciejkula/spotlight
- [ ] https://github.com/Microsoft/Recommenders
- [ ] https://github.com/alibaba/euler
- [ ] https://github.com/alibaba/x-deeplearning/wiki/
- [ ] https://github.com/lyst/lightfm
- [ ] [Surprise: a Python scikit building and analyzing recommender systems](http://surpriselib.com/)
- [ ] [Orange3-Recommendation: a Python library that extends Orange3 to include support for recommender systems.](https://orange3-recommendation.readthedocs.io/en/latest/)
- [ ] [MyMediaLite: a recommender system library for the Common Language Runtime](http://www.mymedialite.net/index.html)
- [ ] http://www.mymediaproject.org/
- [Workshop: Building Recommender Systems w/ Apache Spark 2.x](https://qcon.ai/qconai2019/workshop/building-recommender-systems-w-apache-spark-2x)
- [A Leading Java Library for Recommender Systems](https://www.librec.net/)
- [lenskit: Python Tools for Recommender Experiments](https://lenskit.org/)
- [Samantha - A generic recommender and predictor server](https://grouplens.github.io/samantha/)


# Computational Advertising

Online advertising has grown over the past decade to over $26 billion in recorded revenue in 2010. The revenues generated are based on different pricing models that can be fundamentally grouped into two types: cost per (thousand) impressions (CPM) and cost per action (CPA), where an action can be a click, signing up with the advertiser, a sale, or any other measurable outcome. A web publisher generating revenues by selling advertising space on its site can offer either a CPM or CPA contract. We analyze the conditions under which the two parties agree on each contract type, accounting for the relative risk experienced by each party.

The information technology industry relies heavily on the on-line advertising such as [Google，Facebook or Alibaba].
Advertising is nothing except information, which is not usually accepted gladly. In fact, it is more difficult than recommendation because it is less known of the context where the advertisement is placed.


[Hongliang Jie](http://www.hongliangjie.com/talks/Etsy_ML.pdf) shares 3 challenges of computational advertising in Etsy,
which will be the titles of the following subsections.

<img title="ad" src="https://gokulchittaranjan.files.wordpress.com/2016/06/blog-on-advertising-figures-e1466486030771.png" width="70%" />

* [广告为什么要计算](https://zhuanlan.zhihu.com/p/72092504)
* [计算广告资料汇总](https://www.jianshu.com/p/8c591feb9fc4)
* [ONLINE VIDEO ADVERTISING: All you need to know in 2019](https://strategico.io/video-advertising/)
* [CAP 6807: Computational Advertising and Real-Time Data Analytics](http://www.cse.fau.edu/~xqzhu/courses/cap6807.html)
* [Tutorial: Information Retrieval Challenges in Computational Advertising](http://www.cikm2011.org/tutorials/PM3.html)
* [计算广告](https://dirtysalt.github.io/html/computational-advertising.html)
* [计算广告和机器学习](http://www.52caml.com/)
* https://headerbidding.co/category/adops/
* [Deep Learning Based Modeling in Computational Advertising: A Winning Formula](https://www.omicsonline.org/open-access/deep-learning-based-modeling-in-computational-advertising-a-winning-formula-2169-0316-1000266.pdf)
* [Computational Marketing](https://www.marketing-schools.org/types-of-marketing/computational-marketing.html)
* [Data Science and Analytics in Computational Advertising](https://gokulchittaranjan.wordpress.com/2016/06/22/datascienceinadvertising/)

## Click-Through Rate Modeling


**GBRT+LR**

When the feature vector ${x}$ are given, the tree split the features by GBRT then we transform and input the features to the logistic regression.

[Practical Lessons from Predicting Clicks on Ads at Facebook](https://www.jianshu.com/p/96173f2c2fb4) or the [blog](https://zhuanlan.zhihu.com/p/25043821) use the GBRT to select proper features and LR to map these features into the interval $[0,1]$ as a ratio.
Once we have the right features and the right model (decisions trees plus logistic regression), other factors play small roles (though even small improvements are important at scale).

<img src="https://pic4.zhimg.com/80/v2-fcb223ba88c456ce34c9d912af170e97_hd.png" width = "40%" />

* [聊聊CTR预估的中的深度学习](http://kubicode.me/2018/03/19/Deep%20Learning/Talk-About-CTR-With-Deep-Learning/)
* [Deep Models at DeepCTR](https://deepctr.readthedocs.io/en/latest/models/DeepModels.html)
* [CTR预估算法之FM, FFM, DeepFM及实践](https://blog.csdn.net/john_xyz/article/details/78933253)
* [Turning Clicks into Purchases](https://www.hongliangjie.com/talks/SF_2018-05-09.pdf)
* https://github.com/shenweichen/DeepCTR
* https://github.com/wzhe06/CTRmodel
* https://github.com/cnkuangshi/LightCTR
* https://github.com/evah/CTR_Prediction
* http://2016.qconshanghai.com/track/3025/
* https://blog.csdn.net/u011747443/article/details/68928447

## Conversion Rate Modeling

+ [ ] [Post-Click Conversion Modeling and Analysis for Non-Guaranteed Delivery Display Advertising](http://people.csail.mit.edu/romer/papers/NGDAdvertisingWSDM12.pdf)
+ [ ] [Estimating Conversion Rate in Display Advertising from Past Performance Data](http://wnzhang.net/share/rtb-papers/cvr-est.pdf)
+ [ ] [https://www.optimizesmart.com/](https://www.optimizesmart.com/introduction-machine-learning-conversion-optimization/)

## Bid Optimization

+ [A collection of research and survey papers of real-time bidding (RTB) based display advertising techniques.](https://github.com/wnzhang/rtb-papers)

****

* [Papers on Computational Advertising](https://github.com/wzhe06/Ad-papers)
* [CAP 6807: Computational Advertising and Real-Time Data Analytics](http://www.cse.fau.edu/~xqzhu/courses/cap6807.html)
* [Computational Advertising Contract Preferences for Display Advertising](https://www.soe.ucsc.edu/departments/technology-management/research/computational-advertising)
* [Machine Learning for Computational Advertising, UC Santa Cruz, April 22, 2009, Alex Smola, Yahoo Labs, Santa Clara, CA](http://alex.smola.org/teaching/ucsc2009/)
* [Computational Advertising and Recommendation](https://people.eecs.berkeley.edu/~jfc/DataMining/SP12/lecs/lec12.pdf)
* [Practical Lessons from Predicting Clicks on Ads at Facebook](http://quinonero.net/Publications/predicting-clicks-facebook.pdf)
* http://yelp.github.io/MOE/
* http://www.hongliangjie.com/talks/AICon2018.pdf
* https://sites.google.com/view/tsmo2018/invited-talks
* https://matinathomaidou.github.io/research/
* https://www.usermind.com/
______________________________________________________

## User Engagement

[User engagement measures whether users find value in a product or service. Engagement can be measured by a variety or combination of activities such as downloads, clicks, shares, and more. Highly engaged users are generally more profitable, provided that their activities are tied to valuable outcomes such as purchases, signups, subscriptions, or clicks.](https://mixpanel.com/topics/what-is-user-engagement/)

* [WHAT IS USER ENGAGEMENT?](https://mixpanel.com/topics/what-is-user-engagement/)
* [What is Customer Engagement, and Why is it Important?](https://blog.smile.io/what-is-customer-engagement-and-why-is-it-important)
* [What is user engagement? A conceptual framework for defining user engagement with technology](https://open.library.ubc.ca/cIRcle/collections/facultyresearchandpublications/52383/items/1.0107445)
* [How to apply AI for customer engagement](https://www.pega.com/artificial-intelligence-applications)
* [The future of customer engagement](https://dma.org.uk/event/the-future-of-customer-engagement)
* [Second Uber Science Symposium: Exploring Advances in Behavioral Science](https://eng.uber.com/second-uber-science-symposium-behavioral-science)
* [Measuring User Engagement](https://mounia-lalmas.blog/2013/04/29/measuring-user-engagement/)
* https://uberbehavioralsciencesymposium.splashthat.com/
* https://inlabdigital.com/
* https://www.futurelab.net/
- [The User Engagement Optimization Workshop2](http://www.ueo-workshop.com/)
- [The User Engagement Optimization Workshop1](http://www.ueo-workshop.com/previous-editions/ueo-2013-at-cikm-2013/)
- [EVALUATION OF USER EXPERIENCE IN MOBILE ADVERTISI](http://galjot.si/research)
- [WWW 2019 Tutorial on Online User Engagement](https://onlineuserengagement.github.io/)
- https://www.nngroup.com/

## Labs

- [Recommender Systems](http://csse.szu.edu.cn/csse.szu.edu.cn/staff/panwk/recommendation/index.html)
- https://libraries.io/github/computational-class
- http://www.52caml.com/
- [洪亮劼，博士 – Etsy工程总监](https://www.hongliangjie.com/)
- [Data Mining Machine Learning @The University of Texas at Austin](http://www.ideal.ece.utexas.edu/)
- [Center for Big Data Analytics @The University of Texas at Austin](https://bigdata.oden.utexas.edu/)
- [Multimedia Computing Group@tudelft.nl](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/multimedia-computing/)
- [knowledge Lab@Uchicago](https://www.knowledgelab.org/)
- [DIGITAL TECHNOLOGY CENTER@UMN](https://www.dtc.umn.edu/)
- [The Innovation Center for Artificial Intelligence (ICAI)](https://icai.ai/)
- [Data Mining and Machine Learning lab (DMML)@ASU](http://dmml.asu.edu/)
- [Next Generation Personalization Technologies](http://ids.csom.umn.edu/faculty/gedas/NSFcareer/)
- [Recommender systems & ranking](https://sites.google.com/view/chohsieh-research/recommender-systems)
- [Secure Personalization: Building Trustworthy Recommender Systems](https://www.nsf.gov/awardsearch/showAward?AWD_ID=0430303)
- [Similar grants of  Next Generation Personalization Technologies](https://app.dimensions.ai/details/grant/grant.3063812)
- [Big Data and Social Computing Lab @UIC](https://bdsc.lab.uic.edu/)
- [Web Intelligence and Social Computing](https://www.cse.cuhk.edu.hk/irwin.king/wisc_lab/home)
- [Welcome to the family, Zalando AdTech Lab Hamburg!](https://jobs.zalando.com/tech/blog/zalando-adtech-lab-hamburg/)
- [Data and Marketing Associat](https://dma.org.uk/)
- [Web search and data mining(WSDM) 2019](http://www.wsdm-conference.org/2019/)
- [Web Intelligent Systems and Economics(WISE) lab @Rutgers](https://wise.cs.rutgers.edu/)
- [Ishizuka Lab. was closed. (2013.3) ](http://www.miv.t.u-tokyo.ac.jp/HomePageEng.html)
- [Online Marketing Congress 2017](https://gorrion.io/blog/online-marketing-congress-2017)
- [course-materials of Sys for ML/AI](https://pooyanjamshidi.github.io/mls/course-materials/)
- https://sigopt.com/blog/
- [Web Understanding, Modeling, and Evaluation Lab](http://wume.cse.lehigh.edu/)
- https://knightlab.northwestern.edu/
