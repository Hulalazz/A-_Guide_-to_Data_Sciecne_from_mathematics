# Recommender System

Recommender Systems (RSs) are software tools and techniques providing suggestions for items to be of use to a user.

RSs are primarily directed towards individuals who lack sufficient personal experience or competence to evaluate the potentially overwhelming number of alternative items that a Web site, for example, may offer.

- [ ] https://github.com/hongleizhang/RSPapers
- [ ] https://github.com/maciejkula/spotlight
- [ ] https://github.com/Microsoft/Recommenders
- [ ] https://github.com/YuyangZhangFTD/awesome-RecSys-papers
- [ ] https://github.com/daicoolb/RecommenderSystem-Paper
- [ ] https://github.com/grahamjenson/list_of_recommender_systems
- [ ] https://github.com/benfred/implicit
- [ ] https://www.msra.cn/zh-cn/news/features/embedding-knowledge-graph-in-recommendation-system-i
- [ ] https://www.msra.cn/zh-cn/news/features/embedding-knowledge-graph-in-recommendation-system-ii
- [ ] https://www.msra.cn/zh-cn/news/features/explainable-recommender-system-20170914

## Collaborative Filtering

If we have collected user ${u}$'s explicit evaluation score to the item ${i}$ ,  $R_{[u][i]}$, and all such data makes up a matrix $R=(R_{[u][i]})$ while the user $u$ cannot evaluate all the item so that the matrix is incomplete and missing much data.

Matrix completion is to complete the matrix $X$ with missing elements, such as

$$
\min Rank(Z) \\
s.t. \sum_{(i,j):Observed}(Z_{(i,j)}-X_{(i,j)})^2\leq \delta
$$

![](https://pic3.zhimg.com/80/dc9a2b89742a05c3cd2f025105ba1c4a_hd.png)

SVD is to factorize the matrix into the multiplication of matrices so that
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
C(P,Q) = \sum_{(u,i):Observed}(r_{u,i}-\hat{r}_{u,i})^{2}=\sum_{(u,i):Observed}(r_{u,i}-\sum_f p_{u,f}q_{i,f})^{2}
$$
where $\lambda_u$ is always equal to $\lambda_i$.

Additionally, we can add regular term into the cost function to void over-fitting

$$
C(P,Q) = \sum_{(u,i):Observed}(r_{u,i}-\sum_f p_{u,f}q_{i,f})^{2}+\lambda_u\|P_u\|^2+\lambda_i\|Q_i\|^2.
$$

And the evaluation score is always positive and discrete such as $\{2, 4, 6, 8. 10\}$. This technique is also called **nonnegative matrix factorization**.
***

Another advantage of collaborative filtering or matrix completion is that even the element of matrix is binary or implicit information such as

* [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf),
* [Applications of the conjugate gradient method for implicit feedback collaborative filtering](http://rs1.sze.hu/~gtakacs/download/recsys_2011_draft.pdf),
* https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/
* [a curated list in github.com](https://github.com/benfred/implicit).

WRMF is simply a modification of this loss function:
$$C(P,Q)_{WRMF} = \sum_{(u,i):Observed}c_{u,i}(I_{u,i} - \sum_f p_{u,f}q_{i,f})^{2} + \lambda_u\|P_u\|^2 + \lambda_i\|Q_i\|^2.$$

We make the assumption that if a user has interacted at all with an item, then $I_{u,i} = 1$. Otherwise, $I_{u,i} = 0$.
If we take $d_{u,i}$ to be the number of times a user ${u}$ has clicked on an item ${i}$ on a website, then
$$c_{u,i}=1+\alpha d_{u,i}$$
where $\alpha$ is some hyperparameter determined by cross validation.
The new  term in cost function $C=(c_{u,i})$ is called confidence matrix.

WRMF does not make the assumption that a user who has not interacted with an item does not like the item. WRMF does assume that that user has a negative preference towards that item, but we can choose how confident we are in that assumption through the confidence hyperparameter.

[Alternating least square](http://suo.im/4YCM5f) can give an analytic solution to this optimization problem by setting the gradients equal to 0s.

* https://www.ethanrosenthal.com/2016/10/19/implicit-mf-part-1/
* https://www.ethanrosenthal.com/2016/11/07/implicit-mf-part-2/
* http://datamusing.info/blog/2015/01/07/implicit-feedback-and-collaborative-filtering/
* http://nicolas-hug.com/blog/matrix_facto_1
* http://nicolas-hug.com/blog/matrix_facto_2
* http://nicolas-hug.com/blog/matrix_facto_3

SVD + bias considers the user's preferences or bias.
It predict the scores by
$$
\hat{r}_{u,i} = \mu + b_u + b_i + \left< P_u, Q_u \right>
$$
and the cost function is defined as
$$
\min\sum_{(u,i): Observed}(r_{u,i} - \hat{r}_{u,i})^2 + \lambda (\|P_u\|^2+\|Q_i\|^2+\|b_i\|^2+\|b_u\|^2).
$$

* https://cloud.tencent.com/developer/article/1107364
* https://zhuanlan.zhihu.com/p/42269534
* https://orange3-recommendation.readthedocs.io/en/latest/widgets/svdplusplus.html
* https://www.bbsmax.com/A/KE5Q0M9ZJL/

***

* https://github.com/benfred/implicit
* https://www.zhihu.com/question/47716840/answer/110843844
* http://surpriselib.com/
* https://www.cnblogs.com/Xnice/p/4522671.html
* https://blog.csdn.net/turing365/article/details/80544594
* https://en.wikipedia.org/wiki/Collaborative_filtering
* \url{https://www.wikiwand.com/en/Matrix_factorization_(recommender_systems)}
* https://bugra.github.io/work/notes/2014-04-19/alternating-least-squares-method-for-collaborative-filtering/

## Deep Learning and RS

Deep learning is powerful in processing visual and text information so that it helps to find the interests of users such as
[Deep Interest Network](http://www.cnblogs.com/rongyux/p/8026323.html), [xDeepFM](https://www.jianshu.com/p/b4128bc79df0)  and more.

Deep learning models for recommender system may come from the restricted Boltzman machine.
And deep learning models are powerful information extractors.
Deep learning is really popular in recommender system such as [spotlight](https://github.com/maciejkula/spotlight).


- [ ] https://github.com/robi56/Deep-Learning-for-Recommendation-Systems
- [ ] https://github.com/wzhe06/Reco-papers
- [ ] https://github.com/shenweichen/DeepCTR
- [ ] https://github.com/cyhong549/DeepFM-Keras
- [ ] http://dlrs-workshop.org/
- [ ] https://nycdatascience.com/blog/student-works/deep-learning-meets-recommendation-systems/
- [ ] https://www.ethanrosenthal.com/2016/12/05/recasketch-keras/

## Computational Advertisement

* https://github.com/wzhe06/Ad-papers
