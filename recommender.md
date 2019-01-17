# Recommender System

Recommender Systems (RSs) are software tools and techniques providing suggestions for items to be of use to a user.

RSs are primarily directed towards individuals who lack sufficient personal experience or competence to evaluate the potentially overwhelming number of alternative items that a Web site, for example, may offer.

- [ ] https://github.com/hongleizhang/RSPapers
- [ ] https://github.com/YuyangZhangFTD/awesome-RecSys-papers
- [ ] https://github.com/daicoolb/RecommenderSystem-Paper
- [ ] https://github.com/grahamjenson/list_of_recommender_systems
- [ ] https://github.com/benfred/implicit

## Collaborative Filtering

If we have collected user ${u}$'s evaluation to the item ${i}$ ,  $R_{[u][i]}$, and all such data makes up a matrix $R=(R_{[u][i]})$ while the user $u$ cannot evaluate all the item so that the matrix is incomplete and missing much data.

Matrix completion is to complete the matrix $X$ with missing elements, such as

$$
\min Rank(Z) \\
s.t. \sum_{(i,j):Observed}(Z_{(i,j)}-X_{(i,j)})^2\leq \delta 
$$

![](https://pic3.zhimg.com/80/dc9a2b89742a05c3cd2f025105ba1c4a_hd.png)

SVD is to factorize the matrix into the multiplication of matrices so that 
$$
\hat{R}=P^{T}Q
$$

And we can predict the score $R_{[u][i]}$ via 
$$
\hat{R}_{[u][i]}=\hat{r}_{u,i}=\left<P_u,Q_i\right>=\sum_f p_{u,f}q_{i,f}
$$
where $P_u, Q_i$ is the ${u}$th column of ${P}$ and the ${i}$th column of ${Q}$, respectively.
And we can define the cost function
$$
C(P,Q) = \sum_{(u,i):Observed}(r_{u,i}-\hat{r}_{u,i})^{2}=\sum_{(u,i):Observed}(r_{u,i}-\sum_f p_{u,f}q_{i,f})^{2}.
$$
Additionally, we can add regular term into the cost function to void over-fitting 
$$
C(P,Q) = \sum_{(u,i):Observed}(r_{u,i}-\sum_f p_{u,f}q_{i,f})^{2}+\lambda(\|P_u\|^2+\|Q_i\|^2).
$$

***

* https://www.zhihu.com/question/47716840/answer/110843844
* https://www.cnblogs.com/Xnice/p/4522671.html
* https://blog.csdn.net/turing365/article/details/80544594
* https://en.wikipedia.org/wiki/Collaborative_filtering
* https://bugra.github.io/work/notes/2014-04-19/alternating-least-squares-method-for-collaborative-filtering/

## Deep Learning and RS

Deep learning is powerful in processing visual and text information.

- [ ] https://github.com/robi56/Deep-Learning-for-Recommendation-Systems
- [ ] https://github.com/wzhe06/Reco-papers
- [ ] https://github.com/shenweichen/DeepCTR
- [ ] https://github.com/cyhong549/DeepFM-Keras
  
## Computational Advertisement

* https://github.com/wzhe06/Ad-papers

