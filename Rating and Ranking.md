# Rating and Ranking

The basic idea is the back-feed from the results.
After each game, this data is updated for the participants in the game.
The rating of the winner is increased, and the rating of the loser is decreased.

* https://www.remi-coulom.fr/WHR/WHR.pdf
* http://www.ams.org/notices/201301/rnoti-p81.pdf
* https://www.cs.cornell.edu/jeh/book2016June9.pdf

## Elo Rating

Elo rating is popular at many games such as Go game, soccer and so on.
It supposes that the performance are random and the winning rate is determined by the differences of two players.
If the differences of the scores is beyond some threshold such as ${200}$, it is supposed that the winner probability is ${3/4}$.
And  it is natural if difference of scores is ${0}$, the winning rate is $1/2$.

We assume that
$$P(d) = \frac{1}{1+e^{-\frac{d}{\theta}}}$$
where the parameter $\theta$ is related with the threshold.

For example, the expected performance of player A is $E_{A} = \frac{1}{1 + 10^{-\frac{R_A - R_B}{400}}}$ and $E_{B} = \frac{1}{1 + 10^{-\frac{R_B - R_A}{400}}}=1-E_{A}$.
Supposing Player A was expected to score $E_{A}$ points but actually scored $S_{A}$ points. And the update rule is
$${R}_{A}^{New} = R_{A} + K(S_A - E_A)$$
where $K$ is a constant.

* https://www.wikiwand.com/en/Elo_rating_system
* https://www.wikiwand.com/en/Bradley%E2%80%93Terry_model
* http://www.calvin.edu/~rpruim/fast/R/BradleyTerry/html/BTm.html
* https://homepage.divms.uiowa.edu/~luke/xls/glim/glim/node8.html

## Glicko

The problem with the Elo system that the Glicko system addresses has to do with the
reliability of a player’s rating.

Glickman's principal contribution to measurement is "ratings reliability", called RD, for ratings deviation.
The RD measures the accuracy of a player's rating, with one RD being equal to one standard deviation.
If the player is unrated, the rating is usually set to 1500 and the RD to 350.

1. Determine RD

   The new Ratings Deviation (RD) is found using the old Ratings Deviation $RD_0$:
$$
   RD=\min\{\sqrt{RD_0^2+c_2t}, 350\}
$$
   where ${t}$ is the amount of time (rating periods) since the last competition and '350' is assumed to be the RD of an unrated player. And $c=\sqrt{(350^2-50^2)/100}\simeq 34.6$.
2. Determine New Rating

   The new ratings, after a series of m games, are determined by the following equation:
$$
   r=r_0+\frac{q}{RD^{-2}+d^{-2}}\sum_{i=1}^{m}g(RD_i)(s_i - E(s|r,r_i,RD_i))
$$
   where $g(RD_i)=\{1+\frac{3q^2(RD_i)^2}{\pi^2}\}^{-1/2}$, $E(s|r,r_i,RD_i))=\{1+10^{(\frac{g(RD_i)(r-r_i)}{-400})}\}$, $q=\frac{\ln(10)}{400}\approx 0.00575646273$, $d^{-2} = q^2\sum_{i=1}^{m}[g(RD_i)^2]E(s|r,r_i,RD_i)[1-E(s|r,r_i,RD_i)]$, $r_i$ represents the ratings of the individual opponents. $s_i$ represents the outcome of the individual games. A win is ${1}$, a draw is $\frac {1}{2}$, and a loss is $0$.

3. Determine New Ratings Deviation

   $$RD^{\prime}=\sqrt{(RD^{-2}+d^{-2})^{-1}} .$$

* http://www.glicko.net/research.html
* https://www.wikiwand.com/en/Glicko_rating_system
* https://zhuanlan.zhihu.com/p/21434252
* http://www.glicko.net/glicko.html

## TrueSkill

As shown in the rule to update the score in Elo, it only take the differences of score into consideration.
The TrueSkill system will assume that the distribution of the skill is **location-scale** distribution. In fact, the prior distribution in Elo is **Gaussian distribution**.
The expected performance of the player is his mean of the distribution. The variance is the uncertainty  of the system.

The update rule will update the mean and variance:
$$
\mu_{winner} \leftarrow \mu_{winner} + \frac{\sigma_{winner}^2}{c} \cdot \nu(\frac{(\mu_{winner}-\mu_{loser})}{c},\frac{\epsilon}{c})
$$
$$
\mu_{loser} \leftarrow \mu_{loser} - \frac{\sigma_{loser}^2}{c} \cdot \nu(\frac{(\mu_{winner}-\mu_{loser})}{c},\frac{\epsilon}{c})
$$
$$
\sigma_{winner} \leftarrow \sigma_{winner}^2 [1 - \frac{\sigma_{winner}^2}{c^2} \cdot w(\frac{(\mu_{winner}-\mu_{loser})}{c},\frac{\epsilon}{c}) ]
$$
$$
\sigma_{loser} \leftarrow \sigma_{loser}^2 [1 - \frac{\sigma_{loser}^2}{c^2} \cdot w(\frac{(\mu_{winner}-\mu_{loser})}{c},\frac{\epsilon}{c}) ]
$$
$$
c^2 = 2\beta^2+\sigma^2_{winner}+\sigma^{2}_{loser}
$$

where $\beta^2$ is the average of all players' variances, $\nu(t) =\frac{N(t)}{\Phi(t)}, w(t)=\nu(t)[\nu(t)+1]$. And $N(t),\Phi(t)$ is the PDF and CDF of standard normal distribution, respectively.

- [X] https://www.wikiwand.com/en/TrueSkill
- [X] https://www.jianshu.com/p/c1fbba3af787
- [ ] https://zhuanlan.zhihu.com/p/48737998
- [ ] https://www.wikiwand.com/en/Location%E2%80%93scale_family

## Whole-History Rating

Incremental Rating Systems or dynamical rating systems such as TrueSkill  do not make optimal use of data.
The principle of Bayesian Inference consists in computing a probability distribution
over player ratings ($r$) from the observation of game results ($G$) by inverting the model thanks to Bayes formula:

$$
P(r|G)=\frac{P(G|r)P(r)}{P(G)}
$$

where $P(r)$ is a prior distribution over $r$, and $P(G)$ is a normalizing constant. $P(r|G)$ is called the posterior distribution of $r$
$P(G|r)$ is the Bradley-Terry model, i.e.

$$
P(\text{player $i$ beats player $j$ at time $t$})= \frac{1}{1+10^{-\frac{R_i(t)-R_j(t)}{400}}}
$$

as shown in Elo rating system.

In the dynamic Bradley-Terry model, the prior has two roles. First, a prior
probability distribution over the range of ratings is applied. This way, the rating
of a player with $100\%$ wins does not go to infinity. Also, a prior controls the
variation of the ratings in time, to avoid huge jumps in ratings.
In the dynamic Bradley-Terry model, the prior that controls the variation of
ratings in time is a Wiener process:

$$
r_i(t_1) - r_i(t_1)\sim N(0, |t_2-t_1|w^2) ,
$$

where $w$ is a parameter of the model, that indicates the variability of ratings in time.
The extreme case of $w = 0$ would mean static ratings.

The WHR algorithm consists in computing, for each player, the $r(t)$ function
that maximizes $P(r|G)$. Once this maximum a posteriori (**MAP**) has been computed,
the variance around this maximum is also estimated, which is a way to estimate rating uncertainty.

Formally, Newton’s method consists in updating the rating vector r of one player (the vector of ratings at times when that player played a game) according to this formula

$$
r\leftarrow r - (\frac{\partial^2 \log(p)}{\partial r^2})^{-1} \frac{\partial \log(p)}{\partial r} .
$$

- [ ] https://www.wikiwand.com/en/Bradley%E2%80%93Terry_model
- [ ] https://www.wikiwand.com/en/Ranking
- [ ] https://arxiv.org/pdf/1701.08055v1.pdf
- [X] https://www.remi-coulom.fr/WHR/WHR.pdf

## Ranking

Combining feedback from multiple users to rank a collection of items is an important task.
The ranker, a central component in every search engine, is responsible for the matching between processed queries and indexed documents in information retrieval.
The goal of a ranking system is to find the best possible
ordering of a set of items for a user, within a specific context,
in real-time in recommender system.

In general, we call all those methods that use machine learning technologies to solve the problem of ranking **"learning-to-rank"** methods or **LTR** or **L2R**.
We are designed to compare some indexed document with the query.
The algorithms above are based on feedback to tune the rank or scores of players or documents. The drawback of these methods is that they do not take the features of the players into consideration.
We may use machine learn to predict the scores of players and test it in real data set such as **RankNet, LambdaRank, and LambdaMART**.


* https://github.com/cgravier/RankLib
* http://fastml.com/evaluating-recommender-systems/
* https://github.com/maciejkula/spotlight/tree/master/examples/movielens_explicit
* https://www.microsoft.com/en-us/research/blog/ranknet-a-ranking-retrospective/
* https://blog.csdn.net/cht5600/article/details/54381011
* https://www.ijcai.org/proceedings/2018/0738.pdf
* http://quickrank.isti.cnr.it/research-papers/
* http://learningtorank.isti.cnr.it/
* https://www.hongliangjie.com/2019/01/20/cikm-2018-papers-notes/
* http://www.cs.cornell.edu/people/tj/publications/joachims_etal_17a.pdf
 
![](https://p1.meituan.net/travelcube/58920553566822f1fe059f95eba71d95131646.png)

### RankSVM 

* https://x-algo.cn/index.php/2016/08/09/ranksvm/
* https://www.cnblogs.com/bentuwuying/p/6683832.html


### RankNet

> RankNet is a feedforward neural network model. Before it can be used its parameters must be learned using a large amount of labeled data, called the training set. The training set consists of a large number of query/document pairs, where for each pair, a number assessing the quality of the relevance of the document to that query is assigned by human experts. Although the labeling of the data is a slow and human-intensive task, training the net, given the labeled data, is fully automatic and quite fast. The system used by Microsoft in 2004 for training the ranker was called The Flying Dutchman. from  [RankNet: A ranking retrospective](https://www.microsoft.com/en-us/research/blog/ranknet-a-ranking-retrospective/).

RankNet takes the ranking  as **regression** task.

Suppose that two players $u_i$ and $u_j$ with feature vectors $x_i$ and $x_j$ is presented to the model, which computes the scores $s_i = f(x_i)$ and $s_j = f(x_j)$.
Another output of the model is the probability that $U_i$ should be ranked
higher than $U_j$ via a sigmoid function, thus
$$
P(U_i\triangleleft U_j)=\frac{1}{1+\exp(-\sigma(s_i-s_j))}=\frac{1}{1+\exp[-\sigma(f(x_i)-f(x_j))]}
$$
where the choice of the parameter $\sigma$ determines the shape of the sigmoid.

**Obviously, the idea also occurs in Elo rating.**

We then apply the cross entropy cost function,
which penalizes the deviation of the model output probabilities from the desired
probabilities:
$$
C=-\overline{P_{ij}}\log(P(U_i\triangleleft U_j))-(1-\overline{P_{ij}})\log(1-P(U_i\triangleleft U_j))
$$
where the labeled constant $\overline{P_{ij}}$ is defined as
$$
\overline{P_{ij}}=
\begin{cases}
1, \text{if $U_i$ is labeled higher than $U_j$;}\\
\frac{1}{2}, \text{if $U_i$ is labeled equal than $U_j$;}\\
0, \text{if $U_i$ is labeled lower than $U_j$.}
\end{cases}
$$

The model $f:\mathbb{R}^P\to\mathbb{R}$ can be deep neural network (a.k.a deep learning), which can learned via stochastic gradient descent methods.
The cost function (cross entropy) can be rewritten as
$$
L_{i,j} = -\overline{P_{ij}}[\log(P(U_i\triangleleft U_j))-\log(1-P(U_i\triangleleft U_j))]-\log(1-P(U_i\triangleleft U_j))\\
=-\overline{P_{ij}}[\log(\frac{P(U_i\triangleleft U_j)}{1-P(U_i\triangleleft U_j)}]-\log(1-P(U_i\triangleleft U_j))\\
=-\overline{P_{ij}}[\log(\frac{1}{\exp(-\sigma(s_i-s_j))})]-\log(\frac{\exp(-\sigma(s_i-s_j))}{1+\exp(-\sigma(s_i-s_j))})\\
=\overline{P_{ij}}[-\sigma(s_i-s_j)]+\sigma(s_i-s_j)+\log[1+\exp(-\sigma(s_i-s_j))]\\
=(1-\overline{P_{ij}})\sigma(s_i-s_j)+\log[1+\exp(-\sigma(s_i-s_j))].
$$

So that we can compute the gradient of cost function with respect of $s_i$ and $s_j$, which are outputs of deep neural network.
And we can use backpropagation and stochastic gradient descent to tune the parameters of the deep neural network.

See more deep learning algorithms on ranking  at [https://github.com/Isminoula/DL-to-Rank] or [http://quickrank.isti.cnr.it/research-papers/].

* https://www.ethanrosenthal.com/2016/11/07/implicit-mf-part-2/
* http://sonyis.me/paperpdf/wsdm233-song-2014.pdf

### LambdaRank

**LambdaRank**  introduced the $\lambda_i$ when update of parameters $w$ of the model $f:\mathbb{R}^P\to\mathbb{R}$.
The key observation of LambdaRank is thus that in order to train a model, we do not need the costs themselves: we only need the gradients (of the costs with respect to
the model scores).

You can think of these gradients as little arrows attached to each document in the ranked list, indicating which direction we’d like those documents to move. LambdaRank simply took the RankNet gradients, which we knew worked well, and scaled them by the change in NDCG found by swapping each pair of documents. We found that this training generated models with significantly improved relevance (as measured by NDCG) and had an added bonus of uncovering a further trick that improved overall training speed (for both RankNet and LambdaRank). Furthermore, surprisingly, we found empirical evidence (see also this paper) that the training procedure does actually optimize NDCG, even though the method, although intuitive and sensible, has no such guarantee.

We compute the  gradients of RankNet by:
$$
\frac{\partial L}{\partial w} = \sum_{(i,j)}\frac{\partial L_{i,j}}{\partial w}=\sum_{(i,j)}[\frac{\partial L_{i,j}}{\partial s_i}+\frac{\partial L_{i,j}}{\partial s_j}].
$$

Observe that
$$\frac{\partial L_{i,j}}{\partial s_i} = -\frac{\partial L_{i,j}}{\partial s_j}$$
and define
$$
{\lambda}_{i,j}=\frac{\partial L_{i,j}}{\partial s_i} = -\frac{\partial L_{i,j}}{\partial s_j} = -\frac{\sigma}{1+\exp(\sigma(s_i-s_j))}.
$$
What is more, we can extend it to
$$
{\lambda}_{i,j}=  -\frac{\sigma}{1+\exp(\sigma(s_i-s_j))}|\Delta Z|.
$$
where $\Delta Z$ is the size of the change in some **Information Retrieval Measures** ${Z}$.

And $\lambda_{i}$ with respect to ${i}$-th item is defined as
$$\lambda_i = \sum_{i\in(i,j)}\lambda_{(i,j)}-\sum_{j\in(j,i)}\lambda_{(i,j)}$$

![](https://p0.meituan.net/travelcube/d6142123b31212f4854fd4e53da5831e14664.png)

***

- http://blog.camlcity.org/blog/lambdarank.html
- http://wnzhang.net/papers/lambdafm.pdf
- http://quinonero.net/Publications/predicting-clicks-facebook.pdf
- https://staff.fnwi.uva.nl/e.kanoulas/wp-content/uploads/Lecture-8-1-LambdaMart-Demystified.pdf
- https://liam.page/uploads/slides/lambdamart.pdf


### LambdaMART

**LambdaMART** is the boosted tree version of LambdaRank, which is based on RankNet. It takes the ranking problem as classification problem.

MART stands for [Multiple Additive Regression Tree](http://statweb.stanford.edu/~jhf/MART.html). 
In LambdaRank, we compute the gradient. And we can use this gradient to make up the GBRT. 
> LambdaMART had an added advantage: the training of tree ensemble models can be very significantly sped up over the neural net equivalent (this work, led by O. Dekel, is not yet published). This allows us to train with much larger data sets, which again gives improved ranking accuracy. From [RankNet: A ranking retrospective](https://www.microsoft.com/en-us/research/blog/ranknet-a-ranking-retrospective/).

***
![LambdaMART](https://liam.page/uploads/images/LTR/LambdaMART.png)
***
To implement LambdaMART we just use MART, specifying appropriate gradients
and the Newton step.
The key point is the gradient of the ${\lambda}_i$:
$$w_i = \frac{\partial y_i}{\partial F_{k-1}(\vec{x}_i)}$$
where $\lambda_i = y_i$ is defined in **LambdaRank**.
LambdaRank updates all the weights after each query is examined. The
decisions (splits at the nodes) in LambdaMART, on the other hand, are computed
using all the data that falls to that node, and so LambdaMART updates only a few
parameters at a time (namely, the split values for the current leaf nodes), but using
all the data (since every xi
lands in some leaf). This means in particular that LambdaMART is able to choose splits and leaf values that may decrease the utility for
some queries, as long as the overall utility increases.

- [x] https://liam.page/2016/07/10/a-not-so-simple-introduction-to-lambdamart/
- [X] https://blog.csdn.net/huagong_adu/article/details/40710305
- [X] https://liam.page/uploads/slides/lambdamart.pdf
- [ ] https://arxiv.org/abs/1811.04415
- [ ] https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
- [ ] https://www.microsoft.com/en-us/research/blog/ranknet-a-ranking-retrospective/
- [ ] https://staff.fnwi.uva.nl/e.kanoulas/wp-content/uploads/Lecture-8-1-LambdaMart-Demystified.pdf


**GBRT+LR** can also used to predict the CTR ratio. On short but incomplete word, it is **GBRT + LR** - **gradient boosting regression tree and logistic regression**.
GBRT is introduced at the *Boosting* section. *LR* is to measure the cost as the same in RankNet.

https://arxiv.org/pdf/1811.12776.pdf
https://www.cse.cuhk.edu.hk/irwin.king/_media/presentations/sigir15bestpaperslides.pdf
- [X] https://www.jianshu.com/p/96173f2c2fb4


### Selective Gradient Boosting

![](https://pic4.zhimg.com/80/v2-2880337351fec6ae22cd93addbe5f453_hd.jpg)

- [ ] http://quickrank.isti.cnr.it/selective-data/selective-SIGIR2018.pdf
- [ ] [基于Selective Gradient Boosting的排序方法 - BeyondTheData的文章 - 知乎](https://zhuanlan.zhihu.com/p/55768029)
- [ ] http://quickrank.isti.cnr.it/
- [ ] http://learningtorank.isti.cnr.it/tutorial-ictir17/
- [ ] http://quickrank.isti.cnr.it/research-papers/
- [ ] https://github.com/Isminoula/DL-to-Rank
- [ ] https://github.com/tensorflow/ranking
- [ ] https://maciejkula.github.io/spotlight/index.html#
- [ ] http://quickrank.isti.cnr.it/research-papers/

### LambdaLoss

LambdaRank is a novel algorithm that incorporates
ranking metrics into its learning procedure. The underlying loss that LambdaRank optimizes for remains unknown until now.
Due to this, there is no principled way to advance the LambdaRank algorithm further. The LambdaLoss framework allows
us to define metric-driven loss functions that have clear connection
to different ranking metrics.
A commonly used pairwise loss function is the logistic loss. LambdaRank is a special configuration with a well-defined loss
in the LambdaLoss framework, and thus provide theoretical justification for it. More importantly, the LambdaLoss framework allows
us to define metric-driven loss functions that have clear connection to different ranking metrics.

A learning-to-rank algorithm is to find a ranking model $\Phi$ that
can predict the relevance scores ${s}$ for all documents in a query:
$$\Phi(x): X\to S.$$
We formulate the loss function in a probabilistic manner. Similar to
previous work, we assume that scores of documents ${s}$ determine a distribution over all possible ranked lists or permutations.
Let ${\Pi}$ denote a ranked list and we use ${P(\pi |s) : \pi \in \Pi}$ to denote
the distribution. In our framework, we treat the ranked list ${\pi}$ as
a hidden variable and define the loss based on the likelihood of
observing relevance ${y}$ given ${s}$ (or equivalently ${\Phi}$ and ${x}$) using a
mixture model over ${\Pi}$:
$$
P(y|s)=\sum_{\pi\in\Pi}P(y|s,\pi)P(\pi|s).
$$

We define the as the negative log likelihood based on the maximum likelihood principle:
$$
l(y,s)=-\ln(P(y|s))=-\ln(\sum_{\pi\in\Pi}P(y|s,\pi)P(\pi|s)).
$$

And such a loss can be minimized by the well-known `Expectation-Maximization (EM)` algorithm.

- [ ] https://ai.google/research/pubs/pub47258
- [ ] http://bendersky.github.io/pubs.html
- [ ] http://marc.najork.org/

### Bayesian Personalized Ranking

**Bayesian Personalized Ranking(BPR)** uses implicit information to construct recommender system. The training dat set is $D=\{(u,i,j)\}$
where the data point $(u,i,j)$ represents that the user ${u}$ likes the item ${i}$  more than the item ${j}$, or in another form $U_i\triangleleft U_j$.

The Bayesian formulation of finding the correct personalized ranking for all items $i \in I$ is to maximize
the following posterior probability where $\Theta$ represents
the parameter vector of an arbitrary model class (e.g. matrix factorization).

$$
\prod_{(u,i,j)} P(\Theta|U_i\triangleleft U_j)\approx \prod_{(u,i,j)} P(U_i\triangleleft U_j|\Theta) P(\Theta)\\
P(\Theta|\triangleleft_{U})\approx P(\triangleleft_{U}|\Theta) P(\Theta).
$$

where $P(U_i\triangleleft U_j|\Theta)$ is the likelihood function given the model $\Theta$:

$$P(U_i\triangleleft U_j|\Theta)=\sigma(f_U(i,j|\Theta)), \sigma(x)=\frac{1}{1+e^{-x}}.$$

Here $f_U(i,j|\Theta)$ is an arbitrary real-valued function of
the model parameter vector $\Theta$ which captures the special relationship between user ${U}$, item ${i}$ and item ${j}$.

The aim is to estimate the real function $f_U(i,j|\Theta)$ for every user ${U}$.
And we make some assumption of the data set:

* All users are presumed to act independently
of each other.
* The ordering of each
pair of items $(i, j)$ for a specific user is presumed to be independent
of the ordering of every other pair.

We introduce a general prior density $P(\Theta)$ which is a normal distribution
with zero mean and variance-covariance matrix $\Sigma_{\Theta}$:
$$\Theta\sim N(0, \Sigma_{\Theta}).$$

To reduce the number of unknown
hyperparameters we set
$$\Sigma_{\Theta}=\lambda_{\Theta}I.$$
Now we can formulate the **maximum posterior estimator** to derive our
generic optimization criterion for personalized ranking:
$$
\ln(P(\Theta|\triangleleft_{U}))\\
=\ln(P(\triangleleft_{U}|\Theta) P(\Theta))\\
=\ln (\prod_{(u,i,j)} P(U_i\triangleleft U_j|\Theta) P(\Theta))\\
=\sum_{(u,i,j)}\ln(P(U_i\triangleleft U_j|\Theta))+\ln(P(\Theta)) \\
=\sum_{(u,i,j)}\ln(\sigma(f_U(i,j|\Theta)))-\lambda_{\Theta}\|\Theta\|^2
$$

where $\lambda_{\Theta}$ are model specific regularization parameters.

And we can use stochastic gradient descent to find the parameters $\Theta$.
***
- [ ] http://lipixun.me/2018/01/22/bpr
- [ ] https://liuzhiqiangruc.iteye.com/blog/2073526
- [ ] https://blog.csdn.net/cht5600/article/details/54381011
- [ ] https://blog.csdn.net/qq_20599123/article/details/51315697
- [ ] https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf


**Deep Online Ranking System**

DORS is designed and implemented in a three-level novel architecture, which includes (1) candidate retrieval; (2) learning-to-rank deep neural network (DNN) ranking; and (3) online
re-ranking via multi-arm bandits (MAB).

* [A Practical Deep Online Ranking System in
E-commerce Recommendation](http://www.ecmlpkdd2018.org/wp-content/uploads/2018/09/723.pdf)
* http://www.ecmlpkdd2018.org/

- [ ] https://tech.meituan.com/2019/01/17/dianping-search-deeplearning.html
- [ ] https://academic.microsoft.com/#/detail/2149166361

***
* https://www.wikiwand.com/en/Learning_to_rank
* https://www.wikiwand.com/en/Arrow%27s_impossibility_theorem
* https://plato.stanford.edu/entries/arrows-theorem/
* https://www.math.ucla.edu/~tao/arrow.pdf
* https://www.wikiwand.com/en/Gibbard%E2%80%93Satterthwaite_theorem
* https://arxiv.org/abs/1812.00073
* https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf
