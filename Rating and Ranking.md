# Rating and Ranking

![https://zhuanlan.zhihu.com/p/25443972](https://pic1.zhimg.com/80/v2-ec0751e41981077e932ae0ce2cf6fe48_hd.jpg)

+ [Elasticsearch Learning to Rank: the documentation](https://elasticsearch-learning-to-rank.readthedocs.io/en/latest/core-concepts.html)
+ [Search and information retrieval@Microsoft](https://www.microsoft.com/en-us/research/research-area/search-information-retrieval/)
+ [Information Retrieval and the Web @Google](https://ai.google/research/pubs/?area=InformationRetrievalandtheWeb)
+ [Yandex Research](https://research.yandex.com/)
+ [CIKM 2018 Papers Notes](https://www.hongliangjie.com/2019/01/20/cikm-2018-papers-notes/)
+ <https://www.cse.iitb.ac.in/~soumen/>
+ https://fate-events.github.io/facts-ir/
+ [DocRank: Computer Science Capstone Project with the College of Charleston](https://fullmetalhealth.com/docrank-computer-science-capstone-project-college-charleston/index.html)
+ [Ratings and rankings: voodoo or science?](http://www.andreasaltelli.eu/file/repository/rssa_1059.pdf)
+ [The Science of Ranking Items: from webpages to teams to movies](http://langvillea.people.cofc.edu/RankAggJapan.pdf)
+ [Ranking with Optimization Techniques](http://langvillea.people.cofc.edu/RankbyOptim.pdf)
+ [Rank and Rating Aggregation](http://langvillea.people.cofc.edu/RankAgg.pdf)
+ [Sensitivity and Stability of Ranking Vectors](https://epubs.siam.org/doi/10.1137/090772745)
+ [The Rankability of Data](https://epubs.siam.org/doi/pdf/10.1137/18M1183595)
+ [A Rating-Ranking Method for Crowdsourced Top-k Computation](http://dbgroup.cs.tsinghua.edu.cn/ligl/papers/sigmod18-crowdtopk.pdf)
+ [the (data) science of sports](http://thespread.us/category/ranking.html)

The rating algorithms help to match the players in video games or compare the players in sports.
Ratings is a numerical score to describe the level  of the players' skill based on the results of many competition.
The basic idea is the back-feed from the results to improve the experience. After each game, this data is updated for the participants in the game.


The ranking problem is from information retrieval. Given a query as we type in a search engine, the ranking algorithms are to sort the items
which may answer this query as the PageRank does for web searching. And `search engine optimization (SOE)` can be regarded as the reverse engineer of the ranking algorithms of search engine.

<img title="science of rating and ranking" src="https://images-na.ssl-images-amazon.com/images/I/51bML705X7L._SX353_BO1,204,203,200_.jpg" width="20%" />

They share some techniques although their purpose is different such as the logistic regression.

|[3 R's for the internet age](https://phys.org/news/2016-12-recommendingthree-internet-age.html)|Rating, ranking and recommending|
|---|--- |
| Rating |A rating of items assigns a numerical score to each item. A rating list, when sorted, creates a ranking list.|
| Ranking |A ranking of items is a rank-ordered list of the items. Thus, a ranking vector is a permutation of the integers 1 through n.|
| Recommendation | Recommendation Algorithms predict how each specific user would rate different items she has not yet bought by looking at the past history of her own ratings and comparing them with those of similar users.|

In some sense, rating is to evaluate in a quantity approach, i.e. how much the item is popular; ranking is to evaluate in a quality approach. i.e., whether it is popular or preferred; recommendation is to rate or rank the items if the information is undirected or implicit.

- [ ] [Who’s #1? The Science
of Rating and Ranking](http://www.ams.org/notices/201301/rnoti-p81.pdf)
- [ ] [Who's Number 1? Hodge Theory Will Tell Us](http://www.ams.org/publicoutreach/feature-column/fc-2012-12)
- [ ] [WhoScored Ratings Explained](https://www.whoscored.com/Explanations)
- [ ] [Ranking Algorithm Definition](http://www.meteorsite.com/ranking-algorithm)
- [ ] [EdgeRank](http://edgerank.net/)
- [ ] [SofaScore Statistical Ratings](https://www.sofascore.com/news/sofascore-player-ratings/)
- [ ] [Everything You Need to Know About the TripAdvisor Popularity Ranking](https://www.tripadvisor.com/TripAdvisorInsights/w765)
- [ ] [Deconstructing the App Store Rankings Formula with a Little Mad Science](https://moz.com/blog/app-store-rankings-formula-deconstructed-in-5-mad-science-experiments)

* [Introducing Steam Database's new rating algorithm](https://steamdb.info/blog/steamdb-rating/)
* https://www.cs.cornell.edu/jeh/book2016June9.pdf

## Rating


| [The Four Commandments of a Perfect Rating Algorithm](http://www.atomicfootball.com/af-algorithm.html)|
|----|
|A Perfect Rating Algorithm is self-contained. It should have no "knobs" or "tuning parameters." Knobs mean an algorithm is incomplete.|
|A Perfect Rating Algorithm has a solid statistical foundation. It follows accepted practice.|
|A Perfect Rating Algorithm is able to inherently estimate its own accuracy. A good statistical foundation is normally conducive to this.|
|A Perfect Rating Algorithm is capable of producing either measurable quantities or quantities from which measurables can be derived. For example, the probability that one team will win over another.|

- [ ] [Rating Algorithm for Evaluation of Web Pages: W3C Working Draft](https://www.w3.org/WAI/ER/IG/rating/)
- [ ] [PvP Matchmaking Algorithm](https://wiki.guildwars2.com/wiki/PvP_Matchmaking_Algorithm)
- [ ] [Finding the perfect match by Justin O'Dell on November 20, 2014](https://www.guildwars2.com/en/news/finding-the-perfect-match/)
- [ ] [Ranking of sports teams](http://www.phys.utk.edu/sorensen/ranking/)
- [ ] [PlayeRank: data-driven performance evaluation and player ranking in soccer via a machine learning approach](https://arxiv.org/pdf/1802.04987.pdf)
- [ ] [Massey Ratings Description](https://www.masseyratings.com/theory/massey.htm)
- [ ] [The USCF Rating System, Prof. Mark E. Glickman , Boston University - Thomas Doan
Estima](http://math.bu.edu/people/mg/ratings/rs/)
- [ ] [College Football Ranking Composite](http://www.atomicfootball.com/af-algorithm.html)
- [ ] [A Bayesian Mean-Value Approach with a Self-Consistently Determined Prior Distribution for the Ranking of College Football Teams](https://arxiv.org/abs/physics/0607064)
- [ ] [An overview of some methods for ranking sports teams, Soren P. Sorensen](http://www.phys.utk.edu/sorensen/ranking/Documentation/Sorensen_documentation_v1.pdf)

### Elo Rating

Elo rating is popular at many games such as Go game, soccer and so on.
It supposes that the performance are random and the winning rate is determined by the differences of two players.
If the differences of the scores is beyond some threshold such as ${200}$, it is supposed that the winner probability is ${3/4}$.
And  it is natural if difference of scores is ${0}$, the winning rate is $1/2$.

We assume that

$$
P(d) = \frac{1}{1+e^{-\frac{d}{\theta}}}
$$

where the parameter $\theta$ is related with the threshold.

For example, the expected performance of player A is $E_{A} = \frac{1}{1 + 10^{-\frac{R_A - R_B}{400}}}$ and $E_{B} = \frac{1}{1 + 10^{-\frac{R_B - R_A}{400}}} = 1 - E_{A}$.
Supposing Player A was expected to score $E_{A}$ points but actually scored $S_{A}$ points. And the update rule is
$${R}_{A}^{New} = R_{A} + K(S_A - E_A)$$
where $K$ is a constant.

* https://www.wikiwand.com/en/Elo_rating_system
* [Bradley-Terry model @wikiwand](https://www.wikiwand.com/en/Bradley%E2%80%93Terry_model)
* [Bradley-Terry model and extensions](http://www.calvin.edu/~rpruim/fast/R/BradleyTerry/html/BTm.html)
* [Fitting a Bradley-Terry Model](https://homepage.divms.uiowa.edu/~luke/xls/glim/glim/node8.html)
* [Ties in Paired-Comparison Experiments: A Generalization of the Bradley-Terry Model](https://www.jstor.org/stable/2282923)
* [The Math behind ELO](https://blog.mackie.io/the-elo-algorithm)

### Glicko

The problem with the Elo system that the Glicko system addresses has to do with the
reliability of a player’s rating.

Glickman's principal contribution to measurement is "ratings reliability", called RD, for ratings deviation.
The RD measures the accuracy of a player's rating, with one RD being equal to one standard deviation.
If the player is unrated, the rating is usually set to 1500 and the RD to 350.

> 1. Determine RD
>>   The new Ratings Deviation (RD) is found using the old Ratings Deviation $RD_0$:
>> $$
>>   RD=\min\{\sqrt{RD_0^2+c_2t}, 350\}
>> $$
>>>   where ${t}$ is the amount of time (rating periods) since the last competition and '350' is assumed to be the RD of an unrated player. And $c=\sqrt{(350^2-50^2)/100}\simeq 34.6$.
>
> 2. Determine New Rating
>>   The new ratings, after a series of m games, are determined by the following equation:
>> $$
>>   r=r_0+\frac{q}{RD^{-2}+d^{-2}}\sum_{i=1}^{m}g(RD_i)(s_i - E(s|r,r_i,RD_i))
>> $$
>>>   
>>> where
>>> * $g(RD_i)=\{1+\frac{3q^2(RD_i)^2}{\pi^2}\}^{-1/2}$, $E(s|r,r_i,RD_i))=\{1 + 10^{(\frac{g(RD_i)(r-r_i)}{-400})}\}$,
>>> * $q=\frac{\ln(10)}{400}\approx 0.00575646273$,
>>> * $d^{-2} = q^2\sum_{i=1}^{m}[g(RD_i)^2]E(s|r,r_i,RD_i)[1-E(s|r,r_i,RD_i)]$,
>>> * $r_i$ represents the ratings of the individual opponents.
>>> * $s_i$ represents the outcome of the individual games. A win is ${1}$, a draw is $\frac {1}{2}$, and a loss is $0$.
>
> 3. Determine New Ratings Deviation
>>
>>   $$RD^{\prime}=\sqrt{(RD^{-2}+d^{-2})^{-1}} .$$

* [Mark Glickman's Research](http://www.glicko.net/research.html)
* [Glicko Ratings](http://www.glicko.net/glicko.html)
* https://www.wikiwand.com/en/Glicko_rating_system
* [Java implementation of the Glicko-2 rating algorithm](https://github.com/goochjs/glicko2)


### TrueSkill

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
- [X] [算法_TrueSkill_Python](https://www.jianshu.com/p/c1fbba3af787)
- [ ] [Chapter 3: Meeting Your Match](http://www.mbmlbook.com/TrueSkill.html)
- [ ] [TrueSkill原理简介](https://zhuanlan.zhihu.com/p/48737998)
- [ ] https://www.wikiwand.com/en/Location%E2%80%93scale_family
- [ ] [TrueSkill™ Ranking System](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/)
- [ ] [TrueSkill: the video game rating system](https://trueskill.org/)

### Whole-History Rating

Incremental Rating Systems or dynamical rating systems such as TrueSkill  do not make optimal use of data.
The principle of Bayesian Inference consists in computing a probability distribution
over player ratings ($r$) from the observation of game results ($G$) by inverting the model thanks to Bayes formula:

$$
P(r|G)=\frac{P(G|r)P(r)}{P(G)}
$$

where $P(r)$ is a prior distribution over $r$, and $P(G)$ is a normalizing constant. $P(r|G)$ is called the posterior distribution of $r$:
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


* [X] https://www.remi-coulom.fr/WHR/
* [X] [Whole-History Rating: A Bayesian Rating System for Players of Time-Varying Strength
](https://www.remi-coulom.fr/WHR/WHR.pdf)
* [X] [Scientific Ranking Methods: is risk-free betting possible?](https://www.inria.fr/en/news/news-from-inria/scientific-ranking-methods)

**How to Build a Popularity Algorithm You can be Proud of**

It is  a way to score the posts, articles or something else based on the users' inputs. It is a simple voting system to determine the popularity. It is interesting to select the most popular articles in social media to the subscribers. If all the people in the community likes the same article or item in the same conditions, it is not necessary to build a popularity algorithm. However, our tastes are diverse and dynamical. Not all people like the best apple.

It is an application of collective decision-making theory in some sense.
For example, everyone can upvote or downvote  the answers in stackoverflow in the same question, what is the most popular answer to the question? The following links may help a little.

- [ ] [基于用户投票的排名算法（一）：Delicious和Hacker News](http://www.ruanyifeng.com/blog/2012/02/ranking_algorithm_hacker_news.html)
- [ ] [基于用户投票的排名算法（五）：威尔逊区间](http://www.ruanyifeng.com/blog/2012/03/ranking_algorithm_wilson_score_interval.html)
- [ ] http://www.evanmiller.org/how-not-to-sort-by-average-rating.html
- [ ] http://www.evanmiller.org/rank-hotness-with-newtons-law-of-cooling.html
- [ ] http://www.evanmiller.org/ranking-items-with-star-ratings.html
- [ ] http://www.evanmiller.org/ranking-news-items-with-upvotes.html
- [ ] http://www.evanmiller.org/bayesian-average-ratings.html
- [ ] [Rank products: a simple, yet powerful, new method to detect differentially regulated genes in replicated microarray experiments](https://www.sciencedirect.com/science/article/pii/S0014579304009354)

[This is the “paradox of voting”. Discovered by the Marquis de Condorcet (1785), it shows that possibilities for choosing rationally can be lost when individual preferences are aggregated into social preferences.](https://plato.stanford.edu/entries/arrows-theorem/)

There are some links on the  collective decision-making theory:

* https://www.wikiwand.com/Arrow%27s_impossibility_theorem
* https://plato.stanford.edu/entries/arrows-theorem/
* [Arrow's THeorem by Terence Tao](https://www.math.ucla.edu/~tao/arrow.pdf)
* [Gibbard–Satterthwaite theorem @ wikiwand](https://www.wikiwand.com/en/Gibbard%E2%80%93Satterthwaite_theorem)
* [Do the Math: Why No Ranking System Is No. 1](https://www.scientificamerican.com/article/why-ranking-systems-are-flawed/)

## Ranking

* [Learning to Rank explained](https://everything.explained.today/Learning_to_rank/)
* http://www.cs.cmu.edu/~kdelaros/
* [Ranking in information retrieval](https://www.wikiwand.com/en/Ranking_(information_retrieval))
* [Learning to Rank](https://jimmy-walker.gitbooks.io/rank/L2R.html)
* [Hardened Fork of Ranklib learning to rank library](https://github.com/o19s/RankyMcRankFace)
* [OpenSource Connections](https://github.com/o19s)

Combining feedback from multiple users to rank a collection of items is an important task.
The ranker, a central component in every `search engine`, is responsible for the matching between processed queries and indexed documents in information retrieval.
The goal of a ranking system is to find the best possible ordering of a set of items for a user, within a specific context, in real-time in recommender system.

In general, we call all those methods that use machine learning technologies to solve the problem of ranking **"learning-to-rank"** methods or **LTR** or **L2R**.
We are designed to compare some indexed document with the query.
The algorithms above are based on feedback to tune the rank or scores of players or documents. The drawback of these methods is that they do not take the features of the players into consideration.
We may use machine learn to predict the scores of players and test it in real data set such as **RankNet, LambdaRank, and LambdaMART**.

And it can apply to information retrieval and recommender system.

* https://github.com/cgravier/RankLib
* http://fastml.com/evaluating-recommender-systems/
* http://quickrank.isti.cnr.it/research-papers/
* [Explicit feedback movie recommendations@spotlight](https://github.com/maciejkula/spotlight/tree/master/examples/movielens_explicit)
* [RankNet: A ranking retrospective](https://www.microsoft.com/en-us/research/blog/ranknet-a-ranking-retrospective/)
* [BPR [Bayesian Personalized Ranking] 算法详解及应用实践](https://blog.csdn.net/cht5600/article/details/54381011)
* [Unbiased Learning-to-Rank with Biased Feedback@IJCAI](https://www.ijcai.org/proceedings/2018/0738.pdf)
* [Research and Software on Learning To Rank @ HPC Lab, ISTI-CNR, Pisa, Italy](http://learningtorank.isti.cnr.it/)
* [CIKM 2018 Papers Notes by Hong Liangjie](https://www.hongliangjie.com/2019/01/20/cikm-2018-papers-notes/)
* [Unbiased Learning-to-Rank with Biased Feedback](http://www.cs.cornell.edu/people/tj/publications/joachims_etal_17a.pdf)
* [Boosted Ranking Models: A Unifying Framework for Ranking Predictions](http://www.cs.cmu.edu/~kdelaros/)
* [Tasks Track 2015](http://www.cs.ucl.ac.uk/tasks-track-2015/)
* [TREC 2014 Session Track](http://ir.cis.udel.edu/sessions/)
* [Introduction to Information Retrievel](https://nlp.stanford.edu/IR-book/)
* [Images do npt lie](https://arxiv.org/abs/1511.06746)
![meituan.net](https://p1.meituan.net/travelcube/58920553566822f1fe059f95eba71d95131646.png)

*Learning to Rank* can be classified into [pointwise, pairwise and listless approaches](http://www.l3s.de/~anand/tir15/lectures/ws15-tir-l2r.pdf).
In brief, the pointwise approach is to find a function which can predict the relevance of a query and a given document;
the pairwise approach is to predict the relative  preference of two document when given a query;
the listwise approach is to predict the ranks of documents in a list when given a query.

+ [Ranking Models (2018/2) by Rodrygo Santos](https://homepages.dcc.ufmg.br/~rodrygo/rm-2018-2/)
+ [Temporal Information Retrieval: A course on Information Retrieval with a temporal twist, Lecturer: Dr. Avishek Anand](http://www.l3s.de/~anand/tir15/)
+ [Learn to rank: An overview](https://www.cl.cam.ac.uk/teaching/1516/R222/l2r-overview.pdf)
+ [Yahoo! Learning to Rank Challenge Overview](https://course.ccs.neu.edu/cs6200sp15/extra/07_du/chapelle11a.pdf)
+ [Learning to Rank: From Pairwise Approach to Listwise Approach](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf)
+ [Generalization Analysis of Listwise Learning-to-Rank Algorithms](https://icml.cc/Conferences/2009/papers/101.pdf)
+ [Ranking and Filtering by Weinan Zhang](http://wnzhang.net/teaching/cs420/slides/7-ranking-filtering.pdf)
+ [On the Consistency of Ranking Algorithms](https://www.shivani-agarwal.net/Teaching/E0371/Papers/icml10-ranking-consistency.pdf)
+ [Catarina Moreira's master thesis](http://web.ist.utl.pt/~catarina.p.moreira/coursera.html)
+ [Ranking in Information Retrieval](https://www.cse.iitb.ac.in/internal/techreports/reports/TR-CSE-2010-31.pdf)
- [ ] [Learning Groupwise Scoring Functions Using Deep Neural Networks](https://arxiv.org/abs/1811.04415)
- [ ] https://github.com/tensorflow/ranking

**Training Setup**

A training set for ranking  is denoted as $R=\{(\mathrm{X_i}, y_i)\mid i=1, \dots, m.\}$ where $y_i$ is the ranking of $x_i$, that is, $y_i < y_j$ if $x_i ≻ x_j$, i.e., $x_i$ is preferred to $x_j$ or in the reverse order. In other word, the label $y_i$ is ordinal. By the way, the labels are categorical or  nominal  in most classification tasks.

The ranking function outputs a score for each data object, from which a global
ordering of data is constructed. That is, the target function $F(x_i)$ outputs a score
such that $F(x_i) > F(x_j)$ for any $x_i ≻ x_j$.

- [WHAT IS THE DIFFERENCE BETWEEN CATEGORICAL, ORDINAL AND INTERVAL VARIABLES?](https://stats.idre.ucla.edu/other/mult-pkg/whatstat/what-is-the-difference-between-categorical-ordinal-and-interval-variables/)

**Ranking Metrics**

The metrics or evaluation is different in regression and classification. And many loss function is introduced in maximum likelihood estimation.

![Recall](https://nlp.stanford.edu/IR-book/html/htmledition/img532.png)

`Precision` measures the exactness of the retrieval process. If the actual set of relevant documents is denoted by _I_ and the retrieved set of documents is denoted by _O_, then the precision is given by:
$$Precision= \frac{|O\cap I|}{|O|}.$$

`Recall` is a measure of completeness of the IR process. If the actual set of relevant documents is denoted by _I_ and the retrieved set of documents is denoted by _O_, then the recall is given by:
$$Recall=\frac{|O\cap I|}{|O|}.$$

`F1 Score` tries to combine the precision and recall measure. It is the harmonic mean of the two. If _P_ is the precision and _R_ is the recall then the F-Score is given by:
$$F1 = 2\frac{P\times R}{P+R}.$$

Precision and recall are single-value metrics based on the whole list of documents returned by the system. For systems that return a ranked sequence of documents, it is desirable to also consider the order in which the returned documents are presented. By computing a precision and recall at every position in the ranked sequence of documents, one can plot a precision-recall curve, plotting precision $p(r)$ as a function of recall $r$.
`Average precision` computes the average value of $p(r)$ over the interval from $r=0$ to $r=1$:
$$
AveP=\int_{0}^{1} p(r)\mathrm{d} r.
$$

That is the area under the precision-recall curve. This integral is in practice replaced with a finite sum over every position in the ranked sequence of documents:
$$
AveP= \sum_{k=1}^{n} p(k)\Delta r(k).
$$
where $k$ is the rank in the sequence of retrieved documents, $n$ is the number of retrieved documents, $P(k)$ is the precision at cut-off k in the list, and $\Delta r(k)$ is the change in recall from items $k-1$ to $k$.
****
`Cumulative Gain (CG)` is the predecessor of DCG and does not include the position of a result in the consideration of the usefulness of a result set. In this way, it is the sum of the graded relevance values of all results in a search result list. The CG at a particular rank position _p_ is defined as:
$${CG}_p=\sum_{i=1}^{p}{rel}_i,$$
Where ${rel}_{i}$ is the graded relevance of the result at position _i_.

The premise of `Discounted Cumulative Gain(DCG)` is that highly relevant documents appearing lower in a search result list should be penalized as the graded relevance value is reduced logarithmically proportional to the position of the result.

The traditional formula of _DCG_ accumulated at a particular rank position p is defined as:
$$
{DCG}_p= \sum_{i=1}^{p}\frac{{rel}_i}{{\log}_{2}(i+1)}
$$
For every pair of substantially different ranking functions, it has shown that the NDCG can decide which one is better in a consistent manner.

An alternative formulation of _DCG_ places stronger emphasis on retrieving relevant documents:
$$
{DCG}_p= \sum_{i=1}^{p}\frac{2^{{rel}_i}-1}{{\log}_{2}(i+1)}
$$

For a query, the normalized discounted cumulative gain, or `nDCG`, is computed as:
$$
{nDCG}_p=\frac{{DCG}_p}{{IDCG}_p},
$$
where `IDCG` is ideal discounted cumulative gain,
$$
{IDCG}_p = \sum_{i=1}^{|REL_p|}\frac{2^{{rel}_i}-1}{{\log}_{2}(i+1)}
$$
and ${\displaystyle REL_{p}}$ represents the list of relevant documents (ordered by their relevance) in the corpus up to position _p_.

- [信息检索中的评价指标MAP和NDCG](http://lixinzhang.github.io/xin-xi-jian-suo-zhong-de-ping-jie-zhi-biao-maphe-ndcg.html)
- [Discounted cumulative gain@wikiwand](https://www.wikiwand.com/en/Discounted_cumulative_gain)
- [Evaluating recommender systems](http://fastml.com/evaluating-recommender-systems/)
- [A Short Survey on Search Evaluation](https://staff.fnwi.uva.nl/e.kanoulas/a-short-survey-on-search-evaluation/)
- [Metrics for evaluating ranking algorithms](https://stats.stackexchange.com/questions/159657/metrics-for-evaluating-ranking-algorithms)
- [Metric Learning to Rank](https://bmcfee.github.io/papers/mlr.pdf)
- [Evaluation of ranked retrieval results](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)
- [Evaluation of Ranking @ Stanford CS276](https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf)
- [Evaluation measure in information retrieval @ wikiwand](https://www.wikiwand.com/en/Evaluation_measures_(information_retrieval))
- [mAP（mean average precision）平均精度均值](https://www.jianshu.com/p/82be426f776e)
- [Online User Engagement: Metrics and Optimization.](https://onlineuserengagement.github.io/)
- [Rank and Relevance in Novelty and Diversity Metrics for Recommender Systems](http://ir.ii.uam.es/predict/pubs/recsys11-vargas.pdf)
- [Implementing Triplet Losses for Implicit Feedback Recommender Systems with R and Keras](https://nanx.me/blog/post/triplet-loss-r-keras/)

### RankSVM

> The basic idea of SVMrank is to attempt to minimize the number of misclassified document pairs. This is achieved by modifying the default support vector machine optimization problem, which considers a set of documents, by constraining the optimization problem to perform the minimization of each pair of documents.


Using the techniques of SVM, a global ranking function _F_ can be learned from an ordering R. Assume _F_ is a linear ranking function such that
$$\forall \{(\mathrm{X_i, X_j}): y_i < y_j\}, F(X_i)>F(X_j)\iff w\cdot X_i > w\cdot X_j.$$
The solution can be approximated using SVM techniques by introducing (non-negative) slack variables ${\xi}_{i j}$ and minimizing the upper bound $\sum {\xi}_{i j}$ as follows:
$$
\text{minimize}\qquad L(w, \xi)= C \sum {\xi}_{i j} + {\|w\|}_2^2 \\
\text{subject to } \quad \forall \{(\mathrm{X_i, X_j})\}: w\cdot X_i > w\cdot X_j + 1 -{\xi}_{ij},
\\ \forall {\xi}_{ij}\geq 0.
$$
Then it can trained in the same way as a classifer.
* https://x-algo.cn/index.php/2016/08/09/ranksvm/
* https://www.cnblogs.com/bentuwuying/p/6683832.html
* [Support Vector Machine for Ranking Author: Thorsten Joachims](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html)
* [Ranking SVM for Learning from Partial-Information Feedback](http://www.cs.cornell.edu/people/tj/svm_light/svm_proprank.html)
* [SVM-based Modelling with Pairwise Transformation for Learning to Re-Rank](http://alt.qcri.org/ecml2016/unocanda_cameraready.pdf)


### RankNet

> RankNet is a feedforward neural network model. Before it can be used its parameters must be learned using a large amount of labeled data, called the training set. The training set consists of a large number of query/document pairs, where for each pair, a number assessing the quality of the relevance of the document to that query is assigned by human experts. Although the labeling of the data is a slow and human-intensive task, training the net, given the labeled data, is fully automatic and quite fast. The system used by Microsoft in 2004 for training the ranker was called The Flying Dutchman. from  [RankNet: A ranking retrospective](https://www.microsoft.com/en-us/research/blog/ranknet-a-ranking-retrospective/).

RankNet takes the ranking  as **regression** task.

![RankNet](http://web.ist.utl.pt/~catarina.p.moreira/images/ranknet.png)

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
{\lambda}_{i,j}=  -\frac{\sigma}{1+\exp(\sigma(s_i-s_j))}|\Delta Z|,
$$
where $\Delta Z$ is the size of the change in some **Information Retrieval Measures** ${Z}$.

And $\lambda_{i}$ with respect to ${i}$-th item is defined as
$$\lambda_i = \sum_{i\in(i,j)}\lambda_{(i,j)}-\sum_{j\in(j,i)}\lambda_{(i,j)}$$

<img src=https://p0.meituan.net/travelcube/d6142123b31212f4854fd4e53da5831e14664.png width=50% />

***

- http://blog.camlcity.org/blog/lambdarank.html
- [LambdaFM: Learning Optimal Ranking with Factorization Machines Using Lambda Surrogates](http://wnzhang.net/papers/lambdafm.pdf)
- [Practical Lessons from Predicting Clicks on Ads at Facebook](http://quinonero.net/Publications/predicting-clicks-facebook.pdf)

### LambdaMART

**LambdaMART** is the boosted tree version of LambdaRank, which is based on RankNet. It takes the ranking problem as classification problem.

MART stands for [Multiple Additive Regression Tree](http://statweb.stanford.edu/~jhf/MART.html).
In LambdaRank, we compute the gradient. And we can use this gradient to make up the GBRT.
> LambdaMART had an added advantage: the training of tree ensemble models can be very significantly sped up over the neural net equivalent (this work, led by O. Dekel, is not yet published). This allows us to train with much larger data sets, which again gives improved ranking accuracy. From [RankNet: A ranking retrospective](https://www.microsoft.com/en-us/research/blog/ranknet-a-ranking-retrospective/).

***

<img title="LambdaMART" src = "https://liam.page/uploads/images/LTR/LambdaMART.png" width = 80% />

***

To implement LambdaMART we just use MART, specifying appropriate gradients
and the Newton step.
The key point is the gradient of the ${\lambda}_i$:
$$ w_i = \frac{\partial y_i}{\partial F_{k-1}(\vec{x}_i)} $$
where $\lambda_i = y_i$ is defined in **LambdaRank**.
LambdaRank updates all the weights after each query is examined. The
decisions (splits at the nodes) in LambdaMART, on the other hand, are computed using all the data that falls to that node, and so LambdaMART updates only a few
parameters at a time (namely, the split values for the current leaf nodes), but using all the data (since every $x_i$ lands in some leaf).
This means in particular that LambdaMART is able to choose splits and leaf values that may decrease the utility for some queries, as long as the overall utility increases.

- [x] [LambdaMART 不太简短之介绍](https://liam.page/2016/07/10/a-not-so-simple-introduction-to-lambdamart/)
- [X] [Learning To Rank之LambdaMART的前世今生](https://blog.csdn.net/huagong_adu/article/details/40710305)s
- [X] [LambdaMart Slides](https://liam.page/uploads/slides/lambdamart.pdf)
- [ ] [From RankNet to LambdaRank to LambdaMART: An Overview](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/)
- [ ] [Ranknet a ranking retrospective](https://www.microsoft.com/en-us/research/blog/ranknet-a-ranking-retrospective/)
- [ ] [LambdaMart Demystified](https://staff.fnwi.uva.nl/e.kanoulas/wp-content/uploads/Lecture-8-1-LambdaMart-Demystified.pdf)
- [ ] [Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm](https://arxiv.org/pdf/1809.05818.pdf)

**GBRT+LR** can also used to predict the CTR ratio. On short but incomplete word, it is **GBRT + LR** - **gradient boosting regression tree and logistic regression**.
GBRT is introduced at the *Boosting* section. *LR* is to measure the cost as the same in RankNet.

- [ ] [Learning From Weights: A Cost-Sensitive Approach For Ad Retrieval](https://arxiv.org/abs/1811.12776)
- [X] https://www.jianshu.com/p/96173f2c2fb4
- [ ] [Boosted Ranking Models: A Unifying Framework for Ranking Predictions](http://www.cs.cmu.edu/~kdelaros/)

### LambdaLoss

LambdaRank is a novel algorithm that incorporates ranking metrics into its learning procedure.
The underlying loss that LambdaRank optimizes for remains unknown until now.
Due to this, there is no principled way to advance the LambdaRank algorithm further.
The LambdaLoss framework allows us to define metric-driven loss functions that have clear connection to different ranking metrics.
A commonly used pairwise loss function is the logistic loss. LambdaRank is a special configuration with a well-defined loss
in the LambdaLoss framework, and thus provide theoretical justification for it.
More importantly, the LambdaLoss framework allows us to define metric-driven loss functions that have clear connection to different ranking metrics.

A learning-to-rank algorithm is to find a ranking model $\Phi$ that
can predict the relevance scores ${s}$ for all documents in a query:
$$\Phi(x): X\to S.$$
We formulate the loss function in a probabilistic manner.
Similar to previous work, we assume that scores of documents ${s}$ determine a distribution over all possible ranked lists or permutations.
Let ${\Pi}$ denote a ranked list and we use ${P(\pi |s) : \pi \in \Pi}$ to denote the distribution.
In our framework, we treat the ranked list ${\pi}$ as a hidden variable
and define the loss based on the likelihood of observing relevance ${y}$ given ${s}$ (or equivalently ${\Phi}$ and ${x}$) using a mixture model over ${\Pi}$:
$$
P(y|s)=\sum_{\pi\in\Pi}P(y|s,\pi)P(\pi|s).
$$

We define the as the negative log likelihood based on the maximum likelihood principle:
$$
l(y,s)=-\ln(P(y|s))=-\ln(\sum_{\pi\in\Pi}P(y|s,\pi)P(\pi|s)).
$$

And such a loss can be minimized by the well-known `Expectation-Maximization (EM)` algorithm.

- [ ] [The LambdaLoss Framework for Ranking Metric Optimization](https://ai.google/research/pubs/pub47258)
- [ ] [Michael Bendersky's publication on rankig](http://bendersky.github.io/pubs.html)
- [ ] http://marc.najork.org/

**Essential Loss: Bridge the Gap between Ranking Measures and Loss Functions in Learning to Rank**

We show that the loss functions of these methods are upper bounds of the measure-based ranking errors. As a result, the minimization of these loss functions will lead to the maximization of the ranking measures. The key to obtaining this result is to model ranking as a sequence of classification tasks, and define a so-called essential loss for ranking as the weighted sum of the classification errors of individual tasks in the sequence. We have proved that the essential loss is both an upper bound of the measure-based ranking errors, and a lower bound of the loss functions in the aforementioned methods. Our proof technique also suggests a way to modify existing loss functions to make them tighter bounds of the measure-based ranking errors. Experimental results on benchmark datasets show that the modifications can lead to better ranking performances, demonstrating the correctness of our theoretical analysis.

- [ ] [Essential Loss: Bridge the Gap between Ranking Measures and Loss Functions in Learning to Rank](https://www.microsoft.com/en-us/research/publication/essential-loss-bridge-the-gap-between-ranking-measures-and-loss-functions-in-learning-to-rank/)
- [ ] [RankExplorer: Visualization of Ranking Changes in Large Time Series Data](https://www.microsoft.com/en-us/research/publication/rankexplorer-visualization-ranking-changes-large-time-series-data/)
- [ ] [Revisiting Approximate Metric Optimization in the Age of Deep Neural Networks](https://ai.google/research/pubs/pub48168)
- [ ] [Revisiting Online Personal Search Metrics with the User in Mind](https://ai.google/research/pubs/pub48243)



### AdaRank

In the abstract, the authors wrote:
> Ideally a learning algorithm would train a ranking model that could directly optimize the performance measures with respect to the training data.
> Existing methods, however, are only able to train ranking models by minimizing loss functions loosely related to the performance measures.
> For example, Ranking SVM and RankBoost train ranking models by minimizing classification errors on instance pairs.
> To deal with the problem, we propose a novel learning algorithm within the framework of boosting,
> which can minimize a loss function directly defined on the performance measures.
> Our algorithm, referred to as AdaRank, repeatedly constructs 'weak rankers' on the basis of reweighted training data
> and finally linearly combines the weak rankers for making ranking predictions.
>We prove that the training process of AdaRank is exactly that of enhancing the performance measure used.

- [An Efficient Boosting Algorithm for Combining Preferences](http://jmlr.csail.mit.edu/papers/volume4/freund03a/freund03a.pdf)
- [Concave Learners for Rankboost](http://www.jmlr.org/papers/volume8/melnik07a/melnik07a.pdf)
- [Python implementation of the AdaRank algorithm](https://github.com/rueycheng/AdaRank)
- [AdaRank: a boosting algorithm for information retrieval](https://dl.acm.org/citation.cfm?id=1277809)


### McRank

The ranking problem is cast as (1) multiple classification (“Mc”) (2) multiple ordinal classification, which lead to computationally tractable learning algorithms
for relevance ranking in Web search in `McRank`.

We learn the class probabilities $p_{i,k} = Pr(y_i = k)$, denoted by $\hat{p}_{i,k},\forall k\in [0,1,2,\dots, K-1]$ and define a scoring function for the sample $(\mathrm{X_i}, y_i)$:
$$S_i = {\sum}_{k=0}^{K-1} \hat{p}_{i,k} T(k)$$
where where $T (k)$ is some monotone (increasing) function of the relevance level.
And in this setting as an example  $y_i = 4$ corresponds to a “perfect” relevance and $y_i = 0$ corresponds to a “poor” relevance when $K=5$.

When $T (k) = k$, the scoring function $S_i$ is the `Expected Relevance` of the sample $(\mathrm{X_i}, y_i)$.

![McRank author](https://www.cs.rutgers.edu/files/styles/manual_crop/public/paste_1471815662.png)

A common approach for multiple ordinal classification is to learn the cumulative probabilities $Pr (y_i \leq k)$ instead of the class probabilities $Pr (y_i = k) = p_{i, k}$.

- [McRank: Learning to Rank Using Multiple Classification and Gradient Boosting](http://papers.nips.cc/paper/3270-mcrank-learning-to-rank-using-multiple-classification-and-gradient-boosting.pdf)
- [The news in microsoft 2007](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-classification-and-gradient-boosting/)
- [Ping Li's profile in dblp](https://dblp.org/pers/hd/l/Li_0001:Ping)

### Margin-based Ranking

The algorithm is a modification of RankBoost, analogous to “approximate coordinate ascent boosting.”

- http://rob.schapire.net/
- [Margin-based Ranking and an Equivalence between AdaBoost and RankBoost](http://rob.schapire.net/papers/marginranking.pdf)

### YetiRank and MatrixNet

`PageRank, LambdaRank, MatrixNet` is under the support of commercial firms *Google, Microsoft, Yandex*. The practical ranking algorithms in the search engines are the key to search engine optimization.
Today the word Yandex has become synonymous with Internet search in Russian-speaking countries, just the same as Google in English-speaking countries.

[MatrixNet is a proprietary machine learning algorithm developed by Yandex and used widely throughout the company products. The algorithm is based on **gradient boosting** and was introduced since 2009.](https://www.wikiwand.com/en/MatrixNet)

[Comparative Analysis of Yandex and Google Search Engines by Anna Paananen](https://www.theseus.fi/bitstream/handle/10024/46483/Paananen_Anna.pdf?sequence=1&isAllowed=y) contributes to the comparison of the ranking methods of both of
the search engines, the quality of the results, and the main ranking factors of Yandex and Google.
As summarized in the previous thesis, the key feature of this method is
* its `resistance to overfitting`;
* a multitude of various factors and their combinations;
* allowance to customize a ranking formula for a specific class of search queries.

The difficulty of the analysis of MatrixNet algorithm is that the formula has never been published, unlike Google’s PageRank.
[CatBoost is an algorithm for gradient boosting on decision trees. Developed by Yandex researchers and engineers, it is the successor of the MatrixNet algorithm that is widely used within the company for ranking tasks, forecasting and making recommendations. It is universal and can be applied across a wide range of areas and to a variety of problems.](https://betapage.co/startup/catboost)
+ [Winning The Transfer Learning Track of Yahoo!’s Learning
To Rank Challenge with YetiRank](http://proceedings.mlr.press/v14/gulin11a/gulin11a.pdf)
+ [MatrixNet: New Level of Search Quality](https://yandex.com/company/technologies/matrixnet/)
+ [The Ultimate Guide To Yandex Algorithms](https://salt.agency/blog/the-ultimate-guide-to-yandex-algorithms/)
+ [CERN boosts its search for antimatter with Yandex’s MatrixNet search engine tech](https://www.extremetech.com/extreme/147320-cern-boosts-its-search-for-antimatter-with-yandexs-matrixnet-search-engine-tech)
+ [MatrixNet as a specific Boosted Decision Tree algorithm which is available as a service](https://github.com/yandex/rep/blob/master/rep/estimators/matrixnet.py)

### Selective Gradient Boosting

`Selective Gradient Boosting (SelGB)` is an algorithm addressing the Learning-to-Rank task by focusing on those irrelevant documents
that are most likely to be mis-ranked, thus severely hindering the quality of the learned model.

<img title="Selective Gradient Boosting" src = "https://pic4.zhimg.com/80/v2-2880337351fec6ae22cd93addbe5f453_hd.jpg" width = 80% />

- [ ] [Selective Gradient Boosting for Effective Learning to Rank](http://quickrank.isti.cnr.it/selective-data/selective-SIGIR2018.pdf)
- [ ] [基于Selective Gradient Boosting的排序方法 - BeyondTheData的文章 - 知乎](https://zhuanlan.zhihu.com/p/55768029)
- [ ] http://quickrank.isti.cnr.it/
- [ ] http://quickrank.isti.cnr.it/research-papers/
- [ ] http://learningtorank.isti.cnr.it/tutorial-ictir17/
- [ ] https://maciejkula.github.io/spotlight/index.html#

**QuickScorer and QuickRank**

Given a query-document pair \((q, d_i)\), represented by a feature vector $\mathrm{x}$,
a LtR model based on an additive ensemble of regression trees predicts a relevance score $s(x)$ used for ranking a set of documents.
Typically, a tree ensemble encompasses several binary decision trees, denoted by $T = {T_0, T_1, \dots}$.
Each internal (or branching) node in $T_h$ is associated with a Boolean test over a specific feature $f_{\phi}\in \mathcal{F}$, and a constant threshold $\gamma\in\mathbb{R}$.
Tests are of the form $x[\phi] \leq \gamma$, and, during the visit, the left branch is taken iff the test succeeds.
Each leaf node stores the tree prediction, representing the potential contribution of the tree to the final document score.
The scoring of ${x}$ requires the traversal of all the ensemble’s trees and it is computed as a weighted sum of all the tree predictions.

All the nodes whose Boolean conditions evaluate to _False_ are called false nodes, and true nodes otherwise.
The scoring of a document represented by a feature vector x requires the traversing of all the trees in the ensemble, starting at their root nodes.
If a visited node in N is a false one, then the right branch is taken, and the left branch otherwise.
The visit continues recursively until a leaf node is reached, where the value of the prediction is returned.

The building block of this approach is an alternative method for tree traversal based on bit-vector computations.

- [ ] [QuickScorer: a fast algorithm to rank documents with additive ensembles of regression trees](https://www.cse.cuhk.edu.hk/irwin.king/_media/presentations/sigir15bestpaperslides.pdf)
- [ ] [Official repository of Quickscorer](https://github.com/hpclab/quickscorer)
- [ ] [QuickRank: A C++ suite of Learning to Rank algorithms](http://quickrank.isti.cnr.it/research-papers/)
- http://ecmlpkdd2017.ijs.si/papers/paperID718.pdf
+ [Boosted Ranking Models: A Unifying
Framework for Ranking Predictions](http://www.cs.cmu.edu/~kdelaros/kais2011.pdf)

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

* All users are presumed to act independently of each other.
* The ordering of each pair of items $(i, j)$ for a specific user is presumed to be independent of the ordering of every other pair.

We introduce a general prior density $P(\Theta)$ which is a normal distribution
with zero mean and variance-covariance matrix $\Sigma_{\Theta}$:
$$\Theta\sim N(0, \Sigma_{\Theta}).$$

To reduce the number of unknown hyperparameters we set
$$\Sigma_{\Theta}=\lambda_{\Theta}I.$$

Now we can formulate the **maximum posterior estimator** to derive our generic optimization criterion for personalized ranking:
$$
\ln(P(\Theta|\triangleleft_{U}))\\
  =\ln(P(\triangleleft_{U}|\Theta) P(\Theta))\\
  =\ln (\prod_{(u,i,j)} P(U_i\triangleleft U_j|\Theta) P(\Theta))\\
  =\sum_{(u,i,j)}\ln(P(U_i\triangleleft U_j|\Theta))+\ln(P(\Theta)) \\
  =\sum_{(u,i,j)}\ln(\sigma(f_U(i,j|\Theta)))-\lambda_{\Theta} {\|\Theta\|}^2
$$

where $\lambda_{\Theta}$ are model specific regularization parameters.

And we can use stochastic gradient descent to find the parameters $\Theta$.
***

- [ ] [论文快读 - BPR: Bayesian Personalized Ranking from Implicit Feedback](http://lipixun.me/2018/01/22/bpr)
- [ ] [BPR [Bayesian Personalized Ranking] 算法详解及应用实践](https://liuzhiqiangruc.iteye.com/blog/2073526)
- [ ] [BPR [Bayesian Personalized Ranking] 算法详解及应用实践](https://blog.csdn.net/cht5600/article/details/54381011)
- [ ] [BPR：面向隐偏好数据的贝叶斯个性化排序学习模型](https://blog.csdn.net/qq_20599123/article/details/51315697)
- [ ] [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)
- [ ] [VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1510.01784.pdf)
- [ ] [Top-N Recommendations from Implicit Feedback Leveraging Linked Open Data ?](https://core.ac.uk/display/23873231)

#### Group Preference Based Bayesian Personalized Ranking

However, the two fundamental assumptions made in the pairwise ranking methods,
(1) individual pairwise preference over two items
and (2) independence between two users, may not
always hold.

[GBPR: Group Preference Based Bayesian Personalized Ranking for One-Class Collaborative Filtering](https://www.ijcai.org/Proceedings/13/Papers/396.pdf)  introduce richer interactions among users when the assumption does not hold.

For a typical user ${U}$, in order to calculate the over all likelihood of pairwise preferences (LPP) among all
items ${I}$, Bernoulli distribution over the binary random variable
$\delta(U_i\triangleleft U_j)$ is used in
$$
LLP(U) = \prod_{i, j \in I} P(U_i\triangleleft U_j)^{\delta(U_i\triangleleft U_j)} \times [1 - P(U_i\triangleleft U_j)]^{1- \delta(U_i\triangleleft U_j)}
$$
where $U_i\triangleleft U_j$ denotes that user ${U}$ prefer item ${i}$ to the item ${j}$.

The group preference of users from group ${G}$ on
item ${i}$ can be estimated as the average individual preferences
$$
\hat{r}_{G,i}=\frac{1}{|G|}\sum_{w\in G}\hat{r}_{w,i}.
$$

We assume that the group preference on an
item ${i}$ is more likely to be stronger than the individual preference of user ${u}$ on item ${j}$, if the user-item pair $(u, i)$ is observed and the user-item pair $(u, j)$ is not observed.

To explicitly study the unified effect of group preference
and individual preference,  we combine them linearly
$\hat{r}_{G, u,i} = \rho \hat{r}_{G,i} + (1-\rho)\hat{r}_{u,i}$.

A new criterion called group Bayesian personalized ranking (GBPR) for user ${u}$,
$$GBPR(u) = \prod_{i\in I_{U}}\prod_{j\in (I-I_{U})} P(\hat{r}_{G, u,i} > \hat{r}_{u, j})[1 - P(\hat{r}_{G, u,i} > \hat{r}_{u, j})]$$

where $I_{U}$ is the item set which user ${U}$ has expressed
positive feedback, and item ${i}$ is observed by user
${u}$  and item ${j}$ is not observed

And the user correlations have been introduced via the user group ${G}$. Then we have the following overall
likelihood for all users and all items,
$$
GBPR=\prod_{u\in U}GBPR(u) = \prod_{u\in U}  \prod_{i\in I_{U}}\prod_{j\in (I-I_{U})} P(\hat{r}_{G, u,i} > \hat{r}_{u, j}) [1 - P(\hat{r}_{G, u,i} > \hat{r}_{u, j})].
$$

Following *BPR*, we use $\sigma(\hat{r}_{G, u,i} - \hat{r}_{u, j}) = \frac{1}{1+\exp(-\hat{r}_{G, u,i} + \hat{r}_{u, j})}$ to approximate the probability $P(\hat{r}_{G, u,i} > \hat{r}_{u, j})$. The other probability is similar.

Finally, we reach the objective function of our GBPR,

$$\min_{\Theta} -\frac{1}{2}\ln(GBPR) + \frac{1}{2} \Omega(\Theta)$$

where $\Omega(\Theta)$ is the regularization term used to avoid over-fitting.

See more transfer learning algorithm in [http://csse.szu.edu.cn/staff/panwk/publications/].

* [The paper of GBPR](https://www.ijcai.org/Proceedings/13/Papers/396.pdf)
* [Transfer to Rank for Top-N Recommendation](http://csse.szu.edu.cn/staff/panwk/publications/Journal-TBD-19-CoFiToR-Slides.pdf)
* [The code and data of GBPR](http://csse.szu.edu.cn/staff/panwk/publications/index.html).

#### Collaborative Multi-objective Ranking

The rowwise ranking problem, also known as personalized ranking, aims to build user-specific models such that the correct order of items (in terms of user preference) is most accurately predicted and then items on the top of ranked list will be recommended to a specific
user, while column-wise ranking aims to build item-specific models focusing on targeting users who are most interested in the specific item (for example, for distributing coupons to customers).

The key part of collaborative ranking algorithms is to learn effective user and item latent factors
which are combined to decide user preference scores over items.

In **Collaborative Multi-objective Ranking**, it is to jointly solve row-wise and column-wise
ranking problems through a parameter sharing framework which optimizes three objectives together: to accurately predict rating
scores, to satisfy the user-specific order constraints on all the rated items, and to satisfy the item-specific order constraints.


And the above algorithms are pair-wise algorithms based on the logistic function $\sigma(x)=\frac{1}{1 + \exp(-ax)}$ as the surrogate of zero-one loss such as the  **Bradley-Terry Model** and **Bayesian Personalized Ranking**.

In logistic function, the value of "a" determines the shape of the function. In other words, it tells how close the approximation of logistic function to the zero-one loss. However, in the context of matrix factorization, the change of ${U_u}$ doesn’t necessarily
contribute to the change of approximation to zero-one loss as any change to ${U_u}$ (e.g., double ${U_u}$ ) can be compensated by changing all the item factors $V_i$ accordingly (e.g., reduce $V_i$ by half).


The general idea of matrix factorization is to assume that the rating matrix $R \in \mathbb{R}^{m\times n}$ has low rank and thus it can be approximated by $R = UV^⊺$,
where $U\in \mathbb{R}^{m\times k}$ and $V ∈ \mathbb{R}^{n\times k}$ respectively represent user latent factors and item latent factors, and ${k}$ is the rank of
approximation.
The prediction loss of rating prediction through matrix factorization on the training set is formulated as
$$
L_{pointwise}=\sum_{u}\sum_{i}(r_{ui}-\hat{r}_{ui})^2 + {\lambda}_{U}\sum_{u}{\|U_u\|}_2^2 + {\lambda}_{I}\sum_{i}{\|V_i\|}_2^2 \tag 1
$$

where $r_{ui}$ and $\hat{r}_{ui}$ are respectively the observed and estimated rating scores. The regularized term is set in order to prevent from over-fitting.

By modeling row-wise comparisons of (user-specific) item pairs using Bradley-Terry model together with matrix factorization, it can be formulated as follows

$$
P(u_i\triangleleft u_j )=P(r_{ui}>r_{uj})=\frac{\exp(U_u V_i^T)}{\exp(U_u V_i^T)+\exp(U_u V_j^T)}.
$$

We then minimize negative log likelihood on all the comparisons
of observed item pairs, obtaining the following objective function

$$
L_{row-wise}= -\sum_{u_i\triangleleft u_j }\log(P(r_{ui}>r_{uj})) + {\lambda}_{I}\sum_{i}{\|V_i\|}_2^2 .\tag 2
$$

Symmetric to that in modeling the row-wise comparisons, we model the (item-specific) column-wise comparisons

$$
P(u_i\triangleleft u^{\prime}_i )=P(r_{ui}>r_{u^{\prime}i})=\frac{\exp(U_u V_i^T)}{\exp(U_u V_i^T)+\exp(U_{u^{\prime}} V_j^T)}.
$$

Then, the objective function becomes:

$$
L_{column-wise}= -\sum_{u_i\triangleleft u^{\prime}_i }\log(P(r_{ui}>r_{u^{\prime}i})) + {\lambda}_{I}\sum_{u}{\|U_u\|}_2^2 .\tag 3
$$

We introduce two balance factors $\alpha ∈ [0, 1]$
and $\beta ∈ [0, 1]$, s.t., $\alpha +\beta \le 1$, to combine aforementioned three losses.
The final integrated loss is introduced in the following formulation:
$$
L= \alpha L_{column-wise}  + \beta L_{row-wise} +(1-\alpha -\beta)L_{point-wise}
$$

where balance factors $\alpha$ and $\beta$ are set to model the importance of individual losses.
Intuitively, the weight of each loss function should
be set differently in solving different problems.


* http://column.hongliangjie.com
* [Learning to Rank with Multiple Objective Functions](http://www.cs.toronto.edu/~mvolkovs/www2011_lambdarank.pdf)
* https://sites.google.com/site/hujun1010/


#### Adaptive Boosting Personalized Ranking (AdaBPR)

`AdaBPR (Adaptive Boosting Personalized Ranking)` is a boosting algorithm for top-N item recommendation using users' implicit feedback.
In this framework, multiple homogeneous component recommenders are linearly combined to achieve more accurate recommendation.
The component recommenders are learned based on a re-weighting strategy that assigns a dynamic weight to each observed user-item interaction.

The primary idea of applying boosting for item recommendation is to learn a set of homogeneous component recommenders and then create an ensemble of the component recommenders to predict users' preferences.

Here, we use a linear combination of component recommenders as the final recommendation model
$$f=\sum_{t=1}^{T}{\alpha}_t f_{t}.$$

In the training process, AdaBPR runs for ${T}$ rounds, and the component recommender $f_t$ is created at t-th round by
$$
\arg\min_{f_t\in\mathbb{H}} \sum_{(u,i)\in\mathbb{O}} {\beta}_{u} \exp\{-E(\pi(u,i,\sum_{n=1}^{t}{\alpha}_n f_{n}))\}
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

**Deep Online Ranking System**

DORS is designed and implemented in a three-level novel architecture, which includes (1) candidate retrieval; (2) learning-to-rank deep neural network (DNN) ranking; and (3) online
re-ranking via multi-arm bandits (MAB).
- [ ] https://zhuanlan.zhihu.com/p/57056588
- [ ] [A Practical Deep Online Ranking System in E-commerce Recommendation](http://www.ecmlpkdd2018.org/wp-content/uploads/2018/09/723.pdf)
- [ ] [European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases](http://www.ecmlpkdd2018.org/)
- [ ] [大众点评基于知识图谱的深度学习排序](https://tech.meituan.com/2019/01/17/dianping-search-deeplearning.html)
- [ ] [Log-Linear Models for Label Ranking](https://academic.microsoft.com/#/detail/2149166361)
- [ ] [International Conference on Web Search and Data Mining](http://www.wsdm-conference.org/2019/acm-proceedings.php)
- [ ] [Learning to Rank with Deep Neural Networks](https://github.com/Isminoula/DL-to-Rank)



+ [Adversarial and reinforcement learning-based approaches to information retrieval](https://www.microsoft.com/en-us/research/blog/adversarial-and-reinforcement-learning-based-approaches-to-information-retrieval/)
+ [Cross Domain Regularization for Neural Ranking Models Using Adversarial Learning](https://www.microsoft.com/en-us/research/publication/cross-domain-regularization-neural-ranking-models-using-adversarial-learning/)
+ [Adversarial Personalized Ranking for Recommendation](http://bio.duxy.me/papers/sigir18-adversarial-ranking.pdf)
![adversial IR](https://www.microsoft.com/en-us/research/uploads/prod/2018/06/adversarial.png)

**RankGAN**

![RankGAN](https://x-algo.cn/wp-content/uploads/2018/04/WX20180409-223208@2x-768x267.png)

- https://x-algo.cn/index.php/2018/04/09/rankgan/
- [IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval](https://arxiv.org/pdf/1705.10513.pdf)
- [Adversarial Ranking for Language Generation](http://papers.nips.cc/paper/6908-adversarial-ranking-for-language-generation)

***

* https://www.wikiwand.com/en/Learning_to_rank
* https://arxiv.org/abs/1812.00073
* https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf


### Collaborative Ranking

Collaborative Ranking sounds like collaborative filtering. In fact, collaborative ranking is also used to rank the items based on the feedback from users. [The computation of the Top-N item list for making recommendations is essentially a ranking problem.](http://www.cs.ust.hk/~qyang/Docs/2008/SIGIR297-liu.pdf)
**The general idea of CR is to combine matrix factorization (MF) with learning-to-rank (LTR) techniques for the purpose of accurately recommending interesting items to users.** More matrix factorization for recommender system techniques  includes SVD, regularized SVD, SVD++ and so on.
And in this part we only talk on the Top-N recommendation.

* https://www.cs.rutgers.edu/events/phd-defense-collaborative-ranking-based-recommender-systems
- [ ] [Collaborative ranking-based recommender systems](https://rucore.libraries.rutgers.edu/rutgers-lib/59115/)
- [ ] [Decoupled Collaborative Ranking](https://www.researchgate.net/publication/315874080_Decoupled_Collaborative_Ranking)
- [ ] [Large-scale Collaborative Ranking in Near-Linear Time](http://www.stat.ucdavis.edu/~chohsieh/rf/KDD_Collaborative_Ranking.pdf)
- [ ] [Preference Completion: Large-scale Collaborative Ranking from Pairwise Comparisons](http://proceedings.mlr.press/v37/park15.html)
- [ ] [Local Collaborative Ranking](https://ai.google/research/pubs/pub42242)
- [ ] [SQL-Rank: A Listwise Approach to Collaborative Ranking](http://proceedings.mlr.press/v80/wu18c.html)
- [ ] [VSRank: A Novel Framework for Ranking-Based
Collaborative Filtering](http://users.jyu.fi/~swang/publications/TIST14.pdf)
- [ ] [Machine Learning: recommendation and ranking](https://jhui.github.io/2017/01/15/Machine-learning-recommendation-and-ranking/)
- [ ] [Recommender systems & ranking](https://sites.google.com/view/chohsieh-research/recommender-systems)
- [ ] [Recommendation and ranking by Mark Jelasity](http://www.inf.u-szeged.hu/~jelasity/ddm/graphalgs.pdf)
- [ ] ["Tutorial ：Learning to Rank for Recommender Systems" by](http://www.slideshare.net/kerveros99/learning-to-rank-for-recommender-system-tutorial-acm-recsys-2013)
- [ ] [Rank and Relevance in Novelty and Diversity Metrics for Recommender Systems](http://ir.ii.uam.es/predict/pubs/recsys11-vargas.pdf)

For item recommendation tasks, the accuracy of a recommendation model is usually evaluated using the `ranking metrics`.

* [Ranking Evaluation](http://csse.szu.edu.cn/staff/panwk/recommendation/OCCF/RankingEvaluation.pdf)
* http://fastml.com/evaluating-recommender-systems/


**Top-N recommendation**

* http://glaros.dtc.umn.edu/gkhome/node/1192
* https://www.ijcai.org/Proceedings/16/Papers/339.pdf
* https://blog.csdn.net/lthirdonel/article/details/80021282
* https://arxiv.org/abs/1808.04957v1
* http://ceur-ws.org/Vol-1127/paper4.pdf


### Personalized Search

[The basic functions of a search engine can be described as _crawling, data mining, indexing and query processing_. `Crawling` is the act of sending small programed bots out to collect information. `Data mining` is storing the information collected by the bots. `Indexing` is ordering the information systematically. And `query processing` is the mathematical process in which a person's query is compared to the index and the results are presented to that person.](https://lifepacific.libguides.com/c.php?g=155121&p=1018180)

<img title = "”search process" src = http://www.searchtools.com/slides/images/search-process.gif width=50% />

[Personalised Search fetches results and delivers search suggestions individually for each of its users based on their interests and preferences](https://yandex.com/company/technologies/personalised_search/), which is mined from the information that the search engine has about the user at the given time, such as their location, search history, demographics such as the recommender s

And here search engine and recommender system coincide except the recommender system push some items in order to attract the users' attention while search engine recall the information that the users desire in their mind.

* http://ryanrossi.com/search.php
* https://a9.com/what-we-do/product-search.html
* https://www.algolia.com/
* https://www.cognik.net/
* http://www.collarity.com/
* https://www.wikiwand.com/en/Personalized_search
* [The Mathematics of Web Search](http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/index.html)
* [CSAW: Curating and Searching the Annotated Web](https://www.cse.iitb.ac.in/~soumen/doc/CSAW/)
* [A Gradient-based Framework for Personalization by Liangjie Hong](http://www.hongliangjie.com/talks/Gradient_Indiana_2017-11-10.pdf)
* [Style in the Long Tail: Discovering Unique Interests with Latent Variable Models in Large Scale Social E-commerce](https://mimno.infosci.cornell.edu/info6150/readings/p1640-hu.pdf)
* [Personalised Search in Yandex](https://yandex.com/company/technologies/personalised_search/)
* [Thoughts on Yandex personalized search and beyond](https://www.russiansearchtips.com/2012/12/thoughts-on-yandex-personalized-search-and-beyond/)
* [Yandex filters & algorithms. 1997-2018](https://www.konstantinkanin.com/en/yandex-algorithms/)
* [Google's Personalized Search Explained: How personalization works](https://www.link-assistant.com/news/personalized-search.html)
* [A Better Understanding of Personalized Search](https://www.briggsby.com/better-understanding-personalized-search)
* [Interest-Based Personalized Search](https://www.cpp.edu/~zma/research/Interest-Based%20Personalized%20Search.pdf)
* [Search Personalization using Machine Learning by Hema Yoganarasimhan](https://faculty.washington.edu/hemay/search_personalization.pdf)
* [Web Personalisation and Recommender Systems](https://www.kdd.org/kdd2015/slides/KDD-tut.pdf)
* [Scaling Concurrency of Personalized Semantic Search over Large RDF Data](https://research.csc.ncsu.edu/coul/Pub/BigD402.pdf)
* [Behavior‐based personalization in web search](https://onlinelibrary.wiley.com/doi/full/10.1002/asi.23735)

## IR and Search

If the recommendation is to solve the information overload problem, information retrieval and search technology  is to find the relative entity in web or some data base if the query is given.
[Technically, IR studies the acquisition, organization, storage, retrieval, and distribution of information.](http://www.dsi.unive.it/~dm/Slides/5_info-retrieval.pdf)
Information is in diverse format or form, such as charactor strings(articles), images, voices and videos.
`Ranking` and `Relavance`  is two perpsectives of search.  
In this section, we focus on relavance rather than rank.
If interested in the history of information retrieval, Mark Sanderson and W. Bruce Croft wrote a paper for [The History of Information Retrieval Research](https://ciir-publications.cs.umass.edu/pub/web/getpdf.php?id=1066).


+ [Search and information retrieval@Microsoft](https://www.microsoft.com/en-us/research/research-area/search-information-retrieval/)
+ [Search and information retrieval@Google](https://ai.google/research/pubs/?area=InformationRetrievalandtheWeb)
+ [Web search and mining @Yandex](https://research.yandex.com/publications?themeSlug=web-mining-and-search)
+ [Information Retrieval Lab: A research group @ University of A Coruña (Spain)](https://www.irlab.org/)
+ [ BCS-IRSG: Information Retrieval Specialist Group](https://irsg.bcs.org/)
+ [智能技术与系统国家重点实验室信息检索课题组](http://www.thuir.org/)
+ [The Cochrane Information Retrieval Methods Group (Cochrane IRMG)](https://methods.cochrane.org/irmg/)
+ [SOCIETY OF INFORMATION RETRIEVAL & KNOWLEDGE MANAGEMENT (MALAYSIA)](http://pecamp.org/web14/)
+ [Quantum Information Access and Retrieval Theory)](https://www.quartz-itn.eu/)
+ [Center for Intelligent Information Retrieval (CIIR)](http://ciir.cs.umass.edu/)
+ [InfoSeeking Lab is situated in School of Communication & Information at Rutgers University.](https://infoseeking.org/)
+ http://mlwiki.org/index.php/Information_Retrieval
+ [information and language processing systems](https://ilps.science.uva.nl/)
+ [information retrieval facility](https://www.ir-facility.org/)
+ [Center for Information and Language Processing](https://www.cis.uni-muenchen.de/)
+ [Summarized Research in Information Retrieval for HTA](http://vortal.htai.org/?q=sure-info)
+ [SIGIR](https://sigir.org/)
+ http://cistern.cis.lmu.de/
+ http://hpc.isti.cnr.it/
***
+ [European Conference on Information Retrieval (ECIR 2018)](https://www.ecir2018.org/)
+ [ECIR 2019](http://ecir2019.org/workshops/)
+ [IR @wikiwand](https://www.wikiwand.com/en/Information_retrieval)
+ [Algorithm Selection and Meta-Learning in Information Retrieval (AMIR)](http://amir-workshop.org/)
+ [The ACM SIGIR International Conference on the Theory of Information Retrieval (ICTIR)2019](http://www.ictir2019.org/)
+ [KDIR 2019](http://www.kdir.ic3k.org/)
+ [Advances in Semantic Information Retrieval (ASIR’19)](https://fedcsis.org/2019/asir)
+ [Music Information Retrieval Evaluation eXchange (MIREX 2019)](https://www.music-ir.org/mirex/wiki/MIREX_HOME)
+ [20th annual conference of the International Society for Music Information Retrieval (ISMIR)](https://ismir2019.ewi.tudelft.nl/)
+ [8th International Workshop on Bibliometric-enhanced Information Retrieval](http://ceur-ws.org/Vol-2345/)
+ [ICMR 2019](http://www.icmr2019.org/)
+ [3rd International Conference on Natural Language Processing and Information Retrieval](http://www.nlpir.net/)
+ [FACTS-IR Workshop @ SIGIR 2019](https://fate-events.github.io/facts-ir/)
+ [ACM Conference of Web Search and Data Mining 2019](http://www.wsdm-conference.org/2019/)
+ [SMIR 2014](http://smir2014.noahlab.com.hk/SMIR2014.htm)
+ [2018 PRS WORKSHOP:  Personalization, Recommendation and Search (PRS)](https://prs2018.splashthat.com/)
***
+ [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)
+ [CS 371R: Information Retrieval and Web Search](https://www.cs.utexas.edu/~mooney/ir-course/)
+ [CS 242: Information Retrieval & Web Search, Winter 2019](http://www.cs.ucr.edu/~vagelis/classes/CS242/index.htm)
+ [Winter 2017 CS293S: Information Retrieval and Web Search](https://sites.cs.ucsb.edu/~tyang/class/293S17/)
+ [CS 276 / LING 286: Information Retrieval and Web Search](https://web.stanford.edu/class/cs276/)
+ [Information Retrieval and Web Search 2015](http://web.eecs.umich.edu/~mihalcea/498IR/)
+ [Data and Web Mining](http://www.dsi.unive.it/~dm/)
+ [Neural Networks for Information Retrieval](http://wwwir.com)
+ [Introduction to Search Engine Theory](http://ryanrossi.com/search.php)
+ [INFORMATION RETRIEVAL FOR GOOD](http://romip.ru/russir2018/)
+ [Search user interfaces](http://searchuserinterfaces.com/book/)
+ [Morden Information Retrieval](http://grupoweb.upf.edu/mir2ed/home.php)
+ [Search Engine: Information Retrieval in Practice](http://www.search-engines-book.com/)
+ [Neu-IR: The SIGIR 2016 Workshop on Neural Information Retrieval](https://www.microsoft.com/en-us/research/event/neuir2016/)
