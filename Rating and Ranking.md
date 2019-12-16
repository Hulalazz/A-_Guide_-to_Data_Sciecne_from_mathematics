# Rating and Ranking

<img src="https://pic1.zhimg.com/80/v2-ec0751e41981077e932ae0ce2cf6fe48_hd.jpg" width="80%" />

The rating algorithms help to match the players in video games or compare the players in sports.
Ratings is a numerical score to describe the level  of the players' skill based on the results of many competition.
The basic idea is the back-feed from the results to improve the experience. After each game, this data is updated for the participants in the game.

The ranking problem is from information retrieval. Given a query as we type in a search engine, the ranking algorithms are to sort the items
which may answer this query as the PageRank does for web searching. And `search engine optimization (SOE)` can be regarded as the reverse engineer of the ranking algorithms of search engine.

<img title="science of rating and ranking" src="https://images-na.ssl-images-amazon.com/images/I/51bML705X7L._SX353_BO1,204,203,200_.jpg" width="20%" />

They share some techniques such as the logistic regression although their purpose is different.

|[3 R's for the inter-net age](https://phys.org/news/2016-12-recommendingthree-internet-age.html)|Rating, ranking and recommending|
|---|--- |
| Rating |A rating of items assigns a numerical score to each item. A rating list, when sorted, creates a ranking list.|
| Ranking |A ranking of items is a rank-ordered list of the items. Thus, a ranking vector is a permutation of the integers 1 through n.|
| Recommendation | Recommendation Algorithms predict how each specific user would rate different items she has not yet bought by looking at the past history of her own ratings and comparing them with those of similar users.|

In some sense, rating is to evaluate in a quantity approach, i.e. how much the item is popular; ranking is to evaluate in a quality approach. i.e., whether it is popular or preferred; recommendation is to rate or rank the items if the information is undirected or implicit.

- [ ] [Who’s #1? The Science of Rating and Ranking](http://www.ams.org/notices/201301/rnoti-p81.pdf)
- [ ] [Who Is the Best Player Ever? A Complex Network Analysis of the History of Professional Tennis](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0017249)
- [ ] [WhoScored Ratings Explained](https://www.whoscored.com/Explanations)
- [ ] [Who are the best MMA fighters of all time. A Bayesian study](https://blog.datadive.net/who-are-the-best-mma-fighters-of-all-time-a-bayesian-study/)
- [ ] [Ranking Algorithm Definition](http://www.meteorsite.com/ranking-algorithm)
- [ ] [EdgeRank](http://edgerank.net/)
- [ ] [SofaScore Statistical Ratings](https://www.sofascore.com/news/sofascore-player-ratings/)
- [ ] [Everything You Need to Know About the TripAdvisor Popularity Ranking](https://www.tripadvisor.com/TripAdvisorInsights/w765)
- [ ] [Deconstructing the App Store Rankings Formula with a Little Mad Science](https://moz.com/blog/app-store-rankings-formula-deconstructed-in-5-mad-science-experiments)
- [ ] [Mathematics and Voting: More Than Just Counting Votes](http://www.whydomath.org/node/voting/index.html)
* http://www.whydomath.org/index.html
* [Introducing Steam Database's new rating algorithm](https://steamdb.info/blog/steamdb-rating/)
* https://www.cs.cornell.edu/jeh/book2016June9.pdf
* [Neural Collaborative Ranking](https://arxiv.org/abs/1808.04957v1)

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
- [ ] [The USCF Rating System, Prof. Mark E. Glickman , Boston University - Thomas Doan Estima](http://math.bu.edu/people/mg/ratings/rs/)
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

> 1.  Determine RD
>  The new Ratings Deviation (RD) is found using the old Ratings Deviation $RD_0$:
> $$RD=\min\{\sqrt{RD_0^2+c_2t}, 350\}$$
>  where ${t}$ is the amount of time (rating periods) since the last competition and '350' is assumed to be the RD of an unrated player. And $c=\sqrt{(350^2-50^2)/100}\simeq 34.6$.
>
> 2. Determine New Rating
>   The new ratings, after a series of m games, are determined by the following equation:
> $$r=r_0+\frac{q}{RD^{-2}+d^{-2}}\sum_{i=1}^{m}g(RD_i)(s_i - E(s |r,r_i,RD_i))$$
> where
>  * $g(RD_i)=\{1+\frac{3q^2(RD_i)^2}{\pi^2}\}^{-1/2}$, $E(s | r, r_ i, RD_i))=\{1 + 10^{(\frac{g(RD_i)(r-r_i)}{-400})}\}$,
>  * $q=\frac{\ln(10)}{400}\approx 0.00575646273$,
>  * $d^{-2} = q^2\sum_{i=1}^{m}[g(RD_i)^2]E(s | r, r_i, RD_i)[1-E(s | r, r_i, RD_i)]$,
>  * $r_i$ represents the ratings of the individual opponents.
>  * $s_i$ represents the outcome of the individual games. A win is ${1}$, a draw is $\frac {1}{2}$, and a loss is $0$.
>
> 3. Determine New Ratings Deviation
>
>   $$RD^{\prime}=\sqrt{(RD^{-2}+d^{-2})^{-1}} .$$

* [Mark Glickman's Research](http://www.glicko.net/research.html)
* [Glicko Ratings](http://www.glicko.net/glicko.html)
* https://www.wikiwand.com/en/Glicko_rating_system
* [Java implementation of the Glicko-2 rating algorithm](https://github.com/goochjs/glicko2)


### TrueSkill

As shown in the rule to update the score in Elo, it only take the differences of score into consideration.
The TrueSkill system will assume that the distribution of the skill is **location-scale** distribution. In fact, the prior distribution in Elo is **Gaussian distribution**.
The expected performance of the player is his mean of the distribution. The variance is the uncertainty  of the system.
[We have already noted that skill is an uncertain quantity, and should therefore be included in the model as a random variable. We need to define a suitable prior distribution for this variable. This distribution captures our prior knowledge about a player’s skill before they have played any games. Since we know very little about a player before they play any games, this distribution needs to be broad and cover the full range of skills that a new player might have. Because skill is a continuous variable we can once again use a Gaussian distribution to define this prior.](http://www.mbmlbook.com/TrueSkill_Inferring_the_players_skills.html)

The three assumptions encoded in our model are:

+ Each player has a skill value, represented by a continuous variable with a broad prior distribution.
+ Each player has a performance value for each game, which varies from game to game such that the average value is equal to the skill of that player. The variation in performance, which is the same for all players, is symmetrically distributed around the mean value and is more likely to be close to the mean than to be far from the mean.
+ The player with the higher performance value wins the game.

The update rule will update the mean and variance:
***
$$
\mu_{winner} \leftarrow \mu_{winner} + \frac{\sigma_{winner}^2}{c} \cdot \nu(\frac{(\mu_{winner}-\mu_{loser})}{c},\frac{\epsilon}{c})\\
\mu_{loser} \leftarrow \mu_{loser} - \frac{\sigma_{loser}^2}{c} \cdot \nu(\frac{(\mu_{winner}-\mu_{loser})}{c},\frac{\epsilon}{c})
\\
\sigma_{winner} \leftarrow \sigma_{winner}^2 [1 - \frac{\sigma_{winner}^2}{c^2} \cdot w(\frac{(\mu_{winner}-\mu_{loser})}{c},\frac{\epsilon}{c}) ]
\\
\sigma_{loser} \leftarrow \sigma_{loser}^2 [1 - \frac{\sigma_{loser}^2}{c^2} \cdot w(\frac{(\mu_{winner}-\mu_{loser})}{c},\frac{\epsilon}{c}) ]
\\
c^2 = 2\beta^2+\sigma^2_{winner}+\sigma^{2}_{loser}
$$
where $\beta^2$ is the average of all players' variances, $\nu(t) =\frac{N(t)}{\Phi(t)}, w(t)=\nu(t)[\nu(t)+1]$. And $N(t),\Phi(t)$ is the PDF and CDF of standard normal distribution, respectively.
***
$\color{red}{PS}$: $TrueSkill^{TM}$ is a commercial trademark.
- [X] https://www.wikiwand.com/en/TrueSkill
- [X] [算法_TrueSkill_Python](https://www.jianshu.com/p/c1fbba3af787)
- [ ] [Chapter 3: Meeting Your Match](http://www.mbmlbook.com/TrueSkill.html)
- [ ] [TrueSkill原理简介](https://zhuanlan.zhihu.com/p/48737998)
- [ ] https://www.wikiwand.com/en/Location%E2%80%93scale_family
- [ ] [TrueSkill™ Ranking System](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/)
- [ ] [TrueSkill: the video game rating system](https://trueskill.org/)
- [ ] [Herbrich, R., Minka, T., and Graepel, T. (2007). TrueSkill(TM): A Bayesian Skill Rating System. In Advances in Neural Information Processing Systems 20, pages 569–576. MIT Press.](https://ieeexplore.ieee.org/document/6287323/)
- [ ] https://pypi.org/project/trueskill/

### Edo Historical Chess Rating
In summary, the Edo system is done by
- Obtaining maximum-likelihood estimates of ratings based on tournament and match results by applying a single large Bradley-Terry algorithm using each player in each year of their career as a separate entity, the comparisons between them being:
  - results of real games, and
  - 50% scores in hypothetical games between adjacent years for the same player (30 games),
- Calculating rating deviations - standard errors of the rating estimates from the Bradley-Terry results, and
- Adjusting the ratings by calculating a combined maximum likelihood estimate of the score-based rating and the rating based on the underlying 'prior' rating distribution.

+ [Rating historical chess players](http://www.edochess.ca/Edo.explanation.html)
+ http://www.edochess.ca/


### Whole-History Rating

Incremental Rating Systems or dynamical rating systems such as TrueSkill  do not make optimal use of data.
This idea may be refined by giving a decaying weight to games, either
exponential or linear. With this decay, old games are progressively forgotten,
which allows to measure the progress of players.
The main problem is that the decay of game weights generates a very fast increase in the uncertainty of player ratings.

The weakness of algorithms like `Glicko` and `TrueSkill` lies in the inaccuracies of representing the probability distribution with just one value and one variance for every player, ignoring covariance.
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

In the `dynamic Bradley-Terry model`, the prior has two roles. First, a prior
probability distribution over the range of ratings is applied. This way, the rating
of a player with $100\%$ wins does not go to infinity. Also, a prior controls the variation of the ratings in time, to avoid huge jumps in ratings.
The Bradley-Terry model for paired comparisons gives the probability of winning a game as a function of ratings:
$$P(\text{player i beats player j at time t}) =\frac{\gamma_i(t)}{\gamma_i(t)+\gamma_j(t)}$$

where

+ Player number: $i \in \{1, \dots , N\}$, integer index.
+ Elo rating of player $i$ at time $t$: $R_i(t)$, real number.
+ $\gamma$ rating of player $i$ at time $t$: $\gamma_i(t)$, defined by  $\gamma_i(t)=10^{\frac{R_i(t)}{400}}$
+ Natural rating of player i at time t: $r_i(t) = \ln gamma_i(t) = R_i(t)\frac{\ln 10}{400}$.


In the `dynamic Bradley-Terry model`, the prior that controls the variation of
ratings in time is a Wiener process:

$$
r_i(t_1) - r_i(t_1)\sim N(0, |t_2-t_1|w^2) ,
$$

where $w$ is a parameter of the model, that indicates the variability of ratings in time.
The extreme case of $w = 0$ would mean static ratings.

The `WHR algorithm` consists in computing, for each player, the $r(t)$ function
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

_____
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
* [Arrow's Theorem by Terence Tao](https://www.math.ucla.edu/~tao/arrow.pdf)
* [Gibbard–Satterthwaite theorem @ wikiwand](https://www.wikiwand.com/en/Gibbard%E2%80%93Satterthwaite_theorem)
* [Do the Math: Why No Ranking System Is No. 1](https://www.scientificamerican.com/article/why-ranking-systems-are-flawed/)

## Ranking



* [Learning to Rank explained](https://everything.explained.today/Learning_to_rank/)
* [Learning to Rank with (a Lot of) Word Features](http://ronan.collobert.com/pub/matos/2009_ssi_jir.pdf)
* http://www.cs.cmu.edu/~kdelaros/
* [Ranking in information retrieval](https://www.wikiwand.com/en/Ranking_(information_retrieval))
* [Learning to Rank](https://jimmy-walker.gitbooks.io/rank/L2R.html)
* [Hardened Fork of Ranklib learning to rank library](https://github.com/o19s/RankyMcRankFace)
* [OpenSource Connections](https://github.com/o19s)
* https://mathcitations.github.io/

Combining feedback from multiple users to rank a collection of items is an important task.
The ranker, a central component in every `search engine`, is responsible for the matching between processed queries and indexed documents in information retrieval.
The goal of a ranking system is to find the best possible ordering of a set of items for a user, within a specific context, in real-time in recommender system.

In general, we call all those methods that use machine learning technologies to solve the problem of ranking **"learning-to-rank"** methods or **LTR** or **L2R**.
We are designed to compare some indexed document with the query.
The algorithms above are based on feedback to tune the rank or scores of players or documents. The drawback of these methods is that they do not take the features of the players into consideration.
We may use machine learn to predict the scores of players and test it in real data set such as **RankNet, LambdaRank, and LambdaMART**.

And it can be applied to information retrieval and recommender system.

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
the listwise approach is to predict the ranks of documents in a list when given a query. Recall the following relation:
$$\fbox{Learning = Representation + Evaluation + Optimization}.$$

The representation of the `l2r` is the core and focus.

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
+ [Learning to Efficiently Rank with Cascades](http://lintool.github.io/NSF-projects/IIS-1144034/)
- [ ] [Learning Groupwise Scoring Functions Using Deep Neural Networks](https://arxiv.org/abs/1811.04415)
- [ ] https://github.com/tensorflow/ranking

Learning to rank is about performing ranking using machine learning techniques.
It is based on previous work on ranking in machine learning and statistics, and it also has its own characteristics.
There may be two definitions on learning to rank. In a broad sense, learning to rank refers to any machine learning techniques for ranking.
In a narrow sense, learning to rank refers to machine learning techniques for building ranking models in `ranking creation` and `ranking aggregation`.



### Training Setting

A training set for ranking  is denoted as $R=\{(\mathrm{X_i}, y_i)\mid i=1, \dots, m.\}$ where $y_i$ is the ranking of $x_i$, that is, $y_i < y_j$ if $x_i ≻ x_j$, i.e., $x_i$ is preferred to $x_j$ or in the reverse order. In other word, the label $y_i$ is ordinal. By the way, the labels are categorical or  nominal  in most classification tasks.

The ranking function outputs a score for each data object, from which a global
ordering of data is constructed. That is, the target function $F(x_i)$ outputs a score
such that $F(x_i) > F(x_j)$ for any $x_i ≻ x_j$.

- [WHAT IS THE DIFFERENCE BETWEEN CATEGORICAL, ORDINAL AND INTERVAL VARIABLES?](https://stats.idre.ucla.edu/other/mult-pkg/whatstat/what-is-the-difference-between-categorical-ordinal-and-interval-variables/)

**Point-wise Approach**

The input space of the pointwise approach contains a feature vector of each single
document.
The output space contains the relevance degree of each single document.

The hypothesis space contains functions that take the feature vector of a document
as input and predict the relevance degree of the document.
We usually call such a function $f$ the `scoring function`. Based on the scoring function, one can sort all the documents and produce the final ranked list.
The loss function examines the accurate prediction of the ground truth label for
each single document. In different pointwise ranking algorithms, ranking is modeled
as regression, classification, and ordinal regression, respectively.
Therefore the corresponding regression loss, classification loss, and ordinal regression loss are used as the loss functions.

**The Pairwise Approach**

The input space of the pairwise approach contains pairs of documents, both represented by feature vectors.
The output space contains the pairwise preference (which takes values from ${+1,−1}$) between each pair of documents.
The hypothesis space contains bi-variate functions h that take a pair of documents as input and output the relative order between them.

Note that the loss function used in the pairwise approach only considers the relative order between two documents.
When one looks at only a pair of documents, however, the position of the documents in the final ranked list can hardly be derived.
Furthermore, the approach ignores the fact that some pairs are generated from the documents associated with the same query.
Considering that most evaluation measures for information retrieval are query level and position based,
we can see a gap between this approach and ranking for information retrieval.

**The Listwise Approach**

The input space of the listwise approach contains a set of documents associated with query $q$, e.g., $x = \{x_j\}^m_{j=1}$.
The output space of the listwise approach contains the ranked list (or permutation)
of the documents.
Note that for the listwise approach, the output space that facilitates the learning process is exactly the same as the output space of the task.
In this regard, the theoretical analysis on the listwise approach can have a more direct value to understanding the real ranking problem than the other approaches
where there are mismatches between the output space that facilitates learning and the real output space of the task.

The hypothesis space contains multi-variate functions $h$ that operate on a set of documents and predict their **permutation**.

There are two types of loss functions, widely used in the listwise approach.
For the first type, the loss function is explicitly related to the evaluation measures
(which we call the measure-specific loss function), while for the second type, the loss function
is not (which we call the non-measure-specific loss function).
Note that sometimes it is not very easy to determine whether a loss function is listwise, since some building blocks of a listwise loss may also seem to be pointwise or pairwise

#### Ranking Metrics

The metrics or evaluation is different in regression and classification. And many loss function is introduced in maximum likelihood estimation.

![Recall](https://nlp.stanford.edu/IR-book/html/htmledition/img532.png)

`Precision` measures the exactness of the retrieval process. If the actual set of relevant documents is denoted by _I_ and the retrieved set of documents is denoted by _O_, then the precision is given by:
$$Precision= \frac{|O\cap I|}{|O|}.$$
Precision at position k for query $q$:
$$P@k=\frac{\text{the number of relevant documents in top k results}}{k}.$$

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
AveP= \sum_{k=1}^{n} p(k)\Delta r(k)=\frac{\sum_{k}P@k\times l_k}{\text{the number of relevant documents}}.
$$

where $k$ is the rank in the sequence of retrieved documents, $n$ is the number of retrieved documents, $P(k)$ is the precision at cut-off k in the list, and $\Delta r(k)$ is the change in recall from items $k-1$ to $k$.

`Mean Average Precision` is defined as
$$MAP=\frac{\sum_{q=1}^{Q} AveP(q)}{Q}.$$
****
`Cumulative Gain (CG)` is the predecessor of DCG and does not include the position of a result in the consideration of the usefulness of a result set. In this way, it is the sum of the graded relevance values of all results in a search result list. The CG at a particular rank position _p_ is defined as:
$${CG}_p=\sum_{i=1}^{p}{rel}_i,$$
Where ${rel}_{i}$ is the graded relevance of the result at position _i_.

The premise of `Discounted Cumulative Gain(DCG)` is that highly relevant documents appearing lower in a search result list should be penalized as the graded relevance value is reduced logarithmically proportional to the position of the result.

The traditional formula of _DCG_ accumulated at a particular rank position _p_ is defined as:
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

[`Weighted Approximate-Rank Pairwise (WARP) loss`](http://lyst.github.io/lightfm/docs/examples/warp_loss.html)  directly optimizes the precision@n and is useful when only positive interactions are present (as opposed to e.g. negative ratings or below-average ratings after normalization).

[`Weighted Margin-Rank Batch loss (WMRB)`](https://arxiv.org/pdf/1711.04015.pdf) extends the popular Weighted
Approximate-Rank Pairwise loss (WARP). WMRB uses a new rank
estimator and an efficient batch training algorithm

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
- [How is search different than other machine learning problems?](https://opensourceconnections.com/blog/2017/08/03/search-as-machine-learning-prob/)
- [RankEval: An Evaluation and Analysis Framework for Learning-to-Rank Solutions](https://github.com/hpclab/rankeval)
- [rankeval package](http://rankeval.isti.cnr.it/docs/rankeval.html)
- [	SIGIR 2016 Tutorial on Counterfactual Evaluation and Learning for Search, Recommendation and Ad Placement](http://www.cs.cornell.edu/~adith/CfactSIGIR2016/)
- [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf)
- [Deep Convolutional Ranking for Multilabel Image Annotation](https://arxiv.org/abs/1312.4894)
- [WMRB: Learning to Rank in a Scalable Batch Training Approach](https://arxiv.org/pdf/1711.04015.pdf)


#### Ranking Creation

We can generalize the ranking creation problems already described as a more general task.
Suppose that there are two sets. For simplicity, we refer to them as a set of requests $Q = {q_1, q_2, \cdots , q_i, \cdots , q_M}$ and a set of offerings (or objects) $\mathcal O = {o_1, o_2, \cdots , o_j , \cdots , o_N }$, respectively.
$Q$ can be a set of queries, a set of users, and a set of source sentences in document retrieval,
collaborative filtering, and machine translation, respectively.
$O$ can be a set of documents, a set of items, and a set of target sentences, respectively. Note that $Q$ and $O$ can be infinite sets.
Given an element $q$ of $Q$ and a subset $O$ of $\mathcal O$ ($O \in 2^{\mathcal O}$), we are to rank the elements in $O$ based on the information from $q$ and $O$.

Ranking (ranking creation) is performed with ranking (scoring) function $F(q, O): Q\times \mathcal O^n\to \mathbb{R}^n$:
$$
S_O=F(q, O), \\ \pi=sort_{O}(O),
$$

where $n=|O|$, $q$ denotes an element of $Q$, $O$ denotes a subset of $\mathcal O$, $S_O=$ denotes a set of scores of elements in $O$, and $\pi$ denotes a ranking list (permutation) on $O$ sorted by $S_O$.
Note that even for the same $O$, $F$ can give two different ranking lists with two different q’s.
That is to say, we are concerned with ranking on $O$, with respect to a specific $q$.

#### Ranking Aggregation

We can also define the general ranking aggregation task. Again, suppose that $Q = {q_1, q_2, \cdots , q_i, \cdots , q_M}$ and $\mathcal O = {o_1, o_2, \cdots , o_j , \cdots , o_N }$ are a set of requests and a set of offerings, respectively.
For an element $q$ of $Q and a subset of $O$ of $\mathcal O$, there are $k$ ranking list on $O:\Sigma=\{\pi_i\mid \pi\in\Pi, i=1,\cdots,k\}$, where $\Pi$ is the set of all ranking lists on $O$.
Ranking aggregation takes request $q$ and ranking lists of offerings as input and generates a new ranking list of offerings $\pi$ as output with ranking function $F(q,\Sigma):Q\times \Pi^k\to \mathbb R^n$
$$S_O=F(q, \Sigma),\\ \pi=sort_{S_O}(O).$$

We usually simply define
$$F(q, \Sigma)=F(\Sigma).$$

That is to say, we assume that the ranking function does not depend on the request.

Ranking aggregation is actually a process of combining multiple ranking lists into a single
ranking list, which is better than any of the original ranking lists.

Ranking creation generates ranking based on features of request and offering, while ranking aggregation generates ranking based on ranking of offerings. Note that the output of ranking aggregation can be used as the input of ranking aggregation.

### The Pointwise Approach

According to different machine learning technologies used, the pointwise approach
can be further divided into three subcategories: regression-based algorithms,
classification-based algorithms, and ordinal regression-based algorithms.
For regression-based algorithms, the output space contains real-valued relevance
scores; for classification-based algorithms, the output space contains non-ordered
categories; and for ordinal regression-based algorithms, the output space contains
ordered categories.

Given $x = \{x_j \}_{j=1}^m$,
a set of documents associated with training query $q$,
and the ground truth labels $y = \{y_j \}^m_{j=1}$ of these documents in terms of multiple ordered categories,
suppose a scoring function $f$ is used to rank these documents.
The loss function is defined as the following square loss,
$$L(f;x_i, y_i)=(y_i-f(x_i))^2$$
when the query $q$ is given.
***
The basic conclusion is that the square loss can upper bound the NDCG-based ranking error.
However, according to the above discussions, even if there is a large regression loss,
the corresponding ranking can still be optimal as long as the relative orders between the predictions $f (x_j ) (j = 1, \dots , m)$ are in accordance with those defined by the
ground truth label.
As a result, we can expect that the square loss is a loose bound of the NDCG-based ranking error.

---|---|---|---
---|---|---|---
---|Regression | Classification | Ordinal Regression
Input Space| Single documents $y_i$|Single documents $y_i$|Single documents $y_i$
Output Space | Real values |Non-ordered Categories | Ordinal categories
Hypothesis Space| Scoring function $f(x)$| Scoring function $f(x)$| Scoring function $f(x)$
Loss Function | Regression Loss |Classification Loss | Ordinal regression loss


### Ordinal Regression-Based Algorithms

Ordinal regression takes the ordinal relationship between the ground truth labels
into consideration when learning the ranking model.
Suppose there are $K$ ordered categories. The goal of ordinal regression is to find
a scoring function $f$ , such that one can easily use thresholds $b_1 \leq b_2 \leq\cdots \leq b_{K−1}$
to distinguish the outputs of the scoring function (i.e., $f (x_j ), j = 1, \dots , m$) into
different ordered categories.


#### Perceptron-Based Ranking (PRanking)

The goal of `PRanking` is to find a direction defined by a parameter vector $w$, after projecting the documents
onto which one can easily use thresholds to distinguish the documents into different ordered categories.

On iteration t , the learning algorithm gets an instance $x_j$ associated with query $q$.
Given $x_j$, the algorithm predicts $\hat y_j = \arg\min_{k}\{w^T x_j − b_k < 0\}$.
It then receives the ground truth label $y_j$ .
If the algorithm makes a mistake (i.e., $\hat y_j \not= y_j$ )
then there is at least one threshold, indexed by $k$,
for which the value of $w^T x_j$ is on the wrong side of $b_k$.
To correct the mistake, we need to move the values of $w^T x_j$ and $b_k$ toward each other.
After that, the model parameter $w$ is adjusted by
$w = w + x_j$ , just as in many `perceptron-based algorithms`. This process is repeated until the training process converges.

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

###  Ranking with Large Margin Principles

Given $n$ training queries $\{q_i \}^n_{i=1}$, their associated documents $\mathrm x^{(i)}=\{x_j^{(i)}\}_{j=1}^{m^{(i)}}$
and the corresponding relevance judgments $\mathrm y^{(i)}=\{y_j^{(i)}\}_{j=1}^{m^{(i)}}$, the learning process is defined below, where the adoption of a linear scoring function is assumed.
The constraints basically require every document to be correctly classified into its target ordered category, i.e., for documents in category $k$,
$w^T x_j^{(i)}$ should exceed threshold
$b_{k−1}$ but be smaller than threshold $b_k$ , with certain soft margins (i.e., $1-\epsilon_{j,k-1}^{m^{(i)}}$ and $1-\epsilon_{j,k}^{m^{(i)}}$, respectively.
The term $\frac{1}{2}\|w\|^2$ controls the complexity of model with parameter $w$.

$$\min \frac{1}{2}\|w\|^2 +\lambda \sum_{i=1}^{n}\sum_{j=1}^{m^{(i)}}\sum_{k=1}^{K-1}(\epsilon_{j,k}^{(i)}+\epsilon_{j,k}^{(i)\ast}\\
s.t. w^T x_{j}^{(i)} -b_k \leq -1 + \epsilon_{j,k}^{m^{(i)}}, \text{if $y_j^{(i)}=k$},\\
w^T x_{j}^{(i)} -b_k \geq 1 - \epsilon_{j,k}^{(i)\ast}, \text{if $y_j^{(i)}=k+1$}, \\
\epsilon_{j,k}^{j}\geq 0, \quad \epsilon_{j,k}^{(i)\ast} \geq 0, \\
j = 1, \cdots, m^{(i)}, i=1,\dots, n, k=1,\dots, K-1.
$$

Another strategy is called the sum-of-margins strategy.
 In this strategy, some
additional thresholds $a_k$ are introduced, such that for category k, $b_{k-1}$ is its lower-bound threshold and $a_k$ is its upper-bound threshold.

Accordingly, the constraints become that for documents in category k,$w^T x_{j}^{(i)}$ should exceed threadhold  $b_{k-1}$ but be smaller than threshold $a_k$, with certain soft margins (i.e., $\epsilon_{j,k}^{(i)\ast}$ and $\epsilon_{j,k}^{(i)}$) respectively.

The corresponding learning process can be expressed as follows,
from which we can see that the margin term $\sum_{k=1}^{K}(a_k -b_k)$ really has the meaning of `margin`.
$$\min  \sum_{k=1}^{K}(a_k -b_k)+\lambda \sum_{i=1}^{n}\sum_{j=1}^{m^{(i)}}\sum_{k=1}^{K-1}(\epsilon_{j,k}^{(i)}+\epsilon_{j,k}^{(i)\ast}\\
s.t.  a_k\leq b_k\leq a_{k+1}\\
w^T x_{j}^{(i)} \leq a_k + \epsilon_{j,k}^{(i)}, \text{if $y_j^{(i)}=k$},\\
w^T x_{j}^{(i)}  \geq b_k - \epsilon_{j,k}^{(i)\ast}, \text{if $y_j^{(i)}=k+1$}, \\
\|w\|^2\leq 1, \quad \epsilon_{j,k}^{j}\geq 0, \quad \epsilon_{j,k}^{(i)\ast} \geq 0, \\
j = 1, \cdots, m^{(i)}, i=1,\dots, n, k=1,\dots, K-1.
$$



### The Pairwise Approach

In the pairwise approach, ranking is usually reduced to a classification on document pairs, i.e., to determine which document in a pair is preferred.
That is, the goal of learning is to minimize the number of miss-classified document pairs.
In the extreme case, if all the document pairs are correctly classified, all the documents will be correctly ranked.
Note that this classification differs from the classification in the pointwise approach, since it operates on every two documents under investigation.
A natural concern is that document pairs are not independent, which violates the basic assumption of classification.
The fact is that although in some cases this assumption really does not hold, one can still use classification technology to learn the ranking model.
However, a different theoretical framework is needed to analyze the generalization of the learning process.

---|The Pairwise Approach
---|----
Input Space | Document pairs $x_u, x_v$
Output Space | Preference $y_{u, v}\in \{+1,-1\}$
Hypothesis Space | Preference function $h$
Loss Function | $L(h; x_u, x_v, y_{u,v})$

#### Margin-based Ranking

The algorithm is a modification of `RankBoost`, analogous to “approximate coordinate ascent boosting.”

The margin of ranking function $f$, is defined to be the minimum margin over all crucial pairs,
$$\min_{\{i,k\mid \pi(x_i, x_k)=1\}} f(x_i)-f(x_k)$$

where The values of the truth function $\pi : X \times X \to \{0,1\}$, which is defined over pairs of instances, are
analogous to the “labels” in classification. If $\pi(X_1, X_2)=1$, this means that the pair $X_1, X_2$ is a crucial pair: $X_1$ should be ranked more highly than $X_2$.

Intuitively, the margin tells us how much the ranking function can change before one of the crucial pairs is mis-ranked.

- https://www.shivani-agarwal.net/Teaching/E0371/
- http://rob.schapire.net/
- [Margin-based Ranking and an Equivalence between AdaBoost and RankBoost](http://rob.schapire.net/papers/marginranking.pdf)
- https://www.cs.princeton.edu/~schapire/papers/rankboostmiddle.pdf
- [Efficient Margin-Based Rank Learning Algorithms for Information Retrieval](https://link.springer.com/chapter/10.1007/11788034_12)
- [Ranking with Large Margin Principle: Two Approaches](https://papers.nips.cc/paper/2269-ranking-with-large-margin-principle-two-approaches)



#### RankNet

> `RankNet` is a feed-forward neural network model. Before it can be used its parameters must be learned using a large amount of labeled data, called the training set. The training set consists of a large number of query/document pairs, where for each pair, a number assessing the quality of the relevance of the document to that query is assigned by human experts. Although the labeling of the data is a slow and human-intensive task, training the net, given the labeled data, is fully automatic and quite fast. The system used by Microsoft in 2004 for training the ranker was called The Flying Dutchman. from  [RankNet: A ranking retrospective](https://www.microsoft.com/en-us/research/blog/ranknet-a-ranking-retrospective/).

`RankNet` takes the ranking  as **regression** task.

<img title="RankNet" src="http://web.ist.utl.pt/~catarina.p.moreira/images/ranknet.png"  width="80%" />

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

It is not difficult to verify that the cross entropy loss is an upper bound of the pairwise 0–1 loss.

See more deep learning algorithms on ranking  at [https://github.com/Isminoula/DL-to-Rank] or [http://quickrank.isti.cnr.it/research-papers/].

* https://www.ethanrosenthal.com/2016/11/07/implicit-mf-part-2/
* http://sonyis.me/paperpdf/wsdm233-song-2014.pdf

#### FRank: Ranking with a Fidelity Loss

A new loss function named the fidelity loss is proposed in **: Frank: a ranking method with fidelity loss**, which is defined as:
$$L(f; x_u, x_v, y_{u, v})=1-\sqrt{\hat P_{u, v}P_{u, v}(f)}-\sqrt{(1-\hat P_{u, v})(1-P_{u, v})}.$$

The fidelity was originally used in quantum physics to measure the difference
between two probabilistic states of a quantum.
By comparing the fidelity loss with the cross entropy loss, we can see that the fidelity loss is bounded between 0 and 1,
and always has a zero minimum. These properties are nicer than those of the cross entropy loss.
On the other hand, however, while the cross entropy loss is convex, the fidelity loss becomes non-convex.
Such a non-convex objective is more difficult to optimize and one needs to be careful when performing the optimization.
Furthermore, the fidelity loss is no longer an upper bound of the pairwise
0–1 loss.

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
* [RankSVM原理](https://x-algo.cn/index.php/2016/08/09/ranksvm/)
* [Learning to Rank算法介绍：RankSVM 和 IR SVM](https://www.cnblogs.com/bentuwuying/p/6683832.html)
* [Support Vector Machine for Ranking Author: Thorsten Joachims](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html)
* [Ranking SVM for Learning from Partial-Information Feedback](http://www.cs.cornell.edu/people/tj/svm_light/svm_proprank.html)
* [SVM-based Modelling with Pairwise Transformation for Learning to Re-Rank](http://alt.qcri.org/ecml2016/unocanda_cameraready.pdf)
* [svmlight](http://svmlight.joachims.org/)

### LambdaRank

**LambdaRank**  introduced the $\lambda_i$ when update of parameters $w$ of the model $f:\mathbb{R}^P\to\mathbb{R}$.
The key observation of LambdaRank is thus that in order to train a model, we do not need the costs themselves: we only need the gradients (of the costs with respect to
the model scores).

You can think of these gradients as little arrows attached to each document in the ranked list, indicating which direction we’d like those documents to move. LambdaRank simply took the RankNet gradients, which we knew worked well, and scaled them by the change in NDCG found by swapping each pair of documents. We found that this training generated models with significantly improved relevance (as measured by NDCG) and had an added bonus of uncovering a further trick that improved overall training speed (for both RankNet and LambdaRank). Furthermore, surprisingly, we found empirical evidence (see also this paper) that the training procedure does actually optimize NDCG, even though the method, although intuitive and sensible, has no such guarantee.

We compute the  gradients of RankNet by:
$$
\frac{\partial L}{\partial w} = \sum_{(i, j)}\frac{\partial L_{i, j}}{\partial w}=\sum_{(i, j)}[\frac{\partial L_{i, j}}{\partial s_i}+\frac{\partial L_{i,j}}{\partial s_j}].
$$

Observe that
$$\frac{\partial L_{i, j}}{\partial s_i} = -\frac{\partial L_{i,j}}{\partial s_j}$$
and define

$$
{\lambda}_{i,j}=\frac{\partial L_{i, j}}{\partial s_i} = -\frac{\partial L_{i, j}}{\partial s_j} = -\frac{\sigma}{1 + \exp(\sigma(s_i - s_j))}.
$$

What is more, we can extend it to

$$
{\lambda}_{i,j}=  -\frac{\sigma}{1+\exp(\sigma( s_i -s_j))}|\Delta Z|,
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

* https://github.com/wepe/efficient-decision-tree-notes

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
- [ ] https://www.cnblogs.com/genyuan/p/9788294.html

### Unbiased LambdaMART

Unbiased LambdaMART can jointly estimate the biases at click positions and the biases at unclick positions, and learn an unbiased ranker.

- http://www.hangli-hl.com/uploads/3/4/4/6/34465961/unbiased_lambdamart.pdf
- [Learning to Rank with Selection Bias in Personal Search](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45286.pdf)

### The Listwise Approach

The listwise approach takes the entire set of documents associated with a query in the training data as the input and predicts their ground truth labels.
Note that the listwise approach assumes that the ground truth labels are given in terms of permutations,
while the judgments might be in other forms (e.g., relevance degrees or pairwise preferences).

---|The Listwise Approach|---
---|---|---|
---|Listwise Loss Minimization | Direct Optimization of IR Measure
Input Space| Document set $\mathrm X=\{x_1, x_2, \dots, x_m\}$ |Document set $\mathrm X=\{x_1, x_2, \dots, x_m\}$
Output Space | Permutation |Ordered categories $\mathrm Y=\{y_1, y_2, \dots, y_m\}$
Hypothesis Space | $h(\mathrm X) = sort\circ f(\mathrm X)$ | $h(\mathrm X)=f (\mathrm X)$
Loss Function | Listwise loss| 1-surrogate measure



#### SoftRank

In `SoftRank`, it is assumed that the ranking of the documents is not simply determined by sorting according to the scoring function.
Instead, it introduces randomness to the ranking process by regarding the real score of a document as a random variable whose mean is the score given by the scoring function.
In this way, it is possible that a document can be ranked at any position, of course with different probabilities. For each such possible ranking, one can compute an NDCG value.
Then the expectation of NDCG over all possible rankings can be used as an smooth approximation of the original evaluation measure NDCG.
The detailed steps to achieve this goal are elaborated as follows.


The random variable is governed by a Gaussian distribution whose variance is σs and mean is $f(x_j)$, the original score outputted by the scoring function.
That is,
$$p(s_j)=N(s_j\mid f(x_j),\sigma_s^2).$$

Due to the randomness in the scores, every document has the probability of being ranked at any position.
Specifically, based on the score distribution, the probability of a document $x_u$ being ranked before another document $x_v$ can be deduced as follows:

$$P_{u, v}=\int_{0}^{\infty}N(s\mid f(x_u)-f(x_v), 2\sigma_s^2)\mathrm d s.$$

Suppose we already have document $x_j$ in the ranked list, when adding document $x_u$, if document $x_u$ can beat $x_j$ the rank of $x_j$ will be increased by one.
Otherwise the rank of $x_j$ will remain unchanged. Mathematically, the probability of $x_j$ being ranked at position r (denoted as $P_j(r)$) can be computed as follows:
$$P_{j}^{(u)}=P_{j}^{(u-1)}(r-1)P_{u,j}+P_{j}^{(u-1)}(r)(1-P_{u,j}).$$

Third, with the rank distribution, one computes the expectation of NDCG (referred to as SoftNDCG2), and use (1 − SoftNDCG) as the loss function in learning to rank:
$$L(f;\mathrm{x, y})=1-\frac{1}{Z_m}\sum_{j=1}^{m}(2^{y_j}-1)\sum_{i=1}^{m}\eta(r)P_j(r).$$

#### Plackett-Luce Model Ranking induced by pair-comparisons

In learning to rank, each training sample has been labeled with a relevance score, so the
ground-truth permutation of documents related to the $i$ th query can be easily obtained and
denoted as $\pi_i$, where $\pi(j)$ denotes the index of the document in the $j$ th position of the
ground-truth permutation. We note that $\pi_i$ is not obligatory to be a full rank, as we may
only care about the top K documents.

Consider a ranking function with linear features, the probability of a set of candidate relevant documents $D_i$ associated with a query $q_i$ is defined as
$$p(d_e^i)=\frac{\exp{h(d_e^i)^T w}}{\sum_{d\in D_i } \exp{h(d_e^i)^T w}}$$

The probability of the Plackett-Luce model to generate a rank $\pi_i$ is given as
$$p(\pi_i, w)=\prod_{j=1}^{\|\pi_i\|} \underbrace{p(d_{\pi_{i}(j)}^i\mid C_{i,j})}_{???}\\
p(d_e^i\mid C_{i,j})=p(d)$$

where $C_{i,j}=D_i - \{d_{\pi_{i}(1)}^{i},\dots, d_{\pi_{i}(j-1)}^{i}\}$.

The training objective is to maximize the log-likelihood of all expected ranks over
all queries and retrieved documents with corresponding ranks in the training data with a zero-mean and unit-variance Gaussian prior parameterized by $w$.
$$L=\log\{\prod_{i}p(\pi_i, w)\}-\frac{1}{2} {\|w\|}_2^2.$$

The gradient can be calculated as follows
$$\frac{\partial L}{\partial w}=\sum_{i}\sum_{j}\{h(d_{\pi_{i}(j)}^{i})-\sum_{d\in C_{i, j}} w\}$$

* https://hturner.github.io/PlackettLuce/
* https://blog.csdn.net/fx677588/article/details/52636170
* [Bayesian inference for Plackett-Luce ranking models](https://www.shivani-agarwal.net/Teaching/E0371/Papers/icml09-bayesian-plackett-luce-ranking.pdf)
* https://hturner.github.io/PlackettLuce/articles/Overview.html

#### ListNet

Given the ranking scores of the documents outputted by the scoring function $f$ (i.e., $s = \{s_j \}^m_{j=1}$, where $s_j = f (x_j))$, the `Plackett–Luce` model defines a probability for each possible permutation $\pi$ of the documents, based on the chain rule, as follows:
$$P(\pi\mid s)=\prod_{j=1}^{m}\frac{\phi(s_{\pi^{-1}(j)})}{\sum_{u=1}^{m}\phi(s_{\pi^{-1}(u)})}$$

where $\pi^{−1}(j)$ denotes the document ranked at the j th position of permutation $\pi$
and $\phi$ is a transformation function, which can be linear, exponential, or sigmoid.


Please note that the `Plackett–Luce` model is scale invariant and translation invariant in certain conditions. For example, when we use the exponential function as the
transformation function, after adding the same constant to all the ranking scores,
the permutation probability distribution defined by the `Plackett–Luce` model will not change.
When we use the linear function as the transformation function, after multiplying all the ranking scores by the same constant, the permutation probability distribution will not change.
These properties are quite in accordance with our intuitions on ranking.

With the `Plackett–Luce model`, for a given query $q$, ListNet first defines the permutation probability distribution based on the scores given by the scoring function $f$.
Then it defines another permutation probability distribution $P_y(\pi)$ based on
the ground truth label.

For the next step, ListNet uses the K-L divergence between the probability distribution for the ranking model
and that for the ground truth to define its loss function (denoted as the K-L divergence loss for short).
$$L(f; \mathrm x, \Omega_y)=D( P_y(\pi)\mid\mid P(\pi\mid (f(w, \mathrm x))) ).$$

#### ListMLE

ListMLE is also based on the Plackett–Luce model. For each query q, with the
permutation probability distribution defined with the output of the scoring function, it uses the negative log likelihood of the ground truth permutation as the loss function.
We denote this new loss function as the likelihood loss for short:
$$L(f;\mathrm{x},\pi_y)=-\log P(\pi_y\mid f(w, \mathrm x)).$$

It is clear that in this way the training complexity has been greatly reduced as compared to ListNet,
since one only needs to compute the probability of a single permutation $\pi_y$ but not all the permutations.
Once again, it can be proven that this loss function is convex, and therefore one can safely use a gradient descent method to optimize the loss.

### Cascade Ranking Models

In the information retrieval community, explorations in effectiveness and efficiency have been largely disjoint. This is problematic in that a piecemeal approach may yield ranking models that are impractically slow on web-scale collections or algorithmic optimizations that sacrifice quality to an unacceptable degree. The aim of our work is to develop an integrated framework to building search systems that are both effective and efficient. To this end, we have been exploring a research program, dubbed "learning to efficiently rank", that allows algorithm designers to capture, model, and reason about tradeoffs between effectiveness and efficiency in a unified machine-learning framework.

[Our core idea is to consider the ranking problem as a "cascade", where ranking is broken into a finite number of distinct stages. Each stage considers successively richer and more complex features, but over successively smaller candidate document sets. The intuition is that although complex features are more time-consuming to compute, the additional overhead is offset by examining fewer documents. In other words, the cascade model views retrieval as a multi-stage progressive refinement problem, where each stage balances the cost of exploiting various features with the potential gain in terms of better results. We have explored this notion in the context of linear models and tree-based models.](http://lintool.github.io/NSF-projects/IIS-1144034/)

- https://culpepper.io/publications/gcbc19-wsdm.pdf
- http://zheng-wen.com/Cascading_Bandit_Paper.pdf
- http://lintool.github.io/NSF-projects/IIS-1144034/
- https://www.nsf.gov/awardsearch/showAward?AWD_ID=1144034
- [Ivory: A Hadoop toolkit for web-scale information retrieval research](http://lintool.github.io/Ivory/)

### Relevance Feedback

* After initial retrieval results are presented, allow the user to provide feedback on the relevance of one or more of the retrieved documents.
* Use this feedback information to reformulate the query.
* Produce new results based on reformulated query.
* Allows more interactive, multi-pass process.

### AdaRank

In the abstract, it  claims that:
> Ideally a learning algorithm would train a ranking model that could directly optimize the performance measures with respect to the training data.
> Existing methods, however, are only able to train ranking models by minimizing loss functions loosely related to the performance measures.
> For example, Ranking SVM and RankBoost train ranking models by minimizing classification errors on instance pairs.
> To deal with the problem, we propose a novel learning algorithm within the framework of boosting,
> which can minimize a loss function directly defined on the performance measures.
> Our algorithm, referred to as AdaRank, repeatedly constructs 'weak rankers' on the basis of reweighted training data
> and finally linearly combines the weak rankers for making ranking predictions.
>We prove that the training process of AdaRank is exactly that of enhancing the performance measure used.
***
* **Input**: the set of documents associated with each query
* **Given**: initial distribution  $D_1$ on input queries
* **For** $t=1,\dots, T$
  * Train weak ranker $f_t$ based on distribution $D_t$.
  * Choose $\alpha_t$
  * Update $D_{t+1}( x_u^{(i)}, x_v^{(i)} )=D_{t}( x_u^{(i)}, x_v^{(i)} )\exp(\alpha_t ( f_t(x_u^{(i)}) - f_t(x_u^{(i)}) )$ and then renormalize it.
* **Output**:$f(x)=\sum_{t}\alpha_t f_t(x)$.

- [An Efficient Boosting Algorithm for Combining Preferences](http://jmlr.csail.mit.edu/papers/volume4/freund03a/freund03a.pdf)
- [Concave Learners for Rankboost](http://www.jmlr.org/papers/volume8/melnik07a/melnik07a.pdf)
- [Python implementation of the AdaRank algorithm](https://github.com/rueycheng/AdaRank)
- [AdaRank: a boosting algorithm for information retrieval](https://dl.acm.org/citation.cfm?id=1277809)

### RankBoost

The method of RankBoost adopts AdaBoost for the classification over document pairs.

* **Input**: training data in terms of document pairs
* **Given**: initial distribution $D_1$ on input document pairs
* For $t=1,\dots, T$
  * Train weak ranker $f_t$ based on distribution $D_t$.
  * Choose $\alpha_t=\frac{1}{2}\log\frac{\sum_{i=1}^{n} D_t(i) (1 + M(f_t, x^{(i)}, y^{(i)}))}{\sum_{i=1}^{n} D_t(i) (1 - M(f_t, x^{(i)}, x^{(i)}))}$
  * Update $D_{t+1}(i)=\frac{\exp(-M(\sum_{s=1}^t \alpha_s f_s, x^{(i)}, y^{(i)}))}{\sum_{j=1}^n \exp(-M(\sum_{s=1}^t \alpha_s f_s, x^{(j)}, y^{(j)}))}$ and then renormalize it.
* **Output**: $f(x)=\sum_{t}\alpha_t f_t(x)$.


Here $M(f, x, y)$ represents the evaluation measure.
Due to the analogy to AdaBoost, AdaRank can focus on those hard queries and progressively minimize $1 −M(f, x, y)$.


----

Given a query-document pair $(q, d_i)$, represented by a feature vector $\mathrm{x}$,
a LtR model based on an additive ensemble of regression trees predicts a relevance score $s(\mathrm x)$ used for ranking a set of documents.
Typically, a tree ensemble encompasses several binary decision trees, denoted by $T = {T_0, T_1, \dots}$.
Each internal (or branching) node in $T_h$ is associated with a Boolean test over a specific feature $f_{\phi}\in \mathcal{F}$, and a constant threshold $\gamma\in\mathbb{R}$.
Tests are of the form $x[\phi] \leq \gamma$, and, during the visit, **the `left branch` is taken iff the test `succeeds`.**
Each leaf node stores the tree prediction, representing the potential contribution of the tree to the final document score.
The scoring of $\mathrm{x}$ requires the traversal of all the ensemble’s trees and it is computed as a _weighted sum_ of all the tree predictions.
Such leaf node is named exit leaf and denoted by $e(x)$.

### YetiRank and MatrixNet

`PageRank, LambdaRank, MatrixNet` is under the support of commercial firms *Google, Microsoft, Yandex*, respectively. The practical ranking algorithms in the search engines are the key to search engine optimization.
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

+ [YetiRank: Everybody Lies](http://download.yandex.ru/company/ICML2010-kuralenok.pdf)
+ [Yandex employees won the competition for the best search](https://weekly-geekly.github.io/articles/97689/index.html)
+ [Winning The Transfer Learning Track of Yahoo!’s Learning To Rank Challenge with YetiRank](http://proceedings.mlr.press/v14/gulin11a/gulin11a.pdf)
+ [MatrixNet: New Level of Search Quality](https://yandex.com/company/technologies/matrixnet/)
+ [The Ultimate Guide To Yandex Algorithms](https://salt.agency/blog/the-ultimate-guide-to-yandex-algorithms/)
+ https://yandex.com/

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
- [ ] [Boosted Ranking Models: A Unifying Framework for Ranking Predictions](http://www.cs.cmu.edu/~kdelaros/kais2011.pdf)

### X-CLEaVER

[X-CLEaVER interleaves the iterations of a given gradient boosting learning algorithm with pruning and re-weighting phases. First, redundant trees are removed from the given ensemble, then the weights of the remaining trees are fine-tuned by optimizing the desired ranking quality metric. We propose and analyze several pruning strategies and we assess their benefits showing that interleaving pruning and re-weighting phases during learning is more effective than applying a single post-learning optimization step.](https://dl.acm.org/citation.cfm?doid=3289398.3205453)

- https://dl.acm.org/citation.cfm?doid=3289398.3205453

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

To explicitly study the unified effect of group preference and individual preference,  
we combine them linearly: $\hat{r}_{G, u,i} = \rho \hat{r}_{G,i} + (1-\rho)\hat{r}_{u,i}$.

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

In logistic function, the value of "a" determines the shape of the function. In other words, it tells how close the approximation of logistic function to the zero-one loss.
However, in the context of matrix factorization, the change of ${U_u}$ doesn’t necessarily
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
pre
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



### Borda Count

`Borda Count` is an unsupervised method for ranking aggregation.  Aslam & Montague propose employing Borda Count in meta search.
In such case, Borda Count ranks documents in the final ranking based on their positions in the basic rankings.
More specifically, in the final ranking, documents are sorted according to the numbers of documents that are ranked below them in the basic rankings.
If a document is ranked high in many basic rankings, then it will be ranked high in the final ranking list.

The ranking scores of documents in the final ranking $S_D$ are calculated as
$$S_D=F(\Sigma)=\sum_{i=1}^{K}S_i$$
$$S_i=(s_{i, 1},\cdots, s_{i,j},\cdots, S_{i, n})^T,
\\ s_{i,j}=n-\sigma_{i}(j)$$

where $s_{i,j}$ denotes the number of documents ranked behind $j$ in basic ranking $\sigma_i, \sigma_i(j)$ denotes the rank of document $j$ in basic ranking $\sigma_i$ and $n$ denotes the number of documents.

Unfortunately, the Borda method has one serious drawback in that it is easily manip- ulated. That is, dishonest voters can purposefully collude to skew the results in their favor.

### Cranking

Cranking employs the following probability model

$$P(\pi\mid \theta,\Sigma)=\frac{1}{Z(\theta, \Sigma)}\exp(\sum_{j=1}^{k}\theta_j\cdot d(\pi, \sigma_j))\propto \exp(\sum_{j=1}^{k}\theta_j\cdot d(\pi, \sigma_j))$$

where $\pi$ denotes the final ranking, $\Sigma=(\sigma_1,\cdots,\sigma_k)$ denotes the basic rankings, $d$ denotes the distance between two rankings, and $\theta$ denotes weight parameters.
Distance $d$ can be, for example, Keddal's Tau. Furthermore, $Z$ is the normalizing factor over all the possible rankings defined as $Z(\theta,\Sigma)=\sum_{\pi}\exp(\sum_{j=1}^{k}\theta_j\cdot d(\pi, \sigma_j))$.

It is an extension of ` Mallows model ` in statistics.

In learning, the training data is given as $S=\{(\Sigma_i, \pi_i)\mid i=1, \cdots, m\}$, and the goal is to build the model for ranking aggregation based on the data.
We can consider employing `Maximum Likelihood Estimation` to learn the parameters of the model.
If the final ranking and the basic ranking are all ranking lists in the training data, then the log likelihood function is calculated as follows.
$$L(\theta)=\log\prod_{i=1}^{m}P(\pi_i, \Sigma_I)=\sum_{i=1}^{m}\log\frac{\exp(\sum_{j=1}^k \theta_j\cdot d(\pi_i, \sigma_{i,j}))}{\sum_{\pi}\exp(\sum_{j=1}^{k}\theta_j\cdot d(\pi, \sigma_j))}.$$

We can employ Gradient Descent to estimate the optimal parameters.

### Probabilistic relevance model

> One way to tackle this kind of data is by means of probabilistic modeling, i.e., to learn a probability distribution over the space of ranking, and then to carry out inference tasks in order to get the required information: find the consensus ranking, find the probability that movie A is ranked higher than movie B for a particular user, etc. Unfortunately dealing with ranking data from a probabilistic point of view is a daunting task as the number of permutations of the n objects grows exponentially with n. Furthermore, in many real scenarios the 1 number of possible objects (as the number of possible output web pages of a query in a search engine) can be considered as unlimited. Recently numerous proposals have been done in the machine learning as well as in the statistical communities to create models that can approach these kind of problems. These different approaches can be mainly classified as,
(i) parametric-based approaches,
(ii) low-order statistics approaches and
(iii) non-parametric approaches.
Parametric approaches to deal with permutation data have a long history in the field of statistics as they have been used for more than thirty years. These approaches assume a functional form of the probability distribution over the set of permutations that come defined by a set of parameters. The differences between the parametric models can be related with the kind of data that motivated its discovery.

- http://www.sc.ehu.es/ccwbayes/members/ekhine/tutorial_ranking/info.html
- https://nlp.stanford.edu/IR-book/html/htmledition/probabilistic-information-retrieval-1.html
- https://www.wikiwand.com/en/Probabilistic_relevance_model
- http://www.sc.ehu.es/ccwbayes/members/ekhine/tutorial_ranking/data/slides.pdf

#### Mallows Ranking Models and Generalized Mallows models

It is one of `probabilistic ranking models`.
Mallows model  is defined as a parametric model
$$\mathbb P_{\theta, \pi_0, d}(\pi)=\frac{1}{\Phi(\theta, d)}\exp(-\theta d(\pi, \pi_0))$$
where
+ $\pi$ is a permutations of $\{1, 2, \cdots, N\}$.
+ $\theta > 0$ is the dispersion parameter.
+ $\pi_0$ is the central ranking.
+ $d(\cdot, \cdot)$ is a a discrepancy function of permutations, which is right invariant $d(\pi, d)=d(\pi\circ \sigma^{-1}, id)$ for $\pi, d$ in  permutations of $\{1, 2, \cdots, N\}$.
+ $\Phi(\theta, d)$ is the normalizing constant.

Mallows primarily considered two special cases of this model:

1. Spearman’s rho: $d(\pi, d)=\sum_{i=1}^{n}(\pi(i)-d(i))^2$,
2. Kendall’s tau: $d(\pi, d)=inv(\pi\circ d^{-1})$,

where $inv(\sigma)$ is the number of inversions of permutation $\sigma$, i.e., $inv(\sigma)$ is the number of the set $\{(i, j)\mid i<j , \sigma(i)>\sigma(j)\}$.

It showed that if the central ranking $\pi_0$ is known, the MLE of $\theta$ (or $\vec \theta$) can be easily found by convex optimization. When the number of items $n$ is large, learning a complete
ranking model becomes impracticable.

- [Mallows Ranking Models: Maximum Likelihood Estimate and Regeneration](http://proceedings.mlr.press/v97/tang19a/tang19a.pdf)
- [Consensus ranking under the exponential model](https://www.stat.washington.edu/sites/default/files/files/reports/2007/tr515.pdf)
- [Probabilistic Preference Learning with the Mallows Rank Model](http://www.jmlr.org/papers/volume18/15-481/15-481.pdf)

### HodgeRank

`HodgeRank` apply Hodge theory to the ranking problem. It is considered as an application of modern abstract mathemtics rather than as a technique in machine learning.
The problem is always to evaluate some items and rank them.
Learn a function
$$f:\mathcal X \to \mathcal Y.$$

* Data: Know $f$ on a (very small) subset $\Omega\subset \mathcal X$;
* Model: Know that $f$ belogs to some class of functions $\mathcal F(x, y)$;
* Ranking: Rank objects in some order.
  * Scoring function $f:\mathcal X \to \mathbb{R}$;
  * $f(x)> f(y)\iff x \succ y$, i.e., $x$ is preferred to $y$.

Persi Diaconis, 1987 Wald Memorial Lectures:
> A basic tenet of data analysis is this: If you’ve found some structure, take it out, and look at what’s left.
> Thus to look at second order statistics it is natural to subtract away the observed first order structure.
> This leads to a natural decomposition of the original data into orthogonal pieces.

Hodge decomposition:
$$\text{aggregate pairwise ranking}=\text{consistent}\oplus \text{loclly inconsisitent}\oplus \text{globally inconsistent}$$

Ranking data live on **pairwise comparison graph** $G(E, V)$
where $V$ is the set of alternatives, $E$ is pairs of alternatives to be compared.

The basic models optimize over model class $M$
$$\min_{X\in M}\sum_{\alpha, i, j}w_{i,j}^{\alpha}{(X_{i,j}-Y_{ij}^{\alpha})}^2$$
where
* $Y_{ij}^{\alpha}$ measures preference of $i$ over $j$ of voter $\alpha$.
* $w_{i,j}^{\alpha}$ is metric; 1 if $\alpha$ made comparison for $\{i, j\}$, 0 otherwise.

Kemeny optimization is set the condition
$$M_K =\{ X\in\mathbb{R}^{n\times n}\mid X_{ij} = sign(s_i - s_j), s: V\to \mathbb{R} \}.$$

Relaxed version:
$$M_G =\{ X\in\mathbb{R}^{n\times n}\mid X_{ij} = s_i - s_j, s: V\to \mathbb{R} \}.$$

Previous problem may be reformulated
$$\min_{X\in M_G}{\|X-\bar Y\|}_{F, W}^2=\min_{X\in M_G}\sum_{\{i, j\}\in E}w_{i,j}{(X_{i,j} - \bar Y_{ij})}^2$$

where $w_{i, j}=\sum_{\alpha}w_{i,j}^{\alpha}$ and $\bar Y_{i, j}=\sum_{\alpha}w_{i,j}^{\alpha}Y_{i,j}^{\alpha}/\sum_{\alpha}w_{i,j}^{\alpha}$.

Why not just aggregate over scores directly? Mean score is afirst order statistics and is inadequate because

- most voters would rate just a very small portion of the alternatives,
- different alternatives may have different voters, mean scores affected by individual rating scales.

**Formation of Pairwise Ranking**：Prediction

- Linear Model: average score difference between i and j over all who have rated both,
$$Y_{i,j}=\frac{\sum_{k}(X_{k,i} - X_{k, j}) }{|\{k\mid X_{k,i}, X_{k,j} exist\}|}.$$
- Log-linear Model: logarithmic average score ratio of positive scores,
$$Y_{i,j}=\frac{\sum_{k}(\log X_{k,i} - \log X_{k, j}) }{|\{k\mid X_{k,i}, X_{k,j} exist\}|}.$$
- Linear Probability Model: probability j preferred to i in excess of purely random choice,
$$Y_{i,j}=Pr(k\mid X_{k,j}>X_{k, i})-\frac{1}{2}.$$
- Bradley-Terry Model: logarithmic odd ratio (logit),
$$Y_{i,j}=\log\frac{Pr(k\mid X_{k,j} > X_{k, i})}{Pr(k\mid X_{k,j} < X_{k, i})}.$$
***

pairwise rankings that are gradient of score functions, i.e. consistent or integrable.

Global is ranking given by solution to
$$\min_{s\in C^0}{\|grad\,s-\bar Y\|}_{2,w}.$$
Minimum norm solution is
$$s^{\ast}=-{\Delta}_{0}^{\dagger} div\,\bar Y$$

Divergence is
$$div\,\bar Y(i)=\sum_{j\mid (i,j)\in E}w_{i,j}\bar Y_{i,j}.$$

Graph Laplacian is
$${[{\Delta}_{0}]}_{(i,j)}=
\begin{cases}
\sum_{i} w_{i,i}, &\text{if $i=j$};\\
- w_{(i,j)}, &\text{if $j$ such that $(i,j)\in E$};\\
0, &\text{otherwise}.
\end{cases}$$

- [ ] [Who's Number 1? Hodge Theory Will Tell Us](http://www.ams.org/publicoutreach/feature-column/fc-2012-12)
- [ ] [Statistical Ranking and Combinatorial Hodge Theory](http://repository.ust.hk/ir/Record/1783.1-80467)
- [ ] [HodgeRank on random graphs for subjective video quality assessment](http://repository.ust.hk/ir/Record/1783.1-80463)
- [ ] [HodgeRank with Information Maximization for Crowdsourced Pairwise Ranking Aggregation](http://repository.ust.hk/ir/Record/1783.1-90160)
- [ ] [HodgeRank with Information Maximization for Crowdsourced Pairwise Ranking Aggregation](https://arxiv.org/abs/1711.05957)
- [ ] http://www.stat.uchicago.edu/~lekheng/
- [ ] http://www.math.pku.edu.cn/teachers/yaoy/
- [ ] [AML08: Algebraic Methods in Machine Learning](http://www.gatsby.ucl.ac.uk/~risi/AML08/)
- [ ] [Graph Helmholtzian and Rank Learning](http://www.gatsby.ucl.ac.uk/~risi/AML08/lekhenglim-nips.pdf)

### Differentiable Ranking and Sorting

[Sorting is used pervasively in machine learning, either to define elementary algorithms, such as k-nearest neighbors (k-NN) rules, or to define test-time metrics, such as top-n classification accuracy or ranking losses. Sorting is however a poor match for the end-to-end, automatically differentiable pipelines of deep learning. Indeed, sorting procedures output two vectors, neither of which is differentiable: the vector of sorted values is piecewise linear, while the sorting permutation itself (or its inverse, the vector of ranks) has no differentiable properties to speak of, since it is integer-valued. We propose in this paper to replace the usual $\texttt{sort}$ procedure with a differentiable proxy. Our proxy builds upon the fact that sorting can be seen as an optimal assignment problem, one in which the  values to be sorted are matched to an $\emph{auxiliary}$ probability measure supported on any $\emph{increasing}$ family of  target values. From this observation, we propose extended rank and sort operators by considering optimal transport (OT) problems (the natural relaxation for assignments) where the auxiliary measure can be any weighted measure supported on  increasing values, where $m\not= n$. We recover differentiable operators by regularizing these OT problems with an entropic penalty, and solve them by applying Sinkhorn iterations. Using these smoothed rank and sort operators, we propose differentiable proxies for the classification 0/1 loss as well as for the quantile regression loss.]()

- [Differentiable Ranking and Sorting using Optimal Transport](http://papers.nips.cc/paper/8910-differentiable-ranking-and-sorting-using-optimal-transport)

### Online Learning to Rank

During the past 10-15 years offline learning to rank has had a tremendous influence on information retrieval, both scientifically and in practice. Recently, as the limitations of offline learning to rank for information retrieval have become apparent, there is increased attention for online learning to rank methods for information retrieval in the community. Such methods learn from `user interactions` rather than from a set of labeled data that is fully available for training up front. The time is right for an intermediate-level tutorial on online learning to rank.

- https://staff.fnwi.uva.nl/m.derijke/talks-etc/online-learning-to-rank-tutorial/
- [Online Learning to Rank with Features](http://proceedings.mlr.press/v97/li19f.html)
- [Balancing Exploration and Exploitation in Listwise and Pairwise Online Learning to Rank for Information Retrieval](https://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/hofmannirj13.pdf)
- [Optimizing Ranking Models in an Online Setting](https://arxiv.org/abs/1901.10262)

#### Deep Online Ranking System

DORS is designed and implemented in a three-level novel architecture, which includes (1) candidate retrieval; (2) learning-to-rank deep neural network (DNN) ranking; and (3) online re-ranking via multi-arm bandits (MAB).

- [ ] https://zhuanlan.zhihu.com/p/57056588
- [ ] [A Practical Deep Online Ranking System in E-commerce Recommendation](http://www.ecmlpkdd2018.org/wp-content/uploads/2018/09/723.pdf)
- [ ] [European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases](http://www.ecmlpkdd2018.org/)
- [ ] [大众点评基于知识图谱的深度学习排序](https://tech.meituan.com/2019/01/17/dianping-search-deeplearning.html)
- [ ] [Log-Linear Models for Label Ranking](https://academic.microsoft.com/#/detail/2149166361)
- [ ] [International Conference on Web Search and Data Mining](http://www.wsdm-conference.org/2019/acm-proceedings.php)
- [ ] [Learning to Rank with Deep Neural Networks](https://github.com/Isminoula/DL-to-Rank)

#### Unbiased Learning to Rank

[Implicit feedback (e.g., user clicks) is an important source of data for modern search engines. While heavily biased, it is cheap to collect and particularly useful for user-centric retrieval applications such as search ranking. To develop an unbiased learning-to-rank system with biased feedback, previous studies have focused on constructing probabilistic graphical models (e.g., click models) with user behavior hypothesis to extract and train ranking systems with unbiased relevance signals. Recently, a novel counter- factual learning framework that estimates and adopts examination propensity for unbiased learning to rank has attracted much attention. Despite its popularity, there is no systematic comparison of the unbiased learning-to-rank frameworks based on counterfactual learning and graphical models. In this tutorial, we aim to provide an overview of the fundamental mechanism for unbiased learning to rank. We will describe the theory behind existing frameworks, and give detailed instructions on how to conduct unbiased learning to rank in practice.](https://www.cikm2018.units.it/tutorial8.html)

- [Learning to Rank in theory and practice: FROM GRADIENT BOOSTING TO NEURAL NETWORKS AND UNBIASED LEARNING](http://ltr-tutorial-sigir19.isti.cnr.it/)
- https://dl.acm.org/citation.cfm?id=3334824
- http://ltr-tutorial-sigir19.isti.cnr.it/program-overview/
- [Unbiased Learning to Rank: Theory and Practice Half-day tutorial](https://www.cikm2018.units.it/tutorial8.html)
- [Unbiased Learning to Rank: Counterfactual and Online Approaches](https://arxiv.org/abs/1907.07260)

____
+ [Adversarial and reinforcement learning-based approaches to information retrieval](https://www.microsoft.com/en-us/research/blog/adversarial-and-reinforcement-learning-based-approaches-to-information-retrieval/)
+ [Cross Domain Regularization for Neural Ranking Models Using Adversarial Learning](https://www.microsoft.com/en-us/research/publication/cross-domain-regularization-neural-ranking-models-using-adversarial-learning/)
+ [Adversarial Personalized Ranking for Recommendation](http://bio.duxy.me/papers/sigir18-adversarial-ranking.pdf)

<img title="adversial IR" src="https://www.microsoft.com/en-us/research/uploads/prod/2018/06/adversarial.png" width="50%">

**RankGAN**

<img title="RankGAN" src="https://x-algo.cn/wp-content/uploads/2018/04/WX20180409-223208@2x-768x267.png" width="80%"/>

- https://x-algo.cn/index.php/2018/04/09/rankgan/
- [IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval](https://arxiv.org/pdf/1705.10513.pdf)
- [Adversarial Ranking for Language Generation](http://papers.nips.cc/paper/6908-adversarial-ranking-for-language-generation)
- https://zhuanlan.zhihu.com/p/53691459
- https://zhuanlan.zhihu.com/p/55036597
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
- [ ] [Neural Collaborative Ranking](https://arxiv.org/abs/1808.04957v1)
- [ ] [Decoupled Collaborative Ranking](https://www.researchgate.net/publication/315874080_Decoupled_Collaborative_Ranking)
- [ ] [Large-scale Collaborative Ranking in Near-Linear Time](http://www.stat.ucdavis.edu/~chohsieh/rf/KDD_Collaborative_Ranking.pdf)
- [ ] [Preference Completion: Large-scale Collaborative Ranking from Pairwise Comparisons](http://proceedings.mlr.press/v37/park15.html)
- [ ] [Local Collaborative Ranking](https://ai.google/research/pubs/pub42242)
- [ ] [SQL-Rank: A Listwise Approach to Collaborative Ranking](http://proceedings.mlr.press/v80/wu18c.html)
- [ ] [VSRank: A Novel Framework for Ranking-Based  Collaborative Filtering](http://users.jyu.fi/~swang/publications/TIST14.pdf)
- [ ] [Machine Learning: recommendation and ranking](https://jhui.github.io/2017/01/15/Machine-learning-recommendation-and-ranking/)
- [ ] [Recommender systems & ranking](https://sites.google.com/view/chohsieh-research/recommender-systems)
- [ ] [Recommendation and ranking by Mark Jelasity](http://www.inf.u-szeged.hu/~jelasity/ddm/graphalgs.pdf)
- [ ] ["Tutorial ：Learning to Rank for Recommender Systems" by](http://www.slideshare.net/kerveros99/learning-to-rank-for-recommender-system-tutorial-acm-recsys-2013)
- [ ] [Rank and Relevance in Novelty and Diversity Metrics for Recommender Systems](http://ir.ii.uam.es/predict/pubs/recsys11-vargas.pdf)

For item recommendation tasks, the accuracy of a recommendation model is usually evaluated using the `ranking metrics`.

* [Ranking Evaluation](http://csse.szu.edu.cn/staff/panwk/recommendation/OCCF/RankingEvaluation.pdf)
* http://fastml.com/evaluating-recommender-systems/

### Loss Functions in Ranking

#### LambdaLoss

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

#### Essential Loss

We show that the loss functions of these methods are upper bounds of the measure-based ranking errors. As a result, the minimization of these loss functions will lead to the maximization of the ranking measures. The key to obtaining this result is to model ranking as a sequence of classification tasks, and define a so-called essential loss for ranking as the weighted sum of the classification errors of individual tasks in the sequence. We have proved that the essential loss is both an upper bound of the measure-based ranking errors, and a lower bound of the loss functions in the aforementioned methods. Our proof technique also suggests a way to modify existing loss functions to make them tighter bounds of the measure-based ranking errors. Experimental results on benchmark datasets show that the modifications can lead to better ranking performances, demonstrating the correctness of our theoretical analysis.

First, we propose an alternative representation of the labels of objects (i.e., multi-level ratings). The
basic idea is to construct a permutation set, with all the permutations in the set being consistent with
the labels. The definition that a permutation is consistent with multi-level ratings is given as below.
> Given multi-level ratings $\mathcal L$ and permutation $y$, we say $y$ is consistent with $\mathcal L$, if for
$\forall i, s \in \{1, \dots, n\}$ satisfying $i < s$, we always have $l(y(i)) \geq l(y(s))$, where $y(i)$ represents the index
of the object that is ranked at the i-th position in $y$. We denote $Y_{\mathcal L} = \{\text{y} | \text{y is consistent with $\mathcal L$}\}$.

Second, given each permutation $y \in Y_{\mathcal L}$, we decompose the ranking of objects $\mathrm x$ into several sequential steps. For each step s, we distinguish $x_{y(s)}$, the object ranked at the s-th position in $y$, from
all the other objects ranked below the s-th position in $y$, using ranking function $f$.
Specifically, we denote $\mathrm x_{(s)} = \{x_{y(s)}, \cdots, x_{y(n)}\}$ and define a classifier based on $f$, whose target output is $y(s)$,
$$T_f(\mathrm x_{(s)})=\arg\max_{i\in\{y(s), y(s+1), \dots, y(n)\}}f(x_j).$$
The 0-1 loss for this classification task can be written as follows, where the second equality is based on the definition of $T_f$,
$$l_s(f;\mathrm X_{(s)}, y(s))=\mathbb I_{\{T_f(\mathrm x_{(s)}\not=y(s))\}}=1 -\prod_{i=s+1}^{n}\mathbb I_{\{f(x_{y(s)})> f(x_{y(i)})\}}.$$

Third, we assign a non-negative weight $\beta(s), s=1,2,\cdots, n$ to the classification task at the
s-th step, representing its importance to the entire sequence. We compute the weighted sum of the
classification errors of all individual task,
$$L_{\beta}(f; \mathrm x, y)=\sum_{s=1}\beta(s)(1-\prod_{i=s+1}\mathbb I_{\{f(x_{y(s)})>f(x_{y(i)})\}})$$

Then the essential loss for ranking is defined the minimum value of the weighted sum over all the permutation
$$L_{\beta}(f;\mathrm x, \mathcal L)=\min_{y\in Y_{\mathcal L}} L_{\beta}(f; \mathrm x, y).$$

Denote the ranked list produced by $f$ as $\pi_f$ . Then it is easy to verify that
$$L_{\beta}(f;\mathrm x, \mathcal L)= 0\iff \exists y\in Y_{\mathcal L}\quad satisfing \quad  L_{\beta}(f; \mathrm x, y)=0\iff \pi_f=y\in Y_{\mathcal L}.$$

1) Many pairwise and listwise loss functions are upper bounds of the essential loss.
2) Therefore, the pairwise and listwise loss functions are also upper bounds of (1-NDCG) and (1-MAP).

- [ ] [Essential Loss: Bridge the Gap between Ranking Measures and Loss Functions in Learning to Rank](https://www.microsoft.com/en-us/research/publication/essential-loss-bridge-the-gap-between-ranking-measures-and-loss-functions-in-learning-to-rank/)
- [ ] [RankExplorer: Visualization of Ranking Changes in Large Time Series Data](https://www.microsoft.com/en-us/research/publication/rankexplorer-visualization-ranking-changes-large-time-series-data/)
- [ ] [Revisiting Approximate Metric Optimization in the Age of Deep Neural Networks](https://ai.google/research/pubs/pub48168)
- [ ] [Revisiting Online Personal Search Metrics with the User in Mind](https://ai.google/research/pubs/pub48243)

### QuickRank

QuickRank is an efficient Learning to Rank toolkit providing multithreaded C++ implementation of several algorithms.

+ [ ] [QuickRank: A C++ suite of Learning to Rank algorithms](http://quickrank.isti.cnr.it/research-papers/)
+ [ ] http://lyst.github.io/lightfm/docs/index.html

### Mathematics of Ranking

The mathematics of ranking is an emerging branch of mathematics. 
It has been given a boost with the success of Google. 
The gigantic eigenvalue problem solved to rank web pages 
according to their "importance" is an example of mathematics 
[that are applied in our dayly lives.](https://nalag.cs.kuleuven.be/research/workshops/ranking/proceedings.shtml)

Ranking problems arise in a multitude of domains, 
ranging from elections to web search and from management science 
to drug discovery. Consequently, ranking problems 
have been studied under different guises in many different fields, 
and each field has developed its own mathematical tools for studying ranking. This workshop will bring together for the first time researchers from mathematics, statistics, computer science, operations research, economics and game theory, and from both academic and industry backgrounds, to share their perspectives on ranking problems 
[and on the mathematical tools used to study them.](https://aimath.org/ARCC/workshops/mathofranking.html)


* https://www.stat.uchicago.edu/~lekheng/
* http://www.shivani-agarwal.net/Teaching/teaching.html
* http://www.shivani-agarwal.net/Publications/publications.html
* [AIM Workshop on the Mathematics of Ranking](https://www.stat.uchicago.edu/~lekheng/meetings/mathofranking/)
* [Symposium: The mathematics of ranking](https://nalag.cs.kuleuven.be/research/workshops/ranking/)
* https://aimath.org/ARCC/workshops/mathofranking.html
* https://aimath.org/WWN/mathofranking/
* https://yao-lab.github.io/seminar.html
* https://shenhaihui.github.io/
* http://jhc.sjtu.edu.cn/people/members/faculty/shuai-li.html
* https://shuaili8.github.io/
* https://mathcitations.github.io/recent.html
* https://sites.google.com/site/wcdingwebsite/products-services

### The Rankability of Data

The rankability problem  refers to a dataset's inherent ability 
to produce a meaningful ranking of its items. 
Ranking is a fundamental data science task. 
Its applications are numerous and include web search, data mining, cybersecurity, machine learning, and statistical learning theory. 
Yet little attention has been paid to the question of whether a dataset is suitable for ranking. 
As a result, when a ranking method is applied to an unrankable dataset, 
the resulting ranking may not be reliable. 
The rankability problem asks the following: How can rankability be quantified? 
At what point is a dynamic, time-evolving graph rankable? 
If a dataset has low rankability, c
an modifications be made and which most improve the graph's rankability? 
[We present a combinatorial approach to a rankability measure and then compare several algorithms for computing this new measure.](https://epubs.siam.org/doi/abs/10.1137/18M1183595)

- [The Rankability of Data](https://epubs.siam.org/doi/abs/10.1137/18M1183595)
- https://anderson-data-science.com/research/
- [ Graphs Analytics and Research in Data Science (IGARDS)](https://igards.github.io/)
- http://www.fields.utoronto.ca/talks/Rankability-Data
- [On the graph Laplacian and the rankability of data](https://www.sciencedirect.com/science/article/abs/pii/S0024379519305026)
- https://arxiv.org/abs/1912.00275
- https://anderson-data-science.com/research/
- https://thomasrcameron.com/research.html
- [Random variation and rankability of hospitals using outcome indicators](https://qualitysafety.bmj.com/content/20/10/869)
- https://github.com/IGARDS/ranklib
- https://github.com/IGARDS/rankability_toolbox

### Resource on Ranking and Reating

+ https://papers.nips.cc/paper/8288-contrastive-learning-from-pairwise-measurements.pdf
+ http://lintool.github.io/NSF-projects/IIS-1144034/
+ https://cs.uwaterloo.ca/~jimmylin/projects/index.html
+ [Spectral method and regularized MLE are both optimal for top-K ranking](https://projecteuclid.org/euclid.aos/1558425643)
+ https://europepmc.org/articles/pmc6785035
+ [Learning to Efficiently Rank with Cascades](http://lintool.github.io/NSF-projects/IIS-1144034/)
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
+ [Learning to Rank with Tensorflow](https://quantdare.com/learning-to-rank-with-tensorflow/)
+ http://www.cs.virginia.edu/~hw5x/
+ https://taskintelligence.github.io/WSDM2019-Workshop/#