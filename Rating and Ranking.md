# Rating and Ranking

The basic idea is the backfeed from the results.
After each game, this data is updated for the participants in the game. 
The rating of the winner is increased, and the rating of the loser is decreased

* https://www.remi-coulom.fr/WHR/WHR.pdf
* http://www.ams.org/notices/201301/rnoti-p81.pdf
* https://www.cs.cornell.edu/jeh/book2016June9.pdf

## Elo Rating

Elo rating is popular at many games such as Go game, soccer and so on.
It supposes that the performance are random and the winning rate is determined by the differences of two players.
If the differences of the scores is beyond some threshold such as $200$, it is supposed that the winner probability is 3/4.
And  it is natural if difference of scores is $0$, the winning rate is 1/2.

We assume that
$$ P(d) = \frac{1}{1+e^{-\frac{d}{\theta}}} $$
where the parameter $\theta$ is related with the threshold.

For example, the expected performance of player A is $E_{A} = \frac{1}{1 + 10^{-\frac{R_A - R_B}{400}}}$ and $E_{B} = \frac{1}{1 + 10^{-\frac{R_B - R_A}{400}}}=1-E_{A}$.
Supposing Player A was expected to score $E_{A}$ points but actually scored $S_{A}$ points. And the update rule is 
$${R}_{A}^{New} = R_{A} + K(S_A - E_A)$$
where $K$ is a constant.

* https://www.wikiwand.com/en/Elo_rating_system
  
## Glicko

The problem with the Elo system that the Glicko system addresses has to do with the
reliability of a player’s rating.

Glickman's principal contribution to measurement is "ratings reliability", called RD, for ratings deviation.
The RD measures the accuracy of a player's rating, with one RD being equal to one standard deviation.
If the player is unrated, the rating is usually set to 1500 and the RD to 350.

1. Determine RD
   
   The new Ratings Deviation (RD) is found using the old Ratings Deviation $RD_0$:
   $$RD=\min\{\sqrt{RD_0^2+c_2t}, 350\}$$
   where $t$ is the amount of time (rating periods) since the last competition and '350' is assumed to be the RD of an unrated player. And $c=\sqrt{(350^2-50^2)/100}\simeq 34.6$.
2. Determine New Rating
   
   The new ratings, after a series of m games, are determined by the following equation:
   $$r=r_0+\frac{q}{RD^{-2}+d^{-2}}\sum_{i=1}^{m}g(RD_i)(s_i - E(s|r,r_i,RD_i))$$
   where $g(RD_i)=\{1+\frac{3q^2(RD_i)^2}{\pi^2}\}^{-1/2}$, $E(s|r,r_i,RD_i))=\{1+10^{(\frac{g(RD_i)(r-r_i)}{-400})}\}$, $q=\frac{\ln(10)}{400}\simeq 0.00575646273$, $d^{-2} = q^2\sum_{i=1}^{m}[g(RD_i)^2]E(s|r,r_i,RD_i)[1-E(s|r,r_i,RD_i)]$, $r_i$ represents the ratings of the individual opponents. $s_i$ represents the outcome of the individual games. A win is $1$, a draw is $\frac {1}{2}$, and a loss is $0$.

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
$$\mu_{winner} \leftarrow \mu_{winner} + \frac{\sigma_{winner}^2}{c} \cdot \nu(\frac{(\mu_{winner}-\mu_{loser})}{c},\frac{\epsilon}{c})$$
$$\mu_{loser} \leftarrow \mu_{loser} - \frac{\sigma_{loser}^2}{c} \cdot \nu(\frac{(\mu_{winner}-\mu_{loser})}{c},\frac{\epsilon}{c})$$
$$\sigma_{winner} \leftarrow \sigma_{winner}^2 [1 - \frac{\sigma_{winner}^2}{c^2} \cdot w(\frac{(\mu_{winner}-\mu_{loser})}{c},\frac{\epsilon}{c}) ]$$
$$\sigma_{loser} \leftarrow \sigma_{loser}^2 [1 - \frac{\sigma_{loser}^2}{c^2} \cdot w(\frac{(\mu_{winner}-\mu_{loser})}{c},\frac{\epsilon}{c}) ]$$
$$c^2 = 2\beta^2+\sigma^2_{winner}+\sigma^{2}_{loser}$$

where $\beta^2$ is the average of all players' variances, $\nu(t) =\frac{N(t)}{\Phi(t)}, w(t)=\nu(t)[\nu(t)+1]$. And $N(t),\Phi(t)$ is the PDF and CDF of standard normal distribution, respectively.

- [X] https://www.wikiwand.com/en/TrueSkill
- [X] https://www.jianshu.com/p/c1fbba3af787
- [ ] https://zhuanlan.zhihu.com/p/48737998
- [ ] https://www.wikiwand.com/en/Location%E2%80%93scale_family
- [ ] https://www.wikiwand.com/en/Glicko_rating_system

## Whole-History Rating

Incremental Rating Systems or dynamical rating systems such as TrueSkill  do not make optimal use of data.
The principle of Bayesian Inference consists in computing a probability distribution over player ratings ($r$) from the observation of game results ($G$) by inverting the model thanks to Bayes formula:
$$P(r|G)=\frac{P(G|r)P(r)}{P(G)}$$
where $P(r)$ is a prior distribution over $r$, and $P(G)$ is a normalizing constant. $P(r|G)$ is called the posterior distribution of γ
$P(G|r)$ is the Bradley-Terry model, i.e. 
$$P(\text{player $i$ beats player $j$ at time $t$})= \frac{1}{1+10^{-\frac{R_i(t)-R_j(t)}{400}}}$$
as shown in Elo rating system.

The WHR algorithm consists in computing, for each player, the γ(t) function
that maximizes $P(r|G)$. Once this maximum a posteriori has been computed,
the variance around this maximum is also estimated, which is a way to estimate
rating uncertainty.

- [ ] https://www.wikiwand.com/en/Bradley%E2%80%93Terry_model
- [ ] https://www.wikiwand.com/en/Ranking
- [ ] https://arxiv.org/pdf/1701.08055v1.pdf
- [X] https://www.remi-coulom.fr/WHR/WHR.pdf

## Randking 

Combining feedback from multiple users to rank a collection of items is an important task.


* https://www.wikiwand.com/en/Arrow%27s_impossibility_theorem
* https://plato.stanford.edu/entries/arrows-theorem/
* https://www.math.ucla.edu/~tao/arrow.pdf