# Machine Learning via a Modern Optimization Lens 

The following relation is generally accepted in the machine learning community 

> Learning = Representation + Optimization + Evaluation

although it is not a consensus.


The majority of the central problems of regression, classiﬁcation, and estimation have been addressed using heuristic methods even though they can be
formulated as formal optimization problems. 
While continuous optimization approaches have had a signiﬁcant impact in Machine Learning (ML)/Statistics (S), mixed integer optimization (MIO)
has played a very limited role, primarily based on the belief that MIO models are computationally intractable. 
The last three decades have witnessed 
* (a) algorithmic advances in MIO, which coupled with hardware improvements have resulted in an astonishing over 2 trillion factor speedup in solving MIO problems, 
* (b) signiﬁcant advances in our ability to model and solve very high dimensional robust and convex optimization models.

Our objective in [this course](https://statmodeling.stat.columbia.edu/2019/11/26/machine-learning-under-a-modern-optimization-lens-under-a-bayesian-lens/) is to revisit some of the classical problems in ML/S and demonstrate that they can greatly beneﬁt from a modern optimization treatment. 
The optimization lenses we use in this course include `convex, robust, and mixed integer` optimization. 
In all cases we demonstrate that optimal solutions to large scale instances 
(a) can be found in seconds, 
(b) can be certiﬁed to be optimal/near-optimal in minutes and 
(c) outperform classical heuristic approaches in out of sample experiments involving real and synthetic data.

- https://ryancorywright.github.io/
- https://www.coursicle.com/mit/courses/15/095/
- http://oskali.mit.edu/
- http://www.mit.edu/~yuchenw/
- https://kiranvodrahalli.github.io/about/
- http://www.homepages.ucl.ac.uk/~ucakche/
- https://andre-martins.github.io/
- http://mlli.mit.edu/
- https://deepai.org/profile/michael-lingzhi-li
- http://www.mit.edu/~rahulmaz/research.html
- http://www.mit.edu/~rahulmaz/index.html
- http://niaohe.ise.illinois.edu/forml_group.html
- https://sites.google.com/site/wildml2017icml/
- https://web.stanford.edu/~vcs/talks/MicrosoftMay082008.pdf
- https://r2learning.github.io/
- http://aditya-grover.github.io/

Optimization has a long and distinguished history that has had and continues to have genuine impact in the world.
In a typical optimization problem in the real world, practitioners see optimization as a black-box tool 
where they formulate the problem and they pass it to a solver to find an optimal solution. 
Especially in high dimensional problems typically encountered in real
world applications, it is currently not possible to interpret or intuitively understand the optimal solution. 
[However, independent from the optimization
algorithm used, practitioners would like to understand how problem parameters affect the optimal decisions in order to get intuition and interpretability
behind the optimal solution.](https://arxiv.org/abs/1812.09991)



Philosophical principles of [the book](https://www.dynamic-ideas.com/books/machine-learning-under-a-modern-optimization-lens):

* Interpretability is materially important in the real world.
* Practical tractability not polynomial solvability leads to real world impact.
* NP-hardness is an opportunity not an obstacle.
* ML is inherently linked to optimization not probability theory. Data represent an objective reality; models only exist in our imagination.
* Optimization has a significant edge over randomization.
* The ultimate objective in the real world is prescription,not prediction.


See the book review at [“Machine Learning Under a Modern Optimization Lens” Under a Bayesian Lens](https://statmodeling.stat.columbia.edu/2019/11/26/machine-learning-under-a-modern-optimization-lens-under-a-bayesian-lens/).

Generally speaking, the optimization methods for machine learning are classified into 2 groups: the continuous optimization methods and the discrete optimization methods.

We will focus on the following optimization methods:
(1) mixed integer programming methods (MIP);
(2) alternating direction methods with multipliers (ADMM);
(3) fixed point perspective to unify diverse iterative methods.

Simply speaking, we will focus on the optimization of machine learning in  this course, i.e. how to train the models.
We would formulate the machine learning problem as an optimization problem, supervised or unsupervised.
Specially, we pay more attention on the discrete optimization methods than the continuos ones.

- https://zero-lab-pku.github.io/publication/
- http://cermics.enpc.fr/equipes/optimisation.html
- http://prml.github.io/
- http://www.mbmlbook.com/
- http://madscience.ucsd.edu/dsgl.html
- https://hsnamkoong.github.io/
- https://www.intelligent-optimization.org/LIONbook/

## Nearest Neighbors Search

`Nearest Neighbors Search` is to find the nearest neighbors of the query vector $v$ in a vector set $S$.
For example, the projection into a real convex compact set is a special kind of nearest neighbors search: $Proj_S(x)=\arg\min_{y\in S}\|x-y\|_2$.
Usually  `Nearest Neighbors Search` refers to the projection into a discrete space in computer science.
More Formally,  `Nearest Neighbors Search` is to find the 
$$\arg\max_{p_i}\sigma(p_i, q)$$
given object domain $\mathbb U$ and similarity function $\sigma$;
database $S=\{p_1,p_2,\cdots, p_n\}\subset \mathbb{U}$ is fixed;
the query $q\in\mathbb{U}$.

It is also known as `Best match problem`, `Post office problem`.

Note that the domain $\mathbb U$ may not be numerical.

It seems simple when there is only a few sample while it is hard in high dimension space.
[Unfortunately, the complexity of most existing search algorithms, such as k-d tree and R-tree, grows exponentially with dimension, making them impractical for dimensionality above 15 or so. In nearly all applications, the closest point is of interest only if it lies within a user-specified distance e.](https://www1.cs.columbia.edu/CAVE/projects/search/)
This optimization problem is not the typical optimization problem in machine learning.


- http://www1.cs.columbia.edu/CAVE/projects/nnsearch/
- https://www1.cs.columbia.edu/CAVE/projects/search/
- https://wikimili.com/en/Nearest_neighbor_search
- https://www.jeremyjordan.me/scaling-nearest-neighbors-search-with-approximate-methods/
- [Nearest Neighbors for Modern Applications with Massive Data](https://nn2017.mit.edu/)
- http://simsearch.yury.name/tutorial.html
- https://arxiv.org/pdf/1803.04765.pdf
- http://www.mit.edu/~andoni/thesis/main.pdf
- http://yury.name/
- http://zhangminjia.me/
- http://sss.projects.itu.dk/proximity-workshop.html
- http://people.csail.mit.edu/indyk/icm18.pdf
- https://people.csail.mit.edu/virgi/
- https://www.stat.berkeley.edu/~jon/
- http://www.people.fas.harvard.edu/~junliu/

### Approximate Nearest Neighbors Search

k-Nearest Neighbors Search is to find the top k nearest points of the given point $x$ in the high dimensional space.

<img src="https://www1.cs.columbia.edu/CAVE/projects/nnsearch/images/nn.png" width="80%"/>

- https://eprint.iacr.org/2019/359.pdf
- https://github.com/nmslib/hnswlib
- [Approximate Nearest Neighbor Search as a Multi-Label Classification Problem](https://arxiv.org/abs/1910.08322)
- https://hunch.net/~jl/projects/cover_tree/cover_tree.html
- [Approximate Nearest Neighbors Methods for Learning and Vision](http://groups.csail.mit.edu/vision/vip/nips03ann/description/)


#### Hash


[Nearest neighbor search is a problem of finding the data points from a database such that the distances from them to the query point are the smallest. Learning to hash is one of the major solutions to this problem and has been widely studied recently. ](https://www.microsoft.com/en-us/research/publication/survey-learning-hash/)

- https://www.microsoft.com/en-us/research/publication/survey-learning-hash/
- https://cs.nju.edu.cn/lwj/L2H.html
- https://learning2hash.github.io/
- http://simsearch.yury.name/tutorial.html
- https://github.com/lanl/CompactHash
- https://hunch.net/~jl/projects/hash_reps/index.html

### Vector search

[Deep learning models represent data as vectors, where distance between vectors reflects similarities. Approximate nearest neighbor (ANN) algorithms search billions of vectors, returning results in milliseconds.](https://www.microsoft.com/en-us/ai/ai-lab-vector-search)

- https://www.microsoft.com/en-us/ai/ai-lab-vector-search
- [Optimized Product Quantization for Approximate Nearest Neighbor Search](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Ge_Optimized_Product_Quantization_2013_CVPR_paper.pdf)


### k-Nearest Neighbor Classifier

k-Nearest Neighbor is a typical example of Lazy Learning.
This methods do not transform the raw input and extract any features of the input samples.

- http://vincentfpgarcia.github.io/kNN-CUDA/

### Template Matching

<img src="https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/PR_Figs/noisy_do.gif" with="30%"/>

[Template matching is a natural approach to pattern classification. For example, consider the noisy "D"'s and "O"'s shown above. The noise-free versions shown at the left can be used as templates.](https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/PR_simp/template.htm)
 To classify one of the noisy examples, simply compare it to the two templates. This can be done in a couple of equivalent ways:

1. Count the number of agreements (black matching black and white matching white). Pick the class that has the maximum number of agreements. This is a maximum correlation approach.
2. Count the number of disagreements (black where white should be or white where black should be). Pick the class with the minimum number of disagreements. This is a minimum error approach.


Template matching works well when the variations within a class are due to "additive noise." Clearly, it works for this example because there are no other distortions of the characters -- translation, rotation, shearing, warping, expansion, contraction or occlusion. It will not work on all problems, but when it is appropriate it is very effective. It can also be generalized in useful ways.

- [Template Matching](https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/PR_simp/template.htm)
- [Template Matching in OpenCV](https://www.docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html)
- http://www-cs-students.stanford.edu/~robles/ee368/matching.html


## Sparse Statistical  Learning and Optimization

Statistical Learning consists of three key parts:
(1) pattern recognition (classification); (2) regression; (3) density estimation.
Here we pay more attention to regression than others.
Roughly speaking, the regression is to learn a function $f$ that can response to the continuous variables well， i.e., $f:\mathbb{R}^p\to\mathbb{R}^d$;
the regression is to learn a function $f$ that can response to the discrete variables well; density estimation is aimed at the density of the population when only a few samples are available. 


- [Textbook: The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)
- https://esl.hohoweiya.xyz/index.html
- [A Solution Manual and Notes for: The Elements of Statistical Learning](https://waxworksmath.com/Authors/G_M/Hastie/WriteUp/Weatherwax_Epstein_Hastie_Solution_Manual.pdf)
- [The Elements of Statistical Learning lecture notes](http://www.loyhome.com/elements_of_statistical_learining_lecture_notes/)
- https://padhai.onefourthlabs.in/courses/data-science
- [Statistical Foundations of Data Science](https://orfe.princeton.edu/~jqfan/fan/classes/525/TableOfContent.pdf)
- [An Introduction to Statistical Learning](http://faculty.marshall.usc.edu/gareth-james/ISL/)

As datasets grow wide—i.e. many more features than samples  —the linear model has regained favor as the tool of choice.
We cannot fit linear models with $p > N$ without some constraints. 
Common approaches are
1. Forward stepwise adds variables one at a time and stops when overfitting is detected.
2. Regularized regression fits the model subject to some constraints such as LASSO.

`Bet on Sparsity Principle` is to  use a procedure that does well in sparse problems,
since no procedure does well in dense problems.
By assuming the number of important predictors is small relative to the total number of variables, 
[we can estimate an interpretable model that can yield more stable predictions.](https://www.mcgill.ca/epi-biostat-occh/files/epi-biostat-occh/sb-seminar_announcement-1mar18_0.pdf)

- [Whither the “bet on sparsity principle” in a nonsparse world?](https://statmodeling.stat.columbia.edu/2013/12/16/whither-the-bet-on-sparsity-principle-in-a-nonsparse-world/)
- [Sparsity in Machine Learning](http://www.cs.rpi.edu/~magdon/ps/conference/ICML-NLA2013.pdf)
- [Betting on Sparsity](https://sahirbhatnagar.com/assets/pdf/McGill_talk.pdf)
- [How do we choose our default methods?](http://www.stat.columbia.edu/~gelman/research/published/copss.pdf)
- [Structured Sparsity in Machine Learning: Models, Algorithms, and Applications](http://www.homepages.ucl.ac.uk/~ucakche/agdank/agdank2013presentations/martins.pdf)
- [On Sparsity Inducing Regularization Methods for Machine Learning](https://arxiv.org/abs/1303.6086)
- https://sahirbhatnagar.com/
- [Features of Big Data and sparsest solution in high confidence set](https://orfe.princeton.edu/~jqfan/papers/14/HC-Sparse.pdf)
- [Reflections on Breiman’s Two Cultures of Statistical Modeling1](http://www.stat.columbia.edu/~gelman/research/published/gelman_breiman.pdf)
- [A Closer Look at Sparse Regression](http://www.stat.cmu.edu/~larry/=sml/sparsity.pdf)

There are diverse topics related with sparsity such as feature engineering, variable selection， model selection, robust model, prior choice and so on.

[Simple and highly regularized learning methods must hence be favored as depicted above. Sparse linear models with a small number of nonzero coefficients are a popular choice when confronted with high-dimensional data. It is indeed often the case that all but very few features in high-dimensional data sets are truly relevant.](https://web.mit.edu/vanparys/www/topics/sparse/)

- http://statweb.stanford.edu/~tibs/research.html
- http://akyrillidis.github.io/pubs/
- https://cims.nyu.edu/~cfgranda/
- [sparse machine learning](https://people.eecs.berkeley.edu/~elghaoui/Pubs/cidu2011_final.pdf)
- http://spars2017.lx.it.pt/
- http://spams-devel.gforge.inria.fr/
- https://www.di.ens.fr/~fbach/
- http://www.predictioncenter.org/casp13/index.cgi
- http://www.eecs.harvard.edu/htk/
- https://redwood.berkeley.edu/
- https://sites.google.com/view/mlwithguarantees/home
- http://www.mit.edu/~dbertsim/papers.html
- http://www.mit.edu/~mcopen/research.html
- https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/maschinelles-lernen/news/


> When data is insufficient, one often requires additional information from the application domain to build a mathematical model, followed by numerical methods. Questions to be explored in this project include: 
> (1) how difficult is the process of extracting insights from data? 
> (2) how should reasonable assumptions be taken into account to build a mathematical model? 
> (3) how should an efficient algorithm be designed to find a model solution? More importantly, a feedback loop from insights to data will be introduced, i.e., 
> (4) how to improve upon data acquisition so that information becomes easier to retrieve? As these questions mimic the standard procedure in mathematical modeling, the proposed research provides a plethora of illustrative examples to enrich the education of mathematical modeling.

* [CAREER: Mathematical Modeling from Data to Insights and Beyond](https://nsf.gov/awardsearch/showAward?AWD_ID=1846690&HistoricalAwards=false)

> A common feature of modern big-data approaches to statistics, including lasso, hierarchical  Bayes, deep learning, and Breiman’s own trees and forests, is regularization—estimating lots of  parameters (or, equivalently, forming a complicated nonparametric prediction function) using  some statistical tools to control overfitting, whether by `the use of priors, penalty functions,  cross-validation, or some mixture of these ideas`. All these approaches to regularization continue to be the topic of active research.

-  http://www.stat.columbia.edu/~gelman/research/published/gelman_breiman.pdf

Here we focus on the fields which we can apply the mixed integer programming and ADMM to solve the optimization problem. 

- http://cse.msu.edu/~cse902/S14/ppt/Sparse%20Coding%20and%20Dictionary%20Learning.pdf
- https://arxiv.org/abs/2004.06152

### Mixed Integer Programming


[In statistics](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4166522/)
> Once a model is formulated, its parameters can be estimated by optimization. Because model parsimony is important, models routinely include nondifferentiable penalty terms such as the lasso. This sober reality complicates minimization and maximization.

Optimization is the source power of machine learning.


- [Operations Research & Machine Learning](https://www.euro-online.org/websites/or-in-practice/wp-content/uploads/sites/8/2018/04/Parallel-session-OR-and-machine-learning.pdf)
- [Sparse Learning with Integer Optimization](https://web.mit.edu/vanparys/www/topics/sparse/)
- https://orsolve.com/
- [Deep Learning in Discrete Optimization](http://www.ams.jhu.edu/~wcook12/dl/index.html)
- [2018 Summer School on “Operations Research and Machine Learning”](https://cermics-lab.enpc.fr/summer-school-operations-research-and-machine-learning/)
- [Deep Learning and MIP](http://www.dei.unipd.it/~fisch/papers/slides/2018%20Dagstuhl%20%5bFischetti%20on%20DL%5d.pdf)
- [On Handling Indicator Constraints in Mixed Integer Programming](http://www.dei.unipd.it/~fisch/papers/indicator_constraints.pdf)

<img src="https://ming-zhao.github.io/Business-Analytics/html/_images/docs_optimization_mixed-integer_programming_4_0.png" width="75%"/>

- [MIP for Business-Analytics](https://ming-zhao.github.io/Business-Analytics/html/docs/optimization/mixed-integer_programming.html)
- [nteger Linear Programming formulations in Natural Language Processing](https://ilpinference.github.io/eacl2017/)

### ADMM

[ADMM](https://web.stanford.edu/~boyd/admm.html) is an algorithm that solves convex optimization problems by breaking them into smaller pieces, each of which are then easier to handle.
The smaller pieces are usually solved by proximal operator.

It is argued that [the alternating direction method of multipliers is well suited to distributed convex optimization, and in particular to large-scale problems arising in statistics, machine learning, and related areas.](http://web.stanford.edu/~boyd/papers/admm_distr_stats.html)

We regard the ADMM  as a combination of augmented Lagrangian method and coordinate descent.

ADMM problem form (with $f, g$ convex)
$$\fbox{$\min f(x)+g(z)$}\\ \text{subject to } Ax + Bz = c$$
and we define the augmented Lagrangian function 
$$L_{\rho}(x, z, y) = f(x) + g(z) + y^T (Ax + Bz − c) + (\rho/2)\|Ax + Bz − c\|_2^2.$$
ADMM circularly update the variables:
$$x^{k+1}=\arg\min_{x}L_{\rho}(x, z^k, y^k)$$
$$z^{k+1}=\arg\min_{z}L_{\rho}(x^{k+1}, z, y^k)$$
$$y^{k+1}=y^k-\rho(Ax^{k+1} + Bz^{k+1} − c)$$

- http://web.stanford.edu/~boyd/admm.html
- http://joegaotao.github.io/cn/2014/02/admm
- https://johnmacfarlane.net/
- [Accelerated Alternating Direction Method of Multipliers:An Optimal $O(1/K)$ Nonergodic Analysis](https://zero-lab-pku.github.io/publication/helingshen/jsc19_accelerated_alternating_direction_method_of_multipliers_an_optimal_o_1k_nonergodic_analy/)
- https://zero-lab-pku.github.io/publication/helingshen/icml19_diffenrentiable_linearized_admm/
- [ADMM Algorithmic Regularization Paths for
Sparse Statistical Machine Learning](https://www.math.ucla.edu/~wotaoyin/splittingbook/ch13-hu-chi.pdf)
- http://maths.nju.edu.cn/~hebma/
- [GPU Acceleration of ADMM](https://people.ee.ethz.ch/~gbanjac/pdfs/admm_gpu.pdf)
- https://arxiv.org/abs/1807.07132

####  Proximal algorithms

>  Much like Newton's method is a standard tool for solving unconstrained smooth optimization problems of modest size, proximal algorithms can be viewed as an analogous tool for nonsmooth, constrained, large-scale, or distributed versions of these problems. 
> They are very generally applicable, but are especially well-suited to problems of substantial recent interest involving large or high-dimensional datasets. 
> Proximal methods sit at a higher level of abstraction than classical algorithms like Newton's method: the base operation is evaluating the proximal operator of a function, which itself involves solving a small convex optimization problem. 
> These subproblems, which generalize the problem of projecting a point into a convex set, often admit closed-form solutions or can be solved very quickly with standard or simple specialized methods.
>  Here, we discuss the many different interpretations of proximal operators and algorithms, describe their connections to many other topics in optimization and applied mathematics, survey some popular algorithms, and provide a large number of examples of proximal operators that commonly arise in practice.

The proximal operator of the convex function $f$ evaluated in $z$ is the solution of the following equation:
$$prox_{f, \gamma }(x)=\arg\min_z \frac{1}{2} \|x-z\|_2^2 + \gamma f(z).$$
[There exist several extensions of the previous definition](http://proximity-operator.net/proximityoperator.html) by modifying the quadratic distance into others such as the Bregman divergence.


Note that the iteration in ADMM:
$$x^{k+1}=\arg\min_{x}L_{\rho}(x, z^k, y^k)=\arg\min_{x}f(x) + g(z^k) + (y^k)^T (Ax + Bz^k − c) + (\rho/2)\|Ax + Bz^k − c\|_2^2$$
$$=\arg\min_{x}\fbox{$f(x) + (y^k)^T (Ax)$} + (\rho/2)\|Ax + Bz^k − c\|_2^2$$
which is extended proximal operator.

- [THE ROXIMITY OPERATOR REPOSITORY](http://proximity-operator.net/)
- [Customized proximal point algorithms](http://maths.nju.edu.cn/~hebma/paper/C-PPA.htm)
- [Proximal Algorithms by N. Parikh and S. Boyd](http://web.stanford.edu/~boyd/papers/prox_algs.html)
- http://num.math.uni-goettingen.de/proxtoolbox/
- http://www.nparikh.org/stanford/
- https://web.stanford.edu/~boyd/papers/pdf/prox_slides.pdf
- https://louisenaud.github.io/proximal-operator.html
- https://epfl-lts2.github.io/unlocbox-html/doc/prox/
- http://foges.github.io/pogs/
- http://uclaopt.github.io/TMAC/index.html
- http://web.stanford.edu/class/ee364b/lectures/monotone_slides.pdf

##### Proximal gradient and more

The proximal operator is the projection operator: 
$$prox_{f, \gamma }(x)=\arg\min_z \frac{1}{2} \|x-z\|_2^2 + \gamma f(z)$$
if $f(z)$ is the character function of a convex set.
So we claim that proximal gradient methods are the extension of projected gradient methods.
They are usually to solve the following questions:
$$\min_{x}h(x)+f(x)$$
where $h(x)$ is differentiable and smooth, $f(x)$ is convex.

$$x^{k+1}=\arg\min_{x}\frac{1}{2}\|x-\underbrace{(x^k-\alpha_k\partial h(x^k))}_{\text{gradient descent}}\|_2^2+\fbox{$\delta_{S}(x)$}\tag{Projected subgradient}$$
$$x^{k+1}=\arg\min_{x}\frac{1}{2}\|x-\underbrace{(x^k-\alpha_k\nabla h(x^k))}_{\text{gradient descent}}\|_2^2+\fbox{$\delta_{S}(x)$}\tag{Projected gradient}$$
$$x^{k+1}=\arg\min_{x}D(x,\underbrace{(x^k-\alpha_k\nabla h(x^k))}_{\text{gradient descent}})+\fbox{$\delta_{S}(x)$}\tag{Mirror gradient}$$
$$x^{k+1}=\arg\min_{x}\frac{1}{2}\|x-\underbrace{(x^k-\alpha_k\nabla h(x^k))}_{\text{gradient descent}}\|_2^2+\fbox{$f(x)$}\tag{Proximal gradient}$$
$$x^{k+1}=\arg\min_{x}\frac{1}{2}\|x-\underbrace{(x^k-H_k^{-1}\nabla h(x^k))}_{Newton}\|_2^2+\fbox{$f(x)$}\tag{Proximal Newton}$$

Here $\delta_{S}=1$ if $x\in C$ and $\delta_{S}=\infty$ if $x\notin C$;
$D(x, y)$ is the Bregman divergence. 



- http://cse.lab.imtlucca.it/~bemporad/
- http://stanford.edu/~boyd/papers/monotone_primer.html
- [Reducing Communication in Proximal Newton Methods for Sparse Least Squares Problems](https://www.cs.toronto.edu/~mmehride/papers/ICPPC18.pdf)
- [Proximal-Proximal-Gradient Method](https://www.math.ucla.edu/~wotaoyin/papers/prox_prox_grad.html)
- [A Proximal Gradient Algorithm for Decentralized Composite Optimization](https://www.math.ucla.edu/~wotaoyin/papers/pg_extra.html)
- [Accelerated Proximal Gradient Methods for Convex Optimization](http://www.mit.edu/~dimitrib/PTseng/apgm_mopta08_talk.pdf)
- [Proximal Newton Method](https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/prox-newton.pdf)
- [PROXIMAL NEWTON-TYPE METHODS FOR MINIMIZING COMPOSITE FUNCTIONS](https://stanford.edu/group/SOL/multiscale/papers/14siopt-proxNewton.pdf)
- [Fast L1-L2 Minimization via a Proximal Operator](https://users.math.msu.edu/users/myan/Papers/PDF/JSC2018.pdf)
- https://bayesgroup.github.io/team/arodomanov/incnewton_bmml16_slides.pdf
- https://jasondlee88.github.io/slides/proxnewton_slides_alpine.pdf
- http://cse.lab.imtlucca.it/~bemporad/publications/papers/cdc13-proximal-newton.pdf
- https://arxiv.org/pdf/1010.2847.pdf
- https://arxiv.org/pdf/1412.5154.pdf
- https://parameterfree.com/2019/09/26/online-mirror-descent-i-bregman-version

#### Alternating Direction Methods and Operator Splitting

For some subproblem in ADMM, we can use the coordinate descent to find the solution.
For example,  we can solve the additive separate loss function parallel:
$$\min_{x}\underbrace{\sum_{i=1}^pf_i(x_i)}_{\text{total loss}}=\sum_{i=1}^p\min_{x_i}f_i(x_i).$$ 
The coordinate (descent) methods take advantages of thi properties of the loss function:
$$x_i=\arg\min_{x_i} f_i(x_i)\quad\forall\,\, i\in\{1,2\cdots, p\}.$$
Generally, the loss function is not separable and the variables are coupled so we use the iterative methods to find the minimizers:
$$x_i^{k+1}=\arg\min_{x_i} f(x_1^{k+1},\cdots, x_{i-1}^{k+1}, x_i, x_{i+1}^{k}, \cdots, x_{p}^{k})\quad\forall\,\, i\in\{1,2\cdots, p\}.$$

The optimizer operator of the total loss function is split into some optimizer operators of `partial loss` functions.
The common scheme is to split two operators.
For example, 
$$\min_{x, y}Q(x, y)$$
$$x^{k+1}=\arg\min_{x}Q(x, y^{k})$$
$$y^{k+1}=\arg\min_{y}Q(x^{k+1}, y)$$

- http://faculty.uml.edu/cbyrne/CandT.pdf
- http://curtis.ml.cmu.edu/w/courses/index.php/Alternating_Minimization
- http://www-scf.usc.edu/~yuhao/journal/SimpleParallel-SIOPT-2017.pdf
- https://www.math.ucla.edu/~wotaoyin/papers/bcu/index.html

Operator splitting is also applied to the saddle point finding problem:
$$\arg\min_{y}\max_{x}Q(x, y)$$
$$x^{k+1}=\arg\max_{x}Q(x, y^{k})$$
$$y^{k+1}=\arg\min_{y}Q(x^{k+1}, y)$$

- https://liangjiandeng.github.io/papers/2019/FormVersion_Elastica2019.pdf
- https://hplgit.github.io/fdm-book/doc/pub/book/sphinx/._book018.html
- https://www.math.ucla.edu/~wotaoyin/splittingbook/ch3-macnamara-strang.pdf
- https://epubs.siam.org/doi/book/10.1137/1.9781611970838
- [Some Facts about Operator-Splitting and Alternating Direction Methods](https://www.math.ucla.edu/~wotaoyin/splittingbook/ch2-glowinski-pan-tai.pdf)
- https://www.algorithm-archive.org/contents/split-operator_method/split-operator_method.html
- [Numerical solution of saddle point problems](http://www.mathcs.emory.edu/~benzi/Web_papers/bgl05.pdf)
- https://web.stanford.edu/class/ee392o/

####  Augmented Lagrangian

We  focus on the following problem in this section:
$$\fbox{$\min f(x)+g(z)$}\\ \text{subject to } Ax + Bz = c.$$

The Lagrangian multiplier methods only need the 
$$L(x, y,z)= f(x) + g(z) + \underbrace{y^T (Ax + Bz − c)}_{\text{Linear part}}$$
and  the question may raise in mind: why we choose linear part? Is there an alternative to the linear part?

It is still convex. It is obvious that if $Ax + Bz − c\not=0$, $\max_{y}L(x, y,z)=\infty$.
$$\max_{y}\min_{x,z}L(x, y,z) \leq \min_{x,z}\max_{y}L(x, y,z).$$
`Penalty and barrier` methods are also aimed at the constrained optimization problems via embedding the constrained conditions into the objective function.


<img src="https://tse1-mm.cn.bing.net/th/id/OIP.-vVhmSYToxeF3Y5iWXAfRgHaHD?pid=Api&rs=1" width="40%"/>

- [THE METHOD OF LAGRANGE MULTIPLIERS](http://ramanujan.math.trinity.edu/wtrench/texts/TRENCH_LAGRANGE_METHOD.PDF)
- http://www.cs.cmu.edu/~aarti/Class/10725_Fall17/Lecture_Slides/Augmented-lagrangian.pdf
- [The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices](https://people.eecs.berkeley.edu/~yima/psfile/Lin09-MP.pdf)
- [Part 5: Penalty and augmented Lagrangian methods for equality constrained optimization](http://www.numerical.rl.ac.uk/people/nimg/course/lectures/parts/part5.2.pdf)


ADMM is based on the augmented Lagrangian function 
$L_{\rho}(x, z, y) = f(x) + g(z) + y^T (Ax + Bz − c) + (\rho/2)\|Ax + Bz − c\|_2^2.$
It is the sum of Lagrangian function and the `regular term` $(\rho/2)\|Ax + Bz − c\|_2^2$.
The question may raise in mind: why we choose the squared regularization term? Is there an alternative to the squared regularization term?


- [Bregman Alternating Direction Method of Multipliers](https://www-users.cs.umn.edu/~baner029/papers/14/badmm.pdf)
- [Applications of Lagrangian-Based Alternating Direction Methods and Connections to Split Bregman](https://www.math.uci.edu/~eesser/papers/alg_writeup.pdf)
- [Bregman Iterative Methods, Lagrangian Connections, Dual Interpretations, and Applications](https://www.math.uci.edu/~eesser/papers/overview.pdf)
- http://www.swmath.org/software/20288
- https://www-users.cs.umn.edu/~baner029/
- https://arxiv.org/pdf/1410.8625.pdf
- http://rll.berkeley.edu/gps/faq.html

Now We turn to the  following problem in this section:
$$\fbox{$\min f(x)$}\\ \text{subject to } h_i(x)=0 \quad i\in\{1,2,\cdots, N\}\\ q_j(x)\leq 0 \quad j\in\{1,2\cdots, M\}.$$

The  generalized Lagrangian is 
$$L(x, \mu, \alpha)=f(x)+\sum_{i}\mu_if_i(x)+\sum_{j}\alpha_jq_j(x)$$
where $\alpha_j\in\mathbb{R}^+$.
We can observe that $\max_{\mu,\alpha}\min_{x}L(x, \mu, \alpha)\leq\min_{x}\max_{\mu,\alpha}L(x, \mu, \alpha)$.
And the dual problem is 
$$\min_{x}\max_{\mu,\alpha}L(x, \mu,\alpha).$$

- [Generalized Lagrange multiplier method for solving problems of optimum allocation  of resources](http://www.hpca.ual.es/~jjsanchez/references/Generalized_Lagrange_multiplier_method_for_solving_problems_of_optimum_allocation_of_resources.pdf)
- [Generalized Lagrange Multiplier Method and KKT Conditions With an Application to Distributed Optimization](https://ieeexplore.ieee.org/document/8369146)
- [EXTENDED GENERALIZED LAGRANGIAN MULTIPLIERS FOR MAGNETOHYDRODYNAMICS USING ADAPTIVE MULTIRESOLUTION METHODS](https://www.esaim-proc.org/articles/proc/pdf/2013/05/proc134306.pdf)

###  Fixed point iteration algorithms

[In numerical analysis, fixed point iteration is a method of computing fixed points of iterated functions, which is one of the fundamental functions in computer science. As the name suggests, a process is repeated until an answer is achieved. Iterative techniques are used to find roots of equations, solutions of linear and nonlinear systems of equations, and solutions of differential equations.](https://fixedpoint.fandom.com/wiki/Fixed_Point_Wiki)

The fixed point iteration methods are in the following form:
$$\vec{x}^{k+1}=M(\vec{x}^k)=M^k(\vec{x}^1)$$ 
where $M$ is a mapping or operator.
The operator $M$ is the nonlinear generalization of matrix.

- https://x-datainitiative.github.io/tick/index.html
- https://fixedpointtheoryandapplications.springeropen.com/


Note that the optimal conditions is usually connected with some equation system so that many optimization methods are the fixed point iteration naturally.

For example, we would like to minimize the following objective function
$$\min_{x}\|Ax-b\|_2^2$$
and the optimal condition is that its gradient is zero, i.e., 
$$A^T(Ax-b)=0\iff A^TAx=A^Tb.$$
Additionally, it is equivalent to solve $Ax=b$ when $A$ is invertible.

- http://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
- https://gowerrobert.github.io/pdf/M2_statistique_optimisation/exe_prox.pdf
- https://archive.siam.org/books/mo25/mo25_ch6.pdf
- [Optimization by the Fixed-Point Method](http://www.optimization-online.org/DB_FILE/2007/09/1775.pdf)
- [Fixed-Point Iteration Method for Solving the Convex Quadratic Programming with Mixed Constraints](https://www.scirp.org/pdf/AM_2014012009512564.pdf)
- [Variational Optimization](https://arxiv.org/pdf/1212.4507v2.pdf)
- https://www.math.ucla.edu/~wotaoyin/math285j.16f/
- http://stanford.edu/class/ee367/

#### Fixed point and Operator splitting



- https://matthewktam.github.io/
- https://www.birs.ca/cmo-workshops/2017/17w5030/report17w5030.pdf
- https://www.simonsfoundation.org/event/3rd-workshop-operator-splitting-methods-in-data-analysis/
- http://archive.dimacs.rutgers.edu/Workshops/ADMM/announcement.html
- https://ictp.acad.ro/new-optimization-algorithms-for-neural-network-training-using-operator-splitting-techniques/
- https://www.esat.kuleuven.be/sista/ROKS2013/
- https://pcombet.math.ncsu.edu/
- https://jliang993.github.io/activity.html
- https://people.ok.ubc.ca/bauschke/
- https://engineering.purdue.edu/ChanGroup/project_PnP.html
- https://web.stanford.edu/~yyye/MIT2015.pdf
- https://www.math.ucla.edu/~wotaoyin/summer2016/5_fixed_point_convergence.pdf
- https://www.math.ucdavis.edu/~sqma/publications.html
- [SuperMann: A Superlinearly Convergent Algorithm for Finding Fixed Points of Nonexpansive Operators](https://ieeexplore.ieee.org/document/8675506)

#### Accelerated  Fixed-Point Iteration

Like the Markov chain, the state of the fixed point iteration only depend  on the last iteration.
History matters. 
We make decision not only according to the state  of the last  second around us.
It encourage us to take advantages of the  historical state of the fixed point iteration.

- [Anderson Acceleration for Fixed-Point Iteration](http://users.wpi.edu/~walker/MA590,NLEQ/HANDOUTS/anderson_acceleration_handout.pdf)
- http://users.jyu.fi/~oljumali/teaching/TIES594/14/fixedpoint.pdf
- https://web.stanford.edu/~boyd/papers/nonexp_global_aa1.html
- https://www7.in.tum.de/um/bibdb/esparza/dlt08.pdf
- https://jliang993.github.io/assets/files/slides/admm_neurips19_poster.pdf
- https://jliang993.github.io/assets/files/journal/faster-fista.pdf
- http://web.cse.ohio-state.edu/~belkin.8/

### Subset Selection

The best subset problem with subset size $k$, which is given by the following optimization problem:
$$\min_{\beta}\frac{1}{2}\|Ax-b\|_2^2\quad \text{subject to  } \|x\|_{0}\leq k$$
where the $\|\cdot\|_0$ (pseudo)norm of a vector $x$ counts the number of nonzeros in $x$ and is given by $\sum_{i}^p\mathrm{1}(x_i)$, where $\mathrm{1}(\cdot)$ denotes the indicator function.

Its Lagrangian version is given by
$$L(x)=\frac{1}{2}\|Ax-b\|_2^2+\lambda \|x\|_0\tag{subset}$$
where $\|x\|_0$ is the number of nonzero elements in the vector $x$.
Note that $\|x\|_0=\sum_{i}^p\mathrm{1}(x_i)=\|\operatorname{sgn}(x)\|_1$
where $\operatorname{sgn}(x_i)$ is the sign function and $\operatorname{sgn}(0)=0$ so that $x_i=\operatorname{sgn}(x_i)|x_i|$.

 - [Selection of Subsets of Regression Variables](http://staff.ustc.edu.cn/~zwp/teach/Reg/2981576.pdf)
 - [Best Subset Selection via a Modern Optimization Lens](https://arxiv.org/pdf/1507.03133.pdf)
 - [Fast Best Subset Selection: Coordinate Descent and Local Combinatorial Optimization Algorithms](http://www.mit.edu/~rahulmaz/L0Reg.pdf)
 - [Subset Selection with Shrinkage: Sparse Linear Modeling when the SNR is low](https://arxiv.org/abs/1708.03288)
 
### Variable Selection

[Variable selection in high-dimensional space characterizes many contemporary problems in scientific discovery and decision making.](https://orfe.princeton.edu/~jqfan/papers/07/Screening1.pdf)



- [High-dimensional data and variable selection](https://www.di.ens.fr/appstat/spring-2019/lecture_notes/Lesson7_HighDimensionalData.pdf)
- [A Selective Overview of Variable Selection in High Dimensional Feature Space](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3092303/)
- https://projecteuclid.org/euclid.aos/1247663752
- [Variable Screening in High-dimensional Feature Space](https://orfe.princeton.edu/~jqfan/papers/07/Screening1.pdf)
- http://pages.stat.wisc.edu/~shao/stat992/main.html
- https://arxiv.org/find/all/1/all:+AND+robert+tibshirani/0/1/0/all/0/1
- [Stability Selection](https://stat.ethz.ch/~nicolai/stability.pdf)

### LASSO

> The original motivation for the lasso was interpretability: It is an alternative to subset regression for obtaining a sparse model.

Lasso (“least absolute shrinkage and selection operator”) is a regularization procedure that shrinks regression coefficients toward zero, and in its basic form is equivalent to maximum penalized likelihood estimation with a penalty function that is proportional to the sum of the absolute values of the regression coefficients as an approximation to the $\ell_0$ regularized regression 
$$L(x)=\frac{1}{2}\|Ax-b\|_2^2+\lambda \|x\|_1\tag{LASSO}$$

The loss function is convex while not smooth.
Lasso does variable selection and shrinkage.



- http://statweb.stanford.edu/~tibs/lasso.html
- https://stat.ethz.ch/~geer/papers.html
- [Theory for the Lasso](https://stat.ethz.ch/~geer/Lasso+GLM+group.pdf)
- http://www.princeton.edu/~yc5/ele538b_sparsity/lectures/lasso_algorithm_extension.pdf
- https://statistics.stanford.edu/sites/g/files/sbiybj6031/f/2006-18.pdf
- http://statweb.stanford.edu/~tibs/sta305files/Rudyregularization.pdf
- [The Lasso Problem and Uniqueness](https://www.stat.cmu.edu/~ryantibs/papers/lassounique.pdf)
- [The Lasso with the Simulator](http://faculty.bscb.cornell.edu/~bien/simulator_vignettes/lasso.html)
- https://cermics-lab.enpc.fr/

For more on LASSO, see Robert Tibshirani's articles, lectures or books. 

- [Statistical Learning with Sparsity: The Lasso and Generalizations](http://web.stanford.edu/~hastie/StatLearnSparsity/)
- [In praise of sparsity and convexity](http://statweb.stanford.edu/~tibs/ftp/tibs-copss.pdf)
- [Lasso and Sparsity in Statistics](https://ssc.ca/sites/default/files/data/Members/public/Publications/BookFiles/Book/79-91.pdf)
- [Lasso and glmnet, with applications in GWAS-scale prediction problems](https://web.stanford.edu/~hastie/TALKS/wald_I.pdf)
- [Matrix completion and softImpute, with applications in functional data and collaborative filtering](https://web.stanford.edu/~hastie/TALKS/wald_II.pdf)
- [Graphical model selection with applications in anomaly detection](https://web.stanford.edu/~hastie/TALKS/wald_III.pdf)
- http://www-stat.stanford.edu/~tibs/ftp/covtest-talk.pdf

#### Solver for LASSO

We use diverse methods to solve the following optimization problem:
$$\arg\min_{x}L(x)=\frac{1}{2}\|Ax-b\|_2^2+\lambda \|x\|_1$$
where $\lambda >0$ is constant.
This is a typical convex composite optimization.
The optimal condition is that $0\in\partial L(x)$ where $\partial L(x)$ is the subgradients of the loss function $L(x)$.
If we define $\partial |x|\mid_{x=0}=0$, we can take advantages of the subgradient methods to find the LASSO:
$$x^{k+1}=x^k-\alpha_k[A^T(Ax^k-b)+\lambda \operatorname{sgn}(x^k)]$$
where $\alpha_i>0$.

- https://optimization.mccormick.northwestern.edu/index.php/Subgradient_optimization
- https://people.csail.mit.edu/dsontag/courses/ml16/slides/notes_convexity16.pdf
- http://theory.cs.washington.edu/reading_group/cvxoptJT.pdf

We apply proximal gradient to the LASSO problem:
$$x^{k+1}=\underbrace{\arg\min_{x}\frac{1}{2}\|x-[x^k-\alpha_kA^T(Ax^k-b)]\|_2^2+\lambda \|x\|_1}_{\text{proximal operator}}$$
which is additive and separable.

Since the $\ell_1$ norm is separable, the computation of $x^{k+1}$ reduces to solving a one-dimensional
minimization problem for each of its components, which by simple calculus produces
$$x^{k+1}=\mathbf{T}(x^k-\alpha_kA^T(Ax^k-b))$$

- https://arxiv.org/abs/math/0307152v1
- https://angms.science/doc/CVX/ISTA0.pdf
- http://www.seas.ucla.edu/~vandenbe/236C/lectures/fista.pdf
- https://homes.cs.washington.edu/~kusupati/pubs/kusupati20.pdf
- https://web.iem.technion.ac.il/images/user-files/becka/papers/71654.pdf

LASSO is equivalent to the following optimization:

$$\arg\min_{x}L(x)=\|Ax-b\|_2^2+\lambda \|y\|_1\\ \text{subject to  } x-y=0$$
The augmented Lagrangian function is defined as 
$$L_{\rho}(x, z, y) = \|Ax-b\|_2^2+\lambda \|y\|_1 + z^T (x-y) + (\rho/2)\|x-y\|_2^2.$$

We can use ADMM to find the LASSO:
$$x^{k+1}=\arg\min_{x}\|Ax-b\|_2^2 + (z^k)^T (x-y^k) + (\rho/2)\|x-y^k\|_2^2$$
$$y^{k+1}=\underbrace{\arg\min_{y} \lambda \|y\|_1 +  (z^k)^T(x^{k+1}-y) + (\rho/2)\|x^{k+1}-y\|_2^2}_{\text{proximity operator}}$$
$$z^{k+1}=z^k-\rho(x^{k+1} -y^{k+1})$$

- http://www.optimization-online.org/DB_FILE/2017/08/6149.pdf
- [Safe Screening with Variational Inequalities and Its Application to Lasso](http://proceedings.mlr.press/v32/liuc14.pdf)
- https://online.stat.psu.edu/stat857/node/158/

#### The Lasso as a Quadratic Program

Note that $\|x\|_1=\sum_{i=1}^p|x_i|=\underbrace{ReLU(0,x)}_{(x^+)}+\underbrace{ReLU(0,-x)}_{(x^{-})}$ and $x=x^{+}-x^{-}$.
$$\frac{1}{2}\|Ax-b\|_2^2+\lambda \|x\|_1\iff \frac{1}{2}\|A(x^{+}-x^{-})-b\|_2^2+\lambda (x^{+}+x^{-})\quad\text{subject to }x^{+}>0,x^{-}>0.$$

- https://github.com/Will-Wright/lasso-quadratic-solver
- https://arxiv.org/pdf/1401.2304.pdf
- http://www.aei.tuke.sk/papers/2012/3/02_Bu%C5%A1a.pdf
- https://davidrosenberg.github.io/mlcourse/Archive/2018/Lectures/02c.L1L2-regularization.pdf

#### Entropic Regularization

- [Entropic Regularization of the $\ell_0$ function](https://carma.newcastle.edu.au/resources/jon/entreg.pdf)
- https://carma.newcastle.edu.au/resources/jon/
- http://num.math.uni-goettingen.de/~r.luke/
- http://num.math.uni-goettingen.de/~r.luke/publications/publications.html


### Sparse Bayesian Learning

> "Sparse Bayesian Modelling" describes the application of Bayesian "automatic relevance determination" (ARD) methodology to predictive models 
> that are linear in their parameters. 
> The motivation behind the approach is that one can infer a flexible, nonlinear, predictive model 
> which is accurate and at the same time makes its predictions using only a small number of relevant basis functions 
> which are automatically selected from a potentially large initial set.

- http://staff.ustc.edu.cn/~hchen/signal/SBL.pdf
- http://www.miketipping.com/sparsebayes.htm
- https://sites.google.com/site/researchbyzhang/bsbl
- http://dsp.ucsd.edu/~zhilin/BSBL.html
- http://dsp.ucsd.edu/~zhilin/Software.html
- http://noiselab.ucsd.edu/papers/Gerstoft2016.pdf
- [Sparse Bayesian Multi-Task Learning for Predicting Cognitive Outcomes from Neuroimaging Measures in Alzheimer’s Disease](https://sccn.ucsd.edu/~zhang/Zhang_CVPR2012.pdf)
- http://jmlr.csail.mit.edu/papers/v1/tipping01a.html
- http://spars2017.lx.it.pt/index_files/papers/SPARS2017_Paper_2.pdf
- http://people.ee.duke.edu/~lcarin/li.pdf
- https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html

#### Bayesian Regularization

Bayesian regularization comes from the Bayesian lasso.
From the Bayesian perspective, the loss functions of LASSO is 
$$L(x)=\frac{1}{N}\underbrace{\|Ax-b\|_2^2}_{Gaussian}+\underbrace{\lambda \|x\|_1}_{Laplacian}$$
where the mean square error $\underbrace{\frac{1}{N}\|Ax-b\|_2^2}$ is in the Gaussian distribution and the regularization term $\underbrace{\lambda \|x\|_1}$ is derived from the Laplacian distribution.

From the statistical view, we can derive more sparsity induced regularization terms.

[Perhaps most formally the prior serves to encode information germane to the problem being analyzed, but in practice it often becomes a means of stabilizing inferences in complex, high-dimensional problems. In other settings, the prior is treated as little more than a nuisance, serving simply as a catalyst for the expression of uncertainty via Bayes’ theorem.](http://www.stat.columbia.edu/~gelman/research/published/entropy-19-00555-v2.pdf)

- https://bayesgroup.github.io/
- https://www.bayesfusion.com/
- https://bayesgroup.ru/
- [The Bayesian Lasso](https://people.eecs.berkeley.edu/~jordan/courses/260-spring09/other-readings/park-casella.pdf)
- https://arxiv.org/abs/1304.0001
- [Bayesian Variable Selection and Estimation for Group Lasso](https://projecteuclid.org/download/pdfview_1/euclid.ba/1423083633)
- [Bayesian Regularization Hedibert F. Lopes](http://hedibert.org/wp-content/uploads/2015/12/BayesianRegularization.pdf)
- https://xuan-cao.github.io/publications.html
- http://raybai.net/papers/by-date
- https://faculty.chicagobooth.edu/veronika.rockova/
- [Beyond subjective and objective in statistics](http://www.stat.columbia.edu/~gelman/research/published/gelman_hennig_full_discussion.pdf)
- https://ailab.criteo.com/laplaces-demon-bayesian-machine-learning-at-scale/
- [The Spike-and-Slab LASSO](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1629&context=statistics_papers)
- https://github.com/nyiuab/BhGLM

##### Spike and Slab Prior 

- http://raybai.net/wp-content/uploads/2019/07/MBSP.pdf
- [Penalized Regression, Standard Errors, and Bayesian Lassos](http://archived.stat.ufl.edu/casella/Papers/BL-Final.pdf)
- [Bayesian estimation of sparse signals with a continuous spike-and-slab prior](https://projecteuclid.org/euclid.aos/1519268435)
- http://people.ee.duke.edu/~lcarin/spike_slab.pdf



### Robust  Regression

It dates back to the robust statistics.
For example, we may define a new loss function for the linear regression model so that the outliers would not effect too much on the model:
$$L(x)=\ell(Ax-b)\tag{Robust regression}$$
where $\ell$ is the nonnegative function such as $\ell_1$ function. 

- [ROBUST REGRESSION | R DATA ANALYSIS EXAMPLES](https://stats.idre.ucla.edu/r/dae/robust-regression/)
- https://statistics.stanford.edu/sites/g/files/sbiybj6031/f/2015-04.pdf
- [Variance Breakdown of Huber (M)-Estimators in the High-Dimensional Linear Model](https://statistics.stanford.edu/research/variance-breakdown-huber-m-estimators-high-dimensional-linear-model)
- https://online.stat.psu.edu/stat501/lesson/13/13.3
- [Robust Regression models using scikit-learn](https://ostwalprasad.github.io/machine-learning/Robust-Regression-models-using-scikit-learn.html)
- http://home.olemiss.edu/~xdang/
- https://arxiv.org/pdf/1411.6160.pdf



In a linear regression with data $(X,Y)$, we consider a small perturbation within the neighborhood $\Delta \in \mathcal{U}(q,r)= \{\Delta\in \mathcal{R}^{n\times p}: \max_{\vert\vert \delta \vert\vert_{q} =1 } \vert\vert \delta \Delta \vert\vert_{r} \}$, then the $l_q$ regularized regression is precisely equivalently to the minimax robustness:
$$\displaystyle \min_{\beta}\max_{\Delta\in \mathcal{U}(q,r)} \vert\vert y-(X+\Delta)\beta \vert\vert_{r} = \min_{\beta} \vert\vert y-(X+\Delta)\beta \vert\vert_{r} + \vert\vert \beta \vert\vert_{q} $$
and such equivalence can also be extended to other norms too.

See the [Proof](http://web.mit.edu/dbertsim/www/papers/Statistics/Characterization%20of%20the%20equivalence%20of%20robustification.pdf).
 

### Low-rank Approximation

>  A common modeling assumption in many engineering applications is that the underlying data lies (approximately) on a low-dimensional linear subspace. This property has been widely exploited by classical Principal Component Analysis (PCA) to achieve dimensionality reduction. However, real-life data is often corrupted with large errors or can even be incomplete. Although classical PCA is effective against the presence of small Gaussian noise in the data, it is highly sensitive to even sparse errors of very high magnitude

- https://homes.esat.kuleuven.be/~delathau/
- https://canyilu.github.io/publications/
- http://faculty.cse.tamu.edu/davis/suitesparse.html
- https://www.csie.ntu.edu.tw/~cjlin/libmf/
- http://libfm.org/
- https://people.eecs.berkeley.edu/~yima/matrix-rank/home.html

#### Sparse PCA

The sparse PCA problem can be formulated in many different ways, one of them involves a low-rank approximation problem where the sparsity of the low-rank approximation is penalized:
$$\min_{p, q}\|M-pq^T\|_F+\mu\|p\|_1+\lambda\|q\|_1$$
where $M$ is the data matrix, $\|\cdot\|_F$ is the Frobenius norm, and $\mu \geq 0, \lambda\geq 0$ are parameters.


- https://people.eecs.berkeley.edu/~elghaoui/pubs_sparsepca.html
- https://www.ml.uni-saarland.de/code/sparsePCA/sparsePCA.htm
- https://ryancorywright.github.io/
- [Dealing with curse and blessing of dimensionality through tensor decompositions](https://www.ucl.ac.uk/bigdata-theory/wp-content/uploads/2017/07/deLathauwer-TBD3.pdf)

#### Optimal Shrinkage of Singular Values

Let us measure the denoising performance of a denoiser $\hat{X}$ at a signal matrix $X$ using Mean Square Error,
$$\min_{\hat X}\|X-\hat{X}\|_2^2.$$
The TSVD is an optimal rank-r approximation of the data matrix $X$.

- [Optimal Shrinkage of Singular Values](https://statistics.stanford.edu/sites/g/files/sbiybj6031/f/2016-03.pdf)
- [The Phase Transition of Matrix Recovery from Gaussian Measurements Matches the Minimax MSE of Matrix Denoising](https://statistics.stanford.edu/research/phase-transition-matrix-recovery-gaussian-measurements-matches-minimax-mse-matrix-denoising)
- https://statistics.stanford.edu/research/optimal-hard-threshold-singular-values-4sqrt3

### Sparse Additive Model

As shown as the Taylor expansion, we can approximate diverse functions based on simple basis functions.
Additive models are powerful. 
We can regard the additive models as a linear combination of base learners.
For example, [Super Learner](https://pubmed.ncbi.nlm.nih.gov/17910531/) is motivated by this use of cross validation as a weighted combination of many candidate learners.

Additive models are generalized linear models.
It projects the unknown oracle models into the specified hypothesis space.

- http://www.math.mcgill.ca/yyang/pub.html
- http://www.stat.cmu.edu/~larry/=sml/Spam.pdf
- http://www.cs.cmu.edu/~hanliu/papers/spam07.pdf
- [sail: Sparse Additive Interaction Learning](https://sahirbhatnagar.com/sail/)
- https://neurotree.org/beta/publications.php?pid=183394
- https://github.com/sahirbhatnagar/sail/
- [Super Learner](https://pubmed.ncbi.nlm.nih.gov/17910531/)
- [R/sl3: modern Super Learning with pipelines](https://tlverse.org/sl3/)
- https://mml-book.github.io/
- http://www.svcl.ucsd.edu/projects/taylor_boost/
- [Generalized structured additive regression based on Bayesian P-splines](https://epub.ub.uni-muenchen.de/1702/)
- [Model Selection with Many More Variables than Observations](https://web.stanford.edu/~vcs/talks/MicrosoftMay082008.pdf)
- [Group Sparse Additive Models](https://pubmed.ncbi.nlm.nih.gov/28393154/)
- http://www.u.arizona.edu/~junmingy/papers/Yin-Chen-Xing-ICML12-poster.pdf
- [Convex-constrained Sparse Additive Modeling and Its Extensions](http://www.u.arizona.edu/~junmingy/papers/Yin-Yu-UAI17.pdf)

#### The Smallest Random Forest

A general practice in using random forests is to generate a sufficiently large number of trees, 
although it is subjective as to how large is sufficient. Furthermore, random forests are viewed as “black-box” because of its sheer size. 
In this work, we address a fundamental issue in the use of random forests: how large does a random forest have to be? 
To this end, we propose a specific method to find a sub-forest (e.g., in a single digit number of trees) that can achieve the prediction accuracy of a large random forest (in the order of thousands of trees). 

- [Talk: Search for the Smallest Random Forest](https://publichealth.yale.edu/c2s2/12_209300_5_v1.pdf)
- [Search for the smallest random forest](https://www.intlpress.com/site/pub/files/_fulltext/journals/sii/2009/0002/0003/SII-2009-0002-0003-a011.pdf)
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2822360/

####  Sparse Boosting

Sparse Boosting seeks to minimize the AdaBoost exponential loss of a composite classifier using only a sparse set of base classifiers.

The boosting methodology in general builds on a user-determined base procedure or weak learner 
and uses it repeatedly on modified data which are typically outputs from the previous iterations. 
The final boosted procedure takes the form of linear combinations of the base procedures.
In notation, boosted procedure output a `strong` learner closer to the oracle model $\mathcal{O}$:
$$\sum_{i}^n w_if_i(x)\approx \mathcal{O}(x)$$
where $f_i$ are in the same function family.

The basic idea is to add some sparsity-induce regularization term
$$L(\sum_{i}^n w_if_i(x), y)+\|w\|_1$$
where $\sum_{i}^n w_if_i(x)$ is the linear combination of base learner.

- https://www.ucl.ac.uk/bigdata-theory/professor-peter-buhlmann/
- https://sahirbhatnagar.com/papers/
- https://collaborate.princeton.edu/en/publications/sparse-boosting
- [Sparse Boosting](http://jmlr.csail.mit.edu/papers/volume7/buehlmann06a/buehlmann06a.pdf)
- https://stat.ethz.ch/~nicolai/dantzig_discussion.pdf
- https://www.stat.berkeley.edu/~binyu/ps/rev.pdf
- https://arxiv.org/abs/2006.04059v1
- https://toc.csail.mit.edu/node/1307


We use a model selection criterion in order to achieve a maximal generalization.



#### Gradient Boosted Decision Trees for High Dimensional Sparse Output

Regularized gradient boosting decision trees and gradient boosting soft decision trees are boosting methods in order to make the GBDT more robust.



- [Gradient Boosted Decision Trees for High Dimensional Sparse Output](http://www.stat.ucdavis.edu/~chohsieh/rf/icml_sparse_GBDT.pdf)
- http://www.keerthis.com/
- http://www.huan-zhang.com/
- http://web.cs.ucla.edu/~chohsieh/
- https://www.cs.utexas.edu/~inderjit/
- http://www.cs.columbia.edu/~dhruv/
- https://dl.acm.org/profile/81453617072

### Sparse Coding

[The methods are based on the idea that good feature representation should be high dimensional (so as to facilitate the separability of the categories), should contain enough information to reconstruct the input near regions of high data density, and should not contain enough information to reconstruct inputs in regions of low data density.](https://cs.nyu.edu/~yann/research/sparse/index.html)
This lead us to use `sparsity criteria` on the feature vectors. 
Many of the sparse coding methods we have developed include a feed-forward predictor (a so-called encoder) 
that can quickly produce an approximation of the optimal sparse representation of the input. 
This allows us to use the learned feature extractor in real-time object recognition systems. 
Variant of sparse coding are proposed, including one that uses group sparsity to produce locally invariant features, two methods that separate the "what" from the "where" using temporal constancy criteria, 
and two methods for convolutional sparse coding, where the dictionary elements are convolution kernels applied to images.



- [Sparse Coding for Feature Learning](https://cs.nyu.edu/~yann/research/sparse/index.html)
- [Sparse coding and ‘ICA’](https://redwood.berkeley.edu/wp-content/uploads/2018/08/sparse-coding-ICA.pdf)
- https://dslpitt.org/uai/papers/11/p831-zhu.pdf
- https://sites.google.com/site/sparsereptool/
- https://www.cs.ubc.ca/~schmidtm/MLRG/sparseCoding.pdf
- https://www.cs.ubc.ca/~schmidtm/
- https://github.com/formosa21/Dictionary-learning
- [Atomatic Decomposition ](http://redwood.psych.cornell.edu/discussion/papers/chen_donoho_saunders_AD_by_BP.pdf)


### Compressed Sense

Donoho was one of the first researchers to develop math describing signals that are sparse. Such signals are zero most of the time, with occasional non-zero wiggles.
He first used $L_1 + sparsity$ techniques to
recover a sparse signal that has been blurred in an unknown, arbitrary way (today called `blind deconvolution’). He next used them to recover totally missing data.

- [David Donoho](https://www.mathunion.org/fileadmin/IMU/Prizes/Gauss/David%20Donoho-20180825-DLD-c-names.pdf)
-  https://statistics.stanford.edu/research/deterministic-matrices-matching-compressed-sensing-phase-transitions-gaussian-random
- https://arxiv.org/abs/1508.04924
- http://www.eecs.harvard.edu/htk/publications/
- http://freemind.pluskid.org/machine-learning/a-compressed-sense-of-compressive-sensing-i/
- https://statweb.stanford.edu/~candes/publications/downloads/RIP.pdf
- [Compressive Sensing Resources](http://dsp.rice.edu/cs/)
- https://www.zhihu.com/question/28552876?sort=created
- http://web.eecs.umich.edu/~girasole/csaudio/
- [Near-Optimal Adaptive Compressed Sensing](https://nowak.ece.wisc.edu/acs.pdf)

## Clustering

Teh clustering is the famous algorithms of unsupervised learning.

- [On the Equivalence of Nonnegative Matrix Factorization and Spectral Clustering](http://ranger.uta.edu/~heng/CSE6389_15_slides/On%20the%20Equivalence%20of%20Nonnegative%20Matrix%20Factorization%20and.pdf)
- [Clustering Problems in Optimization Models ](http://www-personal.umich.edu/~murty/clustering-problems-in-optimization-models.pdf)

### Convex Clustering

Assume that we are given $n$ data points and each data point
is described by a $d$ dimensional feature vector.
The convex clustering method clusters data points into groups via solving a convex optimization problem:
$$\min_{P}\|X-P\|_{F}+\alpha\sum_{i<j}w_{i,j}\|P_i-P_j\|_{p}$$
$P \in\mathbb R^{d×n}$ is the matrix consisting cluster centers 
and assignments, $P_i$ is the $i$th column of the matrix $P$ 
and represents the centroid of cluster that the $i$ th data point is assigned, $\alpha$ is a non-negative regularization parameter
that controls the number of clusters. 
$w_{i,j}$ is the weight between $i$th and $j$th data point specified by the user.

> Recently convex clustering has received increasing attentions, which leverages the sparsity inducing norms and enjoys many attractive theoretical properties. However, convex clustering is based on Euclidean distance and is thus not robust against outlier features. Since the outlier features are very common especially when dimensionality is high, the vulnerability has greatly limited the applicability of convex clustering to analyze many real-world datasets.

Specially, we consider the following optimization problem
 $$\min_{P}\sum_{i}\|X_i-P_i\|_{2}^2+\alpha\sum_{i<j}w_{i,j}\|P_i-P_j\|_{1}.$$
And we can use ADMM for this problem as following:
 $$\min_{P}\sum_{i}\|X_i-P_i\|_{2}^2+\alpha\sum_{i<j}w_{i,j}\|Y_{ij}\|_{1}\text{ subject to } P_i-P_j=Y_{ij}\forall  i<j$$
The augmented Lagrangian function is defined as 
$$L_{\rho}(P, Z, Y) = \sum_{i}\|X_i-P_i\|_{2}^2+\alpha\sum_{i<j}w_{i,j}\|Y_{ij}\|_{1} + \sum_{i<j}w_{i,j} Z_{ij}^T (P_i-P_j-Y_{ij}) + (\rho/2)\sum_{i<j}w_{i,j}\|P_i-P_j-Y_{ij}\|_2^2.$$

In an abstract way, it is in the following iteration
$$P^{k+1}=\arg\min_{P}L_{\rho}(P, Z^k, Y^k)$$
$$Y^{k+1}=\arg\min_{Y}L_{\rho}(P^{k+1}, Y, Z^k)$$
$$Z^{k+1}=\arg\max_{Z}L_{\rho}(P^{k+1}, Y^{k+1}, Z).$$
Note that it is additive separate.

- [Robust Convex Clustering Analysis](http://jiayuzhou.github.io/papers/qwangICDM16.pdf)
- [Convex Clustering: Model, Theoretical Guarantee and Efficient Algorithm](https://arxiv.org/pdf/1304.0499.pdf)
- [Splitting Methods for Convex Clustering](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4429070/)
- [Convex Clustering: Model, Theoretical Guarantee and Efficient Algorithm](https://arxiv.org/abs/1810.02677)
- [Weight Selection for Convex Clustering and BiClustering](https://dataslingers.github.io/clustRviz/articles/ClustRVizWeights.html)
- https://github.com/illidanlab/ConvexClustering
- https://stanford.edu/~boyd/papers/pdf/network_lasso.pdf
- https://dataslingers.github.io/clustRviz/
- https://github.com/duckworthd/cvxcluster
- http://ai.stanford.edu/~ajoulin/article/419_icmlpaper.pdf

### Interpretable Clustering

> State-of-the-art clustering algorithms use `heuristics` to partition the feature space and provide little insight into the rationale for cluster membership, limiting their interpretability. In healthcare applications, the latter poses a barrier to the adoption of these methods since medical researchers are required to provide detailed explanations of their decisions in order to gain patient trust and limit liability. 

- http://www.mit.edu/~agniorf/
- https://hwiberg.github.io/
- [Interpretable Clustering: An Optimization Approach](https://hwiberg.github.io/publication/icot/)
- [Interpretable Clustering via Optimal Trees](https://arxiv.org/pdf/1812.00539.pdf)
- [Interpretable clustering using unsupervised binary trees](https://link.springer.com/article/10.1007/s11634-013-0129-3)
- https://deepblue.lib.umich.edu/
- https://mlfinlab.readthedocs.io/en/latest/index.html



### Graph Clustering 

- https://ramyakv.github.io/GraphClusteringWithMissingData_KVOH.pdf
- https://courses.cs.washington.edu/courses/cse522/05au/clustering-flake.pdf
- https://cs.nyu.edu/shasha/papers/GraphClust.html
- https://www.csc2.ncsu.edu/faculty/nfsamato/practical-graph-mining-with-R/slides/pdf/Graph_Cluster_Analysis.pdf
- http://geza.kzoo.edu/~erdi/patent/Schaeffer07.pdf
- [Adaptive Consistency Propagation Method for Graph Clustering](http://crabwq.github.io/pdf/2019%20Adaptive%20Consistency%20Propagation%20Method%20for%20Graph%20Clustering.pdf)

##  Support Vector Machines 

Support vector machines can be applied in classification, regression and ranking.

Support vector machines are one of the most popular linear classifiers.
Like other linear classifiers, its prediction formulae is 
$$f(x)=\operatorname{sgn}(W\cdot x+b)\tag{SVM}$$
for the binary classification task.

Let us begin  with the binary classification with linear separation property so that there is a hyper-line $W\cdot x+ b$ that can classify the samples correctly.

The training data set $\{(x_i, y_i)\}$ are samples drawn from the population via experiment, measure or other observable approaches, where $x\in\mathcal X$ and the binary response $y_i\in\{+1,-1\}$.
We want to find a `correct` predictor $f(x)$  so that the loss is zero, i.e., $\sum_{i} (f(x_i)-y_i)^2=0$.
It is equivalent to solve the equation system $\operatorname{sgn}(W\cdot x_i+b)-y_i=0\quad \forall i$.

Note that the predictor is non-differential, non-linear  and non-convex with respect to the parameters $W, b$ according to the $\operatorname{sgn}$ operator.

- http://image.diku.dk/imagecanon/material/cortes_vapnik95.pdf
- http://www.svms.org/history.html
- https://svmlearning.com/
- https://cml.rhul.ac.uk/publications/vapnik/index.shtml
- https://leon.bottou.org/projects
- https://www.svm-tutorial.com/
- [SVM Tutorial: Classification, Regression, and Ranking](https://x-algo.cn/wp-content/uploads/2016/08/SVM-Tutorial-Classification-Regression-and-Ranking.pdf)
- https://scikit-learn.org/stable/modules/svm.html


### Geometrical Perspective

In diverse conditions [two disjoint convex subsets of a linear topological space can be separated by a continuous linear functional.](https://www.johndcook.com/SeparationOfConvexSets.pdf)

<img src="https://aishack.in/static/img/tut/linear-sep-3d.jpg"  title="Linear separability in 3D space" width=40% />

Suppose that there is a hyper-line $W\cdot x+ b$ that can classify the samples correctly, i.e.,  $y_i=\operatorname{sgn}(W\cdot x_i+b)$ for all $i$, the hyper-line $cW\cdot x+ cb$ can classify the samples correctly
because  $\operatorname{sgn}(W\cdot x_i+b)=\operatorname{sgn}[c(W\cdot x_i+b)]$ if $c>0$.

The optimal hyperplane 
$$W_0\cdot x+ b_0=0$$

is the unique one which separates the training data with a `maximal margin`:
it determines the direction $W/\|W\|_2$ where the distance between the projections of the training vectors of two different classes is maximal.
Here the margin is defined as 
$$\rho(W, b)=\min_{x:y=1}\frac{W\cdot x+b}{\|W\|_2}-\max_{x:y=-1}\frac{W\cdot x+b}{\|W\|_2}$$
where $\frac{W\cdot x+b}{\|W\|_2}$ is the functional margin and $|\frac{W\cdot x+b}{\|W\|_2}|=\operatorname{sgn}(W\cdot x+b)\frac{W\cdot x+b}{\|W\|_2}$ is the geometric margin. 

The margin of  optimal hyper-line  is 
$$\rho(W, b)=\frac{2}{\|W\|_2}.$$
Thus we can find the optimal hyper-line in the term of optimization method
$$\arg\min_{W, b}\frac{2}{\|W\|_2}\\ \text{ subject to  } (W\cdot x_i+b)y_i> 0.$$

And the optimal hyperplane can be written as a linear combination of training vectors:
$$W=\sum_{i}^{\ell}\alpha_i y_i x_i$$
if $y_i(W\cdot x_i+b)=1$ also called as support vectors.

<img src="https://chrisalbon.com/images/machine_learning_flashcards/Support_Vector_Classifier_print.png" width="60%"/>

We can project the data $x_i$ into the optimal hyperplane
$$Proj_{z:W\cdot z+b=0}(x_i)=\arg\min_{z:W\cdot z+b=0}\|x_i-z\|_2^2$$
$$\arg\min_{z}\|x_i-z\|_2^2\\ \text{subject to } W\cdot z+b=0$$
and  the Lagrangian version is that
$$\arg\min_{z}\max_{\lambda}\frac{1}{2}\|x_i-z\|_2^2+\lambda( W\cdot z+b)$$
so that we could obtain $Proj_{z:W\cdot z+b=0}(x_i)=x_i+\lambda W$ and $W\cdot(x_i+\lambda W)+b=0$ so $\lambda=-\frac{W\cdot(x_i)+b}{\|W\|_2^2}$.
- http://image.diku.dk/imagecanon/material/cortes_vapnik95.pdf
- https://www.johndcook.com/SeparationOfConvexSets.pdf 


###  Computational Perspective

- https://github.com/Xtra-Computing/thundersvm
- https://github.com/OrcusCZ/OHD-SVM
- https://github.com/MelvinCaradu/RSVM
- https://leon.bottou.org/projects/svqp

#### Constraint optimization

It is formulated as constraint optimization problem to select the best `correct` predictor as following:
$$\arg\min_{W, b} \|W\|_2^2, \text{  subject to } \operatorname{sgn}(W\cdot x_i+b)-y_i=0\quad \forall i$$
while the constrain is not convex.
Note that $\operatorname{sgn}(W\cdot x_i+b)-y_i=0\quad \iff \quad (W\cdot x_i+b)\cdot y_i> 0$ thus we can rewrite the above problem as convex constrained optimization 
$$\arg\min_{W, b} \frac{1}{2}\|W\|_2^2, \text{  subject to } (W\cdot x_i+b)y_i\geq 0\quad \forall i\tag{Primary problem}$$
which is equivalent to $\arg\min_{W, b} \frac{1}{2}\|W\|_2^2, \text{  subject to } (W\cdot x_i+b)y_i\geq 1\quad \forall i$.

Note that $W\cdot x_i+b=\overline{W}\cdot x_i$, $\overline{W}=W+\frac{1}{\vec{1}\cdot x_i}\vec{1}$ so that the parameter $b$ is not necessary.

We can solve it with Lagrangian method.
We introduce the [Lagrangian duality function](https://www-cs.stanford.edu/people/davidknowles/lagrangian_duality.pdf) $L(W, b, \lambda)=\frac{1}{2}\|W\|_2^2+\sum_{i}\lambda_i[(W\cdot x_i+b)y_i-1]$ where $\lambda_i>0$.
Then we can solve the dual problem of SVM $\arg\min_{W, b}L(W, b, \lambda^*)$ where $\lambda^*=\arg\max_{\lambda}L(W, b, \lambda)$.

It is known that the solution to the optimization problem is determined by the saddle point of this Lagrangian:
$$\arg\max_{\lambda}\min_{W, b}L(W, b, \lambda)\tag{Dual problem}.$$
And note that 
$L(W, b, \lambda)=\frac{1}{2}\|W\|_2^2+\sum_{i}\lambda_i[(W\cdot x_i+b)y_i-1]$ where $\lambda_i[(W_*\cdot x_i+b_*)y_i-1]=0$.

- https://blog.csdn.net/v_july_v/article/details/7624837
- [Solving SVM problems](http://web.mit.edu/dxh/www/svm.html)
- [Constrained Optimization and Support Vector Machines](http://freemind.pluskid.org/machine-learning/constrained-optimization-and-support-vector-machines/)




#### Unconstrained optimization

Linear separability assumption introduces the constraints in the maximal margin classifiers.
Outliers are the samples which are generated in abnormal way and they may be not separated by a hyper-line.

##### Soft margin 

To construct a soft margin separating hyperplane we minimize the functional
$$\frac{1}{2}\|W\|_2^2+(\sum_{i}\varsigma_i)^k\quad\forall k >1$$
under the constraints
$$y_i(Wx_i+b)\geq 1- \varsigma_i$$
$$\varsigma_i\geq 0.$$

And if $\varsigma_i\geq 1$, it means that the sample $x_i, y_i$ is classified wrongly.

When $k=1$, SVM is the standard regularized classifier
$$\min_{W,b, \varsigma}\frac{1}{2}\|W\|_2^2+(\sum_{i}\varsigma_i)^k\quad\forall i\tag{Primal Problem}$$
under the constraints
$$\varsigma_i\geq 1-y_i(Wx_i+b) $$
$$\varsigma_i\geq 0.$$
This is equivalent to
$$\min_{W,b}\frac{1}{2}\|W\|_2^2+\sum_i\underbrace{\max(0, 1-y_i(Wx_i+b))}_{\varsigma_i}\tag{Hinge Loss}.$$

The hinge loss is a surrogate loss function of the 0-1 loss.
The objective function is convex so that we can apply gradient descent to this problem.
We define $L(W. b)=\frac{1}{2}\|W\|_2^2+\sum_i\max(0, 1-y_i(Wx_i+b))$, and we could compute the gradient
$$\frac{\partial L(W, b)}{\partial W}=W+ \sum_{i}\sigma(1-y_i(Wx_i+b))[-y_ix_i]$$
$$\frac{\partial L(W, b)}{\partial b}=\sum_{i}\sigma(1-y_i(Wx_i+b))[-y_i]$$
where $\sigma(\cdot)$ is the binarized ReLU functions.
So that we can apply the gradient-based optimization methods to solve the SVM  problems.

- https://www.cs.utah.edu/~zhe/teach/pdf/svm-sgd.pdf
- https://svivek.com/teaching/lectures/slides/svm/svm-sgd.pdf
- https://leon.bottou.org/projects/sgd

We call $\max(0, 1-y_i(Wx_i+b))$ the hinge loss. 
Note that hinge loss is an upper bound function of 0/1 loss.

<img src="http://fa.bianp.net/blog/static/images/2013/loss_functions.png" />

And the squared hinge loss is also the upper bound of the 0-1 loss function.
$$\min_{W,b}\frac{1}{2}\|W\|_2^2+\sum_i[\underbrace{\max(0, 1-y_i(Wx_i+b))}_{\varsigma_i}]^2\tag{Squared hinge Loss}.$$
And we could compute the (sub)gradient
$$\frac{\partial L(W, b)}{\partial W}=W+ 2\sum_{i}\sigma(1-y_i(Wx_i+b))[1-y_i(Wx_i+b)]_{+}[-y_ix_i]$$
$$\frac{\partial L(W, b)}{\partial b}=\sum_{i}\sigma(1-y_i(Wx_i+b))[1-y_i(Wx_i+b)]_{+}[-y_i].$$

The hinge loss is designed for the non-separate data set.
Note that $\min_{W, b}\sum_{i} (f(x_i)-y_i)^2\geq 0$ so that the solution of maximal margin classifier absolutely belong to the solution of the following optimization problem
$$\arg\min_{W, b}\sum_{i} (f(x_i)-y_i)^2=\arg\min_{W, b}\sum_{i}[\operatorname{sgn}(W\cdot x+b)-y_i]^2.$$


- http://image.diku.dk/imagecanon/material/cortes_vapnik95.pdf


##### Ramp Loss

The ramp loss is a robust but non-convex loss for classification. Compared with other non-convex losses, a local minimum of the ramp loss can be effectively found. 
The effectiveness of local search comes from the piecewise linearity of the ramp loss. 

The ramp loss is defined as
$$\min[\max(0, 1-y_i(Wx_i+b)), 1]=\max(0, 1-y_i(Wx_i+b))-\max(-y_i(Wx_i+b), 0).$$
It is convex difference.
And the SVM with ramp loss id formulated as 
$$\min_{W,b}\frac{1}{2}\|W\|_2^2+\sum_i\min[\max(0, 1-y_i(Wx_i+b)), 1]\tag{Ramp Loss}.$$
It is available to compute the subgradient of this function.

- [Ramp Loss Linear Programming Support Vector Machine](http://jmlr.org/papers/volume15/huang14a/huang14a.pdf)
- http://www.dei.unipd.it/~fisch/papers/indicator_constraints.pdf

We can consider it as `gradient clipping`. 
The ramp function $f(x)=\min[\max(0, x), 1]$ projects the $x\in\mathbb{R}$ into the unit interval $[0, 1]$.
And the ramp is nonzero if and only if the margin $y_i(Wx_i+b)$ is in the interval $(0, 1)$.

- [gradient clipping](https://openreview.net/pdf?id=BJgnXpVYwS)


### Multiclass SVM

> In general, a pattern classifier carves up (or tesselates or partitions) the feature space into volumes called decision regions. All feature vectors in a decision region are assigned to the same category. The decision regions are often simply connected, but they can be multiply connected as well, consisting of two or more non-touching regions.
> 
> <img src="https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/PR_Figs/regions1.gif" width="40%"/>
> 
> The decision regions are separated by surfaces called the decision boundaries. These separating surfaces represent points where there are ties between two or more categories.

<img src="https://storage.ning.com/topology/rest/1.0/file/get/2220280738?profile=original" width="50%" />

- [Decision Boundaries](https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/PR_simp/bndrys.htm)
- [Comparing machine learning classifiers based on their hyperplanes or decision boundaries](https://tjo-en.hatenablog.com/entry/2014/01/06/234155)
- [Learn how each ML classifier works: decision boundary vs. assumed true boundary](https://www.analyticbridge.datasciencecentral.com/profiles/blogs/learn-how-each-ml-classifier-works-decision-boundary-vs-assumed)
- [Multiclass Classification: Margins Revisited, Novelty Detection via Trees, and Visual Object Recognition with Sparse Features Derived from a Convolution Neural Net](https://www.cse.huji.ac.il/~daphna/theses/Lior_Bar_2015.pdf)
- https://svivek.com/teaching/structured-prediction/spring2020/lectures/multiclass.html

A (multiclass) classifier is a function $H : \mathcal X\to \mathcal Y$ that maps an instance $x$ to an element $y$ of $Y$,
where $\mathcal X\subset\mathbb{R}^p, \mathcal Y=\{1,2,\cdots, k\}$.

We modify the binary SVM into multiclass SVM
$$\arg\min_{H} \|H\|_2, \text{  subject to } H(x_i)- y_i=0\quad \forall i$$
where $\|H\|_2^2$ is the $\ell_2$ norm of the mapping $H$.

The feature of SVM is that its decision boundary is linear.
So  we would generalize the support vector to the multiples class case.
In another word, we only focus on the support vectors.
We generalize the hyperplane to subspace, i.e., $H=f(Wx+b)\in\mathbb{R}^k$.
And the support vectors is defined as 
$$(Wx+b)^Ty=1$$
where we encode the element $y$ as one-hot vector.

We may formulate the multiclass SVM as the following optimization problem：
$$\arg\min_{W,b} \|W\|_2, \text{  subject to } (W x_i+b)^Ty_i\geq 1\quad \forall i$$
where $\|W\|_2^2$ is the $\ell_2$ norm of the matrix $W$.
Note that $y_i$ is one-hot so that $\max(W x_i+b)\geq (W x_i+b)^Ty_i$.
The Lagrangian function is defined as following
$$L(W, b, \lambda)=\frac{1}{2}\|W\|_2^2+\sum_{i}\lambda_i[(W x_i+b)^Ty_i-1]$$
so we can compute the gradients
$$\frac{\partial L(W, b, \lambda)}{\partial W}= W+\sum_{i}\lambda_i y_i x_i^T$$
$$\frac{\partial L(W, b, \lambda)}{\partial b}=\sum_{i}\lambda_i y_i$$
$$\frac{\partial L(W, b, \lambda)}{\partial \lambda_i}=(W x_i+b)^Ty_i-1.$$

This direct extension of binary SVM does not obey the principle of binary SVM.

Now we should define the prediction formulae so that we can derive $H(x_i)- y_i=0$  when $(W x_i+b)^Ty_i\geq 1$.
Note that $H(x_i)- y_i=0\iff H(x_i)= y_i\iff \|H(x_i)- y_i\|_{\infty}=0$.
We can use the binarized ReLU $\sigma(z): = \chi_{\{z>0\}}$ which equals 1 if $z > 0$, and $0$ otherwise. 
So that $\sigma(W x_i+b)^Ty_i\geq 1$ and $\|H(x_i)- y_i\|_{\infty}=0\iff\sigma(H(x_i)-y_i-m\vec{1})=0$ where $m=\max{H(x_i)}$.
And we define the prediction formulae
$$H(x)=\sigma(Wx+b)$$
with the assumption that $(W x_i+b)^Ty_i\geq 1$ and $(W x_i+b)^Ty\leq 0$ when $y\not= y_i\in\mathcal{Y}$.
So that we cannot directly extend the binary SVM.
Based on this observation,
$$\arg\min_{W,b}\frac{1}{2} \|W\|_2, \text{  subject to } (W x_i+b)^Ty_i\geq 1, (W x_i+b)^Ty\leq 0, y\not ={y_i}\in\mathcal{Y},\quad \forall i$$
where $y_i$ is one-hot vector.
Note that $(W x_i+b)^Ty_i-(W x_i+b)^Ty\geq 1, y\not ={y_i}$.

The Lagrangian function is defined as following
$$L(W, b, \lambda)=\frac{1}{2}\|W\|_2^2+\sum_{i}\lambda_i[(W x_i+b)^Ty_i-1]-\sum_{y\not=y_i}\beta_{i, y}[(W x_i+b)^Ty]$$

- [Convergence of a Relaxed Variable Splitting Coarse Gradient Descent Method for Learning Sparse Weight Binarized Activation Neural Network](https://www.math.uci.edu/~jxin/RVSCGD_2020.pdf)
- https://www.math.uci.edu/~jxin/xue.pdf

We can consider the non-separate cases for multiclass problems
$$\arg\min_{W,b}\frac{1}{2} \|W\|_2, \text{  subject to } (W x_i+b)^Ty_i\geq 1-\delta_i, (W x_i+b)^Ty\leq \eta_{i,y}, \delta_i\geq 0, \eta_{i,y}\geq 0, y\not ={y_i}\in\mathcal{Y},\quad \forall i.$$
And we can formulate multiclass SVM  as unconstrained optimization problem:
$$L=\frac{1}{2}\|W\|_2^2+\sum_{i}\sum_{y\not=y_i}[\Delta+(W x_i+b)^Ty-(W x_i+b)^Ty_i]_{+}$$

<img src="https://cs231n.github.io/assets/margin.jpg" width="80%" />

where $\Delta$ is positive constant.
Note that 
$$\sum_{y\not=y_i}[\Delta+(W x_i+b)^Ty-(W x_i+b)^Ty_i]_{+}=[(Wx_i+b)+(\Delta-(W x_i+b)^Ty_i) (\vec{1}-y_i)]_{+}^T\vec{1}-[(W x_i+b)^Ty_i]_{+}:=L_{i}$$
so that it is available to compute the subgradient of the loss function
$$\frac{\partial L_i}{\partial W}=\sigma(z_i)[x_i\vec{1}^T-((\vec{1}-y_i)\cdot \vec{1})x_iy_i^T]-\sigma((W x_i+b)^Ty_i)x_i y_i^T$$
where $z_i=(Wx_i+b)+(\Delta-(W x_i+b)^Ty_i) (\vec{1}-y_i)$ and $\sigma()$ is the binarized ReLU function.
And we can use the gradient-based methods to train this model.

- https://nlp.stanford.edu/IR-book/html/htmledition/multiclass-svms-1.html
- [Multi-Class Support Vector Machine via Maximizing Multi-Class Margins](https://www.ijcai.org/Proceedings/2017/0440.pdf)
- [Support vector machines maximizing geometric margins for multi-class classification](https://link.springer.com/article/10.1007/s11750-014-0338-8)
- [Comments on: Support vector machines maximizing geometric margins for multi-class classification](https://link.springer.com/article/10.1007/s11750-014-0341-0)
- http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/



*****
- https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es1999-461.pdf
- https://ljvmiranda921.github.io/notebook/2017/02/11/multiclass-svm/
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- https://shogun-toolbox.org/examples/latest/examples/multiclass_classifier/svm.html
- https://www.csie.ntu.edu.tw/~cjlin/libsvm/
- https://chrisalbon.com/
- https://www.csie.ntu.edu.tw/~cjlin/papers/multisvm.pdf
- https://cvxopt.org/examples/mlbook/mcsvm.html
- [ADMM-Softmax : An ADMM Approach for Multinomial Logistic Regression](ftp://ftp.math.ucla.edu/pub/camreport/cam20-17.pdf)




###  Kernel Methods

XOR problem is a basic non-linear separable problem in computer science while it cannot solve with the SVM directly.
And we should test the linear separability of the data set before we apply the support vector machine.

- [Testing for Linear Separability with Linear Programming in R](https://www.joyofdata.de/blog/testing-linear-separability-linear-programming-r-glpk/)
- [Supervised Learning (contd) Linear Separation](https://courses.cs.washington.edu/courses/csep573/11wi/lectures/22-nns.pdf)

> The machine conceptually implements the following idea: input vectors are non-linearly mapped to a very high dimension feature space. 
In this feature space a linear decision surface is constructed. Special properties of the decision surface ensures high generalization ability of the learning machine. 
The idea behind the support-vector network was previously implemented for the restricted case where the training data can be separated without
errors. 

To construct a hyperplane in a feature space one first has to transform the n-dimensional
input vector $x$ into an $n$-dimensional feature vector through a choice of an N-dimensional vector function $\phi$:
$$\phi:\mathbb{R}^n\to\mathbb{R}^N.$$

Classification of an unknown vector $x$ is done by first transforming the vector to the separating space ($x\mapsto\phi(x)$) and then taking the sign of the function:
$$f(x)=W\cdot \phi(x)+b.\tag{Kernel}$$

<img src="https://nlp.stanford.edu/IR-book/html/htmledition/img1331.png" width="50%" title="Projecting data that is not linearly separable into a higher dimensional space can make it linearly separable."/>

We can rewrite the above problem as convex constrained optimization 
$$\arg\min_{W, b} \frac{1}{2}\|W\|_2^2, \text{  subject to } (W\cdot \phi(x_i)+b)y_i\geq 1\quad \forall i\tag{Primary problem}$$
which is an convex constraint optimization problem with respect to $W, b$.
And according to the optimal condition, $W$ is the linear combination of the support vectors, i.e., $W=\sum_{i}\alpha_i\phi(x_i)$.
And so that we represent the SVM as following:
$$f(x)=\operatorname{sgn}([\sum_{i}\alpha_i\phi(x_i)]\cdot\phi(x)+b)=\operatorname{sgn}(\left<\sum_{i}\alpha_i\phi(x_i), \phi(x)\right>+b).$$

> A kernel is a continuous function that takes two variables $x$ and $y$ and map them to a real value such that $k(x,y)=k(y,x)$.

Since the algorithm can be written entirely in terms of the inner products $\left<x, z\right>$,
this means that we would replace all those inner products with $\left<\phi(x), \phi(z)\right>$. Specifically, given a feature mapping $\phi$, we define the corresponding Kernel to be
$$K(x, z)=\left<\phi(x), \phi(z)\right>.$$
Note that we do not optimize the functions $\phi$.In other word, we pre-specify the kernel functions.
> The kernel trick is to directly specify the inner product by a kernel function.

And the binary SVM with kernel $K$ is represented as
$$f(x)=\operatorname{sgn}(K(W,x)+b).$$

Suppose $K(s,t)$ is a symmetric (that is, $K(t,s)=K(s,t)$), continuous, and nonnegative definite kernel function on $[a,b]\times [a,b]$. 
> Mercer's theorem asserts that there is an orthonormal set of eigenfunctions $\psi_j(x)$ and eigenvalues $\lambda_j$ such that
> $$ K(s,t) = \sum_j^\infty \lambda_j \psi_j(s) \psi_j(t), $$
> where the values and functions satisfy the integral eigenvalue equation
> $$ \lambda_j \psi_j(s) = \int_a^b K(s,t) \psi_j(t). $$

[Kernel functions must be continuous, symmetric, and most preferably should have a positive (semi-) definite Gram matrix. Kernels which are said to satisfy the Mercer’s theorem are positive semi-definite, meaning their kernel matrices have only non-negative Eigen values. The use of a positive definite kernel insures that the optimization problem will be convex and solution will be unique.](http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/)

- http://image.diku.dk/imagecanon/material/cortes_vapnik95.pdf
- [Pattern Recognition and Machine Learning:PRML book](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- https://nlp.stanford.edu/IR-book/html/htmledition/nonlinear-svms-1.html
- http://fourier.eng.hmc.edu/e161/lectures/gaussianprocess/node8.html
- [Mercer’s Theorem, Feature Maps, and Smoothing](http://people.cs.uchicago.edu/~niyogi/papersps/MinNiyYao06.pdf)
- [Mercer's theorem and the Karhunen-Loeve expansion](https://www.chebfun.org/examples/stats/MercerKarhunenLoeve.html)


And we can directly apply the kernel trick to the multiclass support vector machine. 


- http://www.kernel-machines.org/
- https://kernelmethods.blogs.bristol.ac.uk/
- [Kernel Functions for Machine Learning Applications](http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/)
- [Ramp loss K-Support Vector Classification-Regression; a robust and sparse multi-class approach to the intrusion detection problem](https://www.sciencedirect.com/science/article/abs/pii/S0950705117301314)
- [On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines](http://jmlr.csail.mit.edu/papers/volume2/crammer01a/crammer01a.pdf)  
- https://svivek.com/teaching/machine-learning/fall2019/lectures/kernels.html

### Regularized  and Robust SVM

We can use the $\ell_1$ norm to induce some sparsity.

$$L(W, b) =\underbrace{ (1/m) \sum_i^m \left(1 - y_i ( W^T x_i+b) \right)_+ }_{error}+ \lambda \underbrace{\| W\|_1}_{regularizer}$$
where $\left(\cdot\right)_+$ is the hinge loss.

- [Robustness and Regularization of Support Vector Machines](http://jmlr.org/papers/volume10/xu09b/xu09b.pdf)
- [Proximal gradient method for huberized support vector machine](https://xu-yangyang.github.io/papers/PG4HSVM.pdf)
- https://projecteuclid.org/download/pdf_1/euclid.aos/1079120130
- https://pwp.gatech.edu/huan-xu/

#### ADMM for SVM

And we can reduce the model’s complexity by using the $\ell_1$-norm (also
known as LASSO penalty) instead of the Euclidean norm:
$$\arg\min_{W, b} \|W\|_1+\sum_{i}\epsilon_i, \text{  subject to } (W\cdot x_i+b)y_i\geq
 1-\epsilon_i, \epsilon_i\geq 0\quad \forall i.$$

And we can use convex optimization methods to solve such problems.

- https://www.cvxpy.org/examples/machine_learning/svm.html
- [ADMM for Training Sparse Structural SVMs with Augmented $\ell_1$ Regularizers](https://epubs.siam.org/doi/pdf/10.1137/1.9781611974348.77)
- [On Sparsity Inducing Regularization Methods
for Machine Learning](https://arxiv.org/pdf/1303.6086.pdf)
- [Support matrix machine](http://proceedings.mlr.press/v37/luo15.pdf)
- [Semi-supervised Support Tensor Based on Tucker Decomposition](http://www.jsjkx.com/CN/10.11896/j.issn.1002-137X.2019.09.028)

#### Mixed Integer Linear Programming for SVM

- [Feature selection for Support Vector Machines via Mixed Integer Linear Programming](http://repositorio.uchile.cl/bitstream/handle/2250/126859/Feature-selection-for-Support-Vector-Machines-via-Mixed-Integer-Linear-Programming.pdf?sequence=1)
- [A Mixed-Integer Programming Approach to Multi-Class Data Classification ](http://www.optimization-online.org/DB_FILE/2004/11/997.pdf)
- [The Support Vector Machine and Mixed Integer Linear Programming: Ramp Loss SVM with L1-Norm Regularization ](https://scholarscompass.vcu.edu/cgi/viewcontent.cgi?referer=https://cn.bing.com/&httpsredir=1&article=1007&context=ssor_pubs)
- [On Handling Indicator Constraints in Mixed Integer Programming](http://www.dei.unipd.it/~fisch/papers/indicator_constraints.pdf)
- http://burrsettles.com/index.html
- [Support Vector Machine via Sequential Subspace Optimization](https://ie.technion.ac.il/~mcib/sesop_svm_report.pdf)
- https://www.middleprofessor.com/files/applied-biostatistics_bookdown/_book/generalized-linear-models-i-count-data.html


### Structured SVM

Structured output prediction describes the problem of learning a function
$$h : \mathcal{X} \to \mathcal{Y}$$
where $\mathcal{X}$ is the space of inputs, and $\mathcal{Y}$ is the space of (multivariate and structured) outputs. 

- [Structured Prediction CS 6355, Spring 2020](https://svivek.com/teaching/structured-prediction/spring2020/lectures/training.html)
- https://svivek.com/teaching/structured-prediction/spring2020/
- [CS 159: Advanced Topics in Machine Learning: Structured Prediction](https://taehwanptl.github.io/)
- [PyStruct - Structured Learning in Python](https://pystruct.github.io/)
- https://amueller.github.io/

> Unlike regular SVMs, however, which consider only univariate predictions like in classification and regression,  Structured SVM  can predict complex objects y like trees, sequences, or sets.

- [Large Margin Methods for Structured and Interdependent Output Variables](http://www.jmlr.org/papers/volume6/tsochantaridis05a/tsochantaridis05a.pdf)
- https://taehwanptl.github.io/
- http://www.robots.ox.ac.uk/~vedaldi/svmstruct.html
- http://www.cs.cornell.edu/people/tj/svm_light/svm_struct.html
- http://www.cs.cornell.edu/people/tj/
- [Cutting-Plane Training of Structural SVMs](https://www.cs.cornell.edu/people/tj/publications/joachims_etal_09a.pdf)
- [Learning Structural SVMs with Latent Variables](https://researcher.watson.ibm.com/researcher/files/us-pederao/ChunNamYu.pdf)
- [CRF versus SVM-Struct for Sequence Labeling](http://www.keerthis.com/crf_comparison_keerthi_07.pdf)

### Transductive  SVM

Transductive SVM (TSVM) is a well known semi-supervised large margin learning method for binary text classification. 

- http://www.keerthis.com/transp.pdf
- https://www.cs.cornell.edu/people/tj/publications/joachims_99c.pdf
- [Machine Learning with Missing Labels: Transductive SVMs](https://calculatedcontent.com/2014/09/23/machine-learning-with-missing-labels-transductive-svms/)

###  Ensemble of SVM

We can use the ensemble methods to enhance the performance of support vector machines.

- [Ensemble of Exemplar-SVMs for Object Detection and Beyond](http://www.cs.cmu.edu/~tmalisie/projects/iccv11/)
- https://leon.bottou.org/publications/pdf/nips-2004c.pdf
- https://github.com/claesenm/EnsembleSVM
- https://archive.control.lth.se/ls-convex-2015/
- https://github.com/quantombone/exemplarsvm
- https://scikit-learn.org/stable/modules/ensemble.html

### Support Vector Regression

Obviously, classification and regression are different.
In the [history of SVM](http://www.svms.org/history.html), it is later when the algorithm was extended to the case of regression after the original support vector machine for bianry classification.

> Observe that there is predcition formulea in the objective fucntion when training a support vector machine, where we only focus on the linear party.

<img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_svm_regression_001.png" width="80%"/>

- https://www.stat.purdue.edu/~yuzhu/stat598m3/Papers/NewSVM.pdf
- http://alex.smola.org/papers/2003/SmoSch03b.pdf
- https://www.ling.upenn.edu/courses/cogs501/


### Relevance Vector Machines

The relevance vector machine or RVM (Tipping, 2001) is a Bayesian sparse kernel technique for regression and classification 
that shares many of the characteristics of the SVM whilst avoiding its principal limitations.

<img src="http://www.miketipping.com/images/sbprior.png" />

RVM is not a Bayesian interpretation of SVM but rather the method on its own, which adopts the same `functional form`.

It has found  the close connection between the learning procedure of logistic regression and binary SVM.


- https://www.deeplearn.me/1333.html
- https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression
- http://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf
- [Relevance Vector Machines](https://eloquentarduino.github.io/2020/02/even-smaller-machine-learning-models-for-your-mcu/)
- https://github.com/JamesRitchie/scikit-rvm
- https://gaowei262.github.io/
- http://dlib.net/
- http://davidrosenberg.github.io/mlcourse/
- http://topepo.github.io/caret/available-models.html

### Perception, Artificial Neural Network and Beyond

Linear separability is global assumption on the data distribution specially for the binary classification task.
And the XOR problem of the linear classifers leads to the doubt of the perceptron. 

- [A Review of "Perceptrons: An Introduction to Computational Geometry" ](https://core.ac.uk/download/pdf/82206249.pdf)
- [Perceptrons: An Introduction to Computational Geometry](https://mitpress.mit.edu/books/perceptrons)
- [An architectural Solution to the XOR Problem](http://www.mind.ilstu.edu/curriculum/artificial_neural_net/xor_problem_and_solution.php)
- [COGS 501 Mathematical Foundations for the Study of Language and Communication](https://www.ling.upenn.edu/courses/cogs501/)

> Classification by a support-vector network of an unknown pattern is conceptually done by first transforming the pattern into some high-dimensional feature space. An optimal hyperplane constructed in this feature space determines the output. 


The [percetron model](https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf) is the first attempt to model the functions of the brain in history in mathematics:
$$o_j=\varphi(\sum_{i}x_iw_{ij}+\theta_j).$$
<img src="https://pythonawesome.com/content/images/2019/08/act.png" width="70%"/> 

It consiists of two main parts:(1) the linear part $\underbrace{\sum_{i}x_iw_{ij}}_{\text{Inner product}}+\theta_j$; (2) the nonlinear part $\varphi$.

For example, the representtaion of bianry SVM alos follows this form:
$$\operatorname{sgn}(\sum_{i}x_iw_{ij}+\theta_j).$$
There are diverse alternatives of the activation funtion $\varphi(\cdot)$, which leads to diverse models.
And it is a long history of percetron.
And the logistic regression follows this from:
$$\frac{1}{1+\exp(\sum_{i}x_iw_{ij}+\theta_j)}$$
where $\frac{1}{1+\exp(-x)}$ is called logistic fucntion in statsitcis and sigmoid function in pattern recogition.

- [Single-layer Neural Networks (Perceptrons)](https://computing.dcu.ie/~humphrys/Notes/Neural/single.neural.html)

#### Generalized Linear Models  

When the response variables are not in Gaussian distribution, we will come up with the generalized linear models.
From the computational perspective, 
$$\eta(y)=\sum_{i}w_ix_i+b$$
so that $y=\eta^{-1}(\sum_{i}w_ix_i+b)$ where $\eta^{-1}$ is the inverse function of the $\eta$ so that there are must some restriction on the $\eta$. 
For example, the logistic regression can write in 
$$\ln(\frac{1}{y}-1)=\ln(\frac{1-y}{y})=\sum_{i}x_iw_{ij}+\theta_j$$
where $y\in(0, 1)$.
And the logistic regession minimize the negative log-likelihodd function
$$\min_{w, b}\sum_{i}-\ell(y_i, x_i).$$

> Statistically incorporating sparsity into regression
models has received a great deal of attention in the context of the best subset problem, which is the problem of
determining the best k-feature fit in a regression model:
$$\min_{w, b}\sum_{i}-\ell(y_i, x_i),\text{ subject to } \|w\|_0\leq k$$
where the $\ell_0$ (pseudo)norm of a vector $w$ counts the
number of nonzeros in $w$.

- [Logistic Regression: From Art to Science](https://projecteuclid.org/euclid.ss/1504253122)
- https://svivek.com/teaching/lectures/slides/logistic-regression/logistic-regression.pdf

The exponential family of distributions over $x$, given parameters $\eta$, is defined to
be the set of distributions of the form
$$p(x\mid \eta)=h(x)g(\eta)\exp(\eta^T\mu(x))$$
where $x$ may be scalar or vector, and may be discrete or continuous. 
Here $\eta$ are called the natural parameters of the distribution, and $\mu(x)$ is some function of $x$.
The function $g(\eta)$ can be interpreted as the coefficient that ensures that the distribution is normalized.
If we set $y=p(x\mid \eta)$, then we obtain the following conclusion:
$$y=h(x)g(\eta)\exp(\eta^T\mu(x))\iff \eta^T\mu(x)=\ln\frac{y}{h(x)g(\eta)}\iff \ln y= \eta^T\mu(x)+\ln(h(x)g(\eta)).$$

GLM|Neural Network
---|---
Link function | Activation function


- https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-slides-2.pdf
- [The Multiplicative Update Algorithm](https://svivek.com/teaching/machine-learning/fall2019/lectures/multiplicative-update.html)
- https://online.stat.psu.edu/stat504/node/216/
- http://statmath.wu.ac.at/courses/heather_turner/glmCourse_001.pdf
- https://www.guru99.com/r-generalized-linear-model.html

#### Kernel Tricks 

Note that there is no feature transformtion in the perceptron methods.
The linear part does not apply the kernel trick.
$$\varphi(K(x, W)+b)$$

<img src="https://visualstudiomagazine.com/articles/2020/03/19/~/media/ECG/visualstudiomagazine/Images/2020/03/radial_basis_train_3.asxh" width="70%"/>

- [SVC Parameters When Using RBF Kernel](https://chrisalbon.com/machine_learning/support_vector_machines/svc_parameters_using_rbf_kernel/)
- [Structured Prediction with Perceptron: Theory and Algorithms](https://www.gc.cuny.edu/CUNY_GC/media/Computer-Science/Student%20Presentations/Kai%20Zhao/Second_Exam_Survey_Kai_Zhao_12_11_2014.pdf)
- http://image.diku.dk/imagecanon/material/cortes_vapnik95.pdf
- [RBF Network](https://shomy.top/2017/02/26/rbf-network/)
- https://github.com/digantamisra98/Mish



### Deep Kernel

- [Sparse Approximation of Kernel Means](http://web.eecs.umich.edu/~cscott/talks/sparsekernelmean.pdf)
- [ Deep Kernel Learning ](https://arxiv.org/pdf/1511.02222v1.pdf)
- [Exact DKL (Deep Kernel Learning) Regression](https://docs.gpytorch.ai/en/latest/examples/06_PyTorch_NN_Integration_DKL/KISSGP_Deep_Kernel_Regression_CUDA.html)
- https://github.com/maka89/Deep-Kernel-GP

## Optimal Margin Distribution Machine Learning

The margin matters  in classification.

> Support vector machine (SVM) has been one of the most popular learning algorithms, with the central idea of maximizing the minimum margin, i.e., the smallest distance from the instances to the classification boundary. 
Recent theoretical results, however, disclosed that maximizing the minimum margin does not necessarily lead to better generalization performances, and instead, the margin distribution has been proven to be more crucial. In this paper, we propose the Large margin Distribution Machine (LDM), 
which tries to achieve a better generalization performance by optimizing the margin distribution. 

- https://cosx.org/2014/01/svm-series-maximum-margin-classifier
- [Multi-Class Optimal Margin Distribution Machine](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icml17mcODM.pdf)
- [Optimal Margin Distribution Machine](https://arxiv.org/abs/1604.03348v1)
- [最优间隔分布脊回归](http://crad.ict.ac.cn/fileup/1000-1239/HTML/2017-8-1744.shtml)


### Large Margin Distribution Machine

Large margin classifiers are actually trying to maximize the minimum margin.

It is a new direction for algorithm design, i.e., to optimize the margin distribution by maximizing the
margin mean and minimizing the margin variance simultaneously. 

The hard-margin LDM (Large Margin distribution Machine) is  formulated as 
$$\min_{W}\frac{1}{2}\|W\|_2^2+\gamma_{ave}-\gamma_{var}$$
under the constraints
$$y_i(Wx_i+b)\geq 1- \varsigma_i$$
$$\varsigma_i\geq 0.$$
Here $\gamma_{ave},\gamma_{var}$ is the average and variance of the margin, respectively.

It focus on the improvement of the evaluation metric.

- https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/annpr14.pdf
- [From AdaBoost to LDM](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/Adaboost2LDM.pdf)
- [Large Margin Distribution Machine](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/kdd14ldm.pdf)
- [On the Margin Explanation of Boosting Algorithms](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/colt08.pdf)

### Optimal Margin Distribution Clustering

Maximum margin clustering (MMC), which borrows the `large margin` heuristic from support vector machine (SVM),
has achieved more accurate results than traditional clustering methods. 
The intuition is that, for a good clustering, when labels are assigned to different clusters, SVM can achieve a large minimum margin on this data. Recent studies, however, disclosed that maximizing the minimum margin does not necessarily lead to better performance, 
and instead, it is crucial to optimize the `margin distribution`. 
The ODMC (Optimal margin Distribution Machine for Clustering), which tries to cluster the data and achieve optimal margin distribution simultaneously. 


- https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/aaai18odmc.pdf
- [Bayesian Maximum Margin Clustering](https://niug1984.github.io/paper/dai_icdm10.pdf)

### Semi-Supervised Optimal Margin Distribution Machines

Semi-supervised support vector machines is an extension of standard support vector machines with unlabeled instances, and the goal is to find a label assignment of the unlabeled instances, 
so that the decision boundary has the maximal minimum margin on both the original labeled instances and unlabeled instances. 
Recent studies, however, disclosed that maximizing the minimum margin does not necessarily lead to better performance, and instead, it is crucial to optimize the margin distribution.
The `ssODM (SemiSupervised Optimal margin Distribution Machine)` tries to assign the labels to unlabeled instances and to achieve optimal margin distribution simultaneously.

- https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/ijcai18ssodm.pdf

## Non-parametric Learning

Parametric statistics| Nonparametric statistics
---|---
Parametric statistics are any statistical tests based on underlying assumptions about data’s distribution. | [Nonparametric statistics are called distribution-free statistics because they are not constrained by assumptions about the distribution of the population. ](https://www.sciencedirect.com/science/article/pii/B9780128047538000087)

- [Parametric and Nonparametric Statistics](https://www.phdstudent.com/thesis-and-dissertation-survival/statistics/parametric-vs-nonparametric-statistics)
- http://www.cs.columbia.edu/~djhsu/research/
- https://www.isye.gatech.edu/users/arkadi-nemirovski
- https://xu-yangyang.github.io/


### Decision Tree

Decision tree is based on greedy search originally.
Decision tree is a typical recursive partitioning process.
Here we focus on the [optimal decision tree](https://link.springer.com/article/10.1007/s10994-017-5633-9).

> We are given the training data ${\vec{X}}, {\vec{Y}}$, containing n observations ${\vec{x}}_i, y_i$, $i = 1, \ldots , n$, each with $p$ features ${\vec{x}}_i \in \mathbb {R}^p$ and a class label $y_i \in \{1, \ldots , K\}$ indicating which of K possible labels is assigned to this point. We assume without loss of generality that the values for each dimension across the training data are normalized to the 0–1 interval, meaning each ${\vec{x}}_i \in {[0, 1]}^p$.

> Decision tree methods seek to recursively partition ${[0, 1]}^p$ to yield a number of hierarchical, disjoint regions that represent a classification tree. The final tree is comprised of branch nodes and leaf nodes:

> * Branch nodes apply a split with parameters ${\vec{a}}$ and $b$. For a given point $i$, if ${\vec{a}}^T {\vec{x}}_i < b$ the point will follow the left branch from the node, otherwise it takes the right branch. A subset of methods, including CART, produce univariate or axis-aligned decision trees which restrict the split to a single dimension, i.e., a single component of ${\vec{a}}$ will be 1 and all others will be 0.

> * Leaf nodes are assigned a class that will determine the prediction for all data points that fall into the leaf node. The assigned class is almost always taken to be the class that occurs most often among points contained in the leaf node.

Here we focus on optimal decision tree.

- [Decision Tree](https://docs.rapidminer.com/9.2/studio/operators/modeling/predictive/trees/parallel_decision_tree.html)
- https://jack.dunn.nz/
- http://www.mit.edu/~agniorf/
- https://github.com/benedekrozemberczki/awesome-decision-tree-papers
- [Learning Optimal Classification Trees Using a Binary Linear Program Formulation](https://www.aaai.org/ojs/index.php/AAAI/article/view/3978/3856)
- [Learning Optimal Classification Trees: Strong Max-Flow Formulations](https://deepai.org/publication/learning-optimal-classification-trees-strong-max-flow-formulations)

[The problem that has plagued decision tree algorithms since their inception is their lack of optimality, or lack of guarantees of closeness to optimality: decision tree algorithms are often greedy or myopic, and sometimes produce unquestionably suboptimal models. Hardness of decision tree optimization is both a theoretical and practical obstacle, and even careful mathematical programming approaches have not been able to solve these problems efficiently. This work introduces the first practical algorithm for optimal decision trees for binary variables. The algorithm is a co-design of analytical bounds that reduce the search space and modern systems techniques, including data structures and a custom bit-vector library. We highlight possible steps to improving the scalability and speed of future generations of this algorithm based on insights from our theory and experiments.](https://papers.nips.cc/paper/8947-optimal-sparse-decision-trees)

#### Dyadic Decision Trees

The `dyadic decision trees (DDTs)` attain nearly optimal (in a minimax sense)
rates of convergence for a broad range of classification problems.

A dyadic decision tree (DDT) is a decision tree that divides the input space by means of axis-orthogonal dyadic splits.

Dyadic decision trees are constructed by minimizing a complexity penalized empirical risk over an appropriate family of dyadic partitions.
The penalty is data-dependent and comes from a new error deviance bound for trees. 
This new bound is tailored specifically to DDTs and therefore involves substantially smaller constants than bounds derived in more general settings. 
The bound in turn leads to an oracle inequality from which rates of convergence are derived.

<img src="https://image2.slideserve.com/4309520/slide17-n.jpg" width="76%"/>

- [Minimax-Optimal Classification with Dyadic Decision Tree](https://nowak.ece.wisc.edu/ddt.pdf)
- http://web.eecs.umich.edu/~cscott/pubs/thesis.pdf
- http://web.eecs.umich.edu/~cscott/pubs.html
- https://nowak.ece.wisc.edu/
- [Optimal Dyadic Decision Trees](http://doc.ml.tu-berlin.de/publications/publications/BlaSchRozMue07.pdf)

#### Optimal Classification Trees

Motivation: MIO is the natural form for the Optimal Tree problem: 
* Decisions: Which variable to split on, which label to predict for a region 
* Outcomes: Which region a point ends up in, whether a point is correctly classiﬁed.

It is claimed  that optimal decision tree is as powerful as the deep neural networks.

> Theorem: [Optimal classiﬁcation and regression trees with hyperplanes are as powerful as classiﬁcation and regression (feedforward, convolutional and recurrent) neural networks, that is given a NN we can ﬁnd a OCT-H (or ORT-H) that has the same in sample performance.](https://orfe.princeton.edu/pdo/sites/orfe.princeton.edu.pdo/files/Bertsimas_PDO.pdf)


- http://www.mit.edu/~zhuo/
- [Optimal classification trees](http://www.mit.edu/~dbertsim/papers/Machine%20Learning%20under%20a%20Modern%20Optimization%20Lens/Optimal_classification_trees.pdf)
- https://docs.interpretable.ai/stable/OptimalTrees/quickstart/regression/
- http://jack.dunn.nz/papers/OptimalClassificationTrees.pdf

#### Optimal Prescriptive Trees

we propose a tree based algorithm called optimal prescription tree (OPT) that uses either constant or linear models in the leaves of the tree in order to predict the counterfactuals and to assign optimal treatments to new samples. 
We propose an objective function that balances optimality and accuracy. OPTs are interpretable, highly scalable, accommodate multiple treatments and provide high quality prescriptions. We report results involving
synthetic and real data that show that optimal prescriptive trees either outperform or are comparable with several state of the art methods. 
Given their combination of interpretability, scalability, generalizability and
performance, OPTs are an attractive alternative for personalized decision making in a variety of areas such
as online advertising and personalized medicine

- https://docs.interpretable.ai/stable/OptimalTrees/quickstart/prescription/
- https://dspace.mit.edu/handle/1721.1/119280
- http://jack.dunn.nz/papers/OptimalPrescriptiveTrees.pdf

#### Optimal Sparse Decision Trees

The algorithm is a co-design of analytical bounds that reduce the search space and modern systems techniques,
including data structures and a custom bit-vector library.



- https://www.seltzer.com/margo/research
- https://users.cs.duke.edu/~cynthia/
- https://www.andrew.cmu.edu/user/xiyanghu/
- https://arxiv.org/pdf/1904.12847.pdf
- https://github.com/xiyanghu/OSDT

#### Optimal Randomized Classification Trees

> Traditionally, a greedy approach has been used to build the trees, yielding a very fast training process; however, controlling sparsity (a proxy for interpretability) is challenging. In recent studies, optimal decision trees, where all decisions are optimized simultaneously, have shown a better learning performance, especially when oblique cuts are implemented. 



- http://cermics.enpc.fr/~parmenta/frejus/Molero.pdf
- https://arxiv.org/pdf/2002.09191.pdf
- https://www.researchgate.net/profile/Rafael_Blanquero

#### Ensemble of Optimal Trees


- [An Ensemble of Optimal Trees for Class Membership Probability Estimation](https://research-information.bris.ac.uk/ws/portalfiles/portal/120113774/An_Ensemble_of_Optimal_Trees_for_Class_Membership_probability_estimation.pdf)
- https://rdrr.io/cran/OTE/man/OTE-package.html
- [Ensemble of optimal trees, random forest and random
projection ensemble classification](https://link.springer.com/content/pdf/10.1007%2Fs11634-019-00364-9.pdf)
- [An Ensemble of Optimal Trees for Classification and
Regression (OTE)](http://repository.essex.ac.uk/17595/1/OTE-Khan-et-al-preprint-BLG-DRC-17Sept16.pdf)
- http://repository.essex.ac.uk/21533/

## Deep Learning

Deep learning is the powerful representation learning which can train the classifiers with automatic feature extraction.

<img src="http://people.csail.mit.edu/guanghe/locally_linear_files/teaser.png" width="80%"/>

- [Security for Artificial Intelligence](https://aisecure.github.io/PROJECTS/aml.html)
- http://cvlab.cse.msu.edu/
- http://people.csail.mit.edu/davidam/
- http://www.otnira.com/
- http://www.dei.unipd.it/~fisch/
- [Deep Learning and MIP](http://www.dei.unipd.it/~fisch/papers/deep_neural_networks_and_mixed_integer_linear_optimization.pdf)
- [LassoNet](https://arxiv.org/pdf/1907.12207.pdf)
- https://github.com/TimDettmers/sparse_learning
- https://www.mins.ee.ethz.ch/people/show/boelcskei
- https://mat.univie.ac.at/~grohs/
- https://hangzhang.org/CVPR2020/
- https://github.com/hussius/deeplearning-biology
- https://statsinthewild.com/
- https://alammehwish.github.io/dl4kg_eswc_2020/
- http://dorienherremans.com/dlm2017/
- http://cseweb.ucsd.edu/~haosu/
- https://www.norbertwiener.umd.edu/jubilee/Slides/Shen.pdf

There are diverse projects and workshops on deep learning such as the following shown.
- [On Statistical Thinking in Deep Learning](http://www.stats.ox.ac.uk/~teh/research/jsm2019/OnStatisticalThinkinginDeepLearning.pdf)
- https://geometric-relational-dl.github.io/
- https://www.eurandom.tue.nl/event/yes-x-deep-learning-foundations/
- http://cobweb.cs.uga.edu/~shengli/Tusion2019.html
- https://deeplearning-math.github.io/2018spring.html
- http://ubee.enseeiht.fr/skelneton/
- https://www.ece.ufl.edu/deep-learning-workshop-2020/
- http://luoping.me/

[Our understanding of modern neural networks lags behind their practical successes. This growing gap poses a challenge to the pace of progress in machine learning because fewer pillars of knowledge are available to designers of models and algorithms. This workshop aims to close this understanding gap. We solicit contributions that view the behavior of deep nets as natural phenomena, to be investigated with methods inspired from the natural sciences like physics, astronomy, and biology. We call for empirical work that isolates phenomena in deep nets, describes them quantitatively, and then replicates or falsifies them.](https://deep-phenomena.org/)

- https://deep-phenomena.org/
- https://github.com/PacktWorkshops/The-Deep-Learning-Workshop
- https://users.cs.duke.edu/~rongge/stoc2018ml/stoc2018ml.html
- http://ubee.enseeiht.fr/skelneton/workshop.html
- https://www.ieee-security.org/TC/SP2020/workshops.html
- [Limitations of Deep Learning Workshop in Sestri Levante, Italy, June 25-27, 2019](https://cbmm.mit.edu/knowledge-transfer/workshops-conferences-symposia/limitations-deep-learning-workshop)

Here we would not pay attention of the architecture of the deep neural networks.


### Oblique Decision Trees from Derivatives of ReLU Networks

> We show how neural models can be used to realize piece-wise constant functions such as decision trees. 
> It is  proved the equivalence between the class of oblique decision trees and these proposed locally constant neural models.

The work opens up many avenues for future work, from building representations from the derivatives of neural models to the incorporation of more structures, such as the inner randomization of random forest.

- https://people.csail.mit.edu/tommi/tommi.html
- [OBLIQUE DECISION TREES FROM DERIVATIVES OF RELU NETWORKS](https://openreview.net/pdf?id=Bke8UR4FPB)
- https://github.com/guanghelee/iclr20-lcn
- http://people.csail.mit.edu/guanghe/locally_linear

### Lifted Proximal Operator Machines

`LPOM` is block multi-convex in all layer-wise weights and activations. 
This allows us to use block coordinate descent to update the layer-wise weights and activations.

Then the optimality condition of the following minimization problem:
$$\arg\min_{\textbf{X}^{i}}\textbf{1}^Tf(\textbf{X}^{i})\textbf{1}+\frac{1}{2}\|\textbf{X}^{i}-\textbf{W}^{i-1}\textbf{X}^{i-1}\|_F^2$$
is 
$$\mathbb{0}\in \phi^{-1}(\textbf{X}^{i})-\textbf{W}^{i-1}\textbf{X}^{i-1}$$
where where $\textbf{1}$ is an all-one column vector; and $\textbf{X}^{i}$ are matrix; $f(x)=\int_{0}^{x}(\phi^{-1}(y)-y)\mathrm{d} y$.

So the optimal solution of this condition is 
$$\textbf{X}^{i}=\phi(\textbf{W}^{i-1}\textbf{X}^{i-1}).$$

- [Lifted Proximal Operator Machines](https://zhouchenlin.github.io/Publications/2019-AAAI-LPOM.pdf)
- [aaai19_lifted_proximal_operator_machines/](https://zero-lab-pku.github.io/publication/gengzhengyang/aaai19_lifted_proximal_operator_machines/)


### SpiNNaker

> SpiNNaker (a contraction of Spiking Neural Network Architecture) is a million-core computing engine whose flagship goal is to be able to simulate the behaviour of aggregates of up to a billion neurons in real time. It consists of an array of ARM9 cores, communicating via packets carried by a custom interconnect fabric. The packets are small (40 or 72 bits), and their transmission is brokered entirely by hardware, giving the overall engine an extremely high bisection bandwidth of over 5 billion packets/s. Three of the principle axioms of parallel machine design - memory coherence, synchronicity and determinism - have been discarded in the design without, surprisingly, compromising the ability to perform meaningful computations. A further attribute of the system is the acknowledgment, from the initial design stages, that the sheer size of the implementation will make component failures an inevitable aspect of day-to-day operation, and fault detection and recovery mechanisms have been built into the system at many levels of abstraction.


<img src="http://apt.cs.manchester.ac.uk/Images/mesh_ctiff.jpg" width="60%"/>

- http://apt.cs.manchester.ac.uk/projects/SpiNNaker/architecture/
- http://meseec.ce.rit.edu/756-projects/fall2015/2-1.pdf
- http://apt.cs.manchester.ac.uk/projects/SpiNNaker/apps_neural/
- https://www.nowpublishers.com/article/BookDetails/9781680836523
- [Deep Learning Workshop](https://emit.tech/wp-content/uploads/2019/03/DeepLearning_Workshop_EMiT2019.pdf)

### Industrial Deep Learning

In the increasingly digitalized world, it is of utmost importance for various applications to harness the ability to process, understand, and exploit data collected from the Internet. 
For instance, in customer-centric applications such as personalized recommendation, online advertising, and search engines, interest/intention modeling from customers’ behavioral data can not only significantly enhance user experiences but also greatly contribute to revenues. 
Recently, we have witnessed that Deep Learning-based approaches began to empower these internet- scale applications by better leveraging the massive data. 
However, the data in these internet-scale applications are high dimensional and extremely sparse, which makes it different from many applications with dense data such as image classification and speech recognition where Deep Learning-based approaches have been extensively studied. 
For example, the training samples of a typical click-through rate (CTR) prediction task often involve billions of sparse features, how to mine, model and inference from such data becomes an interesting problem, and how to leverage such data in Deep Learning could be a new research direction. The characteristics of such data pose unique challenges to the adoption of Deep Learning in these applications, including modeling, training, and online serving, etc. 
More and more communities from both academia and industry have initiated the endeavors to solve these challenges. This workshop will provide a venue for both the research and engineering communities to discuss the challenges, opportunities, and new ideas in the practice of Deep Learning on high-dimensional sparse data.

- https://dlp-kdd.github.io/
- https://ears2018.github.io/
- https://wsdm2019-dapa.github.io/
- https://www.adkdd.org/
- https://workshop-edlcv.github.io/
- [Compact Deep Neural Network Representation with Industrial Applications](https://openreview.net/group?id=NIPS.cc/2018/Workshop/CDNNRIA#accepted-papers)
- [New Deep Learning Techniques](https://www.ipam.ucla.edu/programs/workshops/new-deep-learning-techniques/?tab=schedule)
- https://realworldml.github.io/

##  Certifiably provably optiaml methods

> A high quality logistic regression model contains various desirable properties: predictive power, interpretability, significance, robustness
to error in data and sparsity, among others.