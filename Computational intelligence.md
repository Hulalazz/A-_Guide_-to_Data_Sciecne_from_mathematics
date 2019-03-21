
# Computational Intelligence

Computational intelligence is rooted in the artificial neural network and evolutionary algorithms.
[No free lunch theorem](https://www.wikiwand.com/en/No_free_lunch_in_search_and_optimization) implies  that searching for the 'best' general purpose black box optimization algorithm is irresponsible as no such procedure is theoretically possible.

- https://cse.buffalo.edu/~rapaport/575/F01/ai.intro.html
- https://arxiv.org/pdf/1307.4186.pdf

Swarm intelligence based algorithms | Bio-inspired (not SI-based) algorithms|Physics and Chemistry based algorithms|Others
---------- | -------------|-----------|---------------
Ant Colony Optimization | Genetic Algorithm |Simulated Annealing|Fuzzy Intelligence
Particle Swarm Optimization | Differential Evolution| Gravitational Search |Tabu Search
More|More|More|More


See the article [A Brief Review of Nature-Inspired Algorithms for Optimization](https://arxiv.org/pdf/1307.4186.pdf) for more nature-inspired optimization algorithms or [Clever Algorithms: Nature-Inspired Programming Recipes](http://www.cleveralgorithms.com/nature-inspired/introduction.html) for nature-inspired AI.

|Computational Intelligence|
|---|
|![CI](https://cis.ieee.org/images/files/slideshow/04mci04-cover1.jpg)|

In [IEEE Computational Intelligence Society](https://cis.ieee.org/about/what-is-ci), computational intelligence is introduced as follows:
> Computational Intelligence (CI) is the theory, design, application and development of biologically and linguistically motivated computational paradigms. Traditionally the three main pillars of CI have been Neural Networks, Fuzzy Systems and Evolutionary Computation. However, in time many nature inspired computing paradigms have evolved. Thus CI is an evolving field and at present in addition to the three main constituents, it encompasses computing paradigms like ambient intelligence, artificial life, cultural learning, artificial endocrine networks, social reasoning, and artificial hormone networks. CI plays a major role in developing successful intelligent systems, including games and cognitive developmental systems. Over the last few years there has been an explosion of research on Deep Learning, in particular deep convolutional neural networks. Nowadays, deep learning has become the core method for artificial intelligence. In fact, some of the most successful AI systems are based on CI.

* https://www.tecweb.org/styles/gardner.html
* http://www.mae.cuhk.edu.hk/~cil/index.htm
* http://cogprints.org/
* http://ai.berkeley.edu/lecture_videos.html
* http://www.comp.hkbu.edu.hk/~hknnc/index.php
* https://www.cleverism.com/artificial-intelligence-complete-guide/
* https://www.eurekalert.org/pub_releases/2018-03/uom-csw032918.php
* [Computational Intelligence: An Introduction](http://papers.harvie.cz/unsorted/computational-intelligence-an-introduction.pdf)


Some packages:

- [ ] https://github.com/facebookresearch/nevergrad
- [ ] https://github.com/DEAP/deap
- [ ] https://github.com/SioKCronin/swarmopt
- [ ] https://cilib.net/
- [ ] http://fuzzy.cs.ovgu.de/nefcon/

Some courses:

- [ ] http://www.cs.armstrong.edu/saad/csci8100/
- [ ] http://www.cis.pku.edu.cn/faculty/system/tanying/CI/CIlecture1.pdf
- [ ] https://people.eecs.berkeley.edu/~russell/papers/mitecs-computational-intelligence.pdf
- [ ] http://www.soft-computing.de/def.html
- [ ] http://mat.uab.cat/~alseda/MasterOpt/index.html


Optimization is to minimize the cost or maximize the utility. Particularly, we call it as numerical optimization if the cost or utility can be formulated in mathematical form.
Some general principles are written in the form of optimization such as **Maximum Entropy Principle**.
It raises in operation research, numerical analysis, computational mathematics and more technological applications such as the engine design.
It is clear that optimization relies on at least perspectives *the objective to optimize* and *the approach to optimize the objective* with or without some constraints.

Another topic on computational intelligence is to search some object or goals in complex and uncertain system.
Optimization is to search the best or optimal configuration in some sense. Search is to explore the system with some fuzzy goals.

Evolution or adaptation is in the level of population, which leads to this diverse earth. And this is an important criteria which differs from other entities. It is related with transfer learning.
Like in ecosystem, what the computational learners can do if the system changes a lot in order to survive?  In numerical optimization, if the objective function changes, we may consider different optimization methods. 


## Swarm Intelligence(SI)

Swarm intelligence is the study of computational systems inspired by the 'collective intelligence'.
Collective Intelligence emerges through the cooperation of large numbers of homogeneous agents in the environment. Swarm intelligence believes that scale matters. And agents in different scale are different: more agents, more power. 


- [ ] Particle Swarm Optimization(PSO)
- [ ] Accelerated PSO
- [ ] Ant Colony Optimization
- [ ] Fish swarm/school

**Particle Swarm Optimization(PSO)**

Individuals in a particle swarm follow a very simple behavior: to emulate the success of
neighboring individuals and their own successes. The collective behavior that emerges
from this simple behavior is that of discovering optimal regions of a high dimensional search space.

In simple terms, the particles are "flown" through a multidimensional search space, where the position
of each particle is adjusted according to its own experience and that of its neighbors.
The position of the particle is changed
by adding a velocity, $v_i(t)$, to the current position, i.e.
$$x_{i}(t+1) = x_{i}(t)+v_i(t+1)$$

with $x_0\sim Unif[x_{min}, x_{max}]$ and $x_{i}(t)$ is the position of particle $i$ at time ${t}$.

Each particle’s velocity is updated using this
equation at time $t$:

$$
v_{i+1}(t)=\omega v_{i}(t)+c_1r_1[\hat{x}_{i}(t)-x_{i}(t)]+c_2r_2[g(t)-x_i(t)]
$$
where the notations will explain in the following table:

|||||||
|--|----|-----|-----|----|--------|
|$\omega$|the inertial coefficient usually in $[0.8, 1.2]$|$c_1,c_2$| acceleration coefficients in [0,2]|$r_1,r_2$|random values in [0,1] at each update|
|$v_i(t)$|the particle’s velocity at time ${t}$|$\hat{x}_{i}(t)$|the particle’s individual best solution as of time${t}$|$g(t)$| the swarm’s best solution as of time $t$|

Inertia Component $\omega v_{i}(t)$:

- Keeps the particle moving in the same direction it was originally heading.
- Lower values speed up convergence, higher values encourage exploring the search space.

Cognitive Component $c_1r_1[\hat{x}_{i}(t)-x_{i}(t)]$:

- Acts as the particle’s memory, causing it to return to its individual best regions of the search space;
- Cognitive coefficient $c_1$ usually close to 2
- Coefficient limits the size of the step the particle takes toward its individual best $\hat{x}_i(t)$.

Social Component $c_2r_2[g(t)-x_i(t)]$:

- Causes the particle to move to the best regions the swarm has found so far;
- Social coefficient $c_2$ usually close to 2;
- Coefficient limits the size of the step the particle takes toward the global best $g$.

It does not compute the gradient of the function. And like accelerated gradient, it uses the previous iteration result.

![](https://cssanalytics.files.wordpress.com/2013/09/pso-graphic.png)


- http://www.cs.armstrong.edu/saad/csci8100/pso_slides.pdf
- http://www.swarmintelligence.org/tutorials.php
- https://github.com/tisimst/pyswarm
- http://mnemstudio.org/particle-swarm-introduction.htm
- https://arxiv.org/ftp/arxiv/papers/1804/1804.05319.pdf

**Ant Colony Optimization**

It is an meta-heuristic optimization for searching for optimal path in the graph based on
behaviour of ants seeking a path between their colony and
source of food.

|Overview of the Concept|
|---|
|1. Ants navigate from nest to food source.|
|2. Shortest path is discovered via pheromone trails.|
|3. Each ant moves at random.|
|4. Pheromone is deposited on path.|
|5. More pheromone on path increases probability of path being followed.|
![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Aco_branches.svg/2000px-Aco_branches.svg.png)

>
Overview of the System:
+ Virtual trail accumulated on path segments;
+ Path selected at random based on amount of "trail" present on possible paths from starting node ;
+ Ant reaches next node, selects next path;
+ Continues until reaches starting node；
+ Finished tour is a solution；
+ Tour is analyzed for optimality.
>
*****
An ant will move from node i to node j with probability
$$
p_{ij} = \frac{(\tau_{ij}^{\alpha})(\eta_{ij}^{\beta})}{\sum (\tau_{ij}^{\alpha})(\eta_{ij}^{\beta})}
$$
 where  $\tau_{ij},\eta_{ij}$ are  the amount and desirablity of pheromone on edge i, j;  $\alpha, \beta$ are parameters to control the influence of $\tau_{ij},\eta_{ij}$.


 Amount of pheromone is updated according to the equation
$$
 \tau_{ij} = (1-\rho)\tau_{ij} +\Delta \tau_{ij}
$$
where $\Delta\tau_{ij}$ is the amount of pheromone deposited, typically given by
$$
\Delta\tau^{k}_{ij} =
\begin{cases}
\frac{1}{L_k}, & \text{if ant $k$ travels from on edge i,j}\\
0, &\text{otherwise}
\end{cases}
$$
where $L_k$ is the cost of the ${k}$ th ant’s tour (typically length).

The three most successful special cases of the ACO metaheuristic are: Ant System, Ant Colony System (ACS), and MAX-MIN Ant System (MMAS).

**Ant System**

Pheromone values are updated by all the ants that have
completed the tour:

$$
\tau_{ij} = (1-\rho)\tau_{ij} +\sum_{k=1}^{m}\Delta \tau_{ij}^{k}.
$$

**Ant Colony System**

Ants in ACS use the pseudorandom proportional rule:

+ Probability for an ant to move from city i to city j depends on a random variable ${q}$ uniformly distributed over [0, 1], and a parameter $q_0$.
+ If $q\leq q_0$, then, among the feasible components, the component that maximizes the product $\tau_{il}\eta_{il}^{\beta}$ is chosen, otherwise the same equation as in Ant System is used.
+ This rule favours exploitation of pheromone information.

Each ant applies it only to the last edge traversed

$$
\tau_{ij} = (1-\rho)\tau_{ij} +\phi \tau_0
$$

where $\tau_0$ is the initial value of the pheromone.

*****
- http://mat.uab.cat/~alseda/MasterOpt/ACO_Intro.pdf
- https://www.ics.uci.edu/~welling/teaching/271fall09/antcolonyopt.pdf
- http://www.aco-metaheuristic.org/
- http://mathworld.wolfram.com/AntColonyAlgorithm.html

**AFSA (artificial fish-swarm algorithm)**

Let $X=(x_1, x_2, \dots, x_n)$ and $X_{v}=(x_1^v, x_2^v, \dots, x_n^v)$ then this process can be
expressed as follows:
$$
x_i^v = x_i + Visual* rand()\\
X_{next} = X +\frac{X_v - X}{\|X_v - X\|}* Step * rand()
$$

where $rand()$ produces random numbers between zero and 1, Step is the step length, and
$x_i$ is the optimizing variable, n is the number of variables. The variables include: ${X}$ is the current position of the AF, Step is the
moving step length, Visual represents the visual distance,

- https://link.springer.com/article/10.1007/s10462-012-9342-2
- http://cloudbus.org/~adel/pdf/AIR2014.pdf
- http://www.mecs-press.org/ijisa/ijisa-v2-n1/IJISA-V2-N1-6.pdf
- http://www.scielo.br/pdf/lajss/v11n1/09.pdf

## Bio-inspired (not SI-based) algorithms

In machine learning, it is always thought in the *representation + evaluation + optimization* way. For specific  tasks, we represent the conditions or inputs in binary code so that the computer can handle them; then a family of mathematical model is choose to transform and process the inputs. And an evaluation as a performance metric is used to show how good a model is such as the loss function in supervised machine learning.
An optimization method is used to find the best model with respect to the specified evaluation. Usually these are not dynamics.

However, it is not the fact in nature. It is not necessary to compute the gradient or the expected regards before making any decision in our brain.
See the differences between the machine learning and  biological learning in their websites:
[Machine Learning Group](http://mlg.eng.cam.ac.uk/) and [Biological Learning Group](http://cbl.eng.cam.ac.uk/Public/BlgHome).



![cam.ac.uk](http://cbl.eng.cam.ac.uk/pub/Public/BlgHome/igp_96ba957c7dfc830bde4791e07e305699_blg-logo.pngs)

In this section, we will focus on the following  topics.

- [ ] Differential evolution
- [ ] Genetic Algorithms
- [ ] Biogeography-based optimization
- [ ] [Neuroscience-Inspired Artificial Intelligence](https://www.ncbi.nlm.nih.gov/pubmed/28728020)

[Terrence J. Sejnowski](https://cnl.salk.edu/) in *The Deep Learning Revolution* asserts that
>The only existence proof that any of the hard problems in artificial intelligence can be solved is the fact that, through evolution, nature has already solved them.

However, the evolution itself is not clear for us since the [Darwin's Theory Of Evolution](https://www.darwins-theory-of-evolution.com/) was proposed.
[Natural selection](https://www.wikiwand.com/en/Natural_selection) is really based on the principles of physics and chemistry, but not all animals live in the world understand physics or chemistry.
From different levels and perspectives, we would research the adaptiveness, intelligence and evolution.

Deep learning is in the level of networks.
Artificial Immune System is in the level of systems.

![Investigation level](https://cnl.salk.edu/images/figures/LevelsOfInvestigation.svg)
- http://biology.ucsd.edu/research/faculty/tsejnowski
- http://learning.eng.cam.ac.uk/Public/

**Evolutionary Computation(EC)**

Evolutionary computation: is likely to be  the next major transition of artificial intelligence.
In EC, core concepts from evolutionary biology — inheritance, random variation, and selection — are harnessed in algorithms that are applied to complex computational problems. The field of EC, whose origins can be traced back to the 1950s and 60s, has come into its own over the past decade. EC techniques have been shown to solve numerous difficult problems from widely diverse domains, in particular producing human-competitive machine intelligence.
[EC presents many important benefits over popular deep learning methods includes a far lesser extent on the existence of a known or discoverable gradient within the search space.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5534026/)

- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5534026/
- [Evolutionary Computation](https://courses.cs.washington.edu/courses/cse466/05sp/pdfs/lectures/10-EvolutionaryComputation.pdf),
- [Introduction to Evolutionary Computing  by A.E. Eiben and J.E. Smith](https://www.cs.vu.nl/~gusz/ecbook/ecbook-course.html),
- [Understanding Evolution](https://evolution.berkeley.edu/evolibrary/home.php )
- https://github.com/lmarti/evolutionary-computation-course
- http://lmarti.com/aec-2014

**Differential Evolution(DE)**

Differential evolution is capable of handling nondifferentiable, nonlinear and multimodal objective functions. It has been used to train neural networks having real and constrained integer weights.

In a population of potential solutions within an n-dimensional search space, a fixed number of vectors are randomly initialized, then evolved over time to explore the search space and to locate the minima of the objective function.

At each iteration, called a generation, new vectors are generated by the combination of vectors randomly chosen from the current population (mutation). The outcoming vectors are then mixed with a predetermined target vector. This operation is called recombination and produces the trial vector. Finally, the trial vector is accepted for the next generation if and only if it yields a reduction in the value of the objective function. This last operator is referred to as a selection. Adopted from [MathWorld](http://mathworld.wolfram.com/DifferentialEvolution.html).

- [DE](https://ir.lib.uwo.ca/cgi/viewcontent.cgi?article=1022&context=wrrr)
- [Differential Evolution (DE) for Continuous Function Optimization (an algorithm by Kenneth Price and Rainer Storn)](http://www1.icsi.berkeley.edu/~storn/code.html)

**Genetic Algorithm**

> Genetic Algorithms (GAs) mimic natural selection in order to solve several types of problems, including function optimization, scheduling, and combinatorial problems. To do this, the GA maintains and "evolves" over time a population of individual solutions to the problem at hand. Each generation of individual solutions (hopefully) becomes better and better, until some individual in the population represents an acceptable solution. Two of the most important characteristics of GAs are the representation used and the genetic operators employed.

> In GAs, representation is how we encode solutions to a problem. For example, individuals in the population of a GA may be simple binary strings, or they may be a series of integers or floating point numbers in some specified range.
> Aside from representation, another facet that may affect GA performance is the set of genetic operators used. Genetic operators are the means by which the population evolves towards a solution. In a simple GA, these operators may be classified as crossover, mutation, and selection operators.



*****
- [Genetic Algorithms](https://www.doane.edu/evolutionary-computation-0).
- http://www.genetic-programming.com/gecco2004tutorial.pdf
- http://mathworld.wolfram.com/GeneticAlgorithm.html
- http://mat.uab.cat/~alseda/MasterOpt/Dziedzic.GA_intro.pdf
- http://mat.uab.cat/~alseda/MasterOpt/Beasley93GA1.pdf
- http://mat.uab.cat/~alseda/MasterOpt/Beasley93GA2.pdf

**Artificial Immune Sysytems**

Artificial immune systems (AISs) are inspired by biological immune systems and mimic these by means of computer simulations. They are seen with interest from immunologists as well as engineers. Immunologists hope to gain a deeper understanding of the mechanisms at work in biological immune systems. Engineers hope that these nature-inspired systems prove useful in very difficult computational tasks, ranging from applications in intrusion-detection systems to general optimization. Moreover, computer scientists identified artificial immune systems as another example of a nature-inspired randomized search heuristic (like evolutionary algorithms, ant colony optimization, particle swarm optimization, simulated annealing, and others) and aim at understanding their potential and limitations. While the relatively new field has its successful applications and much potential its theoretical foundation is still in its infancy. Currently there are several not well connected strands within AIS theory, not even a general agreement on what the central open problems are, and only a weak connection between AIS theory and AIS applications. The main goals of the proposed seminar include bringing together computer scientists and engineers to strengthen the connections within AIS theory, connections to other researchers working on the theory of randomized search heuristics, and to improve connectivity between AIS theory and applications.
[Adopted from Artificial Immune Systems seminar](https://www.dagstuhl.de/en/program/calendar/semhp/?semnr=11172)

- http://www.artificial-immune-systems.org/
- https://www.sciencedirect.com/science/article/pii/S1568494610002723

**Neuroscience-Inspired Artificial Intelligence**

|AI and Beyond|
|:---:|
|![https://zhuanlan.zhihu.com/p/20727283s](https://pic1.zhimg.com/8d02ad62036808353f181a4996aa52e6_1200x500.jpg)|

- [Neuroscience-Inspired Artificial Intelligence](https://www.ncbi.nlm.nih.gov/pubmed/28728020)
- https://www.cbicr.tsinghua.edu.cn/
- https://blog.csdn.net/u013088062/article/details/50489674
- https://zhuanlan.zhihu.com/p/23965227
- https://zhuanlan.zhihu.com/p/30190719
- https://zhuanlan.zhihu.com/p/20726556
- https://zhuanlan.zhihu.com/p/23782226
- https://zhuanlan.zhihu.com/p/23804250
- https://zhuanlan.zhihu.com/p/23979871
- https://www.zhihu.com/question/59408117/answer/164972455
- http://www.psy.vanderbilt.edu/courses/hon1850/Brain_Structure_Function_Chapter.pdf

## Physics and Chemistry based algorithms

- [ ] Simulated annealing
- [ ] Gravitational search
- [ ] Stochastic diffusion search

**Simulated Annealing(SA)**

Simulated Annealing (SA) is an effective and general form of optimization.  It is useful in finding global optima in the presence of large numbers of local optima.  "Annealing" refers to an analogy with thermodynamics, specifically with the way that metals cool and anneal.  Simulated annealing uses the objective function of an optimization problem instead of the energy of a material.

Implementation of SA is surprisingly simple.  The algorithm is basically hill-climbing except instead of picking the best move, it picks a random move.  If the selected move improves the solution, then it is always accepted.  Otherwise, the algorithm makes the move anyway with some probability less than 1.  The probability decreases exponentially with the "badness" of the move, which is the amount $\Delta E$ by which the solution is worsened (i.e., energy is increased.)

- http://www.theprojectspot.com/tutorial-post/simulated-annealing-algorithm-for-beginners/6
- http://mathworld.wolfram.com/SimulatedAnnealing.html
- http://www.cs.cmu.edu/afs/cs.cmu.edu/project/learn-43/lib/photoz/.g/web/glossary/anneal.html
- http://120.52.51.16/web.mit.edu/dbertsim/www/papers/Optimization/Simulated%20annealing
- http://mat.uab.cat/~alseda/MasterOpt/ComprehensiveSimulatedAnnealing.pdf
- 

**Gravitational Search Algorithms(GSA)**

Gravitational Search Algorithms (GSA) are heuristic optimization evolutionary algorithms based on Newton's law of universal gravitation and mass interactions.
GSAs are among the most recently introduced techniques that are not yet heavily explored. They are adapted to the cell placement problem, and it is shown its efficiency in producing high quality solutions in reasonable time. Adopted from [On The Performance of the Gravitational Search Algorithm](https://thesai.org/Downloads/Volume4No8/Paper_11-On_The_Performance_Of_The_Gravitational_Search_Algorithm.pdf)

- http://home.ijasca.com/data/documents/ijasc08_published.pdf
- https://www.sciencedirect.com/science/article/pii/S0020025509001200

**Stochastic Diffusion Search(SDS)**

In contrast to many nature-inspired algorithms, SDS has a strong mathematical framework describing its behaviour and convergence.

- http://www.scholarpedia.org/article/Stochastic_diffusion_search
- http://www.doc.gold.ac.uk/~mas02gw/MSCI11/2010/1/SDS_Review_27%20Sep%202010.pdf

**Fireworks Algorithm**

- http://www.cil.pku.edu.cn/research/fwa/publication/IntroducetoFireworksAlgorithm.pdf
- http://adsabs.harvard.edu/abs/2018EnOp...50.1829G
- https://msdn.microsoft.com/en-us/magazine/dn857364.aspx


## Others

- [ ] Fuzzy Intelligence
- [ ] Tabu Search
- [ ] Differential search algorithm
- [ ] Backtracking optimization search

***

1. https://readyforai.com/
2. http://cleveralgorithms.com/
3. https://course.elementsofai.com/
4. https://intelligence.org/blog/
5. https://github.com/Siyeong-Lee/CIO
6. https://github.com/owainlewis/awesome-artificial-intelligence
7. https://github.com/JordanMicahBennett/Supermathematics-and-Artificial-General-Intelligence
8. https://github.com/gopala-kr/AI-Learning-Roadmap
9. http://emergentcomputation.com/Pseudo.html
10. https://wiki.opencog.org/w/Special:WhatLinksHere/The_Open_Cognition_Project
11. [人工智能(AI)资料大全 - 猿助猿的文章 - 知乎](https://zhuanlan.zhihu.com/p/26183036)
