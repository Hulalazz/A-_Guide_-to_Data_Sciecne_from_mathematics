* https://guides.github.com/features/mastering-markdown/
* https://github.github.com/gfm/
* https://help.github.com/articles/about-writing-and-formatting-on-github/

# Computational Intelligence

Computational intelligence is rooted in the artificial neural network and evolutionary algorithms.
[No free lunch theorem](https://www.wikiwand.com/en/No_free_lunch_in_search_and_optimization) implies  that searching for the ‘best’ general purpose black box optimization algorithm is irresponsible as no such procedure is theoretically possible.

- https://cse.buffalo.edu/~rapaport/575/F01/ai.intro.html
- https://arxiv.org/pdf/1307.4186.pdf

Swarm intelligence based algorithms | Bio-inspired (not SI-based) algorithms|Physics and Chemistry based algorithms|Others
---------- | -------------|-------|----
Ant Colony Optimization | Genetic Algorithm |Simulated Annealing|Fuzzy Intelligence
Particle Swarm Optimization | Differential Evolution| Gravitational Search |Tabu Search
More|More|More|More


See the article [A Brief Review of Nature-Inspired Algorithms for Optimization](https://arxiv.org/pdf/1307.4186.pdf) for more nature-inspired optimization algorithms or [Clever Algorithms: Nature-Inspired Programming Recipes](http://www.cleveralgorithms.com/nature-inspired/introduction.html) for nature-inspired AI.

![CI](https://cis.ieee.org/images/files/slideshow/04mci04-cover1.jpg)

In [IEEE Computtaional Intelligence Society](https://cis.ieee.org/about/what-is-ci), computational intelligence is introduced as follows:
> Computational Intelligence (CI) is the theory, design, application and development of biologically and linguistically motivated computational paradigms. Traditionally the three main pillars of CI have been Neural Networks, Fuzzy Systems and Evolutionary Computation. However, in time many nature inspired computing paradigms have evolved. Thus CI is an evolving field and at present in addition to the three main constituents, it encompasses computing paradigms like ambient intelligence, artificial life, cultural learning, artificial endocrine networks, social reasoning, and artificial hormone networks. CI plays a major role in developing successful intelligent systems, including games and cognitive developmental systems. Over the last few years there has been an explosion of research on Deep Learning, in particular deep convolutional neural networks. Nowadays, deep learning has become the core method for artificial intelligence. In fact, some of the most successful AI systems are based on CI.

* https://www.tecweb.org/styles/gardner.html
* http://www.mae.cuhk.edu.hk/~cil/index.htm
* http://cogprints.org/
* http://ai.berkeley.edu/lecture_videos.html
* http://www.comp.hkbu.edu.hk/~hknnc/index.php
* https://www.cleverism.com/artificial-intelligence-complete-guide/
* https://www.eurekalert.org/pub_releases/2018-03/uom-csw032918.php


Some packages:
- [ ] https://github.com/facebookresearch/nevergrad
- [ ] https://github.com/DEAP/deap
- [ ] https://github.com/SioKCronin/swarmopt

## Swarm Intelligence(SI)

Swarm intelligence is the study of computational systems inspired by the
'collective intelligence'. Collective Intelligence emerges through the cooperation of large numbers of homogeneous agents in the environment.

- [ ] Particle Swarm Optimization(PSO)
- [ ] Accelerated PSO
- [ ] Ant Colony Optimization
- [ ] Fish swarm/school

**Particle Swarm Optimization(PSO)**


- https://github.com/tisimst/pyswarm

**Ant Colony OPtimization**

- http://mat.uab.cat/~alseda/MasterOpt/ACO_Intro.pdf
- https://www.ics.uci.edu/~welling/teaching/271fall09/antcolonyopt.pdf
- http://www.aco-metaheuristic.org/
- http://mathworld.wolfram.com/AntColonyAlgorithm.html

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

**Differential Evolution(DE)**

Differential evolution is capable of handling nondifferentiable, nonlinear and multimodal objective functions. It has been used to train neural networks having real and constrained integer weights.

In a population of potential solutions within an n-dimensional search space, a fixed number of vectors are randomly initialized, then evolved over time to explore the search space and to locate the minima of the objective function.

At each iteration, called a generation, new vectors are generated by the combination of vectors randomly chosen from the current population (mutation). The outcoming vectors are then mixed with a predetermined target vector. This operation is called recombination and produces the trial vector. Finally, the trial vector is accepted for the next generation if and only if it yields a reduction in the value of the objective function. This last operator is referred to as a selection. Adopted from [MathWorld](http://mathworld.wolfram.com/DifferentialEvolution.html).

- [DE](https://ir.lib.uwo.ca/cgi/viewcontent.cgi?article=1022&context=wrrr)
- [Differential Evolution (DE) for Continuous Function Optimization (an algorithm by Kenneth Price and Rainer Storn)](http://www1.icsi.berkeley.edu/~storn/code.html)

**Genetic Algorithm**

> Genetic Algorithms (GAs) mimic natural selection in order to solve several types of problems, including function optimization, scheduling, and combinatorial problems. To do this, the GA maintains and "evolves" over time a population of individual solutions to the problem at hand. Each generation of individual solutions (hopefully) becomes better and better, until some individual in the population represents an acceptable solution. Two of the most important characteristics of GAs are the representation used and the genetic operators employed.

>In GAs, representation is how we encode solutions to a problem. For example, individuals in the population of a GA may be simple binary strings, or they may be a series of integers or floating point numbers in some specified range.
> Aside from representation, another facet that may affect GA performance is the set of genetic operators used. Genetic operators are the means by which the population evolves towards a solution. In a simple GA, these operators may be classified as crossover, mutation, and selection operators.
- [Genetic Algorithms](https://www.doane.edu/evolutionary-computation-0).
- http://mathworld.wolfram.com/GeneticAlgorithm.html

**Artificial Immune Sysytems**

Artificial immune systems (AISs) are inspired by biological immune systems and mimic these by means of computer simulations. They are seen with interest from immunologists as well as engineers. Immunologists hope to gain a deeper understanding of the mechanisms at work in biological immune systems. Engineers hope that these nature-inspired systems prove useful in very difficult computational tasks, ranging from applications in intrusion-detection systems to general optimization. Moreover, computer scientists identified artificial immune systems as another example of a nature-inspired randomized search heuristic (like evolutionary algorithms, ant colony optimization, particle swarm optimization, simulated annealing, and others) and aim at understanding their potential and limitations. While the relatively new field has its successful applications and much potential its theoretical foundation is still in its infancy. Currently there are several not well connected strands within AIS theory, not even a general agreement on what the central open problems are, and only a weak connection between AIS theory and AIS applications. The main goals of the proposed seminar include bringing together computer scientists and engineers to strengthen the connections within AIS theory, connections to other researchers working on the theory of randomized search heuristics, and to improve connectivity between AIS theory and applications.
[Adopted from Artificial Immune Systems seminar](https://www.dagstuhl.de/en/program/calendar/semhp/?semnr=11172)
- http://www.artificial-immune-systems.org/
- https://www.sciencedirect.com/science/article/pii/S1568494610002723


## Physics and Chemistry based algorithms

- [ ] Simulated annealing
- [ ] Gravitational search
- [ ] Stochastic diffusion search

**Simulated Annealing(SA)**

- http://www.theprojectspot.com/tutorial-post/simulated-annealing-algorithm-for-beginners/6
- http://mathworld.wolfram.com/SimulatedAnnealing.html
- http://www.cs.cmu.edu/afs/cs.cmu.edu/project/learn-43/lib/photoz/.g/web/glossary/anneal.html
- http://120.52.51.16/web.mit.edu/dbertsim/www/papers/Optimization/Simulated%20annealing

**Gravitational Search Algorithms(GSA)**

Gravitational Search Algorithms (GSA) are heuristic optimization evolutionary algorithms based on Newton's law of universal gravitation and mass interactions.
GSAs are among the most recently introduced techniques that are not yet heavily explored. They are adapted to the cell placement problem, and it is shown its efficiency in producing high quality solutions in reasonable time. Adopted from [On The Performance of the Gravitational Search Algorithm](https://thesai.org/Downloads/Volume4No8/Paper_11-On_The_Performance_Of_The_Gravitational_Search_Algorithm.pdf)

- http://home.ijasca.com/data/documents/ijasc08_published.pdf
- https://www.sciencedirect.com/science/article/pii/S0020025509001200

**Stochastic Diffusion Search(SDS)**

In contrast to many nature-inspired algorithms, SDS has a strong mathematical framework describing its behaviour and convergence.

- http://www.scholarpedia.org/article/Stochastic_diffusion_search
- http://www.doc.gold.ac.uk/~mas02gw/MSCI11/2010/1/SDS_Review_27%20Sep%202010.pdf

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
