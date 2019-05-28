## Topological Data Analysis

Topological data analysis(TDA) is potential to find better representation of the data.
TDA can visualize the high dimensional data and characterize the intrinsic invariants of the data.
It is close to computational geometry, manifold learning and computational topology.
It is one kind of descriptive representation learning.

As [The NIPS 2012 workshop on Algebraic Topology and Machine Learning](https://sites.google.com/site/nips2012topology/) puts:
> Topological methods and machine learning have long enjoyed fruitful interactions as evidenced by popular algorithms
> like ISOMAP, LLE and Laplacian Eigenmaps which have been borne out of studying point cloud data through the lens of geometry.
> More recently several researchers have been attempting to also understand the algebraic topological properties of data.
> Algebraic topology is a branch of mathematics which uses tools from abstract algebra to study and classify topological spaces.
> The machine learning community thus far has focused almost exclusively on clustering as the main tool for unsupervised data analysis.
> Clustering however only scratches the surface, and algebraic topological methods aim at extracting much richer topological information from data.


![TDA](https://pic4.zhimg.com/v2-bca1bc948527745f786d80427fd816f1_1200x500.jpg)
***

+ https://www.wikiwand.com/en/Topological_data_analysis
+ [TDA overview](https://perfectial.com/blog/topological-data-analysis-overview/)
+ [Studying the Shape of Data Using Topology](https://www.ias.edu/ideas/2013/lesnick-topological-data-analysis)
+ [Topological Data Analysis](https://dsweb.siam.org/The-Magazine/Article/topological-data-analysis-1)
+ [Why TDA works?](https://www.ayasdi.com/blog/bigdata/why-topological-data-analysis-works/)
+ [Topology-Based Active Learning](http://www.sci.utah.edu/publications/Mal2014a/UUSCI-2014-001.pdf)
+ https://sites.google.com/site/nips2012topology/
+ http://outlace.com/TDApart1.html
+ https://www.springer.com/cn/book/9783642150135
+ https://jsseely.github.io/notes/TDA/
+ [Extracting insights from the shape of complex data using topology](https://www.nature.com/articles/srep01236).
+ http://appliedtopology.org/
+ http://www.computational-geometry.org/
+ https://www.h-its.org/event/workshop-grg-2018/
+ https://sites.google.com/view/dragon-applied-topology
+ https://icerm.brown.edu/tripods/tri17-1-gtd/
+ https://www.ipam.ucla.edu/programs/long-programs/geometry-and-learning-from-data-in-3d-and-beyond/
+ http://kurlin.org/index.php#group
+ http://chomp.rutgers.edu/
+ http://chomp.rutgers.edu/Projects/Topological_Data_Analysis.html
+ https://www.jstage.jst.go.jp/article/tjsai/32/3/32_D-G72/_pdf
+ http://www.maths.usyd.edu.au/u/tillmann/cats2017/
+ http://www.maths.ox.ac.uk/groups/topology/
+ https://cs.nyu.edu/~yap/classes/
+ http://www.sci.utah.edu/~beiwang/
+ https://www.cs.montana.edu/tda/
+ https://www.csun.edu/~ctoth/Handbook/

### Topology

Topology focuses on the invariants under continuous mapping.
It pays more attention to the geometrical or discrete properties of the objects such as the number of circles or holes.
It is not distance-based.

> **Definition**: Let $X$ be a non-empty set. A set $\tau$ of subsets of $X$ is said to be a **topology** if
> * $X$ and the empty set $\emptyset$  belong to $\tau$;
> * the union of any number of sets in $\tau$ belongs to $\tau$;
> * the intersection of any two sets inn $\tau$ belongs to $\tau$.
> The pair $(X,\tau)$ is called a **topological space**.

As the definition shows the topology may be really not based on the definition of distance or measure. The set can be countable or discountable.e3

> **Definition**: Let $(X,\tau)$ be a topological space. Then the members of $\tau$ (the subsets of $X$) is said to be **open set**. If $X-S$ is open set, we call $S$ as **closed set**.


![klein bottle](https://www.ics.uci.edu/~eppstein/junkyard/nested-klein-bottles.jpg)

+ https://www.wikiwand.com/en/Topology
+ http://www.topologywithouttears.net/
+ https://www.ics.uci.edu/~eppstein/junkyard/topo.html
+ http://brickisland.net/DDGSpring2016/2016/01/20/reading-2-topology/
+ https://www.ayasdi.com/blog/artificial-intelligence/relationships-geometry-artificial-intelligence/

### TDA

Topological data analysis as one data processing method is selected topic for some students on computer science and applied mathematics.
It is not popular for the statisticians, where there is no estimation and test.

Topological data analysis (TDA) refers to statistical methods that find structure in data. As the
name suggests, these methods make use of topological ideas. Often, the term TDA is used narrowly
to describe a particular method called **persistent homology**.

TDA, which originates from mathematical topology, is a discipline that studies shape. It’s concerned with measuring the shape, by means applying math functions to data, and with representing it in forms of topological networks or combinatorial graphs.

There is another field that deals with the topological and geometric structure of data: computational geometry.
The main difference is that in TDA we treat the data as random points,
whereas in computational geometry the data are usually seen as fixed.

![tda](http://brickisland.net/DDGSpring2016/wp-content/uploads/2016/01/tda-300x208.png)

TDA can be applied to manifold estimation, nonlinear dimension reduction, mode estimation, ridge estimation and persistent homology.

***

+ https://datawarrior.wordpress.com/2015/08/03/tda-1-starting-the-journey-of-topological-data-analysis-tda/
+ https://www.annualreviews.org/doi/10.1146/annurev-statistics-031017-100045
+ http://brickisland.net/cs177fa12/
+ https://github.com/prokopevaleksey/TDAforCNN
+ https://github.com/ognis1205/spark-tda
+ https://github.com/stephenhky/PyTDA
+ https://people.clas.ufl.edu/peterbubenik/intro-to-tda/
+ https://www.math.colostate.edu//~adams/research/
+ http://brickisland.net/DDGSpring2016/2016/01/22/reading-3-topological-data-analysis/
+ https://www.math.upenn.edu/~ghrist/research.html
+ http://web.cse.ohio-state.edu/~dey.8/course/CTDA/CTDA.html
+ https://web.stanford.edu/group/bdl/blog/tda-cme-paper/
+ https://www.math.upenn.edu/~ghrist/research.html
+ http://www.stat.cmu.edu/topstat/presentations.html
+ http://www.stat.cmu.edu/research/statistical-theory-methodology/252
+ https://perfectial.com/blog/topological-data-analysis-overview/
+ http://www.sci.utah.edu/~beiwang/teaching/cs6170-spring-2017/
+ http://web.cse.ohio-state.edu/~dey.8/course/CTDA/CTDA.html
+ http://www.sci.utah.edu/~beiwang/tdaphenomics/tdaphenomics.html
+ https://drona.csa.iisc.ac.in/~vijayn/courses/
+ https://dsweb.siam.org/The-Magazine/Article/topological-data-analysis-1
+ https://www.annualreviews.org/doi/10.1146/annurev-statistics-031017-100045
+ https://www.turing.ac.uk/research/research-projects/scalable-topological-data-analysis

### Computational Topology

Computational topology is the mathematical theoretic foundation of topological data analysis. It is different from the deep neural network that origins from the engineering or the simulation to biological neural network.
Topological data analysis is principle-driven and application-inspired in some sense.

[CS 598: Computational Topology Spring 2013](http://jeffe.cs.illinois.edu/teaching/comptop/) put that
> Potential mathematical topics include the topology of ++cell complexes, topological graph theory, homotopy, covering spaces, simplicial homology, persistent homology, discrete Morse theory, discrete differential geometry, and normal surface theory. Potential computing topics include algorithms for computing topological invariants, graphics and geometry processing, mesh generation, curve and surface reconstruction, VLSI routing, motion planning, manifold learning, clustering, image processing, and combinatorial optimization++.

![bugs-topology](http://jeffe.cs.illinois.edu/teaching/comptop/Fig/codex-bugs.png)

+ https://datawarrior.wordpress.com/
+ http://graphics.stanford.edu/courses/cs468-09-fall/
+ https://graphics.stanford.edu/courses/cs468-02-fall/schedule.html
+ http://people.maths.ox.ac.uk/nanda/source/RSVWeb.pdf
+ https://jeremykun.com/tag/computational-topology/
+ http://jeffe.cs.illinois.edu/teaching/comptop/
+ http://www.enseignement.polytechnique.fr/informatique/INF556/
+ https://topology-tool-kit.github.io/
+ https://www.kth.se/student/kurser/kurs/SF2956?l=en
+ https://cs.nyu.edu/~yap/classes/modeling/06f/
+ https://courses.maths.ox.ac.uk/node/161
+ https://www2.cs.duke.edu/courses/fall06/cps296.1/
+ http://www.math.wsu.edu/faculty/bkrishna/CT_Math574_S12.html
