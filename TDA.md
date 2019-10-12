# Topological Data Analysis

<img src="https://pic3.zhimg.com/v2-fc5ce3dd30b9f253913b833f4a3d6ccb_b.jpg" width="70%" />

Topological data analysis(TDA) is potential to find better representation of the data specially the shape of data.
TDA can visualize the high dimensional data and characterize the intrinsic invariants of the data.
It is close to computational geometry, manifold learning and computational topology.
It is one kind of descriptive representation learning.

> Problems of data analysis share many features with these two fundamental integration tasks:
> (1) how does one infer high dimensional structure from low dimensional representations;
> and (2) how does one assemble discrete points into global structure.

As [The NIPS 2012 workshop on Algebraic Topology and Machine Learning](https://sites.google.com/site/nips2012topology/) puts:
> Topological methods and machine learning have long enjoyed fruitful interactions as evidenced by popular algorithms
> like `ISOMAP, LLE and Laplacian Eigenmaps` which have been borne out of studying point cloud data through the lens of geometry.
> More recently several researchers have been attempting to also understand the algebraic topological properties of data.
> Algebraic topology is a branch of mathematics which uses tools from abstract algebra to study and classify topological spaces.
> The machine learning community thus far has focused almost exclusively on clustering as the main tool for unsupervised data analysis.
> Clustering however only scratches the surface, and algebraic topological methods aim at extracting much richer topological information from data.

![TDA example](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Illustration_of_Typical_Workflow_in_TDA.jpeg/976px-Illustration_of_Typical_Workflow_in_TDA.jpeg)
_______
<img src = "https://www.ics.uci.edu/~eppstein/junkyard/nested-klein-bottles.jpg" width ="30%">

Topological Data Analysis as its name shown takes the advantages of topological properties of data, which makes it different from manifold learning or computational geometry.
_____

+ <https://www.wikiwand.com/en/Topological_data_analysis>
+ [Centre for Topological Data Analysis](https://www.maths.ox.ac.uk/groups/topological-data-analysis)
+ [TDA overview](https://perfectial.com/blog/topological-data-analysis-overview/)
+ [Topological Data Analysis](https://dsweb.siam.org/The-Magazine/Article/topological-data-analysis-1)
+ [Topology-Based Active Learning](http://www.sci.utah.edu/publications/Mal2014a/UUSCI-2014-001.pdf)
+ [The NIPS 2012 workshop on Algebraic Topology and Machine Learning.](https://sites.google.com/site/nips2012topology/)
+ [Topological Data Analysis - Part 4 - Persistent Homology](http://outlace.com/TDApart1.html)
+ [Topological Methods in Data Analysis and Visualization @springer](https://www.springer.com/cn/book/9783642150135)
+ https://jsseely.github.io/notes/TDA/
+ [Applied topology](http://appliedtopology.org/)
+ [WORKSHOP ON TOPOLOGY AND NEUROSCIENCE](http://neurotop2018.org/)
+ https://www.h-its.org/event/workshop-grg-2018/
+ [Dragon Applied Topology Conference](https://sites.google.com/view/dragon-applied-topology)
+ [Computational & Algorithmic Topology, Sydney	](http://www.maths.usyd.edu.au/u/tillmann/cats2017/)
+ [Oxford Topology](http://www.maths.ox.ac.uk/groups/topology/)
+ [Computational Topology and Geometry: G22.3033.007 & G63.2400, Fall 2006](https://cs.nyu.edu/~yap/classes/modeling/06f/)
+ [Computational Topology and Geometry (CompTaG)](https://www.cs.montana.edu/tda/)
+ [Topological Methods for Machine Learning: An ICML 2014 Workshop in Beijing, China](http://topology.cs.wisc.edu/references.html)
+ [Towards topological machine learning](http://bastian.rieck.me/blog/posts/2019/towards_topological_machine_learning/)
+ [Geometry and Topology of Data @ICERM](https://icerm.brown.edu/tripods/tri17-1-gtd/)
+ [Geometry and Learning from Data in 3D and Beyond @IPAM](https://www.ipam.ucla.edu/programs/long-programs/geometry-and-learning-from-data-in-3d-and-beyond/)
+ [Topological Data Analysis: theory, examples, applications](http://kurlin.org/blog/)
+ http://chomp.rutgers.edu/
+ http://chomp.rutgers.edu/Projects/Topological_Data_Analysis.html
+ https://www.jstage.jst.go.jp/article/tjsai/32/3/32_D-G72/_pdf
+ https://scikit-tda.org/
+ https://people.maths.ox.ac.uk/tillmann/TDA2019.html
+ http://dauns.math.tulane.edu/~mathweb/clifford2012/
+ https://www.researchgate.net/profile/Genki_Kusano
+ https://www.researchgate.net/profile/Yasuaki_Hiraoka

## Why TDA?

[One of the key messages around topological data analysis is that data has `shape` and the shape matters.](https://www.ayasdi.com/blog/bigdata/why-topological-data-analysis-works/)
The shape is always not in the term of probability distribution function or cumulant distribution function.
[The basic goal of TDA is to apply topology, one of the major branches of mathematics, to develop tools for studying `geometric features of data`.](https://www.ias.edu/ideas/2013/lesnick-topological-data-analysis)


Perhaps the most elegant demonstration of the dangers of `summary statistics` is `Anscombe’s Quartet`. It’s a group of four datasets that appear to be similar when using typical summary statistics, yet tell four different stories when graphed. Each dataset consists of eleven $(x,y)$ pairs as follows:

<img src="https://heap-analytics.stamp51.com/wp-content/uploads/2014/04/anscombe_quartet.png" width="70%" />

As shown above, the shape matters. And the distribution can not tell us all the information the datasets encode.

Specially in high dimensional space, it is not easy to depict the shape of data sets in the term of probability distribution function and it is almost impossible to visualize or graph them without dimension reduction.
` Capturing all kinds of shape requires different method algebraically.`

Shape of Data:
* Normally defined in terms of a distance metric.
* Euclidean distance, Hamming, correlation distance, etc.
* Encodes similarity.

|[Property](https://web.stanford.edu/class/archive/ee/ee392n/ee392n.1134/lecture/apr9/ayasdi.pdf)|
|---|
|Coordinate Freeness |
|Deformation Invariance|
|Compressed Representation|


[Many of the methods currently being used operate as mechanisms for verifying (or disproving) hypotheses generated by an investigator, and therefore rely on that investigator to formulate good models or hypotheses. For many complex data sets, however, the number of possible hypotheses is very large, and the task of generating useful ones becomes very difficult. In this paper, we will discuss a method that allows exploration of the data, without first having to formulate a query or hypothesis. While most approaches to mining big data focus on pairwise relationships as the fundamental building block1, here we demonstrate the importance of understanding the “shape” of data in order to extract meaningful insights.](https://www.nature.com/articles/srep01236)

[The fundamental idea is that topological methods act as a geometric approach to pattern or shape recognition within data. Recognizing shapes (patterns) in data is critical to discovering insights in the data and identifying meaningful sub-groups. Typical shapes which appear in these networks are “loops” (continuous circular segments) and “flares” (long linear segments). We typically use these template patterns in an informal way, then identify interesting groups using these shapes. For example, we might select groups to be the data points in the nodes concentrated at the end of a flare. ](https://www.nature.com/articles/srep01236)

+ [Anscombe’s Quartet, and Why Summary Statistics Don’t Tell the Whole Story](https://heap.io/blog/data-stories/anscombes-quartet-and-why-summary-statistics-dont-tell-the-whole-story)
+ https://zhuanlan.zhihu.com/p/25547263
+ http://www.matrix67.com/blog/archives/2308
+ [Why TDA works?](https://www.ayasdi.com/blog/bigdata/why-topological-data-analysis-works/)
+ [Studying the Shape of Data Using Topology](https://www.ias.edu/ideas/2013/lesnick-topological-data-analysis)
+ [Towards a topological–geometrical theory of group equivariant non-expansive operators for data analysis and machine learning](https://www.nature.com/articles/s42256-019-0087-3)
+ https://zenodo.org/record/3264851#.XYLmjDYzaM9

## Topology Basics

Topology focuses on the invariants with respect to continuous mapping.
It pays more attention to the geometrical or discrete properties of the objects such as the number of circles or holes.
It is not distance-based as much as differential geometry.

> **Definition**: Let $X$ be a non-empty set. A set $\tau$ of subsets of $X$ is said to be a **topology** if
> * $X$ and the empty set $\emptyset$  belong to $\tau$;
> * the union of any number of sets in $\tau$ belongs to $\tau$;
> * the intersection of any two sets in $\tau$ belongs to $\tau$.

> The pair $(X,\tau)$ is called a **topological space**.

As the definition shows the topology may be really not based on the definition of distance or measure. The set can be countable or discountable.e3

> **Definition**: Let $(X,\tau)$ be a topological space. Then the members of $\tau$ (the subsets of $X$) is said to be **open set**. If $X-S$ is open set, we call $S$ as **closed set**.

From this definition, the open or close set is totally dependent on the set family $\tau$.

Like others in mathematics, the definition of topology is really abstract and strange like the outer space from the eyes of the ordinary living in the earth.
Mathematics texts are almostly written in logic order and for the ideal cases. A good piece of news is that topological data analysis does provide many vivid example and concrete application, which does not only consist of mathematical concepts or theorems.

![spaces](https://jsseely.github.io/notes/assets/spaces.jpg)

> **Definition** A topological space $(X, \tau)$ is said to be connected if $X$ is not the union of two disjoint nonempty open sets. Consequently, a topological space is disconnected if the union of any two disjoint nonempty subsets in $\tau$ produces $X$.

### Simplices and Simplicial Complexes

Topological data analysis employs the use of simplicial complexes, which are complexes of geometric structures called simplices (singular: simplex). TDA uses simplicial complexes because they can approximate more complicated shapes and are much more mathematically and computationally tractable than the original shapes that they approximate.

[Simplices are discrete building blocks for topological spaces.](https://jsseely.github.io/notes/TDA/)

[A simplicial complex is a generalization of a graph, with a few special features. The most particular feature is that simplicial complexes can contain higher order analogs of vertices and edges, referred to as simplices. Simplices can be the familiar vertices and edges of a graph, or triangles drawn between 3 vertices, tetrahedron between 4 vertices, and higher still.](https://sauln.github.io/blog/tda_explanations/)

For example, the probability simplex in $\mathbb{R}^n$ is defined as
$$\sum_{i=1}^{n}x_i=1,\quad x_i\geq 0\quad \forall i\in\{1, 2,  \dots, n\}.$$

In fact, each component in probability simplex is in the interval $[0, 1]$.

> An n-simplex $\sigma$ is the convex hull of $n + 1$ affinely ndependent vertices $S=$
> **Definition** A k-simplex in $X$ is an unordered collection of $k + 1$ distinct elements of $X$.

- [Simplicial Complexes and Simplicial Homology](http://ibykus.sdf.org/website/lang/de/algtop/notes4.pdf?lang=de)
- http://ibykus.sdf.org/website/lang/de/algtop/
- https://simplicial.readthedocs.io/en/latest/simplicialcomplex.html
- [simplicial: Simplicial topology in Python](https://simplicial.readthedocs.io/en/latest/index.html)

![simplex](http://outlace.com/images/TDAimages/simplices2.svg)

[Most often, simplicial complexes are built from the `nerve of a cover`. Intuitively named, a cover of a data set is a collection of subsets of the data such that every data point is in at least one of the subsets. Formally, we say a cover $\{U_i\}$ of a data set X is satisfies the condition that for any $x \in X$, there exists at least on $U_i \in \{U_i\}$ such that $x \in U_i$. In practice, we often have that each point is contained in multiple cover elements. `The nerve is a simplicial complex created from a cover by collapsing each cover element into vertices and connecting vertices when the cover elements had points in common.` If a point was included in two cover elements $U_i$ and $U_j$, then the vertices $\sigma_i, \sigma_j$ would have an edge drawn between them, denoted $\sigma_{ij}$. We continue this process to higher order intersections to create higher order simplices.](https://sauln.github.io/blog/tda_explanations/)

The faces of a simplex are its boundaries.
> **Definition** An `abstract simplex` is any finite set of vertices.

> **Definition**  A `complex` is a collection of multiple simplices.

> **Definition** A `simplicial complex` $\mathcal {K}$ is a set of simplices that satisfies the following conditions:  
>
> 1. Any face of a simplex in $\mathcal {K}$ is also in $\mathcal {K}$.
> 2. The intersection of any two simplices $\sigma_{1}, \sigma_{2}\in \mathcal {K}$ is either $\emptyset$ or a face of both $\sigma_{1}$ and $\sigma_{2}$.

> **Definition (Vietoris-Rips Complex)**
If we have a set of points $P$ of dimension $d$, and $P$ is a subset of $R^d$, then the Vietoris-Rips (VR) complex $V_{\epsilon}(P)$ at scale $\epsilon$ (the VR complex over the point cloud $P$ with parameter $\epsilon$) is defined as:
$$
V_{\epsilon}(P) = \{\sigma\subset P\mid d(u, v)\leq \epsilon,\forall u≠v\in\sigma\}
$$

These VR complexes have been used as a way of associating a simplicial complex to point cloud data sets.

>>>
1. **Flag/clique complexes** : Given a graph (network) $X$, the flag complex or clique complex of $X$ is the maximal simplicial complex $X$ that has the graph as its 1-skeleton: $X^{(1)}=X$.
2. **Banner Complexes**: [From flag complexes to banner complexes](https://sites.math.washington.edu/~novik/publications/banner.pdf)
3. **Nerve Complexes**: **Nerves**
Let $X$ be a topological space and $U = \{U_{\alpha}\}_{\alpha\in A}$ a covering
of $X$.
The nerve of $U$, denoted $N(U)$, is the abstract simplicial complex with vertex set $A$ and where $\{\alpha_0, \cdots , \alpha_k \}$ spans a k-simplex if and only if
$$U_{\alpha_0}\cap\cdots\cap U_{\alpha_k}\not=\emptyset.$$

4. **Dowker Complexes**: For simplicity, let $X$ and $Y$ be finite sets with #R \subset X\times Y$ representing
the ones in a binary matrix (also denoted R) whose columns are indexed by $X$
and whose rows are indexed by $Y$. The `Dowker complex` of $R$ on $X$ is the simplicial
complex on the vertex set $X$ defined by the rows of the matrix $R$. That is, each
row of $R$ determines a subset of $X$: use these to generate a simplex and all its
faces. Doing so for all the rows gives the Dowker complex on $X$. There is a dual
Dowker complex on $Y$ whose simplices on the vertex set $Y$ are determined by the
ones in columns of $R$.
5. **Cell Complexes**: [Cell Complexes: Definitions](http://jeffe.cs.illinois.edu/teaching/comptop/2009/notes/cell-complexes.pdf)

https://ncatlab.org/nlab/show/CW+complex   

<img src="https://jsseely.github.io/notes/assets/toruscomplex.jpg" />

**Nerve Theorem**
$X$ and $U$ as above, $U$ a covering by open sets which is enumerable. Suppose further that for all $\emptyset \not= S \subset A$ we have that $\cap_{s\in S} U_s$
is either empty or contractible. Then $N(U)$ is homotopy equivalent to $X$.

******

> **Definition** A family $\Delta$ of non-empty finite subsets of a set $S$ is an `abstract simplicial complex` if, for every set $X$ in $\Delta$, and every non-empty subset $Y \subset X$, $Y$ also belongs to $\Delta$.

> **Definition** The `n-chain`, denoted $C_n(S)$ is the subset of an oriented abstract simplicial complex $S$ of n-dimensional simplicies.

> **Definition** The `boundary` of an n-simplex $X$ with vertex set $[v_0, v_1, v_2,...v_n]$, denoted $\partial(X)$, is:
$$\partial(X)=\sum_{i=0}^{n}(−1)^i[v_0, v_1, v_2,...v_n],$$
> where the i-th vertex is removed from the sequence.

#### Betti numbers and Persistence Diagram

[Betti numbers and Persistence Diagram (PD) are topological descriptors of a simplicial complex; while Betti numbers count holes of different dimensions, PD tracks the birth and death instances of distinct topological features as the complex is sequentially built piece by piece. Separately, a Minimal Spanning Acycle (MSA) generalizes the notion of a minimal spanning tree to weighted simplicial complexes. ](https://math.duke.edu/events/78736-betti-numbers-persistence-diagrams-and-minimal-spanning-acycles-random-complexes)

- https://topospaces.subwiki.org/wiki/Betti_number
- http://www.math.jhu.edu/~jmb/note/

### Persistent Homology

Persistent homology (henceforth just PH) gives us a way to find interesting patterns in data without having to "downgrade" the data in anyway so we can see it.

[Perhaps the most important idea in applied algebraic topology is persistence. It is a response to the first difficulty that one encounters in attempting to assign topological invariants to statistical data sets: that the topology is not robust and has a sensitive dependence on the length scale at which the data set is being considered. The solution is to calculate the topology (specifically the homology) at all scales simultaneously, and to encode the relationship between the different scales in an algebraic invariant called the persistence diagram.](https://www.birs.ca/workshops/2012/12w5081/report12w5081.pdf)

<img src = "https://pic4.zhimg.com/v2-bca1bc948527745f786d80427fd816f1_1200x500.jpg" width = "50%" />

________________

[Persistent homology “generalizes clustering” in two ways: first, that it includes higher-order homological features in addition to the 0th order feature (i.e. the clusters); second, that it includes a persistence parameter that tells us what homological features exist at which scales. One only has to look to the ubiquity of clustering to see that persistent homology is a sensible thing to do.](https://jsseely.github.io/notes/TDA/)

Robert Ghrist said that
> Homology is the simplest, general, computable invariant of topological data. In its most primal manifestation, the homology of a space $X$ returns a sequence of vector spaces $H•(X)$, the dimensions of which count various types of linearly independent holes in $X$. Homology is inherently linear-algebraic, but transcends linear algebra, serving as the inspiration for homological algebra. It is this algebraic engine that powers the subject.

> **Definition** A `homotopy` between maps, $f_0 \simeq f_1 : X \to Y$ is a continuous 1-parameter family of maps $f_t: X \to Y$.
> A `homotopy equivalence` is a map $f : X \to Y$ with a homotopy inverse, $g: Y \to X$ satisfying $f \circ g \simeq {Id}_Y$ and $g \circ f \simeq {Id}_X$.

![Greedy optimal homotopy and homology generators Written with Kim Whittlesey](http://jeffe.cs.illinois.edu/pubs/pix/gohog.gif)
![HomotopySmall](https://upload.wikimedia.org/wikipedia/commons/7/7e/HomotopySmall.gif)

**Euler Characteristic**


***
* http://outlace.com/TDApart1.html
* http://outlace.com/TDApart2.html
* http://outlace.com/TDApart3.html
* http://outlace.com/TDApart4.html
* http://outlace.com/TDApart5.html
* [Homological Algebra and Data by Robert Ghrist](https://www.math.upenn.edu/~ghrist/preprints/HAD.pdf)
* [homotopy theory](https://ncatlab.org/nlab/show/homotopy+theory)
* [Henry Adams: Persistent Homology](https://github.com/henryadams/Leiden-PersistentHomology/wiki)

___________
+ https://www.wikiwand.com/en/Topology
+ [Topology Without Tears by Sidney A. Morris](http://www.topologywithouttears.net/)
+ [Geometric Topology](https://www.ics.uci.edu/~eppstein/junkyard/topo.html)
+ [Relationships, Geometry, and Artificial Intelligence](https://www.ayasdi.com/blog/artificial-intelligence/relationships-geometry-artificial-intelligence/)

### TDA

Topological data analysis as one data processing method is selected topic for some students on computer science and applied mathematics.
It is not popular for the statisticians, where there is no estimation and test.

Topological data analysis (TDA) refers to statistical methods that find structure in data. As the
name suggests, these methods make use of topological ideas. Often, the term TDA is used narrowly
to describe a particular method called **persistent homology**.

TDA, which originates from mathematical topology, is a discipline that studies shape. It’s concerned with measuring the shape, by means applying math functions to data, and with representing it in forms of topological networks or combinatorial graphs.
> Topological data analysis is more fundamental than revolutionary: such methods are not intended to supplant analytic, probabilistic, or spectral techniques. They can however reveal a deeper basis for why some data sets and systems behave the way they do. It is unwise to wield topological techniques in isolation, assuming that the weapons of unfamiliar "higher" mathematics are clad in incorruptible silver

There is another field that deals with the topological and geometric structure of data: computational geometry.
The main difference is that in TDA we treat the data as random points,
whereas in computational geometry the data are usually seen as fixed.

![tda](http://brickisland.net/DDGSpring2016/wp-content/uploads/2016/01/tda-300x208.png)

TDA can be applied to `manifold estimation, nonlinear dimension reduction, mode estimation, ridge estimation and persistent homology`.

+ [IDAC TDA Workshop: Topological Data Analysis for Discovery in Multi-scalar Biomedical Data – Applications in Musculoskeletal Imaging](https://radiology.ucsf.edu/events/idac-tda-workshop-topological-data-analysis-discovery-multi-scalar-biomedical-data-%E2%80%93)
+ [International Workshop on Topological Data Analysis in Biomedicine (TDA-Bio)  Seattle, WA, October 2, 2016](http://www.sci.utah.edu/~beiwang/acmbcbworkshop2016/)
+ [Deep Learning with Topological Signatures - Persistent Homology and Machine Learning](http://machinelearning.math.rs/Jekic-TDA.pdf)
+ [Topological data analysis for imaging and machine learning](http://math.ens-paris-saclay.fr/version-francaise/formations/master-mva/contenus-/topological-data-analysis-for-imaging-and-machine-learning--377025.kjsp?RH=1242430202531)
+ [Time Series Featurization via Topological Data Analysis](https://arxiv.org/abs/1812.02987)
+ [Topological Analysis and Visualization of Cyclical Behavior in Memory Reference Traces](http://www.cspaul.com/wordpress/publications_choudhury-2012-pv/)
+ http://tdaphenomics.eecs.wsu.edu/
+ https://wasp-sweden.org/topological-data-analysis/
+ http://unboxai.org/
+ [Topological Data Analysis and Persistent Homology](https://donaldpinckney.com/machine%20learning/2019/05/02/tda.html)
+ [An Introduction to Topological Data Analysis for Physicists: From LGM to FRBs](https://arxiv.org/pdf/1904.11044.pdf)

#### Persistence-Way

[Topological analysis using persistent homology](http://www.sci.utah.edu/~beiwang/acmbcbworkshop2016/slides/SvetlanaLockwood.pdf)
* Finds topological invariants in data (# of connected components, enclosed voids, etc.)
* Input: a (density) function, $f$
* Output: topological structures & their *persistence*
* Def: given threshold $t$, the superlevel set $f^{-1}(t)=\{x\mid f(x)\geq t\}$.
  * the true structures are hidden in superlevel sets
  * consider the whole stack of superlevel sets
  * identify structures that often appear (high persistence)
  * Output: persistence diagram – dots representing all structures

[It is beneficial to encode the persistent homology of a data set in the form
of a parameterized version of a Betti number: a barcode.](https://www.math.upenn.edu/~ghrist/preprints/barcodes.pdf)

- http://www.sci.utah.edu/~beiwang/acmbcbworkshop2016/slides/ChaoChen.pdf
- [BARCODES: THE PERSISTENT TOPOLOGY OF DATA](https://www.math.upenn.edu/~ghrist/preprints/barcodes.pdf)

#### TDA Mapper

[The key insight offered by this technique is that many interesting “clusters” in real data are not clusters in the classical sense (as disconnected components), but are the branches of some single connected component. Think about the three “clusters” in the shape $Y$. As simple as this sounds, this insight has been driving real progress in cancer genomics (where the “clusters” are rarely true clusters), and I suspect this method (or some reinvention of it) will find its ways into more fields in due time.](https://jsseely.github.io/notes/TDA/)

<img scr="https://pic4.zhimg.com/80/v2-285bd03f800512b2bbe450e940496a8f_hd.png" width="80%" />

* Apply a filter function to project data onto a lower dimensional space
* Performs partial clustering in the level sets

Mapper is an important tool used in TDA for data
visualization.
* Input
  * point cloud;
  * “filter function;”
  * covering of a metric space;
  * clustering algorithm;
  * various other parameters.
* Output
  * Graph (or higher simplicial complex) which is thought to capture aspects of the topology of the point cloud.

[Mapper gives a multi-resolution, low dimensional picture of the point cloud. It’s highly customizable, and has a track record of revealing structure that clustering and (linear or nonlinear) “projection pursuit” methods miss.](https://jdc.math.uwo.ca/TDA/Herring-Mapper.pdf)

+ [The Mapper Algorithm Western TDA Learning Seminar](https://jdc.math.uwo.ca/TDA/Herring-Mapper.pdf)
+ [Topology ToolKit: Efficient, generic and easy Topological data analysis and visualization](https://topology-tool-kit.github.io/)
+ [Data Visualization with TDA Mapper, Spring 2018](http://homepage.divms.uiowa.edu/~idarcy/COURSES/TDA/SPRING18/3900.html)
+ http://danifold.net/mapper/
+ https://github.com/scikit-tda/kepler-mapper
+ https://www.ayasdi.com/
+ [Extracting insights from the shape of complex data using topology](https://www.nature.com/articles/srep01236).
+ [Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition](http://www.ayasdi.com/wp-content/uploads/2015/02/Topological_Methods_for_the_Analysis_of_High_Dimensional_Data_Sets_and_3D_Object_Recognition.pdf)

---
+ https://zhuanlan.zhihu.com/p/31734839
+ [Interesting Paths in the Mapper](https://arxiv.org/abs/1712.10197)
+ [A header only software library helps to visually discover the insights of high dimensional complex data set.](https://xperthut.github.io/HYPPO-X/)
+ [Toward A Scalable Exploratory Framework for Complex High-Dimensional Phenomics Data](https://www.biorxiv.org/content/10.1101/159954v2)
+ [Machine Learning Explanations with Topological Data Analysis](https://sauln.github.io/blog/tda_explanations/)


#### Density Cluster with TDA


- [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://umap-learn.readthedocs.io/en/latest/index.html)

### Resource

[**Dr Vitaliy Kurlin**](http://kurlin.org/index.php)
[Applied Algebraic Topology (AAT) network in the UK](http://kurlin.org/applied-algebraic-topology.html)

***
[Wang Bei](http://www.sci.utah.edu/~beiwang/) was a PI of [DBI: ABI Innovation: A Scalable Framework for Visual Exploration and Hypotheses Extraction of Phenomics Data using Topological Analytics](http://www.sci.utah.edu/~beiwang/tdaphenomics/tdaphenomics.html).

![Wang Bei](http://www.sci.utah.edu/~beiwang/Bei3.jpg)

+ [A series of blogs on TDA](https://datawarrior.wordpress.com/2015/08/03/tda-1-starting-the-journey-of-topological-data-analysis-tda/)
+ [Topological Data Analysis @ Annual Review of Statistics and Its Application](https://www.annualreviews.org/doi/10.1146/annurev-statistics-031017-100045)
+ [Topological Data Analysis by peterbubenik](https://people.clas.ufl.edu/peterbubenik/intro-to-tda/)
+ [ Applied Algebraic Topology Research Network](https://topology.ima.umn.edu/)
+ [Henry Adams interests in computational topology and geometry, combinatorial topology, and applied topology](https://www.math.colostate.edu//~adams/research/)
+ [Robert Ghrist's research is in applied topology that is, applications of topology to engineering systems, data, dynamics, & more](https://www.math.upenn.edu/~ghrist/research.html)
+ [CSE 5559: Computational Topology and Data Analysis by Tamal K Dey ](http://web.cse.ohio-state.edu/~dey.8/course/CTDA/CTDA.html)
+ [CMU TopStat](http://www.stat.cmu.edu/topstat/presentations.html)
+ [Topological & Functional Data Analysis @ CMU](http://www.stat.cmu.edu/research/statistical-theory-methodology/252)
+ [Topological Data Analysis: an Overview of the World’s Most Promising Data Mining Methodology](https://perfectial.com/blog/topological-data-analysis-overview/)
+ [Index of /~beiwang/teaching/cs6170-spring-2017](http://www.sci.utah.edu/~beiwang/teaching/cs6170-spring-2017/)
+ [Topological Data Analysis: One Applied Mathematician’s Heartwarming Story of Struggle, Triumph, and (Ultimately) More Struggle By Chad Topaz](https://dsweb.siam.org/The-Magazine/Article/topological-data-analysis-1)
+ [Scalable topological data analysis](https://www.turing.ac.uk/research/research-projects/scalable-topological-data-analysis)
+ [Topology, Computation and Data Analysis](https://www.dagstuhl.de/de/programm/kalender/semhp/?semnr=19212)
+ [Topological Data Analysis Learning Seminar, Summer 2018](https://jdc.math.uwo.ca/TDA/)
+ https://www-apr.lip6.fr/~tierny/topologicalDataAnalysisClass.html
* [Topological Data Analysis and Persistent Homology](http://www.science.unitn.it/cirm/TDAPH2018.html)
* http://www.columbia.edu/~jss2219/tda/
* https://github.com/henryadams/Charleston-TDA-ML
* https://github.com/prokopevaleksey/TDAforCNN
* https://github.com/ognis1205/spark-tda
* https://github.com/stephenhky/PyTDA
* http://www.columbia.edu/~jss2219/tda/Resources.html
* [Open Source Software for TDA](http://www.sci.utah.edu/~beiwang/acmbcbworkshop2016/slides/SvetlanaLockwood.pdf)

### Application

+ http://tdaphenomics.eecs.wsu.edu/
+ [Topological Data Analysis of fMRI data：11 Apr 2018 by Manish Saggar](https://web.stanford.edu/group/bdl/blog/tda-cme-paper/)
+ [Topological Data Analysis for Genomics and Applications to Cancer](https://rabadan.c2b2.columbia.edu/courses)
+ [Topological Data Analysis and Machine Learning for Classifying Atmospheric River Patterns in Large Climate Datasets](https://meetingorganizer.copernicus.org/EGU2018/EGU2018-10825.pdf)
+ [DBI: ABI Innovation: A Scalable Framework for Visual Exploration and Hypotheses Extraction of Phenomics Data using Topological Analytics](http://www.sci.utah.edu/~beiwang/tdaphenomics/tdaphenomics.html)
+ [Algebraic topology and neuroscience: a bibliography](http://www.chadgiusti.com/bib.html)
+ [Mass Cytometry and Topological Data Analysis Reveal Immune Parameters Associated with Complications after Allogeneic Stem Cell Transplantation](https://www.cell.com/cell-reports/pdf/S2211-1247(17)31113-0.pdf)
+ [Two Applications of Topological Methods for Neuronal Morphology Analysis](http://www.sci.utah.edu/~beiwang/acmbcbworkshop2016/slides/YusuWang.pdf)
+ [Utilizing Topological Data Analysis to Detect Periodicity](http://www.sci.utah.edu/~beiwang/acmbcbworkshop2016/slides/ElizabethMunch.pdf)
+ [Topological Problems in Molecular Biology, American Mathematical Society Central Section](http://homepage.divms.uiowa.edu/~idarcy/2011workshop.html)
+ [A survey of Topological Data Analysis Methods for Big Data in Healthcare Intelligence](https://www.ripublication.com/ijaer19/ijaerv14n2_34.pdf)

## Computational Topology

Computational topology is the mathematical theoretic foundation of topological data analysis. It is different from the deep neural network that origins from the engineering or the simulation to biological neural network.
Topological data analysis is principle-driven and application-inspired in some sense.

[CS 598: Computational Topology Spring 2013](http://jeffe.cs.illinois.edu/teaching/comptop/) covers the following topics:
> Potential mathematical topics include the topology of **cell complexes, topological graph theory, homotopy, covering spaces, simplicial homology, persistent homology, discrete Morse theory, discrete differential geometry, and normal surface theory. Potential computing topics include algorithms for computing topological invariants, graphics and geometry processing, mesh generation, curve and surface reconstruction, VLSI routing, motion planning, manifold learning, clustering, image processing, and combinatorial optimization**.

<img src = "http://jeffe.cs.illinois.edu/teaching/comptop/Fig/codex-bugs.png" width = 40% />

+ [Computational Algebraic Topology](http://people.maths.ox.ac.uk/nanda/cat/index.html)
+ https://datawarrior.wordpress.com/
+ http://people.maths.ox.ac.uk/tillmann/CAT.html
+ [Theory and Algorithms in Data Science](https://turing-seminar.github.io/)
+ http://graphics.stanford.edu/courses/cs468-09-fall/
+ [CS 468 - Fall 2002: Introduction to  Computational  Topology](https://graphics.stanford.edu/courses/cs468-02-fall/schedule.html)
+ http://people.maths.ox.ac.uk/nanda/source/RSVWeb.pdf
+ [The Čech Complex and the Vietoris-Rips Complex](https://jeremykun.com/tag/computational-topology/)
+ [CS 598: Computational Topology , Spring 2013, Jeff Erickson](http://jeffe.cs.illinois.edu/teaching/comptop/)
+ [INF556 -- Topological Data Analysis (2018-19) Steve Oudot](http://www.enseignement.polytechnique.fr/informatique/INF556/)
+ [SF2956 Topological Data Analysis 7.5 credits](https://www.kth.se/student/kurser/kurs/SF2956?l=en)
+ [Computational Topology and Geometry G22.3033.007 & G63.2400, Fall 2006 @NYU](https://cs.nyu.edu/~yap/classes/modeling/06f/)
+ [C3.9 Computational Algebraic Topology (2016-2017)](https://courses.maths.ox.ac.uk/node/161)
+ [CPS296.1: COMPUTATIONAL TOPOLOGY @Duke](https://www2.cs.duke.edu/courses/fall06/cps296.1/)
+ [Math 574--Introduction to Computational Topology (Spring 2016)](http://www.math.wsu.edu/faculty/bkrishna/CT_Math574_S12.html)
+ [NSF-CBMS Conference and Software Day on Topological Methods in Machine Learning and Artificial Intelligence: May 13–17 and May 18, 2019. Department of Mathematics, College of Charleston, South Carolina](https://blogs.cofc.edu/cbms-tda2019/)
+ [Data science and applied topology](http://cunygc.appliedtopology.nyc/)
+ [Machine Learning Explanations with Topological Data Analysis](https://sauln.github.io/blog/tda_explanations/)
+ [Topological Data Analysis and Machine Learning Theory](https://www.birs.ca/workshops/2012/12w5081/report12w5081.pdf)
+ https://ima.umn.edu/2013-2014/

<img src = "http://www.math.wsu.edu/faculty/bkrishna/pics/MultipleTunnels.png" width= "20%" />
<img src="http://kurlin.org/images/topdatanalysis.png">

## Computational Geometry

https://shapeofdata.wordpress.com/

`Computational geometry` uses some information of samples or local information of the geometrical objects to reconstruct/describe  the whole object.
In computer vision, the task `3D reconstruction` is  a typical example of computational geometry.


+ [Probabilistic Approach to Geometry](https://www.mathsoc.jp/meeting/msjsi08/)
+ [Applied Geometry Lab @Caltech](http://www.geometry.caltech.edu/)
+ [Titane: Geometric Modeling of 3D Environments](https://team.inria.fr/titane/)
+ [Computational Geometry and Modeling G22.3033.007 Spring 2005](https://cs.nyu.edu/~yap/classes/modeling/05s/)
+ [Multi-Res Modeling Group@Caltech](http://www.multires.caltech.edu/research/research.htm)
+ [Geometry in Graphics Group in Computer Science and Engineering@Michigan State University](http://geometry.cse.msu.edu/)
+ [Computational Geometry Week (CG Week) 2019](http://eecs.oregonstate.edu/socg19/)
+ [Computational Geometry and Topology](https://drona.csa.iisc.ac.in/~gsat/Course/CGT/)
+ http://www.computational-geometry.org/
+ [Handbook of Discrete and Computational Geometry —Third Edition— edited by Jacob E. Goodman, Joseph O'Rourke, and Csaba D. Tóth](https://www.csun.edu/~ctoth/Handbook/)
+ http://brickisland.net/DDGSpring2016/2016/01/22/reading-3-topological-data-analysis/
+ http://graphics.stanford.edu/courses/cs468-14-winter/
+ https://drona.csa.iisc.ac.in/~gsat/Course/CGT/
+ http://www.computational-geometry.org/
+ https://project.inria.fr/gudhi/
+ http://web.cse.ohio-state.edu/~wang.1016/courses/788/
+ [Higher Dimensional Geometry Understanding](http://gudhi.gforge.inria.fr/)

![discrete differential geomety](http://brickisland.net/DDGSpring2019/wp-content/uploads/2019/01/cropped-cropped-header.png)

- [MATH:7450 (22M:305) Topics in Topology: Scientific and Engineering Applications of Algebraic Topology](http://homepage.divms.uiowa.edu/~idarcy/AppliedTopology.html#home)

## Geometric Data Analysis

http://cs233.stanford.ed
https://tgda.osu.edu/

`Geometric Data Analysis` and topological data analysis are out of the mainstream of quantitative statistics while the quantity also matters in geometric data analysis.
In conventional statistics, the core concepts are distribution (count in brief) and in/dependence, which is regarded as the reverse engineer of the probability theory. It is supposed that the  data is embedded in some "flat" subspace in $\mathbb{R}^n$ in the past. [Statistics on Manifold](http://bactra.org/notebooks/statistics-on-manifolds.html) and geometry information extends statistics into higher geometrical level.

+ [GEOMETRIC DATA ANALYSIS, U CHICAGO, MAY 20-23 2019](http://appliedtopology.org/geometric-data-analysis-u-chicago-may-20-23-2019/)
+ [Geometric Data Analysis Reading Group](https://www.stat.washington.edu/mmp/geometry/reading-group17/html/gda-home.html)
+ [Foundations of Geometric Methods in Data Analysis](http://www-sop.inria.fr/abs/teaching/centrale-FGMDA/centrale-FGMDA.html)
+ [CS233 Class Schedule for Spring Quarter '17-'18](http://cs233.stanford.edu/)
+ [MA500 Geometric Foundations of Data Analysis](http://www.maths.nuigalway.ie/~mstudies/MA500/)
+ [Special Session on Geometric Data Analysis](http://www.clrc.rhul.ac.uk/slds2015/SS_GDA.html)
+ [Workshop - Statistics for geometric data and applications to anthropology](https://www.frias.uni-freiburg.de/en/events/conferences/workshop-statistics-for-geometric-data-title)
+ [CSIC 5011: Topological and Geometric Data Reduction and Visualization](https://yao-lab.github.io/2019_csic5011/)
+ [4th conference on Geometric Science of Information](https://www.see.asso.fr/en/GSI2019)
+ [Geometric Image Processing @ Department of Computer Science Technion - Israel Institute of Technology](http://gip.cs.technion.ac.il/)

# Optimal Transport

![](https://cedricvillani.org/wp-content/themes/SF-Blueprint-WP/img/Cedric-Villani-Sebastien-Godefroy.jpg)
+ [The geometry of optimal transportation](https://projecteuclid.org/download/pdf_1/euclid.acta/1485890981)
+ [Transformations of PDEs: Optimal Transport and Conservation Laws by Woo-Hyun Cook](http://digitalassets.lib.berkeley.edu/etd/ucb/text/Cook_berkeley_0028E_15678.pdf)
+ [Optimal transport, old and new](https://cedricvillani.org/wp-content/uploads/2012/08/preprint-1.pdf)
+ [Math 3015 (Topics in Optimal Transport). Spring 2010](http://www.pitt.edu/~pakzad/optimaltransport.html)
+ https://optimaltransport.github.io/
+ http://www.math.ucla.edu/~wgangbo/Cedric-Villani.pdf
+ https://pot.readthedocs.io/en/stable/
+ [Optimal Transport @ESI](https://www.esi.ac.at/activities/events/2019/optimal-transport)
+ [Optimal Transport Methods in Density Functional Theory (19w5035)](https://www.birs.ca/events/2019/5-day-workshops/19w5035)
+ [Discrete OT](https://remi.flamary.com/demos/transport.html)
+ [Optimal Transport & Machine Learning](https://sites.google.com/site/nipsworkshopoptimaltransport/)
+ [Optimal Transport and Machine learning course at DS3 2018](https://github.com/rflamary/OTML_DS3_2018)
+ [Hot Topics: Optimal transport and applications to machine learning and statistics](https://www.msri.org/workshops/928)
+ [An intuitive guide to optimal transport for machine learning](https://www.mindcodec.com/an-intuitive-guide-to-optimal-transport-for-machine-learning/)
+ http://faculty.virginia.edu/rohde/transport/
+ http://otml17.marcocuturi.net/
+ https://anr.fr/Project-ANR-17-CE23-0012
+ http://otnm.lakecomoschool.org/program/
+ https://sites.uclouvain.be/socn/drupal/socn/node/113
+ [Topics on Optimal Transport in Machine Learning and Shape Analysis(OT.ML.SA)](https://people.math.osu.edu/memoli.2/courses/cse-topics-2018/)
+ [Optimal Transport in Biomedical Imaging](http://imagedatascience.com/transport/tutorials_isbi18.html)
+ [Optimal transport for documents classification: Classifying news with Word Mover Distance](http://www.lumenai.fr/blog/optimal-transport-for-documents-classification)
+ [Monge-Kantorovich Optimal Transport – Theory and Applications](https://cnls.lanl.gov/MK/)
