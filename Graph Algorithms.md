#### Graph as Data  Structure

http://mat.uab.cat/~alseda/MasterOpt/
http://ryanrossi.com/search.php
https://iss.oden.utexas.edu/

Graph is mathematical abstract or generalization of the connection between entities. It is an important part of discrete mathematics -- graph theory.
And graph processing is widely applied in industry and science such as the `graph convolutional network (GCN)`,  `probabilistic graph model(PGM)` and `knowledge graph`, which are introduced in other chapters.

A graph ${G=(V,E)}$ consists of a finite set of vertices $V(G)$ and a set of edges $E(G)$ consisting of distinct, unordered pairs of vertices, where nodes stand for entities and edges stand for their connections.
It is the foundation of **network science**.
It is  the fact that the feature is nothing except the connection that makes different from the common data.

> Graphs provide a powerful way to represent and exploit these connections.
> Graphs can be used to model such diverse areas as computer vision, natural language processing, and recommender systems. [^12]

The connections can be directed, weighted even probabilistic. It can be studied from the perspectives of matrix analysis and discrete mathematics.   

All data in computer machine is digitalized bits. The primitive goal is to represent graph in computer as one data structure.

> **Definition**:  Let $G=(V, E)$ be a graph with $V(G) = {1,\dots,n}$ and $E(G) = {e_1,\dots, e_m}$. Suppose each
> edge of $G$ is assigned an orientation, which is arbitrary but fixed. The (vertex-edge)
> `incidence` matrix of $G$, denoted by $Q(G)$, is the $n \times m$ matrix defined as follows.
> The rows and the columns of $Q(G)$ are indexed by $V(G)$ and $E(G)$, respectively.
> The $(i, j)$-entry of $Q(G)$ is 0 if vertex $i$ and edge $e_j$ are not incident, and otherwise it
> is $\color{red}{\text{1 or −1}}$ according as $e_j$ originates or terminates at $i$, respectively. We often denote
> $Q(G)$ simply by $Q$. Whenever we mention $Q(G)$ it is assumed that the edges of $G$ are oriented.

|The adjacency matrix|
|:---:|
|<img src = https://cdncontribute.geeksforgeeks.org/wp-content/uploads/adjacencymatrix.png width=60% />|

> **Definition**: Let $G$ be a graph with $V(G) = {1,\dots,n}$ and $E(G) = {e_1,\dots, e_m}$.The `adjacency` matrix of $G$, denoted by $A(G)$, is the $n\times n$ matrix defined as follows. The rows and
> the columns of $A(G)$ are indexed by $V(G)$. If $i \not= j$ then the $(i, j)$-entry of $A(G)$ is $0$ for vertices $i$ and $j$ nonadjacent, and the $(i, j)$-entry is $\color{red}{\text{1}}$ for $i$ and $j$ adjacent. The $(i,i)$-entry of $A(G)$ is 0 for $i = 1,\dots,n.$ We often denote $A(G)$ simply by $A$.

> `Adjacency Matrix` is also used to represent `weighted graphs`. If the $(i, j)$-entry of $A(G)$ is $w_{i, j}$, i.e. $A[i][j] = w_{i, j}$, then there is an edge from vertex $i$ to vertex $j$ with weight $w$.
> The `Adjacency Matrix` of `weighted graphs` $G$ is also called `weight` matrix of $G$, denoted by $W(G)$ or simply by $W$.

See *Graph representations using set and hash* at <https://www.geeksforgeeks.org/graph-representations-using-set-hash/>.

> **Definition**: In graph theory, the degree (or valency) of a vertex of a graph is the number of edges incident to the vertex, with loops counted twice. From the wikipedia page at <https://www.wikiwand.com/en/Degree_(graph_theory)>.
> The degree of a vertex $v$ is denoted $\deg(v)$ or $\deg v$. `Degree matrix` $D$ is a diagonal matrix such that $D_{i,i}=\sum_{j} w_{i,j}$ for the `weighted graph` with $W=(w_{i,j})$.

> **Definition**: Let $G$ be a graph with $V(G) = {1,\dots,n}$ and $E(G) = {e_1,\dots, e_m}$.The `Laplacian` matrix of $G$, denoted by $L(G)$, is the $n\times n$ matrix defined as follows. The rows and
> the columns of $L(G)$ are indexed by $V(G)$. If $i \not= j$ then the $(i, j)$-entry of $L(G)$ is
> $0$ for vertices $i$ and $j$ nonadjacent, and the $(i, j)$-entry is $\color{red}{\text{ −1}}$ for $i$ and $j$ adjacent. The
> $(i,i)$-entry of $L(G)$ is $\color{red}{d_i}$, the degree of the vertex $i$, for $i = 1,\dots,n.$
> In other words, the $(i,i)$-entry of $L(G)$, $L(G)_{i,j}$, is defined by
> $$L(G)_{i,j} = \begin{cases} \deg(V_i) & \text{if $i=j$,}\\ -1  & \text{if $i\not= j$ and $V_i$ and $V_j$ is adjacent,} \\ 0  & \text{otherwise.}\end{cases}$$
> Laplacian matrix of  a  graph $G$ with `weighted matrix` $W$ is ${L^{W}=D-W}$, where $D$ is the degree matrix of $G$.
> We often denote $L(G)$ simply by $L$.

> **Definition**:  A *directed graph* (or `digraph`) is a set of vertices and a collection of directed edges that each connects an ordered pair of vertices. We say that a directed edge points from the first vertex in the pair and points to the second vertex in the pair. We use the names 0 through V-1 for the vertices in a V-vertex graph. Via <https://algs4.cs.princeton.edu/42digraph/>.


`Adjacency List` is an array of lists.  An entry array[i] represents the list of vertices adjacent to the ith vertex. This representation can also be used to represent a weighted graph. The weights of edges can be represented as lists of pairs.
See more representation of graph in computer in <https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/>.
Although the adjacency-list representation is asymptotically at least as efficient as the adjacency-matrix representation, the simplicity of an adjacency matrix may make it preferable when graphs are reasonably small. Moreover, if the graph is unweighted, there is an additional advantage in storage for the adjacency-matrix representation.

***
|Cayley graph of F2 in Wikimedia | Moreno Sociogram 1st Grade|
|:------------------------------:|:---------------------------:|
|![Cayley graph of F2](http://mathworld.wolfram.com/images/eps-gif/CayleyGraph_1000.gif)|![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Moreno_Sociogram_1st_Grade.png/440px-Moreno_Sociogram_1st_Grade.png)|

* http://mathworld.wolfram.com/Graph.html
* https://www.wikiwand.com/en/Graph_theory
* https://www.wikiwand.com/en/Gallery_of_named_graphs
* https://www.wikiwand.com/en/Laplacian_matrix
* https://www.wikiwand.com/en/Cayley_graph
* https://www.wikiwand.com/en/Network_science
* https://www.wikiwand.com/en/Directed_graph
* https://www.wikiwand.com/en/Directed_acyclic_graph
****

It seems that graph theory is partially the application of matrix theory.
[Graph Algorithms in the Language of Linear Algebra](https://epubs.siam.org/doi/book/10.1137/1.9780898719918?mobileUi=0) shows how to leverage existing parallel matrix computation techniques and the large amount of software infrastructure that exists for these computations to implement efficient and scalable parallel graph algorithms. The benefits of this approach are reduced algorithmic complexity, ease of implementation, and improved performance.
__________________________________
Matrix Theory        | Graph Theory|-----|---
---------------------|-------------|-----|---
Matrix Addition      |?   | Spectral Theory |?
Matrix Powder        |?   | Jordan Form     |?
Matrix Multiplication|?   | Rank            |?
Basis                |  ? | Spectra         |?
__________________________________

> **Definition** A `walk` in a digraph is an alternating sequence of vertices and
edges that begins with a vertex, ends with a vertex, and such that for every edge
$\left<u\to v\right>$ in the walk, vertex $u$ is the element just before the edge,
and vertex $v$ is the next element after the edge.

A payoff of this representation is that we can use matrix powers to count numbers
of walks between vertices. The adjacent matrix ${A(G)}^k$ provides a count of the number of length $k$
walks between vertices in any digraph $G$.

**Definition** The length-k walk counting matrix for an n-vertex graph $G$ is
the $n \times n$ matrix $C^{k}$ such that:
$$
C_{uv}^{k} ::= \text{the number of length-k walks from $u$ to $v$}.
$$

> The length-k counting matrix of a digraph $G$ is ${A(G)}^k$, for all $k\in\mathbb{N}$.


> **Definition** A **walk** in an undirected graph is a sequence of vertices, where each
successive pair of vertices are adjacent; informally, we can also think of a walk as
a sequence of edges. A walk is called a **path** if it visits each vertex at most once. For any two vertices $u$ and $v$ in a graph $G$, we say that v is reachable from u
if $G$ contains a walk (and therefore a path) between $u$ and $v$. An undirected
graph is connected if every vertex is reachable from every other vertex.
A **cycle** is a path that starts and ends at the same vertex and has at least one
edge.

#### Shortest Paths

In graph theory, the `shortest path` problem is the problem of finding a path between two vertices (or nodes) in a graph such that the sum of the weights of its constituent edges is minimized.

Given the start node and end node, it is supposed to identify whether there is a path and find the shortest path(s) among all these possible paths.

The distance between the node $u$ and $v$ is the minimal number $k$ that makes $A^{k}_{uv}>0$.

#### $A^{\ast}$ Algorithm

 As introduced in wikipedia, $A^{\ast}$ algorithm has its advantages and disadvantages:
> In computer science, $A^{\ast}$ (pronounced "A star") is a computer algorithm that is widely used in path finding and graph traversal, which is the process of finding a path between multiple points, called "nodes". It enjoys widespread use due to its performance and accuracy. However, in practical travel-routing systems, it is generally outperformed by algorithms which can pre-process the graph to attain better performance, although other work has found A* to be superior to other approaches.


First we learn the **Dijkstra's algorithm**.
Dijkstra's algorithm is an algorithm for finding the shortest paths between nodes in a graph, which may represent, for example, road networks. It was conceived by computer scientist [Edsger W. Dijkstra](https://www.wikiwand.com/en/Edsger_W._Dijkstra) in 1956 and published three years later.

|Dijkstra algorithm|
|:----------------:|
|![Dijkstra algorithm](https://upload.wikimedia.org/wikipedia/commons/5/57/Dijkstra_Animation.gif)|

+ https://www.wikiwand.com/en/Dijkstra%27s_algorithm
+ https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/
+ https://www.wikiwand.com/en/Shortest_path_problem
+ https://www.cnblogs.com/chxer/p/4542068.html
+ [Introduction to A*: From Amit’s Thoughts on Pathfinding](http://theory.stanford.edu/~amitp/GameProgramming/AStarComparison.html)


See the page at Wikipedia [A* search algorithm](https://www.wikiwand.com/en/A*_search_algorithm)

#### Graph adjacency matrix duality

Perhaps even more important is the duality that exists with the fundamental
operation of linear algebra (vector matrix multiply) and a breadth-first search (BFS)
step performed on G from a starting vertex s:
$$
BFS(G, s) \iff A^T v, v(s)=1.
$$

This duality allows graph algorithms to be simply recast as a sequence of linear
algebraic operations. Many additional relations exist between fundamental linear
algebraic operations and fundamental graph operations

+ [Graph Algorithms in the Language of Linear Algebra](https://sites.cs.ucsb.edu/~gilbert/cs240a/slides/old/cs240a-GALA.pdf)
+ [Mathematics of Big Data: Spreadsheets, Databases, Matrices, and Graphs](http://www.mit.edu/~kepner/D4M/MathOfBigData.html)
+ [Dual Adjacency Matrix: Exploring Link Groups in Dense Networks by K. Dinkla  N. Henry Riche  M.A. Westenberg](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.12643)

- [On the p-Rank of the Adjacency Matrices of Strongly Regular Graphs](http://www.kurims.kyoto-u.ac.jp/EMIS/journals/JACO/Volume1_4/q1ur742gt117v044.fulltext.pdf)
- [Matrix techniques for strongly regular graphs and related geometries](http://cage.ugent.be/~fdc/intensivecourse2/haemers2.pdf)


#### Directed Acyclic Graph

`Directed acyclic graph` is the directed graph without any cycles. It is used widely in scheduling, distributed computation.


> **Definition** The acyclic graph is called `forest`. A connected acyclic graph is called a `tree`.

A graph $G$ is a tree if and only if $G$ is a forest and $|V(G)|=|E(G)| + 1$.

#### Graph Partitioning

[The fundamental problem that is trying to solve is that of splitting a large irregular graphs into k parts. This problem has applications in many different areas including, parallel/distributed computing (load balancing of computations), scientific computing (fill-reducing matrix re-orderings), EDA algorithms for VLSI CAD (placement), data mining (clustering), social network analysis (community discovery), pattern recognition, relationship network analysis, etc.
The partitioning is usually done so that it satisfies certain constraints and optimizes certain objectives. The most common constraint is that of producing equal-size partitions, whereas the most common objective is that of minimizing the number of cut edges (i.e., the edges that straddle partition boundaries). However, in many cases, different application areas tend to require their own type of constraints and objectives; thus, making the problem all that more interesting and challenging!

The research in the lab is focusing on a class of algorithms that have come to be known as multilevel graph partitioning algorithms. These algorithms solve the problem by following an approximate-and-solve paradigm, which is very effective for this as well as other (combinatorial) optimization problems.

Over the years we focused and produced good solutions for a number of graph-partitioning related problems. This includes partitioning algorithms for graphs corresponding to finite element meshes, multilevel nested dissection, parallel graph/mesh partitioning, dynamic/adaptive graph repartitioning, multi-constraint and multi-objective partitioning, and circuit and hypergraph partitioning.](http://glaros.dtc.umn.edu/gkhome/views/projects)

+ [Graph Partitioning](http://glaros.dtc.umn.edu/gkhome/views/projects)

![Moore Graphs](https://jeremykun.files.wordpress.com/2016/11/hoffman_singleton_graph_circle2.gif?w=900)

[A Spectral Analysis of Moore Graphs](https://jeremykun.com/2016/11/03/a-spectral-analysis-of-moore-graphs/)


#### Spectral Clustering Algorithm

Spectral method is the kernel tricks applied to [locality preserving projections](http://papers.nips.cc/paper/2359-locality-preserving-projections.pdf) as to reduce the dimension, which is as the data preprocessing for clustering.

In multivariate statistics and the clustering of data, spectral clustering techniques make use of the spectrum (eigenvalues) of the `similarity matrix` of the data to perform dimensionality reduction before clustering in fewer dimensions.
The similarity matrix is provided as an input and consists of a quantitative assessment of the relative similarity of each pair of points in the data set.

**Similarity matrix** is to measure the similarity between the input features $\{\mathbf{x}_i\}_{i=1}^{n}\subset\mathbb{R}^{p}$.
For example, we can use Gaussian kernel function
$$
f(\mathbf{x_i},\mathbf{x}_j)=exp(-\frac{{\|\mathbf{x_i}-\mathbf{x}_j\|}_2^2}{2\sigma^2})
$$
to measure the *similarity* of inputs.
The element of *similarity matrix* $S$ is $S_{i,j}=exp(-\frac{{\|\mathbf{x_i}-\mathbf{x}_j\|}_2^2}{2\sigma^2})$.
Thus $S$ is symmetrical, i.e. $S_{i,j}=S_{j,i}$ for $i,j\in\{1,2,\dots,n\}$.
If the sample size $n\gg p$, the storage of **similarity matrix** is much larger than the original input $\{\mathbf{x}_i\}_{i=1}^{n}$, when we would only preserve the entries above some values.
The **Laplacian matrix** is defined by $L=D-S$ where $D=Diag\{D_1,D_2,\dots,D_n\}$ and
$D_{i}=\sum_{j=1}^{n}S_{i,j}=\sum_{j=1}^{n}exp(-\frac{{\|\mathbf{x_i}-\mathbf{x}_j\|}_2^2}{2\sigma^2})$.

Then we can apply *principal component analysis* to the *Laplacian matrix* $L$ to reduce the data dimension. After that we can perform $K-means$ or other clustering.

* https://zhuanlan.zhihu.com/p/34848710
* *On Spectral Clustering: Analysis and an algorithm* at <http://papers.nips.cc/paper/2092-on-spectral-clustering-analysis-and-an-algorithm.pdf>
* *A Tutorial on Spectral Clustering* at <https://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf>.
* **谱聚类** <https://www.cnblogs.com/pinard/p/6221564.html>.
* *Spectral Clustering* <http://www.datasciencelab.cn/clustering/spectral>.
* https://en.wikipedia.org/wiki/Category:Graph_algorithms
* The course *Spectral Graph Theory, Fall 2015* at<http://www.cs.yale.edu/homes/spielman/561/>.
* <http://swoh.web.engr.illinois.edu/courses/ie532/project.html>.
* https://skymind.ai/wiki/graph-analysis

#### Graph Kernel and Spectral Graph Theory

Like kernels in **kernel methods**, graph kernel is used as functions measuring the similarity of pairs of graphs.
They allow kernelized learning algorithms such as support vector machines to work directly on graphs, without having to do feature extraction to transform them to fixed-length, real-valued feature vectors.

**Definition** : Find a mapping $f$ of the vertices of $G_1$ to the vertices of $G_2$ such that $G_1$ and $G_2$ are identical;
i.e. $(x, y)$ is an edge of $G_1$  if and only if $(f(x),f(y))$ is an edge of $G_2$.
Then ${f}$ is an isomorphism, and $G_1$ and $G_2$ are called `isomorphic`.

No polynomial-time algorithm is known for graph isomorphism.
Graph kernel are convolution kernels on pairs of graphs. A graph kernel makes the whole family kernel methods applicable to graphs.

`Von Neumann diffusion` is defined as
 $$K_{VND}=\sum_{k=0}^{\infty}{\alpha}^{k}{A}^{k}=(I-\alpha A)^{-1}, \alpha\in[0,1].$$

`Exponential diffusion` is defined as $K_{ED}=\sum_{k=0}^{\infty}\frac{1}{k!}{\alpha}^{k}{A}^{k}=\exp(\alpha A)$.
`Katz method` is defined as the truncation of `Von Neumann diffusion`
$$S_K=\sum_{k=0}^{K}{\alpha}^{k}{A}^{k}=(I-\alpha A)^{-1}(\alpha A-\alpha^k A^k).$$

+ https://www.wikiwand.com/en/Graph_product
+ https://www.wikiwand.com/en/Graph_kernel
+ [Graph Kernels](http://people.cs.uchicago.edu/~risi/papers/VishwanathanGraphKernelsJMLR.pdf)
+ [GRAPH KERNELS by Karsten M. Borgwardt](https://www.cs.ucsb.edu/~xyan/tutorial/GraphKernels.pdf)
+ [List of graph kernels](https://github.com/BorgwardtLab/graph-kernels)
+ [Deep Graph Kernel](http://www.mit.edu/~pinary/kdd/YanVis15.pdf)
+ [Topological Graph Kernel on Multiple Thresholded Functional Connectivity Networks for Mild Cognitive Impairment Classification](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4116356/)
+ [Awesome Graph Embedding](https://github.com/benedekrozemberczki/awesome-graph-embedding)
+ [Document Analysis with Transducers](https://leon.bottou.org/publications/pdf/transducer-1996.pdf)


### Computational Graph

Computational graphs are a nice way to think about mathematical expressions, where the mathematical expression will be in the decomposed form and in topological order.

![Computational Graph](https://colah.github.io/posts/2015-08-Backprop/img/tree-eval.png)

[To create a computational graph, we make each of these operations, along with the input variables, into nodes. When one node’s value is the input to another node, an arrow goes from one to another.These sorts of graphs come up all the time in computer science, especially in talking about functional programs. They are very closely related to the notions of dependency graphs and call graphs. They’re also the core abstraction behind the popular deep learning framework `TensorFlow`.](https://colah.github.io/posts/2015-08-Backprop/)

* https://colah.github.io/posts/2015-08-Backprop/
* [Visualization of Computational Graph@chainer.org](https://docs.chainer.org/en/stable/reference/graph.html)
* [Efficiently performs automatic differentiation on arbitrary functions. ](https://github.com/lobachevzky/computational-graph)
__________________________________

* http://ww3.algorithmdesign.net/sample/ch07-weights.pdf
* https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/
* https://www.geeksforgeeks.org/graph-types-and-applications/
* https://algs4.cs.princeton.edu/40graphs/
* http://networkscience.cn/
* http://www.ericweisstein.com/encyclopedias/books/GraphTheory.html
* http://mathworld.wolfram.com/CayleyGraph.html
* http://yaoyao.codes/algorithm/2018/06/11/laplacian-matrix
* The book **Graphs and Matrices** <https://www.springer.com/us/book/9781848829800>
* The book **Random Graph** <https://www.math.cmu.edu/~af1p/BOOK.pdf>
* The book [Graph Signal Processing: Overview, Challenges and Application](https://arxiv.org/pdf/1712.00468.pdf)
* http://www.andres.sc/graph.html
* https://github.com/sungyongs/graph-based-nn
* [Probabilistische Graphische Modelle](https://www-ai.cs.uni-dortmund.de/LEHRE/VORLESUNGEN/PGM/WS1415/index.html)
+ [NetworkX : a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks](https://networkx.github.io/documentation/stable/index.html)
+ [The Neo4j Graph Algorithms User Guide v3.5](https://github.com/neo4j-contrib/neo4j-graph-algorithms)
+ [Matlab tools for working with simple graphs](https://github.com/scheinerman/matgraph)
+ [GSoC 2018 - Parallel Implementations of Graph Analysis Algorithms](https://julialang.org/blog/2019/02/light-graphs)
+ [Graph theory (network) library for visualisation and analysis](http://js.cytoscape.org/)
+ [graph-tool | Efficient network analysis](https://graph-tool.skewed.de/)
+ [JGraphT: a Java library of graph theory data structures and algorithms](https://jgrapht.org/)
+ [Stanford Network Analysis Project](http://snap.stanford.edu/)
