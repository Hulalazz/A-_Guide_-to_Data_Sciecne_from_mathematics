### Graph Algorithms

#### Graph as Data  Structure

Graph is mathematical abstract or generalization of the connection between entries.
A graph ${G}$ consists of a finite set of vertices $V(G)$ and a set of edges $E(G)$ consisting of distinct, unordered pairs of vertices, where nodes stand for entities and edges stand for their connections.
It is the foundation of **network science**.
It is different from the common data where the feature is nothing except the connection.
> Graphs provide a powerful way to represent and exploit these connections.
> Graphs can be used to model such diverse areas as computer vision, natural language processing, and recommender systems. [^12]

The connections can be directed, weighted even probabilistic.

> **Definition**:  Let $G$ be a graph with $V(G) = {1,\dots,n}$ and $E(G) = {e_1,\dots, e_m}$. Suppose each
> edge of $G$ is assigned an orientation, which is arbitrary but fixed. The (vertex-edge)
> `incidence` matrix of $G$, denoted by $Q(G)$, is the $n \times m$ matrix defined as follows.
> The rows and the columns of $Q(G)$ are indexed by $V(G)$ and $E(G)$, respectively.
> The $(i, j)$-entry of $Q(G)$ is 0 if vertex $i$ and edge $e_j$ are not incident, and otherwise it
> is $\color{red}{\text{1 or −1}}$ according as $e_j$ originates or terminates at $i$, respectively. We often denote
> $Q(G)$ simply by $Q$. Whenever we mention $Q(G)$ it is assumed that the edges of $G$ are oriented

> **Definition**: Let $G$ be a graph with $V(G) = {1,\dots,n}$ and $E(G) = {e_1,\dots, e_m}$.The `adjacency` matrix of $G$, denoted by $A(G)$, is the $n\times n$ matrix defined as follows. The rows and
> the columns of $A(G)$ are indexed by $V(G)$. If $i \not= j$ then the $(i, j)$-entry of $A(G)$ is
> $0$ for vertices $i$ and $j$ nonadjacent, and the $(i, j)$-entry is $\color{red}{\text{1}}$ for $i$ and $j$ adjacent. The
> $(i,i)$-entry of $A(G)$ is 0 for $i = 1,\dots,n.$ We often denote $A(G)$ simply by $A$.
> `Adjacency Matrix` is also used to represent `weighted graphs`. If the $(i,i)$-entry of $A(G)$ is $w_{i,j}$, i.e. $A[i][j] = w_{i,j}$, then there is an edge from vertex $i$ to vertex $j$ with weight $w$.
> The `Adjacency Matrix` of `weighted graphs` $G$ is also called `weight` matrix of $G$, denoted by $W(G)$ or simply by $W$.

See *Graph representations using set and hash* at <https://www.geeksforgeeks.org/graph-representations-using-set-hash/>.

> **Definition**: In graph theory, the degree (or valency) of a vertex of a graph is the number of edges incident to the vertex, with loops counted twice. From the Wikipedia page at <https://www.wikiwand.com/en/Degree_(graph_theory)>.
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

It seems that graph theory is the application of matrix theory at least partially.

***
|Cayley graph of F2 in Wikimedia | Moreno Sociogram 1st Grade|
|:------------------------------:|:---------------------------:|
|![Cayley graph of F2](http://mathworld.wolfram.com/images/eps-gif/CayleyGraph_1000.gif)|![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Moreno_Sociogram_1st_Grade.png/440px-Moreno_Sociogram_1st_Grade.png)|

* http://mathworld.wolfram.com/Graph.html
* https://www.wikiwand.com/en/Graph_theory
* https://www.wikiwand.com/en/Gallery_of_named_graphs
* https://www.wikiwand.com/en/Laplacian_matrix
* http://www.ericweisstein.com/encyclopedias/books/GraphTheory.html
* https://www.wikiwand.com/en/Cayley_graph
* http://mathworld.wolfram.com/CayleyGraph.html
* https://www.wikiwand.com/en/Network_science
* https://www.wikiwand.com/en/Directed_graph
* https://www.wikiwand.com/en/Directed_acyclic_graph
* http://ww3.algorithmdesign.net/sample/ch07-weights.pdf
* https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/
* https://www.geeksforgeeks.org/graph-types-and-applications/
* https://github.com/neo4j-contrib/neo4j-graph-algorithms
* https://algs4.cs.princeton.edu/40graphs/
* http://networkscience.cn/
* http://yaoyao.codes/algorithm/2018/06/11/laplacian-matrix
* The book **Graphs and Matrices** <https://www.springer.com/us/book/9781848829800>
* The book **Random Graph** <https://www.math.cmu.edu/~af1p/BOOK.pdf>.

#### A* Algorithm

In computer science, A* (pronounced "A star") is a computer algorithm that is widely used in pathfinding and graph traversal, which is the process of finding a path between multiple points, called "nodes". It enjoys widespread use due to its performance and accuracy. However, in practical travel-routing systems, it is generally outperformed by algorithms which can pre-process the graph to attain better performance, although other work has found A* to be superior to other approaches.
It is draw from [Wikipedia page on A* algorithm](https://www.wikiwand.com/en/A*_search_algorithm).

First we learn the **Dijkstra's algorithm**.
Dijkstra's algorithm is an algorithm for finding the shortest paths between nodes in a graph, which may represent, for example, road networks. It was conceived by computer scientist [Edsger W. Dijkstra](https://www.wikiwand.com/en/Edsger_W._Dijkstra) in 1956 and published three years later.

|Dijkstra algorithm|
|:----------------:|
|![Dijkstra algorithm](https://upload.wikimedia.org/wikipedia/commons/5/57/Dijkstra_Animation.gif)|

+ https://www.wikiwand.com/en/Dijkstra%27s_algorithm
+ https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/
+ https://www.wikiwand.com/en/Shortest_path_problem

See the page at Wikipedia [A* search algorithm](https://www.wikiwand.com/en/A*_search_algorithm)

#### Graph Kernel

Like kernels in **kernel methods**, graph kernel is used as functions measuring the similarity of pairs of graphs.
They allow kernelized learning algorithms such as support vector machines to work directly on graphs, without having to do feature extraction to transform them to fixed-length, real-valued feature vectors.


+ https://www.wikiwand.com/en/Graph_product
+ https://www.wikiwand.com/en/Graph_kernel
+ http://people.cs.uchicago.edu/~risi/papers/VishwanathanGraphKernelsJMLR.pdf
+ https://www.cs.ucsb.edu/~xyan/tutorial/GraphKernels.pdf
+ https://github.com/BorgwardtLab/graph-kernels

#### Spectral Clustering Algorithm

Spectral method is the kernel tricks applied to [locality preserving projections](http://papers.nips.cc/paper/2359-locality-preserving-projections.pdf) as to reduce the dimension, which is as the data preprocessing for clustering.

In multivariate statistics and the clustering of data, spectral clustering techniques make use of the spectrum (eigenvalues) of the `similarity matrix` of the data to perform dimensionality reduction before clustering in fewer dimensions. The similarity matrix is provided as an input and consists of a quantitative assessment of the relative similarity of each pair of points in the data set.

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
