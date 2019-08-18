## Geometric Deep Learning

<img src="https://pic3.zhimg.com/80/fd40dd2ef26a591b5cd0e9d798cd5a67_hd.jpg" width="50%" />

* https://qdata.github.io/deep2Read/
* [Geometric Deep Learning](https://qdata.github.io/deep2Read//2graphs/2architecture/2019/02/22/gnn-Geom/)
* https://qdata.github.io/deep2Read//aReadingsIndexByCategory/#2Graphs
* [The Power of Graphs in Machine Learning and Sequential Decision-Making ](https://graphpower.inria.fr/)
* http://www.ai3sd.org/
* https://heidelberg.ai/2019/07/09/graph-neural-networks.html
* https://sites.google.com/site/rdftestxyz/home
* [Lecture 11: Learning on Non-Euclidean Domains](https://vistalab-technion.github.io/cs236605/lecture_notes/lecture_11/)
* [What Can Neural Networks Reason About?](https://arxiv.org/abs/1905.13211)
* [Deep Geometric Matrix Completion by Federico Monti](http://helper.ipam.ucla.edu/publications/dlt2018/dlt2018_14552.pdf)


[In the last decade, Deep Learning approaches (e.g. Convolutional Neural Networks and Recurrent Neural Networks) allowed to achieve unprecedented performance on a broad range of problems coming from a variety of different fields (e.g. Computer Vision and Speech Recognition). Despite the results obtained, research on DL techniques has mainly focused so far on data defined on Euclidean domains (i.e. grids). Nonetheless, in a multitude of different fields, such as: Biology, Physics, Network Science, Recommender Systems and Computer Graphics; one may have to deal with data defined on non-Euclidean domains (i.e. graphs and manifolds). The adoption of Deep Learning in these particular fields has been lagging behind until very recently, primarily since the non-Euclidean nature of data makes the definition of basic operations (such as convolution) rather elusive. Geometric Deep Learning deals in this sense with the extension of Deep Learning techniques to graph/manifold structured data.](http://geometricdeeplearning.com/)

* [Computational Learning and Memory Group](http://cbl.eng.cam.ac.uk/Public/Lengyel/News)
* [Beyond deep learning](http://beyond-deep-nets.clps.brown.edu/)
* [Cognitive Computation Group @ U. Penn.](https://cogcomp.org/)
* [Computational cognitive modeling](https://brendenlake.github.io/CCM-site/)
* [Mechanisms of geometric cognition](http://hohol.pl/granty/geometry/)
* [Computational Cognitive Science Lab](http://cocosci.princeton.edu/research.php)
* https://jian-tang.com/teaching/graph2019
* [Introducing Grakn & Knowledge Graph Convolutional Networks: Dec 4, 2018 · Paris, France](https://eventil.com/events/introducing-grakn-ai-to-paris)
* [International Workshop on Deep Learning for Graphs and Structured Data Embedding](https://www.aminer.cn/dl4g-sde)
* [HYPERBOLIC DEEP LEARNING: A nascent and promising field](http://hyperbolicdeeplearning.com/papers/)

Images are stored in computer as matrix roughly. The spatial distribution of pixel on the screen project to us a colorful digitalized world.
`Convolutional neural network(ConvNet or CNN)` has been proved to be efficient to process and analyses the images for visual cognitive tasks.
What if we generalize these methods to graph structure which can be represented as adjacent matrix?

|Image | Graph|
|:-----:|:-----:|
|Convolutional Neural Network | Graph Convolution Network|
|Attention|[Graph Attention](http://petar-v.com/GAT/)|
|Gated|[Gated Graph Network](https://zhuanlan.zhihu.com/p/28170197)|
|Generative|[Generative Models for Graphs](http://david-white.net/generative.html)

Advanced proceedings of natural language processing(NLP) shone a light into semantic embedding as one potential approach to `knowledge representation`.
The text or symbol, strings in computer, is designed for natural people to communicate and understand based on the context or situation, i.e., the connections of entities and concepts are essential.
What if we generalize these methods to connected data?



### Hyperbolic Deep Learning

[The hyperbolic space is different from the Euclidean space. It has more capacity.
The volume of a ball grows exponentially with its radius!
Think of a binary tree: the number of nodes grows exponentially with depth.
Hyperbolic geometry is better suited to embed data with an underlying hierarchical/heterogeneous structure.
Let’s make deep learning possible in hyperbolic space.](http://hyperbolicdeeplearning.com/)

<img src="https://hazyresearch.github.io/hyperE/pc.svg"  width="40%" />

- [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112)
- http://hyperbolicdeeplearning.com/papers/
- http://hyperbolicdeeplearning.com/
- https://cla2019.github.io/
***
- http://shichuan.org/
- [Learning Mixed-Curvature Representations in Product Spaces](https://openreview.net/forum?id=HJxeWnCcF7)
- [Hyperbolic Attention Networks ](https://openreview.net/forum?id=rJxHsjRqFQ)
- [Hyperbolic Recommender Systems](https://arxiv.org/abs/1809.01703)
- [Hyperbolic Heterogeneous Information Network Embedding](http://shichuan.org/doc/65.pdf)

### Graph Convolutional Network

If images is regarded as a matrix in computer and text as a chain( or sequence), their representation contain all the spatial and semantic information of the entity.

Graph can be represented as `adjacency matrix` as shown in *Graph Algorithm*. However, the adjacency matrix only describe the connections between the nodes.
The feature of the nodes does not appear. The node itself really matters.
For example, the chemical bonds can be represented as `adjacency matrix` while the atoms in molecule really determine the properties of the molecule.


A simple and direct way is to concatenate the `feature matrix` $X\in \mathbb{R}^{N\times E}$ and `adjacency matrix` $A\in\mathbb{B}^{N\times N}$, i.e., $X_{in}=[X, A]\in \mathbb{R}^{N\times (N+E)}$. What is more, $\mathbb{B}$ is binary where each element of the adjacent  matrix $a_{i,j}$ is ${1}$ if the node ${i}$ is adjacent to the node ${j}$ otherwise 0.

And what is the output? How can deep learning apply to them? And how can we extend the tree-based algorithms such as decision tree into graph-based algorithms?

> For these models, the goal is then to learn a function of signals/features on a graph $G=(V,E)$ which takes as input:

> * A feature description $x_i$ for every node $i$; summarized in a $N\times D$ feature matrix ${X}$ ($N$: number of nodes, $D$: number of input features);
> * A representative description of the graph structure in matrix form; typically in the form of an adjacency matrix ${A}$ (or some function thereof)

> and produces a node-level output $Z$ (an $N\times F$ feature matrix, where $F$ is the number of output features per node). Graph-level outputs can be modeled by introducing some form of pooling operation (see, e.g. [Duvenaud et al., NIPS 2015](http://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints)).

Every neural network layer can then be written as a non-linear function
$$
{H}_{i+1} = \sigma \circ ({H}_{i}, A)
$$
with ${H}_0 = {X}_{in}$ and ${H}_{d} = Z$ (or $Z$ for graph-level outputs), $d$ being the number of layers. The specific models then differ only in how $\sigma$ is chosen and parameterized.

For example, we can consider a simple form of a layer-wise propagation rule
$$
{H}_{i+1} = \sigma \circ ({H}_{i}, A)=\sigma \circ(A {H}_{i} {W}_{i})
$$
where ${W}_{i}$ is a weight matrix for the $i$-th neural network layer and $\sigma (\cdot)$ is is a non-linear activation function such as *ReLU*.

* But first, let us address two limitations of this simple model: multiplication with $A$ means that, for every node, we sum up all the feature vectors of all neighboring nodes but not the node itself (unless there are self-loops in the graph). We can "fix" this by enforcing self-loops in the graph: we simply add the identity matrix $I$ to $A$.

* The second major limitation is that $A$ is typically not normalized and therefore the multiplication with $A$ will completely change the scale of the feature vectors (we can understand that by looking at the eigenvalues of $A$).Normalizing ${A}$ such that all rows sum to one, i.e. $D^{-1}A$, where $D$ is the diagonal node degree matrix, gets rid of this problem.

In fact, the propagation rule introduced in [Kipf & Welling (ICLR 2017)](https://arxiv.org/abs/1609.02907) is given by:
$$
{H}_{i+1} = \sigma \circ ({H}_{i}, A)=\sigma \circ(\hat{D}^{-\frac{1}{2}} \hat{A} \hat{D}^{-\frac{1}{2}} {H}_{i} {W}_{i}),
$$
with $\hat{A}=A+I$, where $I$ is the identity matrix and $\hat{D}$ is the diagonal node degree matrix of $\hat{A}$.
See more details at [Multi-layer Graph Convolutional Networks (GCN) with first-order filters](http://tkipf.github.io/graph-convolutional-networks/).

Like other neural network, GCN is also composite of linear and nonlinear mapping. In details,

1. $\hat{D}^{-\frac{1}{2}} \hat{A} \hat{D}^{-\frac{1}{2}}$ is to normalize the graph structure;
2. the next step is to multiply node properties and weights;
3. Add nonlinearities by activation function $\sigma$.

[See more at experoinc.com](https://www.experoinc.com/post/node-classification-by-graph-convolutional-network) or [https://tkipf.github.io/](https://tkipf.github.io/graph-convolutional-networks/).


<img src = "http://tkipf.github.io/graph-convolutional-networks/images/gcn_web.png" width=50% />
<img src = "https://research.preferred.jp/wp-content/uploads/2017/12/cnn-gcnn.png" width=50% />


$\color{navy}{\text{Graph convolution network is potential to}}\, \cal{reasoning}$ as the blend of $\frak{\text{probabilistic graph model}}$ and $\mit{\text{deep learning}}$.

GCN can be regarded as the counterpart of CNN for graphs so that the optimization techniques such as normalization, attention mechanism and even the adversarial version can be extended to the graph structure.

* [Node Classification by Graph Convolutional Network](https://www.experoinc.com/post/node-classification-by-graph-convolutional-network)
* [GRAPH CONVOLUTIONAL NETWORKS](https://tkipf.github.io/graph-convolutional-networks/)
* https://benevolent.ai/publications

### Spectral ConvNets

Compositional layers of convolutional neural network  can be expressed as

$$
\hat{H}_{i} = P\oplus H_{i-1}         \\
\tilde{H_i} = C_i\otimes(\hat{H}_{t})   \\
Z_i = \mathrm{N}\cdot  \tilde{H_i} \\
H_i = Pooling\cdot (\sigma\circ Z_i)
$$

where $\otimes,\oplus,\cdot$ represent convolution operation, padding and pooling, respectively.

[Xavier Bresson](https://www.ntu.edu.sg/home/xbresson/) gave  a talk on  [New Deep Learning Techniques
FEBRUARY 5-9, 2018](http://helper.ipam.ucla.edu/publications/dlt2018/dlt2018_14506.pdf).
We would ideally like our graph convolutional layer to have:

* Computational and storage efficiency (requiring no more than $O(E+V)$ time and memory);
* Fixed number of parameters (independent of input graph size);
* Localisation (acting on a local neighbourhood of a node);
* Ability to specify arbitrary importances to different neighbours;
* Applicability to inductive problems (arbitrary, unseen graph structures).

|CNN|GCN|---|
|---|---|---|
|padding|?|?|
|convolution|?|Information of neighbors|
|pooling|?|Invariance|

* `Spectral graph theory` allows to redefine convolution in the context of graphs with Fourier analysis.
* Graph downsampling $\iff$ graph coarsening $\iff$ graph partitioning: Decompose ${G}$ into smaller meaningful clusters.
* Structured pooling: Arrangement of the node indexing such that adjacent nodes are hierarchically merged at the next coarser level.


Laplacian operator is represented as a positive semi-definite $n \times n$ matrix:

|Laplacian|Representation|
|---|---|
|Unnormalized Laplacian|$\Delta = \bf{D - A}$|
|Normalized Laplacian|$\Delta = \bf{I -D^{-\frac{1}{2}}AD^{-\frac{1}{2}}}$|
|Random walk Laplacian|$\Delta = \bf{I - D^{-1}A}$|
|$\mathbf A$：Adjacency Matrix|$\mathbf D$: Degree Matrix|

Eigendecomposition of graph Laplacian:

$$\Delta = {\Phi}^{T} {\Lambda} {\Phi}$$

where ${\Phi}$ is the matrix consisting of eigenvectors and ${\Lambda}= diag({\lambda}_1，\dots, {\lambda}_n )$.
In matrix-vector notation, with the $n\times n$ Fourier matrix and a n-dimensional vector $f\in\mathbb{R}^{n}$, it is proven that if $\hat{f}={\Phi}^{T}f$ as the projection of $f$ into the column space  of  $\Phi$, where $\hat{f}_{i}$ the inner product of ${f, \phi_i}$, then $f={\Phi}\hat{f}$ the inverse Fourier transform.

Convolution of two vectors $f=(f_1, \cdots, f_n )^{T}$ and $g=(g_1, \cdots, g_n )^{T}$ is defined as $(f\star g)_i = \sum_{m} g_{(i-m) \,\,\, mod \,\,\,n } \,\cdot f_m$ or in matrix notation
$$ (f\star g)=
\underbrace{\begin{pmatrix}
& g_1, & g_2, & \cdots, & g_{n-1}, & g_n &\\
& g_n, & g_1, & \cdots, & g_{n-2}, & g_{n-1} & \\
& \vdots & \vdots &\ddots & \vdots & \vdots & \\
& g_3, & g_4, & \cdots, & g_{1}, & g_2 &\\
& g_2, & g_3, & \cdots, & g_{n}, & g_1 &\\
\end{pmatrix}}_{\text{Circulant matrix G} }
\begin{pmatrix}
f_1 \\
\vdots \\
f_n
\end{pmatrix} \\
= \underbrace{ {\Phi} {diag(\hat{g}_1, \cdots, \hat{g}_m)} \Phi^T }_{G} \quad f \\
= {\Phi}({\Phi}^Tg\circ {\Phi}^{T} f)
$$

where the last equation is because the matrix multiplication is associative and $\Phi^Tg$ is the $\fbox{Fourier transform}$; the notation $\circ$ is the inner product.
What is more,
$${\Phi}({\Phi}^Tg\circ {\Phi}^{T} f)={\Phi}\hat{g}(\Lambda){\Phi}^{T} f=\hat{g}({\Phi}(\Lambda){\Phi}^{T})f=\hat{g}(\Delta)f$$
where $\Delta$ is the Laplacian of the graph; $\hat{g}(\Lambda)$ is the polynomial of matrix.

****

Graph Convolution: Recursive Computation with Shared Parameters:

* Represent each node based on its neighborhood
* Recursively compute the state of each node by propagating previous
states using relation specific transformations
* Backpropagation through Structure

#### Vanilla spectral graph ConvNets


Every graph convolutional layer starts off with a shared node-wise feature transformation (in order to achieve a higher-level representation), specified by a weight matrix $W$. This transforms the feature vectors into $\vec{g}_i = {\bf W}\vec{h}_i$. After this, the vectors $\vec{g}_i$ are typically recombined in some way at each node.

In general, to satisfy the localization property, we will define a graph convolutional operator as an aggregation of features across neighborhoods;
defining $\mathcal{N}_i$ as the neighborhood of node i (typically consisting of all first-order neighbors of $i$ , including $i$ itself),
we can define the output features of node $i$ as:
$$\vec{h}_i' = \sigma\left(\sum_{j\in\mathcal{N}_i}\alpha_{ij}\vec{g}_j\right)$$
where $\sigma$ is some activation function such as **rectified linear unit (ReLU)** in ConvNet.

#### SplineNets

Parametrize the smooth spectral filter function

- [SplineNets: Continuous Neural Decision Graphs](https://arxiv.org/abs/1810.13118)

#### Spectral graph ConvNets with polynomial filters

Represent smooth spectral functions with polynomials of Laplacian eigenvalues
$$w_{\alpha}(\lambda)={\sum}_{j=0}^r{\alpha}_{j} {\lambda}^j$$

where $\alpha=(\alpha_1, \cdots, \alpha_r)^{T}$ is the vector of filter parameters

Convolutional layer: Apply spectral filter to feature signal ${f}$:
$$w_{\alpha}(\Lambda)f= {\sum}_{j=0}^r{\alpha}_{j} {\Lambda}^j f$$

Such graph convolutional layers are GPU friendly.

#### ChebNet

Graph convolution network  always deal with unstructured data sets where the graph has different size. What is more, the graph is dynamic, and  we need to apply to new nodes without model retraining.

Graph convolution with (non-orthogonal) monomial basis $1, x, x^2, x^3, \cdots$:
$$w_{\alpha}(\lambda)={\sum}_{j=0}^{r}{\alpha}_{j} {\lambda}^{j}$$
Graph convolution with (orthogonal) Chebyshev polynomials
$$w_{\alpha}(\lambda) = {\sum}_{j=0}^{r}{\alpha}_{j} T_j(\lambda)$$
where $T_k(x)$ are the Chebyshev polynomials.

Kipf and Welling proposed the ChebNet (arXiv:1609.02907) to approximate the filter using Chebyshev polynomial.
Application of the filter with the scaled Laplacian:
$$\tilde{\mathbf{\Delta}}=2{\lambda}_{n}^{-1}\mathbf{\Delta-I}$$

$$w_{\alpha}(\tilde{\mathbf{\Delta}})f= {\sum}_{j=0}^r{\alpha}_{j} T_j({\tilde{\mathbf{\Delta}}}) f={\sum}_{j=0}^r{\alpha}_{j}X^{(j)}$$
with
$$X^{(j)}=T_j({\tilde{\mathbf{\Delta}}}) f=2\mathbf{\Delta} X^{(j-1)}-X^{(j-2)}, X^{(0)}=f, X^{(1)}=\tilde{\mathbf{\Delta}} f.$$


* [Graph Convolutional Neural Network (Part I)](https://datawarrior.wordpress.com/2018/08/08/graph-convolutional-neural-network-part-i/)
* https://www.ntu.edu.sg/home/xbresson/
* https://github.com/xbresson

#### Simplified ChebNets

Use Chebychev polynomials of degree $r=2$ and assume $r_2\approx 2$:
$$w_{\alpha}(\tilde{\mathbf{\Delta}})f ={\alpha}_0 f + {\alpha}_1(\mathbf{\Delta-I})f= {\alpha}_0 f - {\alpha}_1\mathbf D^{-\frac{1}{2}}WD^{-\frac{1}{2}} f $$

Further constrain $\alpha=-\alpha_1=\alpha_0$ to obtain a single-parameter filter:
$$w_{\alpha}(\tilde{\mathbf{\Delta}})f ={\alpha}\mathbf{(I-D^{-\frac{1}{2}}WD^{-\frac{1}{2}})} f $$

#### PinSage

<img src=https://pic3.zhimg.com/80/v2-34c698539a34d506ff3f05c24ddd3482_hd.jpg width=70% />
<img src=https://pic2.zhimg.com/80/v2-41f380e6db85ae9173701c33c0f75311_hd.jpg width=70% />

### ChebNet, CayleyNet, MotifNet

In the previous post, the convolution of the graph Laplacian is defined in its **graph Fourier space** as outlined in the paper of Bruna et. al. (arXiv:1312.6203). However, the **eigenmodes** of the graph Laplacian are not ideal because it makes the bases to be graph-dependent. A lot of works were done in order to solve this problem, with the help of various special functions to express the filter functions. Examples include Chebyshev polynomials and Cayley transform.

Graph Convolution Networks (GCNs) generalize the operation of convolution from traditional data (images or grids) to graph data.
The key is to learn a function f to generate
a node $v_i$’s representation by aggregating its own features
$X_i$ and neighbors? features $X_j$ , where $j \in N(v_i)$.



#### CayleyNet

Defining filters as polynomials applied over the eigenvalues of the `graph Laplacian`, it is possible
indeed to avoid any eigen-decomposition and realize convolution by means of efficient sparse routines
The main idea behind `CayleyNet` is to achieve some sort of spectral zoom property by means of Cayley transform:
$$
C(\lambda) = \frac{\lambda - i}{\lambda + i}
$$

Instead of Chebyshev polynomials, it approximates the filter as:
$$
g(\lambda) = c_0 + \sum_{j=1}^{r}[c_jC^{j}(h\lambda) + c_j^{\ast} C^{j^{\ast}}(h\lambda)]
$$
where $c_0$ is real and other $c_j'$ s are generally complex, and ${h}$ is a zoom parameter, and $\lambda'$ s are the eigenvalues of the graph Laplacian.
Tuning ${h}$ makes one find the best zoom that spread the top eigenvalues. ${c}$'s are computed by training. This solves the problem of unfavorable clusters in `ChebNet`.

* [CayleyNets: Graph Convolutional Neural Networks with Complex Rational Spectral Filters](https://arxiv.org/abs/1705.07664)
* [CayleyNets at IEEE](https://ieeexplore.ieee.org/document/8521593)

#### MotifNet

`MotifNet` is aimed to address the directed graph convolution.

<img title="motifNet" src=https://datawarrior.files.wordpress.com/2018/08/f1-large.jpg width=70%/>


* [MotifNet: a motif-based Graph Convolutional Network for directed graphs](https://arxiv.org/abs/1802.01572)
* [Neural Motifs: Scene Graph Parsing with Global Context (CVPR 2018)](https://rowanzellers.com/neuralmotifs/)
* [GCN Part II @datawarrior](https://datawarrior.wordpress.com/2018/08/12/graph-convolutional-neural-network-part-ii/)
* http://mirlab.org/conference_papers/International_Conference/ICASSP%202018/pdfs/0006852.pdf

Minimal inner structures:

* Invariant by vertex re-indexing (no graph matching is required)
* Locality (only neighbors are considered)
Weight sharing (convolutional operations)
* Independence w.r.t. graph size

$$\color{green}{\fbox{${h_i} = f_{GCN}(h_j: j\to i)$} }$$

+ https://github.com/xbresson

#### Higher-order Graph Convolutional Networks

- [Higher-order Graph Convolutional Networks](http://ryanrossi.com/pubs/Higher-order-GCNs.pdf)
- [A Higher-Order Graph Convolutional Layer](http://sami.haija.org/papers/high-order-gc-layer.pdf)
- [MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing](http://proceedings.mlr.press/v97/abu-el-haija19a/abu-el-haija19a.pdf)
***
* https://zhuanlan.zhihu.com/p/62300527
* https://zhuanlan.zhihu.com/p/64498484
* https://zhuanlan.zhihu.com/p/28170197
* [Wavelets on Graphs via Spectral Graph Theory](https://arxiv.org/abs/0912.3848)
* [Spectral Networks and Locally Connected Networks on Graphs](https://arxiv.org/abs/1312.6203)

#### Graph Attention Networks

Graph convolutional layer then computes a set of new node features, $(\vec{h}_{1},\cdots, \vec{h}_{n})$ , based on the input features as well as the graph structure.

Most prior work defines the kernels $\alpha_{ij}$ explicitly (either based on the structural properties of the graph, or as a learnable weight); this requires compromising at least one other desirable property.

In `Graph Attention Networks` the kernels $\alpha_{ij}$ be computed as a byproduct of an attentional mechanism, $a : \mathbb{R}^N \times \mathbb{R}^N \rightarrow \mathbb{R}$, which computes un-normalized coefficients $e_{ij}$ across pairs of nodes $i,j$ based on their features:

$$
e_{ij} = a(\vec{h}_i, \vec{h}_j).
$$

We inject the graph structure by only allowing node $i$ to attend over nodes in its neighbourhood, $j\in\mathcal{N}_{i}$.These coefficients are then typically normalised using the softmax function, in order to be comparable across different neighbourhoods:
$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k\in\mathcal{N}_i}\exp(e_{ik})}.
$$

- http://petar-v.com/GAT/

<img src="https://www.cl.cam.ac.uk/~pv273/images/gat.jpg" width="70%" />

#### Gated Graph Neural Networks

- https://github.com/Microsoft/gated-graph-neural-network-samples

#### Generative Models for Graphs

* [Generative Models for Graphs by David White & Richard Wilson](http://david-white.net/generative.html)

### Application

- [ ] [graph convolution network 有什么比较好的应用task? - 知乎](https://www.zhihu.com/question/305395488/answer/554847680)
- [ ] [Use of graph network in machine learning](https://datawarrior.wordpress.com/2018/09/16/use-of-graph-networks-in-machine-learning/)
- [ ] [Node Classification by Graph Convolutional Network](https://www.experoinc.com/post/node-classification-by-graph-convolutional-network)
- [ ] [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

**GCN for RecSys**

**PinSAGE**
Node’s neighborhood defines a computation graph. The key idea is to generate node embeddings based on local neighborhoods. Nodes aggregate information from
their neighbors using neural networks.

- [ ] [Graph Neural Networks for Social Recommendation](https://paperswithcode.com/paper/graph-neural-networks-for-social)
- [ ] [图神经网+推荐](https://daiwk.github.io/posts/dl-graph-recommendations.html)
- [ ] [Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/abs/1806.01973)
- [ ] [Graph Convolutional Networks for Recommender Systems](https://arxiv.org/abs/1904.12575)

**GCN for Bio & Chem**

- [DeepChem is a Python library democratizing deep learning for science.](https://deepchem.io/docs/notebooks/index.html)
- [Chemi-Net: A molecular graph convolutional network for accurate drug property prediction](https://arxiv.org/ftp/arxiv/papers/1803/1803.06236.pdf)
- [Chainer Chemistry: A Library for Deep Learning in Biology and Chemistry](https://github.com/pfnet-research/chainer-chemistry)
- [Release Chainer Chemistry: A library for Deep Learning in Biology and Chemistry](https://preferredresearch.jp/2017/12/18/chainer-chemistry-beta-release/)
- [Modeling Polypharmacy Side Effects with Graph Convolutional Networks](https://cs.stanford.edu/people/jure/pubs/drugcomb-ismb18.pdf)
- http://www.grakn.ai/?ref=Welcome.AI
- [AlphaFold: Using AI for scientific discovery](https://deepmind.com/blog/alphafold/)
- [A graph-convolutional neural network model for the prediction of chemical reactivity](https://pubs.rsc.org/en/content/articlepdf/2019/sc/c8sc04228d)
- [Convolutional Networks on Graphs for Learning Molecular Fingerprints](http://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints)


**GCN for NLP**

- https://www.akbc.ws/2019/
- http://www.akbc.ws/2017/slides/ivan-titov-slides.pdf
- https://github.com/icoxfog417/graph-convolution-nlp
- https://nlp.stanford.edu/pubs/zhang2018graph.pdf

***
* https://cs.stanford.edu/people/jure/
* https://github.com/alibaba/euler
* https://ieeexplore.ieee.org/document/8439897
* [Higher-order organization of complex networks](http://science.sciencemag.org/content/353/6295/163)
* [Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks](https://arxiv.org/abs/1704.06803)

*****

- http://snap.stanford.edu/proj/embeddings-www/
- http://ryanrossi.com/
- http://www.ipam.ucla.edu/programs/workshops/geometry-and-learning-from-data-tutorials/
- https://zhuanlan.zhihu.com/p/51990489
- https://www.cs.toronto.edu/~yujiali/

*****

* [Python for NLP](https://synalp.loria.fr/python4nlp/)
* [Deep Learning on Graphs: A Survey](https://arxiv.org/abs/1812.04202)
* [Graph-based Neural Networks](https://github.com/sungyongs/graph-based-nn)
* [Geometric Deep Learning](http://geometricdeeplearning.com/)
* [Deep Chem](https://deepchem.io/)
* [GRAM: Graph-based Attention Model for Healthcare Representation Learning](https://arxiv.org/abs/1611.07012)
* https://zhuanlan.zhihu.com/p/49258190
* https://www.zhihu.com/question/54504471
* http://sungsoo.github.io/2018/02/01/geometric-deep-learning.html
* https://rusty1s.github.io/pytorch_geometric/build/html/notes/introduction.html
* [.mp4 illustration](http://tkipf.github.io/graph-convolutional-networks/images/video.mp4)
* [Deep Graph Library (DGL)](https://www.dgl.ai/)
* https://github.com/alibaba/euler
* https://github.com/alibaba/euler/wiki/%E8%AE%BA%E6%96%87%E5%88%97%E8%A1%A8
* https://www.groundai.com/project/graph-convolutional-networks-for-text-classification/
* https://datawarrior.wordpress.com/2018/08/08/graph-convolutional-neural-network-part-i/
* https://datawarrior.wordpress.com/2018/08/12/graph-convolutional-neural-network-part-ii/
* http://www.cs.nuim.ie/~gunes/files/Baydin-MSR-Slides-20160201.pdf
* http://colah.github.io/posts/2015-09-NN-Types-FP/
* https://www.zhihu.com/question/305395488/answer/554847680

***
* http://www.cis.upenn.edu/~kostas/
* https://sites.google.com/site/deepgeometry/slides-1
* https://sites.google.com/site/erodola/publications
* https://people.lu.usi.ch/mascij/publications.html
* http://www.cs.utoronto.ca/~fidler/publications.html
* http://practicalcheminformatics.blogspot.com/
