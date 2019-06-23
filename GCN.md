## Geometric Deep Learning

* http://www.ai3sd.org/
* https://heidelberg.ai/2019/07/09/graph-neural-networks.html
* https://sites.google.com/site/rdftestxyz/home
* [Lecture 11: Learning on Non-Euclidean Domains](https://vistalab-technion.github.io/cs236605/lecture_notes/lecture_11/)
* [Network Embedding](https://shiruipan.github.io/project/effective-network-embedding/)
* [Learning and Reasoning with Graph-Structured Representations, ICML 2019 Workshop](https://graphreason.github.io/index.html)
* [Transformer结构及其应用--GPT、BERT、MT-DNN、GPT-2 - Ph0en1x的文章 - 知乎](https://zhuanlan.zhihu.com/p/69290203)
* [放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)
* [如何评价百度新发布的NLP预训练模型ERNIE？ - 飞桨PaddlePaddle的回答 - 知乎](https://www.zhihu.com/question/316140575/answer/719617103)
* [What Can Neural Networks Reason About?](https://arxiv.org/abs/1905.13211)
* [Deep Geometric Matrix Completion by Federico Monti](http://helper.ipam.ucla.edu/publications/dlt2018/dlt2018_14552.pdf)

[In the last decade, Deep Learning approaches (e.g. Convolutional Neural Networks and Recurrent Neural Networks) allowed to achieve unprecedented performance on a broad range of problems coming from a variety of different fields (e.g. Computer Vision and Speech Recognition). Despite the results obtained, research on DL techniques has mainly focused so far on data defined on Euclidean domains (i.e. grids). Nonetheless, in a multitude of different fields, such as: Biology, Physics, Network Science, Recommender Systems and Computer Graphics; one may have to deal with data defined on non-Euclidean domains (i.e. graphs and manifolds). The adoption of Deep Learning in these particular fields has been lagging behind until very recently, primarily since the non-Euclidean nature of data makes the definition of basic operations (such as convolution) rather elusive. Geometric Deep Learning deals in this sense with the extension of Deep Learning techniques to graph/manifold structured data.](http://geometricdeeplearning.com/)

- [Computational Learning and Memory Group](http://cbl.eng.cam.ac.uk/Public/Lengyel/News)
- [Beyond deep learning](http://beyond-deep-nets.clps.brown.edu/)
- [Cognitive Computation Group @ U. Penn.](https://cogcomp.org/)
- [Computational cognitive modeling](https://brendenlake.github.io/CCM-site/)
- [Mechanisms of geometric cognition](http://hohol.pl/granty/geometry/)
- [Computational Cognitive Science Lab](http://cocosci.princeton.edu/research.php)
- https://jian-tang.com/teaching/graph2019
- [Introducing Grakn & Knowledge Graph Convolutional Networks: Dec 4, 2018 · Paris, France](https://eventil.com/events/introducing-grakn-ai-to-paris)
- [International Workshop on Deep Learning for Graphs and Structured Data Embedding](https://www.aminer.cn/dl4g-sde)
- [HYPERBOLIC DEEP LEARNING: A nascent and promising field](http://hyperbolicdeeplearning.com/papers/)

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

### Graph Embedding

Graph embedding, preprocessing of graph data processing, is an example of representation learning to find proper numerical representation form of graph data structure.

- https://github.com/thunlp/NRLpapers
- https://github.com/thunlp/GNNPapers
- http://snap.stanford.edu/proj/embeddings-www/
- https://arxiv.org/abs/1709.05584
- http://cazabetremy.fr/Teaching/EmbeddingClass.html
- [Awesome Graph Embedding](https://github.com/benedekrozemberczki/awesome-graph-embedding)
- [A Beginner's Guide to Graph Analytics and Deep Learning](https://skymind.ai/wiki/graph-analysis)
- [Representation Learning on Graphs: Methods and Applications](https://www-cs.stanford.edu/people/jure/pubs/graphrepresentation-ieee17.pdf)
- [DOOCN-XII: Network Representation Learning
Dynamics On and Of Complex Networks 2019](http://doocn.org/)
- [15TH INTERNATIONAL WORKSHOP ON
MINING AND LEARNING WITH GRAPHS](http://www.mlgworkshop.org/2019/)
- [Hyperbolic geometry and real life networks](http://www.ec2017.org/celinska/)
- [Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec](https://arxiv.org/abs/1710.02971)

**DeepWalk**

`DeepWalk` is an approach for learning latent representations of vertices in a network, which maps the nodes in the graph into real vectors:
$$
f: \mathbb{V}\to\mathbb{R}^{d}.
$$

[DeepWalk generalizes recent advancements in language modeling and unsupervised feature learning (or deep learning) from sequences of words to graphs. DeepWalk uses local information obtained from truncated random walks to learn latent representations by treating walks as the equivalent of sentences.](http://www.perozzi.net/projects/deepwalk/)
And if we consider the  text as digraph, `word2vec` is an specific example of `DeepWalk`.
Given the word sequence $\mathbb{W}=(w_0, w_1, \dots, w_n)$, we can compute the conditional probability $P(w_n|w_0, w_1, \dots, w_{n-1})$. And
$$P(w_n|f(w_0), f(w_1),⋯, f(w_{n−1}))$$

>
> * DeepWalk  $G, w, d, \gamma, t$
>> * Input: graph $G(V, E)$;
     window size $w$;
     embedding size $d$;
     walks per vertex $\gamma$;
     walk length $t$.
>> * Output:  matrix of vertex representations $\Phi\in\mathbb{R}^{|V|\times d}$
>
>> *  Initialization: Sample $\Phi$ from $\mathbb{U}^{|V|\times d}$;
>>     + Build a binary Tree T from V;
>>     + for $i = 0$ to $\gamma$ do
>>        -  $O = Shuffle(V )$
>>        -  for each $v_i \in O$ do
>>        -  $W_{v_i}== RandomWalk(G, v_i, t)$
>>        -  $SkipGram(Φ, W_{v_i}, w)$
>>        - end for
>>     + end for
***  

> $SkipGram(Φ, W_{v_i}, w)$
* 1. for each $v_j \in W_{v_i}$ do
    + 2. for each $u_k \in W_{v_i}[j − w : j + w]$ do
    + 3.  $J(\Phi)=-\log Pr(u_k\mid \Phi(v_j))$
    + 4.  $\Phi =\Phi -\alpha\frac{\partial J}{\partial \Phi}$
    + 5. end for
* 6. end for

Computing the partition function (normalization factor) is expensive, so instead we will factorize the
conditional probability using `Hierarchical softmax`.
If the path to vertex $u_k$ is identified by a sequence of tree nodes $(b_0, b_1, \cdots , b_{[log |V|]})$,
and $(b_0 = root, b_{[log |V|]} = u_k)$ then
$$
Pr(u_k\mid \Phi(v_j)) =\prod_{l=1}^{[log |V|]}Pr(b_l\mid \Phi(v_j)).
$$


Now, $Pr(b_l\mid \Phi(v_j))$ could be modeled by a binary classifier
that is assigned to the parent of the node $b_l$ as below
$$
Pr(b_l\mid \Phi(v_j))=\frac{1}{1 + \exp(-\Phi(v_j)\cdot \Phi(b_l))}
$$
where $\Phi(b_l)$ is the representation assigned to tree node
$b_l$ ’s parents.



***
- [ ] [The Expressive Power of Word Embeddings](https://arxiv.org/abs/1301.3226)
- [ ] [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652)
- [ ] https://github.com/phanein/deepwalk
- [ ] http://www.perozzi.net/projects/deepwalk/
- [ ] http://www.cnblogs.com/lavi/p/4323691.html
- [ ] https://www.ijcai.org/Proceedings/16/Papers/547.pdf

**node2vec**

`node2vec` is an algorithmic framework for representational learning on graphs. Given any graph, it can learn continuous feature representations for the nodes, which can then be used for various downstream machine learning tasks.

By extending the Skip-gram architecture to networks, it seeks to optimize the following objective function,
which maximizes the log-probability of observing a network neighborhood $N_{S}(u)$ for a node $u$ conditioned on its feature representation, given by $f$:
$$
\max_{f} \sum_{u\in V}\log Pr(N_S(u)\mid f(u))
$$

`Conditional independence` and `Symmetry` in feature space are expected  to  make the optimization problem tractable.
We model the conditional likelihood of every source-neighborhood node pair as a softmax
unit parametrized by a dot product of their features:
$$
Pr(n_i\mid f(u))=\frac{\exp(f(n_i)\cdot f(u))}{\sum_{v\in V} \exp(f(v)\cdot f(u))}.
$$

The objective function simplifies to
$$\max_{f}\sum_{u\in V}[-\log Z_u + \sum_{n_i\in N_S(u)} f(n_i)\cdot f(u).$$

- [ ] https://zhuanlan.zhihu.com/p/30599602
- [ ] http://snap.stanford.edu/node2vec/
- [ ] https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf
- [ ] https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf


**struc2vec**

- [ ] http://jackieanxis.coding.me/2018/01/17/STRUC2VEC/
- [ ] http://www.land.ufrj.br/~leo/struc2vec.html
- [ ] https://arxiv.org/abs/1704.03165
- [Struc2vec: learning node representations from structural identity](https://blog.acolyer.org/2017/09/15/struc2vec-learning-node-representations-from-structural-identity/)

This is a paper about identifying nodes in graphs that play a similar role based solely on the structure of the graph, for example computing the structural identity of individuals in social networks. That’s nice and all that, but what I personally find most interesting about the paper is the meta-process by which the authors go about learning the latent distributed vectors that capture the thing they’re interested in (structural similarity in this case). Once you’ve got those vectors, you can do vector arithmetic, distance calculations, use them in classifiers and other higher-level learning tasks and so on. As word2vec places semantically similar words close together in space, so we want structurally similar nodes to be close together in space.

Struc2vec has four main steps:

> 1. Determine the structural similarity between each vertex pair in the graph, for different neigbourhood sizes.
> 2. Construct a weighted multi-layer graph, in which each layer corresponds to a level in a hierarchy measuring structural similarity (think: ‘at this level of zoom, these things look kind of similar’).
> 3. Use the multi-layer graph to generate context for each node based on biased random walking.
> 4. Apply standard techniques to learn a latent representation from the context given by the sequence of nodes in the random walks.

![](https://adriancolyer.files.wordpress.com/2017/09/struc2vec-sketch-8.jpeg?w=200&zoom=1)

**word2vec**

In natural language processing, the word can be regarded as the node in a graph, which only takes the relation of locality or context.
It is difficult to learn the concepts or the meaning of words. The word embedding technique `word2vec` maps the words to fixed length real vectors:
$$
f: \mathbb{W}\to\mathbb{V}^d\subset \mathbb{R}^d.
$$

The `skip-gram model` assumes that a word can be used to generate the words that surround it in a text sequence.
We assume that, given the central target word, the context words are generated independently of each other.

The conditional probability of generating the context word for the given central target word can be obtained by performing a softmax operation on the vector inner product:
$$
P(w_o|w_c) = \frac{\exp(u_o^T u_c)}{\sum_{i\in\mathbb{V}} \exp(u_i^T u_c)},
$$

where vocabulary index set $V = \{1,2,\dots, |V|-1\}$. Assume that a text sequence of length ${T}$  is given, where the word at time step  ${t}$  is denoted as  $w^{(t)}$.
Assume that context words are independently generated given center words. When context window size is  ${m}$ , the likelihood function of the skip-gram model is the joint probability of generating all the context words given any center word
$$
\prod_{t=1}^{T}\prod_{-m\leq j \leq m, j\not = i}{P}(w^{(t+j)}|w^{(j)}),
$$

Here, any time step that is less than 1 or greater than ${T}$  can be ignored.

The skip-gram model parameters are the central target word vector and context word vector for each individual word. In the training process, we are going to learn the model parameters by maximizing the likelihood function, which is also known as maximum likelihood estimation. his is equivalent to minimizing the following loss function:
$$
-\log(\prod_{t=1}^{T}\prod_{-m\leq j \leq m, j\not = i}{P}(w^{(t+j)}|w^{(j)}))
= \\ -\sum_{t=1}^{T}\sum_{-m\leq j \leq m, j \not= i} \log({P}(w^{(t+j)}|w^{(j)}))).
$$

And we could compute the negative logarithm of the conditional probability
$$
-\log(P(w_o|w_c))
= -\log(\frac{\exp(u_o^T u_c)}{\sum_{i\in\mathbb{V}} \exp(u_i^T u_c)})
\\= -u_o^T u_c + \log(\sum_{i\in\mathbb{V}} \exp(u_i^T u_c)).
$$

Then we could compute the gradient or Hessian matrix of the loss functions to update the parameters such as:

$$
\frac{\partial \log(P(w_o|w_c))}{\partial u_c}
    \\= \frac{\partial }{\partial u_c} [u_o^T u_c - \log(\sum_{i\in\mathbb{V}} \exp(u_i^T u_c))]
    \\= u_o - \sum_{j\in\mathbb{V}}\frac{ \exp(u_j^T u_c)) }{\sum_{i\in\mathbb{V}} \exp(u_i^T u_c))} u_j
    \\= u_o - \sum_{j\in\mathbb{V}}P(w_j|w_c) u_j.
$$

The `continuous bag of words (CBOW)` model is similar to the skip-gram model. The biggest difference is that the CBOW model assumes that the central target word is generated based on the context words before and after it in the text sequence. Let central target word  $w_c$  be indexed as $c$ , and context words  $w_{o_1},…,w_{o_{2m}}$  be indexed as  $o_1,…,o_{2m}$  in the dictionary. Thus, the conditional probability of generating a central target word from the given context word is

$$
P(w_c|w_{o_1},…,w_{o_{2m}}) = \frac{\exp(\frac{1}{2m}u_c^T(u_{o_1}+ \cdots + u_{o_{2m}}))}{\sum_{i\in V}\exp(\frac{1}{2m}u_i^T(u_{o_1}+ \cdots + u_{o_{2m}}))}.
$$

- https://code.google.com/archive/p/word2vec/
- https://skymind.ai/wiki/word2vec
- https://arxiv.org/abs/1402.3722v1
- https://zhuanlan.zhihu.com/p/35500923
- https://zhuanlan.zhihu.com/p/26306795
- https://zhuanlan.zhihu.com/p/56382372
- http://anotherdatum.com/vae-moe.html
- https://d2l.ai/chapter_natural-language-processing/word2vec.html
- https://www.gavagai.io/text-analytics/word-embeddings/

**Doc2Vec**

- [ ] https://blog.csdn.net/Walker_Hao/article/details/78995591
- [Distributed Representations of Sentences and Documents](http://proceedings.mlr.press/v32/le14.pdf)
- [Sentiment Analysis using Doc2Vec](http://linanqiu.github.io/2015/10/07/word2vec-sentiment/)
***
- [Statistical Models of Language](http://cocosci.princeton.edu/publications.php?topic=Statistical%20Models%20of%20Language)
- [Semantic Word Embeddings](https://www.offconvex.org/2015/12/12/word-embeddings-1/)
- [Word Embeddings](https://synalp.loria.fr/python4nlp/posts/embeddings/)
- [GloVe: Global Vectors for Word Representation Jeffrey Pennington,   Richard Socher,   Christopher D. Manning
](https://nlp.stanford.edu/projects/glove/)
- [BERT-is-All-You-Need](https://github.com/Eurus-Holmes/BERT-is-All-You-Need)
- [Word embedding](https://levyomer.wordpress.com/category/word-embeddings/)
- [Open Sourcing BERT: State-of-the-Art Pre-training for Natural Language Processing
Friday, November 2, 2018](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning) ](http://jalammar.github.io/illustrated-bert/)
- [Deep Semantic Embedding](http://smir2014.noahlab.com.hk/paper%204.pdf)
- [无监督词向量/句向量:W2v/Glove/Swivel/ELMo/BERT](https://x-algo.cn/index.php/2018/11/12/3083/)
***
**graph2vec**

- [ ] https://arxiv.org/abs/1707.05005
- [ ] https://allentran.github.io/graph2vec
- [ ] http://humanativaspa.it/tag/graph2vec/

**GB-CENT**

- http://www.hongliangjie.com/talks/GB-CENT_SD_2017-02-22.pdf
- http://www.hongliangjie.com/talks/GB-CENT_SantaClara_2017-03-28.pdf
- http://www.hongliangjie.com/talks/GB-CENT_Lehigh_2017-04-12.pdf
- http://www.hongliangjie.com/talks/GB-CENT_PopUp_2017-06-14.pdf
- http://www.hongliangjie.com/talks/GB-CENT_CAS_2017-06-23.pdf
- http://www.hongliangjie.com/talks/GB-CENT_Boston_2017-09-07.pdf
- [Talk: Gradient Boosted Categorical Embedding and Numerical Trees](http://www.hongliangjie.com/talks/GB-CENT_MLIS_2017-06-06.pdf)
- [Paper: Gradient Boosted Categorical Embedding and Numerical Trees](http://papers.www2017.com.au.s3-website-ap-southeast-2.amazonaws.com/proceedings/p1311.pdf)

**Gaussian Auto Embeddings**

http://koaning.io/gaussian-auto-embeddings.html

**Atom2Vec**

M. V. Diudea, I. Gutman and L. Jantschi wrote in the preface of the book _Molecular Topology_:
[One of the principal goals of chemistry is to establish (causal) relations between the chemical and
physical (experimentally observable and measurable) properties of substance and the
structure of the corresponding molecules. Countless results along these lines have been
obtained, and their presentation comprise significant parts of textbooks of organic,
inorganic and physical chemistry, not to mention treatises on theoretical chemistry](http://www.moleculartopology.com/)

- [ ] [Deep Learning For Molecules and Materials](http://www.rqrm.ca/DATA/TEXTEDOC/03a-total-september2018-v1.pdf)

**tile2Vec**

- [ ] https://ermongroup.github.io/blog/tile2vec/
- [ ] https://arxiv.org/abs/1805.02855

* [RDF2Vec: RDF Graph Embeddings and Their Applications](http://www.semantic-web-journal.net/system/files/swj1495.pdf)
* [EmbedS: Scalable and Semantic-Aware Knowledge Graph Embeddings](https://expolab.org/papers/embeds-slides.pdf)

 **Graph Embedding**

- [ ] https://zhuanlan.zhihu.com/p/33732033
- [ ] [Awesome graph embedding](https://github.com/benedekrozemberczki/awesome-graph-embedding)
- [ ] [Graph Embedding Methods](https://github.com/palash1992/GEM)
- [ ] [Graph Embedding @ deep learning pattern](https://www.deeplearningpatterns.com/doku.php?id=graph_embedding)
- [ ] [Representation learning in graph and manifold](https://rlgm.github.io/)
- [ ] [Learning and Reasoning with Graph-Structured Representations
ICML 2019 Workshop](https://graphreason.github.io/)
- [ ] [LINE: Large-scale Information Network Embedding](https://arxiv.org/abs/1503.03578)
- [ ] [Latent Network Summarization: Bridging Network Embedding and Summarization](http://ryanrossi.com/pubs/Jin-et-al-latent-network-summarization.pdf)
- [ ] [NodeSketch: Highly-Efficient Graph Embeddings via Recursive Sketching](https://exascale.info/assets/pdf/yang2019nodesketch.pdf)
- [ ] http://web.eecs.umich.edu/~dkoutra/group/index.html
- [Graph Embeddings — The Summary](http://sungsoo.github.io/2019/05/26/graph-embedding.html)
- [Graph Embeddings search result @ Arxiv-sanity](http://www.arxiv-sanity.com/search?q=Graph+Embedding)

**Deep Semantic Embedding**

+ http://smir2014.noahlab.com.hk/paper%204.pdf
+ [Deep Embedding Logistic Regression](https://www.cse.wustl.edu/~z.cui/papers/DELR_ICBK.pdf)
+ [DeViSE: A Deep Visual-Semantic Embedding Model](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/41473.pdf)
+ [Deep Visual-Semantic Alignments for Generating Image Descriptions](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)
+ [Semantic Embedding for Sketch-Based 3D Shape Retrieval](http://bmvc2018.org/contents/papers/0040.pdf)

**Hyperbolic Embeddings**

- [Spherical and Hyperbolic Embeddings of Data](https://www.cs.york.ac.uk/cvpr/embedding/index.html)
- [Embedding Networks in Hyperbolic Spaces ](http://bactra.org/notebooks/hyperbolic-networks.html)
- [Characterizing the analogy between hyperbolic embedding and community structure of complex networks ](http://doocn.org/2018/files/radicchi-slides.pdf)
- [Poincaré Embeddings for Learning Hierarchical Representations](https://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations.pdf)
- [Implementing Poincaré Embeddings](https://rare-technologies.com/implementing-poincare-embeddings/)
- [Hyperbolic Embedding search result @Arxiv-sanity](http://www.arxiv-sanity.com/search?q=Hyperbolic+Embeddings)

<img src=https://hazyresearch.github.io/hyperE/pc.svg  width=40% />

- [Hyperbolic Embeddings with a Hopefully Right Amount of Hyperbole](https://dawn.cs.stanford.edu/2018/03/19/hyperbolics/)
- [HyperE: Hyperbolic Embeddings for Entities](https://hazyresearch.github.io/hyperE/)
- [Efficient embedding of complex networks to hyperbolic space via their Laplacian](https://www.nature.com/articles/srep30108)
- [Embedding Text in Hyperbolic Spaces](https://ai.google/research/pubs/pub47117)
- [Hyperbolic Function Embedding: Learning Hierarchical Representation for Functions of Source Code in Hyperbolic Spaces](https://www.mdpi.com/2073-8994/11/2/254/htm)
- http://hyperbolicdeeplearning.com/papers/


--|--
--|--
![combinatorial_tree](https://hazyresearch.github.io/hyperE/pytorch_tree.gif)|![](https://hazyresearch.github.io/hyperE/combinatorial_tree.gif)

+ http://bjlkeng.github.io/posts/hyperbolic-geometry-and-poincare-embeddings/
*****

- https://zhuanlan.zhihu.com/p/47489505
- http://blog.lcyown.cn/2018/04/30/graphencoding/
- https://blog.csdn.net/NockinOnHeavensDoor/article/details/80661180
- http://building-babylon.net/2018/04/10/graph-embeddings-in-hyperbolic-space/
- https://paperswithcode.com/task/graph-embedding

### Graph Convolutional Network


Graph can be represented as `adjacency matrix` as shown in *Graph Algorithm*. However, the adjacency matrix only describe the connections between the nodes.
The feature of the nodes does not appear. The node itself really matters.
For example, the chemical bonds can be represented as `adjacency matrix` while the atoms in molecule really determine the properties of the molecule.

A simple and direct way is to concatenate the `feature matrix` $X\in \mathbb{R}^{N\times E}$ and `adjacency matrix` $A\in\mathbb{B}^{N\times N}$, i.e., $X_{in}=[X, A]\in \mathbb{R}^{N\times (N+E)}$. What is more, $\mathbb{B}$ is binary so that each element of the adjacent  matrix $a_{i,j}$ is ${0}$ or ${1}$.

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

* The second major limitation is that $A$ is typically not normalized and therefore the multiplication with $A$ will completely change the scale of the feature vectors (we can understand that by looking at the eigenvalues of $A$).Normalizing ${A}$ such that all rows sum to one, i.e. $D^{−1}A$, where $D$ is the diagonal node degree matrix, gets rid of this problem.

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
### ChebNet, CayleyNet, MotifNet

In the previous post, the convolution of the graph Laplacian is defined in its **graph Fourier space** as outlined in the paper of Bruna et. al. (arXiv:1312.6203). However, the **eigenmodes** of the graph Laplacian are not ideal because it makes the bases to be graph-dependent. A lot of works were done in order to solve this problem, with the help of various special functions to express the filter functions. Examples include Chebyshev polynomials and Cayley transform.

Graph Convolution Networks (GCNs) generalize the operation of convolution from traditional data (images or grids) to graph data.
The key is to learn a function f to generate
a node $v_i$’s representation by aggregating its own features
$X_i$ and neighbors’ features $X_j$ , where $j \in N(v_i)$.

**ChebNet**

Kipf and Welling proposed the ChebNet (arXiv:1609.02907) to approximate the filter using Chebyshev polynomial. It approximates the filter in terms of Chebyshev polynomial:
$$
g_{\theta}(\Lambda) \approx \sum_{k=0}^{K}\theta_k T_k(\Lambda)
$$
where $T_k(x)$ are the Chebyshev polynomials.

**CayleyNet**

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

***

**MotifNet**

`MotifNet` is aimed to address the directed graph convolution.

<img title="motifNet" src=https://datawarrior.files.wordpress.com/2018/08/f1-large.jpg width=70%/>


* [MotifNet: a motif-based Graph Convolutional Network for directed graphs](https://arxiv.org/abs/1802.01572)
* [Neural Motifs: Scene Graph Parsing with Global Context (CVPR 2018)](https://rowanzellers.com/neuralmotifs/)
* https://datawarrior.wordpress.com/2018/08/12/graph-convolutional-neural-network-part-ii/
* http://mirlab.org/conference_papers/International_Conference/ICASSP%202018/pdfs/0006852.pdf

**Higher-order Graph Convolutional Networks**

- [Higher-order Graph Convolutional Networks](http://ryanrossi.com/pubs/Higher-order-GCNs.pdf)
- [A Higher-Order Graph Convolutional Layer](http://sami.haija.org/papers/high-order-gc-layer.pdf)
- [MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing](http://proceedings.mlr.press/v97/abu-el-haija19a/abu-el-haija19a.pdf)
***
* https://zhuanlan.zhihu.com/p/62300527
* https://zhuanlan.zhihu.com/p/64498484
* https://zhuanlan.zhihu.com/p/28170197
* [Wavelets on Graphs via Spectral Graph Theory](https://arxiv.org/abs/0912.3848)
* [Spectral Networks and Locally Connected Networks on Graphs](https://arxiv.org/abs/1312.6203)

### Application

- [ ] [graph convolution network 有什么比较好的应用task？ - superbrother的回答 - 知乎](https://www.zhihu.com/question/305395488/answer/554847680)
- [ ] [Use of graph network in machine learning](https://datawarrior.wordpress.com/2018/09/16/use-of-graph-networks-in-machine-learning/)
- [ ] [Node Classification by Graph Convolutional Network](https://www.experoinc.com/post/node-classification-by-graph-convolutional-network)
- [ ] [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

**GCN for RecSys**

- https://daiwk.github.io/posts/dl-graph-recommendations.html
- https://arxiv.org/abs/1806.01973
- https://arxiv.org/abs/1904.12575

**GCN for Bio & Chem**

- https://deepchem.io/docs/notebooks/index.html
- [Chemi-Net: A molecular graph convolutional network for accurate drug property prediction](https://arxiv.org/ftp/arxiv/papers/1803/1803.06236.pdf)
- [Chainer Chemistry: A Library for Deep Learning in Biology and Chemistry](https://github.com/pfnet-research/chainer-chemistry)
- [Modeling Polypharmacy Side Effects with Graph Convolutional Networks](https://cs.stanford.edu/people/jure/pubs/drugcomb-ismb18.pdf)
- http://www.grakn.ai/?ref=Welcome.AI
- https://deepmind.com/blog/alphafold/
- [A graph-convolutional neural network model for the prediction of chemical reactivity](https://pubs.rsc.org/en/content/articlepdf/2019/sc/c8sc04228d)

**GCN for NLP**

- https://www.akbc.ws/2019/
- http://www.akbc.ws/2017/slides/ivan-titov-slides.pdf
- https://github.com/icoxfog417/graph-convolution-nlp
- https://nlp.stanford.edu/pubs/zhang2018graph.pdf

***
* https://cs.stanford.edu/people/jure/
* https://github.com/alibaba/euler
* https://ieeexplore.ieee.org/document/8521593
* https://ieeexplore.ieee.org/document/8439897
* [Higher-order organization of complex networks](http://science.sciencemag.org/content/353/6295/163)
* [Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks](https://arxiv.org/abs/1704.06803)


*****

- http://snap.stanford.edu/proj/embeddings-www/
- http://ryanrossi.com/
- http://petar-v.com/GAT/
- http://www.ipam.ucla.edu/programs/workshops/geometry-and-learning-from-data-tutorials/
- https://github.com/Microsoft/gated-graph-neural-network-samples
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
