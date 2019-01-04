## Dimension Reduction

https://www.ayasdi.com/blog/artificial-intelligence/prediction-needs-unsupervised-learning/

Principal component analysis or singular value decomposition can applied to matrix approximation.
The data collected in practice always save in the table form, which can considered as a matrix.
Another techniques similar to PCA is  eigenvalue-eigenvector decomposition.
Dimension reduction is really the topic of data science as data preprocessing .

The basic idea of dimension reduction is that not all information is necessary for a specific task.
The motivation of dimension reduction is **Curse of dimensionality**, limit of storage and computation.
The high dimensional space is not easy for us to visualize, imagine or understand.
The intuitive to high dimensional space is weak for us, the people live in the three dimensional space.
As a preprocessing data method, it helps us to select features and learn proper representation.
A more theoretical and useful topic is data compression, a branch of information theory.
The dimension reduction is related with geometry of data set., which includes manifold learning and topological data analysis.
It is a wonderful review of dimension reduction at \url{https://lvdmaaten.github.io/publications/papers/TR_Dimensionality_Reduction_Review_2009.pdf}.

* https://www.wikiwand.com/en/Data_compression
* https://brilliant.org/wiki/compression/
* https://lvdmaaten.github.io/publications/papers/TR_Dimensionality_Reduction_Review_2009.pdf
* https://www.wikiwand.com/en/Curse_of_dimensionality

### PCA and More

The data in table form can be regarded as matrix in mathematics. And we can apply singular value decomposition  to low rank approximation or nonnegative matrix factorization, which we will talk in *PCA and SVD*.
It is classified as linear techniques.
And it can extend to kernel PCA and {generalized PCA}[https://www.asc.ohio-state.edu/lee.2272/mss/tr892.pdf].

### t-SNE

The basic idea of t-SNE is that the similarity should preserve after dimension reduction.
It maps the data $x$ in high dimensional space $X\subset\mathbb{R}^{p}$ to a low dimensional space $Y\subset\mathbb{R}^{d}$. **t-SNE** is an extension of stochastic neighbor embedding.

**Stochastic Neighbor Embedding (SNE)** starts by converting the high-dimensional Euclidean distances between data points into conditional probabilities that represent similarities.The similarity of data point $x_j$ to data point $x_i$ is the conditional probability, $p_{j|i}$, that $x_i$ would pick $x_j$ as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian centered at $x_i$.
Mathematically, the conditional probability $p_{j|i}$ is given by
$$p_{j|i}=\frac{exp(-\|x_i-x_j\|^2/2\sigma_i^2)}{\sum_{k=\not 1}exp(-\|x_k-x_i\|^2/2\sigma_i^2)}.$$
Because we are only interested in modeling pairwise similarities, we set the value of $p_{i|i}$ to zero. For the low-dimensional counterparts $y_i$ and $y_j$ of the high-dimensional data points $x_i$ and $x_j$, it is possible to compute a similar conditional probability, which we denote by $q_{j|i}$. we model the similarity of map point $y_j$ to map point $y_i$ by
$$q_{j|i}=\frac{exp(-\|y_i-y_j\|^2)}{\sum_{k=\not 1}exp(-\|y_k-y_i\|^2)}.$$ 

SNE minimizes the sum of cross entropy over all data points using a gradient descent
method. The cost function $C$ is given by
$$C=\sum_{i}\sum_{j}p_{j|i}\log(\frac{p_{j|i}}{q_{j|i}}).$$

In the high-dimensional space, we convert distances into probabilities using a Gaussian distribution. In the low-dimensional map, we can use a probability distribution that has much heavier tails than a Gaussian to convert distances into probabilities.
In t-SNE, we employ a **Student t-distribution** with one degree of freedom (which is the same
as a Cauchy distribution) as the heavy-tailed distribution in the low-dimensional map. Using this
distribution, the joint probabilities $q_{i|j}$ are defined as
$$q_{i|j}=\frac{(1+\|y_i-y_j\|^2)^{-1}}{\sum_{j\not= i} (1+\|y_i-y_j\|^2)^{-1}}.$$

* https://blog.paperspace.com/dimension-reduction-with-t-sne/
* https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/
* https://distill.pub/2016/misread-tsne
* https://www.wikiwand.com/en/T-distributed_stochastic_neighbor_embedding

### Auto-Encoder

* https://www.wikiwand.com/en/Nonlinear_dimensionality_reduction
* https://scikit-learn.org/stable/modules/decomposition.html#decompositions
* https://www.cs.toronto.edu/~hinton/science.pdf
* http://www.idm.pku.edu.cn/staff/wangyizhou/papers/GAE-CVPRwDeepVision2014.pdf
