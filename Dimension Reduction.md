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

$\color{aqua}{PS:}$ the dimension reduction is classified into unsupervised learning while it convert to optimization models.

It is a wonderful review of dimension reduction at \url{https://lvdmaaten.github.io/publications/papers/TR_Dimensionality_Reduction_Review_2009.pdf}.

* https://www.wikiwand.com/en/Data_compression
* https://brilliant.org/wiki/compression/
* https://lvdmaaten.github.io/publications/papers/TR_Dimensionality_Reduction_Review_2009.pdf
* https://www.wikiwand.com/en/Curse_of_dimensionality

### PCA and More

The data in table form can be regarded as matrix in mathematics. And we can apply singular value decomposition  to low rank approximation or non-negative matrix factorization, which we will talk in *PCA and SVD*.
It is classified as linear techniques.
And it can extend to kernel PCA and {generalized PCA}[https://www.asc.ohio-state.edu/lee.2272/mss/tr892.pdf].

### ICA

Independent component analysis is to find the latent independent factors that make up the observation, i.e,
$$
X=Bs
$$
where $X$, $s$, $B$ is the observation, latent variable and unknown mixing matrix.
It is based on PCA.

+ https://www.wikiwand.com/en/Independent_component_analysis
+ http://www.gipsa-lab.grenoble-inp.fr/~pierre.comon/FichiersPdf/polyD16-2006.pdf
+ https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf
+ http://arnauddelorme.com/ica_for_dummies/
+ https://sccn.ucsd.edu/wiki/Chapter_09:_Decomposing_Data_Using_ICA
+ http://cs229.stanford.edu/notes/cs229-notes11.pdf
+ https://blog.paperspace.com/dimension-reduction-with-independent-components-analysis/
+ http://deeplearning.stanford.edu/wiki/index.php/Independent_Component_Analysis
+ https://www.zhihu.com/question/28845451/answer/42292804
+ http://www.gipsa-lab.grenoble-inp.fr/~pierre.comon/publications_en.html#book

### Auto-Encoder

Auto-Encoder is a neural network model that compresses the original data and then encodes the compressed information such as
$$\mathbb{R}^{p}\stackrel{\sigma}{\to}\mathbb{R}^{n}\stackrel{\sigma}{\to}\mathbb{R}^{d}$$
where $n\le d=p$ and $\sigma$ is nonlinear activation function.
We can express it in mathematical formula
$$
y=\sigma(x)\in\mathbb{R}^{n},z=\sigma(y)\in\mathbb{R}^{d},
$$
where $x$, $y$, $z$ is the input, the hidden unit and output, respectively.
It is trying to learn an approximation to the identity function, so as to output $z$ that is similar to $x$.

Now suppose we have only unlabeled training examples set ${x^{(1)}, x^{(2)}, x^{(3)}, \dots}$, where $x^{(i)}\in\mathbb{R}^{p}$.

An autoencoder neural network is an unsupervised learning algorithm
that applies backpropagation, setting the target values to be equal to the inputs. I.e., it uses $z^{(i)} = x^{(i)}$. It
is trying to learn an approximation to the identity function, so as to output
$\hat{x}$ that is similar to $x$. The identity function seems a particularly trivial
function to be trying to learn; but by placing constraints on the network,
such as by limiting the number of hidden units, we can discover interesting structure about the data.

Recall that $y^{(j)}$ denotes the activation of hidden unit $j$ in the autoencoder and write $y^{(j)}(x)$ to denote the activation of this hidden unit when the network is given a specific input $x$, i.e., $y^{(j)}(x)=\sigma\circ(x)^{(j)}$.
Let
$$
\hat{\rho}_j=\frac{1}{m}\sum_{i=1}^{m}y^{j}(x^{(i)})
$$
be the average activation of hidden unit $j$ (averaged over the training set).
We will add an extra penalty term to our optimization
objective that penalizes $\rho^j$ deviating significantly from $\rho$.
We will choose the following:
$$
\sum_{j=1}^{n}\rho\log(\frac{\rho}{\hat{\rho}^j})+(1-\rho)\log(\frac{1-\rho}{1-\hat{\rho}^j}).
$$
And we  will minimize the objective function
$$
\sum_{i=1}^{m}L(x^{(i)},z^{(i)})+[\rho\log(\frac{\rho}{\hat{\rho}^j})+(1-\rho)\log(\frac{1-\rho}{1-\hat{\rho}^j})]
$$
via backpropagation, where $L(\cdot , \cdot)$ is a loss function and $z^{(i)}$ is the output of sparse autoencoder when the input is $x^{(i)}$.

If we want to compress the information of the data set, we only need output of hidden units $y^{(i)}=\sigma\circ(x^{(i)})$, which maps the data in higher dimensional space to a low dimensional space
$$
\mathbb{R}^{p}\to\mathbb{R}^{n}\\
x\to \sigma\circ(Wx+b).
$$

* https://en.wikipedia.org/wiki/Autoencoder
* https://www.cs.toronto.edu/~hinton/science.pdf
* https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
* http://ufldl.stanford.edu/wiki/index.php/Stacked_Autoencoders
* https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2016-70.pdf

### t-SNE

The basic idea of t-SNE is that the similarity should preserve after dimension reduction.
It maps the data $x$ in high dimensional space $X\subset\mathbb{R}^{p}$ to a low dimensional space $Y\subset\mathbb{R}^{d}$. **t-SNE** is an extension of stochastic neighbor embedding.

**Stochastic Neighbor Embedding (SNE)** starts by converting the high-dimensional Euclidean distances between data points into conditional probabilities that represent similarities. The similarity of data point $x_j$ to data point $x_i$ is the conditional probability, $p_{j|i}$, that $x_i$ would pick $x_j$ as its neighbor if neighbors were picked in proportion to their probability density under a Gaussian centered at $x_i$.
Mathematically, the conditional probability $p_{j|i}$ is given by
$$
p_{j|i} = \frac{exp(-\|x_i-x_j\|^2/2\sigma_i^2)}{\sum_{k\ne 1} exp(-\|x_k-x_i\|^2/2\sigma_i^2)}.
$$

Because we are only interested in modeling pairwise similarities, we set the value of $p_{i|i}$ to zero. For the low-dimensional counterparts $y_i$ and $y_j$ of the high-dimensional data points $x_i$ and $x_j$, it is possible to compute a similar conditional probability, which we denote by $q_{j|i}$. we model the similarity of map point $y_j$ to map point $y_i$ by
$$q_{j|i}=\frac{exp(-\|y_i-y_j\|^2)}{\sum_{k\ne i}exp(-\|y_k-y_i\|^2)}.$$

SNE minimizes the sum of cross entropy over all data points using a gradient descent
method. The cost function $C$ is given by
$$C=\sum_{i}\sum_{j}p_{j|i}\log(\frac{p_{j|i}}{q_{j|i}}).$$

In the high-dimensional space, we convert distances into probabilities using a Gaussian distribution. In the low-dimensional map, we can use a probability distribution that has much heavier tails than a Gaussian to convert distances into probabilities.
In t-SNE, we employ a **Student t-distribution** with one degree of freedom (which is the same
as a Cauchy distribution) as the heavy-tailed distribution in the low-dimensional map. Using this
distribution, the joint probabilities $q_{i|j}$ are defined as
$$
q_{i|j} = \frac{(1+\|y_i-y_j\|^2)^{-1}}{\sum_{j\not= i} (1+\|y_i -  y_j\|^2)^{-1}}.
$$



* https://blog.paperspace.com/dimension-reduction-with-t-sne/
* https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/
* https://distill.pub/2016/misread-tsne
* https://www.wikiwand.com/en/T-distributed_stochastic_neighbor_embedding

***

* https://www.wikiwand.com/en/Nonlinear_dimensionality_reduction
* https://scikit-learn.org/stable/modules/decomposition.html#decompositions
* http://www.idm.pku.edu.cn/staff/wangyizhou/papers/GAE-CVPRwDeepVision2014.pdf
