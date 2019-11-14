## Dimension Reduction

<img title="https://fineartamerica.com" src="https://images.fineartamerica.com/images/artworkimages/mediumlarge/1/vital-statistics-ian-duncan-anderson.jpg" width="80%"  />

* [CSE254, Intrinsic Dimension 2019](https://yoavfreund.miraheze.org/wiki/CSE254,_Intrinsic_Dimension_2019)
* [Geometric Computation group in the Computer Science Department of Stanford University.](https://geometry.stanford.edu/member/guibas/)
* [CSIC 5011: Topological and Geometric Data Reduction and Visualization
Fall 2019](https://yao-lab.github.io/2019_csic5011/)

Principal component analysis or singular value decomposition can be applied to matrix approximation.
The data collected in practice always save in the table form, which can considered as a matrix. Another techniques similar to PCA is  eigenvalue-eigenvector decomposition. Dimension reduction is really the topic of data science as data preprocessing .

The basic idea of dimension reduction is that not all information is necessary for a specific task.
The motivation of dimension reduction is **Curse of Dimensionality**, limit of storage/computation and so on. The high dimensional space is not easy for us to visualize, imagine or understand. The intuition or insight to high dimensional space is weak for us, the people who live in the three dimensional space.
As a preprocessing data method, it helps us to select features and learn proper representation.  
The dimension reduction is related with geometry of data set, which includes manifold learning and topological data analysis.

All manifold learning algorithms assume that data set lies on a smooth non-linear manifold of low dimension and a mapping

$$
f:\mathbb{R}^D\to\mathbb{R}^d
$$

(where $D\gg d$) can be found by preserving one or more properties of the higher dimension space. For example, dimension reduction seems to be quasi-isometric:
$$
\|f(x)\|\approx \| x\|, \|f(x) -f(y)\|\approx \| x - y\|.
$$

$\color{aqua}{PS:}$ the dimension reduction is classified into unsupervised learning while it can be converted to optimization problems.
Additionally, it will miss some properties of the data set so please do not delete the previous data sets.

What is more, [the __blessings of dimensionality__ include the concentration of measure phenomenon (so-called in the geometry of Banach spaces), which means that certain random fluctuations are very well controlled in high dimensions and the success of asymptotic methods, used widely in mathematical statistics and statistical physics, which suggest that statements about very high-dimensional settings may be made where moderate dimensions would be too complicated.](https://www.math.ucdavis.edu/~strohmer/courses/180BigData/180lecture1.pdf)

It is a wonderful review of dimension reduction at [TiCC TR 2009–005, Dimensionality Reduction: A Comparative Review, Laurens van der Maaten Eric Postma, Jaap van den Herik TiCC, Tilburg University](https://lvdmaaten.github.io/publications/papers/TR_Dimensionality_Reduction_Review_2009.pdf).

A related top is data compression, a branch of information theory, a more  useful and fundamental topic in computer science.

* https://lvdmaaten.github.io/software/
* https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
* http://www.lcayton.com/resexam.pdf
* https://www.wikiwand.com/en/Data_compression
* https://brilliant.org/wiki/compression/
* https://www.wikiwand.com/en/Curse_of_dimensionality
* http://www.cnblogs.com/xbinworld/archive/2012/07/09/LLE.html
* https://www.ayasdi.com/blog/artificial-intelligence/prediction-needs-unsupervised-learning/
* [Nonlinear Dimensionality Reduction](https://cs.nyu.edu/~roweis/lle/related.html)
* [COMPSCI 630: Randomized Algorithms](https://www2.cs.duke.edu/courses/spring18/compsci630/lecture10.pdf)
* [Dimensionality reduction](http://faculty.ucmerced.edu/mcarreira-perpinan/papers/phd-ch04.pdf)
* [Surprises in high dimensions](https://www.math.ucdavis.edu/~strohmer/courses/180BigData/180lecture1.pdf)
* [Linear dimension reduction via Johnson-Lindenstrauss](https://www.math.ucdavis.edu/~strohmer/courses/180BigData/180lecture_jl.pdf)
* [manifold learning with applications to object recognition](https://people.eecs.berkeley.edu/~efros/courses/AP06/presentations/ThompsonDimensionalityReduction.pdf)
* http://bactra.org/notebooks/manifold-learning.html
* http://www.cs.columbia.edu/~jebara/papers/BlakeShawFinalThesis.pdf
* https://www.zhihu.com/question/41106133

### PCA and MDS

The data in table form can be regarded as matrix in mathematics. And we can apply singular value decomposition  to low rank approximation or non-negative matrix factorization, which we will talk in *PCA and SVD*.
It is classified as linear techniques.
And it can extend to kernel PCA and [generalized PCA](https://www.asc.ohio-state.edu/lee.2272/mss/tr892.pdf).

**Multi-Dimension Scaling** is a distance-preserving manifold learning method. Distance preserving methods assume that a manifold can be defined by the pairwise distances of its points.
In distance preserving methods, a low dimensional embedding is obtained from the higher dimension in such a way that pairwise distances between the points remain same. Some distance preserving methods preserve spatial distances (MDS) while some preserve graph distances.

*MDS* is not a single method but a family of methods. MDS takes a dissimilarity or distance matrix ${D}$ where $D_{ij}$ represents the dissimilarity between points ${i}$ and ${j}$ and produces a mapping on a lower dimension, preserving the dissimilarities as closely as possible.
The dissimilarity matrix could be observed or calculated from the given data set. MDS has been widely popular and developed in the field of human sciences like sociology, anthropology and especially in psychometrics.
[From blog.paperspace.com](https://blog.paperspace.com/dimension-reduction-with-multi-dimension-scaling/).

It is a linear map
$$
{X}\in\mathbb{R}^D\to {Z}\in\mathbb{R}^d\\
Z = W^T X
$$

Steps of a Classical MDS algorithm:

Classical MDS uses the fact that the coordinate matrix can be derived by eigenvalue decomposition from ${\textstyle B= Z Z^T}$. And the matrix ${\textstyle B}$ can be computed from proximity matrix ${\textstyle D}$ by using double centering.

+ Set up the squared proximity matrix ${\textstyle D^{(2)}=[d_{ij}^{2}]}$
+ Apply double centering: $B=-{\frac{1}{2}J D^{(2)}J}$ using the centering matrix ${\textstyle J=I-{\frac {1}{n}11'}}$, where ${\textstyle n}$ is the number of objects.
+ Determine the ${\textstyle m}$ largest eigenvalues $\lambda_{1},\lambda_{2},...,\lambda_{m}$ and corresponding eigenvectors ${\textstyle e_{1},e_{2},...,e_{m}}$ of ${\textstyle B}$ (where ${\textstyle m}$ is the number of dimensions desired for the output).
+ Now, ${\textstyle Z=E_{m}\Lambda_{m}^{1/2}}$ , where ${\textstyle E_{m}}$ is the matrix of ${\textstyle m}$ eigenvectors and ${\textstyle \Lambda_{m}}$ is the diagonal matrix of ${\textstyle m}$ eigenvalues of ${\textstyle B}$.

Classical MDS assumes Euclidean distances. So this is not applicable for direct dissimilarity ratings.

* http://www.math.pku.edu.cn/teachers/yaoy/reference/book05.pdf
* http://www.statsoft.com/textbook/multidimensional-scaling
* https://www.springer.com/in/book/9780387251509
* https://www.stat.pitt.edu/sungkyu/course/2221Fall13/lec8_mds_combined.pdf
* https://www.stat.pitt.edu/sungkyu/course/2221Fall13/lec4_pca_slides.pdf
* https://www.ibm.com/support/knowledgecenter/en/SSLVMB_22.0.0/
* https://www.wikiwand.com/en/Multidimensional_scaling


### Locally Linear Embedding

**Locally Linear Embedding(LLE)** is a topology preserving manifold learning method. Topology preservation means the neighborhood structure is intact. Methods like SOM(self-organizing map) are also topology preserving but they assume a predefined lattice for the lower manifold. LLE creates the lattice based on the information contained in the dataset.

<img src = https://s3-us-west-2.amazonaws.com/articles-dimred/lle/lle_main.png width = 50%/>
<img src = https://cs.nyu.edu/~roweis/lle/images/llef2med.gif width = 50%/>

***

1. Compute the neighbors of each data point, $\vec{X}_i$.
2. Compute the weights $W_{ij}$ that best reconstruct each data point $\vec{x_i}$ from its neighbors, minimizing the cost
$\sum_{i}|\vec{X}_i - \sum_{j}W_{ij}\vec{X}_j|^2$ by constrained linear fits.
3. Compute the vectors $\vec{Y}_i$ best reconstructed by the weights $W_{ij}$, minimizing the quadratic form $\sum_{i}|\vec{Y}_i - \sum_{j}W_{ij}\vec{Y}_j|^2$ by its bottom nonzero eigenvectors.

***

* https://cs.nyu.edu/~roweis/lle/
* http://www.robots.ox.ac.uk/~az/lectures/ml/lle.pdf
* http://ai.stanford.edu/~schuon/learning/inclle.pdf
* https://blog.paperspace.com/dimension-reduction-with-lle/

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

An auto-encoder neural network is an unsupervised learning algorithm
that applies back-propagation, setting the target values to be equal to the inputs. I.e., it uses $z^{(i)} = x^{(i)}$. It
is trying to learn an approximation to the identity function, so as to output
$\hat{x}$ that is similar to $x$. The identity function seems a particularly trivial
function to be trying to learn; but by placing constraints on the network,
such as by limiting the number of hidden units, we can discover interesting structure about the data.

Recall that $y^{(j)}$ denotes the activation of hidden unit $j$ in the auto-encoder and write $y^{(j)}(x)$ to denote the activation of this hidden unit when the network is given a specific input $x$, i.e., $y^{(j)}(x)=\sigma\circ(x)^{(j)}$.
Let $\hat{\rho}_j=\frac{1}{m}\sum_{i=1}^{m}y^{j}(x^{(i)})$
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
x\to \sigma\circ(W_1 x+b_1).
$$
Given an compressed data $\{y^{(i)}\}$ via a autoencoder, we can decode it by the output layer $z^{(i)}=\sigma\circ(W_2 y^{(i)} + b_2)$.

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

* https://lvdmaaten.github.io/tsne/
* https://blog.paperspace.com/dimension-reduction-with-t-sne/
* https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/
* https://distill.pub/2016/misread-tsne
* https://www.wikiwand.com/en/T-distributed_stochastic_neighbor_embedding

### ICA

Independent component analysis is to find the latent independent factors that make up the observation, i.e,
$$
x_j = b_{j1}s_1 + b_{j2}s_2 + \dots + b_{jn} s_n,\forall j\in\{1,2,\dots,n\} \\
x=Bs
$$

where $x$, $s$, $B$ is the observation, latent variable and unknown mixing matrix.
The ICA model is a generative model, which means that it describes how the observed data are generated by a process of mixing the components $s_i$.
The independent components are latent variables, meaning that they cannot be directly observed.
Also the mixing matrix $B$ is assumed to be unknown. All we observe is the random vector ${x}$, and we must estimate
both ${B}$ and ${s}$ using it. This must be done under as general assumptions as possible.

In the ICA model, we assume that each mixture $x_j$ as well as each independent component $s_k$ is a random variable.
Without loss of generality, we can assume that both the mixture variables and the independent components have **zero mean**.

Different from the principal components in PCA, the independent  components are not required to be perpendicular to each other.

The starting point for ICA is the very simple assumption that the components $s_i$ are statistically independent.
The fundamental restriction in ICA is that the independent components must be *nongaussian* for ICA to be possible.
And if the mixing matrix  ${B}$ is inversible so that $W=B^{-1}$ and $s=Wx$.

<img title="ICA vs PCA" src="https://ars.els-cdn.com/content/image/1-s2.0-S0957417406001308-gr2.jpg" width = "68%" />

In order to solve $x=Bs$, we assume that each independent component $s_i$ has unit variance: $\mathbb{E}(s_i^2)=1$.
The independence of random variables is not obvious when we do not know their probability distribution. Since independence implies uncorrelatedness, many ICA methods constrain the estimation procedure
so that it always gives uncorrelated estimates of the independent components. This reduces the number of free parameters, and simplifies the problem.

Intuitively speaking, the key to estimating the ICA model is `nongaussianity`. Actually, without `nongaussianity` the
estimation is not possible at all.

The classical measure of `nongaussianity` is kurtosis or the fourth-order cumulant.
The kurtosis of random variable ${y}$ is classically defined by
$$kurt(y)=\mathbb{E}(y^4)-3\{\mathbb{E}(y^2)\}^2.$$
A second very important measure of `nongaussianity` is given by negentropy ${J}$, which is defined as
$$J=H(y_{gauss})-H(y)$$
where $y_{gauss}$ is a Gaussian random variable of the same covariance matrix as ${y}$ and $H(\cdot)$ is the **entropy** function.
Estimating negentropy using the definition would require an estimate (possibly nonparametric) of
the pdf. Therefore, simpler approximations of negentropy are very useful such as
$$J(y)\approx \frac{1}{12}\{\mathbb{E}(y^3)\}^2+\frac{1}{48}kurt(y)^2.$$
Denoting ${B}^{-1}$ by the matrix $W=(w_1,w_2,\dots,w_n)^T$ and $y=Wx$, the log-likelihood takes the form
$$L=\sum_{t=1}^{T} \sum_{i=1}^{n}\log f_i(w_i^Tx(t))+T\log(|det(W)|)$$
where the $f_i$ are the density functions of the $s_i$ (here assumed to be known),
and the $x(t),t = 1,...,T$ are the realizations of $x$.
And we constrain the $y_i(t)=w_i^Tx(t)$ to be **uncorrelated and of unit variance**.

It is proved the surprising result that the principle of network entropy
maximization, or “infomax”, is equivalent to maximum likelihood estimation.

ICA is very closely related to the method called blind source separation (BSS) or blind signal separation and see more on
[Blind Identification of Potentially Underdetermined Mixtures](https://perso.univ-rennes1.fr/laurent.albera/alberasiteweb/pdf/Albe03-PHD.pdf).

***

+ https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf
+ https://research.ics.aalto.fi/ica/
+ http://compneurosci.com/wiki/images/4/42/Intro_to_PCA_and_ICA.pdf
+ https://www.wikiwand.com/en/Independent_component_analysis
+ http://www.gipsa-lab.grenoble-inp.fr/~pierre.comon/FichiersPdf/polyD16-2006.pdf
+ http://arnauddelorme.com/ica_for_dummies/
+ https://sccn.ucsd.edu/wiki/Chapter_09:_Decomposing_Data_Using_ICA
+ http://cs229.stanford.edu/notes/cs229-notes11.pdf
+ [Diving Deeper into Dimension Reduction with Independent Components Analysis (ICA)](https://blog.paperspace.com/dimension-reduction-with-independent-components-analysis/)
+ https://www.zhihu.com/question/28845451/answer/42292804
+ http://www.gipsa-lab.grenoble-inp.fr/~pierre.comon/publications_en.html#book
+ https://www.stat.pitt.edu/sungkyu/course/2221Fall13/lec6_FA_ICA.pdf
+ https://perso.univ-rennes1.fr/laurent.albera/alberasiteweb/pdf/Albe03-PHD.pdf
+ http://fourier.eng.hmc.edu/e161/lectures/ica/index.html
+ http://cis.legacy.ics.tkk.fi/aapo/papers/IJCNN99_tutorialweb/
+ https://www.cs.helsinki.fi/u/ahyvarin/whatisica.shtml


### Exploratory Projection Pursuit

Projection pursuit is a classical exploratory data analysis method to detect interesting low-dimensional structures in multivariate data.
Originally, projection pursuit was applied mostly to data of moderately low dimension.
Peter J Huber said in [Projection Pursuit](https://projecteuclid.org/euclid.aos/1176349519):
> The most exciting features of projection pursuit is that it is one of few multivariate methods able to bypass the "curse of dimensionality" caused by the fact that high-dimensional space is mostly empty.

> PP methods have one serious drawback: their high demand on computer time.


[Discussion by Friedman](https://projecteuclid.org/euclid.aos/1176349535) put that
> Projection pursuit methods are by no means the only ones to be originally ignored for lack of theoretical justification. Factor analysis, clustering, multidimensional scaling, recursive partitioning, correspondence analysis, soft modeling (partial-least-squares), represent methods that were in common sense for many years before their theoretical underpinnings were well understood. Again, the principal justification for their use was that they made sense heuristically and seemed to work well in a wide variety of situations.

It reduces the data of dimension ${m}$ to dimension ${p}$  for visual inspection.
This method consists of defining a measure of information content in two dimension and optimizing that measure or index as
a function of two m-dimensional projection vectors to find the most informative
projection.

The minimal ingredients of an EPP algorithm are then as follows:

> - (1) choose a subspace of the desired dimension and project the data onto the subspace,
> - (2) compute some index of 'information content' for the projection,
> - and (3) iterate 1 and 2 until the index is maximized.

In more mathematical language, EPP involves choosing some
dimension ${p}$ for the projection subspace, defining an index of interestingness for random variables in
dimension ${p}$, say $f(X(\alpha), \alpha)$, and devising some method to optimize ${f}$ over all possible projection unit vectors $\alpha, \|\alpha\|=1$.
Note that in a subspace, the random variable is always directly dependent upon the projection given by ${\alpha}$ but
for convenience the notation is sometimes suppressed.

Assume that the $p$ -dimensional random variable $X$ is sphered and centered, that is, $E(X)=0$ and $Var(X) = {\cal{I}}_p$. This will remove the effect of location, scale, and correlation structure.


Friedman and Tukey (1974) proposed to investigate the high-dimensional distribution of ${X}$ by considering the index

$$
\frac{1}{n}\sum_{i=1}^{n}\hat{f} (\left<\alpha, X_i\right>)
$$

and $\hat{f}(z) = \frac{1}{n}\sum_{j=1}^{n} K (z -\left<\alpha, X_j\right>)$ with some kernel function ${K}$.
If the high-dimensional distribution of $X$ is normal, then each projection $z=\alpha^{\top}X$ is standard normal since $\vert\vert\alpha\vert\vert=1$ and since  ${X}$ has been centered and sphered by, e.g., the [Mahalanobis transformation](https://www.wikiwand.com/en/Whitening_transformation).

The projection pursuit methods can extend to density estimation and regression.

****

* https://projecteuclid.org/euclid.aos/1176349519
* https://projecteuclid.org/euclid.aos/1176349520
* https://projecteuclid.org/euclid.aos/1176349535
* [guynason, Professor of Statistics, University of Bristol](https://people.maths.bris.ac.uk/~magpn/Research/PP/PP.html)
* http://sci-hub.fun/10.1002/wics.23
* [Werner Stuetzle, Department of Statistics, University of Washington](https://www.stat.washington.edu/wxs/Visualization-papers/projection-pursuit.pdf)
* [ICA and Projection Pursuit](http://cis.legacy.ics.tkk.fi/aapo/papers/IJCNN99_tutorialweb/node23.html)
* https://www.pnas.org/content/115/37/9151
* [Projection pursuit in high dimensions](https://www.ncbi.nlm.nih.gov/pubmed/30150379)
* https://github.com/pavelkomarov/projection-pursuit
* [Interactive Projection Pursuit (IPP) by Jiayang Sun, Jeremy Fleischer, Catherine Loader](http://sun.cwru.edu/~jiayang/nsf/ipp.html)
* [Exploratory Projection Pursuit](https://rd.springer.com/chapter/10.1007/978-1-4612-4214-7_9)
* [A Projection Pursuit framework for supervised dimension reduction of high dimensional small sample datasets](https://www.sciencedirect.com/science/article/pii/S0925231214010091)

###  Self Organizing Maps

One source of `ICA` is "general infomax learning principle" well known in machine learning or signal processing community.

However, we can not explain all efficient methods in mathematics or statistics then write it in the textbook.
`Self organizing map` is not well-known. These networks are based
on competitive learning; the output neurons of the network compete among themselves to
be activated or fired, with the result that only one output neuron, or one neuron per group.

Each node has a specific topological position (an x, y coordinate in the lattice) and contains a vector of weights of the same dimension as the input vectors. That is to say, if the training data consists of vectors, V,  of n dimensions:

$$V_1, V_2, V_3, \cdots, V_n.$$

Then each node will contain a corresponding weight vector ${W}$, of ${n}$ dimensions:

$$W_1, W_2, W_3, \cdots, W_n.$$

[Training occurs in several steps and over many iterations](http://www.pitt.edu/~is2470pb/Spring05/FinalProjects/Group1a/tutorial/som.html):

- Each node's weights are initialized.
- A vector is chosen at random from the set of training data and presented to the lattice.
- Every node is examined to calculate which one's weights are most like the input vector. The winning node is commonly known as the `Best Matching Unit (BMU)`.
- The radius of the neighbourhood of the BMU is now calculated. This is a value that starts large, typically set to the 'radius' of the lattice,  but diminishes each time-step. Any nodes found within this radius are deemed to be inside the BMU's neighbourhood.
- Each neighbouring node's (the nodes found in step 4) weights are adjusted to make them more like the input vector. The closer a node is to the BMU, the more its weights get altered.
- Repeat step 2 for N iterations.


***
<img title ="Kohonen" src="http://www.lohninger.com/helpcsuite/img/kohonen1.gif" width=80% />

* [Kohonen Network - Background Information](http://www.lohninger.com/helpcsuite/kohonen_network_-_background_information.htm)
* https://users.ics.aalto.fi/teuvo/
* http://www.ai-junkie.com/ann/som/som1.html
* http://www.mlab.uiah.fi/~timo/som/thesis-som.html
* [Self-Organizing Maps](http://www.pitt.edu/~is2470pb/Spring05/FinalProjects/Group1a/tutorial/som.html
)

### Diffusion map

Diffusion map uses the eigen-vectors to define coordinate  system.

<img src = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Diffusion_map_of_a_torodial_helix.jpg/640px-Diffusion_map_of_a_torodial_helix.jpg" width = "40%" />

https://www.wikiwand.com/en/Diffusion_map

* https://www.wikiwand.com/en/Nonlinear_dimensionality_reduction
* [destiny:An R package for diffusion maps, with additional features for large-scale and single cell data](https://theislab.github.io/destiny/index.html)
* [A short introduction to Diffusion Maps](https://stephanosterburg.github.io/an_introductio_to_diffusion_maps)
* [pydiffmap: an open-source project to develop a robust and accessible diffusion map code for public use.](https://pydiffmap.readthedocs.io/en/master/index.html)
* [MAT 585: Diffusion Maps by Amit Singer](https://www.math.ucdavis.edu/~strohmer/courses/180BigData/Singer_diffusionmaps.pdf)

### Uniform Manifold Approximation and Projection 

`Uniform Manifold Approximation and Projection (UMAP)` is a dimension reduction technique that can be used for visualisation similarly to t-SNE, but also for general non-linear dimension reduction. The algorithm is founded on three assumptions about the data

+ The data is uniformly distributed on Riemannian manifold;
+ The Riemannian metric is locally constant (or can be approximated as such);
+ The manifold is locally connected.

[From these assumptions it is possible to model the manifold with a fuzzy topological structure. The embedding is found by searching for a low dimensional projection of the data that has the closest possible equivalent fuzzy topological structure.](https://umap-learn.readthedocs.io/en/latest/)

- https://github.com/lmcinnes/umap
- https://umap-learn.readthedocs.io/en/latest/
- https://arxiv.org/abs/1802.03426

### Intrinsic Dimension

In [Description Of Intrinsic Dimension 2019](https://yoavfreund.miraheze.org/wiki/Description_Of_Intrinsic_Dimension_2019), Yoav Freund pointed out that:
> It is often the case that very high dimensional data, such as images, can be compressed into low dimensional vectors with small reconstruction error. The dimension of these vectors is the **`intrinsic dimension`** of the data. We will discuss several techniques for estimating intrinsic dimension and for mapping data vectors to their low-dimensional representations. The ultimate goal is to find streaming algorithms can can process very large dataset in linear or sub-linear time.

Methods for identifying the dimension：

* Haussdorff dimension, Doubling dimension, epsilon-cover


## Metric Learning

### Deep Metric Learning

The goal is to capture similarity between embeddings, such that the  projected distance of similar items in the embedding space is smaller  than the dissimilar items.
Compared to the standard distance metric learning, it uses deep neural  networks to learn a nonlinear mapping to the embedding space.
It helps with extreme classification settings with huge number classes, not many examples per class.

#### Siamese Networks

* Left and right legs of the network  have identical structures (siamese);
* Weights are shared between the siamese networks during training;
* Networks are optimized with a  loss function, such as contrastive loss.

- [ ] [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- https://en.wikipedia.org/wiki/Siamese_network

https://people.eecs.berkeley.edu/~efros/courses/AP06/