
### Clustering

There are two different contexts in clustering, depending on how the entities to be clustered are organized.
In some cases one starts from an **internal representation** of each entity (typically an $M$-dimensional vector $x_i$ assigned to entity $i$) and
derives mutual dissimilarities or mutual similarities from the internal representation. In
this case one can derive prototypes (or centroids) for each cluster, for example by averaging the characteristics of the contained entities (the vectors).
In other cases only an **external representation** of dissimilarities is available and the resulting model is an
**undirected and weighted graph** of entities connected by edges. From <https://www.intelligent-optimization.org/LIONbook/>.[^13]
***
The external representation of dissimilarity will be discussed in **Graph Algorithm**.

* http://sklearn.apachecn.org/cn/0.19.0/modules/clustering.html
* https://www.wikiwand.com/en/Cluster_analysis
* https://www.toptal.com/machine-learning/clustering-algorithms

#### K-means cluster

K-means is also called  **Lloyd’s algorithm**.It is a prototype-based clustering method.

* Set the class number $k$ and initialize the centers of the classes $\{c_i, c_2, \dots, c_k\}$.
* Split the data set to $k$ classes by the shortest distance to the center:
   $$\arg\min_{i}\|x-c_i\|$$
*  Update the center of the class $i$:
   $$c_i\leftarrow \frac{\sum_{x}\mathbb{I}(x\in \text{Class i})x}{\sum_{x}\mathbb{I}(x\in \text{Class i})}$$
* Until the stop criterion is satisfied.
***
The indicator function $\mathbb{I}(x\in\text{Class i})$ is defined as
$$
\mathbb{I}(x\in\text{Class i})=
    \begin{cases}
    1, \text{if $x\in\text{Class i}$} \\
    0,  \text{otherwise}.
    \end{cases}
$$

* https://www.wikiwand.com/en/K-means_clustering

#### Hierarchical clustering

**Hierarchical clustering algorithms** are either top-down or bottom-up. Bottom-up algorithms treat each document as a singleton cluster at the outset and then successively merge (or agglomerate) pairs of clusters until all clusters have been merged into a single cluster that contains all documents.
This method is based on the choice of distance measure or metric.
The algorithm works as follows:

* Put each data point in its own cluster.
* Identify the closest two clusters and combine them into one cluster.
* Repeat the above step till all the data points are in a single cluster.

***
![](https://raw.githubusercontent.com/Hulalazz/hierarchical-clustering/master/Results/Centroid.png)


+ https://blog.csdn.net/qq_39388410/article/details/78240037
+ https://www.r-bloggers.com/hierarchical-clustering-in-r-2/
+ http://iss.ices.utexas.edu/?p=projects/galois/benchmarks/agglomerative_clustering
+ https://nlp.stanford.edu/IR-book/html/htmledition/hierarchical-agglomerative-clustering-1.html
+ https://www.wikiwand.com/en/Hierarchical_clustering
+ http://www.econ.upf.edu/~michael/stanford/maeb7.pdf
+ http://www.cs.princeton.edu/courses/archive/spr08/cos424/slides/clustering-2.pdf


#### DBSCAN

Density-based spatial clustering of applications with noise (DBSCAN) is a data clustering algorithm proposed by Martin Ester, Hans-Peter Kriegel, Jörg Sander and Xiaowei Xu in 1996.


* All points within the cluster are mutually density-connected.
* If a point is density-reachable from any point of the cluster, it is part of the cluster as well.


* https://www.wikiwand.com/en/DBSCAN

### Classification

Classification is a basic task in machine learning even in artificial intelligence.
It is supervised with discrete or categorical desired labels.  

* https://en.wikipedia.org/wiki/Category:Classification_algorithms
* https://people.eecs.berkeley.edu/~jordan/classification.html

#### k-Nearest Neighbors

K-NN is simple to implement for high  dimensional data.
It is based on the assumption that the nearest neighbors are similar.

* Find the k-nearest neighbors $\{(p_1,d_0), \dots, (p_k,d_k)\}$ of the a given point $P_0$;
* Vote  the label of the point $P_0$: $d_0=\max{d}$, where $d$ is the number of the desired label.
  
It is distance-based.  

* https://www.wikiwand.com/en/K-nearest_neighbors_algorithm
* http://www.scholarpedia.org/article/K-nearest_neighbor

### Support Vector Machine

Support vector machine  initially is a classifier for the linearly separate data set.
It can extend to kernel methods.

* http://www.svms.org/history.html
* http://www.svms.org/
* http://web.stanford.edu/~hastie/TALKS/svm.pdf
* http://bytesizebio.net/2014/02/05/support-vector-machines-explained-well/

![](https://rescdn.mdpi.cn/entropy/entropy-15-00416/article_deploy/html/images/entropy-15-00416-g001.png)

### Kernel Methods

The kernel methods are to enhance the regression or classification methods via  better nonlinear representation.
It may be regarded as a counterpart of dimension reduction because the kernel methods usually map the low dimensional space into high dimensional space.

+ [Kernel method at Wikipedia](https://www.wikiwand.com/en/Kernel_method);
+ http://www.kernel-machines.org/
+ http://onlineprediction.net/?n=Main.KernelMethods
+ https://arxiv.org/pdf/math/0701907.pdf
+ https://www.ics.uci.edu/~welling/teaching/KernelsICS273B/svmintro.pdf
+ https://people.eecs.berkeley.edu/~jordan/kernels.html

***
![](https://courses.cs.ut.ee/2011/graphmining/uploads/Main/pm.png)
