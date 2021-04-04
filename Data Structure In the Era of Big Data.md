- [Data Structure In the Era of Big Data](#data-structure-in-the-era-of-big-data)
  - [Vector Search Engine](#vector-search-engine)
  - [Probabilistic data structures](#probabilistic-data-structures)
    - [Hashing](#hashing)
      - [Fowler-Noll-Vo Hash](#fowler-noll-vo-hash)
      - [MurmurHash](#murmurhash)
      - [Locality-Sensitive Hashing](#locality-sensitive-hashing)
      - [MinHash](#minhash)
      - [Ranking Preserving Hashing](#ranking-preserving-hashing)
      - [Cuckoo Hashing](#cuckoo-hashing)
      - [Consistent Hashing](#consistent-hashing)
      - [Random Projection and Hashing](#random-projection-and-hashing)
      - [LSH-Sampling](#lsh-sampling)
    - [Bio-Inspired Hashing](#bio-inspired-hashing)
      - [FlyHash](#flyhash)
    - [Bloom filters](#bloom-filters)
      - [Counting Bloom filters](#counting-bloom-filters)
      - [Bloomier Filter](#bloomier-filter)
      - [Blocked Bloom filters](#blocked-bloom-filters)
      - [Multi-set filters](#multi-set-filters)
      - [Quotient Filter](#quotient-filter)
    - [HyperLogLog](#hyperloglog)
    - [Skip Lists](#skip-lists)
    - [Count-min Sketch](#count-min-sketch)
  - [Approximate Nearest Neighbor Searching and Polytope Approximation](#approximate-nearest-neighbor-searching-and-polytope-approximation)
    - [Proximity graphs](#proximity-graphs)
    - [Navigable Small World Graph](#navigable-small-world-graph)
    - [Hierarchical Navigable Small World (HNSW) Graph](#hierarchical-navigable-small-world-hnsw-graph)
    - [KNN graph](#knn-graph)
  - [Compact Data Structures](#compact-data-structures)
  - [Succinct data structures](#succinct-data-structures)
  - [Learned Data Structure](#learned-data-structure)
    - [Learning to Hash](#learning-to-hash)
      - [Boosted LSH](#boosted-lsh)

# Data Structure In the Era of Big Data


- https://dataconomy.com/2017/04/big-data-101-data-structures/
- http://web.stanford.edu/class/cs168/
- https://www.ics.uci.edu/~pattis/ICS-23/
- https://redislabs.com/redis-enterprise/data-structures/
- [Efficient Large Scale Maximum Inner Product Search](http://www.cse.cuhk.edu.hk/systems/hash/gqr/report/bob-slide.pdf)
- https://www.epaperpress.com/vbhash/index.html
- [XSearch: Distributed Indexing and Search in Large-Scale File Systems](http://datasys.cs.iit.edu/projects/xsearch/index.html)
- [Data Structures for Big Data ](http://people.duke.edu/~ccc14/sta-663-2016/A04_Big_Data_Structures.html)
- https://cs.uwaterloo.ca/~imunro/cs840/CS840.html
- https://graphics.stanford.edu/courses/cs468-06-fall/
- [CS369G: Algorithmic Techniques for Big Data](http://web.stanford.edu/class/cs369g/)
- http://www.jblumenstock.com/teaching/course=info251

## Vector Search Engine

Vector search engine (aka neural search engine or deep search engine) [uses deep learning models to encode data sets into meaningful vector representations, where distance between vectors represent the similarities between items.](https://www.microsoft.com/en-us/ai/ai-lab-vector-search)


- https://www.microsoft.com/en-us/ai/ai-lab-vector-search
- https://github.com/textkernel/vector-search-plugin
- https://github.com/pingcap/awesome-database-learning

## Probabilistic data structures

[`Probabilistic data structures` is a common name for data structures based mostly on different `hashing` techniques. Unlike regular (or deterministic) data structures, they always provide approximated answers but with reliable ways to estimate possible errors. Fortunately, the potential losses or errors are fully compensated for by extremely low memory requirements, constant query time, and scaling, three factors that become important in Big Data applications.](https://pdsa.gakhov.com/)

- http://ekzhu.com/datasketch/index.html
- https://github.com/gakhov/pdsa
- https://pdsa.readthedocs.io/en/latest/
- https://iq.opengenus.org/probabilistic-data-structures/
- [Probabilistic Data Structures and Algorithms for Big Data Applications](https://pdsa.gakhov.com/)
- [PROBABILISTIC HASHING TECHNIQUES FOR BIG DATA](https://www.cs.rice.edu/~as143/Doc/Anshumali_Shrivastava.pdf)
- [COMP 480/580 Probabilistic Algorithms and Data Structures](https://www.cs.rice.edu/~as143/COMP480_580_Spring21/index.html)
- [COMS 4995: Randomized Algorithms](http://timroughgarden.org/f19/f19.html)

### Hashing 

To quote the [hash function](https://en.wikipedia.org/wiki/Hash_function) at Wikipedia:
> A hash function is any function that can be used to map data of arbitrary size to fixed-size values. The values returned by a hash function are called hash values, hash codes, digests, or simply hashes. The values are used to index a fixed-size table called a hash table. Use of a hash function to index a hash table is called hashing or scatter storage addressing.

Hashed indexes use a hashing function to compute the hash of the value of the index field. 
The hashing function collapses embedded documents and computes the hash for the entire value but does not support multi-key (i.e. arrays) indexes.

Real life data tends to get corrupted because machines (and humans) are never as reliable as we wish for. One efficient way is make sure your data wasn't unintendedly modified is to generate some kind of hash. [That hash shall be unique, compact and efficient:](https://create.stephan-brumme.com/crc32/)
* unique: any kind of modification to the data shall generate a different hash
* compact: as few bits or bytes as possible to keep the overhead low
* efficient: use little computing resources, i.e. fast and low memory usage

[A hashing model takes an input data-point e.g. an image or document, and outputs a sequence of bits (hash code) representing that data-point.](https://learning2hash.github.io/base-taxonomy/)
> Hashing models can be broadly categorized into two different categories: `quantization` and `projection`. The projection models focus on learning a low-dimensional transformation of the input data in a way that encourages related data-points to be closer together in the new space. In contrast, the quantization models seek to convert those projections into binary by using a thresholding mechanism. The projection branch can be further divided into data-independent, data-dependent (unsupervised) and data-dependent (supervised) depending on whether the projections are influenced by the distribution of the data or available class-labels.

- [Hash function](https://www.jianshu.com/p/bba9b61b80e7)
- https://github.com/caoyue10/DeepHash-Papers
- https://zhuanlan.zhihu.com/p/43569947
- https://www.tutorialspoint.com/dbms/dbms_hashing.htm
- [Indexing based on Hashing](http://www.mathcs.emory.edu/~cheung/Courses/554/Syllabus/3-index/hashing.html)
- https://docs.mongodb.com/manual/core/index-hashed/
- https://www.cs.cmu.edu/~adamchik/15-121/lectures/Hashing/hashing.html
- https://www2.cs.sfu.ca/CourseCentral/354/zaiane/material/notes/Chapter11/node15.html
- https://github.com/Pfzuo/Level-Hashing
- https://thehive.ai/insights/learning-hash-codes-via-hamming-distance-targets
- [Various hashing methods for image retrieval and serves as the baselines](https://github.com/willard-yuan/hashing-baseline-for-image-retrieval)
- http://papers.nips.cc/paper/5893-practical-and-optimal-lsh-for-angular-distance
- [Hash functions: An empirical comparison](https://www.strchr.com/hash_functions)

#### Fowler-Noll-Vo Hash


The basis of the FNV hash algorithm was taken from an idea sent as reviewer comments to the IEEE POSIX P1003.2 committee by Glenn Fowler and Phong Vo back in 1991. In a subsequent ballot round: Landon Curt Noll improved on their algorithm. Some people tried this hash and found that it worked rather well. In an EMail message to Landon, they named it the ``Fowler/Noll/Vo'' or FNV hash.

FNV hashes are designed to be fast while maintaining a low collision rate. The FNV speed allows one to quickly hash lots of data while maintaining a reasonable collision rate. The high dispersion of the FNV hashes makes them well suited for hashing nearly identical strings such as URLs, hostnames, filenames, text, IP addresses, etc.

 - http://www.isthe.com/chongo/tech/comp/fnv/
 - https://create.stephan-brumme.com/fnv-hash/


#### MurmurHash 

[MurmurHash is a `non-cryptographic hash` function suitable for general hash-based lookup. The name comes from two basic operations, multiply (MU) and rotate (R), used in its inner loop. Unlike cryptographic hash functions, it is not specifically designed to be difficult to reverse by an adversary, making it unsuitable for cryptographic purposes.](https://commons.apache.org/proper/commons-codec/apidocs/org/apache/commons/codec/digest/MurmurHash3.html)

- https://sites.google.com/site/murmurhash/
- https://github.com/aappleby/smhasher

#### Locality-Sensitive Hashing

Locality-Sensitive Hashing (LSH) is a class of methods for the nearest neighbor search problem, which is defined as follows: given a dataset of points in a metric space (e.g., Rd with the Euclidean distance), our goal is to preprocess the data set so that we can quickly answer nearest neighbor queries: given a previously unseen query point, we want to find one or several points in our dataset that are closest to the query point. 


- http://web.mit.edu/andoni/www/LSH/index.html
- http://yongyuan.name/blog/vector-ann-search.html
- https://github.com/arbabenko/GNOIMI
- https://github.com/willard-yuan/hashing-baseline-for-image-retrieval
- http://yongyuan.name/habir/
- [4 Pictures that Explain LSH - Locality Sensitive Hashing Tutorial](https://randorithms.com/2019/09/19/Visual-LSH.html)
- https://eng.uber.com/lsh/

#### MinHash

[MinHash was originally an algorithm to quickly estimate the Jaccard similarity between two sets but can be designed as a data structure that revolves around the algorithm. This is a probabilistic data structure that quickly estimates how similar two sets are.](https://iq.opengenus.org/minhash/)

<img src="https://iq.opengenus.org/content/images/2020/09/Example-of-minhash-signatures.jpg" width="80%" />

- [MinHash Tutorial with Python Code](https://mccormickml.com/2015/06/12/minhash-tutorial-with-python-code/)
- [MinHash Sketches: A Brief Survey](http://www.cohenwang.com/edith/Surveys/minhash.pdf)
- https://skeptric.com/minhash-lsh/
- [Asymmetric Minwise Hashing for Indexing Binary Inner Products and Set Containment](http://www.cs.cornell.edu/~anshu/papers/WWW2015.pdf)
- [GPU-based minwise hashing: GPU-based minwise hashing](https://dl.acm.org/doi/10.1145/2187980.2188129)

#### Ranking Preserving Hashing


[Rank Preserving Hashing (RPH) is to explicitly optimize the precision of Hamming distance ranking towards preserving the supervised rank information.](http://math.ucsd.edu/~dmeyer/research/publications/rankhash/rankhash.pdf) 

- [Ranking Preserving Hashing for Fast Similarity Search](https://www.ijcai.org/Proceedings/15/Papers/549.pdf)
- [Unsupervised Rank-Preserving Hashing for Large-Scale Image Retrieval](https://dl.acm.org/doi/pdf/10.1145/3323873.3325038)
- http://math.ucsd.edu/~dmeyer/research/publications/rankhash/rankhash.pdf
- [Order Preserving Hashing for Approximate Nearest Neighbor Search](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/ACMMM13-OrderPreservingHashing.pdf)
- https://songdj.github.io/

#### Cuckoo Hashing

> Cuckoo Hashing is a technique for resolving collisions in hash tables that produces a dictionary with constant-time worst-case lookup and deletion operations as well as amortized constant-time insertion operations.


- [An Overview of Cuckoo Hashing](https://cs.stanford.edu/~rishig/courses/ref/l13a.pdf)
- [Some Open Questions Related to Cuckoo Hashing](https://www.eecs.harvard.edu/~michaelm/postscripts/esa2009.pdf)
- [Practical Survey on Hash Tables](http://romania.amazon.com/techon/presentations/PracticalSurveyHashTables_AurelianTutuianu.pdf)
- [Elastic Cuckoo Page Tables: Rethinking Virtual Memory Translation for Parallelism](https://tianyin.github.io/pub/cuckoo_pt.pdf)
- [MinCounter: An Efficient Cuckoo Hashing Scheme for Cloud Storage Systems](https://www.storageconference.us/2015/Papers/19.Sun.pdf)
- [Bloom Filters, Cuckoo Hashing, Cuckoo Filters, Adaptive Cuckoo Filters and Learned Bloom Filters](http://research.baidu.com/Public/ueditor/upload/file/20180804/1533345837426670.pdf)


#### Consistent Hashing

[Consistent hashing is done to implement scalability into the storage system by dividing up the data among multiple storage servers.](https://nlogn.in/consistent-hashing-system-design/)

- https://courses.cs.washington.edu/courses/cse452/18sp/ConsistentHashing.pdf
- https://nlogn.in/consistent-hashing-system-design/
- https://www.ic.unicamp.br/~celio/peer2peer/structured-theory/consistent-hashing.pdf

#### Random Projection and Hashing

- https://www.cs.utah.edu/~jeffp/teaching/cs7931-S15/cs7931/3-rp.pdf
- [Random Projection and the Assembly Hypothesis](https://simons.berkeley.edu/sites/default/files/docs/14061/rpassemblies.pdf)
- [Random Projections and Sampling Algorithms for Clustering of High-Dimensional Polygonal Curves](https://arxiv.org/abs/1907.06969)
- https://www.dennisrohde.work/uploads/poster_neurips19.pdf
- https://www.dennisrohde.work/
- [Brain computation by assemblies of neurons](https://www.pnas.org/content/117/25/14464)
- https://github.com/wilseypa/rphash

#### LSH-Sampling

- https://www.cs.rice.edu/~bc20/
- https://www.cs.rice.edu/~as143/
- [Locality Sensitive Sampling for Extreme-Scale Optimization and Deep Learning](https://scholarship.rice.edu/bitstream/handle/1911/109187/CHEN-DOCUMENT-2020.pdf?sequence=1&isAllowed=y)
- http://mlwiki.org/index.php/Bit_Sampling_LSH
- [Mutual Information Estimation using LSH Sampling](https://www.cs.rice.edu/~as143/Papers/IJCAI_20.pdf)

### Bio-Inspired Hashing


The fruit fly Drosophila's olfactory circuit has inspired a new locality sensitive hashing (LSH) algorithm, FlyHash. In contrast with classical LSH algorithms that produce low dimensional hash codes, FlyHash produces sparse high-dimensional hash codes and has also been shown to have superior empirical performance compared to classical LSH algorithms in similarity search. 
However, FlyHash uses random projections and cannot learn from data. Building on inspiration from FlyHash and the ubiquity of sparse expansive representations in neurobiology, our work proposes a novel hashing algorithm BioHash that produces sparse high dimensional hash codes in a data-driven manner. We show that BioHash outperforms previously published benchmarks for various hashing methods. 
Since our learning algorithm is based on a local and biologically plausible synaptic plasticity rule, our work provides evidence for the proposal that LSH might be a computational reason for the abundance of sparse expansive motifs in a variety of biological systems. 
We also propose a convolutional variant BioConvHash that further improves performance. From the perspective of computer science, BioHash and BioConvHash are fast, scalable and yield compressed binary representations that are useful for similarity search.

- [Bio-Inspired Hashing for Unsupervised Similarity Search](https://arxiv.org/abs/2001.04907)
- https://mitibmwatsonailab.mit.edu/research/blog/bio-inspired-hashing-for-unsupervised-similarity-search/
- https://deepai.org/publication/bio-inspired-hashing-for-unsupervised-similarity-search
- [https://github.com/josebetomex/BioHash](https://github.com/josebetomex/BioHash)
- http://www.people.vcu.edu/~gasmerom/MAT131/repnearest.html
- https://spaces.ac.cn/archives/8159

#### FlyHash


- https://github.com/dataplayer12/Fly-LSH
- https://science.sciencemag.org/content/358/6364/793/tab-pdf
- https://arxiv.org/abs/1812.01844

### Bloom filters

Bloom filters, counting Bloom filters, and multi-hashing tables

- https://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/
- http://www.cs.jhu.edu/~fabian/courses/CS600.624/slides/bloomslides.pdf
- https://www2021.thewebconf.org/papers/consistent-sampling-through-extremal-process/
- [Cache-, Hash- and Space-Efficient Bloom Filters](http://algo2.iti.kit.edu/singler/publications/cacheefficientbloomfilters-wea2007.pdf)
- [Fluid Co-processing: GPU Bloom-filters for CPU Joins](https://diegomestre2.github.io/files/fluid.pdf)
- https://diegomestre2.github.io/
- [Xor Filters: Faster and Smaller Than Bloom Filters](https://lemire.me/blog/2019/12/19/xor-filters-faster-and-smaller-than-bloom-filters/)

#### Counting Bloom filters

[The same property that results in false positives also makes it difficult to remove an element from the filter as there is no easy means of discerning if another element is hashed to the same bit. Unsetting a bit that is hashed by multiple elements can cause false negatives. Using a counter, instead of a bit, can circumvent this issue. The bit can be incremented when an element is hashed to a given location, and decremented upon removal. Membership queries rely on whether a given counter is greater than zero. This reduces the exceptional space-efficiency provided by the standard Bloom filter.](https://github.com/bitly/dablooms)

#### Bloomier Filter

- https://www.cs.princeton.edu/~chazelle/pubs/soda-rev04.pdf
- https://webee.technion.ac.il/~ayellet/Ps/nelson.pdf

#### Blocked Bloom filters

[Blocked Bloom filters are a cache-efficient variant of Bloom filters, the well-known approximate set data structure. To quote Daniel Lemire, they have unbeatable speed. See the directory benchmarks/ to determine exactly how fast Blobloom is compared to other packages.](https://github.com/greatroar/blobloom)

-  https://github.com/greatroar/blobloom


####  Multi-set filters

-  [Noisy Bloom Filters for Multi-Set Membership Testing](https://cs.nju.edu.cn/daihp/dh/NBF-SIGMETRICS16.pdf)

#### Quotient Filter



- https://ieeexplore.ieee.org/document/8425199

### HyperLogLog

- https://www.cnblogs.com/linguanh/p/10460421.html
- https://www.runoob.com/redis/redis-hyperloglog.html
 

### Skip Lists
[An ordered-key based data structure that allows for competitive performance dictionary or list while implementation remaining relatively easy. This data structure proves that probability can work along with being able to quick index certain items based on probability.](https://ieeexplore.ieee.org/document/8425199)
- http://homepage.cs.uiowa.edu/~ghosh/
- http://ticki.github.io/blog/skip-lists-done-right/
- https://lotabout.me/2018/skip-list/
- [Skip Lists: A Probabilistic Alternative to Balanced Trees](https://www.epaperpress.com/sortsearch/download/skiplist.pdf)

### Count-min Sketch

- https://florian.github.io/count-min-sketch/
- [Count Min Sketch: The Art and Science of Estimating Stuff](https://redislabs.com/blog/count-min-sketch-the-art-and-science-of-estimating-stuff/)
- http://dimacs.rutgers.edu/~graham/pubs/papers/cmencyc.pdf
- https://github.com/barrust/count-min-sketch
- http://hkorte.github.io/slides/cmsketch/
- [Data Sketching](https://cacm.acm.org/magazines/2017/9/220427-data-sketching/fulltext)
- [What is Data Sketching, and Why Should I Care?](http://dimacs.rutgers.edu/~graham/pubs/papers/cacm-sketch.pdf)

## Approximate Nearest Neighbor Searching and Polytope Approximation

- http://www.cse.ust.hk/faculty/arya/publications.html
- https://sunju.org/research/subspace-search/
- http://www.cs.memphis.edu/~nkumar/
- [NNS Benchmark: Evaluating Approximate Nearest Neighbor Search Algorithms in High Dimensional Euclidean Space](https://github.com/DBWangGroupUNSW/nns_benchmark)
- [Datasets for approximate nearest neighbor search](http://corpus-texmex.irisa.fr/)

### Proximity graphs

[A proximity graph is a simply a graph in which two vertices are connected by an edge if and only if the vertices satisfy particular geometric requirements.](http://math.sfsu.edu/beck/teach/870/brendan.pdf)

- http://mzwang.top/2021/03/12/proximity-graph-monotonicity/
- http://math.sfsu.edu/beck/teach/870/brendan.pdf
- [Practical Graph Mining With R](https://www.csc2.ncsu.edu/faculty/nfsamato/practical-graph-mining-with-R/PracticalGraphMiningWithR.html)
- https://graphworkflow.com/decoding/gestalt/proximity/

### Navigable Small World Graph

- http://mzwang.top/about/
- http://web.mit.edu/8.334/www/grades/projects/projects17/KevinZhou.pdf
- [Approximate nearest neighbor algorithm based on navigable small world graphs](https://publications.hse.ru/mirror/pubs/share/folder/x5p6h7thif/direct/128296059)
- [Navigable Small-World Networks](https://www.kth.se/social/upload/514c7450f276547cb33a1992/2-kleinberg.pdf)

### Hierarchical Navigable Small World (HNSW) Graph

   
- https://github.com/js1010/cuhnsw
- https://github.com/nmslib/hnswlib
- https://www.libhunt.com/l/cuda/t/hnsw
- https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces.html

### KNN graph

- [ ] [fast library for ANN search and KNN graph construction](https://github.com/ZJULearning/efanna)
- [ ] [Building KNN Graph for Billion High Dimensional Vectors Efficiently](https://github.com/lengyyy/KNN-Graph)
- [ ] https://github.com/aaalgo/kgraph
- [ ] https://www.msra.cn/zh-cn/news/features/approximate-nearest-neighbor-search
- [ ] https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/37599.pdf


## Compact Data Structures

- [Three Success Stories About Compact Data Structures](https://cacm.acm.org/magazines/2020/11/248206-three-success-stories-about-compact-data-structures/fulltext)
- [Image Similarity Search with Compact Data Structures](https://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p208-lv.pdf)
- [02951 Compact Data Structures](http://www2.compute.dtu.dk/courses/02951/)
- [Computation over Compressed Structured Data](https://www.dagstuhl.de/de/programm/kalender/semhp/?semnr=16431)
- http://www2.compute.dtu.dk/~phbi/
- http://www2.compute.dtu.dk/~inge/
- https://github.com/fclaude/libcds2
- https://diegocaro.github.io/thesis/index.html
- http://www.birdsproject.eu/course-compact-data-structures-during-udcs-international-summer-school-2018/


## Succinct data structures

In computer science, a succinct data structure is a data structure which uses an amount of space that is "close" to the information-theoretic lower bound, 
but (unlike other compressed representations) still allows for efficient query operations.

- https://www.cs.helsinki.fi/group/suds/
- https://www.cs.helsinki.fi/group/algodan/
- http://simongog.github.io/
- http://algo2.iti.kit.edu/gog/homepage/index.html
- https://arxiv.org/abs/1904.02809
- [Succinct Data Structures-Exploring succinct trees in theory and practice](http://www.eecs.tufts.edu/~aloupis/comp150/projects/SuccinctTreesinPractice.pdf)
- [Presentation "COBS: A Compact Bit-Sliced Signature Index" at SPIRE 2019 (Best Paper Award)](https://panthema.net/2019/1008-COBS-A-Compact-Bit-Sliced-Signature-Index/)


[The Succinct Data Structure Library (SDSL)](https://github.com/simongog/sdsl-lite) contains many succinct data structures from the following categories:

* Bit-vectors supporting Rank and Select
* Integer Vectors
* Wavelet Trees
* Compressed Suffix Arrays (CSA)
* Balanced Parentheses Representations
* Longest Common Prefix (LCP) Arrays
* Compressed Suffix Trees (CST)
* Range Minimum/Maximum Query (RMQ) Structures

- https://github.com/simongog/sdsl-lite
- https://github.com/simongog/sdsl-lite/wiki/Literature
- https://github.com/simongog/


## Learned Data Structure

- https://www2.eecs.berkeley.edu/Pubs/TechRpts/2009/EECS-2009-101.pdf
- https://arxiv.org/abs/1908.00672

###  Learning to Hash

By using hash-code to construct index, [we](https://cs.nju.edu.cn/lwj/slides/L2H.pdf) can achieve constant or
sub-linear search time complexity.


Hash functions are learned from a given training dataset.


- https://cs.nju.edu.cn/lwj/slides/L2H.pdf
- [Repository of Must Read Papers on Learning to Hash](https://learning2hash.github.io/)
- [Learning to Hash: Paper, Code and Dataset](https://cs.nju.edu.cn/lwj/L2H.html)
- [Learning to Hash with Binary Reconstructive Embeddings](https://papers.nips.cc/paper/3667-learning-to-hash-with-binary-reconstructive-embeddings)
- http://zpascal.net/cvpr2015/Lai_Simultaneous_Feature_Learning_2015_CVPR_paper.pdf
- https://github.com/twistedcubic/learn-to-hash
- https://cs.nju.edu.cn/lwj/slides/hash2.pdf
- [Learning to hash for large scale image retrieval](https://era.ed.ac.uk/handle/1842/20390)
- http://stormluke.me/learning-to-hash-intro/
- [Learning to Hash for Source Separation](https://saige.sice.indiana.edu/research-projects/BWSS-BLSH/)
- https://github.com/sunwookimiub/BLSH

#### Boosted LSH

-  https://saige.sice.indiana.edu/research-projects/bwss-blsh/
-  http://proceedings.mlr.press/v97/vorobev19a.html