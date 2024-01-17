# Cascade Ranking Models

>Given a query $q$ (context) and a set of documents $D$ (items), 
>the goal is to **order** elements of $D$ such that the resulting ranked list maximizes a user satisfaction metric $Q$ (criteria).

Except the subjective criteria, we can take advantage of NLP to unserstand the query and documnets and apply big data technology to process so many documents.


[A cascaded ranking architecture turns ranking into a pipeline of multiple stages, and has been shown to be a powerful approach to balancing efficiency and effectiveness trade-offs in large-scale search systems.](https://culpepper.io/publications/gcbc19-wsdm.pd)

[Our core idea is to consider the ranking problem as a "cascade", where ranking is broken into a finite number of distinct stages. Each stage considers successively richer and more complex features, but over successively smaller candidate document sets. The intuition is that although complex features are more time-consuming to compute, the additional overhead is offset by examining fewer documents. In other words, the cascade model views retrieval as a multi-stage progressive refinement problem, where each stage balances the cost of exploiting various features with the potential gain in terms of better results. We have explored this notion in the context of linear models and tree-based models.](http://lintool.github.io/NSF-projects/IIS-1144034/)

<img src="https://pic4.zhimg.com/v2-62f364c17d9bfde9a67dfa46699b5b99_b.jpg" width="47%">
<img src="https://pica.zhimg.com/v2-77538392bc888eee2b8c6743576a4cfe_b.jpg" width="47%">

- https://culpepper.io/publications/gcbc19-wsdm.pdf
- [Learning to Efficiently Rank with Cascades](http://lintool.github.io/NSF-projects/IIS-1144034/)
- https://www.nsf.gov/awardsearch/showAward?AWD_ID=1144034
- https://www1.cs.columbia.edu/~gravano/Qual/Papers/singhal.pdf
- https://github.com/frutik/awesome-search
- https://booking.ai/publications/home

> This document set is often retrieved from the collection using a simple unsupervised bag-of-words method, e.g. BM25. 
> This can potentially lead to learning a suboptimal ranking, since many relevant documents may be excluded from the initially retrieved set.

- [Two-Stage Learning to Rank for Information Retrieval](https://bendersky.github.io/pubs/2013-1.pdf)

<img src="https://pic4.zhimg.com/80/v2-ad9ea6d84f344c7e0bec2b61c69d124b_720w.webp" width="80%">

- https://blog.csdn.net/u013510838/article/details/123023259
- https://www.khoury.northeastern.edu/home/vip/teach/IRcourse/IR_surveys/

## Matching and Retrieval

Our aim in matching stage is to exclude the irrelevant documents with the query $q$ from the candidate documents $D$.

From another perspective, it is to choose $k$ from $n$,  
where $n$ is much larger than $k$.

- [Probabilistic n-Choose-k Models for Classification and Ranking](http://www.cs.princeton.edu/~rpa/pubs/swersky2012choose.pdf)
- [Finding the Best of Both Worlds: Faster and More Robust Top-k Document Retrieval](https://web2.qatar.cmu.edu/~mhhammou/SIGIR_20_LazyBM.pdf)
- [Top-k learning to rank: labeling, ranking and evaluation](https://dl.acm.org/doi/10.1145/2348283.2348384)
- [Why Not Yet: Fixing a Top-k Ranking that is Not Fair to Individuals](https://dl.acm.org/doi/abs/10.14778/3598581.3598606)
- [FA*IR: A Fair Top-k Ranking Algorithm](https://arxiv.org/abs/1706.06368)

### Relevance Matching

[A core problem of information retrieval (IR) is relevance matching (RM), where the goal is to rank documents by relevance to a user’s query.](https://aclanthology.org/D19-1540.pdf)

[Given a query and a set of candidate text documents, relevance ranking algorithms determine how **relevant** each text document is for the given query.](https://ieeexplore.ieee.org/document/9177802)

- https://haystackconf.com/
- https://www.mices.co/
- [Search @ Nextdoor: What are our Neighbors Searching For?](https://haystackconf.com/files/slides/haystack2022/Search-at-Nextdoor-What-are-our-neighbors-searching-for-Bojan-Babic.pdf)
- [Ranking Relevance in Yahoo Search](http://www.yichang-cs.com/yahoo/KDD16_yahoosearch.pdf)
- [Search Relevance Methodology](https://www.mices.co/mices2017/slides/MICES2017_Doug_Turnbull_OSC_SearchRelevanceMethodology.pdf)
- https://opensourceconnections.com/blog/2023/03/20/beyond-search-relevance-search-result-quality/
- https://www.sigir.org/sigir2007/tutorial2d.html
- https://haystackconf.com/us2023/talk-2/
- https://www.elastic.co/what-is/search-relevance


#### The Probabilistic Relevance


The assumptions about relevance are as follows:
1. Relevance is assumed to be a property of the document given
information need only, assessable without reference to other
documents; and
2. The relevance property is assumed to be binary.

[Either of these assumptions is at the least arguable.](https://www.khoury.northeastern.edu/home/vip/teach/IRcourse/IR_surveys/robertson_foundations.pdf) 

- [The Probabilistic Relevance Framework: BM25 and Beyond](https://dl.acm.org/doi/10.1561/1500000019)
- https://www.khoury.northeastern.edu/home/vip/teach/IRcourse/IR_surveys/robertson_foundations.pdf

### Semantic Matching

[There are fundamental differences between semantic matching and relevance matching](https://aclanthology.org/D19-1540/):
> Semantic matching emphasizes “meaning” correspondences by exploiting `lexical` information (e.g., words, phrases, entities) and `compositional structures` (e.g., dependency trees), 
> while relevance matching focuses on `keyword matching`. 

[The semantic matching problem in product search seeks to retrieve all semantically relevant products given a user query. Recent studies have shown that extreme multi-label classification (XMC) model enjoys both low inference latency and high recall in real-world scenarios. ](https://dl.acm.org/doi/10.1145/3583780.3614661)

[Extreme multi-label classification (XMC) is the problem of finding the relevant labels for an input, from a very large universe of possible labels.](https://arxiv.org/abs/2004.00198)


- [Relevance under the Iceberg: Reasonable Prediction for Extreme Multi-label Classification](https://dl.acm.org/doi/pdf/10.1145/3477495.3531767)
- [Extreme Multi-label Classification from Aggregated Labels](https://arxiv.org/abs/2004.00198)
- [Build Faster with Less: A Journey to Accelerate Sparse Model Building for Semantic Matching in Product Search](https://dl.acm.org/doi/10.1145/3583780.3614661)
- https://www.khoury.northeastern.edu/home/vip/teach/IRcourse/IR_surveys/ml_for_match-step2.pdf
- https://www.elastic.co/guide/en/elasticsearch/reference/current/semantic-search.html


#### Lexical Matching

[Classical information retrieval systems such as BM25 rely on exact lexical match and carry out search efficiently with inverted list index. ](https://arxiv.org/abs/2104.07186)

- [COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List](https://arxiv.org/abs/2104.07186)
- [NAIL: Lexical Retrieval Indices with Efficient Non-Autoregressive Decoders](https://arxiv.org/abs/2305.14499)
- [A Dense Representation Framework for Lexical and Semantic Matching](https://arxiv.org/abs/2206.09912)
- [SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval](https://arxiv.org/abs/2109.10086)
- https://arxiv.org/abs/2107.05720

#### Embedding-based Retrieval

`DSSM` stands for `Deep Structured Semantic Model`, or more general, `Deep Semantic Similarity Model`.
DSSM can be used to develop latent semantic models that project entities of different types (e.g., queries and documents) into a common low-dimensional semantic space for a variety of machine learning tasks such as ranking and classification. 
For example, in web search ranking, the relevance of a document given a query can be readily computed as the distance between them in that space. 

- [Deep Semantic Similarity Model](https://www.microsoft.com/en-us/research/project/dssm/)

DSSM is extended as the two tower model, where the query and document are represented as vectors via different neural networks.

<img src="https://pic1.zhimg.com/80/v2-82551bd8cd66c0c528cfcdf1732737d0_720w.webp" width="45%">
<img src="https://images.ctfassets.net/7w2tf600vbko/ruXJGQ4DuSAFSaI4Rbjaf/f87ed87fc98de8ec6af6c5f43d4b4082/Embedding_Figure_1.png" width="45%">

[Embedding based retrieval (EBR; a.k.a. vector search) provides an efficient implementation of semantic search and has seen wide adoption in e-commerce. ](https://haystackconf.com/eu2023/talk-13/)

- https://eng.snap.com/embedding-based-retrieval
- https://haystack.deepset.ai/
- [Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632)
- [Embedding-based Product Retrieval in Taobao Search](https://dl.acm.org/doi/abs/10.1145/3447548.3467101)
- [Que2Engage: Embedding-based Retrieval for Relevant and Engaging Products at Facebook Marketplace](https://arxiv.org/abs/2302.11052)
- [From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective](https://download-de.europe.naverlabs.com/splade/presentations/ReNeuIR_splade.pdf)
- [Embedding-based Query Language Models](https://lintool.github.io/robust04-analysis-papers/p147-zamani-2016.pdf)
- https://www.elastic.co/cn/blog/may-2023-launch-information-retrieval-elasticsearch-ai-model
- [Beyond the known: exploratory and diversity search with vector embeddings](https://haystackconf.com/eu2023/talk-5/)
- [Using Vector Databases to Scale Multimodal Embeddings, Retrieval and Generation](https://haystackconf.com/eu2023/talk-8/)
- [Search Engines: Combining Inverted and ANN Indexes for Scale](https://haystackconf.com/eu2023/talk-11/)
- [Evaluating embedding based retrieval beyond historical search results](https://haystackconf.com/eu2023/talk-13/)


### Hybrid Retrieval

[This hypothesis is plausible for methods using very different mechanisms for retrieval because there are many more irrelevant than relevant documents for most queries and corpuses. If methods retrieve relevant and irrelevant documents independently and uniformly at random, this imbalance means it is much more probable for relevant documents to match than irrelevant ones. ](https://www.elastic.co/cn/blog/improving-information-retrieval-elastic-stack-hybrid)

<img src="https://cdn.sanity.io/images/vr8gru94/production/9ae35de5d96d11e99208fc0220ed8ed6ab716bae-2640x608.png" width="80%">

- https://www.elastic.co/cn/blog/improving-information-retrieval-elastic-stack-hybrid
- [Mastering Hybrid Search: Blending Classic Ranking Functions with Vector Search for Superior Search Relevance](https://haystackconf.com/eu2023/talk-10/)
- https://www.pinecone.io/learn/hybrid-search-intro/
- [An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/pdf/2210.11934.pdf)

<img src="https://learn.microsoft.com/zh-CN/azure/search/media/hybrid-search/search-scoring-flow.png" width="70%">

- https://learn.microsoft.com/zh-CN/azure/search/hybrid-search-ranking
- https://opensearch.org/blog/hybrid-search/
- https://qdrant.tech/articles/hybrid-search/
- https://www.pinecone.io/learn/hybrid-search-intro/
- https://milvus.io/docs/hybridsearch.md

### Rank Aggregation

When there is just a single criterion (or "judge") for ranking, the task is relatively easy, 
and is simply a reaction of the judge's opinions and biases. 
In contrast, this paper addresses the problem of computing a "consensus" ranking of the alternatives, 
given the individual ranking preferences of several judges. We call this the `rank aggregation problem`.

- https://www.eecs.harvard.edu/~michaelm/CS222/rank.pdf

## Pre-Ranking

> [Existing pre-ranking systems primarily adopt the two-tower model since the "user-item decoupling architecture" paradigm is able to balance the efficiency and effectiveness.](https://dl.acm.org/doi/abs/10.1145/3511808.3557072)

> [The pre-ranking is widely considered a mini-ranking module, as it needs to rank hundreds of times more items than the ranking under limited latency.](https://arxiv.org/abs/2305.13647)

> [In the pre-ranking stage, vector-product based models with representation-focused architecture are commonly adopted to account for system efficiency. ](https://arxiv.org/abs/2105.07706)

The goal of pre-rank is to select some documents(items) efficiently.


- [COLD: Towards the Next Generation of Pre-Ranking System](https://arxiv.org/abs/2007.16122)
- [Towards a Better Trade-off between Effectiveness and Efficiency in Pre-Ranking: A Learnable Feature Selection based Approach](https://arxiv.org/abs/2105.07706)
- [Rethinking Large-scale Pre-ranking System: Entire-chain Cross-domain Models](https://dl.acm.org/doi/abs/10.1145/3511808.3557683)
- [Rethinking the Role of Pre-ranking in Large-scale E-Commerce Searching System](https://arxiv.org/abs/2305.13647)
- [IntTower: The Next Generation of Two-Tower Model for Pre-Ranking System](https://dl.acm.org/doi/abs/10.1145/3511808.3557072)

## Ranking

>Given a query $q$ (context) and a set of documents $D$ (items), 
>the goal is to **order** elements of $D$ such that the resulting ranked list maximizes a user satisfaction metric $Q$ (criteria).

In cascaded ranking architecture, the set $D$ is generated by matching(recall, retrieval).
Here we focus on learning to rank.


- http://ltr-tutorial-sigir19.isti.cnr.it/program-overview/
- [Efficient and Effective Tree-based and Neural Learning to Rank](https://arxiv.org/pdf/2305.08680.pdf)
- https://aclanthology.org/2023.acl-long.771/

### Learning to Rank

Learning to rank is usually formalized as a supervised learning task, while unsupervised learning and semi-supervised learning formulations are also possible. 
In learning, training data consisting of sets of objects as well as the total or partial orders of the objects in each set is given, and a ranking model is learned using the data. 
In prediction, a new set of objects is given, and a ranking list of the objects is created using the ranking model.

- https://link.springer.com/referenceworkentry/10.1007/978-1-4899-7687-1_893

#### Feature Selection for Learning to Rank

- [A Systematic Study of Feature Selection Methods for Learning to Rank Algorithms](https://dl.acm.org/doi/10.4018/IJIRR.2018070104)
- [FSMRank: feature selection algorithm for learning to rank](https://pubmed.ncbi.nlm.nih.gov/24808475/)
- [Neural Feature Selection for Learning to Rank](https://machinelearning.apple.com/research/neural-feature-selection)


#### LambdaRank

**LambdaRank**  introduced the $\lambda_i$ when update of parameters $w$ of the model $f:\mathbb{R}^P\to\mathbb{R}$.
The key observation of LambdaRank is thus that in order to train a model, we do not need the costs themselves: we only need the gradients (of the costs with respect to
the model scores).

You can think of these gradients as little arrows attached to each document in the ranked list, indicating which direction we’d like those documents to move. LambdaRank simply took the RankNet gradients, which we knew worked well, and scaled them by the change in NDCG found by swapping each pair of documents. We found that this training generated models with significantly improved relevance (as measured by NDCG) and had an added bonus of uncovering a further trick that improved overall training speed (for both RankNet and LambdaRank). Furthermore, surprisingly, we found empirical evidence (see also this paper) that the training procedure does actually optimize NDCG, even though the method, although intuitive and sensible, has no such guarantee.

We compute the  gradients of RankNet by:
$$
\frac{\partial L}{\partial w} = \sum_{(i, j)}\frac{\partial L_{i, j}}{\partial w}=\sum_{(i, j)}[\frac{\partial L_{i, j}}{\partial s_i}+\frac{\partial L_{i,j}}{\partial s_j}].
$$

Observe that
$$\frac{\partial L_{i, j}}{\partial s_i} = -\frac{\partial L_{i,j}}{\partial s_j}$$
and define

$$
{\lambda}_{i,j}=\frac{\partial L_{i, j}}{\partial s_i} = -\frac{\partial L_{i, j}}{\partial s_j} = -\frac{\sigma}{1 + \exp(\sigma(s_i - s_j))}.
$$

What is more, we can extend it to

$$
{\lambda}_{i,j}=  -\frac{\sigma}{1+\exp(\sigma( s_i -s_j))}|\Delta Z|,
$$
where $\Delta Z$ is the size of the change in some **Information Retrieval Measures** ${Z}$.

And $\lambda_{i}$ with respect to ${i}$-th item is defined as
$$\lambda_i = \sum_{i\in(i,j)}\lambda_{(i,j)}-\sum_{j\in(j,i)}\lambda_{(i,j)}$$

<img src=https://p0.meituan.net/travelcube/d6142123b31212f4854fd4e53da5831e14664.png width=50% />

***

- http://blog.camlcity.org/blog/lambdarank.html
- [LambdaFM: Learning Optimal Ranking with Factorization Machines Using Lambda Surrogates](http://wnzhang.net/papers/lambdafm.pdf)
- [Practical Lessons from Predicting Clicks on Ads at Facebook](http://quinonero.net/Publications/predicting-clicks-facebook.pdf)

#### LambdaMART

* https://github.com/wepe/efficient-decision-tree-notes

**LambdaMART** is the boosted tree version of LambdaRank, which is based on RankNet. It takes the ranking problem as classification problem.

MART stands for [Multiple Additive Regression Tree](http://statweb.stanford.edu/~jhf/MART.html).
In LambdaRank, we compute the gradient. And we can use this gradient to make up the GBRT.
> LambdaMART had an added advantage: the training of tree ensemble models can be very significantly sped up over the neural net equivalent (this work, led by O. Dekel, is not yet published). This allows us to train with much larger data sets, which again gives improved ranking accuracy. From [RankNet: A ranking retrospective](https://www.microsoft.com/en-us/research/blog/ranknet-a-ranking-retrospective/).

***

<img title="LambdaMART" src = "https://liam.page/uploads/images/LTR/LambdaMART.png" width = 80% />

***

To implement LambdaMART we just use MART, specifying appropriate gradients
and the Newton step.
The key point is the gradient of the ${\lambda}_i$:
$$ w_i = \frac{\partial y_i}{\partial F_{k-1}(\vec{x}_i)} $$
where $\lambda_i = y_i$ is defined in **LambdaRank**.
LambdaRank updates all the weights after each query is examined. The
decisions (splits at the nodes) in LambdaMART, on the other hand, are computed using all the data that falls to that node, and so LambdaMART updates only a few
parameters at a time (namely, the split values for the current leaf nodes), but using all the data (since every $x_i$ lands in some leaf).
This means in particular that LambdaMART is able to choose splits and leaf values that may decrease the utility for some queries, as long as the overall utility increases.

- [x] [LambdaMART 不太简短之介绍](https://liam.page/2016/07/10/a-not-so-simple-introduction-to-lambdamart/)
- [X] [Learning To Rank之LambdaMART的前世今生](https://blog.csdn.net/huagong_adu/article/details/40710305)s
- [X] [LambdaMart Slides](https://liam.page/uploads/slides/lambdamart.pdf)
- [ ] [From RankNet to LambdaRank to LambdaMART: An Overview](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/)
- [ ] [Ranknet a ranking retrospective](https://www.microsoft.com/en-us/research/blog/ranknet-a-ranking-retrospective/)
- [ ] [LambdaMart Demystified](https://staff.fnwi.uva.nl/e.kanoulas/wp-content/uploads/Lecture-8-1-LambdaMart-Demystified.pdf)
- [ ] [Unbiased LambdaMART: An Unbiased Pairwise Learning-to-Rank Algorithm](https://arxiv.org/pdf/1809.05818.pdf)

**GBRT+LR** can also used to predict the CTR ratio. On short but incomplete word, it is **GBRT + LR** - **gradient boosting regression tree and logistic regression**.
GBRT is introduced at the *Boosting* section. *LR* is to measure the cost as the same in RankNet.

- [ ] [Learning From Weights: A Cost-Sensitive Approach For Ad Retrieval](https://arxiv.org/abs/1811.12776)
- [X] https://www.jianshu.com/p/96173f2c2fb4
- [ ] [Boosted Ranking Models: A Unifying Framework for Ranking Predictions](http://www.cs.cmu.edu/~kdelaros/)
- [ ] https://www.cnblogs.com/genyuan/p/9788294.html


#### lambdaLoss

- https://ml-compiled.readthedocs.io/en/latest/ranking.html
- [The LambdaLoss Framework for Ranking Metric Optimization](https://dl.acm.org/doi/abs/10.1145/3269206.3271784)
- [Revisiting Approximate Metric Optimization in the Age of Deep Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3331184.3331347)
- [On Optimizing Top-𝐾 Metrics for Neural Ranking Models](https://dl.acm.org/doi/pdf/10.1145/3477495.3531849)
- [LambdaRank Gradients are Incoherent](https://dl.acm.org/doi/10.1145/3583780.3614948)

### Click-Through Rate  Prediction

[The purpose of click-through rate (CTR) prediction is to anticipate how likely a person is to click on an advertisement or item. ](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-023-00688-6)

- https://arxiv.org/abs/1810.11921
- https://arxiv.org/abs/2102.07619 
- https://arxiv.org/abs/2105.05563
- https://arxiv.org/abs/2304.13643



## Re-Ranking

Beyond user satisfaction metric,
[the system can re-rank the candidates to consider additional criteria or constraints.](https://developers.google.com/machine-learning/recommendation/dnn/re-ranking?hl=en)

- https://github.com/LibRerank-Community/LibRerank/
- https://librerank-community.github.io/slides-recsys22-tutorial-neuralreranking.pdf
- https://librerank-community.github.io/
- https://arxiv.org/pdf/2202.06602.pdf
- [Personalized Re-ranking for Recommendation](https://www.yongfeng.me/attach/pei-recsys2019.pdf)

### Diversity

- [Diversity in Ranking using Negative Reinforcement](https://arxiv.org/abs/1207.6600)
- [Matching Search Result Diversity with User Diversity Acceptance in Web Search Sessions](https://dl.acm.org/doi/10.1145/3477495.3531880)
- https://ujwalgadiraju.com/Publications/Bias2020.pdf

### Fairness

- https://www.ashudeepsingh.com/
- [Fairness in Ranking: A Survey](https://arxiv.org/abs/2103.14000)
- [Fair ranking: a critical review, challenges, and future directions](https://arxiv.org/abs/2201.12662)
- [Fairness in Ranking, Part I: Score-Based Ranking](https://dl.acm.org/doi/10.1145/3533379)
- [Fairness in Ranking, Part II: Learning-to-Rank and Recommender Systems](https://dl.acm.org/doi/10.1145/3533380)
- [Fairness of Exposure in Rankings](https://www.cs.cornell.edu/~tj/publications/singh_joachims_18a.pdf)
- [Fairness in Recommendation Ranking through Pairwise Comparisons](https://dl.acm.org/doi/abs/10.1145/3292500.3330745)

### Adversarial Ranking

- https://arxiv.org/abs/2002.11293
- https://www.microsoft.com/en-us/research/publication/adversarial-ranking-language-generation/


## Other

- [Cascading Bandits: Learning to Rank in the Cascade Model](http://zheng-wen.com/Cascading_Bandit_Paper.pdf)
- [Pre-training Methods in Information Retrieval](https://arxiv.org/pdf/2111.13853.pdf)


Given a user query, the top matching layer is responsible for providing `semantically relevant` ad candidates to the next layer, while the ranking layer at the bottom concerns more about business indicators (e.g., CPM, ROI, etc.) of those ads. The clear separation between the `matching and ranking` objectives results in a lower commercial return. 
[The Mobius project has been established to address this serious issue.](https://dl.acm.org/doi/abs/10.1145/3292500.3330651)

- [MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu’s Sponsored Search](http://research.baidu.com/uploads/5d12eca098d40.pdf)
- [RankFlow: Joint Optimization of Multi-Stage Cascade Ranking Systems as Flows](https://dl.acm.org/doi/10.1145/3477495.3532050)
- [Joint Optimization of Cascade Ranking Models](https://culpepper.io/publications/gcbc19-wsdm.pdf)

### Calibrated Ranking 

- https://arxiv.org/abs/2211.01494
- https://marc.najork.org/

### Relevance Feedback

* After initial retrieval results are presented, allow the user to provide feedback on the relevance of one or more of the retrieved documents.
* Use this feedback information to reformulate the query.
* Produce new results based on reformulated query.
* Allows more interactive, multi-pass process.

- [Relevance Feedback Algorithms Inspired By Quantum Detection](https://ieeexplore.ieee.org/document/7350145/)
- [Relevance Feedback In Image Retrieval](http://dimacs.rutgers.edu/~billp/pubs/RelevanceFeedbackInImageRetrieval.pdf)
- [Comparing Relevance Feedback Algorithms for Web Search](http://wwwconference.org/proceedings/www2005/docs/p1052.pdf)
- [Relevance Feedback for Best Match Term Weighting Algorithms in Information Retrieval](https://www.ercim.eu/publication/ws-proceedings/DelNoe02/hiemstra.pdf)
