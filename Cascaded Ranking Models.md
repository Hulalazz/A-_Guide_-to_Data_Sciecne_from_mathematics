# Cascade Ranking Models

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

## Matching


### Relevance Matching

[A core problem of information retrieval (IR) is relevance matching (RM), where the goal is to rank documents by relevance to a user’s query.](https://aclanthology.org/D19-1540.pdf)
[Given a query and a set of candidate text documents, relevance ranking algorithms determine how **relevant** each text document is for the given query.](https://ieeexplore.ieee.org/document/9177802)

Our aim is to exclude the irrelevant documents with the query from the candidate documents.


- [Search @ Nextdoor: What are our Neighbors Searching For?](https://haystackconf.com/files/slides/haystack2022/Search-at-Nextdoor-What-are-our-neighbors-searching-for-Bojan-Babic.pdf)
- https://haystackconf.com/
- [Ranking Relevance in Yahoo Search](http://www.yichang-cs.com/yahoo/KDD16_yahoosearch.pdf)

From another perspective, it is to select $k$ from $n$ where $n$ is much larger than $k$.

- [Probabilistic n-Choose-k Models for Classification and Ranking](http://www.cs.princeton.edu/~rpa/pubs/swersky2012choose.pdf)
- [Top-k learning to rank: labeling, ranking and evaluation](https://dl.acm.org/doi/10.1145/2348283.2348384)
- [Why Not Yet: Fixing a Top-k Ranking that is Not Fair to Individuals](https://dl.acm.org/doi/abs/10.14778/3598581.3598606)
- [FA*IR: A Fair Top-k Ranking Algorithm](https://arxiv.org/abs/1706.06368)


### Semantic Matching

[The semantic matching problem in product search seeks to retrieve all semantically relevant products given a user query. Recent studies have shown that extreme multi-label classification (XMC) model enjoys both low inference latency and high recall in real-world scenarios. ](https://dl.acm.org/doi/10.1145/3583780.3614661)

[There are fundamental differences between semantic matching and relevance matching](https://aclanthology.org/D19-1540/):
> Semantic matching emphasizes “meaning” correspondences by exploiting `lexical` information (e.g., words, phrases, entities) and `compositional structures` (e.g., dependency trees), 
> while relevance matching focuses on `keyword matching`. 

[Extreme multi-label classification (XMC) is the problem of finding the relevant labels for an input, from a very large universe of possible labels.](https://arxiv.org/abs/2004.00198)

- [Build Faster with Less: A Journey to Accelerate Sparse Model Building for Semantic Matching in Product Search](https://dl.acm.org/doi/10.1145/3583780.3614661)
- [Relevance under the Iceberg: Reasonable Prediction for Extreme Multi-label Classification](https://dl.acm.org/doi/pdf/10.1145/3477495.3531767)
- [Extreme Multi-label Classification from Aggregated Labels](https://arxiv.org/abs/2004.00198)


### Two Tower Models

`DSSM` stands for `Deep Structured Semantic Model`, or more general, `Deep Semantic Similarity Model`.
DSSM can be used to develop latent semantic models that project entities of different types (e.g., queries and documents) into a common low-dimensional semantic space for a variety of machine learning tasks such as ranking and classification. 
For example, in web search ranking, the relevance of a document given a query can be readily computed as the distance between them in that space. 

- [Deep Semantic Similarity Model](https://www.microsoft.com/en-us/research/project/dssm/)



## Pre-Ranking

[Existing pre-ranking systems primarily adopt the two-tower model since the "user-item decoupling architecture" paradigm is able to balance the efficiency and effectiveness.](https://dl.acm.org/doi/abs/10.1145/3511808.3557072)
Pre-ranking stage is in the middle of a cascade link, 
which is absorbed in preliminarily filtering items (thousands of scale) retrieved from the previous recall stage 
and generating candidates (hundreds of scale) for the subsequent ranking stage.

- [Towards a Better Trade-off between Effectiveness and Efficiency in Pre-Ranking: A Learnable Feature Selection based Approach](https://arxiv.org/abs/2105.07706)
- [Rethinking Large-scale Pre-ranking System: Entire-chain Cross-domain Models](https://dl.acm.org/doi/abs/10.1145/3511808.3557683)
- [COLD: Towards the Next Generation of Pre-Ranking System](https://arxiv.org/abs/2007.16122)
- [IntTower: The Next Generation of Two-Tower Model for Pre-Ranking System](https://dl.acm.org/doi/abs/10.1145/3511808.3557072)

## Ranking

>Given a query $q$ (context) and a set of documents $D$ (items), 
>the goal is to **order** elements of $D$ such that the resulting ranked list maximizes a user satisfaction metric $Q$ (criteria).

In cascaded ranking architecture, the set $D$ is generated by matching(recall, retrieval).
Here we focus on learning to rank.


- http://ltr-tutorial-sigir19.isti.cnr.it/program-overview/
- [Efficient and Effective Tree-based and Neural Learning to Rank](https://arxiv.org/pdf/2305.08680.pdf)

## Re-Ranking

Beyond user satisfaction metric,
[the system can re-rank the candidates to consider additional criteria or constraints.](https://developers.google.com/machine-learning/recommendation/dnn/re-ranking?hl=en)

- https://github.com/LibRerank-Community/LibRerank/
- https://librerank-community.github.io/slides-recsys22-tutorial-neuralreranking.pdf
- https://librerank-community.github.io/
- https://arxiv.org/pdf/2202.06602.pdf
- [Personalized Re-ranking for Recommendation](https://www.yongfeng.me/attach/pei-recsys2019.pdf)

### Rank Aggregation

When there is just a single criterion (or "judge") for ranking, the task is relatively easy, 
and is simply a reaction of the judge's opinions and biases. 
(If simplicity were the only desideratum, dictatorship would prevail over democracy.) 
In contrast, this paper addresses the problem of computing a "consensus" ranking of the alternatives, 
given the individual ranking preferences of several judges. We call this the `rank aggregation problem`.

- https://www.eecs.harvard.edu/~michaelm/CS222/rank.pdf

## Other

- [Cascading Bandits: Learning to Rank in the Cascade Model](http://zheng-wen.com/Cascading_Bandit_Paper.pdf)
- [Pre-training Methods in Information Retrieval](https://arxiv.org/pdf/2111.13853.pdf)


Given a user query, the top matching layer is responsible for providing `semantically relevant` ad candidates to the next layer, while the ranking layer at the bottom concerns more about business indicators (e.g., CPM, ROI, etc.) of those ads. The clear separation between the `matching and ranking` objectives results in a lower commercial return. 
[The Mobius project has been established to address this serious issue.](https://dl.acm.org/doi/abs/10.1145/3292500.3330651)

- [MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu’s Sponsored Search](http://research.baidu.com/uploads/5d12eca098d40.pdf)
- [RankFlow: Joint Optimization of Multi-Stage Cascade Ranking Systems as Flows](https://dl.acm.org/doi/10.1145/3477495.3532050)
- [Joint Optimization of Cascade Ranking Models](https://culpepper.io/publications/gcbc19-wsdm.pdf)

### Relevance Feedback

* After initial retrieval results are presented, allow the user to provide feedback on the relevance of one or more of the retrieved documents.
* Use this feedback information to reformulate the query.
* Produce new results based on reformulated query.
* Allows more interactive, multi-pass process.

- [Relevance Feedback Algorithms Inspired By Quantum Detection](https://ieeexplore.ieee.org/document/7350145/)
- [Relevance Feedback In Image Retrieval](http://dimacs.rutgers.edu/~billp/pubs/RelevanceFeedbackInImageRetrieval.pdf)
- [Comparing Relevance Feedback Algorithms for Web Search](http://wwwconference.org/proceedings/www2005/docs/p1052.pdf)
- [Relevance Feedback for Best Match Term Weighting Algorithms in Information Retrieval](https://www.ercim.eu/publication/ws-proceedings/DelNoe02/hiemstra.pdf)
