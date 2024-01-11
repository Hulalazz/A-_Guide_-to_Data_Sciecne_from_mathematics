# Cascade Ranking Models

[A cascaded ranking architecture turns ranking into a pipeline of multiple stages, and has been shown to be a powerful approach to balancing efficiency and effectiveness trade-offs in large-scale search systems.](https://culpepper.io/publications/gcbc19-wsdm.pd)

[Our core idea is to consider the ranking problem as a "cascade", where ranking is broken into a finite number of distinct stages. Each stage considers successively richer and more complex features, but over successively smaller candidate document sets. The intuition is that although complex features are more time-consuming to compute, the additional overhead is offset by examining fewer documents. In other words, the cascade model views retrieval as a multi-stage progressive refinement problem, where each stage balances the cost of exploiting various features with the potential gain in terms of better results. We have explored this notion in the context of linear models and tree-based models.](http://lintool.github.io/NSF-projects/IIS-1144034/)

<img src="https://pic4.zhimg.com/v2-62f364c17d9bfde9a67dfa46699b5b99_b.jpg">

<img src="https://pica.zhimg.com/v2-77538392bc888eee2b8c6743576a4cfe_b.jpg">

- https://culpepper.io/publications/gcbc19-wsdm.pdf
- http://lintool.github.io/NSF-projects/IIS-1144034/
- https://www.nsf.gov/awardsearch/showAward?AWD_ID=1144034
- https://www1.cs.columbia.edu/~gravano/Qual/Papers/singhal.pdf
- https://github.com/frutik/awesome-search

## Matching: relevance ranker

- [Search @ Nextdoor: What are our Neighbors Searching For?](https://haystackconf.com/files/slides/haystack2022/Search-at-Nextdoor-What-are-our-neighbors-searching-for-Bojan-Babic.pdf)
- https://haystackconf.com/

## Pre-Ranking

- https://arxiv.org/pdf/2105.07706.pdf
- https://dl.acm.org/doi/abs/10.1145/3511808.3557683
- https://arxiv.org/abs/2007.16122

## Ranking

>Given a query $q$ (context) and a set of documents $D$ (items), 
>the goal is to **order** elements of $D$ such that the resulting ranked list maximizes a user satisfaction metric $Q$ (criteria).

- http://ltr-tutorial-sigir19.isti.cnr.it/program-overview/
- [Efficient and Effective Tree-based and Neural Learning to Rank](https://arxiv.org/pdf/2305.08680.pdf)

## Re-Ranking


## Other

- [Cascading Bandits: Learning to Rank in the Cascade Model](http://zheng-wen.com/Cascading_Bandit_Paper.pdf)



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
- https://www.ercim.eu/publication/ws-proceedings/DelNoe02/hiemstra.pdf
