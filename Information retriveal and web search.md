## IR and Search

[RISE: Repository of Online Information Sources Used in Information Extraction Tasks](https://www.isi.edu/info-agents/RISE/)
[AI in Information Retrieval and Language Processing collected by Wlodzislaw Duch](http://www.is.umk.pl/~duch/IR.html)

If the recommendation is to solve the information overload problem, information retrieval and search technology  is to find the relative entity in web or some data base if the query is given.
[Technically, IR studies the acquisition, organization, storage, retrieval, and distribution of information.](http://www.dsi.unive.it/~dm/Slides/5_info-retrieval.pdf)
Information is in diverse format or form, such as character strings(texts), images, voices and videos so that information retrieval has diverse subfields such as [multimedia information retrieval](http://press.liacs.nl/mlew/mir2019.html) and [music infromation retrival](https://musicinformationretrieval.com/index.html). Search engine is considered as a practical application of information retrieval.  

`Relevance`, `Ranking`  and `Context`  are three foundation stones of search. In this section, we focus on relevance more than rank.
If interested in the history of information retrieval, Mark Sanderson and W. Bruce Croft wrote a paper for [The History of Information Retrieval Research](https://ciir-publications.cs.umass.edu/pub/web/getpdf.php?id=1066).

[The basic functions of a search engine can be described as _crawling, data mining, indexing and query processing_. `Crawling` is the act of sending small programed bots out to collect information. `Data mining` is storing the information collected by the bots. `Indexing` is ordering the information systematically. And `query processing` is the mathematical process in which a person's query is compared to the index and the results are presented to that person.](https://lifepacific.libguides.com/c.php?g=155121&p=1018180)


<img title="IR Process" src="https://hsto.org/files/4b9/a9b/1a6/4b9a9b1a60d041b2b4dfeca4b7989586.png" width="50%" />


* https://en.wikipedia.org/wiki/Information_retrieval
* [sease: make research in Information Retrieval more accessible](https://sease.io/)
* [search|hub](http://searchhub.io)
* [Cortical.io](https://www.cortical.io/)
* [FreeDiscovery Engine](http://freediscovery.io/doc/stable/engine/)
* [Index weblogs, mainstream news, and social media with Datastreamer](https://www.datastreamer.io/)
* https://www.omnity.io/
* https://www.luigisbox.com/
* [Some Tutorilas in IR](https://ielab.io/publications/scells-2018-querylab)
* [multimedia information retrieval](http://press.liacs.nl/mlew/mir2019.html)
* [Notes on Music Information Retrieval](https://musicinformationretrieval.com/index.html)
* https://ntent.com/
* http://www.somaproject.eu/

### Information Acquisition and Overload

The first step of information retrieval is to acquise the information itself. The web-scale information brings information overload problem, which `search  engine` or  `web search` attempts to solve.  

* https://iorgforum.org/
* [VII. Information Acquisition](www.science.unitn.it/~pgiorgio/db2/slides/9-infoacquisition.pdf)
* [Automatically modelling and distilling knowledge within AI!](https://ai-distillery.io/)

### Information Organization and Storage: Indexing and Index

Index as data structure is to organize the information efficiently in order to search some specific terms.

First, let us consider the case where we do not remember some key terms as reading some references, the appendices may include index recording the places where the terms firstly appear such as the following images shown.
![chinese](http://www.kfzimg.com/G06/M00/8F/16/p4YBAFsp_5mAHTEFAAY9LXEBT0k044_b.jpg)

Search engine takes advantage of this idea: it is best to store the palces where the terms/words appear in key-value format where the key, values is the terms and their places, respectively.

* [Elasticsearch from the Bottom Up, Part 1](https://www.elastic.co/blog/found-elasticsearch-from-the-bottom-up)
* [Intellectual Foundations for Information Organization and Information](http://people.ischool.berkeley.edu/~glushko/IFIOIR/)
* [Inverted Index versus Forward Index](http://www.darwinbiler.com/inverted-index-vs-forward-index/)
* http://planet.botany.uwc.ac.za/nisl/GIS/GIS_primer/index.htm


### Information Retrieval

<img title = "search process" src = "http://www.searchtools.com/slides/images/search-process.gif" width="50%" />

#### Query Parser: Query Understanding

Query is often some keywords in natural language such as English or Chinese. We use the search engine when we would like to find some information related with the keywords on the web/internet, which means  we do not completely know what the result is. Additionally, all information is digitalized in computer and the computers do not understand the natural language natively.
For example, `synonyms` are different as character or string data structure in computer.
Natural language processing(NLP) or natural language understanding(NLU)  facilitate the computers to comprehend the query.


* [Query Understanding](https://github.com/sanazb/Query-Understanding)
* [Exploring Query Parsers](https://lucidworks.com/post/exploring-query-parsers/)
* [Query Understanding: An efficient way how to deal with long tail queries](https://www.luigisbox.com/blog/query-understanding/)
* [The Art of Tokenization](https://www.ibm.com/developerworks/community/blogs/nlp/entry/tokenization?lang=en)
* https://ntent.com/technology/query-understanding/

<img src="https://ntent.com/wp-content/uploads/2017/01/Query-Understanding2.jpg" width="60%" />


Response | Time|[NLP Pipeline of Query Understanding](http://mlwiki.org/index.php/NLP_Pipeline)
---|---|---
[Query Auto Completion](https://www.jianshu.com/p/c7bc74d3657d)| Before the query input is finished|[Tokenization](http://mlwiki.org/index.php/Tokenization)
[Spelling Correction](https://nlp.stanford.edu/IR-book/html/htmledition/spelling-correction-1.html)| When the query input is finished|[Stop words removal](http://mlwiki.org/index.php/Stop_Words)
[Semantic Analysis](https://quanteda.io/articles/pkgdown/examples/lsa.html)| After the query input is finished|[Text Normalization](http://mlwiki.org/index.php/Text_Normalization)
[Query Suggestion](https://zhuanlan.zhihu.com/p/23693891)|  After the query input is finished|[POS Tagging](http://nlpprogress.com/english/part-of-speech_tagging.html)
[Intention Analysis](https://aiaioo.wordpress.com/tag/intention-analysis/)|  After the query input|[Named Entity Recogition](https://cs230-stanford.github.io/pytorch-nlp.html)



* http://partofspeech.org/
* https://nlpprogress.com/

#### Relevance and Rank

Recall the definition of  `Discounted Cumulative Gain(DCG)`:

$$
{DCG}_p= \sum_{i=1}^{p} \frac{{rel}_i}{\log_{2}(i+1)}
$$

where ${rel}_i$ is the relevance of the document and query.

However, it is discussed how to compute the relevance of the document and query. The document is always text such as html file so natural language processing plays a lead role in computing the relevances.
For other types information retrieval system, it is different to compute the relevance. For example, imagine  search engine is to find and return the images similar on the internet  with the given image query, where the information is almost in pixel format rather than text/string.


`Term frequency(tf)` of a word ${w}$ in a given document ${doc}$ is definded as
$$
tf(w| doc)=\frac{\text{the number of the word ${w}$ in}\,\,\,doc}{\text{the number of words in}\,\,\,doc} .
$$
It is to measure how popular the word ${w}$ in the document $doc$.
`Inverse document frequency(idf)` of a word ${w}$ in a given document list $D=\{doc_i\mid i=1,2,\cdots, N\}$ is defined
$$
idf(w\mid D)=\log\frac{\text{the number of documents in the list $D$} }{\text{the number of document containing the word $w$}+1},
$$
which is to measure how popular the word $w$ i the document list.
**tf-idf** is a rough way of approximating how users value the relevance of a text match, defined as
$$\text{tf-idf}=tf(w| doc)\times idf(w\mid D)$$

**BM25(Best Matching 25)** is a modification of `Binary Independent Model`.
$$
\sum_{i\in Q}\log\frac{ (r_i+0.5)/(R-r_i+0.5) }{ (n_i-r_i+0.5)/(N-R-n_i+r_i+0.5) }\times \frac{(k_1+1)f_i}{ K + f_i}
\times \frac{(k_2+1) {tf}_{tq}}{ K+{tf}_{tq} }
$$

* [BM25 The Next Generation of Lucene Relevance](https://opensourceconnections.com/blog/2015/10/16/bm25-the-next-generation-of-lucene-relevation/)

**TextRank**

[David Ten](https://xang1234.github.io/textrank/) wrote a blog on `TextRank`:
> For keyword extraction we want to identify a subset of terms that best describe the text. We follow these steps:
> 1. Tokenize and annotate with Part of Speech (PoS). Only consider single words. No n-grams used, multi-words are reconstructed later.
> 2. Use syntactic filter on all the lexical units (e.g. all words, nouns and verbs only).
> 3. Create and edge if lexical units co-occur within a window of N words to obtain an unweighted undirected graph.
> 4. Run the text rank algorithm to rank the words.
> 5. We take the top lexical words.
> 6. Adjacent keywords are collapsed into a multi-word keyword.

TextRank model is graph-based derived from Google’s PageRank. It constructs a weighted graph $G$:

* the node set $V$ consists of all sentences in the document;
* the weight is the similarity of each sentence pair, i.e., $w_{i,j}=Similarity (V_i, V_j)$.

The weight of each sentence depends on the weights of its neighbors:
$$WS(V_i)=(1-d)+d\times {\sum}_{V_j\in In(V_i)}\frac{w_{ij}}{\sum_{V_k\in Out(V_j)}}WS(V_j).$$

***
* [TextRank: Bringing Order into Texts](https://www.aclweb.org/anthology/W04-3252)
* [Keyword and Sentence Extraction with TextRank (pytextrank)](https://xang1234.github.io/textrank/)
* https://zhuanlan.zhihu.com/p/41091116
* [TextRank for Text Summarization](https://nlpforhackers.io/textrank-text-summarization/)
* https://www.quantmetry.com/tag/textrank/
* [Textrank学习](https://blog.csdn.net/Silience_Probe/article/details/80699662)

**Text Summarization**

[A summary can defined as “a text that is produced from one or more texts, that conveys important information in the original text(s), and that is no longer than half of the original text(s) and usually significantly less than that”. Automatic text summarization is the process of extracting such a summary from given document(s).](http://sidhant.io/kiss-keep-it-short-and-simple)

* [Gensim: Topic Model for Human](https://radimrehurek.com/gensim/index.html)
* [KISS: Keep It Short and Simple](http://sidhant.io/kiss-keep-it-short-and-simple)
* [NLP buddy](https://nlpbuddy.io/about)
* https://whoosh.readthedocs.io/en/latest/index.html
* https://malaya.readthedocs.io/en/latest/
* [Automatic Text Summarization with Python](https://ai.intelligentonlinetools.com/ml/text-summarization/)
* http://veravandeseyp.com/ai-repository/
* [Text Summarization in Python: Extractive vs. Abstractive techniques revisited](https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/)
* https://pypi.org/project/sumy/
* [自动文本摘要（Auto Text Summarization)](http://www.morrislee.me/%E8%87%AA%E5%8A%A8%E6%96%87%E6%9C%AC%E6%91%98%E8%A6%81%EF%BC%88auto-text-summarization%EF%BC%89/)

In the popular open search engine [ElasticSearch](https://www.elastic.co/cn/products/elasticsearch), the score formula is more complex and complicated.

**Document Similarity**

- [ ] [Document Similarity in Machine Learning Text Analysis with ELMo](https://ai.intelligentonlinetools.com/ml/document-similarity-in-machine-learning-text-analysis-with-elmo/)
- [ ] [Documents similarity](http://text2vec.org/similarity.html)
- [ ] https://copyleaks.com/
- [ ] https://www.wikiwand.com/en/Semantic_similarity
- [ ] https://spacy.io/

#### PageRank for Web Search

`Centrality of network` assigns an importance score based purely on the number of links held by each node.

`Search Engine Optimziation(SEO)` is a bussiness type to boost the website higher.

* http://ryanrossi.com/search.php
* [MGT 780/MGT 795 Social Network Analysis](http://www.analytictech.com/mgt780/)
* [The Anatomy of a Large-Scale Hypertextual Web Search Engine by Sergey Brin and Lawrence Page](http://infolab.stanford.edu/~backrub/google.html)
* [The Mathematics of Google Search](http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/)
* [HITS Algorithm - Hubs and Authorities on the Internet](http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture4/lecture4.html)
* http://langvillea.people.cofc.edu/
* [Google PageRank: The Mathematics of Google](http://www.whydomath.org/node/google/index.html)
* [How Google Finds Your Needle in the Web's Haystack](http://www.ams.org/publicoutreach/feature-column/fcarc-pagerank)
* [Dynamic PageRank](http://ryanrossi.com/dynamic-pagerank.php)

***
* [relevant search](http://manning.com/books/relevant-search)
* [Learning to rank plugin of Elasticsearch](https://github.com/o19s/elasticsearch-learning-to-rank)
* http://mlwiki.org/index.php/Information_Retrieval_(UFRT)
* https://en.wikipedia.org/wiki/List_of_search_engines
* [Open Semantic Search](https://www.opensemanticsearch.org/)
* https://www.seekquarry.com/
* http://l-lists.com/en/lists/qukoen.html
* [20款开源搜索引擎介绍与比较](https://blog.csdn.net/belalds/article/details/80758312)
* [gt4ireval: Generalizability Theory for Information Retrieval Evaluation](https://rdrr.io/cran/gt4ireval/)
* https://daiwk.github.io/posts/nlp.html

#### Comparison and Matching

`Query and Indexed Object` is similar with `Question and Answers`.
The user requested a query then a matched response is supposed to match the query in semantics. Before that we must understand the query.


<img title = "search process" src = "https://ekanou.github.io/dynamicsearch/DynSe2018.png" width="80%" />

* [Deep Semantic Similarity Model](https://www.microsoft.com/en-us/research/project/dssm/)
* [AI in Information Retrieval and Language Processing collected by Wlodzislaw Duch](http://www.is.umk.pl/~duch/IR.html)
* [Deep Learning for Information Retrieval](https://pangolulu.github.io/2016/10/28/deep-ir/)
* [A Deep Relevance Matching Model for Ad-hoc Retrieval](https://arxiv.org/abs/1711.08611)
* [Relevance Matching](https://zhuanlan.zhihu.com/p/39946041)
* https://ekanou.github.io/dynamicsearch/
* http://mlwiki.org/index.php/NLP_Pipeline

### Information Distribution: Search Engine Results Page


[Information Distribution Methods – Information distribution is the timely collection, sharing and distribution of information to the project team. Methods can be portals, collaborative work management tools, web conferencing, web publishing, and when all technology is not available, manual filing systems and hard copy distribution.](http://www.anticlue.net/archives/000804.htm)

* [SERP: GUIDE TO THE GOOGLE SEARCH ENGINE RESULTS (UPDATED 2019 GUIDE)](https://ignitevisibility.com/serp/)

### Personalized Search

[Personalized Search fetches results and delivers search suggestions individually for each of its users based on their interests and preferences](https://yandex.com/company/technologies/personalised_search/), which is mined from the information that the search engine has about the user at the given time, such as their location, search history, demographics such as the recommenders.

And here search engine and recommender system coincide except the recommender system push some items in order to attract the users' attention while search engine recall the information that the users desire in their mind.

* http://ryanrossi.com/search.php
* https://a9.com/what-we-do/product-search.html
* https://www.algolia.com/
* https://www.cognik.net/
* http://www.collarity.com/
* https://www.wikiwand.com/en/Personalized_search
* [The Mathematics of Web Search](http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/index.html)
* [CSAW: Curating and Searching the Annotated Web](https://www.cse.iitb.ac.in/~soumen/doc/CSAW/)
* [A Gradient-based Framework for Personalization by Liangjie Hong](http://www.hongliangjie.com/talks/Gradient_Indiana_2017-11-10.pdf)
* [Style in the Long Tail: Discovering Unique Interests with Latent Variable Models in Large Scale Social E-commerce](https://mimno.infosci.cornell.edu/info6150/readings/p1640-hu.pdf)
* [Personalized Search in Yandex](https://yandex.com/company/technologies/personalised_search/)
* [Thoughts on Yandex personalized search and beyond](https://www.russiansearchtips.com/2012/12/thoughts-on-yandex-personalized-search-and-beyond/)
* [Yandex filters & algorithms. 1997-2018](https://www.konstantinkanin.com/en/yandex-algorithms/)
* [Google's Personalized Search Explained: How personalization works](https://www.link-assistant.com/news/personalized-search.html)
* [A Better Understanding of Personalized Search](https://www.briggsby.com/better-understanding-personalized-search)
* [Interest-Based Personalized Search](https://www.cpp.edu/~zma/research/Interest-Based%20Personalized%20Search.pdf)
* [Search Personalization using Machine Learning by Hema Yoganarasimhan](https://faculty.washington.edu/hemay/search_personalization.pdf)
* [Web Personalization and Recommender Systems](https://www.kdd.org/kdd2015/slides/KDD-tut.pdf)
* [Scaling Concurrency of Personalized Semantic Search over Large RDF Data](https://research.csc.ncsu.edu/coul/Pub/BigD402.pdf)
* [Behavior‐based personalization in web search](https://onlinelibrary.wiley.com/doi/full/10.1002/asi.23735)

### Vertical Domain Search

* [Vertical search](https://en.wikipedia.org/wiki/Vertical_search)
* [Consumer Health Search](https://ielab.io/projects/consumer-health-search.html)
* [Biomedical Data Science Initiative](http://med.stanford.edu/bdsi.html)
* https://clefehealth.imag.fr/
* http://www.bilegaldoc.com/
* [自制AI图像搜索引擎](https://blog.csdn.net/baidu_40840693/article/details/88230418)
* [Building a Content-Based Multimedia Search Engine I: Quantifying Similarity](http://www.deepideas.net/building-content-based-multimedia-search-engine-quantifying-similarity/)
* [Building a Content-Based Multimedia Search Engine II: Extracting Feature Vectors](http://www.deepideas.net/building-content-based-multimedia-search-engine-feature-extraction/)
* [Building a Content-Based Multimedia Search Engine III: Feature Signatures](http://www.deepideas.net/building-content-based-multimedia-search-engine-feature-signatures/)
* [Building a Content-Based Multimedia Search Engine IV: Earth Mover’s Distance](http://www.deepideas.net/building-content-based-multimedia-search-engine-earth-movers-distance/)
* [Building a Content-Based Multimedia Search Engine V: Signature Quadratic Form Distance](http://www.deepideas.net/building-content-based-multimedia-search-engine-signature-quadratic-form-distance/)
* [Building a Content-Based Multimedia Search Engine VI: Efficient Query Processing](http://www.deepideas.net/building-content-based-multimedia-search-engine-efficient-query-processing/)

### Labs and Resources  

#### Labs on Search and Information Retrieval

+ [Search and information retrieval@Microsoft](https://www.microsoft.com/en-us/research/research-area/search-information-retrieval/)
+ [Search and information retrieval@Google](https://ai.google/research/pubs/?area=InformationRetrievalandtheWeb)
+ [Web search and mining @Yandex](https://research.yandex.com/publications?themeSlug=web-mining-and-search)
+ [The Information Engineering Lab](https://ielab.io/)
+ [Information Retrieval Lab: A research group @ University of A Coruña (Spain)](https://www.irlab.org/)
+ [BCS-IRSG: Information Retrieval Specialist Group](https://irsg.bcs.org/)
+ [智能技术与系统国家重点实验室信息检索课题组](http://www.thuir.org/)
+ [The Cochrane Information Retrieval Methods Group (Cochrane IRMG)](https://methods.cochrane.org/irmg/)
+ [SOCIETY OF INFORMATION RETRIEVAL & KNOWLEDGE MANAGEMENT (MALAYSIA)](http://pecamp.org/web14/)
+ [Quantum Information Access and Retrieval Theory)](https://www.quartz-itn.eu/)
+ [Center for Intelligent Information Retrieval (CIIR)](http://ciir.cs.umass.edu/)
+ [InfoSeeking Lab situated in School of Communication & Information at Rutgers University.](https://infoseeking.org/)
+ http://nlp.uned.es/web-nlp/
+ http://mlwiki.org/index.php/Information_Retrieval
+ [information and language processing systems](https://ilps.science.uva.nl/)
+ [information retrieval facility](https://www.ir-facility.org/)
+ [Center for Information and Language Processing](https://www.cis.uni-muenchen.de/)
+ [Summarized Research in Information Retrieval for HTA](http://vortal.htai.org/?q=sure-info)
+ [IR Lab](http://www.ir.disco.unimib.it/)
+ [IR and NLP Lab](https://ir.kaist.ac.kr/about/)
+ [CLEF 2018 LAB](https://ekanou.github.io/dynamicsearch/clef-lab.html)
+ [QANTA: Question Answering is Not a Trivial Activity](https://sites.google.com/view/qanta/home)
+ [SIGIR](https://sigir.org/)
+ http://cistern.cis.lmu.de/
+ http://hpc.isti.cnr.it/
+ http://trec-car.cs.unh.edu/
+ [LETOR: Learning to Rank for Information Retrieval](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval)
+ [A Lucene toolkit for replicable information retrieval research ](https://github.com/castorini/anserini)
+ [Information Overload Research Group](https://iorgforum.org/)
+ [National Information Standards Organization by ANSI](https://www.niso.org/niso-io)
+ [International Society  of Music Information Retrieval](https://ismir.net/)
+ [OpenClinical information retrieval](http://www.openclinical.org/informationretrieval.html)

#### Conference on Information Retrieval

+ https://datanatives.io/conference/
+ [HE 3RD STRATEGIC WORKSHOP ON INFORMATION RETRIEVAL IN LORNE (SWIRL)](https://sites.google.com/view/swirl3/home)
+ [Text Retrieval COnference(TREC)](https://trec.nist.gov/)
+ [European Conference on Information Retrieval (ECIR 2018)](https://www.ecir2018.org/)
+ [ECIR 2019](http://ecir2019.org/workshops/)
+ [IR @wikiwand](https://www.wikiwand.com/en/Information_retrieval)
+ [Algorithm Selection and Meta-Learning in Information Retrieval (AMIR)](http://amir-workshop.org/)
+ [The ACM SIGIR International Conference on the Theory of Information Retrieval (ICTIR)2019](http://www.ictir2019.org/)
+ [KDIR 2019](http://www.kdir.ic3k.org/)
+ [Advances in Semantic Information Retrieval (ASIR’19)](https://fedcsis.org/2019/asir)
+ [Music Information Retrieval Evaluation eXchange (MIREX 2019)](https://www.music-ir.org/mirex/wiki/MIREX_HOME)
+ [20th annual conference of the International Society for Music Information Retrieval (ISMIR)](https://ismir2019.ewi.tudelft.nl/)
+ [8th International Workshop on Bibliometric-enhanced Information Retrieval](http://ceur-ws.org/Vol-2345/)
+ [ICMR 2019](http://www.icmr2019.org/)
+ [3rd International Conference on Natural Language Processing and Information Retrieval](http://www.nlpir.net/)
+ [FACTS-IR Workshop @ SIGIR 2019](https://fate-events.github.io/facts-ir/)
+ [ACM Conference of Web Search and Data Mining 2019](http://www.wsdm-conference.org/2019/)
+ [SMIR 2014](http://smir2014.noahlab.com.hk/SMIR2014.htm)
+ [2018 PRS WORKSHOP:  Personalization, Recommendation and Search (PRS)](https://prs2018.splashthat.com/)
+ [Neu-IR: The SIGIR 2016 Workshop on Neural Information Retrieval](https://www.microsoft.com/en-us/research/event/neuir2016/)
+ [Neu-IR 2017: Workshop on Neural Information Retrieval](https://neu-ir.weebly.com/)
+ [NeuIR Group](http://neuir.org/)
+ [TREC 2019 Fair Ranking Track](https://fair-trec.github.io/)
+ [LEARNING FROM LIMITED OR NOISY DATA
FOR INFORMATION RETRIEVAL](https://lnd4ir.github.io/)
+ [DYNAMIC SEARCH: Develop algorithms and evaluation methodologies with the user in the search loop](https://ekanou.github.io/dynamicsearch/)
+ [European Summer School in Information Retrieval ’15](http://www.rybak.io/european-summer-school-in-information-retrieval-15/)

#### Cources on Information Retrieval and Search

+ [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)
+ [CS 371R: Information Retrieval and Web Search](https://www.cs.utexas.edu/~mooney/ir-course/)
+ [CS 242: Information Retrieval & Web Search, Winter 2019](http://www.cs.ucr.edu/~vagelis/classes/CS242/index.htm)
+ [Winter 2017 CS293S: Information Retrieval and Web Search](https://sites.cs.ucsb.edu/~tyang/class/293S17/)
+ [CS 276 / LING 286: Information Retrieval and Web Search](https://web.stanford.edu/class/cs276/)
+ [Information Retrieval and Web Search 2015](http://web.eecs.umich.edu/~mihalcea/498IR/)
+ [Data and Web Mining](http://www.dsi.unive.it/~dm/)
+ [Neural Networks for Information Retrieval](http://www.nn4ir.com)
+ [Introduction to Search Engine Theory](http://ryanrossi.com/search.php)
+ [INFORMATION RETRIEVAL FOR GOOD](http://romip.ru/russir2018/)
+ [Search user interfaces](http://searchuserinterfaces.com/book/)
+ [Morden Information Retrieval](http://grupoweb.upf.edu/mir2ed/home.php)
+ [Search Engine: Information Retrieval in Practice](http://www.search-engines-book.com/)
+ [Information Retrieval  潘微科](http://csse.szu.edu.cn/csse.szu.edu.cn/staff/panwk/IR201702/index.html)
+ [Information Organization and Retrieval: INFO 202](http://courses.ischool.berkeley.edu/i202/f10/)
+ [Music Information Retrieval @ NYU](http://www.nyu.edu/classes/bello/MIR.html)
