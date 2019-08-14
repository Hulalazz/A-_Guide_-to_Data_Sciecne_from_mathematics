## IR and Search

- [ ] [RISE: Repository of Online Information Sources Used in Information Extraction Tasks](https://www.isi.edu/info-agents/RISE/)
- [ ] [AI in Information Retrieval and Language Processing collected by Wlodzislaw Duch](http://www.is.umk.pl/~duch/IR.html)
- [ ] [Topics in Natural Language Processing (202-2-5381) Fall 2018](https://www.cs.bgu.ac.il/~elhadad/nlp18.html)
- [ ] [CS 124: From Languages to Information Winter 2019 Dan Jurafsky](https://web.stanford.edu/class/cs124/)
- [ ] [CS224U: Natural Language Understanding](https://web.stanford.edu/class/cs224u/)
- [ ] [CS224n: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
- [ ] [Marti A. Hearst](http://people.ischool.berkeley.edu/~hearst/teaching.html)
- [ ] [Applied Natural Language Processing](https://bcourses.berkeley.edu/courses/1453620/assignments/syllabus)
- [ ] [Synthesis Lectures on Information Concepts, Retrieval, and Services](https://www.morganclaypool.com/toc/icr/1/1)
- [ ] [Terrier IR Platform](http://terrier.org/)
- [ ] https://ntent.com/, https://www.clearquery.io/how, https://www.searchhub.io/

If the recommender system is to solve the information overload problem personally, information retrieval and search technology  is to solve that problem generally at the web-scale.
[Technically, IR studies the acquisition, organization, storage, retrieval, and distribution of information.](http://www.dsi.unive.it/~dm/Slides/5_info-retrieval.pdf)
Information is in diverse format or form, such as character strings(texts), images, voices and videos so that information retrieval has diverse subfields such as [multimedia information retrieval](http://press.liacs.nl/mlew/mir2019.html) and [music information retrival](https://musicinformationretrieval.com/index.html). Search engine is considered as a practical application of information retrieval.  

Critical to all search engines is the problem of designing an effective retrieval model that can rank documents accurately for a given query.
A main goal of any IR system is to rank documents optimally given a query so that a highly relevant documents would be ranked above less relevant ones and nonrelevant ones.
`Relevance`, `Ranking`  and `Context`  are three foundation stones of web search. In this section, we focus on relevance more than rank.

If interested in the history of information retrieval, Mark Sanderson and W. Bruce Croft wrote a paper for [The History of Information Retrieval Research](https://ciir-publications.cs.umass.edu/pub/web/getpdf.php?id=1066).

[The basic functions of a search engine can be described as _crawling, data mining, indexing and query processing_. `Crawling` is the act of sending small programed bots out to collect information. `Data mining` is storing the information collected by the bots. `Indexing` is ordering the information systematically. And `query processing` is the mathematical process in which a person's query is compared to the index and the results are presented to that person.](https://lifepacific.libguides.com/c.php?g=155121&p=1018180)

<img title="IR Process" src="https://hsto.org/files/4b9/a9b/1a6/4b9a9b1a60d041b2b4dfeca4b7989586.png" width="50%" />

One of the most fundamental and important challenges is to develop a truly optimal retrieval model that is both effective and efficient
and that can learn form the feedback information over time, which will be talked in `Rating and Ranking`.

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
* http://bit.csc.lsu.edu/~kraft/retrieval.html

### Information Acquisition: Web Crawling

The first step of information retrieval is to acquise the information itself. The web-scale information brings information overload problem, which `search  engine` or  `web search` attempts to solve.  

[Web Crawling By Christopher Olston and Marc Najork](http://infolab.stanford.edu/~olston/publications/crawling_survey.pdf)
> A web crawler (also known as a robot or a spider) is a system for the bulk downloading of web pages.  Web crawlers are used for a variety of purposes.  Most prominently, they are one of the main components of web search engines, systems that assemble a corpus of web pages, index them, and allow users to issue queries against the index and find the web pages that match the queries.  A related use is web archiving (a service provided by e.g., the Internet archive), where large sets of web pages are periodically collected and archived for posterity. A third use is web data mining, where web pages are analyzed for statistical properties, or where data analytics is performed on them (an example would be Attributor, a company that monitors the web for copyright and trademark infringements). Finally, web monitoring services allow their clients to submit standing queries, or triggers, and they continuously crawl the web and notify clients of pages that match those queries (an example would be GigaAlert).

* https://iorgforum.org/
* http://facweb.cs.depaul.edu/mobasher/classes/ect584/
* https://apify.com/
* [Web Crawling By Christopher Olston and Marc Najork](http://infolab.stanford.edu/~olston/publications/crawling_survey.pdf)
* [VII. Information Acquisition](www.science.unitn.it/~pgiorgio/db2/slides/9-infoacquisition.pdf)
* [Automatically modelling and distilling knowledge within AI!](https://ai-distillery.io/)
* [CH. 3: MODELS OF THE INFORMATION SEEKING PROCESS](http://searchuserinterfaces.com/book/sui_ch3_models_of_information_seeking.html)
* https://zhuanlan.zhihu.com/p/70169130

### Information Organization and Storage: Indexing and Index

Index as data structure is to organize the information efficiently in order to search some specific terms.

First, let us consider the case where we do not remember some key terms as reading some references, the appendices may include index recording the places where the terms firstly appear such as the following images shown.

<img src="http://www.kfzimg.com/G06/M00/8F/16/p4YBAFsp_5mAHTEFAAY9LXEBT0k044_b.jpg" width="50%" />

Search engine takes advantage of this idea: it is the best place to store  where the terms/words appear in key-value format where the key, values is the terms and their places, respectively.

#### Index Creation

#### Index Compression

* [Elasticsearch from the Bottom Up, Part 1](https://www.elastic.co/blog/found-elasticsearch-from-the-bottom-up)
* [Intellectual Foundations for Information Organization and Information](http://people.ischool.berkeley.edu/~glushko/IFIOIR/)
* [Inverted Index versus Forward Index](http://www.darwinbiler.com/inverted-index-vs-forward-index/)
* http://planet.botany.uwc.ac.za/nisl/GIS/GIS_primer/index.htm


### Information Retrieval

<img title = "search process" src = "http://www.searchtools.com/slides/images/search-process.gif" width="50%" />

- [Regular Expressions, Text Normalization, Edit Distance](http://web.stanford.edu/~jurafsky/slp3/2.pdf)
- [Introduction to ](https://spark-public.s3.amazonaws.com/cs124/slides/ir-1.pdf)
- [solr-vs-elasticsearch](https://solr-vs-elasticsearch.com/)
- [CH. 4: QUERY SPECIFICATION](http://searchuserinterfaces.com/book/sui_ch4_query_specification.html)
- https://homepages.dcc.ufmg.br/~rodrygo/rm-2019-2/



#### Query Languages

* [Query Languages](http://www.site.uottawa.ca/~diana/csi4107/L5.pdf)
* http://www.odbms.org/

##### Boolean Queries

Keywords combined with Boolean operators: OR AND BUT

##### Phrasal Queries

Retrieve documents with a specific phrase (ordered list of contiguous words)


##### Proximity Queries

List of words with specific maximal distance constraints between terms
Example: “dogs” and “race” within 4 words match “…dogs will begin the race…”

##### Pattern Matching

Allow queries that match strings rather than word tokens.
Requires more sophisticated data structures and algorithms than inverted indices to retrieve efficiently.

**Edit (Levenstein) Distance** is defined as minimum number of character `deletions, additions, or replacements` needed to make two strings equivalent.

- [Minimum	Edit	Distance](https://web.stanford.edu/class/cs124/lec/med.pdf)

**Longest Common Subsequence (LCS)** is the length of the longest subsequence of characters shared by two strings

##### Regular Expressions

Language for composing complex patterns from simpler ones: `Union, Concatenation, Repetition`.

##### Structural Queries

Assumes documents have structure that can be exploited in search, allow queries for text appearing in specific fields.


#### Query Parser: Query Understanding

Query is often some keywords in natural language such as English or Chinese. We use the search engine when we would like to find some information related with the keywords on the web/internet, which means  we do not completely know what the result is. Additionally, all information is digitalized in computer and the computers do not understand the natural language natively.
For example, `synonyms` are different as character or string data structure in computer.
Natural language processing(NLP) or natural language understanding(NLU)  facilitate the computers to comprehend the query.


* [Query Understanding](https://github.com/sanazb/Query-Understanding)
* [Exploring Query Parsers](https://lucidworks.com/post/exploring-query-parsers/)
* [Query Understanding: An efficient way how to deal with long tail queries](https://www.luigisbox.com/blog/query-understanding/)
* [The Art of Tokenization](https://www.ibm.com/developerworks/community/blogs/nlp/entry/tokenization?lang=en)

<img src="https://ntent.com/wp-content/uploads/2017/01/Query-Understanding2.jpg" width="60%" />


Response | Time|
---|---|
[Query Auto Completion](https://www.jianshu.com/p/c7bc74d3657d)| Before the query input is finished|
[Spelling Correction](https://nlp.stanford.edu/IR-book/html/htmledition/spelling-correction-1.html)| When the query input is finished|
[Semantic Analysis](https://quanteda.io/articles/pkgdown/examples/lsa.html)| After the query input is finished|
[Query Suggestion](https://zhuanlan.zhihu.com/p/23693891)| After the query input is finished|
[Intention Analysis](https://aiaioo.wordpress.com/tag/intention-analysis/) |  After the query input|

* http://partofspeech.org/
* https://nlpprogress.com/

##### Query Operations


- Query Reformulation:
  * Query Expansion: Add new terms to query from relevant documents.
  * Term Reweighting: Increase weight of terms in relevant documents and decrease weight of terms in irrelevant documents.
+ https://www.cs.bgu.ac.il/~elhadad/nlp18.html
+ [CH. 6: QUERY REFORMULATION](http://searchuserinterfaces.com/book/sui_ch6_reformulation.html)
+ [Relevance feedback and query expansion](https://nlp.stanford.edu/IR-book/html/htmledition/relevance-feedback-and-query-expansion-1.html)

**Standard Rochio Method**

- https://nlp.stanford.edu/IR-book/html/htmledition/rocchio-classification-1.html
- http://www.cs.cmu.edu/~wcohen/10-605/rocchio.pdf

**Ide Regular Method**

- https://cs.brynmawr.edu/Courses/cs380/fall2006/Class13.pdf
- http://www1.se.cuhk.edu.hk/~seem5680/lecture/rel-feed-query-exp-2016.pdf
- http://web.eecs.umich.edu/~mihalcea/courses/EECS486/Lectures/RelevanceFeedback.pdf
- https://researchbank.rmit.edu.au/eserv/rmit:9503/Billerbeck.pdf

##### Query Auto Completion

The auto complete is a drop-down list populated with suggestions of what one can write in the search box.

The auto complete is a list of suggestions of what one can write in the search box to reach different products or categories.
These suggestions will also be referred to as query suggestions or completions.
After one has written a few letters of the beginning of the query and the list is populated with query suggestions
that in some way match the input. In the normal case matching means that the suggestion starts with the input.


- [Design and Implementation of an Auto Complete Algorithm for E-Commerce](https://www.eit.lth.se/sprapport.php?uid=454)
- https://www.jianshu.com/p/c7bc74d3657d
- https://blog.floydhub.com/gpt2/

##### Spelling Correction

For simplicity let us first consider correction of individual misspelled words (e.g., “elefnat” to “elephant”).
One simple approach to spelling error correction is to **calculate the edit distance between the query word and each of the dictionary words**.
Dictionary words within a fixed range of edit distance or a variable range of edit distance depending on word length are selected as candidates for correction.
There are at least two drawbacks for this approach, however.
First, probabilities of word usages as well as word misspellings are not considered in the model.
Second, context information of correction is not taken into consideration.

To address the issues, probabilistic approaches, both generative approach and discriminative approach, have been proposed.
Suppose that the query word is represented as $q$ and a correction is represented as $c$.
We want to **find the correction $\hat{c}$ having the largest conditional probability $P(c|q)$**.
Different ways of defining the model lead to different methods.

By Bayes’ rule, we can consider finding the correction $\hat c$ having the
largest product of probability $P(c)$ and conditional probability $P(q|c)$
$$\hat c=\arg\max_{c} P(c\mid q)=\arg\max_{c}P(c)P(q\mid c).$$
The former is called source model and the latter channel model.

The source model can be trained by using the document collection and/or search log.
(Due to the wide variety of searches it is better to find the legitimate words from data.)
A straightforward way would be to estimate the probabilities of words based on their occurrences in the dataset with a smoothing technique applied.
The channel model can be defined based on weighted edit distance,
where the model is usually trained by using data consisting of pairs of correct word and misspelled word.

- [Spelling correction](https://nlp.stanford.edu/IR-book/html/htmledition/spelling-correction-1.html)
- [How to Write a Spelling Corrector](http://norvig.com/spell-correct.html)
- [Spelling Correction and the Noisy Channel](https://web.stanford.edu/~jurafsky/slp3/B.pdf)

##### Query Suggestion

When a user provides a root input, such as a search query, these algorithms dynamically retrieve, curate, and present a list of related inputs, such
as search suggestions.
Although ubiquitous in online platforms, a lack of research addressing the ephemerality of their outputs
and the opacity of their functioning raises concerns of transparency and accountability on where inquiry is steered.

<img src="https://github.com/syw2014/query-suggestion/blob/master/doc/qs_workflow.png" width="50%" />

- [How Google Instant’s Autocomplete Suggestions Work](https://searchengineland.com/how-google-instant-autocomplete-suggestions-work-62592)
- [Visual Query Suggestion](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/fp12729-zha.pdf)
- [Query Suggestion @https://www.algolia.com](https://www.algolia.com/doc/guides/getting-insights-and-analytics/leveraging-analytics-data/query-suggestions/)
- [Incremental Algorithms for Effective and Efficient Query Recommendation](http://ir.cs.georgetown.edu/downloads/spire2010broccolo.pdf)
- [Auditing Autocomplete: Suggestion Networks and Recursive Algorithm Interrogation](https://www.shanjiang.me/publications/websci19_paper.pdf)
- https://zhuanlan.zhihu.com/p/23693891
- https://github.com/syw2014/query-suggestion
- https://www.cnblogs.com/wangzhuxing/p/9574630.html
- https://elasticsearch-py.readthedocs.io/en/master/api.html
- https://elasticsearch.cn/article/142

##### Query Expansion

Query expansion is a technique studied intensively and widely in IR.
The basic idea is to enrich the query with additional terms (words or phrases) and to use the expanded query to conduct search in order to circumvent the query-document mismatch challenge.

<img src="https://img-blog.csdn.net/20160611222923822" width="80%" />

+ https://blog.csdn.net/baimafujinji/article/details/50930260
+ https://www.wikiwand.com/en/Query_expansion
+ https://nlp.stanford.edu/IR-book/html/htmledition/query-expansion-1.html
+ https://dev.mysql.com/doc/refman/5.5/en/fulltext-query-expansion.html
+ [Neural Query Expansion for Code Search](https://pldi19.sigplan.org/details/mapl-2019-papers/4/Neural-Query-Expansion-for-Code-Search)

##### Query Relaxation

- http://pike.psu.edu/publications/dongwon-dissertation.pdf

##### Query Segmentation

Query segmentation is to separate the input query into multiple segments, roughly corresponding to natural language phrases, for improving search relevance.

+ https://arxiv.org/abs/1707.07835
+ http://ra.ethz.ch/CDstore/www2011/proceedings/p97.pdf
+ https://github.com/kchro/query-segmenter

##### Query Scoping



##### Query Understanding


Query understanding: query normalization (encoding, tokenization, spelling); query rewriting (expansion, relaxation, segmentation, scoping)


[Levels of Query Understanding](https://ntent.com/technology/query-understanding/#levels):

NTENT’s Search platform choreographs the interpretation of singular query constituents, and the dissemination of relevant answers through a specialized combination of Language Detection, Linguistic Processing, Semantic Processing and Pragmatic Processing.

* Language Detection: The first step is to understand which language the user is using. Sometimes this is obvious, but many things can make this hard. For example, many users find themselves forced to use a keyboard that makes it hard to use accented characters, so they “ascify” their query. Users sometimes use multiple languages within a single query (“code switching”) or proper names that are the same in many languages.
* Linguistic Processing: Every language has its own rules for how text should be broken down into individual words (“tokenized”), whether distinctions of case and accent are significant, how to normalize words to a base form (“lemmatization” or “stemming”), and categorization of words and phrases by parts of speech (“POS tagging”).
* Semantic Processing: A traditional keyword search engine would stop after linguistic processing, but NTENT’s technology goes further, and determines what the user’s words actually mean. Many words have multiple meanings (“homonyms”), and many concepts have multiple ways to express them (“synonyms”). Drawing on many sources of information, such as a large-scale ontology, notability data, and the user’s context (e.g., location), we are able to determine all the possible interpretations of the user’s query, and assign a probability to each one. By distinguishing a particular sense of a word, and by knowing which phrases denote a single concept, we are able to improve the precision of our applications. At the same time, by recognizing that multiple expressions refer to the same concept, and also that broad terms encompass narrower ones (“hyponymy”), we are able to improve recall. Furthermore, NTENT is able to analyze the syntax of how multiple words are combined into composite concepts.
* Intent Detection (Pragmatic Processing): NTENT goes beyond just the surface semantics of the user’s utterance, and develops hypotheses about why they typed what they did: what their information need is; what transactions they intend to perform; what website they’re looking for; or what local facilities they’re trying to find. This inductive reasoning is key to harnessing NTENT’s extensive set of experts to give the user what they want.

|[NLP Pipeline of Query Understanding](http://mlwiki.org/index.php/NLP_Pipeline)|
|---|
|[Tokenization](http://mlwiki.org/index.php/Tokenization)|
|[Stop words removal](http://mlwiki.org/index.php/Stop_Words)|
|[Text Normalization](http://mlwiki.org/index.php/Text_Normalization)|
|[POS Tagging](http://nlpprogress.com/english/part-of-speech_tagging.html)|
|[Named Entity Recogition](https://cs230-stanford.github.io/pytorch-nlp.html)|

* [Query Understanding for Search on All Devices](https://www.wsdm-conference.org/2016/workshops.html)
* https://sites.google.com/site/queryunderstanding/
* https://ntent.com/technology/query-understanding/
* [查询理解(Query Understanding)—查询改写总结](http://www.zhongruitech.com/956268106.html)
* https://www.wikiwand.com/en/Query_understanding
* https://www.luigisbox.com/blog/query-understanding/
* https://github.com/DataEngg/Query-Understanding
* https://docs.microsoft.com/en-us/sharepoint/dev/general-development/customizing-ranking-models-to-improve-relevance-in-sharepoint

##### Intention Analysis

[Intent Analysis goes a level deeper than sentiment analysis and gives an idea of whether a string of text is a complaint, a suggestion or a query.Gauging the intent of messages on social media opens a lot of new possibilities.
It uses Long Short Term Memory (LSTM) algorithms to classify a text into different. LSTMs model sentences as chain of forget-remember decisions based on context. It is trained on social media data and news data differently for handling casual and formal language. We also have trained this algorithm for various custom datasets for different clients.](https://www.paralleldots.com/intent-analysis)

- https://www.paralleldots.com/intent-analysis
- https://aiaioo.wordpress.com/tag/intention-analysis/
-----
#### Relevance and Rank

Recall the definition of  `Discounted Cumulative Gain(DCG)`:

$${DCG}_p= \sum_{i=1}^{p} \frac{{rel}_i}{\log_{2}(i+1)}$$

where ${rel}_{i}$ is the relevance of the document and query.

However, it is discussed how to compute the relevance of the document and query. The document is always text such as html file so natural language processing plays a lead role in computing the relevances.
For other types information retrieval system, it is different to compute the relevance. For example, imagine  search engine is to find and return the images similar on the internet  with the given image query, where the information is almost in pixel format rather than text/string.



---|[A part of Ranking Model]( https://homepages.dcc.ufmg.br/~rodrygo/rm-2018-2/)
----|----
Query-independent ranking| on-document evidence (retrievability, readability, maliciousness); off-document evidence (centrality, popularity, credibility)
Query understanding| query normalization (encoding, tokenization, spelling); query rewriting (expansion, relaxation, segmentation, scoping)
Query-dependent ranking| basic models (algebraic models, probabilistic models, information-theoretic models); proximity models (Markov random fields models); structural models (field-based models); semantic models (latent semantics, explicit semantics)
 Contextual ranking| personalization; diversification; interactivity
Machine-learned ranking| query-document representation; loss functions (pointwise, pairwise, listwise loss); optimization strategies; adaptation strategies (intent-awareness, exploration-exploitation)
Ranking evaluation| behavioral models; evaluation design; evaluation metrics; offline evaluation; online evaluation

The `Machine-learned ranking` and `Ranking evaluation` is discussed in `Rating and Ranking`.

+ https://homepages.dcc.ufmg.br/~rodrygo/rm-2018-2/

|Features/Attributes for ranking|
|---|
|Average Absolute Query Frequency|
|Query Length|
|Average Absolute Document Frequency|
|Document Length|
|Average Inverse Document Frequency|
|Number of Terms in common between query and document|

#### Query-dependent ranking

Query-dependent ranking:
* basic models (algebraic models, probabilistic models, information-theoretic models);
* proximity models (Markov random fields models); structural models (field-based models);
* semantic models (latent semantics, explicit semantics).

- https://en.wikipedia.org/wiki/Ranking
- https://andrewoarnold.com/fp025-geng.pdf
- https://sites.google.com/site/kimducthanh/publication/lrt-queryindependent.pdf
- [Query-Independent Ranking for Large-Scale Persistent Search Systems](https://www.cs.princeton.edu/research/techreps/TR-837-08)

##### TF-IDF

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
$$\text{tf-idf}=tf(w| doc)\times idf(w\mid D).$$

- [tf-idf	weighting	has	many variants](https://spark-public.s3.amazonaws.com/cs124/slides/ir-2.pdf)

##### Robertson-SparckJones Model

**The goal of a probabilistic retrieval model is clearly to retrieve the documents with the highest probability of relevance to the given query.**

Three random variables- the query $Q$, the document $D$ and the relevance $R \in\{0,1\}$.
The goal is to estimate the rank of $D$ based on $P(R=1|Q,D)$.

The basic idea is to compute $Odd(R=1|Q,D)$ using Bayes’ rule
$$Odd(R=1\mid Q, D)=\frac{P(R=1\mid Q, D)}{P(R=0\mid Q, D)}=\frac{P(Q, D\mid R=1)}{P(Q, D\mid R=0)}\frac{P(R=1)}{P(R=0)}.$$

- http://www.cs.cornell.edu/home/llee/papers/idf.pdf
- http://www.minerazzi.com/tutorials/probabilistic-model-tutorial.pdf
- http://www.cs.cornell.edu/courses/cs6740/2010sp/guides/lec05.pdf

##### BM25

The basic idea of `BM25` is to rank documents by the log-odds of their relevance.
Actually `BM25` is not a single model, but defines a whole family of ranking models,
with slightly different components and parameters.
One of the popular instantiations of the model is as follows.

Given a query $q$, containing terms $t_1,\cdots , t_M,$ the BM25 score of a document $d$
is computed as
$$
BM25(d, q)=\sum_{i=1}^{M}\frac{IDF(t_i)\cdot TF(t_i, d)\cdot (k_1 + 1)}{TF(t_i, d)+k_1 \cdot (1-b+b\cdot \frac{LEN(d)}{avdl})}
$$

where $TF(t, d)$ is the term frequency of the $t$ th in the document $d$, $LEN(d)$ is the length(number of words) of document $d$, and $avdl$ is the average document length in the text collection from which document are drawn. $k_1$ and $b$ are free parameters, $IDF(t)$ is the **IDF** weight of the term $t$, computed by $IDF(t)=\log(\frac{N}{n(t)})$ where $N$ is the total number of documents in the collection, and $n(t)$ is the number
of documents containing term $t$ .

* [BM25 The Next Generation of Lucene Relevance](https://opensourceconnections.com/blog/2015/10/16/bm25-the-next-generation-of-lucene-relevation/)
* [Probabilistic IR](http://sifaka.cs.uiuc.edu/czhai/pub/lmir2003-probir.pdf)

##### The language model for information retrieval (LMIR)

`The language model for information retrieval (LMIR)` is an application of the statistical language model on information
retrieval. A statistical language model assigns a probability to a sequence of terms.
When used in information retrieval, a language model is associated with a document.

With query $q$ as input, documents are ranked based on the query likelihood, or the probability that the document’s language model will generate the terms in the query(i.e., $P(q\mid d)$).
By further assuming the independence between terms, one has
$$P(q\mid d)=\prod_{i=1}^{M}P(t_i\mid d)$$
if query $q$ contains terms $t_1,\dots, t_M$.

To learn the document’s language model, a maximum likelihood method is used.
As in many maximum likelihood methods, the issue of smoothing the estimate is critical. Usually a background language model estimated using the entire collection is used for this purpose.
Then, the document’s language model can be constructed
as follows:
$$p(t_i\mid d)=(1-\lambda)\frac{TF(t_i, d)}{LEN(d)}+\lambda p(t_i\mid C)$$
where $p(t_i\mid C)$ is the background language model for term $t_i$, and $\lambda \in[0, 1]$ is a smoothing factor.

##### TextRank

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

##### Text Summarization

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

##### Document Similarity

Document similarity (or distance between documents) is a one of the central themes in Information Retrieval. How humans usually define how similar are documents? Usually documents treated as similar if they are semantically close and describe similar concepts.
`w-shingling`

- [ ] [Document Similarity in Machine Learning Text Analysis with ELMo](https://ai.intelligentonlinetools.com/ml/document-similarity-in-machine-learning-text-analysis-with-elmo/)
- [ ] [Documents similarity](http://text2vec.org/similarity.html)
- [ ] https://copyleaks.com/
- [ ] https://www.wikiwand.com/en/Semantic_similarity
- [ ] https://spacy.io/
- [ ] https://fasttext.cc/

#### Comparison and Matching

`Query and Indexed Object` is similar with `Question and Answers`.
The user requested a query then a matched response is supposed to match the query in semantics. Before that we must understand the query.

The most common way to model similarity is by means of a distance function.
A distance function assigns high values to objects that are dissimilar and small values to objects that are similar, reaching 0 when the two compared objects are the same.
Mathematically, a distance function is defined as follows:

Let $X$ be a set. A function $\delta:X\times X\to \mathbb R$ is called a distance function if it holds for all $x,y\in X$:

* $\delta(x,x)=0$ (reflexivity)
* $\delta(x,y)=\delta(y,x)$ (symmetry)
* $\delta(x,y)≥0$ (non-negativity)

When it comes to efficient query processing, as we will see later, it is useful if the utilized distance function is a metric.

Let $\delta:X\times X\to \mathbb R$ be a distance function. $\delta$ is called a metric if it holds for all $x,y,z\in X$:

* $\delta(x,y)=0\iff x=y$ (identity of indiscernibles)
* $\delta(x,y)\leq \delta(x,z)+\delta(z,y)$ (triangle inequality)

<img title = "search process" src = "https://ekanou.github.io/dynamicsearch/DynSe2018.png" width="80%" />

In a similarity-based retrieval model, it is assumed that the relevance status of a document with respect to a query is correlated with the similarity between the query and the document at some level of representation; the more similar to a query, the more relevant the document is assumed to be.
$\color{red}{Similarity \not= Relevance}$

Similarity matching|Relevance matching
---|---
Whether two sentences are semantically similar|Whether a document is relevant to a query
Homogeneous texts with comparable lengths| Heterogeneous texts (keywords query, document) and very different in lengths
Matches at all positions of both sentences|  Matches in different parts of documents
Symmetric matching function| Asymmetric matching function
Representative task: Paraphrase Identification|Representative task: ad-hoc retrieval

Each search is made up of $\color{red}{Match + Rank}$.

* [Deep Semantic Similarity Model](https://www.microsoft.com/en-us/research/project/dssm/)
* [AI in Information Retrieval and Language Processing collected by Wlodzislaw Duch](http://www.is.umk.pl/~duch/IR.html)
* [Deep Learning for Information Retrieval](https://pangolulu.github.io/2016/10/28/deep-ir/)
* [A Deep Relevance Matching Model for Ad-hoc Retrieval](https://arxiv.org/abs/1711.08611)
* [Relevance Matching](https://zhuanlan.zhihu.com/p/39946041)
* [DeepMatching: Deep Convolutional Matching](http://lear.inrialpes.fr/src/deepmatching/)
* [阿里自主创新的下一代匹配&推荐技术：任意深度学习+树状全库检索](https://zhuanlan.zhihu.com/p/35030348)
* https://www.cnblogs.com/yaoyaohust/p/10642103.html
* https://ekanou.github.io/dynamicsearch/
* http://mlwiki.org/index.php/NLP_Pipeline



##### Learning to Match

User’s intent is explicitly reflected in query such as keywords, questions.
Content is in  Webpages, images.
Key challenge is query-document semantic gap.
Even severe than search, since user and item are two different types of entities and are represented by different features.

Common goal: matching a need (may or may not include an explicit query)
to a collection of information objects (product descriptions, web pages, etc.)
Difference for search and recommendation: features used for matching!

---|Matching | Ranking
---|---------|---
Prediction | Matching degree between a query and a document| Ranking list of documents
Model|$f(q, d)$|$f(q,\{d_1,d_2,\dots \})$
Goal | Correct matching between query and document| Correct ranking on the top

Methods of Representation Learning for Matching:

*  DSSM: Learning Deep Structured Semantic Models for Web Search using Click-through Data (Huang et al., CIKM ’13)
*  CDSSM: A latent semantic model with convolutional-pooling structure for information retrieval (Shen et al. CIKM ’14)
*  CNTN: Convolutional Neural Tensor Network Architecture for Community-Based Question Answering (Qiu and Huang, IJCAI ’15)
*  CA-RNN: Representing one sentence with the other sentence as its
context (Chen et al., AAAI ’18)

DSSM: Brief Summary
+  Inputs: Bag of letter-trigrams as input for improving the scalability and generalizability
+  Representations: mapping sentences to vectors with DNN:
semantically similar sentences are close to each other
+  Matching: cosine similarity as the matching function
+  Problem: the order information of words is missing (bag of
letter-trigrams cannot keep the word order information)

Matching Function Learning:
* Step 1: construct basic low-level matching signals
* Step 2: aggregate matching patterns


- http://staff.ustc.edu.cn/~hexn/papers/www18-tutorial-deep-matching-paper.pdf
- [Deep Learning for Matching in Search and Recommendation](http://www.bigdatalab.ac.cn/~junxu/publications/SIGIR2018-DLMatch.pdf)
- [Deep Learning for Recommendation, Matching, Ranking and Personalization](http://sonyis.me/dnn.html)
- [Tutorials on Deep Learning for Matching in Search and Recommendation](https://www2018.thewebconf.org/program/tutorials-track/tutorial-191/)
- [Framework and Principles of Matching Technologies](http://www.hangli-hl.com/uploads/3/4/4/6/34465961/wsdm_2019_workshop.pdf)
- [Semantic Matching in Search](http://www.hangli-hl.com/uploads/3/1/6/8/3168008/ml_for_match-step2.pdf)

##### Regularized Latent Semantic Indexing

It is a matching method between query and document at topic level based on matrix factorization, which is scale up to large datasets.
The parametric model is expressed in the following form:

$$min_{U, \{v_n\}}\sum_{n=1}^{N}{\|d_n - U v_n\|}_2^2+\underbrace{\lambda_1\sum_{k=1}^K {\|u_k\|}_1}_{\text{topics are sparse}} + \underbrace{\lambda_2\sum_{n=1}^{N}{\|v_n \|}_2^2}_{\text{documents are smooth}}$$

where
- $d_n$ is term representation of doc $n$;
- $U$ represents topics;
- $v_n$ is the topic representation of doc $n$;
- $\lambda_1$ and $\lambda_2$ are regularization parameters.

It is optimized by coordinate descent:
$$u_{mk}=\arg\min_{\bar u_m}\sum_{m=1}^M {\|\bar d_m - V^T \bar u_m\|}_2^2+\lambda_1\sum_{m=1}^{M}{\|\bar u_m\|}_1,\\ v_n^{\ast}=\arg\min_{\{v_n\}}\sum_{n=1}^{N}{\|d_n -U v_n\|}_2^2+\lambda_2\sum_{n=1}^N{\|v_n\|}_2^2=(U^T U + \lambda_2 I)^{-1}U^T {d}_n.$$

- [Regularized Latent Semantic Indexing: A New Approach to Large Scale
Topic Modeling](http://www.hangli-hl.com/uploads/3/1/6/8/3168008/rlsi-tois-revision.pdf)
- https://www.academia.edu/13253156/Hierarchy-Regularized_Latent_Semantic_Indexing
- https://patents.google.com/patent/US8533195B2/en
- http://cse.msu.edu/~cse960/Papers/LSI/LSI.pdf
- https://github.com/JunXu-ICT/rlsi-java-source

##### Partial Least Square (PLS)

The input training data set is $\{(x_i, y_i, r_i)\mid i=1, 2,\cdots, N\}$ where $r_i \in \{+1, -1\}$.

It is to optimize the following cost function
$$\arg\max_{L_x, L_y}\sum_{r_i=+1}\left<L_x x_i, L_y y_i\right>-\sum_{r_i=-1}\left<L_x x_i, L_y y_i\right>\\ s.t. \quad L_x^T L_x=L_y^TL_y=I_k.$$
`Regularized Mapping to Latent Space` will change the constraints
$$\arg\max_{L_x, L_y}\sum_{r_i=+1}\left<L_x x_i, L_y y_i\right>-\sum_{r_i=-1}\left<L_x x_i, L_y y_i\right>\\ s.t. \quad L_x^T L_x=L_y^TL_y=I_k.$$

- https://stats.idre.ucla.edu/wp-content/uploads/2016/02/pls.pdf
- https://www.microsoft.com/en-us/research/publication/learning-bilinear-model-matching-queries-documents/
- https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/

##### Deep Structured Semantic Model

DSSM stands for Deep Structured Semantic Model, or more general, Deep Semantic Similarity Model. DSSM, developed by the MSR Deep Learning Technology Center(DLTC), is a deep neural network (DNN) modeling technique for representing text strings (sentences, queries, predicates, entity mentions, etc.) in a continuous semantic space and modeling semantic similarity between two text strings (e.g., Sent2Vec).

- https://www.microsoft.com/en-us/research/project/dssm/
- https://arxiv.org/pdf/1610.08136.pdf

##### Deep Relevance Matching Model

It is argumented that
> the ad-hoc retrieval task is mainly about relevance matching while most NLP matching tasks concern semantic matching, and there are some fundamental differences between these two matching tasks. Successful relevance matching requires proper handling of the exact matching signals, query term importance, and diverse matching requirements.

A novel deep relevance matching model (DRMM) for ad-hoc retrieval employs a joint deep architecture at the query term level for relevance matching. By using matching histogram mapping, a feed forward matching network, and a term gating network, we can effectively deal with the three relevance matching factors mentioned above.

<img src="https://frankblood.github.io/2017/03/10/A-Deep-Relevance-Matching-Model-for-Ad-hoc-Retrieval/DRMM.jpg" width="80%" />

+ Matching histogram mapping for summarizing each query matching signals
+ Term gating network for weighting the query matching signals
+ Lost word order information (during histogram mapping)

- [A Deep Relevance Matching Model for Ad-hoc Retrieval](https://arxiv.org/abs/1711.08611)
- https://zhuanlan.zhihu.com/p/38344505
- https://frankblood.github.io/2017/03/10/A-Deep-Relevance-Matching-Model-for-Ad-hoc-Retrieval/

##### DeepRank: Text Matching as Image Recognition

Calculate relevance by mimicking the human relevance judgement process

1. Detecting Relevance locations: focusing on locations of query terms when scanning the whole document
2. Determining local relevance: relevance between query and each location context, using MatchPyramid/MatchSRNN etc.
3. Matching signals aggregation

- [Deep Relevance Ranking Using Enhanced Document-Query Interactions](http://nlp.cs.aueb.gr/pubs/EMNLP2018Preso.pdf)
- [DeepRank: A New Deep Architecture for Relevance Ranking in Information Retrieval](https://arxiv.org/pdf/1710.05649.pdf)

Challenges

- [ ] Representation: representing the word level matching signalsas well as the matching positions
- [ ] Modeling: discovering the matching patternsbetween two texts
- [ ] Our solutions
  - [ ] Step 1: representing as matching matrix
  - [ ] Step 2: matching as image recognition

Matching matrix $M_{ij}=\mathbb I_{w_i=v_j}$ or $M_{ij}= \frac{w_i^T v_j}{\|w_i\| \|v_j\|}$ or $M_{ij}=\left<w_i, v_j\right>$.

$$\fbox{MatchPyramid} =\underbrace{Matching\,\, Matrix}_{\text{Bridging the semantic gap between words}}+\underbrace{Hierarchical\,\, Convolution}_{\text{Capturing rich matching patterns}}$$


- http://www.bigdatalab.ac.cn/~junxu/publications/AAAI2016_CNNTextMatch.pdf
- http://www.bigdatalab.ac.cn/~junxu/publications/AAAI2016_BiLSTMTextMatch.pdf

#### Semantic Search

[Alexis Sanders  as an SEO Account Manager at MERKLE | IMPAQT wrote a blog on `semantic search`](https://moz.com/blog/what-is-semantic-search):
> The word "semantic" refers to the meaning or essence of something. Applied to search, "semantics" essentially relates to the study of words and their logic. Semantic search seeks to improve search accuracy by understanding a searcher’s intent through contextual meaning. Through concept matching, synonyms, and natural language algorithms, semantic search provides more interactive search results through transforming structured and unstructured data into an intuitive and responsive database. Semantic search brings about an enhanced understanding of searcher intent, the ability to extract answers, and delivers more personalized results. Google’s Knowledge Graph is a paradigm of proficiency in semantic search.

<img src="https://blog.alexa.com/wp-content/uploads/2019/03/semantic-search-intent.png" width="80%"/>

* [relevant search](http://manning.com/books/relevant-search)
* [Learning to rank plugin of Elasticsearch](https://github.com/o19s/elasticsearch-learning-to-rank)
* [MatchZoo's documentation](https://matchzoo.readthedocs.io/zh/latest/)
* http://mlwiki.org/index.php/Information_Retrieval_(UFRT)
* https://en.wikipedia.org/wiki/List_of_search_engines
* [Open Semantic Search](https://www.opensemanticsearch.org/)
* https://www.seekquarry.com/
* http://l-lists.com/en/lists/qukoen.html
* [20款开源搜索引擎介绍与比较](https://blog.csdn.net/belalds/article/details/80758312)
* [gt4ireval: Generalizability Theory for Information Retrieval Evaluation](https://rdrr.io/cran/gt4ireval/)
* https://daiwk.github.io/posts/nlp.html
* http://www2003.org/cdrom/papers/refereed/p779/ess.html
* https://blog.alexa.com/semantic-search/

#### Query-independent Ranking

`Centrality of network` assigns an importance score based purely on the number of links held by each node.

`Search Engine Optimization(SEO)` is a business type to boost the website higher.
`PageRank` is introduced in `Graph Algorithms`.

* http://ryanrossi.com/search.php
* [MGT 780/MGT 795 Social Network Analysis](http://www.analytictech.com/mgt780/)
* [The Anatomy of a Large-Scale Hypertextual Web Search Engine by Sergey Brin and Lawrence Page](http://infolab.stanford.edu/~backrub/google.html)
* [The Mathematics of Google Search](http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/)
* [HITS Algorithm - Hubs and Authorities on the Internet](http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture4/lecture4.html)
* http://langvillea.people.cofc.edu/
* [Google PageRank: The Mathematics of Google](http://www.whydomath.org/node/google/index.html)
* [How Google Finds Your Needle in the Web's Haysta](http://www.ams.org/publicoutreach/feature-column/fcarc-pagerank)
* [Dynamic PageRank](http://ryanrossi.com/dynamic-pagerank.php)

### Information Distribution: Search Engine Results Page

[Information Distribution Methods – Information distribution is the timely collection, sharing and distribution of information to the project team. Methods can be portals, collaborative work management tools, web conferencing, web publishing, and when all technology is not available, manual filing systems and hard copy distribution.](http://www.anticlue.net/archives/000804.htm)

* [SERP: GUIDE TO THE GOOGLE SEARCH ENGINE RESULTS (UPDATED 2019 GUIDE)](https://ignitevisibility.com/serp/)
* [CH. 5: PRESENTATION OF SEARCH RESULTS](http://searchuserinterfaces.com/book/sui_ch5_retrieval_results.html)
* [CH. 10: INFORMATION VISUALIZATION FOR SEARCH INTERFACES](https://searchuserinterfaces.com/book/sui_ch10_visualization.html)
* [CH. 11: INFORMATION VISUALIZATION FOR TEXT ANALYSIS](https://searchuserinterfaces.com/book/sui_ch11_text_analysis_visualization.html)
* [Match Zoo](https://xieydd.github.io/post/matchzoo/)

#### Keywords Highlight

- https://doc.nuxeo.com/nxdoc/elasticsearch-highlights/

#### Webpage Snapshot

### Neural Information Retrieval

Neural networks or deep learning as a subfield of machine learning, is widely applied in information processing.

> During the opening keynote of the SIGIR 2016 conference, Christopher Manning predicted a significant influx of deep neural network related papers for IR in the next few years.
However, he encouraged the community to be mindful of some of the “irrational exuberance” that plagues the field today.
The first SIGIR workshop on neural information retrieval received an unexpectedly high number of submissions and registrations.
These are clear indications that the IR community is excited by the recent developments in the area of deep neural networks.
This is indeed an exciting time for this area of research and we believe that besides attempting to simply demonstrate empirical progress on retrieval tasks,
our explorations with neural models should also provide new insights about IR itself.
In return, we should also look for opportunities to apply IR intuitions into improving these neural models, and their application to non-IR tasks.

- http://nn4ir.com/
- [Neu-IR: Workshop on Neural Information Retrieval](https://neu-ir.weebly.com/)
- [Topics in Neural Information Retrieval](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/teaching/ss19/topics-in-neural-information-retrieval/)
- https://frankblood.github.io/

### Modeling Diverse Ranking with MDP

Key points:

-  Mimic user top-down browsing behaviors
-  Model dynamic information needs with MDP state

Staets $𝑠_𝑡=[\it Z_𝑡,\it X_𝑡,\mathrm h_𝑡]$ consists of
* sequence of 𝑡 preceding documents, $\it Z_t$ and $\it Z_0=\emptyset$;
* set of candidate documents, $\it X_t$ and $\it X_0 = \it X$
* latent vector $\mathrm h_𝑡$, encodes user perceived utility from preceding documents, initialized with the information needs form the query: $\mathrm h_0=\sigma(V_qq)$

MDP factors | Corresponding diverse ranking factors
---|---
Timesteps | The ranking positions
States | $𝑠_𝑡=[\it Z_𝑡,\it X_𝑡,\mathrm h_𝑡]$
Policy | $\pi(a_t\mid s_t=[\it Z_t, \it X_t,\mathrm h_t])=\frac{\exp\{\mathrm x^T_{m(a_t)}\mathrm U \mathrm h_t\}}{Z}$
Action | Selecting a doc and placing it to rank $\it t+1$
Reward | Based on evaluation measure αDCG, SRecall etc.
State Transition | $s_{t+1}=T(s_t, a_t)=[Z_t\oplus \{\mathrm x_{m(a_t)}\}, \it X_t\setminus \{\mathrm x_{m(a_t)} \}, \sigma(V\mathrm  x_{m(a_t)}+W\mathrm h_t)]$

Here $\mathrm x_{m(a_t)}$ is document embedding.
Model parameters $\Theta=(V_q, U, V, W)$ is to optimize the expected reward.
The goal is maximizing expected return (discounted sum of rewards) of each training query
$$\max_{\Theta} v(q)=\mathbb E_{\pi}G_0=\mathbb E_{\pi}[\sum_{k=0}^{M-1}\gamma^k r_{k+1}].$$

Monte-Carlo stochastic gradient ascent is used to conduct the optimization (REINFORCE algorithm)
$$\nabla_{\Theta}\hat{v}(q)=\gamma^t G_t\nabla_{\Theta}\log(\pi(a_t\mid S_t;\Theta)).$$

+ http://www.bigdatalab.ac.cn/~gjf/papers/2017/SIGIR2017_MDPDIV.pdf
+ [Reinforcement Learning to Rank with Markov Decision Process](http://www.bigdatalab.ac.cn/~junxu/publications/CCF@U_RL4IR.pdf)
+ [Deep and Reinforcement Learning for Information Retrieval](http://cips-upload.bj.bcebos.com/ssatt2018%2FATT9_2_%E4%BF%A1%E6%81%AF%E6%A3%80%E7%B4%A2%E4%B8%AD%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E6%96%B0%E8%BF%9B%E5%B1%95.pdf)
+ [Improving Session Search Performance with a Multi-MDP Model](http://www.thuir.cn/group/~YQLiu/publications/AIRS18Chen.pdf)
+ https://github.com/ICT-BDA/EasyML

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
* [CH. 9: PERSONALIZATION IN SEARCH](https://searchuserinterfaces.com/book/sui_ch9_personalization.html)

### Vertical Domain Search: Beyond String and Texts

As we have learned how to handle text, information retrieval is moving on, to projects in sound and image retrieval, along with electronic provision of much of what is now in libraries.


* [Vertical search](https://en.wikipedia.org/wiki/Vertical_search)


#### Medical Information Retrieval

ChartRequest claims that:
> Requesting medical records is vital to your operations as a health insurance company. From workers’ compensation claims to chronic-condition care, insurance companies require numerous medical records—daily. Obtain records quickly and accurately with our medical information retrieval software. ChartRequest offers a complete enterprise solution for health insurance companies—facilitating swift fulfillment and secure, HIPAA-compliant records release.

##### What is Biomedical and Health Informatics?

`Biomedical and health informatics (BMHI)` is the field concerned with the optimal use of information, often aided by technology, to improve individual health, healthcare, public health, and biomedical research.

- Unified Medical Language System (UMLS)
- Systematized Nomenclature of Medicine--Clinical Terms (SNOMED-CT)
- International Classification of Diseases (ICD)

* https://dmice.ohsu.edu/hersh//whatis/
* https://www.nlm.nih.gov/research/umls/
* http://www.snomed.org/
* https://icd.who.int/en/

##### Why is medical information retrieval important?

To health professionals, applications providing an easy access to validated and up-to-date
health knowledge are of great importance to the dissemination of knowledge and have the potential to impact the quality of care provided by health professionals.
On the other side, the Web opened doors to the access of health information by patients, their family and friends,
making them more informed and changing their relation with health professionals.

To professionals, one of the main and oldest IR applications is *PubMed* from the US National Library of Medicine (NLM) that gives access to the world’s medical research literature.
To consumers, health information is available through different services and with different quality.
Lately, the control over and access to health information by consumers has been a hot topic, with plenty government initiatives all over the world that aim to improve consumer health giving consumers more information and making easier the sharing of patient records.

- http://carlalopes.com/pubs/Lopes_SOA_2008.pdf

##### Why is medical information retrieval difficult?

It is becasue medical information is really professional while critical.


##### How knowledge bases can improve retrieval performance?

- https://slides.com/saeidbalaneshinkordan/medical_information_retrieval#/23
- https://slides.com/saeidbalaneshinkordan/medical_information_retrieval-1-5-6

****
* https://dmice.ohsu.edu/hersh/
* http://www.balaneshin.com/
* https://www.chartrequest.com/
* https://www.a-star.edu.sg/resource
* http://www.bii.a-star.edu.sg/
* [Medical Information Retrieval](http://www.bii.a-star.edu.sg/docs/mig/MedIR.pdf)
* [Information Retrieval: A Health and Biomedical Perspective](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC545137/)
* http://www.khresmoi.eu/
* http://www.imedisearch.com/
* http://everyone.khresmoi.eu/
* https://meshb.nlm.nih.gov/
* http://cbm.imicams.ac.cn/
* [Consumer Health Search](https://ielab.io/projects/consumer-health-search.html)
* [Biomedical Data Science Initiative](http://med.stanford.edu/bdsi.html)
* https://dmice.ohsu.edu/hersh/irbook/
* https://www.hon.ch/en/
* https://search.kconnect.eu/beta/
* https://clefehealth.imag.fr/
* http://www.bilegaldoc.com/

####  Music Information Retrieval

* https://www.music-ir.org/mirex/wiki/MIREX_HOME
* https://www.ismir.net/
* https://github.com/mozilla/DeepSpeech
* https://github.com/mkanwal/Deep-Music
* http://www.nyu.edu/classes/bello/MIR.html
* [Introduction to Mozilla Deep Speech](https://www.simonwenkel.com/2018/09/24/Introduction-to-Mozilla-Deep-Speech.html)
* [The MusArt Music-Retrieval System: An Overview](http://www.dlib.org/dlib/february02/birmingham/02birmingham.html)
* [SGN-24006 Analysis of Audio, Speech and Music Signals Spring 2017 ](http://www.cs.tut.fi/~sgn24006/)
* [Notes on Music Information Retrieval](https://musicinformationretrieval.com/index.html)

#### Multimedia Search Engine

* [自制AI图像搜索引擎](https://blog.csdn.net/baidu_40840693/article/details/88230418)
* [深度学习表征的不合理有效性——从头开始构建图像搜索服务（一）](https://segmentfault.com/a/1190000015570726)
* [深度学习表征的不合理有效性——从头开始构建图像搜索服务（二）](https://yq.aliyun.com/articles/607384)
* [Building a Content-Based Multimedia Search Engine I: Quantifying Similarity](http://www.deepideas.net/building-content-based-multimedia-search-engine-quantifying-similarity/)
* [Building a Content-Based Multimedia Search Engine II: Extracting Feature Vectors](http://www.deepideas.net/building-content-based-multimedia-search-engine-feature-extraction/)
* [Building a Content-Based Multimedia Search Engine III: Feature Signatures](http://www.deepideas.net/building-content-based-multimedia-search-engine-feature-signatures/)
* [Building a Content-Based Multimedia Search Engine IV: Earth Mover’s Distance](http://www.deepideas.net/building-content-based-multimedia-search-engine-earth-movers-distance/)
* [Building a Content-Based Multimedia Search Engine V: Signature Quadratic Form Distance](http://www.deepideas.net/building-content-based-multimedia-search-engine-signature-quadratic-form-distance/)
* [Building a Content-Based Multimedia Search Engine VI: Efficient Query Processing](http://www.deepideas.net/building-content-based-multimedia-search-engine-efficient-query-processing/)

#### Multimodal Search

* http://www.khresmoi.eu/overview/
* [Multi-Task Learning with Neural Networks for Voice Query Understanding on an Entertainment Platform](https://www.kdd.org/kdd2018/accepted-papers/view/multi-task-learning-with-neural-networks-for-voice-query-understanding-on-a)

### Knowledge Graphs

Search is not only on string but also things.

+ [The Second Workshop on Knowledge Graphs and Semantics for Text Retrieval, Analysis, and Understanding](https://kg4ir.github.io/)
+ https://twiggle.com/
+ https://www.clearquery.io/how
+ [The Entity & Language Series: Translation and Language APIs Impact on Query Understanding & Entity Understanding (4 of 5)](https://mobilemoxie.com/blog/the-entity-language-series-translation-and-language-apis-impact-on-query-understanding-entity-understanding-4-of-5/)

### Labs and Resources  

#### Labs on Search and Information Retrieval

+ [Search and information retrieval@Microsoft](https://www.microsoft.com/en-us/research/research-area/search-information-retrieval/)
+ [Search and information retrieval@Google](https://ai.google/research/pubs/?area=InformationRetrievalandtheWeb)
+ [Web search and mining @Yandex](https://research.yandex.com/publications?themeSlug=web-mining-and-search)
+ [The Information Engineering Lab](https://ielab.io/)
+ [Information Retrieval Lab: A research group @ University of A Coruña (Spain)](https://www.irlab.org/)
+ [BCS-IRSG: Information Retrieval Specialist Group](https://irsg.bcs.org/)
+ [智能技术与系统国家重点实验室信息检索课题组](http://www.thuir.org/)
+ [Web Information Retrieval / Natural Language Processing Group (WING)](http://wing.comp.nus.edu.sg/)
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
+ http://www.cs.wayne.edu/kotov/index.html
+ http://www.isp.pitt.edu/research/nlp-info-retrieval-group
+ [LETOR: Learning to Rank for Information Retrieval](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval)
+ [A Lucene toolkit for replicable information retrieval research ](https://github.com/castorini/anserini)
+ [Information Overload Research Group](https://iorgforum.org/)
+ [National Information Standards Organization by ANSI](https://www.niso.org/niso-io)
+ [International Society  of Music Information Retrieval](https://ismir.net/)
+ [Open Clinical information retrieval](http://www.openclinical.org/informationretrieval.html)
+ [Medical Library Association](https://www.mlanet.org/)
+ [AMIA](https://www.amia.org/)
+ [INternational Medical INformatics Association](https://imia-medinfo.org/wp/)
+ [Association of Directors of Information System](https://amdis.org/)

#### Conferences on Information Retrieval

+ https://datanatives.io/conference/
+ [Natural Language Processing in Information Retrieval](http://nlpir.net/)
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

#### Courses on Information Retrieval and Search

+ [CS 276 / LING 286: Information Retrieval and Web Search](https://web.stanford.edu/class/cs276/)
+ [LING 289: History of Computational Linguistics
Winter 2011 ](http://web.stanford.edu/class/linguist289/)
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
+ [Modern Information Retrieval](http://grupoweb.upf.edu/mir2ed/home.php)
+ [Search Engine: Information Retrieval in Practice](http://www.search-engines-book.com/)
+ [Information Retrieval  潘微科](http://csse.szu.edu.cn/csse.szu.edu.cn/staff/panwk/IR201702/index.html)
+ [Information Organization and Retrieval: INFO 202](http://courses.ischool.berkeley.edu/i202/f10/)
+ [Music Information Retrieval @ NYU](http://www.nyu.edu/classes/bello/MIR.html)
+ [Intelligent Information Retrieval](http://facweb.cs.depaul.edu/mobasher/classes/CSC575/)
+ [CSc 7481 / LIS 7610 Information Retrieval Spring 2008](http://www.csc.lsu.edu/~kraft/courses/csc7481.html)
+ [Winter 2016 CSI4107: Information Retrieval and the Internet](http://www.site.uottawa.ca/~diana/csi4107/)
+ [Information Retrieval 2017 Spring](http://berlin.csie.ntnu.edu.tw/Courses/Information%20Retrieval%20and%20Extraction/2020S_IR_Main.htm)
+ [Ranking Model](https://homepages.dcc.ufmg.br/~rodrygo/rm-2018-2/)
+ https://searchpatterns.org/
