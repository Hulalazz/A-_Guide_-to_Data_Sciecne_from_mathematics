## NLP

Since 2018, pre-training has without a doubt become one of the hottest research topics in Natural Language Processing (NLP). By leveraging generalized language models like the BERT, GPT and XLNet, great breakthroughs have been achieved in natural language understanding. However, in sequence to sequence based language generation tasks, the popular pre-training methods have not achieved significant improvements. Now, researchers from Microsoft Research Asia have introduced MASS—a new pre-training method that achieves better results than BERT and GPT.

<img title="The Encoder-Attention-Decoder framework" src="https://www.microsoft.com/en-us/research/uploads/prod/2019/06/MASS-Fig-1.png" width="60%" />
<img title="MASS framework" src="https://www.microsoft.com/en-us/research/uploads/prod/2019/06/MASS-Fig-4.png" width="60%" />

* https://deep-learning-drizzle.github.io/
* https://madewithml.com/
* http://biostat.mc.vanderbilt.edu/wiki/Main/RmS
* https://nlpprogress.com/
* [Tracking Progress in Natural Language Processing](https://ruder.io/tracking-progress-nlp/)
* [A curated list of resources dedicated to Natural Language Processing (NLP)](https://github.com/keon/awesome-nlp)
* https://www.cs.cmu.edu/~rsalakhu/
* https://github.com/harvardnlp
* https://nlpoverview.com/
* http://www.stat.ucla.edu/~sczhu/Courses/UCLA/Stat_232A/Stat_232A.html
* [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/)
* [CS224n: Natural Language Processing with Deep Learning, Stanford / Winter 2019](https://web.stanford.edu/class/cs224n/)
* [Deep Learning for NLP](https://www.comp.nus.edu.sg/~kanmy/courses/6101_1810/)
* https://textprocessing.github.io/
* https://nlpforhackers.io/
* http://www.cs.yale.edu/homes/radev/dlnlp2017.pdf
* https://handong1587.github.io/deep_learning/2015/10/09/nlp.html
* https://cla2018.github.io/dl4nlp_roth.pdf
* https://deep-learning-nlp.readthedocs.io/en/latest/
* https://zhuanlan.zhihu.com/c_188941548
* https://allennlp.org/
* http://aan.how/
* http://www.cs.yale.edu/homes/radev/
* http://www.cs.cmu.edu/~bishan/pubs.html
* http://web.stanford.edu/class/cs224n/index.html
* https://explosion.ai/

<img src="https://pic3.zhimg.com/80/v2-598c67c45b4fe0d0b16a21ee7ba91226_hd.jpg" width="70%"/>

The general building blocks of their model, however, are still found in all current neural language and word embedding models. These are:

1. Embedding Layer: a layer that generates word embeddings by multiplying an index vector with a word embedding matrix;
2. Intermediate Layer(s): one or more layers that produce an intermediate representation of the input, e.g. a fully-connected layer that applies a non-linearity to the concatenation of word embeddings of $n$ previous words;
3. Softmax Layer: the final layer that produces a probability distribution over words in $V$.

> The softmax layer is a core part of many current neural network architectures. When the number of output classes is very large, such as in the case of language modelling, computing the softmax becomes very expensive. 

### Word Embedding and Language Model

Word are always in the string data structure in computer.
> Language is made of discrete structures, yet neural networks operate on continuous data: vectors in high-dimensional space. A successful language-processing network must translate this symbolic information into some kind of geometric representation—but in what form? Word embeddings provide two well-known examples: distance encodes semantic similarity, while certain directions correspond to polarities (e.g. male vs. female).
There is no arithmetic operation on this data structure.
We need an embedding that maps the strings into vectors.

Language Modeling (LM) estimates the probability of a word given the previous words in a sentence: $P(x_t\mid x_1,\cdots, x_{t-1},\theta)$. 
Formally, the model is trained with inputs $(x_1,\cdots, x_{t-1})$ and outputs $Y(x_t)$, where $x_t$ is the output label predicted from the final (i.e. top-layer) representation of a token $x_{t-1}$.

The Neural Probabilistic Language Model can be summarized as follows:

1. associate with each word in the vocabulary a distributed word feature vector (a realvalued vector in $\mathbb{R}^m$),
2. express the joint probability function of word sequences in terms of the feature vectors
of these words in the sequence, and
3. learn simultaneously the word feature vectors and the parameters of that probability
function.

When a word $w$ appears	in	a text,	its	context is	the	set	of	words	
that appear	nearby	(within	a fixed-size	window).
We can use	the	many contexts	of	$w$ to	build up a representation of	$w$.



- https://www.cs.bgu.ac.il/~elhadad/nlp18/nlp02.html
- http://josecamachocollados.com/book_embNLP_draft.pdf
- https://arxiv.org/abs/1906.02715
- https://andre-martins.github.io/pages/project-examples-for-deep-structured-learning-fall-2018.html
- https://ruder.io/word-embeddings-1/index.html
- https://carl-allen.github.io/nlp/2019/07/01/explaining-analogies-explained.html
- https://arxiv.org/abs/1901.09813
- [Analogies Explained Towards Understanding Word Embeddings](https://icml.cc/media/Slides/icml/2019/104(13-11-00)-13-11-00-4883-analogies_expla.pdf)
- https://pair-code.github.io/interpretability/bert-tree/
- https://pair-code.github.io/interpretability/context-atlas/blogpost/ 
- http://disi.unitn.it/moschitti/Kernel_Group.htm   

### Attention

[Attention mechanisms in neural networks, otherwise known as neural attention or just attention, have recently attracted a lot of attention (pun intended). In this post, I will try to find a common denominator for different mechanisms and use-cases and I will describe (and implement!) two mechanisms of soft visual attention.](http://akosiorek.github.io/ml/2017/10/14/visual-attention.html)

[`An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.`  The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.](https://arxiv.org/pdf/1706.03762.pdf)

* https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
* http://nlp.seas.harvard.edu/2018/04/03/attention.html
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* https://arxiv.org/pdf/1811.05544.pdf
* https://arxiv.org/abs/1902.10186
* https://arxiv.org/abs/1906.03731
* https://arxiv.org/abs/1908.04626v1
* [遍地开花的 Attention，你真的懂吗？ - 阿里技术的文章 - 知乎](https://zhuanlan.zhihu.com/p/77307258)
* https://xpqiu.github.io/slides/20200613-CAAI-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E4%B8%AD%E7%9A%84%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%A8%A1%E5%9E%8B.pdf
  
### Transformer

> Representation learning forms the foundation of today’s natural language processing system; Transformer models have been extremely effective at producing word- and sentence-level contextualized representations, achieving state-of-the-art results in many NLP tasks. However, applying these models to produce contextualized representations of the entire documents faces challenges. These challenges include lack of inter-document relatedness information, decreased performance in low-resource settings, and computational inefficiency when scaling to long documents.In this talk, I will describe 3 recent works on developing Transformer-based models that target document-level natural language tasks.

* [Transformer结构及其应用--GPT、BERT、MT-DNN、GPT-2 - 知乎](https://zhuanlan.zhihu.com/p/69290203)
* [放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)
* [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
* [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning) ](http://jalammar.github.io/illustrated-bert/)
* [Universal Transformers](https://mostafadehghani.com/2019/05/05/universal-transformers/)
* [Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View](https://arxiv.org/abs/1906.02762)
* https://arxiv.org/abs/1706.03762
* https://arxiv.org/abs/2002.04723
* https://lena-voita.github.io/posts/emnlp19_evolution.html
* https://www.aclweb.org/anthology/2020.acl-main.385.pdf
* https://math.la.asu.edu/~prhahn/

### BERT 

* https://github.com/tomohideshibata/BERT-related-papers
* https://pair-code.github.io/interpretability/bert-tree/
* https://arxiv.org/pdf/1810.04805.pdf
* https://arxiv.org/pdf/1906.02715.pdf
* [BERT-is-All-You-Need](https://github.com/Eurus-Holmes/BERT-is-All-You-Need)
* [Visualizing and Measuring the Geometry of BERT](https://arxiv.org/abs/1906.02715)
* [BertEmbedding](https://bert-embedding.readthedocs.io/en/latest/api_reference/bert_embedding.html)
* https://pair-code.github.io/interpretability/bert-tree/
* https://zhuanlan.zhihu.com/p/70257427
* https://zhuanlan.zhihu.com/p/51413773
* [MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/abs/1905.02450)
* [SenseBERT: Driving some sense into BERT](https://www.ai21.com/sense-bert)

### GPT

- https://github.com/openai/gpt-2
- [Better Language Models and Their Implications](https://openai.com/blog/better-language-models/)
- https://openai.com/blog/image-gpt/
- https://www.deepspeed.ai/