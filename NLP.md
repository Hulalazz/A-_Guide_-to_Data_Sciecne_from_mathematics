## NLP

Since 2018, pre-training has without a doubt become one of the hottest research topics in Natural Language Processing (NLP). 
By leveraging generalized language models like the BERT, GPT and XLNet, great breakthroughs have been achieved in natural language understanding. 
However, in sequence to sequence based language generation tasks, the popular pre-training methods have not achieved significant improvements. 
Now, researchers from Microsoft Research Asia have introduced MASS—a new pre-training method that achieves better results than BERT and GPT.

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
* [Stat232A: Statistical Modeling and Learning in Vision and Cognition](http://www.stat.ucla.edu/~sczhu/Courses/UCLA/Stat_232A/Stat_232A.html)
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


#### Probabilistic Language Model

Language Modeling (LM) estimates the probability of a word given the previous words in a sentence: $P(x_t\mid x_1,\cdots, x_{t-1},\theta)$. 
Formally, the model is trained with inputs $(x_1,\cdots, x_{t-1})$ and outputs $Y(x_t)$, where $x_t$ is the output label predicted from the final (i.e. top-layer) representation of a token $x_{t-1}$.

- [Probability and Structure in Natural Language Processing](http://www.cs.cmu.edu/~nasmith/psnlp/)
- [Probabilistic Models in the Study of Language](http://idiom.ucsd.edu/~rlevy/pmsl_textbook/text.html)
- [CS598 jhm Advanced NLP (Bayesian Methods)](https://courses.engr.illinois.edu/cs598jhm/sp2013/index.html)
+ [Statistical Language Processing and Learning Lab.](https://staff.fnwi.uva.nl/k.simaan/research_all.html)
+ [Bayesian Analysis in Natural Language Processing, Second Edition](http://www.morganclaypoolpublishers.com/catalog_Orig/product_info.php?products_id=1385)
+ [Probabilistic Context Grammars](https://gawron.sdsu.edu/compling/course_core/lectures/pcfg/prob_parse.htm)
+ https://ofir.io/Neural-Language-Modeling-From-Scratch/
+ https://staff.fnwi.uva.nl/k.simaan/D-Papers/RANLP_Simaan.pdf
- https://staff.fnwi.uva.nl/k.simaan/ESSLLI03.html
- http://www.speech.cs.cmu.edu/SLM/toolkit_documentation.html
- http://www.phontron.com/kylm/
- http://www.lemurproject.org/lemur/background.php
- https://meta-toolkit.org/
- https://github.com/renepickhardt/generalized-language-modeling-toolkit
- http://www.cs.cmu.edu/~nasmith/LSP/

#### Neural  Language Model

The Neural Probabilistic Language Model can be summarized as follows:

1. associate with each word in the vocabulary a distributed word feature vector (a real valued vector in $\mathbb{R}^m$),
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
- [Project Examples for Deep Structured Learning (Fall 2018)](https://andre-martins.github.io/pages/project-examples-for-deep-structured-learning-fall-2018.html)
- https://ruder.io/word-embeddings-1/index.html
- https://carl-allen.github.io/nlp/2019/07/01/explaining-analogies-explained.html
- https://arxiv.org/abs/1901.09813
- [Analogies Explained Towards Understanding Word Embeddings](https://icml.cc/media/Slides/icml/2019/104(13-11-00)-13-11-00-4883-analogies_expla.pdf)
- https://pair-code.github.io/interpretability/bert-tree/
- https://pair-code.github.io/interpretability/context-atlas/blogpost/ 
- http://disi.unitn.it/moschitti/Kernel_Group.htm   


### Attention Mechanism

[`Attention mechanisms` in neural networks, otherwise known as `neural attention` or just `attention`, have recently attracted a lot of attention (pun intended).](http://akosiorek.github.io/ml/2017/10/14/visual-attention.html)

[`An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.`  The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.](https://arxiv.org/pdf/1706.03762.pdf)

* https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* https://arxiv.org/pdf/1811.05544.pdf
* https://arxiv.org/abs/1902.10186
* https://arxiv.org/abs/1906.03731
* https://arxiv.org/abs/1908.04626v1
* [遍地开花的 Attention，你真的懂吗？ - 阿里技术的文章 - 知乎](https://zhuanlan.zhihu.com/p/77307258)

- https://www.jpmorgan.com/jpmpdf/1320748255490.pdf
- [Understanding Graph Neural Networks from Graph Signal Denoising PerspectivesCODE](https://arxiv.org/abs/2006.04386)
- [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)
- https://www.dl.reviews/

******************

Attention distribution is a probability distribution to describe how much we pay attention into the elements in a sequence for some specific task.

For example, we have a query vector $\mathbf{q}$ associated with the task and a list of input vector $\mathbf{X}=[\mathbf{x}_1, \mathbf{x}_2,\cdots, \mathbf{x}_N]$. We can select the input vector smoothly with the weight
$$\alpha_i=p(z=i\mid \mathbf{X}, \mathbf{q})\\
=\frac{\exp(s(\mathbf{x}_i, \mathbf{q}))}{\sum_{j=1}^N\exp(s(\mathbf{x}_j, \mathbf{q}))}$$
where $\alpha_i$ is the attention distribution, $s(\cdot,\cdot)$ is the attention scoring function. $z$ is the index of the vector in $\mathbf{X}$.
The "Scaled Dot-Product Attention" use the modified  dot-product as the scoring function.

- https://chulheey.mit.edu/
- https://arxiv.org/abs/2101.11347
- https://linear-transformers.com/
- https://github.com/idiap/fast-transformers

**********

The attention function between different input vectors is calculated
as follows:
1. Step 1: Compute scores between different input vectors and query vector $S_N$;
2. Step 2: Translate the scores into probabilities such as $P = \operatorname{softmax}(S_N)$;
3. Step 3: Obtain the output as aggregation such as the weighted value matrix with $Z = \mathbb{E}_{z\sim p(\mid \mathbf{X}, \mathbf{q} )}\mathbf{[x]}$.

There are diverse scoring functions and probability translation function, which will calculate the attention distribution in different ways. 


<img src="https://lena-voita.github.io/resources/lectures/seq2seq/attention/computation_scheme-min.png" width="70%"/>

[Efficient Attention](https://cmsflash.github.io/ai/2019/12/02/efficient-attention.html), [Linear Attention](http://proceedings.mlr.press/v119/katharopoulos20a.html) apply more efficient methods to generate attention weights.

[Key-value Attention Mechanism](https://www.aclweb.org/anthology/I17-2049.pdf) and  [Self-Attention](https://www.aclweb.org/anthology/D18-1458.pdf) use different input sequence as following
$$\operatorname{att}(\mathbf{K, V}, \mathbf{q}) =\sum_{j=1}^N\frac{s(\mathbf{K}_j, q)\mathbf{V}_j}{\sum_{i=1}^N s(\mathbf{K}_i, q)}$$
where $\mathbf{K}$ is the key matrix, $\mathbf{V}$ is the value matrix, $s(\cdot, \cdot)$ is the positive similarity function.

Each input token in self-attention receives three representations corresponding to the roles it can play:

* query - asking for information;
* key - saying that it has some information;
* value - giving the information.

We compute the dot products of the query with all keys, divide each by square root of key dimension  $d_k$, and apply a softmax function to obtain the weights on the values as following.


<img src="https://paul-hyun.github.io/assets/2019-12-19/scale_dot_product_attention.png" width="60%"/>

- https://epfml.github.io/attention-cnn/

---------------

Soft Attention: the alignment weights are learned and placed “softly” over all patches in the source image; essentially the same type of attention as in [Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473).
And each output is derived from an attention averaged input.
- Pro: the model is smooth and differentiable.
- Con: expensive when the source input is large.

Hard Attention: only selects one patch of the image to attend to at a time,
which attends to exactly one input state for an output.
- Pro: less calculation at the inference time.
- Con: the model is non-differentiable and requires more complicated techniques such as variance reduction or reinforcement learning to train. [(Luong, et al., 2015)](https://arxiv.org/abs/1508.04025)

#### Soft Attention Mechanism

`Soft Attention Mechanism` is to output the weighted sum of vector with differentiable scoring function:

$$\operatorname{att}(\mathbf{X}, \mathbf{q}) = \mathbb{E}_{z\sim p(z\mid \mathbf{X}, \mathbf{q} )}\mathbf{[x]}$$

where $p(z\mid \mathbf{X}, \mathbf{q} )$ is the attention distribution.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$. The keys and values are also packed together into matrices $K$ and $V$. 
We compute the matrix of outputs as:
$$\operatorname{Attention}(Q, K, V)= [\mathbb{E}_{z\sim p(z\mid \mathbf{K}, \mathbf{Q}_1)}\mathbf{[V]},\cdots,\mathbb{E}_{z\sim p(z\mid \mathbf{K}, \mathbf{Q}_i )}\mathbf{[V]},\cdots, \mathbb{E}_{z\sim p(z\mid \mathbf{K}, \mathbf{Q}_N )}\mathbf{[V]}].$$


- https://arxiv.org/abs/1409.0473
- [Attention Is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)


****************

- https://ayplam.github.io/dtca/
- [Fast Pedestrian Detection With Attention-Enhanced Multi-Scale RPN and Soft-Cascaded Decision Trees](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8883216)
- [DTCA: Decision Tree-based Co-Attention Networks for Explainable
Claim Verification](https://www.aclweb.org/anthology/2020.acl-main.97.pdf)

#### Hard Attention Mechanism

`Hard Attention Mechanism` is to select most likely vector as the output
$$\operatorname{att}(\mathbf{X}, \mathbf{q}) = \mathbf{x}_j$$ 
where $j=\arg\max_{i}\alpha_i$.

It is trained using sampling method or reinforcement learning.

- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- https://github.com/roeeaharoni/morphological-reinflection
- [Surprisingly Easy Hard-Attention for Sequence to Sequence Learning](https://www.cse.iitb.ac.in/~sunita/papers/EMNLP2018.pdf)

#### Sparse Attention Mechanism

The `softmax` mapping  is elementwise proportional to $exp$, therefore it can never assign a weight of exactly zero. 
Thus, unnecessary items are still taken into consideration to some extent.
Since its output sums to one, this invariably means less weight is assigned to the relevant items, potentially harming performance and interpretability.

`Sparse Attention Mechanism` is aimed at generating sparse attention distribution as a trade-off between soft attention and hard attention.


- [Generating Long Sequences with Sparse Transformers](https://d4mucfpksywv.cloudfront.net/Sparse_Transformer/sparse_transformers.pdf)
- [Sparse and Constrained Attention for Neural Machine Translation](https://www.aclweb.org/anthology/P18-2059.pdf)
- https://github.com/vene/sparse-structured-attention
- https://github.com/lucidrains/sinkhorn-transforme
- http://proceedings.mlr.press/v48/martins16.pdf
- https://openai.com/blog/sparse-transformer/
- [Sparse and Continuous Attention Mechanisms](https://papers.nips.cc/paper/2020/file/f0b76267fbe12b936bd65e203dc675c1-Paper.pdf)

#### Graph Attention Networks

GAT introduces the attention mechanism as a substitute for the statically normalized convolution operation. 

* [UNDERSTANDING ATTENTION IN GRAPH NEURAL NETWORKS](https://rlgm.github.io/papers/54.pdf)

<img src="https://dsgiitr.com/images/blogs/GAT/GCN_vs_GAT.jpg" width="70%"/>

- [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- https://petar-v.com/GAT/
- https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
- https://dsgiitr.com/blogs/gat/
- https://www.ijcai.org/Proceedings/2019/0547.pdf
  
### Transformer

> Representation learning forms the foundation of today’s natural language processing system; Transformer models have been extremely effective at producing word- and sentence-level contextualized representations, achieving state-of-the-art results in many NLP tasks. However, applying these models to produce contextualized representations of the entire documents faces challenges. 
> These challenges include lack of inter-document relatedness information, decreased performance in low-resource settings, and computational inefficiency when scaling to long documents. In this talk, I will describe 3 recent works on developing Transformer-based models that target document-level natural language tasks.


`Transformer` is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.


Transformer blocks are characterized by a `multi-head self-attention mechanism`, a `position-wise feed-forward network`, `layer normalization`  modules and `residual connectors`.

<img src="https://lena-voita.github.io/resources/lectures/seq2seq/transformer/model-min.png" width="90%"/>

* [Transformer结构及其应用--GPT、BERT、MT-DNN、GPT-2 - 知乎](https://zhuanlan.zhihu.com/p/69290203)
* [放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)
* [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
* [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning) ](http://jalammar.github.io/illustrated-bert/)
* [Universal Transformers](https://mostafadehghani.com/2019/05/05/universal-transformers/)
* [Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View](https://arxiv.org/abs/1906.02762)
* [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
* [A Survey of Long-Term Context in Transformers](https://www.pragmatic.ml/a-survey-of-methods-for-incorporating-long-term-context/)
* [The Transformer Family](https://lilianweng.github.io/lil-log/2020/04/07/the-transformer-family.html)
* https://www.idiap.ch/~katharas/
* https://arxiv.org/abs/1706.03762
* [Superbloom: Bloom filter meets Transformer](https://arxiv.org/abs/2002.04723)
* [Evolution of Representations in the Transformer](https://lena-voita.github.io/posts/emnlp19_evolution.html)
* https://www.aclweb.org/anthology/2020.acl-main.385.pdf
* https://math.la.asu.edu/~prhahn/
* https://arxiv.org/pdf/1802.05751.pdf
* https://arxiv.org/pdf/1901.02860.pdf
* [Spatial transformer networks](http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf)


- https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html
- [Transformers are graph neural networks](https://thegradient.pub/transformers-are-graph-neural-networks/)
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://linear-transformers.com/)
- [A Unified Understanding of Transformer's Attention via the Lens of Kernel](https://www.aclweb.org/anthology/D19-1443.pdf)


#### BERT 

* https://github.com/tomohideshibata/BERT-related-papers
* https://github.com/google-research/bert
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
* https://github.com/clarkkev/attention-analysis

#### GPT

- https://github.com/openai/gpt-2
- [Better Language Models and Their Implications](https://openai.com/blog/better-language-models/)
- https://openai.com/blog/image-gpt/


#### ViT

- https://github.com/gupta-abhay/ViT
- https://abhaygupta.dev/blog/vision-transformer
- https://arxiv.org/abs/2012.12556v3

### Predictive coding

- https://lorenlugosch.github.io/posts/2020/07/predictive-coding/
- https://github.com/miyosuda/predictive_coding

### Open Library

* https://huggingface.co/transformers/
* https://github.com/huggingface/transformers

#### Megatron-LM

- https://github.com/nvidia/megatron-lm

#### FastMoE

- https://github.com/laekov/fastmoe

#### DeepSpeed

DeepSpeed is a deep learning optimization library that makes distributed training easy, efficient, and effective.

- https://www.deepspeed.ai/

***
- [ ] [Applied Natural Language Processing](https://bcourses.berkeley.edu/courses/1453620/assignments/syllabus)
- [ ] [HanLP：面向生产环境的自然语言处理工具包](http://www.hanlp.com/)
- [ ] [Topics in Natural Language Processing (202-2-5381) Fall 2018](https://www.cs.bgu.ac.il/~elhadad/nlp18.html)
- [ ] [CS 124: From Languages to Information Winter 2019 Dan Jurafsky](https://web.stanford.edu/class/cs124/)
- [ ] [CS224U: Natural Language Understanding](https://web.stanford.edu/class/cs224u/)
- [ ] [CS224n: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
* [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/)
* [CS224n: Natural Language Processing with Deep Learning, Stanford / Winter 2019](https://web.stanford.edu/class/cs224n/)
* [Deep Learning for NLP](https://www.comp.nus.edu.sg/~kanmy/courses/6101_1810/)