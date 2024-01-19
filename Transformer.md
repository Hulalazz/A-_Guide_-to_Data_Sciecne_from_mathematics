# Transformer

* [Transformer结构及其应用--GPT、BERT、MT-DNN、GPT-2 - 知乎](https://zhuanlan.zhihu.com/p/69290203)
* [放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)


`Transformer` is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.


Transformer blocks are characterized by a `multi-head self-attention mechanism`, a `position-wise feed-forward network`, `layer normalization` modules and `residual connectors`.

<img src="https://lena-voita.github.io/resources/lectures/seq2seq/transformer/model-min.png" width="90%"/>


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
- [Understanding Graph Neural Networks from Graph Signal Denoising Perspectives CODE](https://arxiv.org/abs/2006.04386)
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