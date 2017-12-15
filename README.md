# svd-softmax
svd-softmax implemented in Tensorflow by [Koichiro Tamura](http://koichirotamura.com/)

## Paper information

- [SVD-Softmax: Fast Softmax Approximation on Large Vocabulary Neural Networks](https://papers.nips.cc/paper/7130-svd-softmax-fast-softmax-approximation-on-large-vocabulary-neural-networks)
- Kyuhong Shim, Minjae Lee, Iksoo Choi, Yoonho Boo, Wonyong Sung
- published at NIPS 2017


>We propose a fast approximation method of a softmax function with a very large vocabulary using singular value decomposition (SVD). SVD-softmax targets fast and accurate probability estimation of the topmost probable words during inference of neural network language models. The proposed method transforms the weight matrix used in the calculation of the output vector by using SVD. The approximate probability of each word can be estimated with only a small part of the weight matrix by using a few large singular values and the corresponding elements for most of the words. We applied the technique to language modeling and neural machine translation and present a guideline for good approximation. The algorithm requires only approximately 20\% of arithmetic operations for an 800K vocabulary case and shows more than a three-fold speedup on a GPU.



## Requirements

- Tensorflow 1.4

## Why this project?

Since it is very important to redece calculation cost at softmax output in NL tasks, I tried to implement the idea in [SVD-Softmax: Fast Softmax Approximation on Large Vocabulary Neural Networks](https://papers.nips.cc/paper/7130-svd-softmax-fast-softmax-approximation-on-large-vocabulary-neural-networks).


## room for improvement 

### No gradient defined for operation SVD

SVD(singular value decomposition) method in Tensorflow [tf.svd()](https://www.tensorflow.org/api_docs/python/tf/svd) don't support gradient function in Tensorflow Graph. This means that you have to use other training method like NCE.  

### more efficient codes for update Top-N words

Since tensorflow uses static graph, it is difficult to update words by full-view vector multiplication.
If you can know more efficient way to implement, please tell me.

