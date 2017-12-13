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

Since It is very importance to redece calculation cost at softmax output in NL tasks, I tried to implement the idea in SVD-Softmax: Fast Softmax Approximation on Large Vocabulary Neural Networks](https://papers.nips.cc/paper/7130-svd-softmax-fast-softmax-approximation-on-large-vocabulary-neural-networks).
However, there are some problems at the implement of svd-softmax in Tensorflow. So I would like to discuss them and I would like someone to tell me the solutions. 

## Problems to solve

### No gradient defined for operation SVD

SVD(singular value decomposition) method in Tensorflow [tf.svd()](https://www.tensorflow.org/api_docs/python/tf/svd) don't support gradient function in Tensorflow Graph. If you would like to use SVD-softmax in training, you have to implemnt trainable svd-function by yourself.  

### Too slow SVD-softmax in GPU

Even when using svd-softmax in evaluation, calculation of svd-softmax is too slow.
For example, I tried to use svd-softmax in [Transformer](https://arxiv.org/abs/1706.03762) using following hyperparameters or enviroments.

- vocabulary size = 30000
- hidden units = 256
- window size = 256
- num of full view = 2048
- JPO Japanses-Chinese corpus (1 milion pairs) 
- 4x TITAN X (Pascal) (Liquid cooling
- Ubuntu 16.04.1 LTS


However, the calculation time is as follows in my experiments.

- calculation full-softmax(codes are as follows): about 0.4sec

```
logits = tf.matmul(self.dec_output, tf.transpose(self.weights))
logits = tf.nn.bias_add(logits, self.biases)
logits = tf.nn.softmax(logits)
```
- tf.svd() line 32 in svd_softmax.py: about 2.5sec

That is, calucation of SVD is slower than the speed of calculation of full-softmax.

I don't know how to deal with this problem, so please tell me the solution if you can.

