# Differential Privacy together with TensorFlow 2.x.

This tutorial exemplifies the use of the [TensorFlow Privacy](https://github.com/tensorflow/privacy) together with the model structure as recommended for [Tensorflow 2.x.](https://www.tensorflow.org/)

### What is Differential Privacy? 

The general aim of _differential privacy_ is to avail sensitive data, e.g., data containing people's personal information, without divulging the personal identification of individuals. This can be achieved by introducing a minimum distraction to the data of a sensitive dataset. 

&emsp; &emsp; _"  ...differential privacy forms data anonymously via injecting noise into the dataset studiously."_
     
The introduced distraction should be immense enough that it is capable of protecting privacy, while at the same time limited enough so that the data is still useful for analysis.

In the case of learning algorithms, differential privacy is, practically, achieved by injecting noise during the training of a model. For example, in the case of _Differentially Private Stochastic Gradient Descent_ (DP-SGD)[[1]](#references) (as used in the code examples in this repository), the approach for bolster _differential privacy_ is to apply noise to the computed _gradients of the loss_ (with respect to the model parameters), using the following steps:

1. Clip gradients, per training example included in a batch, to ensure each gradient has a known _maximum Euclidean norm_.

2. Add _random noise_ to the clipped gradients.

Consequently, there are two parameters that affect the _privacy_ in this approach:

* __L2 Norm Clipping__: the cumulative gradient across all network parameters, from each batch, will be clipped so that its _L2 Norm_ is at most this value.

* __Noise Multiplier__: governs the amount of noise added during training. Generally, more noise results in better privacy and lower utility.

## Dependencies

In addition to an installation of TensorFlow 2.x (see general dependencies [Decentralized AI](https://github.com/aidotse/DecentralizedAI)), the following prerequisites are also required in order to train TensorFlow models using differential privacy:

* `scipy`

* `tensorflow-privacy`
 
Both libraries can easily be installed by:

```
pip install scipy tensorflow-privacy
```

## Usage

The example code in this repository is using the _MNIST_ dataset (for now), to illustrate how the training metrics (_loss_ and _accuracy_) are affected by the use of _differential privacy_. The code example support (for now) two types of model structures: 

1. Fully connected _Multi-Layer Perceptron_ (MLP), and... 

2. Simple _Convolutional Neural Network_ (CNN). 

All model structures are befined (using the [TensorFlow Functional API](https://www.tensorflow.org/guide/keras/functional)) in the file `model.py`. Training a model and visualizing the training metrics is simply done by:

    python3 train.py

The above command will train a model with default parameters. However, both the use of model, as well as parameters regarding the training of the model, can be changed through various command-line arguments:

*  `--model {mlp,cnn}` &emsp; &emsp;...type of learning model used, either _MLP_ or _CNN_ (_datatype:_ `string`, _default_: `mlp`).  

*  `--opt {sgd,adam}` &emsp; &emsp; ..._optimizer_ used for training, either _SGD_ or _Adam_ (_datatype:_ `string`, _default_: `sgd`).

*  `--lr N` &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; ..._learning rate_ for the training  (_datatype:_ `float`, _default_: `0.1`).

*  `--epochs N`&emsp; &emsp; &emsp; &emsp;...number of training _epochs_ (_datatype:_ `int`, _default_: `15`) 

*  `--batch-size N` &emsp; &emsp;  ...training _batch size_ (_datatype:_ `int`, _default_: `32`)

*  `--dp` / `--no-dp` &emsp; &emsp; ...using _differential privacy_ or not (_default_: `dp`).

In addition, there are two command-line arguments specific for training with _differential privacy_ (se section __What is Differential Privacy?__ above):
  
*  `--l2_norm_clip N` &emsp; &emsp; ...clipping norm (_datatype:_ `float`, _default_: `0.9`).

*  `--noise_mult N` &emsp; &emsp; &emsp;...ratio of the _standard deviation_ to the clipping norm (_datatyp:_ `float`, _default_: `0.9`).

For example, training a _CNN model_ with _differential privacy_ using an _SGD optimizer_, can be done by the following command:

    python3 train.py --model cnn --opt sgd

  ...while traning a "vanilla" _MLP model_ (i.e. without _differential privacy_) using _Adam optimizer_ with a _learning rate of 0.01_, can be done by the following command:
  
    python3 train.py --model mlp --opt adam --lr 0.01 --no-dp 

## Results

The images below are a couple of examples of resulting outputs of running the example code with various settings. In all the cases, an _SGD optimizer_ was used for traning. In the case of _training without differential privacy_, the remaining parameters were used as default. In the case of _training with differential privacy_, the _learning rate_ was increased to _0.5_ (to prevent the training from getting stuck in a "local minimum"). Also, in the case of _training with differential privacy_, the _batch size_ was increased to _300_ (as for training with differential privacy, the batch size should, preferably, be evenly dividable with the total number of training samples, e.g., evenly dividable with _60000 training samples_ in the case of the _MNIST dataset_). 

![](./images/mlp_no-dp.png "Multi-Layer Perceptron (MLP) without Differentially Privacy (DP)") ![](./images/mlp_dp.png "Multi-Layer Perceptron (MLP) with Differentially Privacy (DP)") 

![](./images/cnn_no-dp.png "Convolutional Neural Network (CNN) without Differentially Privacy (DP)") ![](./images/cnn_dp.png "Convolutional Neural Network (CNN) with Differentially Privacy (DP)") 

### How to Measure Privacy? 

In the examples above, the measure of _privacy_ is expressed through `epsilon`. Practically applied, the measure for _differential privacy guarantee_ is expressed through the following two parameters [[2]](#references):

* `delta` -- bounds the _probability of privacy guarantee_ not holding. A fixed value set, as a rule of thumb, to be less than the inverse of the training data size (i.e., the population size).

* `epsilon` -- measures the _strength of privacy guarantee_. In the case of differentially private machine learning, it gives a bound on how much the probability of particular model output can vary by including (or removing) a single training sample.  

Given a fixed _delta_ value for a model, the _epsilon_ is calculated based on the _batch size_, _noise multiplier_, and the umber of _trained epochs_. Hence, as seen in the examples above, the _strength of privacy guarantee_ (_epsilon_) of a trained model is linearly proportional to the number of trained epochs - meaning an increased probability of variety in model output, for a singular training sample, with respect to the number of trained epochs. 

Interpreting the _epsilon_ value is, however, at times difficult. The value is merely an _indication of privacy_ for a trained model. In reality, the likelihood of a certain output for a single training sample must further be confirmed, e.g., by purposely insert secrets in the training dataset and measure the likelihood that those secrets are leaked during inference [[3]](#references).

## Further Reading

A detailed tutorial regarding the technical details of _Machine Learning with Differential Privacy in TensorFlow_ can be found in this [blog](http://www.cleverhans.io/privacy/2019/03/26/machine-learning-with-differential-privacy-in-tensorflow.html).

## References
[1] M. Abadi, et al. ["Deep learning with differential privacy."](https://dl.acm.org/doi/pdf/10.1145/2976749.2978318?casa_token=HLroUey_9GQAAAAA:XJpCJz8GF9AZFuOaMoDEqy-aKWpnYUKBHhPy1bwvP709x0l6ofIs_NuhAyhd5pDsxxOxBwLc_kk) Proceedings of the 2016 ACM SIGSAC conference on computer and communications security. 2016.

[2] I. Mironov, T. Kunal, and Z. Li. ["Rényi Differential Privacy of the Sampled Gaussian Mechanism."](https://arxiv.org/pdf/1908.10530.pdf) arXiv preprint arXiv:1908.10530. 2019.

[3] N. Carlini, et al. ["The secret sharer: Evaluating and testing unintended memorization in neural networks."](https://www.usenix.org/system/files/sec19-carlini.pdf) 28th USENIX Security Symposium (USENIX Security 19). 2019.
