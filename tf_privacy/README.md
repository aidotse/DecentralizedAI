# Differential Privacy together with TensorFlow 2.x.

This tutorial exemplifies the use of the [TensorFlow Privacy](https://github.com/tensorflow/privacy) together with the model structure as recommended for [Tensorflow 2.x.](https://www.tensorflow.org/)

## Dependencies

This tutorial uses [TensorFlow](https://www.tensorflow.org/) to define machine learning models using TensorFlow 2.x (i.e., no use of traditional `tf.Estimators`, etc., whatsoever!). An installing of TensorFlow (>= 2.0) is, therefore, a pre-requisite. You can find instructions [here](https://www.tensorflow.org/install/). For better performance, it is also recommended to install TensorFlow with GPU support (detailed instructions on how to do this are available in the TensorFlow installation documentation).

In addition, the following prerequisites are required:

* `scipy`

* `tensorflow-privacy`
 
Both libraries can easily be installed by:

```
pip install scipy tensorflow-privacy
```

## Usage

## Results

![](./images/mlp_no-dp.png "Multi-Layer Perceptron (MLP) without Differentially Privacy (DP)") ![](./images/mlp_dp.png "Multi-Layer Perceptron (MLP) with Differentially Privacy (DP)") 

![](./images/cnn_no-dp.png "Convolutional Neural Network (CNN) without Differentially Privacy (DP)") ![](./images/cnn_dp.png "Convolutional Neural Network (CNN) with Differentially Privacy (DP)") 


## References
