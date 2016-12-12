# credit-classifier

## Introduction 
This is a small tech demonstration of analyzing credit data from Hamburg University. The analyzer can analyze some data collected by a bank giving a loan. The dataset consists of 1000 datapoints of categorical and numerical dataas well as a good credit vs bad credit metric which has been assigned by bank employees. The dataset is fully anonymized. The code in this repository contains a linear regression and a neural network machine learning models to try to predict the credit rating that a bank employee would assign given some datapoints.

## Requirements
Python<br>
TensorFlow<br>
Scikit Learn

## The dataset
The dataset consists of 1000 datatpoints each with 20 variables (dimensions) 7 are numerical and 13 are categorical. The categorical data is encoded according to terms which have meaning to the bankers such as current employment timeframe:

* A71 : unemployed
* A72 :       ... < 1 year
* A73 : 1  <= ... < 4 years
* A74 : 4  <= ... < 7 years
* A75 :       .. >= 7 years
	      
Or what kind of housing the person has:

* A151 : rent
* A152 : own
* A153 : for free
	         
Other data provided in the dataset contains numerical information such as:

* Duration of the loan in months
* Credit amount
* Age in years

See the Description file attached to the repository for further details on the data. Source:

Professor Dr. Hans Hofmann<br>
Institut für Statistik und Ökonometrie<br>
Universität Hamburg<br>
FB Wirtschaftswissenschaften<br>
Von-Melle-Park 5<br>
2000 Hamburg 13

### Dataset preprocessing  
##### Categorical Data
Because the majority of this dataset is categorical data we must first digitize the categorical variables into numbers from which our ML (machine algorithms) can learn. The most common way of encoding categorical variables is One-Hot-Encoding<br>
 [One-Hot on Wikipedia](https://en.wikipedia.org/wiki/One-hot)<br>
 [scikit learn encoding categorical variable](http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)<br>
 
One-Hot encoding is basically assigning a binary state machine state to each unique value in the set of categorical variables.For example if there are 3 categorical variables {A151, A152, A153} they will be encoded as {001, 010, 100}. This is important because since we cannot easily estimate how close or how far the variables should be to each other if they were on the same axis. The downside of One-Hot encoding is that it increases the number of dimensions and therefore increases the computational complexity of any models using the data. 

##### Binary Categorical Data

An exception to the above rule is when the categorical variable consists of only two unique values. In that case the difference can be encoded as either 0 vs 1 or -1 vs +1 with the latter being better for building ML models.

##### Numerical Data

Numerical data should be normalized even though it is not strictly neccessary, but it has been shown that normalization of numerical data can lead to faster training of neural network weights.

#### Implementation of dataset preproccessing
You can find the implementation of my dataset preprocessor which does the above three operations in createNumericalData.py

## Classification

### Linear regression
A linear regression simply tries to linearly separate the model by finding an equation of a line which can match some ideal parameters (to be found by a cost function). The linear regression can be modeled by finding a combination of **W** (weights) and b (bias term) which when combined can give us the line separating our data.

**Y** = **X** * **w** + b

Where: **X** = {x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>} and **W** = {w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>n</sub>}

Therefore **Y** can be expanded to:

**Y** = w<sub>1</sub>x<sub>1</sub> + w<sub>1</sub>x<sub>2</sub> + ... + w<sub>n</sub>x<sub>n</sub> + b

#### Finding optimal w and b
Let: w<sub>0</sub> = b <br>
The bias term b can be thought of as part of the weights vector with degree of zero. The optimal weights can be found using a classical method called least square error minimization. The sum of the square errors can be calculated with the following equation:

E(w) =  **Σ**<sub>1</sub><sup>n</sup> (y<sub>i</sub> - x<sub>i</sub><sup>T</sup>w)<sup>2</sup> = (y-Xw)<sup>T</sup>(y-Xw)

Finding the vetor w that minimizes the above formula is the solution that offers a best fit model. This can be solved for using a gradient descent method.

#### Cost Matrix
The  dataset requires the use of a cost matrix (see below). In the credit/loan problem it is worse to predict a customer as good when he is bad (false positive) vs predicting a customer as bad when he is good (false negative).

|          |Good |   Bad  |
|:--------:|:---:|:------:|
| **Good** | 0   |     1  |
|  **Bad** | 5   |    0   |

This cost matrix has been incorporated into the naive least square errors function described above. The false positives are multiplied by 5 and false negatives are multiplied by 1.

See code in trainLinearRegression.py

### Neural Network

```
            Hidden     Hidden
Input       Layer1     Layer2
Layer      +-----+    +-----+
           |     |    |     |
+-----+    |     +--> |     |
|     +--> +-----+    +-----+
|     |    +-----+    +-----+     Output
+-----+    |     +--> |     |     Layer
+-----+    |     |    |     |    +-----+
|     |    +-----+    +-----+    |     |
|     +--> +-----+    +-----+ +> |     |
+-----+    |     |    |     |    +-----+
+-----+    |     +--> |     |    +-----+
|     |    +-----+    +-----+    |     |
|     +--> +-----+    +-----+ +> |     |
+-----+    |     |    |     |    +-----+
+-----+    |     +--> |     |
|     +--> +-----+    +-----+     2 neurons
|     |    +-----+    +-----+
+-----+    |     +--> |     |
           |     |    |     |
 59 n      +-----+    +-----+

            124 n      124 n
```

The neural network is built according to the above diagram. There are 59 neurons associated with the input variables (one for each feature in the adjusted dataset), there are two hidden layers of 124 neurons each and finally an output layer corresponding to the 2 classes (bad credit vs good credit). Each one of the neurons is connected with all neurons in the previous and next layers.

The network is initialized with random weights from a Gaussian distribution with {μ = 0, σ = 0.1}, a learning rate of 10<sup>-4</sup> and 3000 epochs of training for each fold and is trained by minimizing the cost function described using an Adam Optimizer [3].

See code in trainNeuralNet.py

### 10-Fold cross validation

To use a fair comparison of the performance of the two networks I am using 10-Fold cross validation and only recording the values from the validation set. There should not be any overfitting in the 

## Results

#### Linear regression model

Mean validation precision of 10 runs: 0.703<br>
Standard deviation: 0.0702<br>
Confusion matrix:

|          | Good |  Bad |
|:--------:|:----:|:----:|
| **Good** |  60  |  10  |
|  **Bad** |  15  |  15  |


#### Neural network model
Much better results from the neural network:

Mean validation precision of 10 runs: 0.903<br>
Standard deviation: 0.124<br>
Confusion matrix:

|          | Good |  Bad |
|:--------:|:----:|:----:|
| **Good** |  67  |   3  |
|  **Bad** |   5  |  25  |

#### Results Table
|                             | Precision |   Recall  |  F-Score  |
|:---------------------------:|:---------:|:---------:|:---------:|
| **Linear regression model** |   0.703   |    0.678  |   0.683   |
|   **Neural network model**  |   0.902   |    0.890  |   0.895   |

The very small standard deviations included in the results signify that the results are repeatable with respect to the folds used in k-fold cross validation.

## Analysis 
Neural networks, even a simple one as above can both obtain a greater overall accuracy and minimize the number of false positives  when compared to linear regressions. Neural nets can be easily improved by adding more layers. The only downside of neural nets is that it is harder to describe which features are considered important by neural nets as opposed to by a single weight vector in a linear regression.


## References 
[1] Sola, J., & Sevilla, J. (1997). Importance of input data normalization for the application of neural networks to complex industrial problems. Nuclear Science, IEEE Transactions on, 44(3), 1464-1468.<br>
[2] Duan, K., Keerthi, S. S., Chu, W., Shevade, S. K., & Poo, A. N. (2003). Multi-category classification by soft-max combination of binary classifiers. In Multiple Classifier Systems (pp. 125-134). Springer Berlin Heidelberg.<br>
[3] Kingma, D., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.[Link](http://arxiv.org/pdf/1412.6980v7.pdf)<br>

