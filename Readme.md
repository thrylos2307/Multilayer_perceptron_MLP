# MLP
****
## Overview
Multilayer perceptron is field of artificial neural networks ,which works in feedforward mechanism, Neural networks are simplified versions of the human brain
and consist of input, hidden and output layers . The objective of the neural network is to transform the
inputs into meaningful outputs. Multi-layer perceptron
(MLP) is the most common neural network model.
I have used the [sklearn wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine) dataset for the model.<br>
_____
## Model
MLP model :

<p align="center">
  <img width="600" height="300" src="img/neurons.jpg">
</p>
<p align="center"><strong>FIGURE 1:</strong> NN layer.<p align="center">
The neurons are arranged in different
layers, in each layer the nodes receive inputs only from the
nodes in the preceding layer and pass their outputs only to
the nodes of the next layer. <br>

1. Input layer consist of input features of a sample of dataset on which we will train our model.
2. Hidden layer consist of neurons in which weight function applies and then forward them to next layer after applying activation function.
3. Output Layer consist of neurons = number of classes that we have for one vs all model.

____
## Weights and shape
<p align="center">
  <img width="600" height="250" src="img/shape.jpg">
</p>
<p align="center"><strong>FIGURE 2:</strong> Input feature matrix and weight matrix.<p align="center">

The input feature matrix(X) consist of all sample points as shown above ,each column consist of feature: let we have n input feature, each row consist of a sample of dataset :let we have m samples. So input matrix would have shape m x n (X<sub>mxn</sub>).

**Weights**:This is the basic component of NN where we intialized them ,transform the input to next layer and  get the results after applyting activation and update them as to minimize the error.

Our feed forward equation would be :
<img src="https://render.githubusercontent.com/render/math?math=\beta%2b\theta_{1}x_{1}%2b\theta_{2}x_{2}....%2b\theta_{n}x_{n}"> for 1 neuron as we do in regression,where <img src="https://render.githubusercontent.com/render/math?math=\theta=weights">.
So 1 neurons would get input from all the neurons of previous layer and if we are having k neurons in layer L and lets say we have p neurons from previous layer ,thus all neurons from previous layer would feed forward their inputs, hence we would come up with the weight matrix for L with the shape k x p(W<sub>k x p</sub>)  and similarly for bias weights (k x 1).

To avoid vanishing/exploding problem and the weights doesn't turn to be 0 or inf after few iterations,i initialize them with Yoshua Bengio weight initialization.
<img src="https://render.githubusercontent.com/render/math?math=W^{l}=np.random.rand(shape)*(\sqrt{1/(neurons\_in\_layer\[l\]%2b neurons\_in\_layer\[l-1\])})"><br>
One can use the Xavier initialization also.

____
## Forward Propagation

Each neuron would have 2 component, feedforward(Z) and activation function(A) this together consist of forward propagation as shown in Figure 1.<br>

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=Z=W*X%2bb">
</p>
The size of W is k x p ,and input matrix is n x m, here p=n (p=previous layer nuron)  and b is also having size k x 1,so it would be broadcast to W*X,thus Z would come to be size of k x m.
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=Z_{kxm}=W_{kxn}*X_{nxm}%2bb_{kx1}">
</p>
For input layer X(mxn) so we would transpose it.<br>
Second component is activation function(A) , there are many types of activation function like : sigmoid,relu,tanh,softmax,leakyrelu. 

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=A_{kxm}=G(Z_{kxm})"><br>
where is G activation function sigmoid/Relu/tanh..
</p>
Now will forward this activation value as input to next layer so general equation would be :
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=Z^{l}=W^{l}*A^{l-1}%2bb^{l}"><br>
where <strong>l</strong> is current layer 
</p>

<p align="center">
  <img width="600" height="350" src="img/output.jpg">
</p>
<p align="center">FIGURE 3: Output<p align="center">

Finally we would get the results from output layer after forward propagation from all layer ,let say we have 4 class classes in our dataset then in output layer would have shape (4 x 1) for one sample input, then for m samples we would get the output matrix of shape 4 x m.  For e.g [0 0 1 0],shows sample belongs to class 3.<br>
 We would use sigmoid for last layer and then by defining threshold(say 0.5) we can classify for each output of neuron to be 0 or 1. It might be the case that we can get more 1 in output as after sigmoid fucnction we get more values to be greater than 0.5 e.g.[0 1 1 1],then this is case of miss classification and then to get the more accurate result we need to update the weights such that this missclassification can be decrease.
 
**Note:** I have declare mapp function for the threshold ,one can go with it or can go without threshold value as the cost comes we can go with the highest value among all the values of outcome as we are implementing one-vs-all.  
____
## Backword Propagation
In order to update the weights we need to find the error ,which can be find using loss function , i am using log loss function .


<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=L=-(Y*log(A^{output})%2b(1-Y)*log(1-A^{output}))"><br>
</p>

As we need to minimize the error ,we need to update weights accordingly. We update the weights from output layer in backword direction, to get best weigths ,bias,Z and activation , we will take derivative of loss fuction wrt to W,b,Z,activation.
1. <img src="https://render.githubusercontent.com/render/math?math=dA=(dL/dA)=-Y/A%2b(1-Y)/(1-A)">
2. <img src="https://render.githubusercontent.com/render/math?math=dZ=(dL/dA)*(dA/dZ)=(-Y/A%2b(1-Y)/(1-A))*(1-A)*A=A-Y">

dA/dZ=derivative of activation function ,so for the output layer we can use dZ=A-Y when sigmoid is used in last layer<br>

3. <img src="https://render.githubusercontent.com/render/math?math=dW=(dL/dA)*(dA/dZ)*(dZ/dW)=dZ*(dZ/dW)=dZ*(A^{previous_layer})">
4. <img src="https://render.githubusercontent.com/render/math?math=db=(dL/dA)*(dA/dZ)*(dZ/db)=dZ*(dZ/db)=dZ*1">

so,finally we would have :<br>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=dZ^{output\_layer}=A^{predicted}-Y^{given}">
</p>
for hidden layer:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=dZ^{l}=(W^{l%2b1})*(dZ^{l%2b1})*derivative\_of\_activation">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=dW^{l}=dZ^{l}*A^{l-1}">
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=db^{l}=dZ^{l}">
</p>

dZ=A-Y, The shape of A and Y should be same if A=[class1 class2 class3 class4] ,then also Y=[class1 class2 class3 class4] ,in order to get series(all sample output class(Y))into this ,we'll use [one hot endcoding](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html). 

Uncomment the lines in [t_mlp.ipynb](t_mlp.ipynb) in training part to take input for number of hidden layer,number of neurons in each hidden layer, activation used for a particular layer. If one does not input any hidden layer the model would work as single layer perceptron.



