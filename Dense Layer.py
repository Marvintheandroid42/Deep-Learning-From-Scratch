import numpy as np

class Dense():

  def __init__(self, input_dim, output_dim):

    #using the random normal sampling to get weights -1 < x < 1

    self.W = np.random.randn(input_dim, output_dim)
    self.b = np.random.randn(output_dim, 1)

    self.input_dim = input_dim
    self.output_dim = output_dim

  def forward(self, X):

    #making sure the input matrix is of the shape (input_dim, n)

    if X.shape[0] != self.input_dim:

      X = X.T

    self.input = X

    # Weighted sum gives an output with shape (output_dim, n)

    z = np.dot(self.W.T, X) + self.b

    return z
  
  def backward(self, input_grad, learning_rate): #shape of input grad from the activation is (output_dim, n)

    weight_grad = (1/input_grad.shape[1]) * np.dot(self.input, input_grad.T) #(input_dim, output_dim)

    #need to add the (1/input_grad.shape[1]) as the dot product is the aggregate of all the data points
    #so in order to take the mean we need to divide by the number of data points as the sum for the mean 
    #is already done by the dot product, need to carry the 1/n term from the loss function into the update

    bias_grad = np.mean(input_grad, axis=1).reshape(-1,1) #(output_dim, 1)

    #we dont need to carry the term for the bias as we are already taking the mean using the numpy function!

    output_grad = np.dot(self.W, input_grad) #(input_dim, n)

    self.W = self.W - learning_rate * (weight_grad)

    self.b = self.b - learning_rate * (bias_grad)

    return output_grad
