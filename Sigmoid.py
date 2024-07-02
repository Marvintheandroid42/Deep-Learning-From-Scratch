class Sigmoid():

  def activation(self, x):

    return 1 / (1 + np.exp(-1 * x))

  def forward(self, X):

    #X is of the shape (output_dim of dense layer, n)

    self.input = X

    #Output is shape (output_dim, n)

    return self.activation(X)

  def backward(self, input_grad): #input_grad is of the shape (output_dim, n)

    #(output_dim, n) .* (output_dim, n) = (output_dim, n)

    output_grad = self.activation(self.input) * self.activation(1 - self.input)

    return input_grad * output_grad