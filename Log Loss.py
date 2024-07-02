class Log_Loss(): #need to compute the forward propogation druing every epoch

  def forward(self, y_hat, y): #both arrays should have the shape (1, n)

    if y.shape[0] != 1:

      y = y.T

    
    if y_hat.shape[0] != 1:

      y_hat = y_hat.T
    

    self.y_hat = y_hat

    self.y = y

    return -1 * np.mean(y * np.log(y_hat) + (1-y)*np.log(1-y_hat), axis=1)

  
  def backward(self):

    return -1 * ((self.y / self.y_hat) - ((1-self.y)/(1-self.y_hat))) #(1, n) shape 
    