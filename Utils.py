#UTILS 

def logits_to_classes(y_hat, threshold):

  y_hat[np.where(y_hat >= threshold)] = 1

  y_hat[np.where(y_hat < threshold)] = 0

  return y_hat

def accuracy(y_hat, y, verbose=True):

  if y_hat.shape[1] != 1:

    y_hat = y_hat.T
  
  if y.shape[1] != 1:

    y = y.T


  acc = np.round(len(np.where(y_hat == y)[0]) / len(y), decimals=3) * 100

  if verbose == True:

    print('ACCURACY: ', acc, '%')

  return acc