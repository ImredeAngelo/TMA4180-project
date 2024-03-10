from autograd import jacobian, numpy as np
from scipy import linalg

def newtons_method(f, x0, max_iterations = 4):
    '''TODO (This method does not work since the jacobian is not invertible)'''
    x = x0
    for i in range(max_iterations):
        J = jacobian(f)(x0) # Get pseudoinverse
        invJ = np.append(linalg.pinv(J), [0] * len(x0)).reshape(len(x0), len(x0))
        x = x - (invJ @ x)

        print(f"{i + 1}: ", x, sep = "\n", end="\n\n")
    
    return x