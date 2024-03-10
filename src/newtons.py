from autograd import jacobian
from scipy import linalg

def newtons_method(f, x0):
    'Return a single step of newtons method'
    J = jacobian(f)(x0)

    print("Jacobian: ", J)

    xn = x0 - linalg.inv(J)*f(x0)
    print("Next: ", xn)

    return xn