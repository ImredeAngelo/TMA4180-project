import autograd.numpy as np

# TODO: It might be worth ignoring the fixed points p, as they 
#       always have the same effect on the objective function
class Energy:
    def __init__(self, l, p, mg, k = 3):
        assert(len(l) == len(mg))

        self.l = l
        self.p = p
        self.k = k
        self.mg = mg

    def cable_elast(self, x, i, j):
        '''The elastic energy of the cable connecting i and j'''
        x = np.concatenate((self.p, x))
        L = self.l[i][j]
        if(L <= 0):
            return 0

        d = np.linalg.norm(x[i] - x[j])
        if d < L:
            return 0
        else:
            return self.k/(L**2) * (d - L)**2
        
    def ext(self, x):
        '''Get the external load of the system'''
        x = np.concatenate((self.p, x))
        E = 0
        for i in range(len(self.l)):
            E += self.mg[i] * x[i][2]
        return E