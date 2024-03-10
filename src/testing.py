import autograd.numpy as np
from scipy.sparse import bsr_array
from energy import Energy
from newtons import newtons_method

class TestCase():
    def __init__(self, l, p, mg, k, solution):
        self.l = l
        self.p = p
        self.mg = mg
        self.k = k
        self.solution = solution
        self.test_cases = []
        self.E = Energy(l, p, mg, k)

    def objective(self, x):
        '''
        The objective function of the problem.
        Must be overriden for each problem.
        '''
        pass

    def run(self, initial_x):
        '''
        Run optimization algorithm here.
        Must be overriden for each problem.
        '''
        pass

    def test(self, error = 10e-3):
        '''
        Run all test cases
        '''
        for i in range(len(self.test_cases)):
            result = self.run(self.test_cases[i])
            print(f"Test #{i + 1}:", end=" ")
            
            # TODO:
            if np.linalg.norm(result - self.solution) > error:
                print(f"FAILED - Returned {result} but solution was {self.solution}")
                continue

            print("success")

class TestCase1(TestCase):
    def __init__(self):
        # Define l as (sparse) matrix
        # Two nodes i and j are connected if l[i][j] > 0 
        r = np.array([0, 1, 2, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7])
        c = np.array([4, 5, 6, 7, 0, 5, 7, 1, 4, 6, 2, 5, 7, 3, 4, 6])
        data = np.array([3] * len(r))

        # Fixed parameters
        super().__init__(
                    l = bsr_array((data, (r, c)), shape=(8,8)).toarray(),
                    p = np.array([(5, 5, 0), (-5, 5, 0), (-5, -5, 0), (5, -5, 0)]), 
                    mg = [1/6] * 8, 
                    k = 3,
                    solution = [(2, 2, -3/2), (-2, 2, -3/2), (-2, -2, -3/2), (2, -2, -3/2)]
                )

        # Define initial conditions for unit testing 
        # TODO: Pass as parameter (?)
        self.test_cases = [
            np.array([(10., 1., 0.), (-1., 1., 0.), (-6., -1., 0.), (6., -1., 0.)])
        ]

    def objective(self, x):
        total = self.E.ext(x)
        for i in range(len(self.l)):
            for j in range(i, len(self.l)):
                total += self.E.cable_elast(x, i, j)
        return total

    def run(self, x0):
        # TODO: Run optimization algorithm here
        return newtons_method(self.objective, x0)

case = TestCase1()
case.test()
