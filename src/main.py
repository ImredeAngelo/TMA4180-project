import numpy as np
from energy import Energy
    
def problem_1(p, x):
    '''
    Defines test case 1
    
    x - Initial conditions
    '''
    solution = [(2, 2, -3/2), (-2, 2, -3/2), (-2, -2, -3/2), (2, -2, -3/2)] # Used for validation

    # Parameters
    mg = [1/6] * len(l)
    l = [
        [ 0, 0, 0, 0, 3, 0, 0, 0 ], # 1 - 5
        [ 0, 0, 0, 0, 0, 3, 0, 0 ], # 2 - 6
        [ 0, 0, 0, 0, 0, 0, 3, 0 ], # 3 - 7
        [ 0, 0, 0, 0, 0, 0, 0, 3 ], # 4 - 8
        [ 3, 0, 0, 0, 0, 3, 0, 3 ], # 5 - 1, 6, 8
        [ 0, 3, 0, 0, 3, 0, 3, 0 ], # 6 - 2, 5, 7
        [ 0, 0, 3, 0, 0, 3, 0, 3 ], # 7 - 3, 6, 8
        [ 0, 0, 0, 3, 3, 0, 3, 0 ]  # 8 - 4, 5, 7
    ]

    # Objective function
    def objective(x):
        energy = Energy(l, p, x, mg, 3)
        total = energy.ext()
        for i in range(len(l)):
            for j in range(i, len(l)):
                total += energy.cable_elast(i, j)
        return total

    # Debugging
    print(objective(x))
    print(objective(solution))

    return np.concatenate((p, x)), np.concatenate((p, solution)), l