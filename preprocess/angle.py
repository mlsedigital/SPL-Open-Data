## Mostly a script to return the kinematic angles. 

## Time to flex that mat223/4 shit

## The idea is that we can effectively find the angle between any 3 points (A,B,C)
## retrieve A->B = B-A, B->C = C-B
## A->B dot B->C =  |B->C|*|A->B| cos(theta) rearrange to find theta

## Gonna add other .py files here when I get the chance. 

import numpy as np
import math

def angle_finder(A, B, C):
    """
    Retrieves the angle between points, A,B,C, by using the dot product

    Args:
        A: (np.array) An array of size 3
        B: (np.array) An array of size 3
        C: (np.array) An array of size 3
    
    Return:
        angle: (np.float) the angle between the 3 points
    """
    A_B = A-B
    B_C = B-C

    dotproduct = np.dot(A_B, B_C)
    A_Bmagnitude = np.linalg.norm(A_B)
    B_Cmagnitude = np.linalg.norm(B_C)

    cos_theta = dotproduct /  (A_Bmagnitude * B_Cmagnitude)
    print(np.arccos(cos_theta) * 180/math.pi)

    return np.arccos(cos_theta) * 180/math.pi




    
 