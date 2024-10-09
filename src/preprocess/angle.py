 

## TODO: Change this for the orientation matrices


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




    
 