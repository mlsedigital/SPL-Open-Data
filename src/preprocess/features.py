import numpy as np
import math

def angle_finder_shared(A, B, C, D):
    """
    Retrieves the angle between points, A,B,C,D by using the dot product. 

    When you have like a shared point, just use it again

    Args:
        A: (np.array) An array of size 3
        B: (np.array) An array of size 3
        C: (np.array) An array of size 3
        D: (np.array) An array of size 3
    
    Return:
        angle: (np.float) the angle between the 3 points
    """
    A_B = A-B
    D_C = C-D

    dotproduct = np.dot(A_B.T, D_C)
    A_Bmagnitude = np.linalg.norm(A_B)  
    D_Cmagnitude = np.linalg.norm(D_C)

    cos_theta = dotproduct /  (A_Bmagnitude * D_Cmagnitude)

    return np.arccos(cos_theta) * 180/math.pi


def velocity_calc(data, column, groupby_col='trial_id'):
    """ 
    Does some quick velocity calculations for different bodyparts. 
    Follows this process:
    1. Groupby by the groupby object
    2. runs pd.shift() method for the column with the specific movement_axis (x, y, z, or all)
    3. differences the 2 
    4. divides by 30 to give us velocity
    
    Args:
        data (pd.DataFrame): the dataset
        column (list of strings): Columns
        groupby_col (string or list of strings): what to groupby by (eventually may have to factor multiple features)
    
    returns:
        Nonetype
    """
    group = data.groupby(groupby_col)
    
    for col in column:
        shifted_col = group[col].shift(1)
        data[f'VELOCITY_{col}'] = (data[col] - shifted_col) * 30
        
    return data
    


    
 