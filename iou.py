import numpy as np
import matplotlib.pyplot as plt

def area_intersection(R, d):
    """
    R = radius of the 2 circles
    d = distance between the center of the 2 circles
    
    """
    A = (2*R**2)*np.arccos(d/(2*R))
    B = (R**2)-((0.5*d)**2)
    return A-d*np.sqrt(B)

def iou(center1, center2, R):
        d = np.linalg.norm(center1 - center2)
        if d >= 2*R :
            intersection = 0
        else :
            intersection = area_intersection(R,d)
        union = 2*np.pi*R**2 - intersection
        iou = intersection / union
        return iou

"""
if __name__ == "__main__":
    score = iou(np.array([0,0]), np.array([0,1]), 1)
    print(score)
"""