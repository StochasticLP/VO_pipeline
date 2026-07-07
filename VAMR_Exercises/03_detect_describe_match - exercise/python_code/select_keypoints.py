import numpy as np
import line_profiler

'''
def selectKeypoints(scores, num, r):
    """
    Selects the num best scores as keypoints and performs non-maximum supression of a (2r + 1)*(2r + 1) box around
    the current maximum.
    """    
    keypoints = np.zeros((num, 2), dtype=int)

    for i in range(num):                                      
        row, col = np.unravel_index(np.argmax(scores), scores.shape)
        keypoints[i] = [row, col]
        
        scores[row-r:row+r, col-r:col+r] = 0

    return keypoints.T
'''
@line_profiler.profile
def selectKeypoints(scores, num, r):
    scores = scores.copy()
    scores[:r+1, :] = 0    # top
    scores[-r-1:, :] = 0   # bottom
    scores[:, :r+1] = 0    # left
    scores[:, -r-1:] = 0   # right
    
    keypoints = np.zeros((num, 2), dtype=int)
    suppressed = np.zeros(scores.shape, dtype=bool)
    
    # sort all candidates once
    flat_indices = np.argsort(scores.ravel())[::-1]
    rows, cols = np.unravel_index(flat_indices, scores.shape)
    
    count = 0
    for row, col in zip(rows, cols):
        if count == num:
            break
        if not suppressed[row, col]:
            keypoints[count] = [row, col]
            count += 1
            r0, r1 = max(0, row-r), min(scores.shape[0], row+r+1)
            c0, c1 = max(0, col-r), min(scores.shape[1], col+r+1)
            suppressed[r0:r1, c0:c1] = True
    
    return keypoints.T
