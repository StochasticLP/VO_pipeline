import numpy as np
from scipy.spatial.distance import cdist
import line_profiler

@line_profiler.profile
def matchDescriptors(query_descriptors, database_descriptors, match_lambda):
    """
    Returns a 1xQ matrix where the i-th coefficient is the index of the database descriptor which matches to the
    i-th query descriptor. The descriptor vectors are MxQ and MxD where M is the descriptor dimension and Q and D the
    amount of query and database descriptors respectively. matches(i) will be -1 if there is no database descriptor
    with an SSD < lambda * min(SSD). No elements of matches will be equal except for the -1 elements.
    """
    len = query_descriptors.shape[1]
    matches = np.full(len, -1)
    
    dist = cdist(query_descriptors.T, database_descriptors.T, 'sqeuclidean')
    
    min_dist = np.min(dist[dist > 0])
    delta = match_lambda * min_dist
    
    for i in range(len):
        idist = dist[i]
        done = False
        while not done:
            min_dist = np.min(idist)
            if (min_dist > delta):
                done = True
                break
            j = np.argmin(idist)
            if j in matches:
                idist[j] = delta + 1
            else:
                matches[i] = j
                done = True
                break
    return matches
    
