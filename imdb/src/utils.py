import numpy as np

def get_dummies_based_variables(response, independent, data, k):
    count = data[independent].value_counts()

    top_k = list(count[:k].index)

    dummy = data.loc[:, independent].copy()

    for i in range(0, len(data)): 
        if dummy[i] not in top_k:
            dummy[i] = np.nan

    dummy = dummy.str.get_dummies()

    return(dummy)