# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 21:13:19 2025

@author: mdaks
"""

import numpy as np
import pandas as pd
df=pd.read_csv("C:\\Users\\mdaks\\Downloads\\EVS_US (1).csv")
obj_df.columns
a=df['Make and model']
cleanup_nums = {"Vehicle Type": {"Truck": 9, "Pickup truck": 8,"SUV (7 seats)": 7, "SUV": 6, "Station wagon": 5, "Hatchback": 4,"Sedan": 3, "Crossover": 2, "Coupe":1 }}
#To convert the columns to numbers using replace :
obj_df = df.replace(cleanup_nums)
obj_df['Price Per Mile of Range'] = obj_df['Price Per Mile of Range'].astype('int64')
def weighted_product_method(criteria, weights):
    """
    Apply the Weighted Product Method (WPM) to a set of criteria and weights.

    Parameters:
    criteria (numpy.ndarray): A 2D array where each row represents an alternative and each column represents a criterion.
    weights (numpy.ndarray): A 1D array of weights corresponding to the criteria.

    Returns:
    numpy.ndarray: A 1D array of scores for each alternative.
    """
    # Normalize the weights
    weights = weights / np.sum(weights)
    
    # Calculate the weighted product for each alternative
    scores = np.prod(np.power(criteria, weights), axis=1)
    
    return scores

 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(df)

# # Example usage
# criteria = np.array([
#     [7, 9, 8],  # Alternative 1
#     [6, 8, 7],  # Alternative 2
#     [8, 7, 9]   # Alternative 3
# ])

# weights = np.array([0.5, 0.3, 0.2])
b=scaler.fit_transform(obj_df[['Base Price (MSRP)','Price Per Mile of Range', 'Vehicle Type']])
b=pd.DataFrame(b,columns=['Base Price (MSRP)','Price Per Mile of Range', 'Vehicle Type'])
b['Base Price (MSRP)']=1-b['Base Price (MSRP)']

scores = weighted_product_method(np.array(b), weights)
df['scores']=scores
print("Scores for each alternative:", scores)
df.to_excel("output.xlsx")