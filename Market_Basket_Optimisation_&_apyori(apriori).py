# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 16:16:14 2023

@author: rajsu
"""

import numpy as np
import  pandas as pd

dataset=pd.read_csv("Market_Basket_Optimisation.csv",header=None)
transaction=[]
for i in range(7501):
    
        transaction.append([str(dataset.values[i,j])for j in range(20)])
        
from apyori import apriori
rules=apriori(transaction,min_support=0.005,min_confidence=0.2,min_lift=3,min_length=2)


result=list(rules)
