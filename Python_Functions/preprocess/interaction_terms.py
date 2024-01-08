# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 14:05:26 2023

@author: cfris
"""

from sklearn.preprocessing import PolynomialFeatures

def interaction_terms(data):
    #create columns of interaction terms that are a each 2-sensor combination multiplied together 
    #for CO, temp, humidity sensor, it would be CO*temp, CO*humidity, temp*humidity
    poly = PolynomialFeatures(degree=2,interaction_only=True, include_bias=False)
    data = poly.fit_transform(data)
    return data