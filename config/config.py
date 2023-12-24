# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:43:25 2023

@author: User
"""



import yaml

def load_yaml(filename):
    
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        
    return data

args = load_yaml('config/config.yaml')