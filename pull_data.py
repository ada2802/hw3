# -*- coding: utf-8 -*-
"""
pull_data.py (5 points) 
When this is called using python pull_data.py in the command line, 
this will go to the 2 Kaggle urls provided below, authenticate using 
your own Kaggle sign on, pull the two datasets, and save as .csv files 
in the current local directory. The authentication login details (aka secrets) 
need to be in a hidden folder (hint: use .gitignore). There must be a data check step to ensure the data has been pulled correctly and clear commenting and documentation for each step inside the .py file. 

Training dataset url: https://www.kaggle.com/c/titanic/download/train.csv 
Scoring dataset url: https://www.kaggle.com/c/titanic/download/test.csv

@author: Ada
"""
#
from kaggle.api.kaggle_api_extended import KaggleApi

import os

download_path ="C:\\Users\\Ada\\Desktop\\CUNY_SPS_DA\\622 ML\\HW1 Titanic"

api = KaggleApi()

api.authenticate()

api.competition_download_files('titanic', path = download_path)

    
    
