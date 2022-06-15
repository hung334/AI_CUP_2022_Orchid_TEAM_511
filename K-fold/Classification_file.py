# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:13:45 2022

@author: chen_hung
"""

import pandas as pd
import os
import shutil

if __name__ == '__main__':

    file_data = pd.read_csv("label.csv")
    
    for i in range(219):
        path = './{}'.format(i)
        if not os.path.isdir(path):
            os.mkdir(path)
            
    for file_name , class_ in file_data.values:
        print(file_name , class_)
        source = './{}'.format(file_name)
        destination = './{}/{}'.format(class_,file_name)
        shutil.move(source,destination)

