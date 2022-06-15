# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:05:40 2022

@author: chen_hung
"""

import os
import shutil
import random

def is_number(s):
  try:
    float(s) # for int, long and float
  except ValueError:
    try:
      complex(s) # for complex
    except ValueError:
      return False
  return True

if __name__ == '__main__':
    
    
    # save_path = os.path.join('C:\Users\chen_hung\Desktop\AI_CUP_fold',)
    # if not os.path.isdir(path):
    #         os.mkdir(path)
    
    o_path = r"C:\Users\chen_hung\Desktop\training"
    o_img  = os.listdir(o_path)
    
    save_path = r'C:\Users\chen_hung\Desktop\AI_CUP_fold\AI_CUP_fold'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if not os.path.isdir(os.path.join(save_path,"train")):
                os.mkdir(os.path.join(save_path,"train"))
    if not os.path.isdir(os.path.join(save_path,"val")):
                os.mkdir(os.path.join(save_path,"val"))
    #class_img = 
    
    for class_ in  o_img:
        if(is_number(class_)):
            class_path = os.path.join(o_path,class_)
            img_list = os.listdir(class_path)
            random.shuffle(img_list)
            train_list = img_list[0:7]
            val_list = img_list[7:10]
            
            if not os.path.isdir(os.path.join(save_path,"train",class_)):
                os.mkdir(os.path.join(save_path,"train",class_))
            if not os.path.isdir(os.path.join(save_path,"val",class_)):
                os.mkdir(os.path.join(save_path,"val",class_))
            for file_name in train_list:
                    source = os.path.join(o_path,class_,file_name)
                    destination = os.path.join(save_path,'train',class_,file_name)
                    shutil.copy(source,destination)
            for file_name in val_list:
                    source = os.path.join(o_path,class_,file_name)
                    destination = os.path.join(save_path,'val',class_,file_name)
                    shutil.copy(source,destination)
