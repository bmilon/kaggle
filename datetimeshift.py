# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:47:49 2019

@author: mbhattac
"""


import pytz 

def  datetimeshift(a):
    return a.replace(tzinfo=pytz.timezone("UTC"))
    

