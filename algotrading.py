#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'solve' function below.
#
# The function is expected to return an INTEGER.
# The function accepts 2D_INTEGER_ARRAY profits as parameter.
#

def solve1(profits,iminus1=-1,iminus2=-1,currprofit=0,currday=0):
    
    if currday == len(profits):
        return currprofit # doubut
    
    newprofit=0; oldprofit=0;
    
    for algos in set([0,1,2,3]):
        if (algos == iminus1) or  (algos == iminus2):
            pass
        else:
            newprofit= solve(profits,iminus2,algos,currprofit+profits[currday][algos],currday+1)            
            if newprofit  > oldprofit:
                oldprofit = newprofit
    
    return oldprofit


def solve1():
    
    kick=len(profits)
    register = [0] * kick 
    for counter1 in range(0,int (math.pow(4,kick))):
        carry=1
        for counter2 in range(kick,0):
            carry =( register[counter2] + carry) % 3
            register[counter2] = math.ceil(( register[counter2] + carry) / 3)
            
        print(register)            
            

if __name__ == '__main__':
    global profits 
    fptr = open('input06.txt', 'r')

    q = int(fptr.readline().strip())

    for q_itr in range(q):
        w = int(fptr.readline().strip())

        profits = []

        for _ in range(w):
            profits.append(list(map(int,fptr.readline().strip().split())))
            
        result = solve1()
        

        print(str(result) + '\n')

    fptr.close()
