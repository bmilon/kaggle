#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'solve' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY a
#

def solve(n, a):
    
    if n > len(a):
        return 0 
    
    a.sort(reverse=True)
    soilder=len(a) 
    invaders=n
    while soilder >  0:
        if  a[0] > sum(a)/invaders:
            a.pop(0)
            invaders-=1 
        else:
            return sum(a)/invaders
            
    # Print your result

if __name__ == '__main__':
    first_multiple_input = input().rstrip().split()

    n = int(first_multiple_input[0])

    k = int(first_multiple_input[1])

    a = list(map(int, input().rstrip().split()))

    print(solve(n, a))
