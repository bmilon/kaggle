#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'solve' function below.
#
# The function is expected to return an INTEGER_ARRAY.
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER m
#  3. INTEGER_ARRAY h
#  4. 2D_INTEGER_ARRAY rounds
#

def powerset(h):
    return (2*len(h)*sum(h))
#    h=set(h)
#    
#    if len(h) == 1:
#        return sum(h)
#    else:
#        return (2*len(h)*sum(h))
    
    
def solve(n, m, h, rounds):
    # Write your code here
    answers=[];
    for counter in range(0,len(rounds)):
        #print('new round')
        allheight=[];
        start=rounds[counter][0]
        end=rounds[counter][1]
        
        for counter2 in range(start-1,end):
            for counter3 in range(counter2+1,end+1):
                # find all subsequences
                if powerset(h[counter2:counter3]) >=  rounds[counter][2]:
                    maxheight=max(h[counter2:counter3])
                    allheight.append(maxheight)
                    #print(h[counter2:counter3])
                    #print(maxheight)
                    
        if len(allheight) == 0 :
            answers.append(-1)
        else:
            answers.append(min(allheight))
        
        
    return answers

    
if __name__ == '__main__':
    fptr = open('input02_basketball.txt', 'r')

    first_multiple_input = fptr.readline().rstrip().split()

    n = int(first_multiple_input[0])

    m = int(first_multiple_input[1])

    h = list(map(int, fptr.readline().rstrip().split()))

    rounds = []

    for _ in range(m):
        rounds.append(list(map(int, fptr.readline().rstrip().split())))

    answer = solve(n, m, h, rounds)

    #fptr.write('\n'.join(map(str, answer)))
    #fptr.write('\n')
    print(answer)
    fptr.close()
