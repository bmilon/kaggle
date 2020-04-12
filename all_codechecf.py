





import sys 

T = sys.stdin.readline().split() 
T = int(T[0])


for testcases in range(T):
    people = sys.stdin.readline() 
    people = int(people)
    spots = sys.stdin.readline().split() 
    location = []
    for sp in range(len(spots)):
        if spots[sp] == '1':
            location.append(sp)
            if len(location) > 1 :
                if  (location[-1] - location[-2])  < 6:
                    print('NO')
                    sp = len(spots) 
    
    if sp != len(spots):
        print('YES')
    # show locations
    

1        
5 3        
5 4 8 6 9        
2 3 5




import sys 

T = sys.stdin.readline() #input('')
T = int(T)


for testcases in range(T):
    N,Q= input().split() 
    N,Q=int(N),int(Q)
    #print(N,Q)
    scores = sys.stdin.readline().split()
    scores=[int(a) for a in scores ]
    thoughts = sys.stdin.readline().split()
    thoughts = [int(a) for a in thoughts]
    #print(scores)
    #print(thoughts)
    for t in thoughts:
        print(max(scores[0:t]))
    



T = int(input(''))


for testcases in range(T):
    N,Q= input().split() 
    N,Q=int(N),int(Q)
    #print(N,Q)
    scores = input().split()
    scores=int(scores)
    
    
    
    