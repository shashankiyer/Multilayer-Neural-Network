import random
import math
import numpy as np
import sys
import os

class samplespace:

    def __init__( self , prob ):
        counter = 1
        while ( prob < 10 ):
            prob = prob * 10
            counter = counter * 10
        self.sample = np.ones(counter)

        for i in range ( 0 , int( prob ) ):
            self.sample[i] = 0    

    def le (self):
        return len(self.sample)

class sticky:
    di = {'A' : 'C' , 'B' : 'D' , 'C' : 'A' , 'D' : 'B'}
    dis = {'A' : ['C' , 'B' , 'D'] , 'B' : ['C' , 'D' , 'A'] , 'C' : ['D' , 'B' , 'A'] , 'D' : ['C' , 'B' , 'A']}
    dinu = {'A' : '1' , 'B' : '2' , 'C' : '3' , 'D' : '4'}
    def mutate ( self , prob , s , sstring):
        x = random.randint( 0 , 2 )
        y = random.randint( 0 , s.le()-1 )
        if prob == 1 or s.sample[y] == 0:
            return sticky.dis[sstring][x]
        return sstring

    def __init__( self , prob , from_ends , s):
        self.string = ''
        arr = [ 'A' , 'B' , 'C' , 'D' ]
        w = ''
        for i in range ( 0 , 20 ):
            x = random.randint( 0 , 3 )
            self.string += arr[x]
            w += sticky.di[arr[x]]

        w = w[::-1]
        self.string += w
        list1 = list(self.string)
        
        for i in range ( 0 , from_ends ):
            list1[i] = self.mutate ( prob , s , list1[i])
        for i in range (from_ends , len(list1) - from_ends):
            list1[i] = self.mutate ( 1 , s , list1[i])
        for i in range ( len(list1) - from_ends , len(list1) ):
            list1[i] = self.mutate ( prob , s , list1[i])
        self.string = ''.join(list1)

        self.convert()

    def convert(self):
        list1 = list(self.string)
        for i in range ( 0 , 40 ):
            list1[i] = list1[i]        
        self.string = ''.join(list1)
            
    




if __name__ == '__main__' :
    if len(sys.argv) != 5:
        print ("Please supply the correct arguments")
        raise SystemExit(1)
       
    the_file = open( str(sys.argv[ 4 ]) , "w+")

    prob = float (sys.argv[ 2 ])
    s = samplespace ( prob )
    from_ends = int (sys.argv[ 3 ])
    for i in range ( 0 , int(sys.argv[1]) ):
        ss = sticky ( prob , from_ends , s )
        the_file.write( getattr( ss , 'string' ) + '\n')
        print (getattr( ss , 'string' ))


            
