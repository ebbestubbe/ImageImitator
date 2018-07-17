import numpy as np
import random
class Circle:
    def __init__(self,p0,p1,r,color):
        self.p0 = p0
        self.p1 = p1
        self.r = r
        self.color = color

def generate_random_circle(shape):
    p0 = random.randint(0,shape[0])
    p1 = random.randint(0,shape[1])
    r = 5
    #r = random.randint(1,max(shape))
    #color = 0.0001
    color = random.random()*2-1.0
    return Circle(p0,p1,r,color)
