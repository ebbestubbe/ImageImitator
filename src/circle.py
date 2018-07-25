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
    #r = random.randint(3,10)
    color = 0.1
    #min_color = -0.4
    #max_color = 0.4
    #color = random.random()*(max_color-min_color)+min_color
    return Circle(p0,p1,r,color)
    