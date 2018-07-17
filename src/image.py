import numpy as np
from skimage import io,draw


class Image:
    def __init__(self, shape, circles=None):
        self.image = np.zeros(shape) #the values to save in a map.
        self.circles = []

        if(circles is not None):
            self.add_circles(circles)
    
    def add_circles(self,circles):
        for c in circles:
            self.add_circle(c)
    
    def add_circle(self,c,index=None):
        if(index==None):
            self.circles.append(c)
        else:
            self.circles.insert(index,c)
        rr,cc = draw.ellipse(c.p0,c.p1,c.r,c.r,self.image.shape)
        self.image[rr,cc] += c.color
        #print("circle added: p0: {0}; p1: {1}; r: {2}, c: {3}".format(c.p0,c.p1,c.r,c.color))

    def pop_circle(self, index=None):
        if(index is None):
            c = self.circles.pop()
        else:
            c = self.circles.pop(index)
        rr,cc = draw.ellipse(c.p0,c.p1,c.r,c.r,self.image.shape)
        self.image[rr,cc] -= c.color
        return c
        #print("circle removed: p0: {0}; p1: {1}; r: {2}, c: {3}".format(c.p0,c.p1,c.r,c.color))
    """
    def shuffle(self,index):
        c = self.pop_circle(index)
        p0 = random.randint(0,self.image.shape[0])
        p1 = random.randint(0,self.image.shape[1])
        c = Circle(p0,p1,c.r,c.color)
        self.add_circle(c)
    """
    def get_normalized_image(self):
        
        #maxwith = np.zeros(self.image.shape)
        #normalized = np.maximum(self.image,maxwith)
        
        #minwith = np.ones(self.image.shape)
        #normalized = np.minimum(self.image,minwith)
        
        normalized = self.image - np.amin(self.image)
        if(np.abs(np.amax(normalized)) > 0.00001):
            normalized = normalized/np.amax(normalized)
        return normalized
    
    def get_image(self):
        return self.image
    def print_circles(self):
        for c in self.circles:
            print("p0: {0}; p1: {1}; r: {2}; c: {3}".format(c.p0,c.p1,c.r,c.color))