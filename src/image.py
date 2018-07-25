import numpy as np
from skimage import io,draw
from circles import Circles


class Image:
    def __init__(self, shape, circles=None,background = 0):
        self.image = np.ones(shape)*background #the values to save in a map.
        
        if(circles == None):
            self.circles = Circles()
        else:
            self.circles = circles
            for c in self.circles.circles:
                self.draw_circle(c)
    def __lt__(self, other):
        return self.circles.circles[0].color < other.circles.circles[0].color


    #def add_circles(self,circles):
    #    for c in circles:
    #        self.add_circle(c)
    
    def add_circle(self,c,index=None):
        self.circles.add_circle(c,index)
        draw_circle(c)
        #rr,cc = draw.ellipse(c.p0,c.p1,c.r,c.r,self.image.shape)
        #self.image[rr,cc] += c.color
        #print("circle added: p0: {0}; p1: {1}; r: {2}, c: {3}".format(c.p0,c.p1,c.r,c.color))

    def pop_circle(self, index=None):
        c = self.circles.pop_circle(index)
        draw_anti_circle(c)
        #rr,cc = draw.ellipse(c.p0,c.p1,c.r,c.r,self.image.shape)
        #self.image[rr,cc] -= c.color
        return c
        #print("circle removed: p0: {0}; p1: {1}; r: {2}, c: {3}".format(c.p0,c.p1,c.r,c.color))
    def draw_anti_circle(self,c):
        rr,cc = draw.ellipse(c.p0,c.p1,c.r,c.r,self.image.shape)
        self.image[rr,cc] -= c.color
    
    def draw_circle(self,c):
        rr,cc = draw.ellipse(c.p0,c.p1,c.r,c.r,self.image.shape)
        self.image[rr,cc] += c.color
        
    def ncircles(self):
        return self.circles.ncircles()

    def get_normalized_image(self):
        
        #maxwith = np.zeros(self.image.shape)
        #normalized = np.maximum(self.image,maxwith)
        
        #minwith = np.ones(self.image.shape)
        #normalized = np.minimum(self.image,minwith)
        
        normalized = self.image - np.amin(self.image)
        if(np.abs(np.amax(normalized)) > 0.00001):
            normalized = normalized/np.amax(normalized)
        return normalized
    def get_normalized_image2(self):
        
        maxwith = np.zeros(self.image.shape)
        normalized = np.maximum(self.image,maxwith)
        
        minwith = np.ones(self.image.shape)
        normalized = np.minimum(normalized,minwith)
        
        return normalized
        
    def get_shape(self):
        return self.image.shape
    def get_image(self):
        return self.image
    def print_circles(self):
        for c in self.circles:
            print("p0: {0}; p1: {1}; r: {2}; c: {3}".format(c.p0,c.p1,c.r,c.color))