from circle import Circle
from flattened_circles import Flattened_circles
class Circles:
    def __init__(self, circles = None):
        if(circles==None):  
            self.circles = []
        else:     
            self.circles = circles

        #self.convertToParameters()
    def add_circle(self, circle, index=None):
        if(index==None):
            self.circles.append(circle)
        else:
            self.circles.insert(index,circle)

    def pop_circle(self, index=None):
        if(index is None):
            c = self.circles.pop()
        else:
            c = self.circles.pop(index)
        return c
    def ncircles(self):
        return len(self.circles)

    def flatten(self):
        return parameters