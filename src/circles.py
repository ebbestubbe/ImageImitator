from circle import Circle
class Circles:
    def __init__(self, circles = None):    
        self.circles = circles

        #self.convertToParameters()
    def addCircle(self, circle, index=None):
        if(index==None):
            self.circles.append(circle)
        else:
            self.circles.insert(index,circle)

    def popCircle(self, index=None):
        if(index is None):
            c = self.circles.pop()
        else:
            c = self.circles.pop(index)
    