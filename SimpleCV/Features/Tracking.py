from SimpleCV.base import *
from SimpleCV.ImageClass import *
from SimpleCV.Features.Features import Feature, FeatureSet

class CAMShift(Feature):
    def __init__(self, img, bb, ellipse):
        self.ellipse = ellipse
        self.bb = bb
        self.image = img
        self.x, self.y, self.w, self.h = self.bb
        self.center = self.getCenter()
        self.area = self.getArea()

    def getCenter(self):
        return (self.x+self.w/2,self.y+self.h/2)
        
    def getArea(self):
        return self.w*self.h
        
    def getImage(self):
        return self.img
        
    def getBB(self):
        return self.bb
        
    def getEllipse(self):
        return self.ellipse
        
    def drawBB(self, color=(0, 0, 255), width=1):
        self.image.drawRectangle(self.x,self.y,self.w,self.h,color,width)
    
    def draw(self, color=(0, 0, 255), width=1):
        self.image.drawCircle(self.center, 1, color, width)
