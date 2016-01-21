from __future__ import division
import numpy as np
import cv2
from scipy import signal as sp
import scipy 
import Image
import scipy.ndimage as ndi
import math
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import zoom

class object:
    
    def __init__(self):
        self.l = 0
        self.r = 0
        self.t = 0
        self.b = 0

class minMaxPx:
    
    def __init__(self,val):
        self.min = val
        self.max = val
        self.avg = val

class regionMerge:
    
    def __init__(self, filename):
        self.filename = filename
        print 'preprocessing'
        self.preprocess()
        print 'defining labels'
        self.defineLabels()
        print 'filling avg values'
        self.fillAvgInten()
        print 'filling supergrid uniformly'
        self.fillSuperGridUniformInten()
        print 'filling crack points'
        self.fillCracks()
        print 'Updating Perimeters'
        self.updatePerimeters()
        print 'Removing weak common edges'
        self.removeWeakCommEdge()
        print 'copying edges'
        self.copyBackEdges()
        print 'check output'
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',self.img1)
        plt.imshow(self.img1)
        #plt.colorbar()
        plt.show()
        
    def preprocess(self):
        self.img1 = cv2.imread(self.filename,0);
        self.img2 = ndi.filters.gaussian_filter(self.img1, 1.2)       
        self.img = zoom(self.img2, 0.5)
        [self.w,self.h] = self.img.shape
        self.obj = np.array([ object() for x in range(self.h*self.w+1)])
        self.Label = np.array([[ 0 for x in range(self.h)] for y in range(self.w)])
        self.out = np.array([[ 0 for x in range(self.h)] for y in range(self.w)])

        ## to find the minimum, maximum and average intensities in a region
        self.regionAvgInten = np.array([ minMaxPx(0) for x in range(self.h*self.w+1)]) 
        
        ## Make supergrid of size (2*m-1,2*n-1) of original image
        self.superGrid = np.array([[[ 0 for intenLabel in range(2)] for x in range(2*self.h-1)] for y in range(2*self.w-1)])
        [self.superW,self.superH,l] = self.superGrid.shape
        
    
    def copyBackEdges(self):
        for i in range(self.superW):
            for j in range(self.superH):
                if self.superGrid[i,j,0] == 0 or (i-1>=0 and j-1>=0 and \
                (self.superGrid[i-1,j,0] == 0 or \
                ( i+1< self.superW and self.superGrid[i+1,j,0] == 0)) and \
                (( (j+1 < self.superH and \
                self.superGrid[i,j+1,0] == 0) or (j-1 >=0 and self.superGrid[i,j-1,0] == 0)))):
                    self.img1[i,j] = 255
                    
    def min(self,x,y):
        return x if x < y else y
    
    def removeWeakCommEdge(self):
        thres2 = 0.8
        for i in range(self.superW):
            for j in range(self.superH):
                R1 = 0
                R2 = 0
                if self.superGrid[i,j,0] == 0:
                    [R1,R2] = self.neighbrRegions(i,j)
                    if R1 != 0 and R2 != 0:
                        W = self.commPeri[R1,R2]
                        L1 = self.regionPeri[R1]
                        L2 = self.regionPeri[R2]
                        confidence = W/min(L1,L2) 
                        if confidence >= thres2 :
                            self.mergeRegions(R1,R2)
           
                                
    def mergeRegions(self,R1,R2):
        reg = 0
        if self.regionAvgInten[R1].avg < self.regionAvgInten[R2].avg:
            reg = self.regionAvgInten[R1].avg
            replaceByReg = self.regionAvgInten[R2].avg
            R = R2
        else:
            reg = self.regionAvgInten[R2].avg
            replaceByReg = self.regionAvgInten[R1].avg
            R = R1
            
        for x in range(2*self.obj[reg].l,2*self.obj[reg].r):
            for y in range(2*self.obj[reg].t,2*self.obj[reg].b):
                if self.superGrid[x,y,0] == reg:
                    self.superGrid[x,y,0] = replaceByReg
                    self.superGrid[x,y,1] = R
                if self.superGrid[x,y,0] == 0:
                    [R3,R4] = self.neighbrRegions(x,y)
                    if (R3 == R1 and R4 == R2) or (R3 == R2 and R4 == R1):
                        self.superGrid[x,y,0] = replaceByReg
                        self.superGrid[x,y,1] = R
                        
    def neighbrRegions(self,i,j):
        label1 = 0
        label2 = 0
        w = 0
        x = 0
        y = 0
        z = 0
        
        if i-1 >= 0:
            if self.superGrid[i-1,j,0] != 0 :
                w = self.superGrid[i-1,j,1]
        if j-1 >= 0:
            if self.superGrid[i,j-1,0] != 0 :
                x = self.superGrid[i,j-1,1]
        if i+1 < self.superW:
            if self.superGrid[i+1,j,0] != 0 :
                y = self.superGrid[i+1,j,1]
        if j+1 < self.superH:
            if self.superGrid[i,j+1,0] != 0 :
                z = self.superGrid[i,j+1,1]
        label1 = w
        if label1 == 0:
            label1 = x
        else:
            label2 = x
        if label1 == 0:
            label1 = y
        else:
            label2 = y
        if label1 == 0:
            label1 = z
        else:
            label2 = z
        return label1,label2
        
    
    def updatePerimeters(self):
        self.commPeri = np.array([[ 0 for x in range(self.label+1)] for y in range(self.label+1)])
        self.regionPeri = np.array([ 0 for x in range(self.label+1)]) 
        
        for i in range(self.superW):
            for j in range(self.superH):
                R1 = 0
                R2 = 0
                if self.superGrid[i,j,0] == 0:
                    [R1,R2] = self.neighbrRegions(i,j)
                    if R1 != 0:
                        self.regionPeri[R1] = self.regionPeri[R1] + 1
                    if R2 != 0:
                        self.regionPeri[R2] = self.regionPeri[R2] + 1
                        if R1 != 0:
                            self.commPeri[R1,R2] = self.commPeri[R1,R2] + 1
                            self.commPeri[R2,R1] = self.commPeri[R2,R1] + 1
                   
    def neighbourCracksIfEdge(self,i,j):
        count = 0
        inten = 0
        lab = 0
        if i-1 >= 0:
            if self.superGrid[i-1,j,0] != 0:
                inten = self.superGrid[i-1,j,0]
                lab = self.superGrid[i-1,j,1]
            else:
                count = count + 1
        if j-1 >= 0:
            if self.superGrid[i,j-1,0] != 0:
                inten = self.superGrid[i,j-1,0]
                lab = self.superGrid[i,j-1,1]
            else:
                count = count + 1
        if i+1 < self.superW:
            if self.superGrid[i+1,j,0] != 0:
                inten = self.superGrid[i+1,j,0]
                lab = self.superGrid[i+1,j,1]
            else:
                count = count + 1
        if j+1 < self.superH:
            if self.superGrid[i,j+1,0] != 0:
                inten = self.superGrid[i,j+1,0]
                lab = self.superGrid[i,j+1,1]
            else:
                count = count + 1
        if count >2:
            return 0,0
        else:
            return inten,lab
                                             
        
    ## 0 signifies weak edge
    ## 1 value signifies strong edge between pixels
    def fillCracks(self):
        for i in range(self.w):
            for j in range(self.h):
                
                ## fill vertical cracks [2i,2j+1]
                if 2*j+2 < self.superH and 2*i < self.superW:
                    diff = self.superGrid[2*i,2*j,0] - self.superGrid[2*i,2*j+2,0]
                    if diff != 0:
                        self.superGrid[2*i,2*j+1,0],self.superGrid[2*i,2*j+1,1] = 0,0
                    else:
                        self.superGrid[2*i,2*j+1,0],self.superGrid[2*i,2*j+1,1] = self.superGrid[2*i,2*j,0],self.superGrid[2*i,2*j,1]
                        
                ## fill horizontal cracks [2i+1,j]
                if 2*j < self.superH and 2*i+2 < self.superW:
                    diff = self.superGrid[2*i,2*j,0] - self.superGrid[2*i+2,2*j,0]
                    if diff != 0:
                        self.superGrid[2*i+1,2*j,0],self.superGrid[2*i+1,2*j,1] = 0,0
                    else:
                        self.superGrid[2*i+1,2*j,0],self.superGrid[2*i+1,2*j,1] = self.superGrid[2*i,2*j,0],self.superGrid[2*i,2*j,1]
                     
                ##crack edge crossing points
                if 2*i+1 < self.superW and 2*j+1 < self.superH:
                    self.superGrid[2*i+1,2*j+1,0],self.superGrid[2*i+1,2*j+1,0] = self.neighbourCracksIfEdge(2*i+1,2*j+1)  
                  
    def fillSuperGridUniformInten(self):
        ## fill superGrid intensity values
        for i in range(self.w):
            for j in range(self.h):
                ## filling intensities uniformly according to average values corresponding to labels
                self.superGrid[2*i,2*j,0] = self.regionAvgInten[self.Label[i,j]].avg
                
                ## fill superGrid region Labels
                self.superGrid[2*i,2*j,1] = self.Label[i,j]
        
    def fillAvgInten(self):
        for i in range(self.regionAvgInten.shape[0]):
            self.regionAvgInten[i].avg = (self.regionAvgInten[i].min + self.regionAvgInten[i].max)/2;
            
    def detectEdge(self):
        self.output = scipy.zeros(self.Label.shape)
        w = self.output.shape[1]
        h = self.output.shape[0]
    
        for y in range(1, h-1 ):
            for x in range(1, w-1 ):
                patch = self.Label[y-1:y+1, x-1:x+1]
                
                maxP = patch.max()
                minP = patch.min()
                Cross = False
                if minP !=maxP:
                    Cross = True
                if Cross:
                    self.output[y, x] = 255
                
        return self.output
        
    def defineLabels(self):
        label = 0
        ### visit each pixel and check if the label is not already set . Give the pixel a label and recurse to the neihbours
        for i in range(self.w):
            for j in range(self.h):
                if self.Label[i,j] == 0:
                    label = label + 1
                    currInt = self.img[i,j]
                    self.obj[label].l = i
                    self.obj[label].r = i
                    self.obj[label].t = j
                    self.obj[label].b = j
                    self.recurLabel(i,j,label,currInt);
        self.label = label
     
        
    def recurLabel(self,i,j,label,currInt):
        ##return at object boundary conditions
        thres1 = 3.8
        if (i < 0 or i >= self.w or j < 0 or j >= self.h) or \
            self.Label[i,j] != 0 or \
            self.img[i,j] < currInt - thres1 or self.img[i,j] > currInt + thres1:
            return
        
        currInt = self.img[i,j]
        
        if self.regionAvgInten[label].min > currInt:
            self.regionAvgInten[label].min = currInt
        elif self.regionAvgInten[label].max < currInt:
            self.regionAvgInten[label].max = currInt
        
        ## if label already not set and the pixel intensity satisfies homogenity criteria
        ## set Label
        self.Label[i,j] = label
        if self.obj[label].l > i:
            self.obj[label].l = i 
        elif self.obj[label].r < i: 
            self.obj[label].r = i 
        if self.obj[label].t > j:
            self.obj[label].t = j 
        elif self.obj[label].b < j: 
            self.obj[label].b = j
        self.recurLabel(i-1,j+1,label,currInt)
        self.recurLabel(i,j+1,label,currInt)
        self.recurLabel(i+1,j+1,label,currInt)
        self.recurLabel(i+1,j,label,currInt)
        self.recurLabel(i+1,j-1,label,currInt)

if __name__ == "__main__":
    hough = regionMerge('./Peppers.jpg');