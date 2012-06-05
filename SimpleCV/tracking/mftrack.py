from SimpleCV import Camera, Image, Display
from cv2 import *
from cv2.cv import *
import math
import time
import numpy as np
import scipy.spatial.distance as spsd

def fbtrack(imgI, imgJ, bb, numM=10, numN=10,margin=5,winsize_ncc=10):
    """
    **SUMMARY**
    
    Forward-Backward tracking using Lucas-Kanade Tracker
    
    **PARAMETERS**
    
    imgI - Image contain Object with known BoundingBox (Numpy array)
    imgJ - Following image (Numpy array)
    bb - Bounding box represented through 2 points (x1,y1,x2,y2)
    numM - Number of points in height direction.
    numN - Number of points in width direction.
    margin - margin (in pixel)
    winsize_ncc - size of the search window at each pyramid level in LK tracker (in int)
    
    **RETURNS**
    
    newbb - Bounding box of object in track in imgJ
    scaleshift - relative scale change of bb
    
    """
    nPoints = numM*numN
    sizePointsArray = nPoints*2
    
    pt = getFilledBBPoints(bb, numM, numN, margin)
    fb, ncc, status, ptTracked = lktrack(imgI, imgJ, pt, nPoints, winsize_ncc)
    
    nlkPoints = 0
    for i in range(nPoints):
        nlkPoints += status[i][0]

    startPoints = []
    targetPoints = []
    fbLKCleaned = [0.0]*nlkPoints
    nccLKCleaned = [0.0]*nlkPoints
    M = 2
    nRealPoints = 0
    
    for i in range(nPoints):
        if ptTracked[M*i] is not None:
            startPoints.append((pt[2 * i],pt[2*i+1]))
            targetPoints.append((ptTracked[2 * i], ptTracked[2 * i + 1]))
            fbLKCleaned[nRealPoints]=fb[i]
            nccLKCleaned[nRealPoints]=ncc[i][0][0]
            nRealPoints+=1
            
    medFb = getMedian(fbLKCleaned)
    medNcc = getMedian(nccLKCleaned)
    
    nAfterFbUsage = 0
    for i in range(nlkPoints):
        if fbLKCleaned[i] <= medFb and nccLKCleaned[i] >= medNcc:
            startPoints[nAfterFbUsage] = startPoints[i]
            targetPoints[nAfterFbUsage] = targetPoints[i]
            nAfterFbUsage+=1

    newBB, scaleshift = predictBB(bb, startPoints, targetPoints, nAfterFbUsage)
    
    return (newBB, scaleshift)

def calculateBBCenter(bb):
    """
    
    **SUMMARY**
    
    Calculates the center of the given bounding box
    
    **PARAMETERS**
    
    bb - Bounding Box represented through 2 points (x1,y1,x2,y2)
    
    **RETURNS**
    
    center - A tuple of two floating points
    
    """
    center = (0.5*(bb[0] + bb[2]),0.5*(bb[1]+bb[3]))
    return center
    
def getFilledBBPoints(bb, numM, numN, margin):
    """
    
    **SUMMARY**
    
    Creates numM x numN points grid on Bounding Box
    
    **PARAMETERS**
    
    bb - Bounding Box represented through 2 points (x1,y1,x2,y2)
    numM - Number of points in height direction.
    numN - Number of points in width direction.
    margin - margin (in pixel)
    
    **RETURNS**
    
    pt - A list of points (pt[0] - x1, pt[1] - y1, pt[2] - x2, ..)
    
    """
    pointDim = 2
    bb_local = (bb[0] + margin, bb[1] + margin, bb[2] - margin, bb[3] - margin)
    if numM == 1 and numN == 1 :
        pts = calculateBBCenter(bb_local)
        return pts
    
    elif numM > 1 and numN == 1:
        divM = numM - 1
        divN = 2
        spaceM = (bb_local[3]-bb_local[1])/divM
        center = calculateBBCenter(bb_local)
        pt = [0.0]*(2*numM*numN)
        for i in range(numN):
            for j in range(numM):
                pt[i * numM * pointDim + j * pointDim + 0] = center[0]
                pt[i * numM * pointDim + j * pointDim + 1] = bb_local[1] + j * spaceM
                
        return pt
        
    elif numM == 1 and numN > 1:
        divM = 2
        divN = numN - 1
        spaceN = (bb_local[2] - bb_local[0]) / divN
        center = calculateBBCenter(bb_local)
        pt = [0.0]*((numN-1)*numM*pointDim+numN*pointDim)
        for i in range(numN):
            for j in range(numN):
                pt[i * numM * pointDim + j * pointDim + 0] = bb_local[0] + i * spaceN
                pt[i * numM * pointDim + j * pointDim + 1] = center[1]
        return pt
        
    elif numM > 1 and numN > 1:
        divM = numM - 1
        divN = numN - 1
    
    spaceN = (bb_local[2] - bb_local[0]) / divN
    spaceM = (bb_local[3] - bb_local[1]) / divM

    pt = [0.0]*((numN-1)*numM*pointDim+numM*pointDim)
    
    for i in range(numN):
        for j in range(numM):
            pt[i * numM * pointDim + j * pointDim + 0] = float(bb_local[0] + i * spaceN)
            pt[i * numM * pointDim + j * pointDim + 1] = float(bb_local[1] + j * spaceM)
    return pt

def getBBWidth(bb):
    """
    
    **SUMMARY**
    
    Get width of the bounding box
    
    **PARAMETERS**
    
    bb - Bounding Box represented through 2 points (x1,y1,x2,y2)
    
    **RETURNS**
    
    width of the bounding box
    
    """
    return bb[2]-bb[0]+1
    
def getBBHeight(bb):
    """
    
    **SUMMARY**
    
    Get height of the bounding box
    
    **PARAMETERS**
    
    bb - Bounding Box represented through 2 points (x1,y1,x2,y2)
    
    **RETURNS**
    
    height of the bounding box
    
    """
    return bb[3]-bb[1]+1
    
def predictBB(bb0, pt0, pt1, nPts):
    """
    
    **SUMMARY**
    
    Calculates the new (moved and resized) Bounding box.
    Calculation based on all relative distance changes of all points
    to every point. Then the Median of the relative Values is used.
    
    **PARAMETERS**
    
    bb0 - Bounding Box represented through 2 points (x1,y1,x2,y2)
    pt0 - Starting Points
    pt1 - Target Points
    nPts - Total number of points (eg. len(pt0))
    
    **RETURNS**
    
    bb1 - new bounding box
    shift - relative scale change of bb0
    
    """
    ofx = []
    ofy = []
    for i in range(nPts):
        ofx.append(pt1[i][0]-pt0[i][0])
        ofy.append(pt1[i][1]-pt0[i][1])
    
    dx = getMedianUnmanaged(ofx)
    dy = getMedianUnmanaged(ofy)
    ofx=ofy=0
    
    lenPdist = nPts * (nPts - 1) / 2
    dist0=[]
    for i in range(nPts):
        for j in range(i+1,nPts):
            temp0 = ((pt0[i][0] - pt0[j][0])**2 + (pt0[i][1] - pt0[j][1])**2)**0.5
            temp1 = ((pt1[i][0] - pt1[j][0])**2 + (pt1[i][1] - pt1[j][1])**2)**0.5
            dist0.append(float(temp1)/temp0)
            
    shift = getMedianUnmanaged(dist0)
    
    s0 = 0.5 * (shift - 1) * getBBWidth(bb0)
    s1 = 0.5 * (shift - 1) * getBBHeight(bb0)
    
    bb1 = (abs(bb0[0] + s0 + dx),
           abs(bb0[1] + s1 + dy),
           abs(bb0[2] + s0 + dx), 
           abs(bb0[3] + s1 + dy))
              
    return (bb1, shift)

def lktrack(img1, img2, ptsI, nPtsI, winsize_ncc=10, win_size_lk=4, method=CV_TM_CCOEFF_NORMED):
    """
    **SUMMARY**
    
    Lucas-Kanede Tracker with pyramids
    
    **PARAMETERS**
    
    img1 - Previous image or image containing the known bounding box (Numpy array)
    img2 - Current image
    ptsI - Points to track from the first image
           Format ptsI[0] - x1, ptsI[1] - y1, ptsI[2] - x2, ..
    nPtsI - Number of points to track from the first image
    winsize_ncc - size of the search window at each pyramid level in LK tracker (in int)
    method - Paramete specifying the comparison method for normalized cross correlation 
             (see http://opencv.itseez.com/modules/imgproc/doc/object_detection.html?highlight=matchtemplate#cv2.matchTemplate)
    
    **RETURNS**
    
    fb - forward-backward confidence value. (corresponds to euclidean distance between).
    ncc - normCrossCorrelation values
    status - Indicates positive tracks. 1 = PosTrack 0 = NegTrack
    ptsJ - Calculated Points of second image
    
    """
    template_pt = []
    target_pt = []
    fb_pt = []
    ptsJ = [0.0]*len(ptsI)
    
    for i in range(nPtsI):
        template_pt.append((ptsI[2*i],ptsI[2*i+1]))
        target_pt.append((ptsI[2*i],ptsI[2*i+1]))
        fb_pt.append((ptsI[2*i],ptsI[2*i+1]))
    
    template_pt = np.asarray(template_pt,dtype="float32")
    target_pt = np.asarray(target_pt,dtype="float32")
    fb_pt = np.asarray(fb_pt,dtype="float32")
    
    target_pt, status, track_error = calcOpticalFlowPyrLK(img1, img2, template_pt, target_pt, 
                                     winSize=(win_size_lk, win_size_lk), flags = OPTFLOW_USE_INITIAL_FLOW,
                                     criteria = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 10, 0.03))
                                     
    fb_pt, status_bt, track_error_bt = calcOpticalFlowPyrLK(img2,img1, target_pt,fb_pt, 
                                       winSize = (win_size_lk,win_size_lk),flags = OPTFLOW_USE_INITIAL_FLOW,
                                       criteria = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 10, 0.03))
    
    for i in range(nPtsI):
        if status[i] == 1 and status_bt[i] == 1:
            status[i]=1
        else:
            status[i]=0
    ncc = normCrossCorrelation(img1, img2, template_pt, target_pt, status, winsize_ncc, method)
    fb = euclideanDistance(template_pt, target_pt)
    
    for i in range(nPtsI):
        if status[i] == 1 and status_bt[i] == 1:
            ptsJ[2 * i] = target_pt[i][0]
            ptsJ[2 * i + 1] = target_pt[i][1]
        else:
            ptsJ[2 * i] = None
            ptsJ[2 * i + 1] = None
            fb[i] = None
            ncc[i] = None
            
    return fb, ncc, status, ptsJ
    
def euclideanDistance(point1,point2):
    """
    **SUMMARY**
    
    Calculates eculidean distance between two points
    
    **PARAMETERS**
    
    point1 - vector of points
    point2 - vector of points with same length
    
    **RETURNS**
    
    match = returns a vector of eculidean distance
    """
    match=[]
    n = len(point1)
    for i in range(n):
        match.append(spsd.euclidean(point1[i],point2[i]))
    return match

def normCrossCorrelation(img1, img2, pt0, pt1, status, winsize, method=CV_TM_CCOEFF_NORMED):
    """
    **SUMMARY**
    
    Calculates normalized cross correlation for every point.
    
    **PARAMETERS**
    
    img1 - Image 1.
    img2 - Image 2.
    pt0 - vector of points of img1
    pt1 - vector of points of img2
    status - Switch which point pairs should be calculated.
             if status[i] == 1 => match[i] is calculated.
             else match[i] = 0.0
    winsize- Size of quadratic area around the point
             which is compared.
    method - Specifies the way how image regions are compared. see cv2.matchTemplate
    
    **RETURNS**
    
    match - Output: Array will contain ncc values.
            0.0 if not calculated.
 
    """
    match = []
    nPts = len(pt0)
    for i in range(nPts):
        if status[i] == 1:
            patch1 = getRectSubPix(img1,(winsize,winsize),tuple(pt0[i]))
            patch2 = getRectSubPix(img2,(winsize,winsize),tuple(pt1[i]))
            match.append(matchTemplate(patch1,patch2,method))
        else:
            match.append(0.0)
    return match

def getMedianUnmanaged(a):
    low = 0
    high = len(a) - 1
    median = (low + high) / 2
    while True:
        if high < 0:
            return 0
        if high <= low:
            return a[median]
        if high == low + 1:
            if a[low] > a[high]:
                a[low],a[high] = a[high],a[low]
            return a[median]
        middle = (low+high)/2
        if a[middle] > a[high]:
            a[middle],a[high] = a[high],a[middle]
        if a[low] > a[high]:
            a[low],a[high] = a[high],a[low]
        if a[middle] > a[low]:
            a[middle],a[low] = a[low],a[middle]
        a[middle],a[low+1] = a[low+1],a[middle]
        
        ll = low + 1
        hh = high
        
        while True:
            while True:
                ll +=1
                try:
                    if a[low] > a[ll]:
                        break                    
                except IndexError:
                    break
            while True:
                hh -= 1
                try:
                    if a[hh] > a[low]:
                        break
                except IndexError:
                    break
            if hh < ll:
                break
            try:
                a[ll],a[hh] = a[hh],a[ll]
            except IndexError:
                break
        try:
            a[low],a[hh] = a[hh],a[low]
        except IndexError:
            break
        if hh <= median:
            low = ll
        if hh >= median:
            high = hh - 1

def getMedian(a):
    median = getMedianUnmanaged(a)
    return median

def mftrack():
    cam = Camera()
    p1 = None
    p2 = None
    d = Display()
    while d.isNotDone():
        try:
            img = cam.getImage()
            a=img.save(d)
            dwn = d.leftButtonDownPosition()
            up = d.leftButtonUpPosition()
            
            if dwn:
                p1 = dwn
            if up:
                p2 = up
                break

            time.sleep(0.1)
        except KeyboardInterrupt:
            break
    if not p1 or not p2:
        return None
        
    bb = img.drawBB(p1,p2,width=5)
    i = img.copy()
    img.save(d)
    time.sleep(0.5)
    img1 = cam.getImage()
    print "fbtrack"
    while True:
        try:
            newbb, shift = fbtrack(i.getGrayNumpy(),img1.getGrayNumpy(), bb, 12, 12, 3, 12)
            print newbb, shift
            img1.drawBB((newbb[0],newbb[1]),(newbb[2],newbb[3]),width=5)
            img1.save(d)
            time.sleep(0.1)
            i = img1.copy()
            img1 = cam.getImage()
            bb = newbb
        except KeyboardInterrupt:
            break

mftrack()
