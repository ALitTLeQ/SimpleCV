from SimpleCV import *

def camshift():
    #cam = VirtualCamera("Python/SimpleCV/vid3.mp4","video")
    cam = Camera()
    img = cam.getImage()
    d = Display()
    bb1 = getBBFromUser(cam, d)
    print bb1
    bb2 = getBBFromUser(cam,d)
    print bb2
    fs1=[]
    fs2=[]
    while True:
        try:
            img1 = cam.getImage()
            fs1 = img1.track("camshift",fs1,img,bb1,num_frames=5)
            fs1[-1].drawBB()
            fs2 = img1.track("camshift",fs2,img,bb2,num_frames=5)
            fs2[-1].drawBB()
            fs1.drawAll()
            fs2.drawAll()
            img1.show()
        except KeyboardInterrupt:
            break
    
def getBBFromUser(cam, d):
    p1 = None
    p2 = None
    img = cam.getImage()
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

            time.sleep(0.05)
        except KeyboardInterrupt:
            break
    if not p1 or not p2:
        return None
    
    xmax = np.max((p1[0],p2[0]))
    xmin = np.min((p1[0],p2[0]))
    ymax = np.max((p1[1],p2[1]))
    ymin = np.min((p1[1],p2[1]))
    return (xmin,ymin,xmax-xmin,ymax-ymin)

camshift()
