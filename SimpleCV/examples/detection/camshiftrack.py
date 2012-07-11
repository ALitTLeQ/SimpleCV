from SimpleCV import *

def camshift():
    cam = Camera()
    img = cam.getImage()
    d = Display()
    bb = getBBFromUser(cam, d)
    print bb
    while True:
        try:
            img1 = cam.getImage()
            bb = img1.track(img, "camshift", bb)
            #print bb
            img = img1
            img1.drawRectangle(bb[0],bb[1],bb[2],bb[3])
            img1.show()
            time.sleep(0.05)
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
