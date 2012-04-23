import gtk
from threading import Thread
import gobject
import time
gtk.gdk.threads_init()

class Display():
    def __init__(self,title="imagegtk"):
        """
        Constructs gtk window and adds widgets
        parameters:
            title - A string for window title
        """
        self.img = None
        self.img_gtk = None
        self.img_thrd = None
        self.img_thrd_flag = False
        self.interval = 0.1
        self.mouseX = 0
        self.mouseY = 0
        self.mouse_rawX = 0
        self.mouse_rawY = 0
        self.done = False
        self.thrd = None
        self.mouseLeft = 0
        self.mouseRight = 0
        self.mouseMiddle = 0
        self.leftButtonDown = 0
        self.rightButtonDown = 0
        
        self.win = gtk.Window()
        self.win.set_title(title)
        self.win.connect("delete_event",self.leave_app)
        self.box = gtk.VBox(False,2)
        self.win.add(self.box)
        self.box.show()
        self.image_box = gtk.EventBox()
        self.image_box.connect("motion_notify_event",self.motion_callback)
        self.image_box.connect("button_press_event",self.press_callback)
        self.image_box.connect("button_release_event",self.release_callback)
        self.box.pack_start(self.image_box,False,False,2)
        
    def show(self,image):
        """
        Summary:
            Creates a thread to show image and calls show_image
            
        Parameters:
            image - cv2.cv.iplimage or iplimage
        """
        if self.img_thrd:
            # Thread exists. Hence change the flag to end the thread.
            self.img_thrd_flag = False
        self.img_thrd = Thread(target = self.__showimage,args=(image,),name="show_image")
        self.img_thrd.start()
    
    def __showimage(self,image):
        """
        Summary:
            Creates pixbuf from image data. Sets image from pixbuf.
            Creates a thread to call gtk.main()
            
        Parameters:
            image - cv2.cv.iplimage or iplimage
            
        NOTE:
            This function shouldn't be called directly
        """
        if self.done == True and self.img_gtk is None:
            self.done = False
            self.__init__()
        elif self.done == True:
            self.done = False
            
        self.img = image
        self.img_thrd_flag = True
        
        if self.img_gtk is None:
            self.img_flag=0
            self.img_gtk = gtk.Image()
            self.image_box.add(self.img_gtk)
            
        self.img_pixbuf = gtk.gdk.pixbuf_new_from_data(self.img.tostring(),
                                                        gtk.gdk.COLORSPACE_RGB,
                                                        False,
                                                        self.img.depth,
                                                        self.img.width,
                                                        self.img.height,
                                                        self.img.width*self.img.nChannels)
        self.img_gtk.set_from_pixbuf(self.img_pixbuf)
        self.img_gtk.show()
        self.win.show_all()
        
        if not self.img_flag:
            self.__threadgtk()
            self.img_flag=1                     
            
        while (not self.done and self.img_thrd_flag) :
            time.sleep(self.interval/2)
    
    def __threadgtk(self):
        self.thrd = Thread(target=gtk.main, name = "GTK thread")
        self.thrd.daemon = True
        self.thrd.start()
    
    def leave_app(self,widget,data=None):
        self.quit()
    
    def release_callback(self,widget,event):
        self.mouseX = int(event.x)
        self.mouseY = int(event.y)
        self.mouse_rawX = int(event.x_root)
        self.mouse_rawY = int(event.y_root)
        self.setReleaseButtonState(event.button)
    
    def press_callback(self,widget,event):
        self.mouseX = int(event.x)
        self.mouseY = int(event.y)
        self.mouse_rawX = int(event.x_root)
        self.mouse_rawY = int(event.y_root)
        self.setPressButtonState(event.button)
        
    def setPressButtonState(self, mode):
        if mode == 1:
            self.leftButtonDown = 1
            self.mouseLeft = 1
        elif mode == 2:
            self.rightButtonDown = 1
            self.mouseLeft = 1
        elif mode == 3:
            self.mouseMiddle = 1
            
    def setReleaseButtonState(self, mode):
        if mode == 1:
            self.leftButtonDown = 0
            self.mouseLeft = 0
        elif mode == 2:
            self.rightButtonDown = 0
            self.mouseLeft = 0
        elif mode == 3:
            self.mouseMiddle = 0
        
    def motion_callback(self,widget,event):
        self.mouseX = int(event.x)
        self.mouseY = int(event.y)
        self.mouse_rawX = int(event.x_root)
        self.mouse_rawY = int(event.y_root)
    
    def isDone(self):
        return self.done
    
    def quit(self):
        self.done = True
        self.img_gtk = None
        self.win.destroy()
