from SimpleCV.base import np, warnings
from SimpleCV.ImageClass import Image
try:
    import pywt
    print "pywt imported"
except ImportError:
    warnings.warn("DWT requires pywt library which SimpleCV can't find.")

class DWT:
    def __init__(self):
        pass

    @classmethod
    def dwt2(self, img, wavelet="haar", mode="sym"):
        npimg = img.getGrayNumpyCv2()
        a, (h, v, d) = pywt.dwt2(npimg, wavelet=wavelet, mode=mode)
        imgs = [Image(img, cv2image=True) for img in [a, h, v, d]]
        return imgs

    @classmethod
    def idwt2(self, imgs, wavelet="haar", mode="sym"):
        a, (h, v, d) = imgs
        np_a = a.getGrayNumpyCv2()
        np_h = h.getGrayNumpyCv2()
        np_v = v.getGrayNumpyCv2()
        np_d = d.getGrayNumpyCv2()
        img = pywt.idwt2((np_a, (np_h, np_v, np_d)), wavelet=wavelet, mode=mode)
        retVal = Image(img, cv2image=True)
        return retVal

    @classmethod
    def _getDescriptor(self, img, levels=4):
        coeffs = pywt.wavedec2(img, "haar", level=levels)
        descriptor = []
        stdlist = []
        for i in xrange(1,levels):
            for transform in coeffs[i]:
                stdval = np.std(transform)
                descriptor.append(stdval/2.0**(i-1))
                stdlist.append(stdval)
        stdval = np.std(coeffs[0])
        descriptor.append(stdval/2.0**(i-1))
        stdlist.append(stdval)
        descriptor.append(np.average(coeffs[0]))
        stdlist.append(np.average(coeffs[0]))
        return descriptor, stdlist

    @classmethod
    def _getContentDescriptor(self, img):
        imgY = img.toYCrCb()
        npimg = imgY.getNumpyCv2()
        Y = npimg[:, :, 0]
        Cr = npimg[:, :, 1]
        Cb = npimg[:, :, 2]
        YDescriptor, YStd = self._getDescriptor(Y)
        CrDescriptor, CrStd = self._getDescriptor(Cr)
        CbDescriptor, CbStd = self._getDescriptor(Cb)
        descriptor = YDescriptor + CrDescriptor + CbDescriptor
        stdlist = YStd + CrStd + CbStd
        return np.array(descriptor), np.array(stdlist)

    @classmethod
    def getSimilarity(self, img1, img2):
        descriptor1, stdlist1 = self._getContentDescriptor(img1)
        descriptor2, stdlist2 = self._getContentDescriptor(img2)
        des = np.abs(descriptor1 - descriptor2)
        print des
        if not np.any(des):
            warnings.warn("Both descriptors are similar")
            return 0
        std = np.std((stdlist1, stdlist2), axis=0)
        print std
        # still issues here
        cost = np.divide(des,stdlist1)
        return sum(cost)
