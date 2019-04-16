import cv2
import numpy as np
import random


def ComputeHist(img):
    h, w = img.shape
    hist, bin_edge = np.histogram(img.reshape(1, w * h), bins=list(range(257)))
    return hist


def ComputeMinLevel(hist, rate, pnum):
    sum = 0
    for i in range(256):
        sum += hist[i]
        if (sum >= (pnum * rate * 0.01)):
            return i


def ComputeMaxLevel(hist, rate, pnum):
    sum = 0
    for i in range(256):
        sum += hist[255 - i]
        if (sum >= (pnum * rate * 0.01)):
            return 255 - i


def LinearMap(minlevel, maxlevel):
    if (minlevel >= maxlevel):
        return []
    else:
        newmap = np.zeros(256)
        for i in range(256):
            if (i < minlevel):
                newmap[i] = 0
            elif (i > maxlevel):
                newmap[i] = 255
            else:
                newmap[i] = (i - minlevel) / (maxlevel - minlevel) * 255
        return newmap


def CreateNewImg(img):
    h, w, d = img.shape
    newimg = np.zeros([h, w, d])
    for i in range(d):
        imgmin = np.min(img[:, :, i])
        imgmax = np.max(img[:, :, i])
        imghist = ComputeHist(img[:, :, i])
        minlevel = ComputeMinLevel(imghist, 8.3, h * w)
        maxlevel = ComputeMaxLevel(imghist, 2.2, h * w)
        newmap = LinearMap(minlevel, maxlevel)
        if (newmap.size == 0):
            continue
        for j in range(h):
            newimg[j, :, i] = newmap[img[j, :, i]]
    return newimg



def add_noise(ndarray):
    ntime=random.randint(10,150)
    for i in range(ntime):
        y=random.randint(5,ndarray.shape[0]-5)
        x= random.randint(5, ndarray.shape[1] - 5)
        noise=np.random.randint(0,255,[1,1,3])
     #   print noise
        ndarray[y,x]=noise
    return ndarray
def erodergb(ndarray):

    erosion = cv2.dilate(ndarray, kernel=None, iterations=1)
    return erosion

def tiue(ndarray):

    ndarray=np.array(ndarray,np.float).copy()
    ndarray-=5
    for i in range(3):
        if random.random()>=0.5:
            ndarray[:,:,i]+=random.randint(0,10)
            if np.min(ndarray[:,:,i])<0:
                ndarray[:,:,i]-=np.min(ndarray[:,:,i])
            ndarray[:,:,i]=255.0*(np.array(ndarray[:,:,i]/float(np.max(ndarray[:,:,i]))))
    m=np.array(ndarray,np.uint8).copy()
    return m


def adjust_gamma(image):
    gamma=random.random()*2+1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def lab(image):
    #### it work very well
    # image=np.array(image,np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab", lab)
    # cv2.waitKey(0)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))


    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # final=np.array(final,np.float32)
    # cv2.imshow("lab", final)
    # cv2.waitKey(0)
    return final

def function_line(Image):
    # its finall tool
    # print(Image.dtype)
    Image=np.array(Image,np.uint8)
    funct_list=[add_noise]
    Image=lab(Image)

    for i in range(len(funct_list)):
        if random.random()>0.7:
            Image=funct_list[i](Image)
            # cv2.imshow('after'+str(i),Image)
            # cv2.waitKey(0)
    Image = np.array(Image, np.float32)
    return Image

def test_function(Image):
    print(Image.dtype)
    return Image

if __name__=='__main__':
    image=cv2.imread('/Disk4/xkp/dataset/1.jpg')
    # image=np.array(image,np.float32)
    # image=lab(image)
    # print(image.dtype)

    image=lab(image)
    cv2.imshow('duibidu',image)

    # image=adjust_gamma(image)
    # cv2.imshow('gamma',image)
    cv2.waitKey(0)



# TODO
# ruihua              (no)
# baipingheng         (no)
# dui bi du la shen   (ok)
#