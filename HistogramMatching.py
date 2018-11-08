import matplotlib
matplotlib.use('agg')

import numpy as np
import cv2
import sys

import histogram as h
import cumulative_histogram as ch
import scipy.stats as ss

import matplotlib.pyplot as plt 

print(sys.argv)

#%matplotlib inline
#plt.style.use('classic') 
#plt.rcParams['figure.figsize'] = (20, 8)

##'/home/dipayan/Pictures/1.jpg'
##'/home/dipayan/Pictures/4.png'
img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
img_ref = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

height = img.shape[0]
width = img.shape[1]
pixels = width * height

height_ref = img_ref.shape[0]
width_ref = img_ref.shape[1]
pixels_ref = width_ref * height_ref

hist = h.histogram(img)
hist_ref = h.histogram(img_ref)

cum_hist = ch.cumulative_histogram(hist)
cum_hist_ref = ch.cumulative_histogram(hist_ref)

prob_cum_hist = cum_hist / pixels

prob_cum_hist_ref = cum_hist_ref / pixels_ref

K = 256
new_values = np.zeros((K))

print('Preproccessing done...')

for a in np.arange(K):
    j = K - 1
    while True:
        new_values[a] = j
        j = j - 1
        if j < 0 or prob_cum_hist[a] > prob_cum_hist_ref[j]:
            break

print('Cumulative sum found...')

for i in np.arange(height):
    for j in np.arange(width):
        a = img.item(i,j)
        b = new_values[a]
        img.itemset((i,j), b)

cv2.imwrite('/home/dipayan/Pictures/output/matched.jpg', img)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()




ori_img = cv2.imread(sys.argv[1],0)
chn_img = cv2.imread('/home/dipayan/Pictures/output/matched.jpg',0)

match_pixel = chn_img.shape[0]*chn_img.shape[1]
c_hist_m = (ch.cumulative_histogram(h.histogram(chn_img)))/match_pixel

f, ax = plt.subplots(2,4,figsize=(20,8))

ax[0][0].imshow(ori_img, cmap='gray')
ax[0][1].plot(range(256),prob_cum_hist)
ax[0][2].plot(range(256), h.histogram(ori_img)/pixels)
ax[0][3].plot(range(256),prob_cum_hist_ref)
ax[0][3].plot(range(256),c_hist_m)

ax[1][0].imshow(chn_img, cmap='gray')
ax[1][1].plot(range(256), c_hist_m)
ax[1][2].plot(range(256), h.histogram(chn_img)/match_pixel)
ax[1][3].plot(range(256), hist_ref/pixels_ref)
ax[1][3].plot(range(256), h.histogram(chn_img)/match_pixel)

f.savefig('output1.png')




def get_plot(image, pdf, cdf):
    K = 256
    _new_value = np.zeros((K))
    
    _heigth = image.shape[0]
    _width = image.shape[1]
    _pixels = _heigth * _width
    _hist = h.histogram(image)
    _c_hist = ch.cumulative_histogram(_hist)
    _prob_cum_hist = _c_hist / _pixels

    
    
    for a in np.arange(K):
        j = K - 1
        while True:
            _new_value[a] = j
            j = j - 1
            if j < 0 or _prob_cum_hist[a] > cdf[j]:
                break
            
    
    for i in np.arange(_heigth):
        for j in np.arange(_width):
            a = image.item(i,j)
            b = _new_value[a]
            image.itemset((i,j), b)
        
    cv2.imwrite('_matched_img.jpg',image)
        
    _ori_img = cv2.imread(sys.argv[1],0)
    _chn_img = cv2.imread('_matched_img.jpg',0)
    
    _match_pixel = _chn_img.shape[0]*_chn_img.shape[1]
    _c_hist_m = (ch.cumulative_histogram(h.histogram(_chn_img)))/_match_pixel
    
    fig, ax = plt.subplots(2,4)
    ax[0][0].imshow(_ori_img, cmap='gray')
    ax[0][0].set_title('Original image')
    ax[0][1].plot(range(256),prob_cum_hist)
    ax[0][1].set_title('Original prob_cum_hist')
    ax[0][2].plot(range(256), h.histogram(_ori_img) / pixels)
    ax[0][2].set_title('Original Normalized Histogram')
    ax[0][3].plot(range(256), cdf)
    ax[0][3].plot(range(256), _c_hist_m)
    ax[0][2].set_title('Original and other Histogram')
    
    ax[1][0].imshow(_chn_img, cmap='gray')
    ax[1][0].set_title('changed image')
    ax[1][1].plot(range(256), _c_hist_m)
    ax[1][1].set_title('changed prob_cum_hist')
    ax[1][2].plot(range(256), h.histogram(_chn_img) / _match_pixel)
    ax[1][2].set_title('changed Normalized Histogram')
    ax[1][3].plot(range(256), pdf)
    ax[1][3].plot(range(256), h.histogram(_chn_img) / _match_pixel)
    ax[0][2].set_title('changed pdf Histogram')


#normal distribution
x = np.linspace(0, 256, 256)
pdf = ss.norm.pdf(x, 128, 20)
cdf = ss.norm.cdf(x, 128, 20)
img1 = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
get_plot(img1, pdf, cdf)
f.savefig('output2.png')