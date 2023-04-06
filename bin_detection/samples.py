

'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import os
import cv2
import numpy as np
from roipoly import RoiPoly
from matplotlib import pyplot as plt

if __name__ == '__main__':

    # read the first training image
    #f = ['data/trial/blue','data/trial/green']
    folder = 'data/training/otherblue'
    i = 0
    for filename in os.listdir(folder):

        # X[i] = img[0, 0].astype(np.float64)/255
        # i += 1

        # filename = '0001.jpg'
        img = cv2.imread(os.path.join(folder, filename))
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # display the image and use roipoly for labeling
        fig, ax = plt.subplots()
        ax.imshow(img_RGB)
        my_roi = RoiPoly(fig=fig, ax=ax, color='r')

    # get the image mask
        mask = my_roi.get_mask(img_RGB)

    # display the labeled region and the image mask
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('%d pixels selected\n' % img_RGB[mask, :].shape[0])

        ax1.imshow(img_RGB)
        ax1.add_line(plt.Line2D(
            my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
        ax2.imshow(mask)

        plt.show(block=True)

        mat_RGB_temp = img_RGB[mask == 1].astype(np.float64)/255
        mat_YUV_temp = img_YUV[mask == 1].astype(np.float64)/255

        if i == 0:
            mat_RGB = mat_RGB_temp
            mat_YUV = mat_YUV_temp
            i += 1
        else:
            mat_RGB = np.append(mat_RGB, mat_RGB_temp, axis=0)
            mat_YUV = np.append(mat_YUV, mat_YUV_temp, axis=0)

    #trans_mat = np.array([[1/3, 1/3, 1/3], [0, -1, 1], [1, -1, 0]])
    #np.savetxt('bin_blue_RGB.txt', mat, fmt="%s")

    # mat_YUV = np.zeros((len(mat), 3))
    # for i in range(len(mat)):
    #     mat_YUV[i] = np.matmul(trans_mat, mat[i])

    np.savetxt('otherblue_RGB.txt', mat_RGB, fmt="%s")
    np.savetxt('otherblue_YUV.txt', mat_YUV, fmt="%s")
