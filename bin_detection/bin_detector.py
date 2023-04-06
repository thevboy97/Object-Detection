'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops



class BinDetector():
    def __init__(self):
        '''
                Initilize your bin detector with the attributes you need,
                e.g., parameters of your classifier
        '''
        self.theta_1, self.theta_2, self.theta_3, self.theta_4 = np.loadtxt('bin_detection/bin_training_theta_RGB.txt')
        self.mu_1, self.mu_2,  self.mu_3, self.mu_4 = np.loadtxt('bin_detection/bin_training_mu_RGB.txt')
        self.sigma_1, self.sigma_2, self.sigma_3, self.sigma_4 = np.loadtxt('bin_detection/bin_training_sigma_binblue_RGB.txt'),np.loadtxt('bin_detection/bin_training_sigma_otherblue_RGB.txt'), np.loadtxt('bin_detection/bin_training_sigma_green_RGB.txt'), np.loadtxt('bin_detection/bin_training_sigma_brown_RGB.txt')

        # self.theta_1, self.theta_2, self.theta_3, self.theta_4 = np.loadtxt('bin_training_theta_RGB.txt')
        # self.mu_1, self.mu_2,  self.mu_3, self.mu_4 = np.loadtxt('bin_training_mu_RGB.txt')
        # self.sigma_1, self.sigma_2, self.sigma_3, self.sigma_4 = np.loadtxt('bin_training_sigma_binblue_RGB.txt'),np.loadtxt('bin_training_sigma_otherblue_RGB.txt'), np.loadtxt('bin_training_sigma_green_RGB.txt'), np.loadtxt('bin_training_sigma_brown_RGB.txt')    

        pass

    def segment_image(self, img):
        '''
                Obtain a segmented image using a color classifier,
                e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
                call other functions in this class if needed

                Inputs:
                        img - original image
                Outputs:
                        mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE

        # Replace this with your own approach

        # convert image to matrix
        #img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        shp = np.shape(img)
        #dim = shp[0]*shp[1]
        # X = np.reshape(test_img_RGB, (dim, 3))

        mask_img = np.zeros((shp[0],shp[1]))
        
        det_sigma_1 = np.linalg.det(self.sigma_1)
        det_sigma_2 = np.linalg.det(self.sigma_2)
        det_sigma_3 = np.linalg.det(self.sigma_3)
        det_sigma_4 = np.linalg.det(self.sigma_4)


        inv_sigma_1 = np.linalg.inv(self.sigma_1)
        inv_sigma_2 = np.linalg.inv(self.sigma_2)
        inv_sigma_3 = np.linalg.inv(self.sigma_3)
        inv_sigma_4 = np.linalg.inv(self.sigma_4)

        for i in range(shp[0]):
            for j in range(shp[1]):
                pdf_1 = np.log(self.theta_1) - 0.5 * np.log(det_sigma_1)-0.5 *(img[i,j]/255-self.mu_1).dot(inv_sigma_1).dot(img[i,j]/255-self.mu_1)
                pdf_2 = np.log(self.theta_2) - 0.5 * np.log(det_sigma_2)-0.5 *(img[i,j]/255-self.mu_2).dot(inv_sigma_2).dot(img[i,j]/255-self.mu_2)
                pdf_3 = np.log(self.theta_3) - 0.5 * np.log(det_sigma_3)-0.5 *(img[i,j]/255-self.mu_3).dot(inv_sigma_3).dot(img[i,j]/255-self.mu_3)
                pdf_4 = np.log(self.theta_4) - 0.5 * np.log(det_sigma_4)-0.5 *(img[i,j]/255-self.mu_4).dot(inv_sigma_4).dot(img[i,j]/255-self.mu_4)

                if np.argmax([pdf_1, pdf_2, pdf_3, pdf_4]) == 0:
                    mask_img[i,j] = 1
                else:
                    mask_img[i,j] = 0

        # plt.imshow(mask_img)
        # plt.show()
        return mask_img
        # convert mask matrix to image

        # YOUR CODE BEFORE THIS LINE
        ################################################################





    def get_bounding_boxes(self, img):
        '''
                Find the bounding boxes of the recycling bins
                call other functions in this class if needed

                Inputs:
                        img - original image
                Outputs:
                        boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
                        where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE


        #img = cv2.imread(os.path.join(folder,filename))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
		#mask = self.segment_image(img)

        lab = label(img) 
        props = regionprops(lab)
        boxes = []

        for prop in props:
            h =abs(prop.bbox[0]-prop.bbox[2])
            w = abs(prop.bbox[1]-prop.bbox[3])
            
            if 0.8<h/w<2.5 and w*h>5000:	
	

                temp = list(prop.bbox)
                temp[0], temp[1] = temp[1], temp[0]
                temp[2], temp[3] = temp[3], temp[2]
                boxes.append(temp)

        	#cv2.rectangle(img, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 2)	
			
		#print(boxes)
		#plt.imshow(img)
		#plt.show()


        # img = img.astype(np.uint8)
        # shp = np.shape(img)
        # contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # boxes = []
        # for con in contours:
        #     x,y,w,h = cv2.boundingRect(con)
        #     #area = cv2.contourArea(c)
        #     ar = float(w/h)
        #     #ext = float(area)/(w*h)

        #     if w/h>1.25 or w/h<0.3 or w*h<3000:
        #         continue
        #     if ar<1.1 or w*h>0.6*(shp[0]*shp[1]):
        #         # cv2.rectangle(img,x,y)
        #         boxes.append([x,y,x+w,y+h])


        # Replace this with your own approach
        # x = np.sort(np.random.randint(img.shape[0], size=2)).tolist()
        # y = np.sort(np.random.randint(img.shape[1], size=2)).tolist()
        # boxes = [[x[0], y[0], x[1], y[1]]]
        # boxes = [[182, 101, 313, 295]]

        # YOUR CODE BEFORE THIS LINE
        ################################################################

        return boxes

    def train(self):
        '''
        The training code. Include how you process data, how to train the model.
        Can be multiple functions.
        '''
        # process_data(dataset)
        # save_model(xxx)
        ext = ['_RGB.txt', '_YUV.txt']
        #folder = 'data/training'
        for i in ext:
            X1 = np.loadtxt('binblue'+i, dtype=float)
            X2 = np.loadtxt('otherblue'+i, dtype=float)
            X3 = np.loadtxt('green'+i, dtype=float)
            X4 = np.loadtxt('brown'+i, dtype=float)
            y1, y2, y3, y4 = np.full(X1.shape[0], 1),np.full(X2.shape[0], 2), np.full(X3.shape[0], 3), np.full(X4.shape[0], 4)

            # train using GNB
            len_y = len(y1)+len(y2)+len(y3)+len(y4)
            theta_1,theta_2, theta_3, theta_4 = len(
                y1)/len_y, len(y2)/len_y, len(y3)/len_y, len(y4)/len_y
            mu_1,mu_2, mu_3, mu_4 = np.average(X1, axis=0),np.average(X2, axis=0),  np.average(X3, axis=0), np.average(X4, axis=0)
            sigma_1,sigma_2, sigma_3, sigma_4 = np.cov(
                X1.T), np.cov(X2.T), np.cov(X3.T), np.cov(X4.T)

            np.savetxt('bin_training_theta'+i, [
                theta_1,theta_2, theta_3, theta_4], fmt="%s")
            np.savetxt('bin_training_mu'+i, [
                mu_1, mu_2, mu_3, mu_4], fmt="%s")
            np.savetxt('bin_training_sigma_binblue' +
                       i, sigma_1, fmt="%s")
            np.savetxt('bin_training_sigma_otherblue' +
                       i, sigma_2, fmt="%s")
            np.savetxt('bin_training_sigma_green'+i, sigma_3, fmt="%s")
            np.savetxt('bin_training_sigma_brown'+i, sigma_4, fmt="%s")
        pass


# b = BinDetector()
# b.train()

# b = BinDetector()
# b.segment_image('data/validation/0061.jpg')
