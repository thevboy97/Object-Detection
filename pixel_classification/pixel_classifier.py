'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''


import numpy as np
from pixel_classification.generate_rgb_data import read_pixels


class PixelClassifier():
    # theta_1, theta_2, theta_3 = 0, 0, 0
    # mu_1, mu_2, mu_3 = np.zeros(3), np.zeros(3), np.zeros(3)
    # sigma_1, sigma_2, sigma_3 = np.zeros(
    #     (3, 3)), np.zeros((3, 3)), np.zeros((3, 3))

    def __init__(self):
        '''
                Initilize your classifier with any parameters and attributes you need
        '''
        #self.YOUR_MODEL = load_model(xxx)
        self.theta_1, self.theta_2, self.theta_3 = 0.36599891716296695, 0.3245804006497022, 0.3094206821873308
        self.mu_1 = np.array(
            [0.7525060911938769, 0.34808562478245886, 0.3489122868082153])
        self.mu_2 = np.array(
            [0.3506091677705277, 0.735514889859201, 0.32949353219186034])
        self.mu_3 = np.array(
            [0.34735903110150423, 0.33111351277169015, 0.7352649546257737])
        self.sigma_3 = np.array([[0.05458537840583403, 0.008552820244218098, 0.017173502589262955],
                                 [0.008552820244218098, 0.05688307626400782,
                                     0.018308486881832904],
                                 [0.017173502589262955, 0.018308486881832904, 0.035771903528807804]])
        self.sigma_1 = np.array([[0.037086702210567674, 0.018440783894231063, 0.018632848266471717],
                                 [0.018440783894231063, 0.062014562077284355,
                                     0.008581635746996158],
                                 [0.018632848266471717, 0.008581635746996158, 0.06206845784048821]])
        self.sigma_2 = np.array([[0.055781152694426586, 0.01765326729606052, 0.008739554602480016],
                                 [0.01765326729606052, 0.03481496393718223,
                                     0.01702340107506035],
                                 [0.008739554602480016, 0.01702340107506035, 0.056068642903265964]])
        pass

    def classify(self, X):
        '''
                Classify a set of pixels into red, green, or blue

                Inputs:
                  X: n x 3 matrix of RGB values
                Outputs:
                  y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
        '''
        ################################################################
        # YOUR CODE AFTER THIS LINE
        y = np.zeros(len(X))
        for i in range(len(X)):
            pdf_1 = np.log(self.theta_1) - 0.5 * \
                np.log(np.linalg.det(self.sigma_1))-0.5 * \
                (X[i]-self.mu_1).dot(np.linalg.inv(self.sigma_1)).dot(X[i]-self.mu_1)
            pdf_2 = np.log(self.theta_2) - 0.5 * \
                np.log(np.linalg.det(self.sigma_2))-0.5 * \
                (X[i]-self.mu_2).dot(np.linalg.inv(self.sigma_2)).dot(X[i]-self.mu_2)
            pdf_3 = np.log(self.theta_3) - 0.5 * \
                np.log(np.linalg.det(self.sigma_3))-0.5 * \
                (X[i]-self.mu_3).dot(np.linalg.inv(self.sigma_3)).dot(X[i]-self.mu_3)

            y[i] = np.argmax([pdf_1, pdf_2, pdf_3])+1

        return y
        # Just a random classifier for now
        # Replace this with your own approach

        #y = predict(self.YOUR_MODEL, X)
        # return y

        # YOUR CODE BEFORE THIS LINE
        ################################################################

    def train(self):
        '''
        The training code. Include how you process data, how to train the model.
        Can be multiple functions.
        '''
        # process_data(dataset)
        # save_model(xxx)

        folder = 'data/training'
        X1 = read_pixels(folder+'/red', verbose=True)
        X2 = read_pixels(folder+'/green')
        X3 = read_pixels(folder+'/blue')
        y1, y2, y3 = np.full(X1.shape[0], 1), np.full(
            X2.shape[0], 2), np.full(X3.shape[0], 3)

        # train using GNB
        len_y = len(y1)+len(y2)+len(y3)
        theta_1, theta_2, theta_3 = len(
            y1)/len_y, len(y2)/len_y, len(y3)/len_y
        mu_1, mu_2, mu_3 = np.average(X1, axis=0), np.average(
            X2, axis=0), np.average(X3, axis=0)
        sigma_1, sigma_2, sigma_3 = np.cov(
            X1.T), np.cov(X2.T), np.cov(X3.T)

        np.savetxt('Training_theta.txt', [
                   theta_1, theta_2, theta_3], fmt="%s")
        np.savetxt('Training_mu.txt', [
                   mu_1, mu_2, mu_3], fmt="%s")
        np.savetxt('Training_sigma_red.txt', sigma_1, fmt="%s")
        np.savetxt('Training_sigma_green.txt', sigma_2, fmt="%s")
        np.savetxt('Training_sigma_blue.txt', sigma_3, fmt="%s")
        pass

# # folder_test = 'data/validation/blue'

# # X = grd.read_pixels(folder_test)
# # myPixelClassifier = PixelClassifier()
# # y = myPixelClassifier.classify(X)

# # print('Precision: %f' % (sum(y == 1)/y.shape[0]))
