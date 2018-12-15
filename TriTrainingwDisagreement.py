import numpy as np
import sklearn
import math

class TriTrainingwDisagreement():

    def __init__(self, classifier):
        """
        args:
            classifier - classifier, with .fit, .predict, .score API (refer to classifiers of sklearn)
        """
        # Initialize
        self.clf = [sklearn.base.clone(classifier) for i in range(3)]

    def measure_error(self, j, k):
        """
        args:
                j - int, classifier index
                k - int, classifier index
        return:
                float, classification_error
        """
        y_predict_j = self.clf[j].predict(self.X_label)
        y_predict_k = self.clf[k].predict(self.X_label)
        return (1 - np.sum((y_predict_j == y_predict_k) & (y_predict_j == self.y_label)) / np.sum(y_predict_j == y_predict_k))

    def fit(self, X_label, y_label, X_unlabel):
        """
        args:
                X_label - labeled train feature vector (ndarray of size, # of samples * # of features), features are numeric numbers
                y_label - labeled train label vector (ndarray of size, # of samples), labels are numeric numbers
                X_unlabel - test feature vector (ndarray of size, # of samples * # of features), features are numeric numbers
        """
        self.X_label = X_label
        self.y_label = y_label

        classification_error_current = [0.5, 0.5, 0.5]
        classification_error = [0.5, 0.5, 0.5]
        pseudo_label_size_current = [0, 0, 0]
        pseudo_label_size = [0, 0, 0]
        # pseudo_label_index used to compare and check if tri-training can be stopped, when two iterations have the same label_index, means tri-training can be stopped
        X_pseudo_label_index = [[], [], []]
        X_pseudo_label_index_current = [[], [], []]

        feature_size = self.X_label.shape[1]

        # Train each classifier with bootstrampped subset
        for i in range(3):
            X_resample, y_resample = sklearn.utils.resample(self.X_label, self.y_label)  # BootstrapSample(L)
            self.clf[i].fit(X_resample, y_resample)  # Learn(Si)

        iteration = 0
        while (True):

            update = [False, False, False]

            iteration = iteration + 1
            for i in range(3):
                X_pseudo_label_index_current[i] = X_pseudo_label_index[i]

            # Step3.1 Set Li = empty set, Li denotes the new pseudo label set determined by tri-training iteration for classifier i
            # X_pseudo_label_index, contains the data record index (in the full unlabelled set) of the new pseudo label set determined by tri-training iteration for classifier i
            # X_pseudo_label, contains the features for new pseudo label set determined by tri-training iteration for classifier i
            # y_pseudo_label, contains the labels (not ground truth label, but pseudo label calculated by tri-training iteration) for new pseudo label set determined by tri-training iteration for classifier i
            X_pseudo_label_index = [[], [], []]
            X_pseudo_label = [[], [], []]
            y_pseudo_label = [[], [], []]

            # Step 3.2 Loop through all the data record in unlabelled set
            for i in range(3):
                j, k = np.delete(np.array([0, 1, 2]), i)
                classification_error[i] = self.measure_error(j, k)
                if classification_error[i] < classification_error_current[i]:
                    # Step 3.2 If classifier j,k aggrees with the label for one data record, and not agree with classifier i, in unlabelled set,
                    # then add the data record into Li
                    y_predict_j = self.clf[j].predict(X_unlabel)
                    y_predict_k = self.clf[k].predict(X_unlabel)
                    y_predict_i = self.clf[i].predict(X_unlabel)
                    y_pseudo_label[i] = y_predict_j[np.logical_and(y_predict_j==y_predict_k,y_predict_j!=y_predict_i)]
                    X_pseudo_label_index[i] = np.where(np.logical_and(y_predict_j==y_predict_k,y_predict_j!=y_predict_i))

                    pseudo_label_size[i] = len(X_pseudo_label_index[i])
                    #print("classification_error: {}, classification_error_current: {}, pseudo_label_size: {}, pseudo_label_size_current: {}".format(classification_error[i], classification_error_current[i], pseudo_label_size[i],pseudo_label_size_current[i]))

                    if pseudo_label_size_current[i] == 0:
                        pseudo_label_size_current[i] = math.floor(classification_error[i] / (classification_error_current[i] - classification_error[i]) + 1)
                    if pseudo_label_size_current[i] < pseudo_label_size[i]:
                        if ((classification_error[i] * pseudo_label_size[i]) < (classification_error_current[i] * pseudo_label_size_current[i])):
                            update[i] = True
                        elif pseudo_label_size_current[i] > (classification_error[i] / (classification_error_current[i] - classification_error[i])):
                            resample_size = math.ceil(classification_error_current[i] * pseudo_label_size_current[i] / classification_error[i] - 1)
                            X_pseudo_label_index[i], y_pseudo_label[i] = sklearn.utils.resample(X_pseudo_label_index[i],y_pseudo_label[i],replace=False,n_samples=resample_size)
                            pseudo_label_size[i] = len(X_pseudo_label_index[i])
                            update[i] = True

            # Step 3.3 Train all the three classifiers with Li + original labelled data set
            for i in range(3):
                if update[i] == True:
                    #print("number of pseudo labels added for classifier {} is: {}".format(i,len(X_pseudo_label_index[i])))
                    X_pseudo_label[i] = np.array(X_unlabel[X_pseudo_label_index[i]])
                    self.clf[i].fit(np.concatenate((X_pseudo_label[i], self.X_label), axis=0),np.concatenate((np.array(y_pseudo_label[i]), self.y_label), axis=0))
                    classification_error_current[i] = classification_error[i]
                    pseudo_label_size_current[i] = pseudo_label_size[i]

            # Stop tri-training process, if the pseudo label data set added in current tri-training iteration
            # is the same for last tri-training iteration for all classifiers
            if (np.array_equal(X_pseudo_label_index[0], X_pseudo_label_index_current[0]) & np.array_equal(X_pseudo_label_index[1], X_pseudo_label_index_current[1])
                    & np.array_equal(X_pseudo_label_index[2], X_pseudo_label_index_current[2])):
                break

    def predict(self, X_test):
        """
        args:
                X_test - test feature vector (ndarray of size, # of samples * # of features), features are numeric numbers
        return:
                array of size (# of test samples), with values as predicted label 1 or 0
        """
        I = self.clf[0].predict(X_test)
        J = self.clf[1].predict(X_test)
        K = self.clf[2].predict(X_test)
        I[J == K] = J[J == K]
        return I

    def score(self, X_test, y_test):
        """
        args:
                X_test - test feature vector (ndarray of size, # of samples * # of features), features are numeric numbers
                y_test - test label vector (ndarray of size, # of samples), labels are numeric numbers
        return:
                float, accuracy_score of predicted value by the tri-training (with disagreement) classifier against groud truth
        """
        return sklearn.metrics.accuracy_score(y_test, self.predict(X_test))