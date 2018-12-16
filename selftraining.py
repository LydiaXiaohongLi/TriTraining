import numpy as np
import sklearn

class SelfTraining:
    def __init__(self, classifier):
        self.classifier = sklearn.base.clone(classifier)
    
    def fit(self, L_X, L_y, U_X, tau):
        improve =  True
        self.iter = 0
        while improve and len(U_X) !=0:
            self.classifier.fit(L_X, L_y)
            U_prob = self.classifier.predict_proba(U_X)
            U_label = self.classifier.predict(U_X)
            label_index = np.argmax(U_prob, axis = 1)>tau

            if sum(label_index) ==0:
                improve = False
            self.iter += 1
            L_X = np.append(L_X, U_X[label_index], axis=0)
            L_y = np.append(L_y, U_label[label_index])
            U_X = np.delete(U_X, np.where(label_index), axis=0)


    def predict(self, X):
        return self.classifier.predict(X)
        
    def score(self, X, y):
        return sklearn.metrics.accuracy_score(y, self.predict(X))
