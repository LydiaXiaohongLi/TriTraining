Overview

In this project, a semi-supervised learning interface using Tri-training [1] and Tri-training with disagreement [2] method have been implemented. Tri-training algorithm takes in three classifiers of different models or same model each trained with a subset of original labeled data using bootstrap sampling. Further unlabeled data will be added to a model’s training set if it can achieve agreed label prediction by other two models. Training stops when all the three models do not change anymore. Tri-training with disagreement algorithm works similarly to Tri-training algorithm except it only adds unlabeled data to one’s training set if one’s prediction disagrees with the other two models’ agreed prediction.

Implementation

TriTraining

This method is based on the pseudo code from this paper[1]. In this method, three classifiers were used to label data in each iteration. For each classifier, a proxy labeled dataset was created by the prediction of the other two classifiers. A certain number of data from this dataset would be added to the training set, after which the proxy labeled dataset will be cleared for the next run. The number of added data is controlled by the errors of the other two classifiers, which could be 0 if they have worse performance than the last run. The training process will stop when no proxy labeled data could be added to the training set.
Note that the unlabeled dataset will not be changed after each iteration.

Tri Training with Disagreement

The Tri Training with disagreement wrapped in TriTrainingwDisagreement class is coded in python and implements the algorithm from Tri Training paper [1] and adjusts for pseudo label updating only if one’s prediction disagrees with the other two models’ agreed prediction based on Tri Training with disagreement paper [2]. The TriTrainingwDisagreement class implements fit(), predict(), score() API, detailed usage refer to the session below. On a separate note, part of the implementation also uses python’s math, numpy and sklearn packages. Math package is used here for floor, ceiling math functions, numpy package used for numpy array manipulations as the fit(), predict(), score() APIs accepts/returns numpy input/output, and sklearn package used for its resample function for subsample/boostramp as required in the original algorithm [1] and accuracy score function for the score() API. 


Usage

To use these methods, users should try
from tritraining import TriTraining
from tritraining import TriTrainingwDisagreement

To initialize Self-training, users should assign a sklearn classifier which would be used in training. The TriTraining and TriTraining with Disagreement methods could initialize with a sklearn classifier (or any classifier that has fit(), predict(), score() API that is of same input/output format and function of sklearn classifier), which would be cloned three times. Or, they could initialize with a list contains three classifiers, then three of them would be used for tri-training.

Like sklearn classifiers, each methods provides three basic function APIs: fit(), predict() and score(). Those functions are similar to those of sklearn classifiers, but some of the parameters have been modified based on the semi-supervised algorithms. In particular, fit() function need not only the labeled dataset: L_X and L_y, but also the unlabeled data U_X. All of three should be numpy arrays, and U_X should have the same number of attributes as L_X. 


Contribution

Xunyi implemented the TriTraining and Self-training.

Xiaohong implemented the TriTraining with disagreement

Reference

[1] Zhou, Z.-H., & Li, M. (2005). Tri-Training: Exploiting Unlabled Data Using Three Classifiers. IEEE Trans.Data Eng., 17(11), 1529–1541.

[2] Søgaard, A. (2010). Simple semi-supervised training of part-of-speech taggers. Proceedings of the ACL 2010 Conference Short Papers.

