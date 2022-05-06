#!/usr/bin/env python

from sklearn.linear_model import LogisticRegression
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

def main():
    transform = transforms.Compose([
            transforms.Resize(255),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ])
    grey_flattened_trn = datasets.ImageFolder('train/', transform)
    grey_flattened_tst = datasets.ImageFolder('test/', transform)
    grey_flattened_valid = datasets.ImageFolder('valid/', transform)
    image_grey_flat, label = grey_flattened_trn[0]
    print(image_grey_flat.shape)  # [307200] = 480*640, resize(255) => 86700
    trn_dataloader = DataLoader(dataset=grey_flattened_trn, batch_size=len(grey_flattened_trn))
    tst_dataloader = DataLoader(dataset=grey_flattened_tst, batch_size=len(grey_flattened_tst))
    val_dataloader = DataLoader(dataset=grey_flattened_valid, batch_size=len(grey_flattened_valid))

    logModel = LogisticRegression()
    svm_classifier_rbf = svm.SVC(kernel='rbf')  # accuracy train: .446, test: .230 elapsed: train: 28 min, score, 8 min

    svm_classifier_poly_3 = svm.SVC(kernel='poly')  # accuracy: train: .458 test: .220 elapsed: train: 7 min, score: 6 min default degree is 3
    svm_classifier_poly_5 = svm.SVC(kernel='poly', degree=5)  # accuracy: train: .536 test: .18 valid: .37
    svm_classifier_poly_8 = svm.SVC(kernel='poly', degree=8)  # accuracy: train: .611 test: .24 valid: .34
    svm_classifier_poly_10 = svm.SVC(kernel='poly', degree=10)  # accuracy: train: .674 test: .21 valid: .33
    svm_classifier_poly_12 = svm.SVC(kernel='poly', degree=12)  # accuracy: train: .711 test: .22 valid: .32
    svm_classifier_poly_50 = svm.SVC(kernel='poly', degree=50)  # accuracy: train: .734 test: .21 valid: .230

    # 7-8 to train 5-6 to score 3
    svm_classifier_linear_1 = svm.SVC(kernel='linear')  # C = 1 accuracy: train: .999 test: .350  valid: .390
    svm_classifier_linear_100 = svm.SVC(kernel='linear', C=100)  # accuracy: train: .999 test: .350
    svm_classifier_linear_10000 = svm.SVC(kernel='linear', C=10000)  # accuracy: train: .999 test: .350
    svm_classifier_linear_01 = svm.SVC(kernel='linear', C=.01)  # accuracy: train: .999 test: .330 valid: .360 ----
    svm_classifier_linear_001 = svm.SVC(kernel='linear', C=.001)  # accuracy: train: .976 test: .320 valid: .370 ----
    svm_classifier_linear_0001 = svm.SVC(kernel='linear', C=.0001)  # accuracy: train: .924 test: .270 valid: .380
    svm_classifier_linear_00001 = svm.SVC(kernel='linear', C=.00001)  # accuracy: train: .463 test: .230
    svm_classifier_linear_000001 = svm.SVC(kernel='linear', C=.000001)  # accuracy: train: .426 test: .210

    # 10s range to train 2 min to score 3  train: . test: . valid: .
    kneighbors_1 = KNeighborsClassifier(n_neighbors=1) # train: .999 test: .14 valid: .21 --
    kneighbors_2 = KNeighborsClassifier(n_neighbors=2) # train: .719 test: .14 valid: .22 --
    kneighbors_3 = KNeighborsClassifier(n_neighbors=3) # train: .606 test: .210 valid: .26 --
    kneighbors_4 = KNeighborsClassifier(n_neighbors=4) # train: .536 test: .18 valid: .26 --
    kneighbors_5 = KNeighborsClassifier(n_neighbors=5) # train: .484 test: .23 valid: .24 --
    kneighbors_7 = KNeighborsClassifier(n_neighbors=7) # train: .434 test: .18 valid: .24 --
    kneighbors_50 = KNeighborsClassifier(n_neighbors=50) # train: .346 test: .24 valid: .23
    kneighbors_80 = KNeighborsClassifier(n_neighbors=80) # train: . test: . valid: .


    kneighbors_2_dist = KNeighborsClassifier(n_neighbors=2, weights='distance') # train: .999 test: .180 valid: .250
    kneighbors_7_dist = KNeighborsClassifier(n_neighbors=7, weights='distance') # train: .999 test: .230 valid: .260
    kneighbors_50_dist = KNeighborsClassifier(n_neighbors=50, weights='distance') # train: .999 test: .23 valid: .21



    model = kneighbors_7

    now = datetime.now()
    start_time = now.strftime("%H:%M:%S")
    print('Training start :)', start_time)
    for i, (x_trn, y_trn) in enumerate(trn_dataloader):
        model.fit(x_trn, y_trn)
        now = datetime.now()
        end_time = now.strftime("%H:%M:%S")
        print('Training end :)', end_time)
        print('Train Accuracy : %1.3f' % model.score(x_trn, y_trn))

    for i, (x_tst, y_tst) in enumerate(tst_dataloader):
        print('Test Accuracy : %1.3f' % model.score(x_tst, y_tst))
    for i, (x_val, y_val) in enumerate(val_dataloader):
        print('Valid Accuracy : %1.3f' % model.score(x_val, y_val))
    now = datetime.now()
    end_time = now.strftime("%H:%M:%S")
    print('Program end :)', end_time)

if __name__ == '__main__':
    main()

