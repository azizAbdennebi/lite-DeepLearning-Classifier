from tensorflow import keras
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from keras.datasets.cifar import load_batch
from keras import backend
import numpy as np
import os
import pickle
# opening datasets
path = "datasets"
num_train_samples = 50000
x_train = np.empty((num_train_samples, 3, 32, 32), dtype="uint8")
y_train = np.empty((num_train_samples,), dtype="uint8")
for i in range(1, 6):
    fpath = os.path.join(path, "data_batch_" + str(i))
    (x_train[(i - 1) * 10000 : i * 10000, :, :, :],y_train[(i - 1) * 10000 : i * 10000],) = load_batch(fpath)
fpath = os.path.join(path, "test_batch")
x_test, y_test = load_batch(fpath)
y_train = np.reshape(y_train, (len(y_train), 1))
y_test = np.reshape(y_test, (len(y_test), 1))
if backend.image_data_format() == "channels_last":
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)
x_test = x_test.astype(x_train.dtype)
y_test = y_test.astype(y_train.dtype)
print(x_train.shape,x_test.shape)
#normalization
x_train=x_train/255.0
x_test=x_test/255.0
#sklearn expects i/p to be 2d array model.fit(x_train, y_train)=> reshape to 2d array
nsamples, nx, ny, nrgb= x_train.shape
x_train2=x_train.reshape((nsamples, nx*ny*nrgb))
#so, eventually,model.predict() should also be a 2d array
nsamples, nx, ny, nrgb= x_test.shape
x_test2=x_test.reshape((nsamples, nx*ny*nrgb))
print(x_train2.shape,x_test2.shape)
print(y_train.shape,y_test.shape)
#KNN algorithm
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train2, y_train)
y_pred_knn=knn.predict(x_test2)
print(y_pred_knn)
score=accuracy_score(y_pred_knn,y_test)
print(classification_report(y_pred_knn,y_test))
# save the model to disk
filename = 'knnClassification10_model.sav'
pickle.dump(knn, open(filename, 'wb'))
print("done")
