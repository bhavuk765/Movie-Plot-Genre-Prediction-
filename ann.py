
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle

dataset = pd.read_csv('final_dataset.csv',,quoting = 3, names = ['id','summary','name','genre'])
const=2000


gen_list=['Comedy','Action','Drama','Thriller','Horror','Romance Film']
d={}
dataset['genre'].value_counts
for i in gen_list:
    d[i]=sum([1 if i in genre else 0 for genre in dataset['genre']])
with open('count.pkl','wb') as file:
    pickle.dump(corpus,file)
with open('corpus.pkl','rb') as file:
    corpus=pickle.load(file)
my_var=[[] for _ in range(const)]
gen_list=['Comedy','Action','Drama','Thriller','Horror','Romance Film']

for i in range(const):
    for j in gen_list:
        my_var[i].append(int(j in dataset['genre'][i]))


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
cv2=cv.fit(corpus[:const])
X = cv.fit_transform(corpus[:const]).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, my_var, test_size = 0.20, random_state = 1)
y_train=np.asarray(y_train)
y_test=np.asarray(y_test)

classifier = Sequential()
classifier.add(Dense(X_train.shape[1]//3, activation = 'relu', input_shape = (X_train.shape[1],)))
classifier.add(Dropout(0.2))
classifier.add(Dense(X_train.shape[1]//5, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(6, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

filepath="weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history=classifier.fit(X_train, y_train, batch_size = 256,callbacks=callbacks_list, epochs = 50,verbose=1,validation_split=0.1)
y_pred_val = classifier.predict(X_test)
y_pred = (y_pred_val > 0.5)

acc=[]
for i in range(6):
    y_test_single=[]
    y_pred_single=[]
    for j in y_test:
        y_test_single.append(j[i])
    for j in y_pred:
        y_pred_single.append(j[i])
    cm = confusion_matrix(y_test_single, y_pred_single)
    print(cm)
    acc.append(accuracy_score(y_pred_single,y_test_single))
print(*(list(zip(gen_list,acc))),sum(acc)/6,sep='\n')

'''Output:
    Epoch 1/20
16000/16000 [==============================] - 10s 618us/step - loss: 0.5721 - acc: 0.7470

Epoch 00011: acc did not improve from 0.99976
Epoch 12/20
16000/16000 [==============================] - 6s 366us/step - loss: 0.0054 - acc: 0.9994

('Comedy', 0.62975)
('Action', 0.77025)
('Drama', 0.51625)
('Thriller', 0.75525)
('Horror', 0.85925)
('Romance Film', 0.76925)
0.7067'''

import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
