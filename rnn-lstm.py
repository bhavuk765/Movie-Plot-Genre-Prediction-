import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM
#from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import re



dataset = pd.read_csv('final_dataset.csv',quoting = 3, names = ['id','summary','name','genre'])
const=5000


gen_list=['Comedy','Action','Drama','Thriller','Horror','Romance Film']
my_var=[[] for _ in range(const)]

for i in range(const):
    for j in gen_list:
        my_var[i].append(int(j in d2['genre'][i]))


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
max_features = 5000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(dataset['summary'].values)
max_review_length=100

X2 = tokenizer.texts_to_sequences(dataset['summary'].values[:const])
X2 = pad_sequences(X2,max_review_length)




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, my_var, test_size = 0.2, random_state = 4)
y_train=np.asarray(y_train)
y_test=np.asarray(y_test)



embedding_vector_length=32
model = Sequential()
model.add(Embedding(max_features, embedding_vector_length, input_length=max_review_length))
model.add(SpatialDropout1D(0.5))
model.add(CuDNNLSTM(50))

model.add(Dense(6, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

filepath="lstm-weights-improvement.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history=model.fit(X_train, y_train, batch_size = 256,callbacks=callbacks_list, epochs = 55,verbose=1,validation_split=0.06)

y_pred_val = model.predict(X_test)
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

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
'''Output:

('Comedy', 0.6075)
('Action', 0.79625)
('Drama', 0.52325)
('Thriller', 0.754)
('Horror', 0.83475)
('Romance Film', 0.7245)
0.7167083333333334
'''
