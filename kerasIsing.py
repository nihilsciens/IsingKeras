###########
# IMPORTS #
###########
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
import math
import numpy

################
# PREPARATIONS #
################
# fix random seed for reproducibility
numpy.random.seed(7)
# load data set and test set (they have to have same dimensions)
dataset = numpy.loadtxt("IsingValues.txt", delimiter=",")
testset = numpy.loadtxt("IsingTest.txt", delimiter=",")
# note shape of indata
size_y = dataset.shape[0]
size_x = dataset.shape[1]
border = size_x - 2
# split into input (X) and output (Y) variables
X = dataset[:,0:border]
Y = dataset[:,border:size_x]
# do the same for test data
Xt = testset[:,0:border]
Yt = testset[:,border:size_x]
# convert input to matrices
length = int(math.sqrt(size_x))
X_i = numpy.empty([size_y, length, length])
X_it = numpy.empty([size_y, length, length])
for i in range(size_y):
	for j in range(length):
		X_it[i,j,:] = Xt[i,j*length:(j+1)*length]
		X_i[i,j,:] = X[i,j*length:(j+1)*length]

#########
# MODEL #
#########
# Define model
model = Sequential()
model.add(Dense(length, input_shape=[length, length], activation='relu'))
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

###################
# BACKPROPAGATION #
###################
# Fit the model
model.fit(X_i, Y, epochs=500, batch_size=10)

##############
# EVALUATION #
##############
# evaluate test data
scores = model.evaluate(X_it, Yt)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

###########
# PREDICT #
###########
# calculate predictions
predictions = model.predict(X_it)
# check accuracy
diff = numpy.rint(predictions) - Yt
res = numpy.empty([size_y, 1], dtype='str')
for i in range(size_y):
    res[i] = 'i'
    if diff[i,0] == 0 and diff[i,1] == 0:
        res[i] = 'c'
# print predictions
print(numpy.concatenate([predictions, Yt, res],axis=1))
