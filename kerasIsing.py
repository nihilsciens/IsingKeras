###########
# IMPORTS #
###########
from keras.models import Sequential
from keras.layers import Dense
import numpy

################
# PREPARATIONS #
################
# fix random seed for reproducibility
numpy.random.seed(7)
# load data set and test set
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

#########
# MODEL #
#########
# Define model
model = Sequential()
model.add(Dense(border, input_dim=border, activation='relu'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

###################
# BACKPROPAGATION #
###################
# Fit the model
model.fit(X, Y, epochs=1500, batch_size=10)

##############
# EVALUATION #
##############
scores = model.evaluate(Xt, Yt)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

###########
# PREDICT #
###########
# calculate predictions
predictions = model.predict(Xt)
# print predictions
print(predictions)
