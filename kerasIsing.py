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
# load data set
dataset = numpy.loadtxt("IsingValues.txt", delimiter=",")
# note shape of indata
size_y = dataset.shape[0]
size_x = dataset.shape[1]
border = size_x - 2
# split into input (X) and output (Y) variables
X = dataset[:,0:border]
Y = dataset[:,border:size_x]

#########
# MODEL #
#########
# Define model
model = Sequential()
model.add(Dense(22, input_dim=100, activation='relu'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid')) # Dense(1, ....)
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

###################
# BACKPROPAGATION #
###################
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

##############
# EVALUATION #
##############
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

###########
# PREDICT #
###########
# calculate predictions
predictions = model.predict(X)
# print predictions
print(predictions)
