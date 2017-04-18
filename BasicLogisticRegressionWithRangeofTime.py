import numpy as np
import theano
import theano.tensor as T
import pandas as pd
import random
from collections import Counter
from matplotlib import pyplot as plt
import sys
import time

tpre = time.time()
rng = np.random

#Retrieve dataset
dataname = "BCICIV_calib_ds1a_cnt.txt"
Srate = 100
start_ofst = int(0*Srate)
end_ofst = int(5*Srate)
resolution = int(Srate*.2) #Lower number is higher resolution
frontstrip = int(Srate*14) #number is seconds to cut from the beginning
endstrip = int(Srate*5) #number is seconds to cut from end
file_reso = int(Srate*.4) #naumber is how ofter (in seconds) data is taken (this lowers samplerate or resolution)
markerSize = int(Srate*.5)
train_set_size = .6 #percent of data to use for training vs percent to use for testing
#Start to get memmory errors if the two resolution numbers multiply to be less than .005
#(This is because the min they can be is .01, .005 choose the same data point for each part of the for loop these numbers work in, and thus create an infinite loop.)

markername = "BCICIV_calib_ds1a_mrk.txt"
eegnames = [str(i) for i in range(59)]
txtdata = []
df = pd.read_csv(dataname, sep='\t', header = None, names=eegnames)
mf = pd.read_csv(markername, sep='\t', header = None, names = ['timestamp', 'class'])

reg = 10000 #regularization parameter
acc = []
deltashrink = .875
deltagrow = 1.001
training_steps = 750 #training steps
num_classes = 3 #one for left, one for foot, and one for neither

data = np.array(df)
markers = np.array(mf)
oldN, oldfeats = data.shape

#### Setting time points in the range of "markersize" after the marker timepoint to the marker class ####
Marks = [1 for i in range(oldN)]
for mark in markers:
    for i in range(markerSize):
        Marks[int(mark[0]+i)] = int(mark[1]+1)

Marks = Marks[frontstrip:len(Marks)-endstrip]

#### DownSampling Entire Dataset ####
interdata = [] # data before shuffling and splitting into datasets
intermarks = [] # marks corresponding with above data.
timepoint = 0
while timepoint < len(Marks):
    if timepoint % file_reso == 0 or Marks[timepoint] != 1:
        interdata.append(np.reshape(data[timepoint+start_ofst:timepoint+end_ofst:resolution],-1)) # Downsampling the feature vector
        intermarks.append(Marks[timepoint])
    timepoint+=1
interN = len(interdata)
randindex = [i for i in range(interN)] #create index list
rng.shuffle(randindex) #randomize list
finaldata = []
finalmarks = []
testdata = []
testmarks = []
train_actual_size = interN*train_set_size
for i in range(interN): #pick "train_set_size" of the data and markers and put it into the training set
    if i < train_actual_size:
        finaldata.append(interdata[randindex[i]])
        finalmarks.append(intermarks[randindex[i]])
    else:
        testdata.append(interdata[randindex[i]])
        testmarks.append(intermarks[randindex[i]])
finaldata = np.array(finaldata)
finalmarks = np.array(finalmarks, dtype = 'int32')
testdata = np.array(testdata)
testmarks = np.array(testmarks, dtype = 'int32')
print('trainsizes')
print(finaldata.shape)
print(finalmarks.shape)
print('\ntestsizes')
print(testdata.shape)
print(testmarks.shape)


print(finalmarks)
N, feats = finaldata.shape

initw = rng.randn(feats,num_classes)
initb = rng.randn(num_classes)

ttrain = time.time()
learnrate = theano.shared(.0001, name="learnrate") #learning rate


#Declare Theano Symbolics
x = T.dmatrix('x')
y = T.ivector('y')
z = T.dmatrix('z')

#Initialize weight vector randomly (use shared so they stay the same after the generation)
w = theano.shared(initw, name="w")

#Initialize bias vector b
b = theano.shared(initb, name="b")


#construct formulas with Symbolics
sigma = T.nnet.softmax(T.dot(x,w)+b) #SoftMax of the classes
prediction = T.argmax(sigma, axis=1) #The most probable class.

### Z Fuction ###
zsigma = T.nnet.softmax(T.dot(z,w)+b) #SoftMax of the classes
zprediction = T.argmax(zsigma, axis=1)

xent = -T.mean(T.log(sigma)[T.arange(y.shape[0]),y]) #Cross Entropy Function For Multi Class Optimization
cost = xent.mean() + reg * (w**2).sum() #cost function with regularization
gw, gb = T.grad(cost, [w,b]) #Gradient computation

#Instantiate formulas with theano.function
train = theano.function(inputs = [x,y],
                        outputs = [prediction, cost],
                        updates = ((w, w - learnrate*gw), (b, b - learnrate*gb), (learnrate, learnrate*deltagrow)))
predict = theano.function(inputs = [z], outputs = zprediction)

#Train
old_err = sys.float_info.max
for i in range(training_steps):
    print('\ntrainstep' + str(i))
    pred, err = train(finaldata,finalmarks)
    if err > old_err*1.0001:
        learnrate.set_value(learnrate.get_value()*deltashrink)
    old_err = err
    print(err)
    print(learnrate.get_value())

ttest = time.time()
finalpred = predict(testdata)
answers = np.array(testmarks)
print(finalpred)
print(answers)
c = Counter(finalpred)
d = Counter(answers)
print(c)
print(d)
finalpred2 = finalpred*5
PosNeg = finalpred2 - answers

PosNegCount = {8:['PP', 0], -2:['NP', 0], 0:['NN', 0], 10:['PN', 0], 9:['PO', 0], -1:['NO',0], 3:['OP', 0], 4:['OO', 0], 5:['ON', 0]}

for j in PosNeg:
    try:
        PosNegCount.get(j)[1] += 1
    except:
        print(j)
PP = PosNegCount.get(8)[1]
OO = PosNegCount.get(4)[1]
NN = PosNegCount.get(0)[1]
PN = PosNegCount.get(10)[1]
PO = PosNegCount.get(9)[1]
ON = PosNegCount.get(5)[1]
OP = PosNegCount.get(3)[1]
NO = PosNegCount.get(-1)[1]
NP = PosNegCount.get(-2)[1]

Npred = len(answers)

print(PosNegCount)
print("Final model:")
print("True Positive Rate:")
print(PP/(PP+OP+NP))
print("True Neutral Rate:")
print(OO/(PO+OO+NO))
print("True Negative Rate:")
print(NN/(PN+ON+NN))
accuracyList = [1 if finalpred[i] == answers[i] else 0 for i in range(Npred)]
right = sum(accuracyList)
acc.append(right/Npred)
print(acc)


tfinish = time.time()
print("Preprocessing Time: %f" %(ttrain-tpre))
print("Training Time: %f" %(ttest-ttrain))
print("Testing Time: %f" %(tfinish-ttest))


## TODO
"""
Add cross-validation SciKit or manual (Try SciKit first)
Understand why values are only ever perfect or completely off
Determine whether it is correctly training and testing (Appears to be somewhat correct?, maybe fake data is just very good fake data, try with non fake data)
Add Backtracking Search Algorithm

########### Long Stretch: Network Versions ##############


#### Done:

Networks:
LDA

Functions:
Sigmoid

Descent Algorithms:
Gradient Descent (Backpropagation Algorithm)

Parameterization:
Flexible Learn Rate with Growth and Shrink

Other:




#### TBD (To Be Done):

Networks:
MLP
Convolutional**
Fuzzy?
RNN/LSTM*
RBF or Radial Basis Function*

Functions:
RELU?
SVM**

Descent Algorithms:
LBFGS

Parameterization:
Cross Validation
Backtracking Search Algorithm

Other:
CSP*

"""
