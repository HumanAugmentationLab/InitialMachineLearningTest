import numpy as np
import theano
import theano.tensor as T
import pandas as pd
rng = np.random

#Retrieve dataset
dataname = "BCICIV_calib_ds1a_cnt.txt"
Srate = 100
start_ofst = int(.5*Srate)
end_ofst = int(3*Srate)
resolution = int(Srate*.1) #Lower number is higher resolution
frontstrip = int(Srate*14) #number is seconds to cut from the beginning
endstrip = int(Srate*4) #number is seconds to cut from end
file_reso = int(Srate*.05) #naumber is how ofter (in seconds) data is taken (this lowers samplerate or resolution)
markerSize = int(Srate*.5)
#Start to get memmory errors if the two resolution numbers multiply to be less than .005

markername = "BCICIV_calib_ds1a_mrk.txt"
eegnames = [str(i) for i in range(59)]
#print(eegnames)

txtdata = []
df = pd.read_csv(dataname, sep='\t', header = None, names=eegnames)
mf = pd.read_csv(markername, sep='\t', header = None, names = ['timestamp', 'class'])

#print(df.loc[3:7])
#print(mf.loc[:,'timestamp'])
reg = .0001 #regularization parameter
learnrate = .001 #learning rate
training_steps = 50 #training steps
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

#### DownSampling Entire Data ####
finaldata = []
finalmarks = []
i = 0
for timepoint in Marks:
    if i % file_reso == 0 or Marks[i] != 1:
        finaldata.append(np.reshape(data[timepoint+start_ofst:timepoint+end_ofst:resolution],-1)) # Downsampling the feature vector
        finalmarks.append(Marks[i])
    i+=1
finaldata = np.array(finaldata)
finalmarks = np.array(finalmarks, dtype = 'int32')
print(finaldata.shape)
print(finalmarks.shape)
#print(len(timedata))

N, feats = finaldata.shape

#Declare Theano Symbolics
x = T.dmatrix('x')
y = T.ivector('y')

#Initialize weight vector randomly (use shared so they stay the same after the generation)
w = theano.shared(rng.randn(feats,num_classes), name="w")

#Initialize bias vector b
b = theano.shared(.01, name="b")

#construct formulas with Symbolics
#p_1 = 1/(1+T.exp(-T.dot(x,w)-b)) #Calculate accuracy prediction
#prediction = p_1 > .5
sigma = T.nnet.softmax(T.dot(x,w)+b) #SoftMax of the classes
prediction = T.argmax(sigma, axis=1) #The most probable class.
#xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)
xent = -T.mean(T.log(sigma)[T.arange(y.shape[0]),y]) #Cross Entropy Function
cost = xent.mean() + reg * (w**2).sum() #cost function with regularization
gw, gb = T.grad(cost, [w,b]) #Gradient computation

#Instantiate formulas with theano.function
train = theano.function(inputs = [x,y],
                        outputs = [prediction, cost],
                        updates = ((w, w - learnrate*gw), (b, b - learnrate*gb)))
predict = theano.function(inputs = [x], outputs = prediction)

#Train
for i in range(training_steps):
    print('trainstep' + str(i))
    pred, err = train(finaldata,finalmarks)
    print(err)

finalpred = predict(finaldata)
answers = np.array(finalmarks)
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

print(PosNegCount)
#Print Results
print("Final model:")
print(w.get_value())
print(b.get_value())
print("True Positive Rate:")
print(PP/(PP+OP+NP))
print("True Neutral Rate:")
print(OO/(PO+OO+NO))
print("True Negative Rate:")
print(NN/(PN+ON+NN))
accuracyList = [1 if finalpred[i] == answers[i] else 0 for i in range(N)]
#for i in range(N):
#    accuracyList.append(finalpred[0] == answers[0])
right = sum(accuracyList)
print(N-right)
print(right/N)
print(N)
#print("False Positive Rate:")
#print(FP/(FP+TN))
#print("False Neutral Rate:")
#print()
#print("False Negative Rate:")
#print()
