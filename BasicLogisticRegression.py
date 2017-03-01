import numpy as np
import theano
import theano.tensor as T
import pandas as pd
from collections import Counter
rng = np.random

#Retrieve dataset
dataname = "BCICIV_calib_ds1a_cnt.txt"
Srate = 100
markername = "BCICIV_calib_ds1a_mrk.txt"
eegnames = [str(i) for i in range(59)]
#print(eegnames)

txtdata = []
df = pd.read_csv(dataname, sep='\t', header = None, names=eegnames)
mf = pd.read_csv(markername, sep='\t', header = None, names = ['timestamp', 'class'])

#print(df.loc[3:7])
#print(mf.loc[:,'timestamp'])
reg = .01 #regularization parameter
learnrate = .1 #learning rate
training_steps = 200 #training steps
num_classes = 3 #one for left, one for foot, and one for neither

data = np.array(df)
markers = np.array(mf)

N, feats = data.shape
crapMarks = [1 for i in range(N)]
for mark in markers:
    crapMarks[int(mark[0])] = int(mark[1]+1)


#print(crapMarks[2070:2100])

#Declare Theano Symbolics
x = T.dmatrix('x')
y = T.ivector('y')

#Initialize weight vector randomly (use shared so they stay the same after the generation)
w = theano.shared(rng.randn(feats,num_classes), name="w")

#Initialize bias vector b
b = theano.shared(0.01, name="b")

#construct formulas with Symbolics
#p_1 = 1/(1+T.exp(-T.dot(x,w)-b)) #Calculate accuracy prediction
#prediction = p_1 > .5
sigma = T.nnet.softmax(T.dot(x,w)+b) #SoftMax of the classes
prediction = T.argmax(sigma, axis=1) #The most probable class.
#xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)
xent = -T.mean(T.log(sigma)[T.arange(y.shape[0]),y])
cost = xent.mean() + reg * (w**2).sum() #cost function with regularization
gw, gb = T.grad(cost, [w,b]) #Gradient computation

#Instantiate formulas with theano.function
train = theano.function(inputs = [x,y],
                        outputs = [cost],
                        updates = ((w, w- learnrate*gw), (b, b- learnrate*gb)))
predict = theano.function(inputs = [x], outputs = prediction)

#Train
for i in range(training_steps):
    print('trainstep' + str(i))
    train(data,crapMarks)

finalpred = predict(data)
answers = np.array(crapMarks)
finalpred2 = finalpred*5
PosNeg = finalpred2 - answers
#c = Counter(finalpred2)
#d = Counter(answers)
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
#print("False Positive Rate:")
#print(FP/(FP+TN))
#print("False Neutral Rate:")
#print()
#print("False Negative Rate:")
#print()
