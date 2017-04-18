# InitialMachineLearningTest
Initial test to do machine learning on BCI data using python.


## Singular Perceptron Neural Network (or Linear Discriminant Analysis ??)

This setup used a single perceptron, or one network of weights leading from inputs straight to outputs with no hidden layer.  This allows for relatively quick training but greatly limits the complexity of the solutions the network generates.

In order to get the data into a format that could be used in an LDA but still classify data over a measure of time, The information from the 59 electrodes at each time point was added to timepoints after the designated time as well to get a range of information from the EEG at the current timestep and future timesteps.  The reason we look at the future data in order to categorize whether the person is thinking of left foot or right hand, is because the data specifies when the person was told to start thinking of which class, rather than when the person was actually thinking of said class.  This way we can train the network to categorize what the person was told to do based on the reaction they have to said instruction in the proceeding seconds.

###First Dataset Results (ds1b [developed first on the second dataset.])
Accuracy: ~70-75% with 750 trials (~75-80% with 7500 trials)
Preprocessing Time: 1.669661s
Training Time: 64.005466s
Testing Time: 0.046811s

In order to maximize the result of this data a lot of messing with parameters was done and without cross validation.  This means that the data could very well be skewed to only working well on this data.  A good way to test the validity of these parameters for the training is to simply use the same parameters on another set of data and see how different the accuracy is.

###Second Dataset Results (ds1a [tested on the first dataset])
Accuracy: ~50%
Preprocessing Time: 1.881018
Training Time: 65.029943
Testing Time: 0.042454

This shows that the lack of cross validation caused a significant amount of the increased accuracy in the First Dataset's Results
