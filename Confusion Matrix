# I mapped the outcome variables as follows to get a better understanding of which all target variables got misclassified

y = y.map({'Still Birth' : 0, 'Abortion' : 1, 'Live Birth': 2, 'Infant Death' : 3, 'Miscarriage' : 4})

#Now lets take a look at the no. of instances of each class present in the entire dataset:
#No. of instances of Still Birth:
(y == 0).sum() # output : 56

#No. of instances of Abortion:
(y == 1).sum() # output : 58

#No. of instances of Live Birth : 
(y == 2).sum() # Output : 1797 !

#No. of instances of Infant Death:
(y == 3).sum() # Output : 8 !
#No. of instances of Miscarriage:
(y == 4).sum() # Output : 11 !

# As can be seen, an overwhelming majority of the instances are Live Birth, and only 8 and 11 instances of Infant Death and
# Miscarriage are present.

#Because of this imbalance in classes, I manually set the weights for each class in the classifier (the dictionary in the code below)
rf = RandomForestClassifier(n_estimators = 500, max_features = 35, max_depth = 12, random_state = 0,  class_weight = \
        ({0:1000, 1:1000, 2:0.01, 3:1000, 4:1000}), ).fit(X_train, y_train)
# On a side note, this modification changes the cross_val_score to 87.65

#Now for the confusion matrix on the train set:
# The code for the confusion matrix:
confusion = confusion_matrix(y_train, rf.predict(X_train))
confusion

# The output of the above code is as follows:

array([[  45,    0,    0,    0,    0],
       [   0,   40,    0,    0,    0],
       [  21,   32, 1391,    0,    0],
       [   0,    0,    0,    7,    0],
       [   0,    0,    0,    0,    8]], dtype=int64)
	   
# As is evident, even during training, and even after assigning weights to individual classes, there is a high tendency to 
# misclassify instances of Still Birth and Abortion as Live Birth. This hints at possible similarities in the values of the 
# attributes between the three classes.

# test set confusion matrix:

array([[  0,   1,  10,   0,   0],
       [  1,   6,  11,   0,   0],
       [ 12,   5, 336,   0,   0],
       [  0,   0,   1,   0,   0],
       [  0,   0,   3,   0,   0]], dtype=int64)
	   
# Here, none of the instances have been classified as Miscarriage or Infant Death!
# Upon inspection:

(y_test == 3).sum() # Output : 1
(y_test == 4).sum() # Output : 3

#So there is only one instance of Infant Death and 3 instances of Miscarriage in the actual test set.

# Code for evaluation metrics:
FP = confusion.sum(axis=0) - np.diag(confusion)  
FN = confusion.sum(axis=1) - np.diag(confusion)
TP = np.diag(confusion)
TN = confusion.sum() - (FP + FN + TP)

# Sensitivity
TPR = TP/(TP+FN)
print (TPR)
# Out : [ 0.          0.33333333  0.95184136  0.          0.        ] 

# Specificity
TNR = TN/(TN+FP) 
print (TNR)
# Out : [ 0.96533333  0.98369565  0.24242424  1.          1.        ]

# Positive predictive value
PPV = TP/(TP+FP)
print (PPV)
# Out : [ 0.          0.5         0.93074792         nan         nan] 
# Here, there are no TPs or FPs in classes 3 and 4, and hence division by zero leads to NaN


# Negative predictive value
NPV = TN/(TN+FN)
print (NPV)
# Out : [ 0.97050938  0.96791444  0.32        0.99740933  0.99222798]

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print (ACC)
# Out : [ 0.93782383  0.95336788  0.89119171  0.99740933  0.99222798]
