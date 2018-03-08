# pregnancy-outcome 
Predict the outcome of childbirth (Live Birth/Still Birth/Miscarriage/Abortion/Infant Death).

Data is spread across 3 Excel files, which are merged on the ID of the mother-to-be.
The data sets contain data from the Ante Natal Care checkups, and of the socio-economic status of the mother-to-be.

There is huge class imbalance in the data set (expected, as the number of Live Births are always exponentially greater than the negative outcomes), and to add to the misery, much of the data is missing for most features.
This has been partially solved by assigning class weights in the call for the Random Forest classifier.

This is just a baseline Random Forest Classifier score, as I couldn't get time to experiment with other classifiers or neural networks.

I'm attaching only one plot here that shows the interaction between features and the outcomes, and some other screenshots that I think would make things easier to understand.
Age Effect.png shows the effect of Age on the final outcome.
Top Risks.png shows some of the risky symptoms and their corresponding counts in the entire data set in descending order.
Risk as Features.png shows the transformation of symptoms into individual features (line 204 in code).
Feature Importances.png shows the top 20 features, which the classifier thinks is the most influential. The code for generating the plot is in 'importances.py'

# Inferences and conclusions:
 Given the highly imbalanced nature of classes in the data set, it would be asking too much of any classifier to accurately predict each instance, that too when there seems to be similarities in attribute values between classes.
 A possible way to compensate for the class imbalance would be to sample out a sub set of instances where the no. of instances of the negative class are in greater number.
 But this suffers from the shortcoming of insufficient data. As shown earlier, out of 1930 total instances, ~1800 are of the 'Live Birth' class. Even taken in its entirety during training (without train-test split), the data would only have either 56,58, 8, or 11 instances of each class. Training on such a small sample wouldn't be incredibly effective, and even if it was, we run the risk of overfitting to the negative class.

The outcome can be made to be a binary distribution( 'Live Birth' or Not), to get a slightly better performance, but even then, the total amount of negative classes would only be 133, as oppposed to 1797 postive.
