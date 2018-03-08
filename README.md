# pregnancy-outcome 
Predict the outcome of childbirth (Live Birth/Still Birth/Miscarriage/Abortion/Infant Death).

The data set contains data from the Ante Natal Care checkups and of the socio-economic status of the mother-to-be.
There is huge class imbalance in the data set (expected, as the number of Live Births are always exponentially greater than the negative outcomes), and to add to the misery, much of the data is missing for most features.
This has been partially solved by assigning class weights in the call for the Random Forest classifier.

This is just a baseline Random Forest Classifier score, as I couldn't get time to experiment with other classifiers or neural networks.

I'm attaching only one plot here that shows the interaction between features and the outcomes, and some other screenshots that I think would make things easier to understand. And the code is a .py file that has been copied into NotePad++, so for best aesthetics, NotePad++ would be recommended as the viewer.

The first image (countplot) shows the effect of Age on the final outcome.
The second image shows some of the risky symptoms and their corresponding counts in the entire data set in descending order.
The third image shows the transformation of symptoms into individual features (line 204 in code).

