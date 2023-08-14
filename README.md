# Deep Learning and Neural Networks - *The Alpha Soup Charity Challenge*

![Screen Shot 2023-08-08 at 10 49 31 PM](https://github.com/PsCushman/deep-learning-challenge/assets/122395437/5c0ca181-2d3c-4492-b206-e975cef028ff)

## Data Preprocessing

*What variable(s) are the target(s) for your model?*

The 'IS_SUCCESSFUL' column from application_df is the target variable. In other words, we are trying to predict wether the money was used effectively.
 
*What variable(s) are the features for your model?*

The feature variables we used are:

AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested

*What variable(s) should be removed from the input data because they are neither targets nor features?*

The "EIN" and "NAME" columns are identification columns that provide unique identifiers for each organization. However, in this case, they have no impact on the target variable and can be dropped without affecting the model's accuracy.

## Compiling, Training, and Evaluating the Model

*How many neurons, layers, and activation functions did you select for your neural network model, and why?*

In my first neural network model, I used a two-layer architecture with a specific choice for the number of neurons, layers, and activation functions.
I chose two ReLu activtion to introduce non-linearity shapes and allow the model to learn complex relationships between the input features and the target variable.

Additionally, I used a single neuron in the output layer (units=1) with a sigmoid activation function (activation="sigmoid") to model the binary classification problem. The sigmoid activation function maps the output to a range between 0 and 1, representing the probability of the positive class.

In summary, the model architecture with the chosen number of neurons, layers, and activation functions attempt to balance complexity and simplicity, allowing the model to learn and generalize well on the given classification task.

*Were you able to achieve the target model performance?*

NO! I was only able to achieve 73%. The target model performance which was 75%.


*What steps did you take in your attempts to increase model performance?*

Well... 

## Increasing neurons and epochs

![Screen Shot 2023-08-09 at 9 00 46 AM](https://github.com/PsCushman/deep-learning-challenge/assets/122395437/05b2d022-488f-438e-a427-ad96a577bb08)


By increasing the number of neurons in a layer, the model becomes more expressive and can capture complex patterns in the data. This allows for better representation of the underlying relationships between the features and the target variable, potentially leading to higher accuracy.

Increasing the number of epochs gives the model more opportunities to learn from the data and adjust the weights. It allows the model to refine its predictions and find better parameter values, which can lead to improved accuracy. However, it's important to find a balance as increasing epochs excessively can lead to overfitting.

While increasing the number of neurons increased the speed at which the model got to around 73% accuracy, it did not improve the model. As for increasing the epochs, I increased them all the way to 500 without any success improving the model's performance.

In fact, I added neurons to mutiple models, but any more than the amount I chose did not increase the model's perforamce on the test data. While the accuracy did inprove on the training model, it tended to overfit the data if I increased them much higher. 

I chose 50 epoch to avoid anymore overfitting and because after a certain amount of epoch, it wasn't doing much to improve the model.

So while, it increased the speed, increasing nodes and epochs did not get me over the 75% accuracy threshold.

## Adding Layers


Adding more layers can provide the model with additional capacity to capture and represent intricate relationships within the data. Each layer can learn different levels of abstraction, enabling the model to extract more meaningful features and potentially improving accuracy. Deep models with multiple layers have the ability to learn hierarchical representations of the data, which can be advantageous for complex problems.

Similarly, I had the same results with 2,3, and 4 hidden layers. Increasing the layers over 2 hidden layers did not increase the model's performance by much at all, still topping off at around 73%. While there were times when the acruraccy of the training data improved, it resulted in an orverfitted model that did not produce great results for the test data.

## Adjusting Activation Functions

Introducing a different activation function, such as tanh, can affect how the model interprets and transforms the inputs. Different activation functions have different properties and can capture different types of non-linearities. I used tanh, relu, and sigmoid in any number of combinations, but with little success. In the end, begining with a ReLu activation and additonal ReLu layer, finishing with an output layer sigmoid, achieved my best results. 

![Screen Shot 2023-08-13 at 8 59 27 PM](https://github.com/PsCushman/deep-learning-challenge/assets/122395437/0796f31d-6ca4-48b1-bd88-b2710091ce94)


With all that, I was still only able to achieve 73 or less%

## Checking Feature Importance

I tried a different algorithm, Random Forest, to get some insights into the importance of different features. Random Forest features importance based on how effectively each feature contributes to the overall prediction. This analysis can help identify the key predictor columns, allowing you to focus on the most informative features and potentially improve accuracy.


![Screen Shot 2023-08-09 at 9 04 03 PM](https://github.com/PsCushman/deep-learning-challenge/assets/122395437/085537c9-5ee4-4744-bebc-95df6c9b6628)


Analysing feature importance helps determine which attributes have the most significant impact on the output. By identifying and selecting the most important attributes, you can reduce the noise and complexity in the model. Focusing on the most relevant features can enhance the model's ability to capture meaningful patterns and improve accuracy.

![Screen Shot 2023-08-09 at 8 53 24 AM](https://github.com/PsCushman/deep-learning-challenge/assets/122395437/3cd8b984-7757-4dfa-af03-5f38419f4f64)

I tried using the top 3, top 10, and top 25 most important features, but I never got a model that was better than 73%. In the end, using all the features provided the best model for getting the best results. However, I did get a score that was only 2 percentage points off the the best model by using only 10 features.

*10 most Important Features*
![Screen Shot 2023-08-13 at 9 04 59 PM](https://github.com/PsCushman/deep-learning-challenge/assets/122395437/e92f7e0c-01a9-40d5-980e-06f76fe9de4c)


## Utilizing PCA

Since feature importance did not provide any relief, I turned to a PCA model. Similar results. Using PCA with 10 elements with the same combinations of activations, neurons and epochs with which I had greatest sucess, did not improve my results.

![Screen Shot 2023-08-09 at 8 46 39 AM](https://github.com/PsCushman/deep-learning-challenge/assets/122395437/ce0adf64-4b38-494a-a6d8-b6ecde624de4)

The PCA model only achieved 72% accuracy and so another door was closed.

![Screen Shot 2023-08-13 at 8 58 35 PM](https://github.com/PsCushman/deep-learning-challenge/assets/122395437/e884966b-6341-48ef-a391-c5073d550b51)

## Turning to the Automated Optimiser

![Screen Shot 2023-08-09 at 1 44 10 PM](https://github.com/PsCushman/deep-learning-challenge/assets/122395437/d53db0ae-af4c-438e-b47d-3228227fa38c)

The Keras Automated Optimiser, like other hyperparameter tuners, systematically explores various combinations of hyperparameters, such as activation functions, number of layers, number of neurons, and epochs. This exploration was supposed to give me all the answers and save me by identifying the most optimal combination of hyperparameters. Instead of manually trying out anymore  combinations, I turned to the the optimiser to leverage its search algorithms to find the best configuration.



However, what I found was that, fround the data we have, no model using the hyperparamters I set could cross the 75% acuracy threhold. And so, I was stuck at 73%.

![Screen Shot 2023-08-09 at 9 19 32 PM](https://github.com/PsCushman/deep-learning-challenge/assets/122395437/9fbb6300-19f4-4a2a-97ee-8f4521ddf471)


# Conclusion

The deep learning model that I have developed was unable to achieve accuracy higher than 73%. I makes me think it might not be possible with the methods I have at my disposile right now.
