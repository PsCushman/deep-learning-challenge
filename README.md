# Deep Learning and Neural Networks - *The Alpha Soup Charity Challenge*

![Screen Shot 2023-08-08 at 10 49 31 PM](https://github.com/PsCushman/deep-learning-challenge/assets/122395437/5c0ca181-2d3c-4492-b206-e975cef028ff)

## Data Preprocessing

*What variable(s) are the target(s) for your model?*

The 'IS_SUCCESSFUL' column from application_df is the target variable, this is what we are trying to predict. This shows if the money was used effectively.

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

Identification columns: The "EIN" and "NAME" columns are identification columns that typically provide unique identifiers for each organization. These columns usually have no direct impact on the target variable and can be dropped without affecting the model's accuracy.

## Compiling, Training, and Evaluating the Model

*How many neurons, layers, and activation functions did you select for your neural network model, and why?*

In my first neural network model, I used a two-layer architecture with a specific choice for the number of neurons, layers, and activation functions.
I chose two ReLu activtion to introduce non-linearity shapes and allow the model to learn complex relationships between the input features and the target variable.

Additionally, I used a single neuron in the output layer (units=1) with a sigmoid activation function (activation="sigmoid") to model the binary classification problem. The sigmoid activation function maps the output to a range between 0 and 1, representing the probability of the positive class.

In summary, the model architecture with the chosen number of neurons, layers, and activation functions aimed to strike a balance between complexity and simplicity, allowing the model to learn and generalize well on the given classification task.

*Were you able to achieve the target model performance?*

NO! I was only able to achieve 73%. The target model performance which was 75%.


*What steps did you take in your attempts to increase model performance?*

Well... 

## Increasing neurons and epochs


By increasing the number of neurons in a layer, the model becomes more expressive and can capture complex patterns in the data. This allows for better representation of the underlying relationships between the features and the target variable, potentially leading to higher accuracy.

Increasing the number of epochs gives the model more opportunities to learn from the data and adjust the weights. It allows the model to refine its predictions and find better parameter values, which can lead to improved accuracy. However, it's important to find a balance as increasing epochs excessively can lead to overfitting.

While it increased the speed of and I needed  I was only able to achieve 73%.

## Adding Layers:


Adding more layers can provide the model with additional capacity to capture and represent intricate relationships within the data. Each layer can learn different levels of abstraction, enabling the model to extract more meaningful features and potentially improving accuracy. Deep models with multiple layers have the ability to learn hierarchical representations of the data, which can be advantageous for complex problems.

I was only able to achieve 73%

## Adjusting Activation Functions:


Introducing a different activation function, such as tanh, can affect how the model interprets and transforms the inputs. Different activation functions have different properties and can capture different types of non-linearities. By using tanh, it introduce a different non-linearity that may better suit the problem at hand, potentially leading to increased accuracy.


Again, I was only able to achieve 73 or less%

## Checking Feature Importance

I tried a different algorithm, Random Forest, to get some insights into the importance of different features. Random Forest features importance based on how effectively each feature contributes to the overall prediction. This analysis can help identify the key predictor columns, allowing you to focus on the most informative features and potentially improve accuracy.
Identifying feature importance and selecting relevant attributes:

Analysing feature importance helps determine which attributes have the most significant impact on the output. By identifying and selecting the most important attributes, you can reduce the noise and complexity in the model. Focusing on the most relevant features can enhance the model's ability to capture meaningful patterns and improve accuracy.

## Caving to the Automated Optimiser

Automated optimisers, like hyperparameter tuners, systematically explore various combinations of hyperparameters, such as activation functions, number of layers, number of neurons, and epochs. 

This exploration was supposed to give me all the answers and save me by identifying the most optimal combination of hyperparameters for your specific problem, potentially leading to higher accuracy. It saves you from manually trying out different combinations and allows the optimiser to leverage its search algorithms to find the best configuration.

However, the best it could muster is 73%.

# Conclusion

The deep learning model that I have developed was unable to achieve accuracy higher than 73%. I makes me think it might not be possible with the methods I have at my disposile right now.

However, I might try:

Adding more data or increasing the size of my training dataset. 

Further cleaning a prepocessing of the data, maybe try a PCA model.

Addressing high bias and outliers.
