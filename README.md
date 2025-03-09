

### Report on the Neural Network Model for Alphabet Soup

#### Overview of the Analysis
The purpose of this analysis is to develop a deep learning model to predict the success of funding applications for Alphabet Soup, a non-profit organization. The goal is to create a neural network model that can accurately classify whether an application will be successful based on various features of the application data.

#### Results

##### Data Preprocessing

- **Target Variable:**
  - The target variable for the model is `IS_SUCCESSFUL`, which indicates whether the funding application was successful.

- **Feature Variables:**
  - The features for the model include all other variables in the dataset except for the target variable and those identified as unnecessary. These features include:
    - `AFFILIATION_CompanySponsored`
    - `AFFILIATION_Independent`
    - `APPLICATION_TYPE_T4`
    - `APPLICATION_TYPE_T19`
    - `ORGANIZATION_Association`
    - `APPLICATION_TYPE_Other`
    - `APPLICATION_TYPE_T6`
    - `ORGANIZATION_Trust`
    - `CLASSIFICATION_Other`
    - `CLASSIFICATION_C3000`
    - `APPLICATION_TYPE_T5`
    - `USE_CASE_ProductDev`
    - `APPLICATION_TYPE_T8`
    - `INCOME_AMT_10M-50M`
    - `APPLICATION_TYPE_T7`
    - `APPLICATION_TYPE_T3`
    - `INCOME_AMT_1M-5M`
    - `CLASSIFICATION_C1200`
    - `INCOME_AMT_100000-499999`
    - `INCOME_AMT_50M+`
    - `USE_CASE_Preservation`
    - `INCOME_AMT_5M-10M`
    - `ORGANIZATION_Co-operative`
    - `CLASSIFICATION_C2100`
    - `INCOME_AMT_1-9999`
    - `INCOME_AMT_10000-24999`
    - `CLASSIFICATION_C2000`
    - `CLASSIFICATION_C1000`
    - `USE_CASE_Heathcare`
    - `ASK_AMT`
    - `USE_CASE_CommunityServ`
    - `ORGANIZATION_Corporation`
    - `USE_CASE_Other`
    - `AFFILIATION_National`

- **Removed Variables:**
  - The following variables were removed from the input data because they are neither targets nor useful features based on permutation importance analysis:
    - `AFFILIATION_Other`
    - `STATUS`
    - `AFFILIATION_Family/Parent`
    - `AFFILIATION_Regional`
    - `SPECIAL_CONSIDERATIONS_Y`
    - `SPECIAL_CONSIDERATIONS_N`
    - `INCOME_AMT_25000-99999`
    - `INCOME_AMT_0`

##### Compiling, Training, and Evaluating the Model

- **Neurons, Layers, and Activation Functions:**
  - The huperparameter search from the keras_tuner returned an optimal number of units in the first densely-connected layer is 26, the optimal number of hidden layers is 4, and the optimal activation function is ReLU.
  - The neural network model consists of 4 hidden layers with the following configuration:
      - First Hidden Layer: 26 units, ReLU activation
      - Second Hidden Layer: 16 units, ReLU activation
     - Third Hidden Layer: 11 units, ReLU activation
      - Fourth Hidden Layer: 6 units, ReLU activation
   - The output layer has 1 unit with a Sigmoid activation function for binary classification.

- **Model Performance:**
  - The model was trained with early stopping to prevent overfitting. The early stopping callback monitored the validation loss and stopped training if it did not improve for 10 consecutive epochs.
  - The model's performance on the test data across multiple attempts is as follows:
    - Attempt 1: Loss: 0.5704, Accuracy: 0.7258
    - Attempt 2: Loss: 0.5683, Accuracy: 0.7247
    - Attempt 3: Loss: 0.5605, Accuracy: 0.7290
    - Attempt 4: Loss: 0.5865, Accuracy: 0.7072
    - Attempt 5: Loss: 0.5544, Accuracy: 0.7307
    - Attempt 6: Loss: 0.5521, Accuracy: 0.7321
    - Attempt 7: Loss: 0.5483, Accuracy: 0.7389

- **Steps to Increase Model Performance:**
  - Implemented early stopping to prevent overfitting.
  - Added dropout layers to reduce overfitting.
  - Performed hyperparameter tuning to find the optimal number of neurons, layers, and dropout rates.
  - Removed less important features based on permutation importance analysis.
  - Added an additional hidden layer to the neural network.
  - Used Keras Tuner to further optimize the model. The hyperparameter search was completed with the best validation accuracy of 0.7389.

##### Comparison with Other Models

- **Logistic Regression Accuracy:** 0.7247
- **Decision Tree Accuracy:** 0.7134
- **Random Forest Accuracy:** 0.7139
- **SVM Accuracy:** 0.7304
- **KNN Accuracy:** 0.7097
- **Gradient Boosting Accuracy:** 0.7276

#### Summary
The deep learning model achieved an accuracy of approximately 0.7321 in the 6th attempt, which is close to the target performance of 0.75. The model was optimized by tuning hyperparameters, adding dropout layers, removing less important features, and adding an additional hidden layer. 

