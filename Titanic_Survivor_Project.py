import pandas as pd
from sklearn import model_selection
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import Counter


# This function mutates, and also returns, the targetDF DataFrame.
# Mutations are based on values in the sourceDF DataFrame.
# You'll need to write more code in this function, to complete it.
def preprocess(targetDF, sourceDF):
    # For the Sex attribute, replace all male values with 0, and female values with 1.
    # (For this historical dataset of Titanic passengers, only "male" and "female" are listed for sex.)
    targetDF.loc[:, "Sex"] = targetDF.loc[:, "Sex"].map(lambda v: 0 if v=="male" else v)
    targetDF.loc[:, "Sex"] = targetDF.loc[:, "Sex"].map(lambda v: 1 if v=="female" else v)
    
    # Fill not-available age values with the median value.
    targetDF.loc[:, 'Age'] = targetDF.loc[:, 'Age'].fillna(sourceDF.loc[:, 'Age'].median())
    
	# -------------------------------------------------------------
	# Problem 4 code goes here, for fixing the error
    targetDF.loc[:, "Embarked"] = targetDF.loc[:, "Embarked"].fillna(sourceDF.loc[:, "Embarked"].mode()[0])
    targetDF.loc[:, "Embarked"] = targetDF.loc[:, "Embarked"].map(lambda v: 0 if v == "C" else v)
    targetDF.loc[:, "Embarked"] = targetDF.loc[:, "Embarked"].map(lambda v: 1 if v == "Q" else v)
    targetDF.loc[:, "Embarked"] = targetDF.loc[:, "Embarked"].map(lambda v: 2 if v == "S" else v) 
    
    # -------------------------------------------------------------
	# Problem 5 code goes here, for fixing the error
    targetDF.loc[:, "Fare"] = targetDF.loc[:, "Fare"].fillna(sourceDF.loc[: , "Fare"].median())

	
# You'll need to write more code in this function, to complete it.
def buildAndTestModel():
    titanicTrain = pd.read_csv("/Users/aaryanjha/Desktop/hw06/Data/train.csv")
    preprocess(titanicTrain, titanicTrain)
	
	# -------------------------------------------------------------
	# Problem 4 code goes here, to make the LogisticRegression object.
    model = LogisticRegression(solver = 'liblinear')
    input_Columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    output_Column = "Survived"
    inputDF = titanicTrain.loc[: , input_Columns]
    outputSeries = titanicTrain.loc[: , output_Column] 
    c_v_score = model_selection.cross_val_score(model, inputDF, outputSeries, cv = 3, scoring='accuracy')
    avg_accuracy = np.mean(c_v_score)
    print(avg_accuracy)
	
	# -------------------------------------------------------------
	# Problem 5 code goes here, to try the Kaggle testing set
    titanicTest = pd.read_csv("/Users/aaryanjha/Desktop/hw06/Data/test.csv")
    preprocess(titanicTest, titanicTrain)
    test_InputDF = titanicTest.loc[:, input_Columns]    
    model = LogisticRegression(solver = 'liblinear')
    model.fit(inputDF, outputSeries)
    predictions = model.predict(test_InputDF)
    print(predictions, Counter(predictions), sep="\n")
    
    # -------------------------------------------------------------
    submitDF = pd.DataFrame({"PassengerId": titanicTest.loc[:, "PassengerId"], 
                             "Survived": predictions})
    submitDF.to_csv("data/submission.csv", index=False)


	
	
def test06():
    buildAndTestModel()    
