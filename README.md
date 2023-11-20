# Lung-cancer-prediction-Using-Machine-Learning
## INTRODUCTION:
  Lung cancer poses a formidable global health challenge, ranking among the most
prevalent and lethal cancers. Its insidious nature often leads to diagnoses at advanced stages,
underscoring the urgent need for predictive tools that can facilitate early detection and
intervention. In this context, machine learning emerges as a transformative force in
healthcare. By harnessing computational algorithms to analyze diverse patient data, including
demographic information, medical history, and genetic factors, machine learning models hold
the promise of identifying individuals at risk of developing lung cancer. Successfully
navigating these challenges opens avenues for optimized resource allocation, targeted
screening, and a paradigm shift towards personalized medicine. This introduction sets the
stage for exploring how machine learning can be a catalyst in reshaping the landscape of lung
cancer diagnosis, underscoring its potential to be a vital tool in the ongoing fight against this
pervasive disease.
## PROBLEM STATEMENT:
Develop a robust machine learning model for the early detection and prediction of lung cancer based on diverse medical data, including but not limited to patient demographics, clinical history, genetic markers, and imaging studies. The primary objective is to create an accurate and reliable predictive tool that can assist healthcare professionals in identifying individuals at high risk of developing lung cancer, enabling early intervention and improving overall patient outcomes. The model should be capable of handling various data types, addressing imbalances in the dataset, and demonstrating high sensitivity and specificity in predicting the likelihood of lung cancer. Additionally, the system should be interpretable, providing insights into the key features influencing predictions, to enhance trust and understanding among healthcare practitioners.
## SCOPE OF THE PROJECT:
The scope of a lung cancer prediction project encompasses various aspects, including the
objectives, methodologies, technologies, and potential impact. Here is an outline of the scope
for a lung cancer prediction project:
Algorithm Selection and Development: Investigate and implement suitable machine learning
and deep learning algorithms for lung cancer prediction. Experiment with various models
such as logistic regression, support vector machines, and convolutional neural networks
(CNNs)
## METHODOLOGY:
The methodology for lung cancer prediction using machine learning begins with the comprehensive collection and preprocessing of diverse medical data, incorporating patient demographics, clinical history, genetic markers, and imaging studies. Following data splitting into training, validation, and test sets, the selection of suitable machine learning algorithms, such as logistic regression, support vector machines, random forests, and deep learning, is crucial. Model training addresses class imbalances, and hyperparameter tuning optimizes performance using validation data. Evaluation metrics like sensitivity, specificity, and ROC-AUC guide the assessment of model performance on the test set. Prioritizing model interpretability, techniques are implemented to provide insights into feature importance and decision processes. The deployment phase involves creating a user-friendly interface for healthcare professionals while adhering to ethical and regulatory standards. Continuous improvement is emphasized through ongoing monitoring, updating with new data, and exploration of emerging technologies. Comprehensive documentation ensures transparency, reproducibility, and potential future enhancements to the predictive model.
## WORKING PROCESS:         
![image](https://github.com/Aadheeshwar-AIDS/Lung-cancer-prediction-/assets/93427205/6be52e1d-4c58-49e8-b5eb-f4d94c210c64)
## PROGRAM
### importing the necessary packages
```python
 
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
```
```python
#reading the dataset using pandas 
data = pd.read_csv("/content/survey lung cancer.csv") 
dat
#checking some information of our dataset 
data.info() 
#checking for null values 
data.isnull().any() 
#coverting from float to integer 
encoder = LabelEncoder() 
data['LUNG_CANCER']=encoder.fit_transform(data['LUNG_CANCER']) 
data['GENDER'] = encoder.fit_transform(data['GENDER']) 
x = data.drop(columns = 'LUNG_CANCER') 
y = data['LUNG_CANCER'] 
y
```
### Visualizing Data
```python
#visualizing the people affected with lungcancer with respect to age 
fig,ax = plt.subplots(1,3,figsize=(20,6)) 
sns.distplot(data['AGE'],ax=ax[0]) 
sns.histplot(data =data,x='AGE',ax=ax[1],hue='LUNG_CANCER',kde=True) 
sns.boxplot(x=data['LUNG_CANCER'],y=data['AGE'],ax=ax[2]) 
plt.suptitle("Visualizing AGE column",size=20) 
warnings.filterwarnings('ignore') 
plt.show()
```
```python
#splitting the data as training data and testing data 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2) 
x.shape,x_train.shape,x_test.shape
```

### training the model 
```python
model = LogisticRegression() 
model.fit(x_train,y_train) 
#checking the accuracy of training data 
x_train_prediction = model.predict(x_train) 
training_data_accuracy = accuracy_score(x_train_prediction,y_train) 
print(training_data_accuracy) 
#checking the accuracy of testing data 
x_test_prediction = model.predict(x_test) 
testing_data_accuracy = accuracy_score(x_test_prediction,y_test) 
print(testing_data_accuracy)
```
### Building a prediction model
```python
input_data = [1 ,69 ,1, 2 ,2 ,1 ,1 ,2, 1, 2 ,2, 2 ,2, 2 ,2] 
#converting the input data into numpy array 
input_data_as_numpy_array = np.asarray(input_data) 
#reshaping the input data 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) 
prediction = model.predict(input_data_reshaped) 
print(prediction) 
if prediction[0]==0: 
print("The Patient does not Lung Cancer") 
else: 
print("The Patient Have Lung Cancer")
```

## Output:
### Displaying the values in the dataset:
![image](https://github.com/Aadheeshwar-AIDS/Lung-cancer-prediction-/assets/93427205/49849855-55ac-436a-a21b-befaeb5bc90c)
### Checking for NULL values:
![image](https://github.com/Aadheeshwar-AIDS/Lung-cancer-prediction-/assets/93427205/801f3437-4fe6-42bb-b5d6-61bf852ed2e8)
### Displaying the values in Y:
![image](https://github.com/Aadheeshwar-AIDS/Lung-cancer-prediction-/assets/93427205/01992dae-00fe-4aa2-97e2-0923e4acef75)
### Visualizing Data:
![image](https://github.com/Aadheeshwar-AIDS/Lung-cancer-prediction-/assets/93427205/7d4d63c0-4559-4d9b-b050-04009ed84b1f)
### Accuracy of training data:
![image](https://github.com/Aadheeshwar-AIDS/Lung-cancer-prediction-/assets/93427205/d38020bd-4cad-4b07-8cd4-fcbaa9da8d88)
### Accuracy of testing data:
![image](https://github.com/Aadheeshwar-AIDS/Lung-cancer-prediction-/assets/93427205/f8c5021a-12db-441c-aadb-841acf9dae9f)
### Prediction of a model:
![image](https://github.com/Aadheeshwar-AIDS/Lung-cancer-prediction-/assets/93427205/fcd4bd8d-de7d-4cd7-9488-a385fc6b9a1d)

## RESULT:
Upon successful completion of the lung cancer prediction project, the model is anticipated to demonstrate reliable performance metrics, such as high sensitivity and specificity, on the test dataset. The system should be capable of accurately identifying individuals at high risk of developing lung cancer based on diverse medical data, enabling early intervention and improved patient outcomes. The interpretability of the model would offer insights into the key features influencing predictions, fostering trust among healthcare practitioners. The deployed model would serve as a valuable tool for clinicians, contributing to early detection and personalized healthcare decisions in the context of lung cancer. Continuous monitoring and updates to the model would ensure its adaptability to new data and evolving healthcare practices, thereby enhancing its long-term effectiveness.









