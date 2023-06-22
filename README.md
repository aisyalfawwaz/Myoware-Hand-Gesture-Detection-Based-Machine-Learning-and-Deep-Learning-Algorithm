# Myoware Hand Gesture Detection Based Machine Learning and Deep Learning Algorithm

This project aims to classify hand gestures based on the data collected by MyoWare sensors using Machine Learning and Deep Learning algorithms. The data has been preprocessed and visualized using Python libraries, and then fed into several models to evaluate their performance in classifying hand gestures.

## Requirements
- Python 3
- pandas
- numpy
- sklearn
- tensorflow
- xgboost
- seaborn
- matplotlib
- catboost

## Dataset
The dataset used in this project is stored in a CSV file named `all_labelled.csv`, which contains 1,800 samples of hand gestures. Each sample has 5 columns, the first 4 represent the readings from MyoWare sensors, and the last column represents the corresponding label of the hand gesture.

## Data Preprocessing
- The dataset is loaded using pandas, and the shape and first 5 rows of the data are printed.
- The data is then visualized using the `plot_data()` function to see the trend of each sensor reading over time.
- The data is split into input features (`x`) and target variable (`y`), and then further split into training and testing sets with a 0.25 test size using the `train_test_split()` function.
- The input features are then scaled using StandardScaler from sklearn.

## Models
Several Machine Learning and Deep Learning models were used to classify hand gestures:
- Random Forest Classifier
- XGBoost Classifier
- CatBoost Classifier
- Convolutional Neural Network (CNN)

The accuracy of each model is evaluated using the `accuracy_score` function from sklearn.metrics. The confusion matrix of each model is plotted using the `confusion_matrix` function from sklearn.metrics.

## Conclusion
The models' performance is evaluated based on their accuracy and confusion matrix. The results showed that the CNN model achieved the highest accuracy score of 89.44%, followed by the CatBoost model with 86.56%, the XGBoost model with 84.44%, and lastly, the Random Forest model with 80.44%.

The confusion matrix of each model is plotted to see the model's performance in predicting each label, where the CNN model has the least false predictions compared to the other models.

Overall, this project successfully classified hand gestures using Machine Learning and Deep Learning models with high accuracy, which has potential applications in the field of Human-Computer Interaction.

![Poster](https://i.ibb.co/Cv55mpy/A0-300-DPI-1.png)

