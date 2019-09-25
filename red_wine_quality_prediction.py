import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,mean_absolute_error

#load the dataset using pandas
red_wine_data = pd.read_csv('winequality-red.csv', sep=';')

#---------------- Data Visualization ----------------------------

print(red_wine_data.head())

#test for null values and check correct datatypes 
print('\nTest for null values and check correct datatypes')
assert red_wine_data.notnull().all().all()
red_wine_data.info()


#---------------- Data Analysis --------------------------------

#distribution of quality ranks for red wine
red_wine_data['quality'] = pd.Categorical(red_wine_data['quality'])
sns.countplot(x="quality", data=red_wine_data)
plt.xlabel("Quality level of wine (0-10 scale)")
plt.title('Distribution of quality ranks for red wine')
plt.show()

#correlation between different features plotted in a heat map
correlation = red_wine_data.corr()
figure = plt.subplots(figsize=(10,10))
sns.heatmap(correlation,vmax=1,square=True,annot=True,cmap='Reds')
plt.title('Correlation between different features')
plt.show()

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

#ploting the regression between different features and label(quality)
sns.pairplot(red_wine_data,x_vars=features,y_vars='quality',kind='reg',size=7,aspect=0.5)
plt.title('Regression between different features and quality')
plt.show()


#------------------- Decision Tree Classification --------------------------------------

#drop target variable
x=red_wine_data.drop('quality', axis=1)
y=red_wine_data.quality

#split dataset into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,train_size=0.8,random_state = 0)

# Show the results of the split
print("\nTraining set has {} samples.".format(x_train.shape[0]))
print("Testing set has {} samples.".format(x_test.shape[0]))

clf=tree.DecisionTreeClassifier()
#build decision tree classifier from the training set
clf.fit(x_train, y_train)

#predict class or regression value for x
y_pred = clf.predict(x_test)

#converting the numpy array to list
x_list=np.array(y_pred).tolist()

#printing first twenty predictions
print("\nPredicted quality of first twenty wines:")
for i in range(0,20):
    print(x_list[i])
    
#printing first twenty expectations
print("\nExpected quality first twenty wines:")
print(y_test.head(n=20))

#mean accuracy on the given test data and label(quality).
confidence = clf.score(x_test, y_test)
print("\nMean accuracy on the given test data and label:",confidence)
print("\nMean accuracy on the given test data and label(As a percentage): {}%".format(int(round(confidence * 100))))


plt.scatter(y_test, y_pred)
plt.xlabel('True Quality')
plt.ylabel('Predicted Quality')
plt.title('Predicted Quality Against True Quality of Red Wines')
plt.show()

print('\nDecision Tree score for test set: %f' % clf.fit(x_train, y_train).score(x_test, y_test))

#show main classification metrics
print('\nClassification Report')
print(classification_report(y_test, y_pred))

#confusion matrix to evaluate the accuracy of a classification
print('\nConfusion Metrix')
print(confusion_matrix(y_test, y_pred))

#calculate accuracy classification score
print('\nAccuracy Classification Score:')
print(accuracy_score(y_test, y_pred))

#calculate Mean Absolute Error
print('\nMean Absolute Error:')
print(mean_absolute_error(y_test, y_pred))

tree.export_graphviz(clf, out_file='tree.dot')

print('\nFeature Importances for Decision tree\n')
for importance,feature in zip(clf.feature_importances_,['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']):
    print('{}: {}'.format(feature,importance))
#------------------- After Scaling-------------------

#
#scaled_x=scale(x)
#
##split dataset into training and testing data
#scaled_x_train, scaled_x_test, y_train, y_test = train_test_split(scaled_x, y,test_size=0.2,train_size=0.8,random_state = 0)
#
## Show the results of the split
#print("\nTraining set has {} samples.".format(scaled_x_train.shape[0]))
#print("Testing set has {} samples.".format(scaled_x_test.shape[0]))
#
#scaled_clf=tree.DecisionTreeClassifier()
##build decision tree classifier from the training set
#scaled_clf.fit(scaled_x_train, y_train)
#
##predict class or regression value for x
#scaled_y_pred = clf.predict(scaled_x_test)
#
##converting the numpy array to list
#scaled_x_list=np.array(scaled_y_pred).tolist()
#
##printing first twenty predictions
#print("\nPredicted quality of first twenty wines:")
#for i in range(0,20):
#    print(scaled_x_list[i])
#    
##printing first twenty expectations
#print("\nExpected quality first twenty wines:")
#print(y_test.head(n=20))
#
##mean accuracy on the given test data and label(quality).
#confidence = clf.score(scaled_x_test, y_test)
#print("\nMean accuracy on the given test data and label:",confidence)
#print("\nMean accuracy on the given test data and label(As a percentage): {}%".format(int(round(confidence * 100))))
#
#
#plt.scatter(y_test, scaled_y_pred)
#plt.xlabel('True Quality')
#plt.ylabel('Predicted Quality')
#plt.title('Predicted Quality Against True Quality of Red Wines')
#plt.show()
#
#print('\nDecision Tree score for test set: %f' % clf.fit(scaled_x_train, y_train).score(scaled_x_test, y_test))
#
##show main classification metrics
#print('\nClassification Report')
#print(classification_report(y_test, scaled_y_pred))
#
##confusion matrix to evaluate the accuracy of a classification
#print('\nConfusion Metrix')
#print(confusion_matrix(y_test, scaled_y_pred))
#
##calculate accuracy classification score
#print('\nAccuracy Classification Score:')
#print(accuracy_score(y_test, scaled_y_pred))
#
##calculate Mean Absolute Error
#print('\nMean Absolute Error:')
#print(mean_absolute_error(y_test, scaled_y_pred))

