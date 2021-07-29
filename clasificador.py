import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn import svm 

#clases
classes = ['Iris setosa', 'Iris Versicolour', 'Iris Virginica']

#dataset
iris = datasets.load_iris()

#Datos
X = iris.data
y = iris.target

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Definiendo el rango de parametros
C_range = [1,10,100, 1000]
gamma_range = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]

#Definiendo el modelo SVC con Radial Basis Kernel(rbfk)
model = GridSearchCV(
    svm.SVC(kernel='rbf'), 
    param_grid=
        {
            #Parametros del grid search
            'C':C_range,
            'gamma':gamma_range,
        },
    cv=5    #Number of cross validations
)

#Entrenando el modelo
model.fit(X_train, y_train)

#Mostrando los mejores parametros
print("Los mejores parametros son", model.best_params_ ,"con una precicion de %0.2f", model.best_score_)

predictions = model.predict(X_test)
acc = accuracy_score(y_test , predictions)

print("accuracy: ", acc)
