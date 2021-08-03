import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from sklearn import svm 

import pandas as pd

def getMVSKCSV(filename):
    data = pd.read_csv(filename, header=None)
    data = np.array(data)
    labels = data[:,1]                  #todos los labels
    values = data[:,range(2,6)]       #Tomamos los valores del linear binary pattern
    return labels, values

def classificate(label):
    if label==' grid':
        return 0
    elif label==' gauzy':
        return 1
    elif label==' grooved':
        return 2 
    return -1

#clases
classes, data = getMVSKCSV('momentos.csv')
target = [classificate(c) for c in classes]

#Normalizacion de datos
scaler = MinMaxScaler()

#Datos
X = scaler.fit_transform(data)
y = target

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
        scoring='accuracy', n_jobs=8,
)

#Entrenando el modelo
model.fit(X_train, y_train)

#Mostrando los mejores parametros
print("Los mejores parametros son:", model.best_params_ ,"con una precicion de:", model.best_score_)

predictions = model.predict(X_test)
acc = accuracy_score(y_test , predictions)

print("Mientras que la precision del Test es: ", acc)