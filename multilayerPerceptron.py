##
import numpy as np
import operator
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.models import Model

## Number of water samples for training and testing
train_quantity = 144
test_quantity = 36

## Import training data
data = pd.read_csv(r'Lake_Train.csv')
df = pd.DataFrame(data)
train_data = np.array(df)
input_data_train = np.nan_to_num(train_data)
test = pd.read_csv(r'Lake_Test.csv')
dft = pd.DataFrame(test)
test_data = np.array(dft)
input_data_test = np.nan_to_num(test_data)

## Normalize train dataset
def preprocess(x):
    y = np.copy(x)
    max_v = np.max(x, axis=0)
    min_v = np.min(x, axis=0)
    for i in range(np.size(max_v)):
        if (max_v[i] - min_v[i] > 0):
            y[:, i] = (x[:, i] - min_v[i]) / (max_v[i] - min_v[i])
        else:
            continue
    return y, max_v, min_v  # max and min vectors are needed to renormalize
input_data_train, max_v, min_v = preprocess(input_data_train)

## Normalize test dataset
def preprocess_test(x, max_v, min_v):
    y = np.copy(x)
    for i in range(np.size(max_v)):
        if (max_v[i] - min_v[i] > 0):
            y[:, i] = (x[:, i] - min_v[i]) / (max_v[i] - min_v[i])
        else:
            continue
    return y
input_data_test = preprocess_test(input_data_test, max_v, min_v)

## Import train and test concentrations
train_concentration = pd.read_csv(r'Lake_Train_Concentrations.csv')
dfc = pd.DataFrame(train_concentration)
train_concentration = np.array(dfc)
test_concentration = pd.read_csv(r'Lake_Test_Concentrations.csv')
dfct = pd.DataFrame(test_concentration)
test_concentration = np.array(dfct)

## Separate concentration for naphthenic acids (NAs) and phenol for training and testing
train_Naph = train_concentration[:,0]
train_Phenol = train_concentration[:,1]
train_Naph.shape = [train_quantity, 1]
train_Phenol.shape = [train_quantity, 1]
test_Naph = test_concentration[:,0]
test_Phenol = test_concentration[:,1]
test_Naph.shape = [test_quantity, 1]
test_Phenol.shape = [test_quantity, 1]

## ANN to predict naphthenic acids (NAs) concentrations. 1 hidden layer with nodes half of the EEM input
def naph(input):
    layer_1 = layers.Dense(units=2400, activation='elu')(input)
    output = layers.Dense(units=1, activation='elu')(layer_1)
    return output
inputs = layers.Input(shape=(np.shape(input_data_train)[1],))
naph_output = naph(inputs)
naph_model = Model(inputs=inputs, outputs=naph_output)
naph_model.compile(optimizer='adam', loss='mean_squared_error')
naph_model.summary()

## Train the ANN
deep_auto_history = naph_model.fit(
    x=input_data_train,
    y=train_Naph,
    epochs=500,
    batch_size=20,
    shuffle=True,
    validation_split=0.2
)

## Make predictions of NAs concentrations with the test dataset
naph_predicted = naph_model.predict(input_data_test)

## Evaluation metrics for the model. Mean absolute error (MAE) and R2
def mae(real, predicted):
    mae = np.sum(abs(real-predicted), axis=0)
    mae= mae/len(real)
    mae = np.nanmean(mae)
    return mae

def r2(real, predicted):
    mean = np.mean(real, axis=0)
    first_errors_autoencoder = np.sum((real-predicted)**2, axis=0)
    second_errors_autoencoder = np.sum((real-mean)**2, axis=0)
    r2_autoencoder = 1-first_errors_autoencoder/second_errors_autoencoder
    r2_autoencoder[r2_autoencoder == -inf] = nan
    r2 = np.nanmean(r2_autoencoder)
    return r2

mae_naph = float("{:.3f}".format(mae(test_Naph, naph_predicted)))
r2_naph = float("{:.3f}".format(r2(test_Naph, naph_predicted)))

## Plot predicted concentrations vs real concentrations
plt.figure(figsize=(10,10))
plt.title('1 dense layer', fontsize=20)
plt.scatter(test_Naph, naph_predicted, c='crimson', label="Naphthenic")

x = np.linspace(0, 100)
plt.plot(x, x, c='black', linestyle='dotted')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Measured NAs (mg/L) and Phenol (${\mu}$g/L)', fontsize=20)
plt.ylabel('Predicted NAs (mg/L) and Phenol (${\mu}$g/L)', fontsize=20)
plt.axis('square')
plt.grid(linestyle='dotted', linewidth=1)
plt.legend(fontsize=20)
plt.ylim(0, 100)
plt.xlim(0, 100)
plt.show()
