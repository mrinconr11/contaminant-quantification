##
from keras.layers import Input, Dense, Conv2D
from keras.layers import Activation, BatchNormalization, Flatten, MaxPooling2D
from keras.models import Model
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt

## Import training data
data = pd.read_csv(r'Creek_Train.csv')
df = pd.DataFrame(data)
train_data = np.array(df)
input_data_train = np.nan_to_num(train_data)
test = pd.read_csv(r'Creek_Test.csv')
dft = pd.DataFrame(test)
test_data = np.array(dft)
input_data_test = np.nan_to_num(test_data)

## Number of excitation/emission wavelengths in the excitation-emission matrix (EEM)
## Number of water samples
n_ex = 160
n_em = 240
train_quantity = 144
test_quantity = 36

## Remove scattering the excitation emission matrix (EEM)
def rem_scatt(x, nex, nem):
    ## ranges of the excitation-emission spectra
    ex = range(240, 602, 2)
    em = range(250, 802, 2)
    ind = []
    bw = 12
    for i in range(0, len(ex)):
        for j in range(0, len(em)):
            # if em[j] <= (ex[i]+bw) and em[j] >= (ex[i]-bw):
            if em[j] <= (ex[i] + bw):
                ind.append((i) * nem + (j))
            # if em[j] <= (ex[i]*2+bw) and em[j] >= (ex[i]*2-bw):
            if em[j] >= (ex[i] * 2 - bw):
                ind.append((i) * nem + (j))
    ## Make scattering 0
    ind = np.reshape(ind, (1, len(ind)))
    x[:, ind] = 0.
    return x
input_data_train= rem_scatt(input_data_train, 181, 276)

#Reduce matrix size for CNN
input_data_train = np.reshape(input_data_train, (-1, 181, 276))
input_data_train = input_data_train[:, 0:80, 0:176]
input_data_train = np.reshape(input_data_train, (-1, 80*176))

## Normalize EEMs
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

# Input style for CNN, requires 4D tensor
input_data_train_CNN = np.reshape(input_data_train, (-1, 80, 176, 1))
input_data_test = rem_scatt(input_data_test, 181, 276)

# Reduce matrix size for CNN, requires 4D tensor
input_data_test = np.reshape(input_data_test, (-1, 181, 276))
input_data_test = input_data_test[:, 0:80, 0:176]
input_data_test = np.reshape(input_data_test, (-1, 80*176))

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

input_data_test_CNN = np.reshape(input_data_test, (-1, 80, 176, 1))

atrain_concentration = pd.read_csv(r'../CNN/Creek_Train_Concentrations.csv')
dfc = pd.DataFrame(train_concentration)
train_concentration = np.array(dfc)
test_concentration = pd.read_csv(r'../CNN/Creek_Test_Concentrations.csv')
dfct = pd.DataFrame(test_concentration)
test_concentration = np.array(dfct)

## Separate concentration for naphthenic acids (NAs) and phenol for training and testing
train_Naph = train_concentration[:, 0]
train_Phenol = train_concentration[:, 1]
train_Naph.shape = [train_quantity, 1]
train_Phenol.shape = [train_quantity, 1]
test_Naph = test_concentration[:, 0]
test_Phenol = test_concentration[:, 1]
test_Naph.shape = [test_quantity, 1]
test_Phenol.shape = [test_quantity, 1]

## CNN for NAs, 5 convolutional layers. Predicts NAs concentrations
def naph(input):
    layer1 = Conv2D(32, (7, 7), padding='same')(input)
    layer1 = BatchNormalization()(layer1)
    layer1 = Activation('elu')(layer1)
    layer1 = MaxPooling2D((2,2))(layer1)
    layer2 = Conv2D(32, (7, 7), padding='same')(layer1)
    layer2 = BatchNormalization()(layer2)
    layer2 = Activation('elu')(layer2)
    layer3 = Conv2D(32, (7, 7), padding='same')(layer2)
    layer3 = BatchNormalization()(layer3)
    layer3 = Activation('elu')(layer3)
    layer4 = Conv2D(32, (7, 7), padding='same')(layer3)
    layer4 = BatchNormalization()(layer4)
    layer4 = Activation('elu')(layer4)
    layer5 = Conv2D(32, (7, 7), padding='same')(layer4)
    layer5 = BatchNormalization()(layer5)
    layer5 = Activation('elu')(layer5)
    layer5 = MaxPooling2D((2, 2))(layer5)
    flatten = Flatten()(layer5)
    dense = Dense(units=10, activation='elu')(flatten)
    output = Dense(units=1, activation='elu')(dense)
    return output

## The input is the EEM
inputs = Input(shape=(80, 176, 1))
naph_output = naph(inputs)
naph_model = Model(inputs=inputs, outputs=naph_output)
naph_model.compile(optimizer='adam', loss='mean_squared_error')

## Train the CNN
deep_auto_history = naph_model.fit(
    x=input_data_train_CNN,
    y=train_Naph,
    epochs=500,
    batch_size=32,
    shuffle=True,
    validation_split=0.2
)

## Make predictions of NAs concentration with the test data set
naph_predicted = naph_model.predict(input_data_test_CNN)

## The evaluation metrics are the mean absolute error (MAE) and R2
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
plt.title('NAs concentrations', fontsize=20)
plt.scatter(test_Naph, naph_predicted, c='crimson', label="Naphthenic")

x = np.linspace(0, 100)
plt.plot(x, x, c='black', linestyle='dotted')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Measured NAs (mg/L)', fontsize=20)
plt.ylabel('Predicted NAs (mg/L)', fontsize=20)
plt.axis('square')
plt.grid(linestyle='dotted', linewidth=1)
plt.legend(fontsize=20)
plt.ylim(0, 100)
plt.xlim(0, 100)
plt.show()
y = 0
