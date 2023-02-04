##
from keras.layers import Input, Dense, Conv2D
from keras.layers import Activation, BatchNormalization, Flatten, MaxPooling2D
from keras.models import Model
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
from Dense.functions import mae, rem_scatt, preprocess_test, preprocess, r2
## Training data
data = pd.read_csv(r'Creek_Train_largo.csv')
df = pd.DataFrame(data)
train_data = np.array(df)
input_data_train = np.nan_to_num(train_data)
test = pd.read_csv(r'Creek_Test_largo.csv')
dft = pd.DataFrame(test)
test_data = np.array(dft)
input_data_test = np.nan_to_num(test_data)

n_ex = 160
n_em = 240
train_quantity = 144
test_quantity = 36

input_data_train= rem_scatt(input_data_train, 181, 276)

#Reduce matrix size for CNN
input_data_train = np.reshape(input_data_train, (-1, 181, 276))
input_data_train = input_data_train[:, 0:80, 0:176]
input_data_train = np.reshape(input_data_train, (-1, 80*176))

input_data_train, max_v, min_v = preprocess(input_data_train)

# Input style for CNN, requires 4D tensor
input_data_train_CNN = np.reshape(input_data_train, (-1, 80, 176, 1))
input_data_test = rem_scatt(input_data_test, 181, 276)

# Reduce matrix size for CNN, requires 4D tensor
input_data_test = np.reshape(input_data_test, (-1, 181, 276))
input_data_test = input_data_test[:, 0:80, 0:176]
input_data_test = np.reshape(input_data_test, (-1, 80*176))

input_data_test = preprocess_test(input_data_test, max_v, min_v)

input_data_test_CNN = np.reshape(input_data_test, (-1, 80, 176, 1))

atrain_concentration = pd.read_csv(r'../CNN/Creek_Train_Concentrations_largo.csv')
dfc = pd.DataFrame(train_concentration)
train_concentration = np.array(dfc)
test_concentration = pd.read_csv(r'../CNN/Creek_Test_Concentrations_largo.csv')
dfct = pd.DataFrame(test_concentration)
test_concentration = np.array(dfct)

train_Naph = train_concentration[:, 0]
train_Phenol = train_concentration[:, 1]
train_Naph.shape = [train_quantity, 1]
train_Phenol.shape = [train_quantity, 1]
test_Naph = test_concentration[:, 0]
test_Phenol = test_concentration[:, 1]
test_Naph.shape = [test_quantity, 1]
test_Phenol.shape = [test_quantity, 1]
##
mae_naph_test = []
r2_naph_test = []
mae_phenol_test = []
r2_phenol_test = []
naph_predicted_test = []
phenol_predicted_test = []

for i in range(1):
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

    inputs = Input(shape=(80, 176, 1))
    naph_output = naph(inputs)
    naph_model = Model(inputs=inputs, outputs=naph_output)
    naph_model.compile(optimizer='adam', loss='mean_squared_error')
    deep_auto_history = naph_model.fit(
        x=input_data_train_CNN,
        y=train_Naph,
        epochs=500,
        batch_size=32,
        shuffle=True,
        validation_split=0.2
    )
    naph_predicted = naph_model.predict(input_data_test_CNN)
    mae_naph = float("{:.3f}".format(mae(test_Naph, naph_predicted)))
    r2_naph = float("{:.3f}".format(r2(test_Naph, naph_predicted)))
    mae_naph_test.append(mae_naph)
    r2_naph_test.append(r2_naph)
    naph_predicted_test.append(naph_predicted)

    def phenol(input):
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

    inputs = Input(shape=(80, 176, 1))
    phenol_output = phenol(inputs)
    phenol_model = Model(inputs=inputs, outputs=phenol_output)
    phenol_model.compile(optimizer='adam', loss='mean_squared_error')
    deep_auto_history = phenol_model.fit(
        x=input_data_train_CNN,
        y=train_Phenol,
        epochs=500,
        batch_size=32,
        shuffle=True,
        validation_split=0.2
    )
    phenol_predicted = phenol_model.predict(input_data_test_CNN)
    mae_phenol = float("{:.3f}".format(mae(test_Phenol, phenol_predicted)))
    r2_phenol = float("{:.3f}".format(r2(test_Phenol, phenol_predicted)))
    mae_phenol_test.append(mae_phenol)
    r2_phenol_test.append(r2_phenol)
    phenol_predicted_test.append(phenol_predicted)
    phenol_model.save('phenol_model_creek.h5')
##
r2_index_naph, r2_value_naph = max(enumerate(r2_naph_test), key=operator.itemgetter(1))
mae_index_naph, mae_value_naph = min(enumerate(mae_naph_test), key=operator.itemgetter(1))
r2_index_phenol, r2_value_phenol = max(enumerate(r2_phenol_test), key=operator.itemgetter(1))
mae_index_phenol, mae_value_phenol = min(enumerate(mae_phenol_test), key=operator.itemgetter(1))

labels = ['Naphthenic','Phenol']
concentration_predicted = [naph_predicted_test[mae_index_naph], phenol_predicted_test[mae_index_phenol]]
concentration_predicted = np.array(concentration_predicted)
concentration_predicted.shape = [2, test_quantity]
concentration_predicted = concentration_predicted.transpose()

real_datafrane = pd.DataFrame(test_concentration, columns=labels)
predicted_dataframe = pd.DataFrame(concentration_predicted, columns=labels)

plt.figure(figsize=(10,10))
plt.title('5 CNN layers', fontsize=20)
plt.scatter(real_datafrane['Naphthenic'], predicted_dataframe['Naphthenic'], c='crimson', label="Naphthenic")
plt.scatter(real_datafrane['Phenol'], predicted_dataframe['Phenol'], c='blue', label='Phenol')

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
y = 0