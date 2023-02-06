##
import numpy as np
import operator
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from functions import preprocess, preprocess_test, mae, r2

train_quantity = 144
test_quantity = 36

data = pd.read_csv(r'../Dense/Lake_Train_cut.csv')
df = pd.DataFrame(data)
train_data = np.array(df)
input_data_train = np.nan_to_num(train_data)
test = pd.read_csv(r'../Dense/Lake_Test_cut.csv')
dft = pd.DataFrame(test)
test_data = np.array(dft)
input_data_test = np.nan_to_num(test_data)

input_data_train, max_v, min_v = preprocess(input_data_train)
input_data_test = preprocess_test(input_data_test, max_v, min_v)

train_concentration = pd.read_csv(r'../Dense/Lake_Train_Concentrations.csv')
dfc = pd.DataFrame(train_concentration)
train_concentration = np.array(dfc)
test_concentration = pd.read_csv(r'../Dense/Lake_Test_Concentrations.csv')
dfct = pd.DataFrame(test_concentration)
test_concentration = np.array(dfct)

train_Naph = train_concentration[:,0]
train_Phenol = train_concentration[:,1]
train_Naph.shape = [train_quantity, 1]
train_Phenol.shape = [train_quantity, 1]
test_Naph = test_concentration[:,0]
test_Phenol = test_concentration[:,1]
test_Naph.shape = [test_quantity, 1]
test_Phenol.shape = [test_quantity, 1]
##
mae_naph_test = []
r2_naph_test = []
mae_phenol_test = []
r2_phenol_test = []
#naph_predicted_test = []
#phenol_predicted_test = []

for i in range(1):
    def naph(input):
        layer_1 = layers.Dense(units=2400, activation='elu')(input)
        output = layers.Dense(units=1, activation='elu')(layer_1)
        return output
    inputs = layers.Input(shape=(np.shape(input_data_train)[1],))
    naph_output = naph(inputs)
    naph_model = Model(inputs=inputs, outputs=naph_output)
    naph_model.compile(optimizer='adam', loss='mean_squared_error')
    naph_model.summary()
    deep_auto_history = naph_model.fit(
        x=input_data_train,
        y=train_Naph,
        epochs=500,
        batch_size=20,
        shuffle=True,
        validation_split=0.2
    )
    naph_predicted = naph_model.predict(input_data_test)
    mae_naph = float("{:.3f}".format(mae(test_Naph, naph_predicted)))
    r2_naph = float("{:.3f}".format(r2(test_Naph, naph_predicted)))
    mae_naph_test.append(mae_naph)
    r2_naph_test.append(r2_naph)
    #naph_predicted_test.append(naph_predicted)
    #naph_model.save('naph_modelDense_Creek.h5')

    def phenol(input):
        layer_1 = layers.Dense(units=2400, activation='elu')(input)
        output = layers.Dense(units=1, activation='elu')(layer_1)
        return output
    inputs = layers.Input(shape=(np.shape(input_data_train)[1],))
    phenol_output = phenol(inputs)
    phenol_model = Model(inputs=inputs, outputs=phenol_output)
    phenol_model.compile(optimizer='adam', loss='mean_squared_error')
    deep_auto_history = phenol_model.fit(
        x=input_data_train,
        y=train_Phenol,
        epochs=500,
        batch_size=20,
        shuffle=True,
        validation_split=0.2
    )
    phenol_predicted = phenol_model.predict(input_data_test)
    mae_phenol = float("{:.3f}".format(mae(test_Phenol, phenol_predicted)))
    r2_phenol = float("{:.3f}".format(r2(test_Phenol, phenol_predicted)))
    mae_phenol_test.append(mae_phenol)
    r2_phenol_test.append(r2_phenol)
    #phenol_predicted_test.append(phenol_predicted)
    #phenol_model.save('phenol_modelDense_Creek.h5')
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
plt.title('1 dense layer', fontsize=20)
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

