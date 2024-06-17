import matplotlib.pyplot as plt
import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset_path = 'auto-mpg.data-original'
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration','Model Year','Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values = "?", comment='\t',
                          sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
dataset.isna().sum()
dataset = dataset.dropna()
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


train_labels = train_dataset.pop('MPG')
train_labels = test_dataset.pop('MPG')


train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# Model = keras.sequentian([
# layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())])
# layers.Dense(64, activation=tf.nn.relu)
# layers.Dense(1)
#  ])
#
#
# optimizer = tf.keras.optimizers.RMSprop(0.001)
#
# Model.compile(loss='mean_squared_error',
#               optimizer=optimizer,
#               metrics=['mean_absolute_error', 'mean_sequared_error'] )
#
# Model.summary()
#
# EPOCHS = 1000
#
# history = Model.fit(
#     normed_train_data, train_labels,
#     epochs=EPOCHS, validation_split = 0.2)
#
# loss, mae, mse = Model.evaluate(normed_test_data, test_labels, verbose=0)
#
# test_predictions = Model.predict(normed_test_data).flatten()
#
# plt.scatter(test_labels, test_predictions)
# plt.xlabel('True values [MPG]')
# plt.ylebel('predictions [MPG]')
# plt.axis('equal')
# plt.axis('square')
# plt.xlim([0,plt.xlim()[1]])
# plt.ylim([0,plt.xlim()[1]])
# plt.plot([-100, 100], [-100, 100])
