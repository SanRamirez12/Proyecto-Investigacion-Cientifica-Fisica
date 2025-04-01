from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model=Sequential([
Dense(units=25,activation="ReLU"),
Dense(units=15,activation ="ReLU"),
Dense(units=1,activation ="sigmoid"),
])

print(model.shape())
