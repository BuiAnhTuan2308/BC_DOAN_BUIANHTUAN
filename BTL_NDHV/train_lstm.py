import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split

# Đọc dữ liệu
bodyswing_df = pd.read_csv("SWING.txt")
handswing_df = pd.read_csv("HANDSWING.txt")
clapping_df = pd.read_csv("CLAPPING.txt")
donothing_df= pd.read_csv("DO NOTHING.txt")
lefthand_df = pd.read_csv("LEFT HAND MOVE.txt")
X = []
y = []
no_of_timesteps = 10



# Xử lý dữ liệu cho hành động "swing body"
dataset = bodyswing_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(1)  # Gán nhãn cho hành động "swing body"


# Xử lý dữ liệu cho hành động "swing hand"
dataset = handswing_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(0)  # Gán nhãn cho hành động "swing hand"


# Xử lý dữ liệu cho hành động "clapping"
dataset = clapping_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(2)  # Gán nhãn cho hành động "clapping"

# Xử lý dữ liệu cho hành động "Do Nothing"
dataset = donothing_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(4)  # Gán nhãn cho hành động "Do Nothing"

# Xử lý dữ liệu cho hành động "left hand move"
dataset = lefthand_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps:i, :])
    y.append(3)  # Gán nhãn cho hành động "left hand move"


X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Xây dựng mô hình
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=5, activation="softmax"))  # Số lượng hành động là 3: swing hand, swing body, clapping
model.compile(optimizer="adam", metrics=['accuracy'], loss="sparse_categorical_crossentropy")

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=16, batch_size=32, validation_data=(X_test, y_test))
# Lưu mô hình
model.save("model.h5")
