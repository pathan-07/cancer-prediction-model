import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
import streamlit as st

# Load the dataset
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Drop unnecessary columns
data = data.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Convert diagnosis to binary values
data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Define features and target
X = data.drop(columns='diagnosis')
y = data['diagnosis']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = keras.Sequential([
    keras.layers.Dense(30, input_shape=(X_train.shape[1],), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=2)

# Streamlit app
st.title("Breast Cancer Detection Model")

st.sidebar.header("User Input Parameters")

def user_input_features():
    mean_radius = st.sidebar.slider('Mean Radius', float(X['radius_mean'].min()), float(X['radius_mean'].max()), float(X['radius_mean'].mean()))
    texture_mean = st.sidebar.slider('Texture Mean', float(X['texture_mean'].min()), float(X['texture_mean'].max()), float(X['texture_mean'].mean()))
    perimeter_mean = st.sidebar.slider('Perimeter Mean', float(X['perimeter_mean'].min()), float(X['perimeter_mean'].max()), float(X['perimeter_mean'].mean()))
    area_mean = st.sidebar.slider('Area Mean', float(X['area_mean'].min()), float(X['area_mean'].max()), float(X['area_mean'].mean()))
    smoothness_mean = st.sidebar.slider('Smoothness Mean', float(X['smoothness_mean'].min()), float(X['smoothness_mean'].max()), float(X['smoothness_mean'].mean()))
    
    data = {
        'radius_mean': mean_radius,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Add missing features and fill with mean values
for col in X.columns:
    if col not in df.columns:
        df[col] = X[col].mean()

# Normalize user input
df_scaled = scaler.transform(df)

# Make predictions
prediction = model.predict(df_scaled)
st.subheader('Prediction')
st.write('Malignant' if prediction > 0.5 else 'Benign')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
st.subheader('Model Accuracy')
st.write(f'Accuracy: {accuracy:.2f}')

# Confusion matrix
y_pred = (model.predict(X_test) > 0.5).astype("int32")
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(plt.gcf())

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
st.pyplot(plt.gcf())
