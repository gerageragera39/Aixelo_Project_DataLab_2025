# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load and preprocess the data
data = df_45_bandgap_joined_clean.reset_index().drop(columns=['MOF', 'CBM_PBE', 'VBM_PBE', 'Direct_PBE'])

# Exclude the first row and first column
data = data.iloc[1:, 1:]

# Ensure the data is numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
data_clean = data.dropna()

# Split features and target variable
X = data_clean.drop(columns=['BG_PBE'])
y = data_clean['BG_PBE']

# Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Applying PCA for dimensionality reduction (optional)
#pca = PCA(n_components=20)  # Increase the number of components for more variance retention
#X_train_pca = pca.fit_transform(X_train)
#X_valid_pca = pca.transform(X_valid)
#X_test_pca = pca.transform(X_test)

# Define the neural network model with additional layers and dropout
model = Sequential([
    Dense(256, input_dim=X_train_pca.shape[1], activation='relu'),  # Increase number of neurons
    Dropout(0.3),  # Increase dropout rate
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model with a reduced learning rate
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')  # Adjust learning rate

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # Increase patience
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6)  # Adjust patience and min_lr

# Train the model with increased epochs
history = model.fit(X_train_pca, y_train, validation_data=(X_valid_pca, y_valid), epochs=500, batch_size=32, verbose=1,
                    callbacks=[early_stopping, reduce_lr])

# Evaluate the model on the test set
loss = model.evaluate(X_test_pca, y_test)
print(f'Test loss: {loss}')

# Make predictions
y_pred = model.predict(X_test_pca)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# Plot the loss over epochs
plt.figure(figsize=(10, 7))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss Over Epochs')
plt.legend()
plt.show()

# Scatter plot of predictions vs true values
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('True BG_PBE')
plt.ylabel('Predicted BG_PBE')
plt.title('Predicted vs True BG_PBE')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line of perfect prediction
plt.show()

# Applying PCA to the full scaled data for the cumulative explained variance plot
pca_full = PCA().fit(X_train)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.show()