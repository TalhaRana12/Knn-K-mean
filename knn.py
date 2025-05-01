import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# --- Load the Data ---
try:
    # Load the RT_IOT2022 dataset
    data = pd.read_csv("RT_IOT2022_transformed.csv")
    print("Successfully loaded RT_IOT2022_shuffled_temp.csv")
    #print("\nOriginal Data Head:")
    #print(data.head())
    #print("\nData Info:")
    #data.info()
    print(f"\nData Shape: {data.shape}")

except FileNotFoundError:
    print("Error: 'RT_IOT2022_shuffled_temp.csv' not found.")
    print("Please make sure the file is in the correct directory.")
    exit()

# Handle missing values by imputing or dropping them
if data.isnull().values.any():
    print("Missing values detected in the dataset.")
    print("Imputing missing values with the mean for numeric columns.")
    data.fillna(data.mean(numeric_only=True), inplace=True)
    print("Missing values have been handled.")
#data=data.sample(n=10000, random_state=42)  # Sample 10,000 rows for faster processing
# --- Preprocessing Steps ---

# 1. Separate features (X) and target variable (y)
# Identify features and target. Assuming 'Attack' is the target.  Adjust as needed.
features = [col for col in data.columns if col != 'Attack_type']  # Use all columns except 'Attack'
target = 'Attack_type'

# Check if columns exist before proceeding
if not all(col in data.columns for col in features):
    print(f"Error: One or more feature columns not found in the CSV.")
    exit()
if target not in data.columns:
    print(f"Error: Target column '{target}' not found in the CSV.")
    exit()

X = data[features]
y = data[target]

print(f"\nFeatures (X) shape: {X.shape}")
print("Features Head:")
print(X.head())
print(f"\nTarget (y) shape: {y.shape}")
print("Target Head:")
print(y.head())

# 2. Split the data into training and test sets (BEFORE scaling)
# stratify=y ensures the proportion of attack types is similar in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# print("\n--- After Splitting ---")
# print(f"X_train shape: {X_train.shape}")
# print(f"X_test shape: {X_test.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"y_test shape: {y_test.shape}")
# print("\nProportion of classes in y_train:")
# print(y_train.value_counts(normalize=True))
# print("\nProportion of classes in y_test:")
# print(y_test.value_counts(normalize=True))


# 3. Normalize the FEATURES (X) using StandardScaler
# Identify numeric columns for scaling
numeric_X_train = X_train.select_dtypes(include=np.number)
numeric_X_test = X_test.select_dtypes(include=np.number)


# Check if there are any numeric columns
if numeric_X_train.shape[1] == 0:
    print("Error: No numeric columns found in the data.  KNN requires numeric features.")
    exit()
# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler ONLY on the training data
scaler.fit(numeric_X_train)

# Transform both the training and testing data using the fitted scaler
X_train_scaled = scaler.transform(numeric_X_train)
X_test_scaled = scaler.transform(numeric_X_test)

# Create DataFrames from the scaled NumPy arrays
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numeric_X_train.columns, index=numeric_X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numeric_X_test.columns, index=numeric_X_test.index)

# Combine scaled numeric data with non-numeric data
X_train_combined = pd.concat([X_train_scaled_df, X_train.drop(columns=numeric_X_train.columns, errors='ignore')], axis=1)
X_test_combined = pd.concat([X_test_scaled_df, X_test.drop(columns=numeric_X_test.columns, errors='ignore')], axis=1)


# X_train_scaled and X_test_scaled are now NumPy arrays
# print("\n--- After Scaling ---")
# print("Scaled Training Data (X_train_scaled) Head (first 5 rows as numpy array):")
# print(X_train_scaled[:5])
# print("\nMean of scaled training features (should be close to 0):")
# print(X_train_scaled.mean(axis=0))
# print("\nStandard Deviation of scaled training features (should be close to 1):")
# print(X_train_scaled.std(axis=0))

# print("\nScaled Testing Data (X_test_scaled) Head (first 5 rows as numpy array):")
# print(X_test_scaled[:5])

# Now you have:
# X_train_scaled: Scaled features for training the model
# X_test_scaled: Scaled features for testing the model
# y_train: Target variable for training
# y_test: Target variable for testing

# --- Ready for Model Training ---
# print("\nData preprocessing complete. Ready for model training using:")
# print("X_train_combined, y_train, X_test_combined, y_test")

# --- Step 4: Train the KNN Model ---
# Initialize the KNN classifier with k=3 (a common starting point)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model using the scaled training data
knn.fit(X_train_combined, y_train)

print("KNN model trained successfully.")

# --- Step 5: Evaluate the Model ---
print("\n--- Step 5: Evaluating Model Performance ---")

# Use the trained model to make predictions on the unseen scaled test data
y_pred = knn.predict(X_test_combined)

print("Predictions made on the test set.")

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Generate a detailed classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

# Optional: Display a few actual vs predicted values for comparison
print("\nSample of Actual vs. Predicted values:")
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df.head(10).to_string())

# --- Step 6: Visualizing K vs. Accuracy ---
print("\n--- Step 6: Visualizing K vs. Accuracy ---")

# Define a range of k values to test
k_range = range(1, 21)
accuracy_scores = []

# Loop through each value of k
print(f"Testing k values from {k_range.start} to {k_range.stop - 1}...")
for k in k_range:
    # 1. Initialize KNN classifier with the current k
    knn = KNeighborsClassifier(n_neighbors=k)

    # 2. Train the model
    knn.fit(X_train_combined, y_train)

    # 3. Predict on the test set
    y_pred = knn.predict(X_test_combined)

    # 4. Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # 5. Store the accuracy
    accuracy_scores.append(accuracy)

# Find the k value that gave the highest accuracy
best_k_index = np.argmax(accuracy_scores)
best_k = k_range[best_k_index]
best_accuracy = accuracy_scores[best_k_index]

print(f"\nBest accuracy found: {best_accuracy:.4f} at k={best_k}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracy_scores, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=8)
plt.title('Accuracy vs. K Value for KNN Classifier')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.annotate(f'Best k={best_k}\nAccuracy={best_accuracy:.4f}',
             xy=(best_k, best_accuracy),
             xytext=(best_k + 1, best_accuracy - 0.01),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.xticks(k_range)
plt.grid(True)
plt.show()
