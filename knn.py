import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import unique_labels

# --- Load the Data ---
try:
    # Load the RT_IOT2022 dataset
    data = pd.read_csv("RT_IOT2022_transformed.csv")
    print("\nData Info:")
    data.info()
    print(f"\nData Shape: {data.shape}")

except FileNotFoundError:
    print("Error: 'RT_IOT2022_shuffled_temp.csv' not found.")
    print("Please make sure the file is in the correct directory.")
    exit()
data=data.sample(n=10000, random_state=42)  # Sample 10,000 rows for faster processing  
# Handle missing values by imputing or dropping them
if data.isnull().values.any():
    print("Missing values detected in the dataset.")
    print("Imputing missing values with the mean for numeric columns.")
    data.fillna(data.mean(numeric_only=True), inplace=True)
    print("Missing values have been handled.")

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


# Scale the dataset
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
numerical_features = df_scaled.select_dtypes(include=['number']).columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
numeric_X_train = X_train.select_dtypes(include=np.number)
numeric_X_test = X_test.select_dtypes(include=np.number)


# Check if there are any numeric columns
if numeric_X_train.shape[1] == 0:
    print("Error: No numeric columns found in the data.  KNN requires numeric features.")
    exit()
# Initialize the scaler
scaler = StandardScaler()
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
#k means
k_values = [5, 10, 20]
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(df_scaled[numerical_features])
    cluster_labels = kmeans.labels_
    df_scaled[f'cluster_{k}'] = cluster_labels

print(df_scaled.head())
# Select two principal components or relevant numerical features
x_axis = 'fwd_pkts_tot'
y_axis = 'bwd_pkts_tot'
# Create scatter plots for each k value
k_values = [5, 10, 20]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust figure size
for i, k in enumerate(k_values):
    ax = axes[i]
    ax.scatter(df_scaled[x_axis], df_scaled[y_axis], c=df_scaled[f'cluster_{k}'], cmap='viridis')
    ax.set_title(f'K-means Clustering (k={k})')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

plt.tight_layout()
plt.show()
print("K-Means clustering completed and summary statistics generated.")
#NAIVE BAYES
model = GaussianNB()

# 2. Train the model
model.fit(X_train, y_train)
print("\nNaive Bayes model trained.")

# 3. Make predictions
y_pred = model.predict(X_test)
print("Predictions made on test set.")

# 4. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Filter out labels with no true samples
def filter_labels_with_no_true_samples(y_true, y_pred):
    labels = unique_labels(y_true, y_pred)
    true_label_counts = {label: (y_true == label).sum() for label in labels}
    valid_labels = [label for label, count in true_label_counts.items() if count > 0]
    return valid_labels

# Get valid labels
valid_labels = filter_labels_with_no_true_samples(y_test, y_pred)

# Update classification_report and other metrics to use valid labels
report = classification_report(y_test, y_pred, labels=valid_labels, zero_division=0)
print("\nClassification Report:")
print(report)

# --- Plotting the Confusion Matrix ---
from sklearn.metrics import ConfusionMatrixDisplay

# Plot the confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.title("Confusion Matrix")

# Rotate x-axis labels to prevent overlap
plt.xticks(rotation=45, ha='right')  # or rotation=90 for vertical
plt.tight_layout()  # Adjust layout to prevent clipping
plt.xticks(rotation=90)
plt.show()

# --- Plotting Accuracy ---
plt.figure(figsize=(8, 6))
plt.bar(['Accuracy'], [accuracy], color='lightgreen')
plt.ylim(0, 1)
plt.title("Naive Bayes Model Accuracy")
plt.ylabel("Accuracy")
plt.show()

# --- Random Forest Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Train the model
model.fit(X_train, y_train)
print("\nRandom Forest model trained.")

# 3. Make predictions
y_pred = model.predict(X_test)
print("Predictions made on test set.")

# 4. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

# 5. Plot the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
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
report = classification_report(y_test, y_pred, zero_division=0)
print("\nClassification Report:")
print(report)

# Optional: Display a few actual vs predicted values for comparison
print("\nSample of Actual vs. Predicted values:")
comparison_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
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
# knn graph Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracy_scores, color='green', linestyle='solid', marker='^',
         markerfacecolor='blue', markersize=8)
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

# Define the metrics for each model
acc = accuracy_score(y_test, knn.predict(X_test_combined))
acc_nb = accuracy_score(y_test, model.predict(X_test))  # Naive Bayes accuracy
acc_rf = accuracy_score(y_test, model.predict(X_test))  # Random Forest accuracy

precision = classification_report(y_test, knn.predict(X_test_combined), output_dict=True, zero_division=0)['weighted avg']['precision']
precision_nb = classification_report(y_test, model.predict(X_test), output_dict=True, zero_division=0)['weighted avg']['precision']
precision_rf = classification_report(y_test, model.predict(X_test), output_dict=True, zero_division=0)['weighted avg']['precision']

recall = classification_report(y_test, knn.predict(X_test_combined), output_dict=True, zero_division=0)['weighted avg']['recall']
recall_nb = classification_report(y_test, model.predict(X_test), output_dict=True, zero_division=0)['weighted avg']['recall']
recall_rf = classification_report(y_test, model.predict(X_test), output_dict=True, zero_division=0)['weighted avg']['recall']

f1 = classification_report(y_test, knn.predict(X_test_combined), output_dict=True, zero_division=0)['weighted avg']['f1-score']
f1_nb = classification_report(y_test, model.predict(X_test), output_dict=True, zero_division=0)['weighted avg']['f1-score']
f1_rf = classification_report(y_test, model.predict(X_test), output_dict=True, zero_division=0)['weighted avg']['f1-score']

#compare the results of KNN, Naive Bayes, and Random Forest
# ---------------- Plots ----------------
models = ['KNN', 'Naive Bayes', 'Random Forest']
accuracy = [acc, acc_nb, acc_rf]
precision = [precision, precision_nb, precision_rf]
recall = [recall, recall_nb, recall_rf]
f1 = [f1, f1_nb, f1_rf]

x = range(len(models))
plt.figure(figsize=(12, 6))
plt.plot(x, accuracy, label='Accuracy', marker='o')
plt.plot(x, precision, label='Precision', marker='s')
plt.plot(x, recall, label='Recall', marker='^')
plt.plot(x, f1, label='F1 Score', marker='x')
plt.xticks(x, models)
plt.title("Model Comparison on Classification Metrics")
plt.xlabel("Models")
plt.ylabel("Score")
plt.ylim(0.7, 1.0)
plt.legend()
plt.grid(True)
plt.show()
