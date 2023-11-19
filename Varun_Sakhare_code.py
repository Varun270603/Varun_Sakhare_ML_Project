# Title: Prediction of Mortality Rate of Heart Failure Patients Admitted to ICU
# Name: Varun Sakhare
# Registration No./Roll No.: 21301
# Institute/University Name: IISER Bhopal
# Program/Stream: EECS


import numpy as np
import pandas as pd
from collections import Counter
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Read data
data = pd.read_csv('training_data.csv')
targets = pd.read_csv('training_data_targets.csv')

def get_class_statistics(labels):
    class_statistics = Counter(labels)
    print('\n Class \t\t Number of Instances \n')
    for item in class_statistics:
        print(f'\t{item}\t\t\t{class_statistics[item]}')

datas = targets.iloc[:, :-1]
labels = targets['outcomes']
get_class_statistics(labels)

# Fill missing values
columns_to_fill = ['heart rate', 'BMI', 'Systolic blood pressure', 'Diastolic blood pressure', 'Respiratory rate',
                   'temperature', 'SP O2', 'Urine output', 'Neutrophils', 'Basophils', 'Lymphocyte', 'PT', 'INR',
                   'Creatine kinase', 'glucose', 'Blood calcium', 'PH', 'PCO2']

for column in columns_to_fill:
    data[column].fillna(data[column].median(), inplace=True)

data['Lactic acid'].fillna(data['Lactic acid'].median(), inplace=True)

# Standardize data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Feature selection
k_best_features = 10
outcomes = labels
selector = SelectKBest(score_func=mutual_info_classif, k=k_best_features)
data_selected = selector.fit_transform(data_standardized, outcomes)

selected_feature_indices = selector.get_support(indices=True)
selected_features = data.columns[selected_feature_indices]

print("\nSelected Features:")
print(selected_features)

binary_selected_features = [feature for feature in selected_features if len(data[feature].unique()) == 2]
multivalue_selected_features = [feature for feature in selected_features if len(data[feature].unique()) > 2]

# One-hot encode binary features
onehot_encoder = OneHotEncoder()
hot_binary_data = onehot_encoder.fit_transform(data[binary_selected_features]).toarray()

hot_data = np.hstack((hot_binary_data, data[multivalue_selected_features].values))
data = data.drop(columns=binary_selected_features)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(hot_data, labels, test_size=0.5, random_state=42, stratify=labels)

# Oversample with SMOTE
smote = SMOTE(random_state=10)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

# Display class statistics
print("\n Training Data ")
get_class_statistics(Y_train)

print("\n Validation Data ")
get_class_statistics(Y_test)

print("\n Training Data after Oversampling")
get_class_statistics(Y_train_resampled)

# Define classifiers
classifiers = [
    ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=50, min_samples_split=2, random_state=42, class_weight='balanced')),
]

# Train and evaluate classifiers
for clf_name, clf in classifiers:
    clf.fit(X_train_resampled, Y_train_resampled)
    predicted = clf.predict(X_test)
    class_names = [str(item) for item in list(Counter(Y_test).keys())]

    # Display classification report
    print("\n ##### Classification Report for {} ##### \n".format(clf_name))
    print(classification_report([str(label) for label in Y_test], [str(pred) for pred in predicted], target_names=class_names))

    # Display precision, recall, and F1-score
    pr = precision_score(Y_test, predicted, average='macro')
    print('\n Precision:\t' + str(pr))

    rl = recall_score(Y_test, predicted, average='macro')
    print('\n Recall:\t' + str(rl))

    fm = f1_score(Y_test, predicted, average='macro')
    print('\n F1-Score:\t' + str(fm))

    # Read test data
test_data = pd.read_csv('test_data.csv')

# Fill missing values in the test data
for column in columns_to_fill:
    test_data[column].fillna(test_data[column].median(), inplace=True)

test_data['Lactic acid'].fillna(test_data['Lactic acid'].median(), inplace=True)

# Standardize test data
test_data_standardized = scaler.transform(test_data)

# Feature selection for test data
test_data_selected = selector.transform(test_data_standardized)

# One-hot encode binary features for test data
hot_binary_test_data = onehot_encoder.transform(test_data[binary_selected_features]).toarray()

hot_test_data = np.hstack((hot_binary_test_data, test_data[multivalue_selected_features].values))
test_data = test_data.drop(columns=binary_selected_features)

# Predict using the trained classifier
predicted_labels = clf.predict(hot_test_data)

# Save predicted labels to a text file
np.savetxt("Varun_Sakhare_predicted_labels.txt", predicted_labels, fmt="%s")

print("Predicted labels saved to Varun_Sakhare_predicted_labels.txt.")