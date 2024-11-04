import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Load embeddings and labels
print("Loading embeddings and labels...")
with open('models/embeddings.pkl', 'rb') as f:
    X_train, y_train = pickle.load(f)
print("Embeddings and labels loaded.")

# Encode the labels
print("Encoding labels...")
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
print("Labels encoded.")

# Train the SVM classifier
print("Training SVM classifier...")
classifier = SVC(kernel='linear', probability=True)
classifier.fit(X_train, y_train_encoded)
print("SVM classifier trained.")

# Save the trained classifier and label encoder
print("Saving classifier and label encoder...")
with open('models/svm_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
print("Classifier and label encoder saved successfully.")
