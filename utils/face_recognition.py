import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score, classification_report

# Path for the face recognition model and data
FACE_RECOGNITION_MODEL_PATH = os.path.join("models", "face_recognition_model")
# Function to encode faces
def encode_faces(faces, model_name='default'):
    encoded_faces = []
    for face in faces:
        face = cv2.resize(face, (128, 128))
        face = face / 255.0
        face = face.reshape(1, -1)  # Flatten
        encoded_faces.append(face)
    return np.vstack(encoded_faces)

def train_face_recognition_model(known_faces_dir):
    # Collect face images and labels
    labels = []
    face_embeddings = []
    for person in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                 image_path = os.path.join(person_dir, image_name)
                 img = cv2.imread(image_path)
                 if img is None:
                     print(f"Error reading image: {image_path}")
                     continue
                 faces_info = [(0, 0, img.shape[1], img.shape[0])] #Assume face occupies the entire image
                 encoded_face = encode_faces([img])
                 face_embeddings.append(encoded_face)
                 labels.append(person)
    
    if not labels:
        print("No faces for training were found. Please check the images in the faces folder.")
        return None
    
    face_embeddings = np.vstack(face_embeddings)
    # Encode the labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    # Train the SVM classifier
    X_train, X_test, y_train, y_test = train_test_split(face_embeddings, labels, test_size=0.2, random_state=42)

    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {accuracy*100:0.2f}%")
    print(classification_report(y_test, y_pred))
    # Save trained model and label encoder
    model_path = os.path.join(FACE_RECOGNITION_MODEL_PATH, "svm_model.pkl")
    encoder_path = os.path.join(FACE_RECOGNITION_MODEL_PATH, "label_encoder.pkl")
    os.makedirs(FACE_RECOGNITION_MODEL_PATH, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(svm_model, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Face recognition model and encoder have been saved in the folder: {FACE_RECOGNITION_MODEL_PATH}")
    return  svm_model, label_encoder

def load_face_recognition_model():
    model_path = os.path.join(FACE_RECOGNITION_MODEL_PATH, "svm_model.pkl")
    encoder_path = os.path.join(FACE_RECOGNITION_MODEL_PATH, "label_encoder.pkl")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print("Loaded pre-trained model and encoder")
        return model, label_encoder
    except FileNotFoundError:
        print("Pre-trained face recognition model and/or encoder not found. Please train first.")
        return None, None

def recognize_faces(img, faces_info):
    # Load model and encoder
    model, label_encoder = load_face_recognition_model()
    if model is None or label_encoder is None:
        print("Face recognition model not available, defaulting to unknown")
        return [((x, y, w, h), "Unknown") for x, y, w, h in faces_info] # Default to Unknown
    # Extract faces from original image
    faces = [img[y:y+h, x:x+w] for (x, y, w, h) in faces_info]
    # Encode the face embeddings
    if not faces:
        print("No faces were found for recognition.")
        return []
    
    face_embeddings = encode_faces(faces)
    
    # Predictions
    predictions = model.predict(face_embeddings)
    probabilities = model.predict_proba(face_embeddings)

    recognized_faces = []
    for i, (x, y, w, h) in enumerate(faces_info):
       
       predicted_label = label_encoder.inverse_transform([predictions[i]])[0]
       recognized_faces.append(((x,y,w,h), predicted_label))
    return recognized_faces

if __name__ == '__main__':
    # Example Usage for Training Model
    known_faces_dir = "data/faces"
    if not os.path.exists(known_faces_dir):
        print(f"No directory named {known_faces_dir} was found, creating it.")
        os.makedirs(known_faces_dir)
        print("Please add directories with face images to the path")
        exit()

    # Train or load model.
    if os.path.exists(FACE_RECOGNITION_MODEL_PATH):
        model, label_encoder = load_face_recognition_model()
        if model is None or label_encoder is None:
           model, label_encoder = train_face_recognition_model(known_faces_dir)
    else:
        model, label_encoder = train_face_recognition_model(known_faces_dir)
