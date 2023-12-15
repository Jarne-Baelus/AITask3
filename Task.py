import streamlit as st
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import warnings

# Disable warnings
warnings.filterwarnings("ignore")

# Function to display images
def display_images(data_folder, categories, sample_count=5):
    for category in categories:
        folder_path = os.path.join(data_folder, category.replace(' ', '_'))
        st.write(f"## {category} Images")
        class_counts = len(os.listdir(folder_path))

        # Visualize the first few images
        sample_images = os.listdir(folder_path)[:sample_count]
        for img_name in sample_images:
            img_path = os.path.join(folder_path, img_name)
            img = Image.open(img_path)
            st.image(img, caption=f"{category} - {img_name}", use_column_width=True)

# Variables to store results
test_results = None
conf_matrix = None
class_report = None

def train_and_evaluate_model(train_gen, val_gen, test_gen, categories, progress_bar):
    global test_results, conf_matrix, class_report

    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten layer
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(5, activation='softmax'))  # Assuming 5 classes

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    st.write("## Model Summary")
    model.summary()

    st.write("## Training the Model")
    history = model.fit(train_gen, epochs=5, validation_data=val_gen)

    st.write("## Training and Validation Loss Plot")
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot()

    st.write("## Evaluating the Model on Test Set")
    test_results = model.evaluate(test_gen)
    st.write("Test Loss:", test_results[0])
    st.write("Test Accuracy:", test_results[1])

    st.write("## Confusion Matrix")
    test_predictions = model.predict(test_gen)
    test_pred_labels = np.argmax(test_predictions, axis=1)
    test_true_labels = test_gen.classes
    conf_matrix = confusion_matrix(test_true_labels, test_pred_labels)
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot()

    st.write("## Classification Report")
    class_report = classification_report(test_true_labels, test_pred_labels, target_names=categories)
    st.write(class_report)

    # Save the model and history to files
    model.save('saved_model.h5')
    joblib.dump(history.history, 'training_history.joblib')

def load_model_and_history():
    try:
        model = load_model('saved_model.h5')
        history = joblib.load('training_history.joblib')
        return model, history
    except (OSError, IOError) as e:
        return None, None

# Your data and model paths
data_folder = "."
root_folder = "."
train_path = os.path.join(root_folder, "train")
val_path = os.path.join(root_folder, "val")
test_path = os.path.join(root_folder, "test")

# Your categories list
categories = ["Dog", "Cat", "Bird", "Spider", "Elephant"]

# Assuming you have defined categories and
# paths
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Streamlit app
st.title("Image Classification Streamlit App")

# Check if the model is trained
model_trained = st.session_state.get('model_trained', False)

if not model_trained:
    # Button to trigger model training
    if st.button("Train Model"):
        # Train and evaluate the model
        train_and_evaluate_model(train_gen, val_gen, test_gen, categories, st.progress_bar)
        # Set the model_trained flag to True
        st.session_state.model_trained = True

# Load the model and history
model, history = load_model_and_history()

# Dropdown for selecting different parts of the output
output_selection = st.selectbox("Select Output:", ["Model Summary", "Loss Plot", "Evaluation Results", "Confusion Matrix", "Classification Report"])

# Display additional information based on the selected dropdown
if output_selection == "Model Summary":
    st.write("## Model Summary")
    model.summary()
elif output_selection == "Loss Plot":
    st.write("## Training and Validation Loss Plot")
    if history:
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        st.pyplot()
    else:
        st.write("Model has not been trained.")
elif output_selection == "Evaluation Results":
    if test_results:
        st.write("## Evaluating the Model on Test Set")
        st.write("Test Loss:", test_results[0])
        st.write("Test Accuracy:", test_results[1])
    else:
        st.write("Model has not been trained.")
elif output_selection == "Confusion Matrix":
    if conf_matrix is not None:
        st.write("## Confusion Matrix")
        plt.figure(figsize=(8, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot()
    else:
        st.write("Model has not been trained.")
elif output_selection == "Classification Report":
    if class_report:
        st.write("## Classification Report")
        st.write(class_report)
    else:
        st.write("Model has not been trained.")

# Display images
display_images(data_folder, categories)
