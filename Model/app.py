import streamlit as st
import pickle
from PIL import Image
import numpy as np

# Loading the models and classification reports from the model.pkl file
svm_classification_report, knn_classification_report, dt_classification_report, svm_model, knn_model, dt_model = pickle.load(open('./model.pkl', 'rb'))

st.title('Image Classification with Different Models')


uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])


if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)

    # Preprocessing  the image for model prediction
    image = Image.open(uploaded_image)
    image = image.resize((15, 15))
    image_array = np.array(image).flatten()


    # Reshaping the flattened array to have a shape (1, -1)
    image_array_reshaped = image_array.reshape(1, -1)

    # Unpacking  the models and make predictions
    svm_prediction = svm_model.predict(image_array_reshaped)
    knn_prediction = knn_model.predict(image_array_reshaped)
    dt_prediction = dt_model.predict(image_array_reshaped)

    # Printing the predictions for debugging
    st.text(f'SVM Prediction: {svm_prediction}')
    st.text(f'KNN Prediction: {knn_prediction}')
    st.text(f'Decision Tree Prediction: {dt_prediction}')

    # Displaying  the final predictions
    st.success(f'SVM Prediction: {svm_prediction[0]}')
    st.success(f'KNN Prediction: {knn_prediction[0]}')
    st.success(f'Decision Tree Prediction: {dt_prediction[0]}')