import streamlit as st
import numpy as np
import pickle
from PIL import Image
import requests
from io import BytesIO
import extra_streamlit_components as stx
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json
import plotly.graph_objects as go
import pandas as pd
import time
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import base64

# Set page configuration
st.set_page_config(page_title="Crop Management System", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Background image
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=3200&q=80");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# Load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_plant = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_jR229r.json")

# Load the crop recommendation model
@st.cache_resource
def load_crop_model():
    return pickle.load(open('cropreco', 'rb'))

crop_model = load_crop_model()

# Load the plant disease prediction model and class indices
@st.cache_resource
def load_disease_model_and_classes():
    model = load_model('plant_disease_prediction_model.h5')
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    class_labels = {int(k): v for k, v in class_indices.items()}  # Ensure keys are integers
    return model, class_labels

disease_model, class_labels = load_disease_model_and_classes()

# Crop dictionary
crop_dict = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut', 6: 'papaya', 
    7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon', 11: 'grapes', 
    12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil', 16: 'blackgram', 
    17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas', 20: 'kidneybeans', 
    21: 'chickpea', 22: 'coffee'
}

# Soil type dictionary
soil_types = {
    1: "Loamy Soil: A well-balanced soil type with a mix of sand, silt, and clay. It is fertile and suitable for most crops.",
    2: "Sandy Soil: Coarse-textured soil with high drainage capacity but lower fertility. Suitable for crops like watermelon and groundnuts.",
    3: "Clayey Soil: Fine-textured soil that retains water well but has poor drainage. Suitable for crops like paddy and wheat.",
    4: "Silty Soil: Smooth-textured soil with good fertility, often found near water bodies. Suitable for crops like maize and sugarcane."
}

# Function to get crop benefits
def get_crop_benefits(crop):
    benefits = {
        'rice': "High in carbohydrates, good source of vitamin B",
        'maize': "Rich in fiber, vitamins, and minerals",
        'jute': "Eco-friendly fiber, used in textiles and paper",
        'cotton': "Natural fiber, breathable and comfortable",
        'coconut': "Rich in healthy fats, vitamins, and minerals",
        'papaya': "High in vitamin C and antioxidants",
        'orange': "Excellent source of vitamin C and fiber",
        'apple': "Rich in antioxidants and dietary fiber",
        'muskmelon': "High in vitamins A and C, good for hydration",
        'watermelon': "Low in calories, high in lycopene",
        'grapes': "Rich in antioxidants, good for heart health",
        'mango': "High in vitamin A and C, boosts immunity",
        'banana': "Good source of potassium and vitamin B6",
        'pomegranate': "High in antioxidants, anti-inflammatory properties",
        'lentil': "High in protein and fiber, good for heart health",
        'blackgram': "Rich in protein, fiber, and B vitamins",
        'mungbean': "High in protein and antioxidants",
        'mothbeans': "Rich in protein and dietary fiber",
        'pigeonpeas': "Good source of protein and minerals",
        'kidneybeans': "High in protein and fiber, good for digestion",
        'chickpea': "Rich in protein and fiber, helps in weight management",
        'coffee': "Contains antioxidants, may improve brain function"
    }
    return benefits.get(crop, "Benefits information not available.")

# Function to get seasonal information
def get_seasonal_info(crop):
    seasons = {
        'rice': "Typically grown in warm and humid climates, best planted in spring or early summer.",
        'maize': "Warm-season crop, usually planted in late spring or early summer.",
        'jute': "Grows best in hot and humid climates, typically planted in spring.",
        'cotton': "Warm-season crop, planted in spring when soil temperatures are above 60°F (15°C).",
        'coconut': "Thrives in tropical climates, can be planted year-round in suitable areas.",
        'papaya': "Grows best in tropical and subtropical climates, can be planted year-round in warm areas.",
        'orange': "Best planted in spring after the threat of frost has passed.",
        'apple': "Typically planted in spring or fall, depending on the variety and climate.",
        'muskmelon': "Warm-season crop, planted in spring after the last frost.",
        'watermelon': "Warm-season crop, planted in spring when soil temperatures reach 70°F (21°C).",
        'grapes': "Usually planted in early spring or fall when the vine is dormant.",
        'mango': "Best planted at the beginning of the rainy season in tropical climates.",
        'banana': "Can be planted year-round in tropical climates, spring in subtropical areas.",
        'pomegranate': "Best planted in spring or fall in areas with mild winters.",
        'lentil': "Cool-season crop, typically planted in early spring or fall.",
        'blackgram': "Warm-season crop, usually planted in late spring or early summer.",
        'mungbean': "Warm-season crop, planted in late spring or early summer.",
        'mothbeans': "Drought-resistant crop, usually planted at the beginning of the rainy season.",
        'pigeonpeas': "Warm-season crop, typically planted in late spring or early summer.",
        'kidneybeans': "Warm-season crop, planted in spring after the last frost.",
        'chickpea': "Cool-season crop, planted in fall in warmer climates, spring in cooler areas.",
        'coffee': "Best planted at the beginning of the rainy season in tropical or subtropical climates."
    }
    return seasons.get(crop, "Seasonal information not available.")

# Preprocess the uploaded image
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Custom components
def custom_button(label, key, help=None):
    return st.button(label, key=key, help=help, use_container_width=True)

# Animated progress bar
def progress_bar():
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"Processing: {i+1}%")
        time.sleep(0.01)
    status_text.text("Processing complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()

# Prediction visualization
def plot_prediction(predictions, class_labels):
    df = pd.DataFrame({
        'Class': [class_labels[i] for i in range(len(predictions[0]))],
        'Probability': predictions[0] * 100
    })
    df = df.sort_values('Probability', ascending=False).head(5)
    
    fig = go.Figure(data=[go.Bar(
        x=df['Probability'],
        y=df['Class'],
        orientation='h',
        marker=dict(
            color='rgba(76, 175, 80, 0.6)',
            line=dict(color='rgba(76, 175, 80, 1.0)', width=2)
        )
    )])
    fig.update_layout(
        title='Top 5 Prediction Probabilities',
        xaxis_title='Probability (%)',
        yaxis_title='Class',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

# Information cards
def info_card(title, content):
    st.markdown(f"""
    <div class="card">
        <h3>{title}</h3>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

# Main App
def main():
    st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🌿 Crop Management System</h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="header-description">
        Harness the power of AI to optimize your crop management. 
        Get crop recommendations and identify plant diseases to protect your harvest.
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    selected = option_menu(
        menu_title=None,
        options=["Crop Recommendation", "Disease Prediction", "About", "Help"],
        icons=["plant", "bug", "info-circle", "question-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "rgba(0,0,0,0.5)"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "green"},
        }
    )

    if selected == "Crop Recommendation":
        st.header("Crop Recommendation")
        col1, col2 = st.columns(2)
        with col1:
            N = st.number_input("Nitrogen content in soil", min_value=0, max_value=140, value=50)
            P = st.number_input("Phosphorus content in soil", min_value=5, max_value=145, value=50)
            K = st.number_input("Potassium content in soil", min_value=5, max_value=205, value=50)
            temperature = st.number_input("Temperature (°C)", min_value=8.0, max_value=44.0, value=25.0, step=0.1)
        with col2:
            humidity = st.number_input("Humidity (%)", min_value=14.0, max_value=100.0, value=70.0, step=0.1)
            ph = st.number_input("Soil pH", min_value=3.5, max_value=10.0, value=6.5, step=0.1)
            rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=300.0, value=100.0, step=0.1)
            soil_type = st.selectbox("Soil Type", options=list(soil_types.keys()), format_func=lambda x: soil_types[x])

        if custom_button("Get Crop Recommendation", key="crop_recommend"):
            try:
                features = np.array([[N, P, K, temperature, humidity, ph, rainfall, soil_type]])
                prediction = crop_model.predict(features)
                crop_name = crop_dict[prediction[0]]
                
                st.success(f"The recommended crop is: **{crop_name.capitalize()}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Benefits:**")
                    st.write(get_crop_benefits(crop_name))
                with col2:
                    st.markdown("**Seasonal Information:**")
                    st.write(get_seasonal_info(crop_name))
                
                crop_image_url = f"https://source.unsplash.com/featured/?{crop_name},crop"
                st.image(crop_image_url, caption=f"{crop_name.capitalize()} (Image source: Unsplash)", use_column_width=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please check your input values and try again.")

    elif selected == "Disease Prediction":
        st.header("Plant Disease Prediction")
        uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if custom_button("Predict Disease", key="predict_disease"):
                progress_bar()
                img_array = preprocess_image(image)
                predictions = disease_model.predict(img_array)
                predicted_index = np.argmax(predictions)

                try:
                    predicted_class = class_labels[predicted_index]
                    st.success(f"🌟 Predicted Disease: {predicted_class}")
                    
                    fig = plot_prediction(predictions, class_labels)
                    st.plotly_chart(fig, use_container_width=True)
                except KeyError:
                    st.error(f"Error: Predicted class index {predicted_index} not found in class labels.")
                    st.write("### Debug Information:")
                    st.write(f"Predicted Index: {predicted_index}")
                    st.write("Available Class Labels:")
                    st.json(class_labels)

    elif selected == "About":
        st.header("About the Project")
        st.write("""
        The Advanced Crop Management System is a state-of-the-art AI-powered tool designed to revolutionize 
        agricultural practices. By leveraging machine learning techniques and agricultural expertise, our system 
        provides two key functionalities:

        1. Crop Recommendation: Based on soil composition and environmental factors, the system suggests 
           the most suitable crops for cultivation, optimizing yield and resource utilization.

        2. Plant Disease Prediction: Using computer vision and deep learning, the system can analyze images 
           of plant leaves to identify potential diseases, enabling early intervention and crop protection.

        This integrated approach aims to empower farmers and agricultural professionals with data-driven 
        insights, leading to more sustainable and productive farming practices.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            info_card("Crop Recommendation Model", "Random Forest Classifier trained on extensive agricultural data")
        with col2:
            info_card("Disease Prediction Model", "Convolutional Neural Network (CNN) based on EfficientNet")

        st.subheader("How it Works")
        st.write("""
        Crop Recommendation:
        1. User Input: Farmers input soil composition and environmental data.
        2. Data Processing: The system processes and normalizes the input data.
        3. Model Prediction: Our trained model analyzes the data to recommend suitable crops.
        4. Results Display: The system provides crop recommendations along with benefits and seasonal information.

        Plant Disease Prediction:
        1. Image Upload: Users upload a high-resolution image of a plant leaf.
        2. Preprocessing: The image is resized, normalized, and augmented if necessary.
        3. AI Analysis: Our advanced CNN model analyzes the image, extracting key features.
        4. Disease Prediction: The model predicts the most likely disease based on learned patterns.
        5. Results Visualization: Predictions and confidence levels are displayed in an intuitive format.
        """)

    elif selected == "Help":
        st.header("How to Use")
        st.write("""
        Follow these steps to get the most out of our Advanced Crop Management System:

        For Crop Recommendation:
        1. Navigate to the 'Crop Recommendation' tab.
        2. Input the required soil and environmental data accurately.
        3. Click on the 'Get Crop Recommendation' button.
        4. Review the recommended crop, its benefits, and seasonal information.

        For Plant Disease Prediction:
        1. Go to the 'Disease Prediction' tab.
        2. Ensure you have a clear, well-lit image of the plant leaf you want to analyze.
        3. Click on the 'Choose a leaf image...' button and select your image file.
        4. Once the image is uploaded and displayed, click on the 'Predict Disease' button.
        5. Wait for the AI model to process the image and generate results.
        6. Review the predicted disease and the confidence levels for different classes.

        For best results in disease prediction:
        - Use high-resolution images with good lighting conditions.
        - Ensure the leaf fills most of the frame and is in focus.
        - Include images of both the top and bottom of the leaf if possible.
        - Avoid using blurry, overexposed, or underexposed images.
        - If multiple leaves are affected, submit individual images for each leaf.

        Remember, while our system provides valuable insights, it's always recommended to consult with 
        agricultural experts for comprehensive advice on crop management and disease control.
        """)

        st.subheader("Supported Crops for Recommendation")
        supported_crops = list(crop_dict.values())
        st.write(", ".join(crop.capitalize() for crop in supported_crops))

        st.subheader("Supported Plants for Disease Prediction")
        supported_plants = ["Tomato", "Potato", "Apple", "Cherry", "Corn", "Grape", "Peach", "Pepper", "Strawberry"]
        st.write(", ".join(supported_plants))

    # Footer
    st.markdown("""
    <div class="footer">
        <p>Crop Management System </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

