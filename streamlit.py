import streamlit as st
from backend import chatbot_response

# Streamlit UI Setup
st.title("Fashion Assistant 🛍️")
st.write("Get fashion recommendations, analyze clothing images, and more!")

# Input Selection
input_type = st.radio("Choose input type:", ["Text Query", "Image Upload"])

if input_type == "Text Query":
    user_input = st.text_area("Enter your fashion-related question:")
    if st.button("Ask"):  # Process user query
        if user_input.strip():
            response = chatbot_response(user_input)
            st.write("**Response:**", response)
        else:
            st.warning("Please enter a question!")

elif input_type == "Image Upload":
    uploaded_image = st.file_uploader("Upload a clothing image", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Convert image to bytes
        image_bytes = uploaded_image.read()
        image_input = f"image:{uploaded_image.name}"  # Pass image filename to chatbot_response
        
        # Get clothing features and recommendations
        response = chatbot_response(image_input)
        st.write("**Recommendation:**", response)
