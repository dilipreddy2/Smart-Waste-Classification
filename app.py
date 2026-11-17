
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="centered"
)

# ----------------------------------
# 2. WASTE INFORMATION
# ----------------------------------

waste_info = {
    "cardboard": {
        "description": "Cardboard is a recyclable paper-based material.",
        "tip": "Keep cardboard clean and dry before recycling."
    },
    "glass": {
        "description": "Glass is a recyclable material used in bottles and jars.",
        "tip": "Separate glass items and follow local recycling guidelines."
    },
    "metal": {
        "description": "Metal waste includes recyclable cans and metal containers.",
        "tip": "Keep metal containers clean before recycling."
    },
    "paper": {
        "description": "Paper is a commonly recyclable material.",
        "tip": "Keep paper dry and separate from food waste."
    },
    "plastic": {
        "description": "Plastic waste includes bottles and packaging.",
        "tip": "Check local recycling rules for different plastic types."
    },
    "trash": {
        "description": "This category contains general waste that may not fit the other classes.",
        "tip": "Check local waste disposal guidelines before discarding it."
    }
}

# ----------------------------------
# 3. LOAD FINE-TUNED MODEL
# ----------------------------------

@st.cache_resource
def load_waste_model():
    return tf.keras.models.load_model(
        "model/best_waste_model_finetuned.keras"
    )

model = load_waste_model()

# ----------------------------------
# 4. LOAD CLASS NAMES
# ----------------------------------

with open("model/class_names.txt", "r") as file:
    class_names = file.read().splitlines()

# ----------------------------------
# 5. APPLICATION HEADING
# ----------------------------------

st.title("♻️ Smart Waste Classification")

st.write(
    "Upload a waste image to identify its category "
    "using a fine-tuned MobileNetV2 deep learning model."
)

st.divider()

# ----------------------------------
# 6. IMAGE UPLOAD
# ----------------------------------

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Waste Image",
        use_container_width=True
    )

    # ----------------------------------
    # 7. CLASSIFY WASTE
    # ----------------------------------

    if st.button("Classify Waste", type="primary"):

        with st.spinner("Analyzing waste image..."):

            # Resize image
            resized_image = image.resize((224, 224))

            # Convert image to array
            image_array = np.array(resized_image)

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # MobileNetV2 preprocessing
            image_array = preprocess_input(
                image_array.astype(np.float32)
            )

            # Model prediction
            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            # Get highest prediction
            predicted_index = int(np.argmax(predictions))

            predicted_class = class_names[predicted_index]

            confidence = float(
                predictions[predicted_index]
            ) * 100

            # Get top 3 predictions
            top_indices = np.argsort(predictions)[::-1][:3]

        # ----------------------------------
        # 8. DISPLAY RESULT
        # ----------------------------------

        st.divider()

        st.subheader("Classification Result")

        st.success(
            f"Predicted Category: {predicted_class.capitalize()}"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        # Confidence warning
        if confidence < 60:
            st.warning(
                "The model is not very confident about this prediction. "
                "Try uploading a clearer image."
            )

        # ----------------------------------
        # 9. TOP 3 PREDICTIONS
        # ----------------------------------

        st.subheader("Top 3 Predictions")

        for index in top_indices:

            category = class_names[index]

            score = float(predictions[index]) * 100

            st.write(
                f"**{category.capitalize()}** - {score:.2f}%"
            )

            st.progress(float(predictions[index]))

        # ----------------------------------
        # 10. RECYCLING INFORMATION
        # ----------------------------------

        st.divider()

        st.subheader("Recycling Information")

        info = waste_info.get(predicted_class)

        st.write(info["description"])

        st.info("♻️ " + info["tip"])

        # ----------------------------------
        # 11. DISCLAIMER
        # ----------------------------------

        st.caption(
            "Note: This is an AI prediction, not a guaranteed "
            "identification. Check local recycling guidance."
        )

# ----------------------------------
# 12. FOOTER
# ----------------------------------

st.divider()

st.caption(
    "Smart Waste Classification | Powered by Fine-Tuned MobileNetV2"
)
