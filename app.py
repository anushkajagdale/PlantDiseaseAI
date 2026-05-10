import streamlit as st
from PIL import Image
import os
from utils import predict_image, TF_AVAILABLE
import tempfile

# Page configuration
st.set_page_config(
    page_title="🌿 Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar information
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Plant Disease Detection System** uses deep learning to classify plant leaf images as:
    - 🟢 **Healthy** - No disease detected
    - 🔴 **Diseased** - Disease detected
    
    **Model**: MobileNetV2 Transfer Learning
    **Classes**: Binary Classification (Healthy/Diseased)
    """)
    
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. Upload a clear image of a plant leaf
    2. Supported formats: JPG, JPEG, PNG
    3. The model will analyze and classify the leaf
    4. View confidence score and results
    """)

# Main content
st.title("🌿 Plant Disease Detection System")
st.write("Upload a leaf image, and the model will classify it as Healthy or Diseased.")

# Show warning if running in demo mode
if not TF_AVAILABLE:
    st.warning(
        "⚠️ **Demo Mode**: The ML model is not available locally (TensorFlow not installed). "
        "Predictions shown are random for demo purposes only. "
        "The full AI model will work when deployed to Render or when TensorFlow is installed.",
        icon="🧪"
    )

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

        # Save temporarily and predict
        temp_path = None
        try:
            # Try creating temp file with explicit permissions
            import tempfile as tf_module
            with tf_module.NamedTemporaryFile(delete=False, suffix=".jpg", mode='wb') as tmp_file:
                img.save(tmp_file, format='JPEG')
                temp_path = tmp_file.name
            
            # Verify temp file was created
            if not os.path.exists(temp_path):
                raise RuntimeError(f"Failed to create temporary file at {temp_path}")
            
            if os.path.getsize(temp_path) == 0:
                raise RuntimeError("Temporary file is empty - image save failed")

            # Make prediction
            label, confidence = predict_image(temp_path)

            # Display results
            st.success(f"✅ Prediction: **{label}**")
            
            # Create progress bar for confidence
            st.metric("Confidence Score", f"{confidence * 100:.2f}%")
            
            # Add visual feedback
            if label == "Healthy":
                st.balloons()
                st.success("This leaf appears to be healthy! 🌱")
            else:
                st.warning("Disease detected. Consider treatment. 🏥")
                
        except PermissionError as pe:
            st.error(f"❌ Permission Error: Cannot write to temporary storage. {str(pe)}")
            st.info("This is a server configuration issue. Contact support.")
        except RuntimeError as re:
            st.error(f"❌ File System Error: {str(re)}")
            st.info("Temporary file creation failed. Please try again.")
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as cleanup_error:
                    st.warning(f"⚠️ Could not clean up temp file: {cleanup_error}")
                
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.info("Please try with another image or check if the model is properly loaded.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<p style='color: gray; font-size: 12px;'>
Plant Disease AI v1.0 | Powered by TensorFlow & Streamlit
</p>
</div>
""", unsafe_allow_html=True)
