#loading libraries
import numpy as np
import keras
from PIL import Image,ImageOps
import streamlit as st
import cv2
from streamlit_option_menu import option_menu

#heading
def main():
    st.title("BRAIN TUMOR DETECTION")
    st.markdown('<span style="font-size: 20px; color: red">UPLOAD THE MRI TO SCAN</span>', unsafe_allow_html=True)
    
    

if __name__== '__main__':
    main()

#loading the model and predicting

def getResult(img,weights_file):
    weights_file = r"C:\\Users\\Nishanth S\\Desktop\\class\\streamlit project mri\\braintumorcat.h5"
    model=keras.models.load_model(weights_file)
    data=np.ndarray(shape=(1, 64, 64, 3), dtype=np.float32)
    image=img
    size=(64,64)
    image=ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array=np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0]=normalized_image_array
    prediction=model.predict(data)
    return np.argmax(prediction)

#masking image of tumor

def mask(img):

    image = Image.open(uploaded_file)
    img = np.array(image)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 0, 150, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 8))
    refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    masked_image = cv2.bitwise_and(img, img, mask=refined_mask)
    return masked_image

#building streamlit

uploaded_file=st.file_uploader("Browse a file ...", type=["jpg"])
if uploaded_file is not None:
    col1,col2=st.columns(2)
    with col1:
        image=Image.open(uploaded_file)
        st.subheader("Uploaded MRI")
        st.image(image,width=250)
        label=getResult(image,"C:\\Users\\Nishanth S\\Downloads\\brain tumor detection\\brain tumor detection\\BrainTumorcategorical.h5")
    if label==0:
        st.markdown('<h2 style="color: red;">THIS MRI SCAN IS HEALTHY</h2>', unsafe_allow_html=True)
        
    else:
        st.markdown('<h2 style="color: red;">TUMOR HAS BEEN DETECTED ON THIS MRI SCAN</h2>', unsafe_allow_html=True)
        st.markdown('<span style="font-size: 20px; color: #">Adjust the number of iterations to get clear and sharpened masked image of the tumor.</span>', unsafe_allow_html=True)
        iterations = st.slider("Number of Iterations", min_value=0, max_value=20, value=4, step=1)
        
        with col2:
            st.subheader("Masked image of Tumor")
            st.image(mask(image),width=250)
        
        