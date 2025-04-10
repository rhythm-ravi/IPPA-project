# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image

# # Image processing functions
# def apply_smoothing(image):
#     return cv2.GaussianBlur(image, (7, 7), 1.5)

# def apply_sharpening(image):
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5,-1],
#                        [0, -1, 0]])
#     return cv2.filter2D(image, -1, kernel)

# def detect_corners(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     gray = np.float32(gray)
#     dst = cv2.cornerHarris(gray, 2, 3, 0.04)
#     dst = cv2.dilate(dst, None)
#     result = image.copy()
#     result[dst > 0.01 * dst.max()] = [255, 0, 0]
#     return result

# def edge_detection(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     edges = cv2.Canny(gray, 100, 200)
#     return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

# def min_filter(image):
#     return cv2.erode(image, np.ones((3, 3), np.uint8))

# def max_filter(image):
#     return cv2.dilate(image, np.ones((3, 3), np.uint8))

# def mean_filter(image):
#     return cv2.blur(image, (5, 5))

# def median_filter(image):
#     return cv2.medianBlur(image, 5)

# # Operation map
# operations = {
#     "Smoothing": apply_smoothing,
#     "Sharpening": apply_sharpening,
#     "Detection of corners": detect_corners,
#     "Edge Detection": edge_detection,
#     "Minimum filter": min_filter,
#     "Maximum filter": max_filter,
#     "Mean filter": mean_filter,
#     "Median filter": median_filter
# }

# # Streamlit UI
# st.title("🧪 Image Processing Tool")

# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     image_np = np.array(image)

#     st.image(image_np, caption="Original Image", use_column_width=True)

#     operation = st.selectbox("Choose an operation", list(operations.keys()))

#     if st.button("Apply Operation"):
#         processed_image = operations[operation](image_np)
#         st.image(processed_image, caption=f"Result: {operation}", use_column_width=True)


import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Image processing functions
def apply_smoothing(image):
    return cv2.GaussianBlur(image, (7, 7), 1.5)

def apply_sharpening(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    result = image.copy()
    result[dst > 0.01 * dst.max()] = [255, 0, 0]
    return result

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def min_filter(image):
    return cv2.erode(image, np.ones((3, 3), np.uint8))

def max_filter(image):
    return cv2.dilate(image, np.ones((3, 3), np.uint8))

def mean_filter(image):
    return cv2.blur(image, (5, 5))

def median_filter(image):
    return cv2.medianBlur(image, 5)

def grayscale_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# Operation map
operations = {
    "Grayscale Transformation": grayscale_transform,
    "Smoothing": apply_smoothing,
    "Sharpening": apply_sharpening,
    "Detection of corners": detect_corners,
    "Edge Detection": edge_detection,
    "Minimum filter": min_filter,
    "Maximum filter": max_filter,
    "Mean filter": mean_filter,
    "Median filter": median_filter
}

# Streamlit UI
st.title(f"🧪 Image Processing Tool\n - Rhythm Ravi")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image_np, caption="Original Image", use_column_width=True)

    operation = st.selectbox("Choose an operation", list(operations.keys()))

    if st.button("Apply Operation"):
        processed_image = operations[operation](image_np)
        st.image(processed_image, caption=f"Result: {operation}", use_column_width=True)
