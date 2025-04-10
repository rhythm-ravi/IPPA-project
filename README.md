# IPPA-project
IPPA project
# Image Processing Tool

This project provides a simple **Image Processing Tool** built with **Streamlit** and **OpenCV**. It allows users to apply various image processing techniques to enhance or analyze images. The tool supports multiple operations like smoothing, sharpening, edge detection, corner detection, and filtering.

## Features

- **Grayscale Transformation**: Convert images to grayscale.
- **Smoothing (Gaussian Blur)**: Apply a Gaussian blur to smooth the image.
- **Sharpening**: Enhance edges and details in the image.
- **Edge Detection**: Detect edges using the **Canny edge detection** algorithm.
- **Corner Detection**: Detect corners using the **Harris corner detection** method.
- **Minimum, Maximum, Mean, and Median Filters**: Apply common filters to remove noise and smooth images.

## Requirements

To run the project, you'll need Python 3.x and the following dependencies:

- `streamlit`: For the web interface.
- `opencv-python`: For image processing functions.
- `numpy`: For array manipulations.
- `Pillow`: For image loading and conversion.

You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt
