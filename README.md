# Trip-Highlight-Extractor

## Streamlit Cloud Deployment

This app uses OpenCV (`cv2`) which requires native system libraries that are not present by default on Streamlit Cloud.

Two files handle this automatically when you deploy to Streamlit Cloud:

- **`packages.txt`** – tells Streamlit Cloud to `apt-get install` the required system libraries (`libgl1` and `libglib2.0-0`) before starting the app. Without these, `import cv2` raises an `ImportError` at runtime.
- **`requirements.txt`** – pins OpenCV to `opencv-python-headless==4.9.0.80`, a stable headless build that works reliably in Linux server environments without a display.