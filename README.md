
# Lumbar Spine Muscle Segmentation from MRI

This repository contains code for a web application that performs lumbar spine muscle segmentation from MRI images. The segmentation is based on a U-NET neural network, and the application allows users to upload their MRI images for processing.

## Features

- Upload .raw and .mhd files for processing.
- Utilizes a pre-trained U-NET model for image segmentation.
- Sends processed images by email.
- Provides information about the segmentation process.

## Prerequisites

- Python 3
- Libraries: SimpleITK, NumPy, Flask, Flask-Mail, PyTorch

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/LumbarSpineSegmentation.git
   cd LumbarSpineSegmentation
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python app.py
   ```

4. Visit `http://127.0.0.1:5000/` in your web browser.

## Configuration

- Set the necessary configurations in the `app.py` file, including mail server details and data paths.

## Folder Structure

- `CNNs`: Contains the U-NET model and utility functions.
- `templates`: HTML templates for the web application.
- `uploads_folder`: Temporary folder to store uploaded files.

## About LaPIM

The repository is part of the Laboratorio de Procesamiento de Imágenes médicas (LaPIM), which belongs to the CEMSC3 and University Center for Medical Imaging (CEUNIM) of UNSAM. LaPIM focuses on researching medical image processing methods and their translation to clinical and basic science research.

## Credits

LaPIM (Laboratorio de Procesamiento de Imágenes médicas)

## Contact

For more information or inquiries, please contact 

---

Feel free to customize it further based on your preferences and specific details about the project.
