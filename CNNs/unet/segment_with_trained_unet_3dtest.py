import SimpleITK as sitk
import numpy as np
import os
from unet_3d import Unet
import torch
from utils import writeMhd
from utils import multilabel
from utils import maxProb
from utils import FilterUnconnectedRegions
from flask import Flask, render_template, request, redirect, url_for
import os
import time  # Import the time module

app = Flask(__name__, template_folder='templates')  # Set the template folder

# Configura la ubicación para guardar archivos cargados temporalmente
app.config['UPLOAD_FOLDER'] = 'C:/Users/DELL/MuscleSegmentation/CNNs/unet/uploads_folder/'

# Resto de tu código...


############################ DATA PATHS ##############################################
dataPath = '../../Data/LumbarSpine3D/InputImages/'
outputPath = '../../Data/LumbarSpine3D/InputImages/'
modelLocation = '../../Data/LumbarSpine3D/PretrainedModel/'
# Image format extension:
extensionImages = 'mhd'

if not os.path.exists(dataPath):
    os.makedirs(dataPath)

modelName = os.listdir(modelLocation)[0]
modelFilename = modelLocation + modelName

######################### CHECK DEVICE ######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device.type == 'cuda':
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('Total memory: {0}. Reserved memory: {1}. Allocated memory:{2}. Free memory:{3}.'.format(t,r,a,f))

######################### MODEL INIT ######################
multilabelNum = 8
torch.cuda.empty_cache()
model = Unet(1, multilabelNum)
model.load_state_dict(torch.load(modelFilename, map_location=device))
model = model.to(device)

###################### READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS #####################################################
# Look for the folders or shortcuts:
imageNames = []
imageFilenames = []
i = 0

    # ... (previous code)

# Route to render the upload form
@app.route('/')
def upload_form():
    return render_template('upload.html')



# Route to process the uploaded image
@app.route('/process', methods=['POST'])
def process_image():
    if 'raw_file' not in request.files or 'mhd_file' not in request.files:
        return redirect(request.url)

    raw_file = request.files['raw_file']
    mhd_file = request.files['mhd_file']

    # Check if either file is empty
    if raw_file.filename == '' or mhd_file.filename == '':
        return redirect(request.url)
    if raw_file:
    # If the uploaded file is a .raw file, pass it directly to writeMhd
        raw_file_path = os.path.join(app.config['UPLOAD_FOLDER'], raw_file.filename)
        raw_file.save(raw_file_path)
    
    if mhd_file:
        # Save the .mhd file temporarily
        mhd_file_path = os.path.join(app.config['UPLOAD_FOLDER'], mhd_file.filename)
        mhd_file.save(mhd_file_path)
        processed_file_path = mhd_file_path
        # Process the .mhd file
        sitkImage = sitk.ReadImage(mhd_file_path)
        image = sitk.GetArrayFromImage(sitkImage).astype(np.float32)
        image = np.expand_dims(image, axis=0)

        with torch.no_grad():
            input = torch.from_numpy(image).to(device)
            output = model(input.unsqueeze(0))
            output = torch.sigmoid(output.cpu().to(torch.float32))
            outputs = maxProb(output, multilabelNum)
            output = ((output > 0.5) * 1)
            output = multilabel(output.detach().numpy())

            # Now, you can process the image as intended and save the result
            # Replace 'outputPath' with the appropriate path
            writeMhd(output.squeeze(0).astype(np.uint8), outputPath + 'processed_image.mhd', sitkImage)


    return f"Image processed successfully! <a href='{processed_file_path}' download>Download Processed File</a>"


if __name__ == '__main__':
    app.run(debug=True, port=8080)
