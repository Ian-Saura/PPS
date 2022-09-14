#! python3
import SimpleITK
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from Utils import swap_labels
#import winshell

############################### CONFIGURATION #####################################
DEBUG = 0 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1

############################## REGISTRATION PARAMETER FILES ######################
similarityMetricForReg = 'NMI'
parameterFilesPath = '..\\Data\\Elastix\\'
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg
paramFileNonRigid = paramFileAffine#'Par0000bspline_500'
############################## AUGMENTATION PARAMETERS ##########################
rotationValues_deg = range(-10, 10+1, 5)
############################### IMAGES AVAILABLE ###################################
atlasNamesImplantOrNotGood = ['7286867', '7291398', '7300327', '7393917', 'L0469978', 'L0483818', 'L0508687', 'L0554842']
dataPath = '..\\Data\\LumbarSpine2D\\Segmented\\' # Base data path.
outputPath = '..\\Data\\LumbarSpine2D\\TrainingSet\\' # Base data path.
outputAugmentedLinearPath = '..\\Data\\LumbarSpine2D\\TrainingSetAugmentedLinear\\' # Base data path.
outputAugmentedNonLinearPath = '..\\Data\\LumbarSpine2D\\TrainingSetAugmentedNonLinear\\' # Base data path.
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
if not os.path.exists(outputAugmentedLinearPath):
    os.makedirs(outputAugmentedLinearPath)
if not os.path.exists(outputAugmentedNonLinearPath):
    os.makedirs(outputAugmentedNonLinearPath)
# Get the atlases names and files:
# Look for the folders or shortcuts:
files = os.listdir(dataPath)
# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
tagInPhase = '_I'
tagLabels = '_labels'
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images
atlasLabelsFilenames = [] # Filenames of the label images
for filename in files:
    name, extension = os.path.splitext(filename)
    # Substract the tagInPhase:
    atlasName = name[:-len(tagInPhase)]
    # Check if filename is the in phase header and the labels exists:
    filenameLabels = dataPath + atlasName + tagLabels + '.' + extensionImages
    if name.endswith(tagInPhase) and extension.endswith(extensionImages) and os.path.exists(filenameLabels) \
            and (atlasName not in atlasNamesImplantOrNotGood):
        # Atlas name:
        atlasNames.append(atlasName)
        # Intensity image:
        atlasImageFilenames.append(filename)
        # Labels image:
        atlasLabelsFilenames.append(filenameLabels)
print("Number of atlases images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))



################################### REFERENCE IMAGE FOR THE REGISTRATION #######################
indexReference = 10
referenceSliceImage = sitk.ReadImage(dataPath + atlasImageFilenames[indexReference])
print('Reference image: {0}. Voxel size: {1}'.format(atlasImageFilenames[indexReference], referenceImage.GetSize()))

################################### READ IMAGES, EXTRACT SLICES AND REGISTER IMAGES TO THE REFERENCE ########################################
for i in range(0, len(atlasNames)):

    ############## 1) READ IMAGE WITH LABELS #############
    # Read target image:
    atlasSliceImage = sitk.ReadImage(dataPath + atlasImageFilenames[i])
    atlasSliceLabel = sitk.ReadImage(atlasLabelsFilenames[i])

    # Cast the image as float:
    atlasSliceImage = sitk.Cast(atlasSliceImage, sitk.sitkFloat32)
    # Rigid registration to match voxel size and FOV.
    ############## 1) RIGID REGISTRATION #############
    # elastixImageFilter filter
    elastixImageFilter = sitk.ElastixImageFilter()
    # Parameter maps:
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                   + paramFileRigid + '.txt'))
    # Registration:
    elastixImageFilter.SetFixedImage(referenceSliceImage)
    elastixImageFilter.SetMovingImage(atlasSliceImage)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    # Get result and apply transform to labels:
    # Get the images:
    atlasSliceImage = elastixImageFilter.GetResultImage()
    # Apply transform:
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetMovingImage(atlasSliceLabel)
    transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
    transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
    transformixImageFilter.Execute()
    atlasSliceLabel = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)
    # write the 2d images:
    sitk.WriteImage(atlasSliceImage, outputPath + atlasNames[i] + '.' + extensionImages)
    sitk.WriteImage(atlasSliceLabel, outputPath + atlasNames[i] + tagLabels + '.' + extensionImages)
    # Show images:
    if DEBUG:
        slice = sitk.GetArrayFromImage(atlasSliceImage)
        labels = sitk.GetArrayFromImage(atlasSliceLabel)
        plt.subplot(1,2,1)
        plt.imshow(slice, cmap='gray')
        plt.imshow(labels, cmap='hot', alpha=0.5)

    ################################### AUGMENTATE WITH REFLECTION AND ROTATION ########################################
    for reflectionX in [1,-1]:
        ############## Reflection ######################
        scale = SimpleITK.ScaleTransform(2, (reflectionX, 1))
        for rotAngle_deg in rotationValues_deg:
            rotation2D = sitk.Euler2DTransform()
            rotation2D.SetAngle(np.deg2rad(rotAngle_deg))
            # Composite transform:
            composite = sitk.Transform(scale)
            composite.AddTransform(rotation2D)
            #scale.SetScale((-1,1))
            # Apply transform:
            atlasSliceImageTransformed = sitk.Resample(atlasSliceImage, composite, sitk.sitkLinear, 0)
            atlasSliceLabelransformed = sitk.Resample(atlasSliceLabel, composite, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            # Change the labels side:
            if reflectionX == -1:
                for l in range (1,5):
                    atlasSliceLabelransformed = swap_labels(atlasSliceLabelransformed, label1=l, label2=l+4)

            # write the 2d images:
            sitk.WriteImage(atlasSliceImageTransformed, outputAugmentedLinearPath + atlasNames[i] + '_refX' + str(reflectionX) + '_rotDeg' + str(rotAngle_deg) +'.' + extensionImages)
            sitk.WriteImage(atlasSliceLabelransformed, outputAugmentedLinearPath + atlasNames[i] + '_refX' + str(reflectionX) + '_rotDeg' + str(rotAngle_deg) +  tagLabels  + '.' + extensionImages)
            # Show images:
            if DEBUG:
                slice = sitk.GetArrayFromImage(atlasSliceImageTransformed)
                labels = sitk.GetArrayFromImage(atlasSliceLabelransformed)
                plt.subplot(1, 2, 2)
                plt.imshow(slice, cmap='gray')
                plt.imshow(labels, cmap='hot', alpha=0.5)
                plt.show()

    ################################### AUGMENTATE WITH NONLINEAR TRANSFORMATIONS ########################################
    for j in range(0, len(atlasNames)):
        # Image to realign to:
        fixedSliceImage = sitk.ReadImage(dataPath + atlasImageFilenames[j])
        ############## NONRIGID REGISTRATION #############
        # elastixImageFilter filter
        elastixImageFilter = sitk.ElastixImageFilter()
        # Parameter maps:
        parameterMapVector = sitk.VectorOfParameterMap()
        parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                       + paramFileNonRigid + '.txt'))
        # Registration:
        elastixImageFilter.SetFixedImage(fixedSliceImage)
        elastixImageFilter.SetMovingImage(atlasSliceImage)
        elastixImageFilter.SetParameterMap(parameterMapVector)
        elastixImageFilter.Execute()
        # Get result and apply transform to labels:
        # Get the images:
        atlasSliceImageDeformed = elastixImageFilter.GetResultImage()
        # Apply transform:
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetMovingImage(atlasSliceLabel)
        transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
        transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
        transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
        transformixImageFilter.Execute()
        atlasSliceLabelDeformed = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)
        # write the 2d images:
        sitk.WriteImage(atlasSliceImageDeformed, outputAugmentedNonLinearPath + atlasNames[i] + '_' + atlasNames[j] + '.' + extensionImages)
        sitk.WriteImage(atlasSliceLabelDeformed, outputAugmentedNonLinearPath + atlasNames[i] + '_' + atlasNames[j] + tagLabels + '.' + extensionImages)
        # Show images:
        # slice = sitk.GetArrayFromImage(atlasSliceImageDeformed)
        # labels = sitk.GetArrayFromImage(atlasSliceLabelDeformed)
        # plt.subplot(1, 2, 2)
        # plt.imshow(slice, cmap='gray')
        # plt.imshow(labels, cmap='hot', alpha=0.5)
        # plt.show()


