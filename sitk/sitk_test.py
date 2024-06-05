from PIL import Image
import numpy as np
import SimpleITK as sitk

# Load the 4D TIFF image using PIL
tiff_image = Image.open('A:\Git Repos\worm-warp\stacks\exp240202_01_E.tif')

# Extract all frames (timepoints)
image_sequence = []
try:
    while True:
        image_sequence.append(np.array(tiff_image))
        tiff_image.seek(tiff_image.tell() + 1)
except EOFError:
    pass

# Convert list of 3D arrays to a 4D numpy array
image_stack_4d = np.stack(image_sequence, axis=0)

# Convert the numpy 4D array to a SimpleITK image
sitk_image_stack_4d = sitk.GetImageFromArray(image_stack_4d)

def extract_3d_timepoint(image, timepoint):
    size = list(image.GetSize())
    size[0] = 1  # Extract only one timepoint
    index = [timepoint, 0, 0, 0]  # Start at the specified timepoint
    return sitk.Extract(image, size, index)

timepoints = [sitk.Extract(sitk_image_stack_4d, list(sitk_image_stack_4d.GetSize())[:-1] + [0], [0, 0, 0, i]) for i in range(sitk_image_stack_4d.GetSize()[3])]

def register_images(fixed_image, moving_image):
    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings
    registration_method.SetMetricAsMeanSquares()

    # Interpolator settings
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Transformation settings
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Execute the registration
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32))

    return final_transform

reference_image = timepoints[0]
registered_timepoints = [reference_image]

for i in range(1, len(timepoints)):
    moving_image = timepoints[i]
    transform = register_images(reference_image, moving_image)
    registered_image = sitk.Resample(moving_image, reference_image, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    registered_timepoints.append(registered_image)

def combine_into_4d_image(image_list):
    # Convert to numpy arrays and stack
    np_images = [sitk.GetArrayFromImage(img) for img in image_list]
    np_4d_image = np.stack(np_images, axis=0)

    # Convert back to SimpleITK image
    sitk_4d_image = sitk.GetImageFromArray(np_4d_image)
    return sitk_4d_image

# Combine the registered timepoints into a 4D image
registered_image_stack_4d = combine_into_4d_image(registered_timepoints)

sitk.WriteImage(registered_image_stack_4d, 'registered_image_stack_4d.tiff')
