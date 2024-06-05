import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
import SimpleITK as sitk
import tifffile as tiff
import datetime
import os

def create_dicom_file(filename, numpy_array):
    # Create a DICOM dataset
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.ImplicitVRLittleEndian
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Add some metadata
    ds.PatientName = "Test^Firstname"
    ds.PatientID = "123456"
    ds.Modality = "CT"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.SOPClassUID = pydicom.uid.CTImageStorage

    ds.is_little_endian = True
    ds.is_implicit_VR = True

    # Set the pixel data
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.NumberOfFrames, ds.Rows, ds.Columns = numpy_array.shape
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelData = numpy_array.tobytes()

    # Set the file meta information
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    # Save the DICOM file
    ds.save_as(filename)
    print(f"File saved as {filename}")

def save_numpy_as_itk(filename, numpy_array, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    # Convert numpy array to SimpleITK image
    itk_image = sitk.GetImageFromArray(numpy_array)
    
    # Set spacing (voxel size) and origin (image position)
    itk_image.SetSpacing(spacing)
    itk_image.SetOrigin(origin)

    # Save the image
    sitk.WriteImage(itk_image, filename)
    print(f"File saved as {filename}")

# Convert numpy array to DICOM
#data = np.load('output/pvd_test.npy')  # Load your 3D neuron data
data = tiff.imread('output/thresh_stack.tif')
filename = "output/pvd_test.nii"
save_numpy_as_itk(filename, data)
