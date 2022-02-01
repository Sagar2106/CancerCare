import os
import pydicom
import cv2


# make it True if you want in PNG format
PNG = 1
# Specify the .dcm folder path
folder_path = "D:\\CancerCare\\Lung CT\\The IQ-OTHNCCD lung cancer dataset\\Train cases d"
# Specify the output jpg/png folder path
jpg_folder_path = "D:\\CancerCare\\Lung CT\\The IQ-OTHNCCD lung cancer dataset\\Train Cases j"
images_path = os.listdir(folder_path)
for n, image in enumerate(images_path):
    ds = pydicom.dcmread(os.path.join(folder_path, image))
    pixel_array_numpy = ds.pixel_array
    if PNG == 1:
        image = image.replace('.dcm', '.jpg')
    else:
        image = image.replace('.dcm', '.png')
    cv2.imwrite(os.path.join(jpg_folder_path, image), pixel_array_numpy)
    print('image converted')
