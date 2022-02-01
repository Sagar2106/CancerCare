# CancerCare
----------------------------------------------------------------------------------
Bachelors Final Year Project - Lung Cancer Detection System by classifying CT Scan Images using kNN and GLCM
Conducted by Sagar Sonawane, Aayush Kamdar, and Vihaan Sharma. 
----------------------------------------------------------------------------------
A lung cancer detection system made by using various Image Processing (Gabor Filter, Otsu's Thresholding) and Machine Learning(GLCM, kNN Classification) methods by using the CT scan images in the IQ-OTHNCCD lung cancer dataset.

The CT images were first converted into jpg images then were enhanced using Gabor Filter and Otsu's Thresholding which were then run by a feature extraction process(GLCM) and several features such as energy, entropy, correlation, homogeniety, dissimilarity, and ASM were stored in a csv file. This csv file was used as test and train input for kNN classification. The classifications made by kNN classifier based on the mentioned csv file resulted in a 99.11% training accuracy and a test accuracy of 92.37%.

This project was presented as a paper to the 2nd International Conference on Innovations in Computational Intelligence and Computer Vision (ICICV-2021), Springer, Manipal University, Jaipur. 
