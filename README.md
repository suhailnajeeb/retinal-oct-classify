# Classification of Retinal Diseases from OCT scans using Convolutional Neural Networks

Biomedical image classification for diseases is a lengthy and manual process. However recent progress in computer vision has enabled detection and classification of medical images using machine intelligence a more feasible solution. We explore the possibility of automated detection and classification of retinal abnormalities from retinal OCT scan images of patients. We develop an algorithm to detect the region of interest from a retinal OCT scan and use a computationally inexpensive single layer convolutional neutral network structure for the classification process. Our model is trained on an open source retinal OCT dataset containing 83,484 images of various tunnel disease patients and provides a feasible classification accuracy. 

# Retinal OCT & Diagnosed Eye Diseases

Optical Coherence Tomography (OCT) is an advanced imaging technique which uses coherent light sources to capture images of the optical scattering media up to micrometer resolutions. The process is comparable to ultrasound. Light sources are used instead of sound waves for OCT scans. Optical Coherence Tomography has vast applications in medical imaging. With Retinal OCT scans, the detailed images are revolutionizing early detection and treatment of eye conditions such as -

1. Choroidal NeoVascularization (CNV)
2. Diabetic Macular Edema (DME); and 
3. Drusen etc.

<p align="center">
  <img src="https://raw.githubusercontent.com/suhailnajeeb/retinal-oct-classify/master/figures/fig1.png"><br>
  (a) CNV   (b) DME   (c) DRUSEN
</p>

# Workflow

<p align="center">
  <img src="https://raw.githubusercontent.com/suhailnajeeb/retinal-oct-classify/master/figures/fig2.png">
</p>

## Segmentation

<p align="center">
  <img src="https://raw.githubusercontent.com/suhailnajeeb/retinal-oct-classify/master/figures/fig3.png"><br>
  Region of Interest
</p>

* Border fill-in
* Thresholding: Otsu's Binary Thresholding
* Image Opening
* Image Dilation
* Cropped Out Region of Interest

<p align="center">
  <img src="https://raw.githubusercontent.com/suhailnajeeb/retinal-oct-classify/master/figures/fig4.png"><br>
  (a) Sample image containing redundant white borders (b) Sample image after filling in the white borders with dark pixels (c) Sample image after Otsu’s Binarization (d) Sample image after opening operation (e) Sample image after further dilation (f) Cropped out Region of Interest form sample image
</p>

## CNN Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/suhailnajeeb/retinal-oct-classify/master/figures/fig5.png"><br>
  CNN Architecture
</p>

## Accuracy

<p align="center">
  <img src="https://raw.githubusercontent.com/suhailnajeeb/retinal-oct-classify/master/figures/accuracy.jpg">
</p>

## Confusion Matrix

<p align="center">
  <img src="https://raw.githubusercontent.com/suhailnajeeb/retinal-oct-classify/master/figures/confusion_matrix_without_normalization.png"><br>
  Confusion Matrix
</p>
