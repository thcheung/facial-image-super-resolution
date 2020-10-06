This readme file is the description of the folder "FYP"

--------------------Files Description--------------------

"datasets":                   datasets of facial images

"docs":                       presentation slides, interim report and final report

"gui":                        the GUI for demonstration
      "gui.py":               the GUI script using python
      
"src":
      "deep-learning":        source codes for CNN and GAN in python
      "machine-learning":     source codes for PCA, LLE, SR in C++
      "mahcine-learning-gui": source codes for PCA, LLE, SR into a single exe for demo
      
"tools":
      "crop.py":              cropped the images to another size
      "down.py":              downsampling the images
      "norm.py":              face detection and alignment
      "up.py"  :              upscaling the images using bicubic interpolation
      
"workspace":                  the programs used for super-resolution
      "pca.exe":              SR using PCA
      "lle.exe":              SR using LLE
      "ssr.exe":              SR using SR
      "cnn.py" :              SR using CNN
      "gan.py" :              SR using GAN
      "measure.m":            measure the PSNR and SSIM
      "models" :              the trainned models for CNN and GAN

--------------------Programs Description--------------------

0. Direct to the folder "workspace"
1. Place the HR trainning samples into folder "sample_hr"
2. Place the LR trainning samples into folder "sample_lr"
3. Place the LR input samples into folder "test_lr"
4. Run either "pca.exe", "lle.exe", "ssr.exe", "cnn.py" and "gan.py"
5. For CNN and GAN, copy the model from "models" to "logs" and rename it as "test.pth"
6. The results are outputted to the folders: "test_pca", "test_lle", "test_ssr", "test_cnn", "test.gan"
7. run measure.m in MATLAB to measure the PSNR and SSIM of the super-resolved images

--------------------GUI Description--------------------

0. Direct to the folder "gui"
1. Run "gui.py"
2. click 'load' to load the LR image as the input
3. Select the parameters including upscaling factor, dataset and method
4. click 'start' to run the SR
5. click 'save' to save the SR image
