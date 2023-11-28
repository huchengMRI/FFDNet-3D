# FFDNet-3D
A 3D generalization of FFDNet, based on the codes by Matias Tassano, Julie Delon, and Thomas Veit, An Analysis and Implementation of the FFDNet Image Denoising Method, Image Processing On Line, 9 (2019), pp. 1–25. https://doi.org/10.5201/ipol.2019.231
To train the model, first prepare the training samples (patches) by running prepare_patches.py, which reads the data from folder ./data/rgb/train and ./data/rgb/val/ and generates patches for training and validation.
Then run train_3d.py to train the model and save the model in ./models
Finally one can test the denoising on a NIFTI file by running test_ffdnet_ipol_3d.py.
