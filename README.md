<h2>Tensorflow-Tiled-Image-Segmentation-BreCaHAD (2024/08/06)</h2>

This is the first experiment of Tiled Image Segmentation for BreCaHAD:A Dataset for Breast Cancer Histopathological Annotation and Diagnosis based on
the <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/1ppmWc-AxMFZeiu66_GwTFJ5lgHc-y3jz/view?usp=sharing">
Tiled-BreCaHAD-ImageMask-Dataset</a>, which was derived by us from the original 
dataset 
<a href="https://figshare.com/articles/dataset/BreCaHAD_A_Dataset_for_Breast_Cancer_Histopathological_Annotation_and_Diagnosis/7379186">
BreCaHAD: A Dataset for Breast Cancer Histopathological Annotation and Diagnosis
</a>
<br>
<br>
Please see also <a href="https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-BreCaHAD">Tiled-ImageMask-Dataset-BreCaHAD</a>
<br>
<br> 
In this experiment, we employed the following strategy:
<b>
<br>
1. We trained and validated a TensorFlow UNet model using the Tiled-BreCaHAD-ImageMask-Dataset, which was tiledly-splitted to 512x512
 and reduced to 512x512 image and mask dataset.<br>
2. We applied the Tiled-Image Segmentation inference method to predict the tumor regions for the mini_test images 
with a resolution of 1360x1024 pixels. 
<br><br>
</b>  
Please note that Tiled-BreCaHAD-ImageMask contains two types of images and masks:<br>
1. Tiledly-splitted to 512x512 image and mask files.<br>
2. Size-reduced to 512x512 image and mask files.<br>
Namely, this is a mixed set of Tiled and Non-Tiled ImageMask Datasets.

<hr>
<b>Actual Image Segmentation for Images of 1360x1024 pixels</b><br>
As shown below, the tiled inferred masks look somewhat similar to the ground truth masks, but they lack precision in certain areas.
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: tiled inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/images/10011.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/masks/10011.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test_output_tiled/10011.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/images/10021.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/masks/10021.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test_output_tiled/10021.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/images/10024.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/masks/10024.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test_output_tiled/10024.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we have used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Oral Cancer Segmentation.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The original BreCaHAD dataset used here has been taken from the following figshare web-site;<br>
<a href="https://figshare.com/articles/dataset/BreCaHAD_A_Dataset_for_Breast_Cancer_Histopathological_Annotation_and_Diagnosis/7379186">
BreCaHAD: A Dataset for Breast Cancer Histopathological Annotation and Diagnosis
</a>
<br>
<br>

Aksac, Alper; Demetrick, Douglas J.; Özyer, Tansel; Alhajj, Reda (2019). <br>
BreCaHAD: A Dataset for Breast Cancer Histopathological Annotation and Diagnosis. <br>
figshare. Dataset. https://doi.org/10.6084/m9.figshare.7379186.v3
<br>
DOI:https://doi.org/10.6084/m9.figshare.7379186.v3<br>
<br>
<b>Version 3</b><br>

Dataset posted on 2019-01-28, 23:22 <br>
authored by Alper Aksac, Douglas J. Demetrick, Tansel Özyer, Reda Alhajj<br>
<br>
<b>License</b>: CC BY 4.0<br><br>

<h3>
<a id="2">
2 Tiled-BreCaHAD ImageMask Dataset
</a>
</h3>
 If you would like to train this Tiled-BreCaHAD Segmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/1ppmWc-AxMFZeiu66_GwTFJ5lgHc-y3jz/view?usp=sharing">
Tiled-BreCaHAD-ImageMask-Dataset</a>,
<br>
Please expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Tiled-BreCaHAD
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>

<b>Tiled-BreCaHAD Dataset Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/Tiled-BreCaHAD-ImageMask-Dataset-M1_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is enough large to use for a training set for our segmentation model. 
<br>

<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained Tiled-BreCaHAD TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/Tiled-BreCaHAD and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>

<pre>
; train_eval_infer.config
; 2024/08/05 (C) antillia.com

[model]
model          = "TensorflowUNet"
generator      = False
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = False
normalization  = False
num_classes    = 1
base_filters   = 16
base_kernels   = (7,7)
num_layers     = 8
dropout_rate   = 0.05
learning_rate  = 0.0001
clipvalue      = 0.5
dilation       = (1,1)
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
show_summary   = False

[train]
epochs        = 100
batch_size    = 2
steps_per_epoch  = 200
validation_steps = 80
patience      = 10
metrics       = ["dice_coef", "val_dice_coef"]

model_dir     = "./models"
eval_dir      = "./eval"
image_datapath = "../../../dataset/Tiled-BreCaHAD/train/images/"
mask_datapath  = "../../../dataset/Tiled-BreCaHAD/train/masks/"

epoch_change_infer     = True
epoch_change_infer_dir = "./epoch_change_infer"
epoch_change_tiledinfer     = True
epoch_change_tiledinfer_dir = "./epoch_change_tiledinfer"
num_infer_images       = 1

create_backup  = False

learning_rate_reducer = True
reducer_factor     = 0.3
reducer_patience   = 5
save_weights_only  = True

[eval]
image_datapath = "../../../dataset/Tiled-BreCaHAD/valid/images/"
mask_datapath  = "../../../dataset/Tiled-BreCaHAD/valid/masks/"

[test] 
image_datapath = "../../../dataset/Tiled-BreCaHAD/test/images/"
mask_datapath  = "../../../dataset/Tiled-BreCaHAD/test/masks/"

[infer] 
images_dir    = "./mini_test/images"
output_dir    = "./mini_test_output"

[tiledinfer] 
overlapping   = 64
images_dir    = "./mini_test/images"
output_dir    = "./mini_test_tiled_output"

[segmentation]
colorize      = True
black         = "black"
white         = "green"
blursize      = None

[image]
;color_converter = None
color_converter = "cv2.COLOR_BGR2HSV_FULL"
gamma           = 0

[mask]
blur      = False
blur_size = (3,3)
binarize  = False
;threshold = 128
threshold = 80

[generator]
debug        = False
augmentation = True

[augmentor]
vflip    = True
hflip    = True
rotation = True
angles   = [30, 60, 90, 120, 180, 210, 240, 270, 300,330]
shrinks  = [0.8]
shears   = [0.1]

deformation = True
distortion  = True
sharpening  = False
brightening = False
barrdistortion = True

[deformation]
alpah     = 1300
sigmoids  = [8.0,]

[distortion]
gaussian_filter_rsigma= 40
gaussian_filter_sigma = 0.5
distortions           = [0.02, 0.03]

[barrdistortion]
radius = 0.3
amount = 0.3
centers =  [(0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.7, 0.7)]

[sharpening]
k        = 1.0

[brightening]
alpha  = 1.2
beta   = 10  
</pre>
<hr>
<b>Model parameters</b><br>
Defined small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16 
base_kernels   = (7,7)
num_layers     = 8
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback. 
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.3
reducer_patience   = 5
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer and epoch_change_tiledinfer callbacks.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_change_tiledinfer  = True
epoch_change_tiledinfer_dir = "./epoch_change_tiledinfer"
num_infer_images         = 1
</pre>

By using these callbacks, on every epoch_change, the inference procedures can be called
 for an image in <b>mini_test</b> folder. These will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/asset/epoch_change_infer.png" width="1024" height="auto"><br>
<br>
<br>
<b>Epoch_change_tiled_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/asset/epoch_change_tiledinfer.png" width="1024" height="auto"><br>
<br>
<br>
In this experiment, we manually terminated the training process at epoch 34.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/asset/train_console_output_at_epoch_34.png" width="720" height="auto"><br>
<br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Tiled-BreCaHAD.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/asset/evaluate_console_output_at_epoch_34.png" width="720" height="auto">
<br><br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) score for this test dataset is not low, and dice_coef not high as shown below.<br>
<pre>
loss,0.2354
dice_coef,0.713
</pre>


<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Tiled-BreCaHAD.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>


<h3>
6 Tiled Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Tiled-BreCaHAD.<br>
<pre>
./4.tiled_infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetTiledInferencer.py ./train_eval_infer_aug.config
</pre>

<hr>
<b>Tiled inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/asset/mini_test_output_tiledinfer.png" width="1024" height="auto"><br>
<br>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Tiled-inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/images/10013.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/masks/10013.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test_output_tiled/10013.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/images/10021.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/masks/10021.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test_output_tiled/10021.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/images/10023.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/masks/10023.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test_output_tiled/10023.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/images/10031.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/masks/10031.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test_output_tiled/10031.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/images/10035.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/masks/10035.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test_output_tiled/10035.jpg" width="320" height="auto"></td>
</tr>
</table>

<br>
<br>
<!--
  -->
<b>Comparison of Non-tiled inferred mask and Tiled-Inferred mask</b><br>
As shown below, the tiled inferencer provides clearer and better results than non-tiled inferencer.<br>
<br>
<table>
<tr>
<th>Mask (ground_truth)</th>

<th>Non-tiled-inferred-mask</th>
<th>Tiled-inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/masks/10013.jpg" width="320" height="auto"></td>

<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test_output/10013.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test_output_tiled/10013.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/masks/10022.jpg" width="320" height="auto"></td>

<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test_output/10022.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test_output_tiled/10022.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test/masks/10040.jpg" width="320" height="auto"></td>

<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test_output/10040.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-BreCaHAD/mini_test_output_tiled/10040.jpg" width="320" height="auto"></td>
</tr>
</table>
<br>

<h3>
Reference
</h3>

<b>1. BreCaHAD: A Dataset for Breast Cancer Histopathological Annotation and Diagnosis</b><br>
Aksac, Alper; Demetrick, Douglas J.; Özyer, Tansel; Alhajj, Reda (2019). <br>
BreCaHAD: A Dataset for Breast Cancer Histopathological Annotation and Diagnosis. <br>
figshare. Dataset. https://doi.org/10.6084/m9.figshare.7379186.v3<br>
<a href="https://figshare.com/articles/dataset/BreCaHAD_A_Dataset_for_Breast_Cancer_Histopathological_Annotation_and_Diagnosis/7379186">
https://figshare.com/articles/dataset/BreCaHAD_A_Dataset_for_Breast_Cancer_Histopathological_Annotation_and_Diagnosis/7379186
</a><br>
<br>
<b>2. BreCaHAD: a dataset for breast cancer histopathological annotation and diagnosis</b><br>
Alper Aksac1, Douglas J. Demetrick, Tansel Ozyer and Reda Alhajj<br>
<a href="https://bmcresnotes.biomedcentral.com/counter/pdf/10.1186/s13104-019-4121-7.pdf">
https://bmcresnotes.biomedcentral.com/counter/pdf/10.1186/s13104-019-4121-7.pdf</a>

<br>
<br>
<b>3. Tiled-ImageMask-Dataset-BreCaHAD</b><br>
Toshiyuki Arai antillia.com<br>
<a href="https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-BreCaHAD">
https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-BreCaHAD
</a>
<br>



