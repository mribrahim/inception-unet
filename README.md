# Improved U-Nets with Inception Blocks for Building Detection
This repository consists Inception Unet versions of classical Unet architecture for image segmentation. 
In the paper, a new deep learning architecture has been developed by combining inception blocks with the convolutional layers 
of the original U-Net architecture to achieve remarkably high performance in building detection. 

You can train your model by using **[Massachusetts Buildings Dataset]** https://www.cs.toronto.edu/~vmnih/data/

To train Unet, Inception or UnetV2 model

```
import unet, Inception, unetV2

x, y = ... # range [0,1] normalized images and ground truth map

model = unetV2.get_unet_plus_inception()
model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

model.fit(x,y)
```

