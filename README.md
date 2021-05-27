# Dolin-Segmentation-UNet

This study proposes a model for the automatic segmentation of dolins. Our model was applied in an area where many dolines are observed in the northwestern part of Sivas city, Turkey. The U-Net model with transfer learning methods was used for this task. A dataset consisting of 374 images was prepared using orthophoto images having a resolution of 0.3×0.3 meter.

* image and mask data were created.
* patches were created
* İmages not including doline were deleted
* dataset were created as train, validation and test
* segmentation (U-Net model)
* prediction of new area
* creating .shp file (vector data)

<p align="left">
  <img src="predicted_image.jpg" width="1024" alt="Dolin segmentation by Dr.Ali POLAT(2021)">
</p>
Figure showes predicted dolines from orthophoto for a new area.
