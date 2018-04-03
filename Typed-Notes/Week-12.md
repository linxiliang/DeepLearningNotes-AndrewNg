# Week 12 Deep Learning

### Object Detection
* Classification -- try to tell what the object in the picture is
* Classification with localization -- try to tell what the object in picture is and where the object is located.
* Detection -- Detect all objects and locate them all on the image.

###### Notations
The upper left corner of the picture is denoted by (0,0), and the lower right of the model is (1,1). Then, we can learn about four points about the location of the object (object box). The output/prediction

$$ y = \left[\begin{array}{c}
p_c \\
b_x \\
b_y \\
b_h \\
b_w \\
c_1 \\
c_2 \\
c_3 \\
\end{array}\right]$$
where $p_c$ probability of an object exists, the $b$s are the locations of the object ($b_x$ and $b_y$ are the coordinates of the center of the object, $b_h$ and $b_w$ denotes the height and width of the object respectively), and the $c$s are the class of the object. If an object exists with car, then,
$$ y = \left[\begin{array}{c}
1 \\
b_x \\
b_y \\
b_h \\
b_w \\
0 \\
1 \\
0 \\
\end{array}\right]$$
and if no object exists,
$$ y = \left[\begin{array}{c}
0 \\
? \\
? \\
? \\
? \\
? \\
? \\
? \\
\end{array}\right]$$

For this, the loss function can be defined as
$$  \mathcal{L}(\hat{y}, y) =
  \begin{cases}
      ∑^{n_y}_{i} (\hat{y}_{i}-y_{i})^2 & \text{if $y_{1}=1$} \\
      (\hat{y}_{1}-y_{1})^2 & \text{if $y_{1}=1$}
  \end{cases}
$$
We can modify the loss function as we need such as using the logistic regression loss function and the softmax loss function.

###### Landmark Detection
Instead of detecting box, we can generalize the idea to identifying important locations such as identifying 42 landmarks/points on a face. The landmarks need to be consistent across different training samples.

###### Object Detection
* Crop and center objects, and build a ConV net to identify objects.
* Crop the images, and then do sliding window across the image (similar to convolution operation) to do predictions. We can also change the size of window. The computational cost is very high for sliding window. [We can potentially use the probability outputs to decide the next cropping instead of a fixed stride.]

###### Convolution Implementation of Sliding Windows (Overfeat Paper)
We can implement Fully Connected Layers as convolutional layers such as using 1×1 filters so that we have 1×1×nodes of the FC layers as dimension of the later layers. Hence, if the input image is larger (almost by definition since we generally crop and center images for our ConvNet in the first step), we apply the same convolution layers on the bigger images. The output will not be 1×nodes, but instead will be number of possible vertical slides × number of possible horizontal slides × nodes. Each of the horizontal and vertical positions then represent a possible image. This method works since a lot of the sliding window computations are redundant and the above calculation takes advantage of this redundancy and compute the final output more efficiently.

###### Bounding Box Predictions (I'm almost certain that we can use Reinforcement learning to improve the boxing algorithm.)
We can use YoLo (You only look once, a little hard to read) algorithm to improve bounding box prediction.

* Put a grid on the image
* Apply the classification and location algorithm to each grid cell and compute the output.
* Assign the object to the grid box which contains the center of the object

###### Intersection Over Union (IOU)
Computes the sizes of the intersection of the two boxes (true box and the prediction box) and the union of the two boxes. Then, we divide the union intersection size by union size. People generally use 0.5 as a correct classification.

###### No-Max Suppression
This method is used to deal with multiple detections of the same object
* Discard all boxes with low probability of an object (e.g. $p_c≤0.6$).
* Select the box with the highest probability of an object for the remaining boxes.
* Suppress all boxes/grid cells with high overlap with the selected boxes (they are detecting the same object).
* Select the box with the highest probability of an object for the remaining boxes.
* Suppress all boxes/grid cells with high overlap with the selected boxes (they are detecting the same object).
* Iterate ... until all boxes are either selected or suppressed.

###### Anchor Boxes
It allows detecting multiple objects in the same grid cell/box. Previously, we assign each object in training image to a grid cell that contains that object's midpoint.

With two anchor boxes
* Each object is assigned to a grid cell that contains object's midpoint and anchor box for the grid cell with highest IoU.
* The output along the third dimension will be doubled because we use two Anchor boxes to allow detection of two objects.
* We can potentially use K-means cluster to choose the anchor boxes if needed.

###### YoLo Algorithm
The output $y$ will be of the following dimensions
$$\text{horizontal grids} × \text{vertical grids} × \text{# of anchors} × \text{(# probabilities + positions + # classes})$$

* Run the objection location and objection identification algorithm for each of the grid box with anchor boxes
* For each grid call, get 2 predicted bounding boxes
* Get rid of low probability predictions
* For each class, use non-max suppression to general final prediction.

###### Region Proposal (R-CNN: Regions with CNN)
(Rich feature hierarchies for accurate object detection and semantic segmentation)
Only run algorithm on "interesting" regions.
* First run a segmentation algorithm, and try to find regions with interesting details (proposed regions)
* Run image classifier with only proposed regions -- output label + bounding boxes

People have tried to improve the computational speed of the algorithm and invented Fast-RCNN and Faster-RCNN.  

[I think the fact a NN is of fixed length is a little crazy. In theory, the NN should stop if it can reasonably judge the situation. Why should the computation goes through the entire algorithm? For example, we humans don't look a picture more carefully unless we cannot tell what's in it.]
