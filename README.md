# Mamba-Enhanced Sparse CNN for High-Precision Instance-Level Tree Segmentation in Diverse Forests
In order to achieve comprehensive and accurate tree resources inventory in small woodlands or local forests with div-erse tree species, an improved forest point cloud instance segmentation method based on sparse convolutional neural network is proposed. Meanwhile, the study develops a parameterized procedural generation pipeline that produces ecologically realistic synthetic forest scenes, effectively bridging the data gap for robust model develop-ment and evaluation. 
# Introduction of Simulated forest point cloud synthesis pipeline
To simulate the forest inventory task in a real situation, our study proposes a new pipeline to solve the problem faced by the FOR-instance dataset. To some extent, it solves the problem of lack of large and fine real forest land point clouds in current research, making it more con-sistent with the detection of small forest land in the real forest inventory process. The real situation is the scene where the driver’s walking trajectory distance is about 100 meters. Below is the result of our pipeline.  
  
![result](https://github.com/Cocktail-salad/MAMBA-TREE-SEG/blob/master/Figures/Figure8.jpg)

# Results of ours Mamba-Enhanced model
we take the visualization of simulated local forest dataset slices of TreeLearn dataset and FOR-instance as an example, and show the com-parison of instance segmentation results between our model and the baseline model below. For the comparison of instance segmentation results of closely intertwined trees, which are common for local forest trees, our optimization model can clearly reduce or even solve its instance assignment problem. This means that the model introduces a multi-dimensional attention mechanism through Mamba, which can capture and process information on multi-dimensional point cloud information at the same time.

![result](https://github.com/Cocktail-salad/MAMBA-TREE-SEG/blob/master/Figures/Figure9.jpg)

Meanwhile, in the above figure, we study the comparison of the evaluation results of the dataset in the x, y and z axis directions. The improved sparse convolutional neural network based on Mamba shows better segmentation accuracy in vertical and horizontal directions, and can effec-tively capture data features, which is superior to traditional methods. 

![result](https://github.com/Cocktail-salad/MAMBA-TREE-SEG/blob/master/Figures/Figure10.jpg)

# Data Preparation & Model Training
The code will be made available after the paper is accepted
# Citation
```
@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```
# Acknowledgements
The code is built based on [Treelearn](https://github.com/ecker-lab/TreeLearn).
