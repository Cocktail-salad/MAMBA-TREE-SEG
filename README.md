# Mamba-Enhanced Sparse CNN for High-Precision Instance-Level Tree Segmentation in Diverse Forests
This project introduces an enhanced forest point cloud instance segmentation method using sparse convolutional neural networks, designed for accurate tree resource inventories in small woodlands and mixed-species forests. 
# Simulation Pipeline
To address the challenges of real-world forest inventory tasks, we propose a novel pipeline that improves upon the limitations of existing FOR-instance datasets. 
# Key Contributions:

    Realistic Simulation: Generates synthetic forest point clouds that closely mimic real small-scale inventory conditions

    Improved Dataset: Mitigates the lack of fine-grained, real-world forest LiDAR data for training and evaluation

    Field-Aligned Design: Optimized for practical use cases, such as handheld or backpack-mounted mobile scanning in dense forests

# Example Output:
  
![result](https://github.com/Cocktail-salad/MAMBA-TREE-SEG/blob/master/Figures/Figure8.jpg)

# Results of ours Mamba-Enhanced model
The following figure shows how our partial forest slice compares this model with the baseline method. The results show that our method significantly improves the segmentation accuracy of dense clustering trees -a common challenge in real-world forest surveys. Our model addresses instance assignment errors that are common in overlapping tree crowns.

![result](https://github.com/Cocktail-salad/MAMBA-TREE-SEG/blob/master/Figures/Figure9.jpg)

We achieved higher accuracy in all axes with multi-directional feature extraction using mamba (Figure below). This would be beneficial to distinguish between dense crowns and individual trees in complex forest structures.

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
