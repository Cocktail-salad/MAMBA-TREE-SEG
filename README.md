# Mamba-Enhanced Sparse CNN for High-Precision Instance-Level Tree Segmentation in Diverse Forests
This project introduces an enhanced forest point cloud instance segmentation method using sparse convolutional neural networks, designed for accurate tree resource inventories in small woodlands and mixed-species forests. To address data scarcity challenges, we've developed a parametric procedural generation system that creates ecologically realistic synthetic forest environments, enabling more robust model training and evaluation.
# Realistic Forest Inventory Simulation Pipeline
To address the challenges of real-world forest inventory tasks, we propose a novel pipeline that improves upon the limitations of existing FOR-instance datasets. Our solution tackles the scarcity of large-scale, high-quality real forest point clouds, particularly for small woodland areas—matching practical scenarios like field surveys with ~100-meter walking trajectories.
# Key Contributions:

    Realistic Simulation: Generates synthetic forest point clouds that closely mimic real small-scale inventory conditions

    Improved Dataset: Mitigates the lack of fine-grained, real-world forest LiDAR data for training and evaluation

    Field-Aligned Design: Optimized for practical use cases, such as handheld or backpack-mounted mobile scanning in dense forests

# Example Output:
  
![result](https://github.com/Cocktail-salad/MAMBA-TREE-SEG/blob/master/Figures/Figure8.jpg)

# Results of ours Mamba-Enhanced model
We compare our model against baseline methods using simulated forest slices from the TreeLearn and FOR-instance datasets. The results demonstrate that our approach significantly improves segmentation accuracy for densely clustered trees—a common challenge in real-world forest inventories.By integrating a multi-dimensional attention mechanism via Mamba, our model effectively resolves instance assignment errors in overlapping tree canopies. This enhancement allows simultaneous processing of complex 3D point cloud features, enabling more robust segmentation in dense woodland environments.

![result](https://github.com/Cocktail-salad/MAMBA-TREE-SEG/blob/master/Figures/Figure9.jpg)

Our Mamba-enhanced sparse CNN outperforms traditional methods in 3D forest point cloud segmentation, achieving higher accuracy across all axes with superior multi-directional feature extraction - particularly beneficial for distinguishing dense canopies and individual trees in complex forest structures.

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
