## MLPerf HPC Datasets

The purpose of this note is to document the available datasets for MLPerf HPC benchmarking.

Please provide an appropriate amount of detail, including overall dataset size and structure.

### Climate analytics segmentation
Paper: https://arxiv.org/abs/1810.01993 
Description: Spatio-temporal simulated climate data for weather phenomena tracking problems.
Structure: 16-channel 2D images of size (768, 1152, 16) with per-pixel 3-class target labels
Size: 20TB
Access: Available on request; will make public


### CosmoFlow
Paper: https://arxiv.org/abs/1808.04728 
Description: Cosmology simulations of matter distribution for parameter regression problems.
Structure: 3D volumetric data, single-channel, shape (128,128,128)
Size: 1.4TB (TF records)
Access: Hosted at NERSC. Can/will be made public


### ImageNet-22k
Paper: http://www.image-net.org/papers/imagenet_cvpr09.pdf 
Description: The full imagenet dataset contains 21,841 synsets from wordnet and 14,197,087 images. They aim to have ~1000 images per synset (class). The following walkthrough from 2015 uses MXNet and some compression techniques to train an Inception model (http://proceedings.mlr.press/v37/ioffe15.pdf ) on the full dataset as described above:  https://github.com/dmlc/dmlc.github.io/blob/master/_posts/2015-10-27-training-deep-net-on-14-million-images.markdown 
Structure: 2D Image data (224x224x3): 14,197,087 images in 21,841 classes
Size: 1.31 TB


### Tencent ML-Images
https://github.com/Tencent/tencent-ml-images
Combination of Imagenet and Open-Images. A multi-label image database with 18M images and 11K categories, dubbed Tencent ML-Images, which is the largest publicly available multi-label image database until now.
128 GPUs, the throughout (i.e., the number of processed images per second) of MPI+NCCL is up to 11077 img/sec, The whole training process with 60 epochs takes 90 hours, i.e., 1.5 hours per epoch

The main statistics of ML-Images are summarized in ML-Images.

| Train images | Validation images | Classes | Trainable Classes | Avg tags per image | Avg images per class |
|:------------:|:-----------------:|:-------:|:-----------------:|:------------------:|:--------------------:|
|  17,609,752  |      88,739       |  11,166 |     10,505        |        8           |       1447.2         |

The paper here (Jan 8th 2019) describes the approach: https://arxiv.org/pdf/1901.01703.pdf
TL;DR: Move from single-tag to multi-tag per image to train on hierarchical classes 

“To the best of our knowledge, the largest public multi-label image database is Open Images [3], which includes about 9 million images and 6 thousand categories, and with about 20% label noises. However, only CNN models with multi-label outputs have been trained on Open Images, while its generalization to other vision tasks like single-label image classification has not been studied in [4]. “


| Checkpoints | Train and finetune setting |  Top-1 acc (on Val 224) | Top-5 acc (on Val 224) | Top-1 acc (on Val 299)| Top-5 acc (on Val 299)| 
| :---------: |  :-----------------------: |  :-------: |  :------: |  :------: |  :------: | 
| MSRA ResNet-101 | train on ImageNet      | 76.4       |      92.9 |     --    |    --     |
| Google ResNet-101 ckpt1 | train on ImageNet, 299 x 299 | -- |  -- | 77.5 | 93.9 |
| Our ResNet-101 ckpt1 | train on ImageNet | 77.8 | 93.9 | 79.0 | 94.5 |
| Google ResNet-101 ckpt2 | Pretrain on JFT-300M, finetune on ImageNet, 299 x 299 | -- | -- | 79.2 | 94.7 |
| Our ResNet-101 ckpt2 | Pretrain on ML-Images, finetune on ImageNet | 78.8 | 94.5 | 79.5 | 94.9 |
| Our ResNet-101 ckpt3 | Pretrain on ML-Images, finetune on ImageNet 224 to 299 | 78.3 | 94.2 | 80.73 | 95.5 |
| Our ResNet-101 ckpt4 | Pretrain on ML-Images, finetune on ImageNet 299 x 299 | 75.8 | 92.7 | 79.6 | 94.6 |


Using this as a reference target accuracy, we can project that it would take 


### OpenImages V4
Link: https://storage.googleapis.com/openimages/web/index.html
Description:
Structure:
Size:


### Language Modeling
Link: http://commoncrawl.org/the-data/ , model: https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html
Description (from wikipedia): Common Crawl is a nonprofit 501(c)(3) organization that crawls the web and freely provides its archives and datasets to the public.[1][2] Common Crawl's web archive consists of petabytes of data collected since 2011.[3] It completes crawls generally every month.[4]
Structure: Text
Projected Scale: 1,000s of GPUs
Size: Petabytes
1 GPU can train on about X in 1 month: 1GB
Access: Public, http://commoncrawl.org/terms-of-use/


### Deep Heuristic Learning
Link: https://arxiv.org/pdf/1611.09940.pdf 
Description: Reinforcement learning algorithms can be used to find improved heuristics for specific or classical problems in combinatorial optimization such as SAT, traveling salesman, graph coloring, etc.  Although it is clear from prior work on these topics that HPC systems can find better solutions than heuristics, doing so may not be practically useful because most users don’t have access to an HPC cluster.  Machine learning methods provide a different perspective to this problem because although training is hard (e.g. NP-HARD), inference is easy (e.g. P-TIME, parallel, etc).  So you can use an HPC system to find better heuristics, and then deploy them to users for specific problem domains who will use them in an inference setting (e.g. on a laptop or a phone), where the computational demands would be far less.
Structure: Simulation
Size: Practically Infinite
Projected Scale: 10,000s of GPUs
Access: Unrestricted public


### Neutrino dataset
Link: deeplearnphysics.org
https://github.com/DeepLearnPhysics/larcv2
One paper for all datasets, it’s written but needs to be posted
Description: There are 4 tasks worth benchmarking: neutrino classification (2D and 3D) and image segmentation (2D and 3D).  All 4 tasks are sparse convolutional neural networks.
Structure: All datasets are currently in ROOT format.  There is a preliminary implementation of hdf5 but need to convert the datasets and test it.  Work is in progress here.
Size: Number of example images ~ 50k.  Size on disk is ~ 100GB for dense representation
Access: Open access data
http://deeplearnphysics.org/DataChallenge/


### Neuro-imaging datasets
Paper: https://doi.org/10.1101/200675
Description: This project will develop a computational pipeline for neuroscience that will extract brain-image-derived mappings of neurons and their connections from electron microscope datasets too large for today’s most powerful systems. Ultimately the pipeline will be used to analyze an entire cubic centimeter of electron microscopy data. It uses Google’s FFN image segmentation algorithm. Flood-filling networks (FFNs) provide a method for automated segmentation that uses convolutional neural networks with an additional recurrent pathway that allows the iterative optimization and extension of individual neuronal processes. FFN has two input channels: one for the 3D image data, which is generally obtained from electron microscopy, and one for the current prediction of the object shape.

Structure of datasets: For training there are three input files. 1) Coordinates file as a TFrecord file 2) Data files (images) in hdf5 3) Labels (groundtruth) in hdf5.

Size: FIB-25 datasets can be obtained from Google's FFN repo at https://github.com/google/ffn. The sizes are (1) 113M (2) 22M (3) 2.2G

Access: Open ( https://github.com/google/ffn)


### Convergent Beam Electron Diffraction (CBED) Patterns of the Structural Properties of Materials
Link: https://doi.ccs.ornl.gov/ui/doi/70
Description: State of the art electron microscopes produce focused electron beams with atomic dimensions and allow to capture diffraction patterns arising from the interaction of incident electrons with nanoscale material volumes. Backing out the local atomic structure of said materials requires compute- and time-intensive analyses of these diffraction patterns (known as convergent beam electron diffraction, CBED). Traditional analyses of CBED requires iterative numerical solutions of partial differential equations and comparison with experimental data to refine the starting material configuration. This process is repeated anew for every newly acquired experimental CBED pattern and/or probed material. In this data, we used newly developed multi-GPU and multi-node electron scattering simulation codes on the Summit supercomputer to generate CBED patterns from over 60,000 materials (solid-state materials), representing nearly every known crystal structure. Briefly, a data sample from this data set is given by a 3-d array formed by stacking 3 CBED patterns simulated from the same material at 3 distinct material projections (i.e. crystallographic orientations). Each CBED pattern is a 2-d array (512x512 pixels) with float 32-bit image intensities. Associated with each data sample in the data set is a host of material attributes or properties which are, in principle, retrievable via analysis of this CBED stack. These consists of the crystal space group the material belongs to, atomic lattice constants and angles, chemical composition, to name but a few. Of note is the crystal space group attributed (or label). All possible spatial arrangements of atoms in any solid (crystal) material obey symmetry conditions described by 230 unique mathematical discrete space groups. Data generation utilized an award of computer time provided by the INCITE program. 
Size: ~50 TB
(Initially 550 GB data has been provided. Soon the whole dataset will be made public)


### MRI dataset (Biomedical image analysis)
BraTS: Image segmentation in multimodal magnetic resonance imaging (MRI) scans.
https://www.med.upenn.edu/sbia/brats2018.html. 
Has good exposure, thanks to Intel AI. Works well with 3D-Unet. 
Paper: https://ieeexplore.ieee.org/document/6975210 (529 citations)


### HEP-GAN dataset


