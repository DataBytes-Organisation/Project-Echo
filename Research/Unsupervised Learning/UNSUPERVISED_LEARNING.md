
## Notes of unsupervised classification methods

This research note is about capturing the techniques that may be applicable to performing unsupervised learning and classification tasks on audio data.  For simplicity, the process of generating the initial 2D image representation of a 5 second audio clip is left out (although combinations of different audio representation techiques such as mel-specs, mfccs and discrete wavelet transforms) would have an impact on performance.

Unsupervised image classification techniques are a subset of machine learning methods that aim to identify and categorize patterns within images without relying on pre-labeled data. These approaches are particularly useful for cases where labeled data is scarce, expensive to generate, or when the goal is to uncover unknown structures within the data. The performance of these methods can be evaluated using ground truth labelled data.

![General Approach Phases](Unsupervised_General_Approach.jpg)

To help speed up generation of these notes, GPT-4 was used to assist writing summary descriptions of algorithms found in literature.

### Feature Extraction Methods

#### Autoencoder

Autoencoders are a type of neural network that learns to encode and decode images in an unsupervised manner. The encoder compresses the input image into a lower-dimensional latent space, while the decoder reconstructs the image from the latent space. By training the autoencoder to minimize the reconstruction error, it learns to capture essential features from the images, effectively performing dimensionality reduction. The lower-dimensional representations can then be used for clustering or other unsupervised classification tasks.

Idea: We could use a combination of supervised feature extraction method combined with unsupervised clustering methods: 

First build an auto-encoded that is trained in a supervised way to create a vector representation of 2 image inputs known to be of different classes, with a combined loss function that uses a weighted sum combination of losses:

- reconstruction error: in being able to reconstruct the two original feature images
- classification error: in being able to only represent latent space features that reduce classification error.

The classification error would be a cross-entropy loss via a fully connected layer attached to the latent space representation.

#### Histogram of Oriented Gradients (HOG)

HOG captures the distribution of gradient directions in an image. It divides the image into small cells, calculates the gradient magnitude and orientation for each pixel, and creates a histogram of gradient orientations for each cell. The histograms from all cells are concatenated to form the final feature vector. HOG is particularly useful for object detection and recognition tasks.

Wiki Reference: https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

#### Scale-Invariant Feature Transform (SIFT)

SIFT is a robust local feature descriptor that is invariant to scale, rotation, and illumination changes. SIFT identifies keypoints in the image, calculates their scale and orientation, and generates a 128-dimensional descriptor for each keypoint based on the local gradient information. These descriptors can be used as feature vectors for various tasks.

Wiki Reference: https://en.wikipedia.org/wiki/Scale-invariant_feature_transform

#### Speeded-Up Robust Features (SURF)

SURF is a faster and more efficient alternative to SIFT that is also invariant to scale, rotation, and illumination changes. SURF uses integral images for fast computation and approximates the Laplacian of Gaussian with box filters to detect keypoints. It computes Haar wavelet responses in the neighborhood of each keypoint to create a descriptor. The SURF descriptors can be used as feature vectors for various applications.

Wiki Reference: https://en.wikipedia.org/wiki/Scale-invariant_feature_transform


#### Deep Learning-based Features

Convolutional Neural Networks (CNNs) have proven to be powerful tools for image feature extraction. By training a CNN on a large dataset, the network learns hierarchical feature representations that can be used as feature vectors. You can use pre-trained models like VGG, ResNet, or Inception and extract features from the intermediate layers (e.g., fully connected layers or last convolutional layers) to generate a vector representation of the image.

#### Hidden Markov Models (HMMs) Features

In the paper (Bird Species Recognition Using Unsupervised
Modeling of Individual Vocalization Elements, 2019) the temporal evolution of features in modells using HMMs.

Here is an artical on using this idea to feed into clustering:

https://franky07724-57962.medium.com/using-keras-pre-trained-models-for-feature-extraction-in-image-clustering-a142c6cdf5b1

Papers with code has papers on 'deep clustering' tasks which seem to map here: https://paperswithcode.com/task/deep-clustering

### Dimension Reduction Algorithms

In order to improve the performance of downstream clustering methods, the application of dimensionality reduction techniques are typically applied.  These techniques reduce the dimensionality of the extracted features and generally improve the robustness and performance of the classfication algorithms.

#### Principal Component Analysis (PCA)

PCA is a widely used linear dimensionality reduction technique that seeks to project the original data into a lower-dimensional subspace while preserving the maximum variance. PCA computes eigenvectors and eigenvalues of the data covariance matrix and selects the top principal components as the new basis for the lower-dimensional space. In unsupervised image classification, PCA can help reveal the underlying structure of the data and improve clustering performance by reducing noise and redundancy.

Wiki Reference: https://en.wikipedia.org/wiki/Principal_component_analysis

### t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE is a non-linear dimensionality reduction technique that focuses on preserving local neighborhood relationships in the lower-dimensional space. It minimizes the divergence between probability distributions that represent pairwise similarities in the original and reduced spaces. t-SNE is particularly useful for visualizing high-dimensional image data and can aid in understanding the structure and relationships between different classes or clusters.

https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding


### Clustering-Based Methods

Scikit learn has a good page on this topic showing a range of ML techniques for clustering: https://scikit-learn.org/stable/modules/clustering.html



#### K-Means Clustering

K-means is an unsupervised clustering algorithm that partitions a set of images into K distinct groups, where each image belongs to the cluster with the nearest mean. The algorithm iteratively updates the centroids of the clusters and assigns images to the closest centroid until convergence. K-means is simple to implement, but the choice of K and initial centroids can significantly impact the results.

Wiki Reference: https://en.wikipedia.org/wiki/K-means_clustering

#### Hierarchical Clustering

Hierarchical clustering builds a tree-like structure to represent the relationships between images. The algorithm starts by treating each image as a separate cluster and iteratively merges the closest clusters until all images belong to a single cluster. Hierarchical clustering can use different distance metrics and linkage criteria to determine the similarity between clusters, which can influence the structure of the resulting dendrogram.

Wiki Reference: https://en.wikipedia.org/wiki/Hierarchical_clustering


#### Summary / Recommendation

Unsupervised techniques can be used in both feature extraction and classification aspects of the problem.  This summary proposes using standard library implementations to assist us in getting a working solution by end of T1 2023, steering away from implementing complex algorithms suggested by some papers.

Application of unsupervised learning techniques can produce high accuracy (~97%)classification results.

Suggest our Echo Engine have variants or plugins to allow the researcher to choose different pipelines for processing the audio.

At least one of the prototyping paths (engine variants) we explore on project echo should include building upon unsupervised techniques (both in extracting features and classification)

For Feature learning / Extractions I suggest:

- Application of deep learning and pre-trained image vectorisation: 1D and 2D Auto-encoders including variants such as variational autoencoders for structured latent space representation or transfer learning from pre-trained image classifiers.
Sinusoidal breakdown using FFTs (see Bird Species Recognition Using Unsupervised Modelling of Individual Vocalization Elements, 2019)
- Application of 'traditional' image processing techniques (e.g. SURF) see OpenCV library for other techniques: https://docs.opencv.org/3.4/db/d27/tutorial_py_table_of_contents_feature2d.html

For Unsupervised classification suggest:

- Application of non-deep learning clustering techniques (see https://scikit-learn.org/stable/modules/clustering.html#clustering)
- Application of dimensionality reduction techniques (see https://scikit-learn.org/stable/modules/decomposition.html#decompositions)

Suggest we also setup/design a generalised pipeline for model training that has several phases (some may simply be pass-through)

- Data Load (dealing with files and database endpoints)
- Data Cleaning (noise removal, rescaling, normalising, resampling)
- Data Segmentation (identifying 5 second clips containing vocalisations, includes trigger/detection aspects)
- 1D Data Augmentation library of techniques
- 2D Data Augmentation library of techniques
- Feature Extraction
- Feature Vectorisation
- Species Classification options (binary, multi-class and multi-label)

- This would allow us to have a family of techniques which could be combined into an "ensemble" to get even better overall performance.

References:

Acevedo, M. A., Corrada-Bravo, C. J., Corrada-Bravo, H., Villanueva-Rivera, L. J., & Aide, T. M. (2009). Automated classification of bird and amphibian calls using machine learning: A comparison of methods. Ecological Informatics, 4(4), 206-214. https://doi.org/10.1016/j.ecoinf.2009.06.005 

Aide, T. M., Corrada-Bravo, C., Campos-Cerqueira, M., Milan, C., Vega, G., & Alvarez, R. (2013). Real-time bioacoustics monitoring and automated species identification. PeerJ, 2013(1), Article e103. https://doi.org/10.7717/peerj.103 

Esmaeilpour, M., Cardinal, P., & Koerich, A. L. (2020). Unsupervised feature learning for environmental sound classification using weighted cycle-consistent generative adversarial network. Applied Soft Computing, 86, 105912. 

Jancovic, P., & Köküer, M. (2019). Bird Species Recognition Using Unsupervised Modeling of Individual Vocalization Elements. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 27(5), 932-947. https://doi.org/10.1109/TASLP.2019.2904790 

Michaud, F., Sueur, J., Le Cesne, M., & Haupert, S. (2023). Unsupervised classification to improve the quality of a bird song recording dataset. Ecological Informatics, 74, N.PAG-N.PAG. https://doi.org/10.1016/j.ecoinf.2022.101952 

Morfi, V., Bas, Y., Pamula, H., Glotin, H., & Stowell, D. (2019). NIPS4Bplus: a richly annotated birdsong audio dataset. PeerJ Computer Science, 5, e223. https://doi.org/10.7717/peerj-cs.223 

Mutanu, L., Gohil, J., Gupta, K., Wagio, P., & Kotonya, G. (2022). A Review of Automated Bioacoustics and General Acoustics Classification Research. Sensors (14248220), 22(21), 8361. https://doi.org/10.3390/s22218361 

Olaode, A., Naghdy, G., & Todd, C. (2014). Unsupervised classification of images: a review. International Journal of Image Processing, 8(5), 325-342. 

Stowell, D., & Plumbley, M. D. (2014). Automatic large-scale classification of bird sounds is strongly improved by unsupervised feature learning [Article]. PeerJ, 2014(1), Article e488. https://doi.org/10.7717/peerj.488 

Todor, G. (2017). Computational Bioacoustics : Biodiversity Monitoring and Assessment (Vol. 00004). De Gruyter. 
