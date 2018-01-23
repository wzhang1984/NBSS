# Classifying tumors by supervised network propagation   

## Software overview
  
We develop a general algorithmic framework by adapting the Supervised Random Walk (SRW) algorithm (Backstrom and Leskovec, 2010) with a novel loss function designed specifically for cancer subtype classification. The package is called **Network-Based Supervised Stratification (NBS^2)**.  
  
## The package
  
* [__SRW_v044__](./SRW_v044.py) This software package contains all the functions of NBS^2.  
  
## Analysis scripts
  
* [__simulation_100x1000__](./simulation_100x1000.ipynb) Script to perform the simulation (**Fig. 2**).  
  
* [__data_processing_BRCA__](./data_processing_BRCA.ipynb) Script to processes PathwayCommons interaction features and Breast Cancer mutation profiles.  
  
* [__SRW_cookbook_BRCA__](./SRW_cookbook_BRCA.ipynb) Run the NBS^2 package to classify breast cancer subtypes.   
  
## Equations
  
* [__equations_v044__](./equations_v044.ipynb) This document contains equations of the algorithm.  
  
## Figures
  
| | | |
|:---:|:---:|:---:|
| ![Fig. 1](./images/Figure_1_method.PNG) |   | ![Fig. 2](./images/Figure_2_simulation.PNG) |
| ![Fig. 3](./images/Figure_BRCA_learning_curve.PNG) |   | ![Fig. 4](./images/Figure_BRCA_subnets.PNG) |

| **Fig. 1. Workflow of the Network-Based Supervised Stratification (NBS^2) of cancer subtypes.** NBS^2 takes three input data sets, represented by red arrows: 1) a molecular network where each edge is annotated by a set of features *x*, and each feature is assigned a initial weight *w*; 2) a tumor-by-gene matrix representing the mutation profile of a cohort; and 3) the defined subtype of each tumor. In each iteration, NBC compute an activation score a for each edge (**Eq. 3**), calculate a transition matrix *Q* (**Eq. 2**), perform a random walk (**Eq. 1**), and compute the value of the cost function *J*(*w*) (**Eq. 4**). Training the classifier is conducted iteratively using gradient descent. To minimize *J*(*w*), the algorithm calculates the partial derivative of *J*(*w*) with respect to the edge feature weights *w* using the chain rule (**Eqs. 7-11**), and updates *w* accordingly. Upon convergence, the algorithm outputs the final feature weights *w*, transition matrix *Q* and propagated mutation profiles *P*, which together defines the classification model. | | **Fig. 2. Experiments on simulated data.** (**A**) (Upper) Simulated mutation dataset including characteristic genes of two subtypes (gene 1-10) and an Frequently Mutated Gene (FMG) (gene 16). Mutated genes are shown in dark red and non-mutated genes are shown in white. A reduced set of tumors and genes (10 x 20) is shown as an example; the full simulation is (100 ✕ 1000). An edge-by-feature matrix is also used as a input for the supervised random walk. (Middle left) Unsupervised and (middle right) supervised random walk of mutations over a simulated gene interaction network. Shades of red show propagated mutation values for tumor sample #7. (Bottom) Propagated mutation profiles following (left) unsupervised and (right) supervised random walk. (**B, C**) Principal components analysis (PCA) of the full simulation (100✕1000) between (**B**) unsupervised random walk-based tumor stratification and (**C**) supervised random walk-based tumor classification. |  
  
**Fig. 3. Performance of breast cancer subtypes classification.** Values of cost function (left y-axis) and classification accuracy (right y-axis) are plotted against the number of iterations of NBC on the (**A**) training data and (**B**) validation data. Dash line indicates the accuracy of tumor stratification based on unsupervised random walk and non-propagated mutation profiles on the validation data.
  
**Fig. 4. Subnetworks of breast cancer subtypes.** Subnetworks characterizing breast cancer subtypes extracted from Pathway Commons by NBC, defined as a set of genes with significantly different network-transformed scores between different subtypes (ANOVA FDR < 0.05) and their molecular interactions with higher than average activation scores (> 0.01) learned from NBC. The pie chart represents the relative ratio of the average propagated mutation score for the four subtypes. For example, the large cyan area on _ERBB2_ represents that its average propagation score is much higher in HER2 tumors than other subtypes.
