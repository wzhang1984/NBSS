# Supervised Random Walk

let $\frac{\partial F(w)}{\partial w_l}=2\lambda(w)+\sum_u\frac{2(||p_u-C_b||_2^2(p_u-C_a)^T(\frac{\partial p_u}{\partial w_l}-\frac{\partial C_a}{\partial w_l})-||p_u-C_a||_2^2(p_u-C_b)^T(\frac{\partial p_u}{\partial w_l}-\frac{\partial C_b}{\partial w_l}))}{\max (||p_u-C_a||_2^2,||p_u-C_b||_2^2)^2}$


[SRW_v041.py](./SRW_v041.py) contains the functions for Supervised Random Walk

[test.ipynb](./test.ipynb) contains two toy examples for testing the partial derivatives and the gradient descent functions  
  
[data_processing.ipynb](./data_processing.ipynb) contains the code for processing PathwayCommons edge features and Breast Cancer tumor mutation data  

[SRW_cookbook_BRCA.ipynb](./SRW_cookbook_BRCA.ipynb) contains some example code for classifying Breast Cancer samples into five known subtypes   
