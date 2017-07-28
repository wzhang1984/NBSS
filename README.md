# Supervised Random Walk

[SRW.py](./SRW.py) contains the functions for supervised random walk

[test.ipynb](./test.ipynb) contains two toy examples for testing the partial derivatives and the functions for gradient descent



Reference: https://arxiv.org/pdf/1011.4071.pdf
However, this package has a different objective function:
J = lam*||w||^2 + sum_over_u_and_v(y*sum((p_u-p_v)^2))
where pu and pv are the propagation scores of 2 samples, 
and y (1, 0) or (1, -1) indicates whether the 2 samples belong to the same group.
Thus the goal is to minimize the difference of samples within each group
(and optionally maximize the difference between different groups)
