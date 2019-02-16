# Probabilistic_Graphical_Models
python implementation of classic probabilistic graphical models 


## HMM (Hidden Markov Model)
I implement a tensor version (using numpy) of HMM for three classic problems of HMM. **For any non-sequential/non-recursive, I bypass the for-loop by using tensor operations.** Thus, my numpy implementation could be easily adapted by tensorflow, pytorch.

1. The Evaluation problem: (forward/backward messages) Given 𝜆=(𝜋,𝐴,𝐵)  and a sequence of observation  𝑂=(𝑜1,𝑜2,𝑜𝑇) , estimate  P(𝑂|𝜆).

2. The Learning problem: (Baum-Welch algorithm, aka EM-algorithm) Given  𝑂=(𝑜1,𝑜2,𝑜𝑇) , estimate best fit parameters  𝜆=(𝜋,𝐴,𝐵).

3. The Decoding problem: (Viterbi's algorithm) Given  𝜆=(𝜋,𝐴,𝐵)  and a sequence of observation  𝑂=(𝑜1,𝑜2,𝑜𝑇) , estimate hidden states.


Please see directly the jupyter notebook HMM.ipynb for more details.

Project ongoing: (check list)

- tensor version of basic HMM ...done.
- discrete observation space...done.
- D trajectories version of HMM...done.
- Problem 1, 2...done.
- Problem 3...partially.
- continue version (with gaussian kernel) ... partially.
- D trajectories, continue version
- Monitor likelihood in the Problem 2.





