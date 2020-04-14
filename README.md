# Automated instance segmentation of nuclei in DAPI fluorescent stained slide images from photoimmunotherapy clinical trial
An pytorch implementation of Attention U-Net architecture for nuclei segmentation in DAPI stained WSI's

# Results
Some Sample Result, you can refer to the [Results/](Results/) folder for **details**.

  ![GT Nuclei][50%](Results/GT_R026_nuclei.jpg)
  **GT Nuclei**

  ![GT Boundary][50%](Results/GT_R026_bound.jpg)
   **GT Boundary**
   ![GT Nuclei][50%](Results/nuclei_R026.jpg)
  **Predictions Nuclei**

  ![GT Boundary][50%](Results/bound_R026.jpg)
   **Predictions Boundary**

# Implementation details

**Library** : Pytorch version 1.3.1<br/>
**GPU** : Tesla V100-SXM2-32GB<br/>
**Number of epochs** : 200<br/>

 