# Diffusion Model for Generative Denoising on OrganAMNIST

A conditional(Diffusion Posterior Sampling) generative denoising framework using diffusion model for the OrganAMNIST dataset from MedMNIST. 
The implementation is based on score-based generative modeling through stochastic differential equations (SDEs).

Follows the Variance Preserving Stochastic Differential Equation (VP-SDE) formulation:

![VP-SDE](https://latex.codecogs.com/svg.image?\color%7Bwhite%7D%7Bdx=-\frac{1}{2}\beta(t)x\,dt+\sqrt{\beta(t)}dw%7D)

where β(t) follows a cosine schedule. 
The forward process gradually adds noise to transform data distribution into a standard normal distribution, while the reverse process performs denoising.

The score network s_θ(x_t, t) is trained with a combined loss function:

![Loss function](https://latex.codecogs.com/svg.image?\color%7Bwhite%7D%7B\mathcal{L}=\mathcal{L}_{\text{noise}}+\lambda_{\text{freq}}\mathcal{L}_{\text{freq}}%7D)

where:
- ![Noise loss](https://latex.codecogs.com/svg.image?\color%7Bwhite%7D%7B\mathcal{L}_{\text{noise}}=\mathbb{E}_{t,x_0,\epsilon}\bigl\|s_\theta(x_t,t)-\epsilon\bigr\|_2^2%7D)
- ![Frequency loss](https://latex.codecogs.com/svg.image?\color%7Bwhite%7D%7B\mathcal{L}_{\text{freq}}=\mathbb{E}_{t,x_0,\epsilon}\bigl\|\mathcal{F}(\hat{x}_0)-\mathcal{F}(x_0)\bigr\|_1%7D)

The frequency domain loss $\mathcal{L}_{\text{freq}}$ improves the reconstruction of high-frequency details.

The model uses a U-Net architecture with self-attention modules containing:
- Time-dependent embeddings
- Skip connections
- Self-attention mechanism (8 heads)
- 6.6M parameters

### Forward Process (Data → Noise)
![Forward Process](ppt/forward_process.gif)

### Reverse Process (Noise → Data)
<img src="ppt/reverse_process_denoising.gif" width="450" alt="Reverse Process"/>

### Denoising Performance
The diffusion model achieves effective generative denoising on the OrganAMNIST dataset with:
- Noise level σ = 0.3
- 1000 sampling steps
- Improved PSNR from 17.21 dB to 19.66 dB (+2.44 dB)
- Comparable SSIM (0.5372 → 0.5364)

![Denoising Results](ppt/final_comparison.png)
