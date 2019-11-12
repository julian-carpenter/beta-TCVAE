### TF 2 implementation of the beta-TCVAE ###

I'm a researcher at the [Max-Born-Institute Berlin](https://mbi-berlin.de/homepage). This is written and maintained as part of the ongoing
analysis of [diffraction data](https://en.wikipedia.org/wiki/Diffraction) obtained during
[coherent diffraction imaging experiments](https://en.wikipedia.org/wiki/Coherent_diffraction_imaging) in our group.
Feel free to take what you need but please don't expect any support on issues with this code.

However, I'm grateful for any bugs reported.

#### About this implementation:
Penalizing the total correlation instead of the KL between the prior and q(z) helps to disentangle the latent representations.

See:
* [betaTCVAE: Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1802.04942)
* [factorVAE: Disentangling by Factorising](https://arxiv.org/abs/1802.05983)
* [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://arxiv.org/abs/1811.12359)

**I added**:
* Laplace Prior
* Custom 15-Layer Resnet in Encoder and Decoder
* Use Pre-Activated Residual Blocks instead of plain convolutional layers
    1) [He et al. // Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
    2) [He et al. // Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
* Use Self-Attention in the encoder and the decoder
    1) [Zhang et al. // Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)
* Use linear warm-up learning rate scheduler (Not using RAdam, but Adam with warmup)
    1) [Liu et al. // On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)
