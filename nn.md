### Optimizers

* GD
    * direction of fastest local increase of a function.
        * Taylor expansion for $\|h\|=1$: $f(x+\alpha h) = f(x) + \alpha \langle \nabla f(x), h \rangle + o(\alpha)$
        * We want to find minimal value: $f(x) + \alpha \langle \nabla f(x), h \rangle < f(x)$
        * Cauchy–Schwarz inequality: $\langle \nabla f(x), h \rangle \geq \| \nabla f(x) \|_2 \| h \|_2 = \| \nabla f(x) \|_2$
        * where $\langle \nabla f(x), h \rangle$ takes minimal value when $h \sim - \nabla f(x) = - \frac{\nabla f(x)}{\| \nabla f(x) \|_2}$
    * hence $x_{k+1} = x_k - \alpha \nabla f(x_k)$
    * fastest descent: $\alpha_k = \arg \min_\alpha f \big( x_k - \alpha \nabla f(x_k) \big)$

* SGD
    * $f(x) = \sum_i L(x, y_i)$
    * $f(x) = \mathbb{E} \big[ L(x, \xi) \big], \xi \sim y$
    * $\nabla f(x) = \mathbb{E} \big[ \nabla L(x, y) \big]$
    * unbiased Monte-Carlo estimate: $\tilde{\nabla} f(x) = \frac{1}{B} \sum_{i=1}^{B} \nabla L(x, \xi_i)$

* Momentum
    * $v_{k+1} = \beta_k v_k - \alpha_k \nabla f(x_k)$
    * $x_{k+1} = x_k + v_{k+1}$

* Nesterov Momentum
    * $v_{k+1} = \beta_k v_k - \alpha_k \nabla f(x_k + \beta_k v_k)$
    * $x_{k+1} = x_k + v_{k+1}$

* AdaGrad
    * $G_{k+1} = G_k + \big( \nabla f (x_k) \big)^2$
    * $x_{k+1} = x_k - \frac{\alpha}{\sqrt{G_{k+1} + \varepsilon}} \nabla f(x_k)$

* RMSPROP
    * $G_{k+1} = \gamma  G_k + (1 - \gamma) \big( \nabla f (x_k) \big)^2$
    * $x_{k+1} = x_k - \frac{\alpha}{\sqrt{G_{k+1} + \varepsilon}} \nabla f(x_k)$

* Adam
    * $v_{k+1} = \beta_1 v_k + (1-\beta_1) \nabla f(x_k)$
    * $G_{k+1} = \beta_2  G_k + (1 - \beta_2) \big( \nabla f (x_k) \big)^2$
    * $x_{k+1} = x_k - \frac{\alpha}{\sqrt{G_{k+1} + \varepsilon}} v_{k+1}$

* AdamW
    * $v_{k+1} = \beta_1 v_k + (1-\beta_1) \nabla f(x_k)$
    * $G_{k+1} = \beta_2  G_k + (1 - \beta_2) \big( \nabla f (x_k) \big)^2$
    * $x_{k+1} = x_k - (\frac{\alpha}{\sqrt{G_{k+1} + \varepsilon}} v_{k+1} + \lambda x_k)$


* Second Order
    * Newton method (aka метод касательных)
    * $f(x + \Delta x) \sim f(x) + \langle \nabla f(x), \Delta x \rangle + \frac{1}{2} \langle H(x) \Delta x, \Delta x \rangle$
    * $\Delta x = H^{-1}(x) \nabla f(x)$
    * $x_{k+1} = x_k - \alpha H_k^{-1} \nabla f(x_k)$

* Quasi-Newthon methods
    * Secant method (aka метод секущих)
    * BFGS and L-BFGS

* Proximal Methods


### Regularization

#### Loss function

* L2 term (weight decay):
    * $L = L + \lambda \sum w^2$
    * usage with normalization layers: weight decay affects all trainable parameters, so using normalization layers with weight decay is not recommended, because normalization layers weights do not cause overfitting in the same way standard weights do.
    * weight decay is mathematically equivalent to L2 regularization, but is preferred since it does not complicate loss gradient 
* L1 term
    * $L = L + \lambda \sum |w|$
    * used in sparse autoencoders
        * resulting embedding has sparse features
        * valuable for interpretability
* Focal Loss for class imbalance
    * $- \sum_i (1-p_i)^\gamma \log p_i$
    * adds weight of $(1-p_i)^\gamma$ to each object
    * $\gamma = 0$ -> cross-entropy
    * $\gamma > 0$ -> the greater $p_i$ (well-classified examples), the smaller the weight
    * smaller weight for easily classified points
    * focus gradient updates on rare, hard-to-classify samples
* Entropy term (e.g. for diverse recommendations)
* Contrastive (triplet) loss


#### Structural changes

* model simplification
    * freeze part of weights, tune only portion of weights
        * lora (w' = w + b*a)
        *  other techniques
* dropout
    * $x^{k+1} = \frac{1}{1-p} x^k \cdot mask$, $mask \sim Bernoulli(1-p)$
    * $\nabla_{x^k} L = \frac{1}{1-p} \nabla_{x^{k+1}} L \cdot mask$
    * $\frac{1}{1-p}$ обеспечиват неизменность дисперсии и матожидания выходов слоя: $\mathbb{E}[\hat{x}] = (1-p) \mathbb{E}[x] +  p\mathbb{E}[0x]$
    * dropout with convolution networks: standard dropout is not used, because neighboring pixels contain enough information to guess the masked value. Solution: Dropout2d - randomly mask entite channels


#### Data Augmentations

* Geometric Transformations
    * Flipping (mirroring)
    * Rotation
    * Scaling and Cropping
    * Shearing (simulates a slanted or stretched appearance)
    * Translation
* Color Space Transformations
    * Brightness & Contrast (Increasing or decreasing overall light levels and contrast)
    * Saturation & Hue (Adjusting the intensity and shade of colors)
* Noise & Kernel Filters
    * Blurring
    * Noise Injection
* Erasing & Occlusion
    * Cutout & Random Erasing: Randomly blackening or removing a rectangular section of an image.
    * Coarse Dropout: Removing scattered smaller patches to simulate partial occlusion.
* Advanced Mixing Strategies
    * MixUp: Superimposing two images on top of each other with varying levels of transparency to blend their class labels.
    * CutMix: Cutting a patch from one image and pasting it onto another image, forcing the model to recognize multiple objects in a single scene.
    * Mosaic: Combining four images into a single training tile (heavily used in modern YOLO models).


#### Other Regularization Techniques

* early stopping
* feature selection
    * correlation
    * overfitting
    * L1
    * catboost feature importance
* feature extraction (extract meaningful information from unstructured data)
    * pretrained models
    * bag of words / tf-idf
    * etc
* dimensionality reduction
    * pca
    * cluster ids
    * etc
* add data similair to problematic points
* active-learning
    1. model is trained on a small corpus of labelled data
    2. given a larger corpus models selects samples it is most uncertain about to be labelled
    3. traing corpus of labelled data is exanded
    4. repeat


### No/Bad Convergence (Vanishing/Exploding Gradients)

##### Normalization Layers

* batchnorm
    * $x^{k+1} = \frac{x^k - \mu}{\sigma + \varepsilon} \gamma + \beta$, where $\gamma$ and $\beta$ are trainable
    * during training: use exponential moving average for $\mu$ and $\sigma$, during inference: use last saved value
    * controls distribution of output features of layer $k$, so that inputs to layer $k+1$ have consistent distribution (initial proposition)
    * modern proposition: smoothes out loss plane (ландшафт), reduces amount of local minima (за счет нормированности признаков)
    * another explanation: acts as a slight regularizer due to the noise introduced by calculating statistics on random mini-batches.
    * Target Usage: Convolutional Neural Networks (CNNs) and feed-forward networks.
    * one can use higher learning rate
    * BatchNorm1d: for $(B, C)$: $2C$ trainable parameters
    * BatchNorm1d: for $(B, C, L)$: $2C$ trainable parameters
    * BatchNorm2d: for $(B, C, H, W)$: $2C$ trainable parameters
    * normalizes across every dimension except $C$
    * every channel in $C$ has its own scale and shift
* layernorm
    * computes the mean and variance across the hidden features, independent of other samples in the batch
    * suitable for small batches
    * suitable for sequential  data: can average over feature dim alone, without mixing length dim 
    * normalizes over the last specified [shape] dimensions
    * for $(B, L, C)$ and `shape=[C]`: normalizes over C; $C$ trainable parameters
    * for $(B, C, H, W)$ and `shape=[H, W]`: normalizes over H, W; $H \times W$ trainable parameters
* instancenorm
    * used in style transfer: originally designed for this task, Instance Normalization helps neural networks become invariant to the contrast and color of individual images. By normalizing each channel independently, it detaches the "style" factors from the underlying content.
    * used when disturbances from other samples are not wanted in the current sample
    * same normalization as batchnorm, but without averaging over batch dim -> every instance in batch has its own normalization values
* rmsnorm
    * works just like layernorm, but does not center data around 0, only normalizes
    * faster than layernorm


##### Weight Initialization

* Xavier
    * inputs have distribution with 0 mean and non-zero variance
    * 0-symmetric activation function (tanh)
    * $w \sim N(0, 1/n_{in})$ to preserve variance in forward pass: $Var(output) = Var(input)$
    * $w \sim N(0, 1/n_{out})$ to preserve variance in backward pass: $Var(output) = Var(input)$
    * $w \sim N(0, 2/(n_{in}+n_{out}))$ to balance forward and backward pass
* Normalized Xavier
    * inputs have distribution with 0 mean and non-zero variance
    * $w \sim U \Big[ -\frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}\Big]$
* Kaiming
    * from ResNet
    * for ReLU activations
    * $w \sim N(0, 2/n_{in})$


##### Skip Connections

* $y = F(x) + x$
* $\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + 1$, $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} (\frac{\partial F}{\partial x} + 1)$
* is called residual, because layer is now learning not the target (say $H$), but the residual $H-x$
* Prevents Vanishing Gradients
* Preserves Feature Information
* ResNet - allow for very deep convolutional networks
* U-Net - pass high-resolution spatial maps from contracting path to expanding path
* Transformers - in both attention and dense blocks


##### Activation Functions

* ReLU
    * $\max(0, x)$
    * extremely fast to compute both forward and backward
    * does not train neurons with negative values
* Leaky ReLU / Parametric ReLU
    * $\max(\alpha x, x)$
    * fixes dying neurons problem
* tanh
    * benefit of zero-centered bount outputs
    * suffers from vanishing gradients
* GELU (Gaussian Error Linear Unit)
    * $x \Phi(x)$
    * Multiplies the input by the probability that a Gaussian (normal) distribution variable is less than that input.
    * mimcs ReLU + Dropout: Instead of a coin flip (like Dropout), the probability of the neuron dropping to 0 should depend on how small the input is.
    * Large positive inputs pass through almost entirely
    * Large negative inputs are suppressed
    * Inputs near zero act probabilistically, allowing the network to retain a mix of states. This behaves similarly to applying dropout and zoneout, which acts as a form of stochastic regularization during training.
* Swith / SiLU (Sigmoid Linear Unit)
    * $x \sigma(\beta x)$, $\beta = 1$ for SiLU
    * acts similair to GeLU, but easier to compute
    * EfficientNet
    * modern YOLO families


#### Local Minima

* LR Scheduling
* Warmup
* Alternative optimizers that avoid local minima
