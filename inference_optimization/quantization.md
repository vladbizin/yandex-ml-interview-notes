# Quantization for inference

## 1. Formats and memory

- **Floating-point downcasting**
    - FP32 → FP16 or BF16 halves raw tensor storage
    - FP16 has more fraction bits than BF16
    - BF16 has the same exponent width as FP32
- **Low-bit formats**
    - INT8 and INT4 store integer codes; FP8 uses a floating-point format
- **Weight budget**
    - For $N$ weights at $b$ bits each, raw storage is $Nb/8$ bytes
    - Total device memory also includes KV cache and temporary activations

## 2. Linear quantization

- **Symbols**
    - $x$: original real value
    - $q$: stored integer code
    - $\hat{x}$: reconstructed value
    - $s>0$: scale, or spacing between adjacent reconstructed levels
    - $z$: integer zero point; code $z$ represents real zero exactly
    - $q_{\min},q_{\max}$: permitted integer limits
- **Mapping**

$$
q=\operatorname{clip}\!\left(\operatorname{round}(x/s)+z,\ q_{\min},\ q_{\max}\right),
\qquad \hat{x}=s(q-z)
$$

- **Asymmetric range**
    - Choose a real range $[a,b]$ that includes zero, with $a<b$
    - $s=(b-a)/(q_{\max}-q_{\min})$
    - $z=\operatorname{clip}(\operatorname{round}(q_{\min}-a/s),q_{\min},q_{\max})$
    - Useful for uneven ranges, such as mostly positive activations
- **Symmetric range**
    - Use $z=0$ and a range centered on zero
    - One common INT8 convention uses $[-127,127]$ with $s=\max|x|/127$
- **Two errors**
    - **Rounding:** nearby values share a code; without clipping, nearest rounding gives $|x-\hat{x}|\leq s/2$
    - **Clipping:** values outside the covered range saturate at an endpoint

## 3. Granularity and packing

- **Per tensor**
    - One scale and zero point for the whole tensor
    - Low metadata cost
    - One outlier can make the scale too coarse
- **Per channel**
    - Separate parameters for each channel
- **Per group**
    - Separate parameters for small blocks of weights, such as 128 values along the input dimension
    - Reduces error
    - Needs more metadata and kernel support
- **Packing**
    - Low-bit codes must share bytes to give the intended storage saving
    - Four 2-bit codes fit in one byte; two 4-bit codes fit in one byte
    - Keeping each 4-bit code in an INT8 tensor still uses one byte per value
    - Example: 128 INT4 weights plus one FP16 scale use $64+2=66$ bytes, or 4.125 bits/weight (assumes symmetric quantization with no zero-point storage or padding)

## 4. Quantized storage versus quantized compute

- **Notation**
    - W4A16 means 4-bit weights and 16-bit activations
    - Does not specify the accumulator type, scale layout, or KV-cache type
- **Weight-only**
    - Store packed low-bit weights; keep activations in FP16/BF16
    - A fused kernel unpacks and dequantizes weight tiles during multiplication
    - This reduces weight reads without materializing the full floating-point matrix
- **Weights and activations**
    - Both operands use integer codes
    - $i$ indexes input features; subscripts $x,w$ identify input and weight parameters
    - For a dot product with one scale per operand, omit bias:

$$
\hat{y}=s_xs_w\sum_i(q_{x,i}-z_x)(q_{w,i}-z_w)
$$

- **Note**
    - Norms, softmax, and sensitive layers can stay at higher precision

## 5. Quantization approaches

- **Post-training quantization (PTQ)**
    - Convert a trained model without gradient-based retraining
    - Simple weight rounding needs only the weights
    - Advanced methods can use calibration inputs
- **Static activation quantization**
    - Run representative calibration inputs and collect activation statistics
    - Choose ranges and fix quantization parameters before inference
- **Dynamic activation quantization**
    - Compute activation parameters from the current input at runtime
    - Adapts to changing ranges, but adds reduction and conversion work
    - Weights can still be quantized in advance
- **Quantization-aware training (QAT)**
    - Simulate quantize/dequantize effects during training or fine-tuning
    - Fake quantization usually keeps floating-point tensors, while exposing rounding and clipping error
    - Use approximate gradients through rounding, commonly a straight-through estimator
    - Convert to the target low-bit representation for deployment
- **Calibration tradeoff**
    - Min/max preserves the observed range; rare outliers can waste most levels
    - Clipping outliers gives smaller steps for common values, but larger errors at the tails
    - Match calibration data to real inputs; small tensor error does not guarantee task quality

## 6. Honorable quantization methods

- **Problem**
    - A few large activation channels can make low-bit quantization inaccurate
    - Their effect depends on layer inputs, not only on the weight magnitudes
- **Generalized Post-Training Quantization (GPTQ)**
    - Use calibration inputs and approximate second-order information
    - Quantize weights while adjusting remaining weights to limit layer-output error
- **Activation-aware Weight Quantization (AWQ)**
    - Use activation statistics to choose channel scaling before weight quantization
    - Activation-aware does not mean that activations are also quantized
- **SmoothQuant**
    - Rescale activation channels and apply inverse scaling to the matching weights
    - The transformation preserves the full-precision linear operation before quantization
    - Move some quantization difficulty from activations to weights to enable W8A8

## 7. Best use-cases

- **Small-batch decode**
    - Often limited by reading weights; weight-only quantization can reduce those bytes
    - Unpacking and dequantization must cost less than the saved memory traffic
- **Prefill and large batches**
    - More tokens reuse each weight
    - Matrix compute can become the main cost
    - Low-precision compute needs suitable kernels and hardware
- **Long contexts**
    - Quantize KV separately to reduce cache memory

## Sources

- **DeepLearning AI courses:** [Quantization Fundamentals](https://www.deeplearning.ai/courses/quantization-fundamentals), [Quantization in Depth](https://www.deeplearning.ai/courses/quantization-in-depth)
- **Mapping and arithmetic:** [Hugging Face: Quantization concepts](https://huggingface.co/docs/optimum/concept_guides/quantization)
- **Ranges, granularity, QAT:** [PyTorch: Practical Quantization](https://pytorch.org/blog/quantization-in-practice/)
- **LLM methods:** [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978),d [SmoothQuant](https://arxiv.org/abs/2211.10438)d
