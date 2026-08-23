# realesrgan-to-safetensors

Converts a Real-ESRGAN RRDBNet `.pth` checkpoint to safetensors for InferKitMLX's `NFKMLXRealESRGAN`.

MLX loads safetensors/npz, not PyTorch `.pth`. This tool rewrites the official release into safetensors,
keeping the reference RRDBNet parameter names and PyTorch convolution layout `[out, in, kH, kW]`. The
Swift loader (`NFKMLXRealESRGAN.loadWeights(into:from:)`) transposes convolution weights to MLX's
channels-last layout `[out, kH, kW, in]` when it loads, so this tool does not transpose.

## Use

```bash
pip install torch safetensors

# Get the weights (~67 MB) from the official release:
curl -L -o RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

python convert.py RealESRGAN_x4plus.pth RealESRGAN_x4plus.safetensors
```

`--half` stores float16 (smaller). The default is float32, which matches the float32 image tensors the
backend feeds in; feed float16 input if you convert with `--half`.

The `RealESRGAN_x4plus_anime_6B` model converts the same way; it has 6 RRDB blocks, so build the net
with `blocks: 6`.

## Load in InferKitMLX

Host `RealESRGAN_x4plus.safetensors` in a Hugging Face repo and build through `NFKMLXHub`, or load a
local file with `NFKMLXModelRegistry.backend(named:weightsURL:)`:

```objc
[NFKMLXRealESRGAN register];
NSError *error = nil;
id<NFKInferenceBackend> upscaler =
    [NFKMLXHub backendNamed:@"real-esrgan-x4"
                       repo:@"your-org/real-esrgan-mlx"
                weightsPath:@"RealESRGAN_x4plus.safetensors"
                   revision:nil cacheDirectoryURL:nil error:&error];
```

The parameter names must match the reference RRDBNet (`conv_first.*`, `body.N.rdbM.convK.*`,
`conv_body.*`, `conv_up1.*`, `conv_up2.*`, `conv_hr.*`, `conv_last.*`).
