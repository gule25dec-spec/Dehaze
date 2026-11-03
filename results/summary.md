# Dehazing results (same images, same metric code for every model)

## RESIDE_SOTS_indoor

| Model | Checkpoint | N | PSNR (dB) | SSIM | MAE | MUSIQ | LPIPS | s/img |
|---|---|---|---|---|---|---|---|---|
| Hazy input (no dehazing) |  | 500 | 11.97 | 0.6947 | 0.2297 | 46.28 | 0.2000 |  |
| DEA-Net | PSNR4131_SSIM9945.pth | 500 | 41.22 | 0.9938 | 0.0058 | 50.10 | 0.0029 | 0.055 |
| ConvIR | its-base.pkl | 500 | 42.60 | 0.9948 | 0.0049 | 50.14 | 0.0025 | 0.114 |
| CoA | EMA_r.pth | 500 | 18.55 | 0.8302 | 0.0912 | 50.50 | 0.1769 | 0.035 |
| BiLaLoRA | LoRA.pth | 500 | 19.61 | 0.8062 | 0.0816 | 60.73 | 0.1806 | 0.061 |
| Ground truth |  | 50 |  |  |  | 50.11 |  |  |

## O_HAZE

| Model | Checkpoint | N | PSNR (dB) | SSIM | MAE | MUSIQ | LPIPS | s/img |
|---|---|---|---|---|---|---|---|---|
| Hazy input (no dehazing) |  | 45 | 13.74 | 0.6096 | 0.1748 | 52.82 | 0.4231 |  |
| DEA-Net | PSNR3659_SSIM9897.pth | 45 | 16.53 | 0.6843 | 0.1225 | 53.36 | 0.3652 | 0.403 |
| ConvIR | ohaze-base.pkl | 45 | 25.35 | 0.8045 | 0.0428 | 53.93 | 0.3226 | 0.833 |
| CoA | EMA_r.pth | 45 | 13.84 | 0.6830 | 0.1806 | 52.75 | 0.4056 | 0.115 |
| BiLaLoRA | LoRA.pth | 45 | 15.34 | 0.6929 | 0.1440 | 60.12 | 0.3381 | 0.447 |
| Ground truth |  | 45 |  |  |  | 63.16 |  |  |

## O_HAZE_test41-45

| Model | Checkpoint | N | PSNR (dB) | SSIM | MAE | MUSIQ | LPIPS | s/img |
|---|---|---|---|---|---|---|---|---|
| Hazy input (no dehazing) |  | 5 | 14.97 | 0.6382 | 0.1488 | 51.80 | 0.3648 |  |
| DEA-Net | PSNR3659_SSIM9897.pth | 5 | 18.77 | 0.7279 | 0.0929 | 51.25 | 0.3050 | 0.403 |
| ConvIR | ohaze-base.pkl | 5 | 24.64 | 0.8201 | 0.0497 | 54.05 | 0.2776 | 0.833 |
| CoA | EMA_r.pth | 5 | 15.83 | 0.7446 | 0.1365 | 50.18 | 0.3394 | 0.115 |
| BiLaLoRA | LoRA.pth | 5 | 18.30 | 0.7447 | 0.0983 | 58.94 | 0.2543 | 0.447 |
| Ground truth |  | 5 |  |  |  | 61.81 |  |  |

## NH_HAZE

| Model | Checkpoint | N | PSNR (dB) | SSIM | MAE | MUSIQ | LPIPS | s/img |
|---|---|---|---|---|---|---|---|---|
| Hazy input (no dehazing) |  | 55 | 11.40 | 0.4449 | 0.2203 | 54.00 | 0.5631 |  |
| DEA-Net | PSNR3659_SSIM9897.pth | 55 | 12.29 | 0.4954 | 0.1975 | 54.92 | 0.5117 | 0.359 |
| ConvIR | nhhaze-base.pkl | 55 | 26.10 | 0.7954 | 0.0380 | 61.37 | 0.2041 | 0.763 |
| CoA | EMA_r.pth | 55 | 11.04 | 0.5560 | 0.2284 | 56.43 | 0.4986 | 0.099 |
| BiLaLoRA | LoRA.pth | 55 | 12.07 | 0.5836 | 0.1898 | 62.64 | 0.4399 | 0.402 |
| Ground truth |  | 55 |  |  |  | 64.26 |  |  |

## NH_HAZE_test51-55

| Model | Checkpoint | N | PSNR (dB) | SSIM | MAE | MUSIQ | LPIPS | s/img |
|---|---|---|---|---|---|---|---|---|
| Hazy input (no dehazing) |  | 5 | 11.34 | 0.4182 | 0.2254 | 54.99 | 0.5828 |  |
| DEA-Net | PSNR3659_SSIM9897.pth | 5 | 12.55 | 0.4669 | 0.1914 | 55.75 | 0.5380 | 0.359 |
| ConvIR | nhhaze-base.pkl | 5 | 20.66 | 0.6910 | 0.0732 | 58.86 | 0.2892 | 0.763 |
| CoA | EMA_r.pth | 5 | 10.32 | 0.5103 | 0.2522 | 57.81 | 0.5157 | 0.099 |
| BiLaLoRA | LoRA.pth | 5 | 11.87 | 0.5530 | 0.1896 | 63.47 | 0.4729 | 0.402 |
| Ground truth |  | 5 |  |  |  | 62.21 |  |  |
