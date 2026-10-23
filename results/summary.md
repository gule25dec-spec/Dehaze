# Dehazing results (same images, same metric code for every model)

## RESIDE_SOTS_indoor

| Model | Checkpoint | N | PSNR (dB) | SSIM | MAE | s/img |
|---|---|---|---|---|---|---|
| Hazy input (no dehazing) |  | 500 | 11.97 | 0.6947 | 0.2297 |  |
| FFA-Net | its_train_ffa_3_19.pk | 500 | 36.36 | 0.9879 | 0.0103 |  |
| MSBDN-DFF | model.pkl | 500 | 32.91 | 0.9808 | 0.0148 | 0.059 |
| AECR-Net | ITS_train.pk | 500 | 37.14 | 0.9893 | 0.0098 | 0.051 |
| DEA-Net | PSNR4131_SSIM9945.pth | 500 | 41.22 | 0.9938 | 0.0058 | 0.055 |

## O_HAZE

| Model | Checkpoint | N | PSNR (dB) | SSIM | MAE | s/img |
|---|---|---|---|---|---|---|
| Hazy input (no dehazing) |  | 45 | 13.74 | 0.6096 | 0.1748 |  |
| FFA-Net | ots_train_ffa_3_19.pk | 45 | 15.50 | 0.6499 | 0.1398 | 3.292 |
| MSBDN-DFF | model.pkl | 45 | 16.73 | 0.6733 | 0.1194 | 0.380 |
| AECR-Net | NH_train.pk | 45 | 19.78 | 0.7432 | 0.0845 | 0.388 |
| DEA-Net | PSNR3659_SSIM9897.pth | 45 | 16.53 | 0.6843 | 0.1225 | 0.403 |

## NH_HAZE

| Model | Checkpoint | N | PSNR (dB) | SSIM | MAE | s/img |
|---|---|---|---|---|---|---|
| Hazy input (no dehazing) |  | 55 | 11.40 | 0.4449 | 0.2203 |  |
| FFA-Net | ots_train_ffa_3_19.pk | 55 | 11.94 | 0.4677 | 0.2061 | 2.978 |
| MSBDN-DFF | model.pkl | 55 | 12.45 | 0.4873 | 0.1930 | 0.343 |
| AECR-Net | NH_train.pk | 55 | 21.88 | 0.7199 | 0.0617 | 0.348 |
| DEA-Net | PSNR3659_SSIM9897.pth | 55 | 12.29 | 0.4954 | 0.1975 | 0.359 |

## NH_HAZE_test51-55

| Model | Checkpoint | N | PSNR (dB) | SSIM | MAE | s/img |
|---|---|---|---|---|---|---|
| Hazy input (no dehazing) |  | 5 | 11.34 | 0.4182 | 0.2254 |  |
| FFA-Net | ots_train_ffa_3_19.pk | 5 | 12.00 | 0.4431 | 0.2079 | 2.978 |
| MSBDN-DFF | model.pkl | 5 | 12.83 | 0.4648 | 0.1875 | 0.343 |
| AECR-Net | NH_train.pk | 5 | 19.92 | 0.6717 | 0.0818 | 0.348 |
| DEA-Net | PSNR3659_SSIM9897.pth | 5 | 12.55 | 0.4669 | 0.1914 | 0.359 |
