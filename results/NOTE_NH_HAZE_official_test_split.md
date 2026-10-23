# NH-HAZE: use the official test split for AECR-Net

AECR-Net's NH-HAZE checkpoint (NH_train.pk) was trained on NH-HAZE images 01-45,
so its score over all 55 images (21.88 dB) is inflated. Official split: 01-45 train,
46-50 validation, 51-55 test.

| Model | PSNR test 51-55 | SSIM test 51-55 | PSNR all 55 | SSIM all 55 |
|---|---|---|---|---|
| Hazy input | 11.34 | 0.4182 | 11.40 | 0.4449 |
| FFA-Net (OTS) | 12.00 | 0.4431 | 11.94 | 0.4677 |
| MSBDN-DFF (RESIDE) | 12.83 | 0.4648 | 12.45 | 0.4873 |
| AECR-Net (NH-HAZE) | 19.92 | 0.6717 | 21.88* | 0.7199* |
| DEA-Net (OTS) | 12.55 | 0.4669 | 12.29 | 0.4954 |

*includes AECR-Net's own training images. AECR-Net paper reports 19.88 dB on the NH-HAZE test set.
