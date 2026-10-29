# Model selection before final test (2026-10-28)

Selection used validation only. Checkpoint = best validation mIoU; threshold = validation mIoU optimum.
The frozen final-test candidate was not evaluated before this table was written.

| arm | seed | best step | threshold | val mIoU | val OA | bF1 | bIoU | nbF1 | nbIoU | AP | params | test used |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tnet_off | 1234 | 6999 | 0.71 | 0.951814 | 0.985616 | 0.991286 | 0.982723 | 0.958824 | 0.920905 | 0.999646 | 2,733,058 | False |
| tnet_off | 2345 | 7999 | 0.74 | 0.947847 | 0.984354 | 0.990516 | 0.981211 | 0.955331 | 0.914483 | 0.999639 | 2,733,058 | False |
| tnet_off | 3456 | 7999 | 0.63 | 0.951203 | 0.985411 | 0.991159 | 0.982473 | 0.958297 | 0.919934 | 0.999691 | 2,733,058 | False |
| tnet_on | 1234 | 6999 | 0.54 | 0.926756 | 0.977592 | 0.986402 | 0.973168 | 0.936365 | 0.880345 | 0.999331 | 3,536,139 | False |
| tnet_on | 2345 | 6999 | 0.67 | 0.926436 | 0.977393 | 0.986265 | 0.972902 | 0.936153 | 0.879970 | 0.999334 | 3,536,139 | False |
| tnet_on | 3456 | 7999 | 0.65 | 0.931102 | 0.978804 | 0.987106 | 0.974540 | 0.940489 | 0.887663 | 0.999357 | 3,536,139 | False |

## Mean ± sample std (3 seeds)

| arm | tuned mIoU | 0.5 mIoU | tuned bF1 | tuned nbF1 | AP | params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tnet_on | 0.928098 ± 0.002606 | 0.922773 ± 0.003748 | 0.986591 ± 0.000451 | 0.937669 ± 0.002444 | 0.999341 ± 0.000014 | 3,536,139 |
| tnet_off | 0.950288 ± 0.002136 | 0.940754 ± 0.006931 | 0.990987 ± 0.000413 | 0.957484 ± 0.001883 | 0.999659 ± 0.000028 | 2,733,058 |

## Decision

- All six validation-selected models (2 arms × 3 seeds) will be evaluated once on the frozen final-test candidate,
  each with its own validation threshold.
- No threshold tuning, checkpoint change, or model selection on the test set.
- Primary comparison: mean ± std across seeds, reported for both arms.

## Caveats

- The validation region contains one main building; validation-based selection has limited scene diversity.
- Seed 1234 tnet_on ran on physical GPU 1 under contention; the other five runs used GPU 0. Runtime is not comparable.
- Test results will not be used to choose between arms or seeds.
