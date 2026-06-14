# V3 Model Selection Analysis

Reused 200-cell grid: 189 cells parsed across 3 sweep directories.

Analyzable (dataset, backbone) combinations: 25


## Headline

**Decisions differ in 10 of 25 (40.0%) (dataset, backbone) combinations.**

When MSE-best and SDSC-best disagree, switching to SDSC-best costs **+0.62% MSE** on average while gaining **+0.37% SDSC**.


## Per-(dataset, backbone) selection table

| dataset | backbone | MSE-best loss | SDSC-best loss | hybrid-best | differ? | MSE cost % | SDSC gain % |
|---|---|---|---|---|---|---|---|
| ECL | DLinear | hybrid | hybrid | hybrid |   | +0.00% | +0.00% |
| ECL | PatchTST | hybrid | mse | hybrid | ✓ | +3.47% | +0.10% |
| ECL | iTransformer | hybrid | mse | mse | ✓ | +0.09% | +0.05% |
| ETTh1 | DLinear | dilate | dilate | dilate |   | +0.00% | +0.00% |
| ETTh1 | PatchTST | mse | sdsc | sdsc | ✓ | +0.05% | +0.21% |
| ETTh1 | SimMTM | mse | sdsc | sdsc | ✓ | +0.50% | +1.22% |
| ETTh1 | iTransformer | snr | snr | snr |   | +0.00% | +0.00% |
| ETTh2 | DLinear | dilate | dilate | dilate |   | +0.00% | +0.00% |
| ETTh2 | PatchTST | pcc | pcc | pcc |   | +0.00% | +0.00% |
| ETTh2 | SimMTM | zcr | dtw | zcr | ✓ | +0.43% | +0.03% |
| ETTh2 | iTransformer | snr | snr | snr |   | +0.00% | +0.00% |
| ETTm1 | DLinear | snr | snr | snr |   | +0.00% | +0.00% |
| ETTm1 | PatchTST | pcc | mse | mse | ✓ | +0.03% | +0.08% |
| ETTm1 | SimMTM | snr | snr | snr |   | +0.00% | +0.00% |
| ETTm1 | iTransformer | pcc | pcc | pcc |   | +0.00% | +0.00% |
| ETTm2 | DLinear | dilate | dilate | dilate |   | +0.00% | +0.00% |
| ETTm2 | PatchTST | dilate | dilate | dilate |   | +0.00% | +0.00% |
| ETTm2 | SimMTM | dilate | snr | snr | ✓ | +0.05% | +0.04% |
| ETTm2 | iTransformer | snr | snr | snr |   | +0.00% | +0.00% |
| Traffic | DLinear | dtw | mse | mse | ✓ | +0.60% | +1.71% |
| Traffic | PatchTST | hybrid | hybrid | hybrid |   | +0.00% | +0.00% |
| Weather | DLinear | dilate | dilate | dilate |   | +0.00% | +0.00% |
| Weather | PatchTST | snr | hybrid | snr | ✓ | +0.32% | +0.06% |
| Weather | SimMTM | mse | hybrid | pcc | ✓ | +0.64% | +0.15% |
| Weather | iTransformer | dtw | dtw | dtw |   | +0.00% | +0.00% |

## Loss-mode choice counts across all (dataset, backbone) combos

| loss | chosen as MSE-best | chosen as SDSC-best | chosen as Hybrid-best |
|---|---|---|---|
| dilate | 6 | 5 | 5 |
| dtw | 2 | 2 | 1 |
| hybrid | 4 | 4 | 3 |
| mse | 3 | 4 | 3 |
| pcc | 3 | 2 | 3 |
| sdsc | 0 | 2 | 2 |
| snr | 6 | 6 | 7 |
| zcr | 1 | 0 | 1 |