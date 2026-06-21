# AAAI27 protocol v2 cross-backbone results

- ZCR catastrophic threshold (in-domain): MSE acc − ZCR acc > 5%
- Loss-neutrality threshold (in-domain): |SDSC acc − MSE acc| ≤ 2%
- Cross-domain TOST margin: ±3% accuracy, BH-FDR q=0.05

## AC-CL2-3: Loss-neutrality on in-domain (|SDSC − MSE| ≤ 2%)

| backbone | dataset | MSE mean | SDSC mean | ZCR mean | |SDSC−MSE| | ZCR drop | neutral? | catastrophic? |
|---|---|---|---|---|---|---|---|---|
| SimMTM-Cls | in_Epilepsy | 0.9496 | 0.9482 | 0.9471 | 0.13% | +0.25% | ✓ |   |
| TFC | in_Epilepsy | 0.9343 | 0.9343 | 0.9343 | 0.00% | +0.00% | ✓ |   |
| TFC | in_Gesture | 0.6083 | 0.6083 | 0.6083 | 0.00% | +0.00% | ✓ |   |
| TS2Vec | in_Epilepsy | 0.9575 | 0.9575 | 0.9575 | 0.00% | +0.00% | ✓ |   |
| TS2Vec | in_Gesture | 0.7139 | 0.7139 | 0.7139 | 0.00% | +0.00% | ✓ |   |
| TS2Vec | in_HAR | 0.8138 | 0.8138 | 0.8138 | 0.00% | +0.00% | ✓ |   |

**Loss-neutrality**: 6/6 in-domain cells pass ≤2% threshold.
**ZCR catastrophic (AC-CL2-2)**: 0/6 in-domain cells show > 5% drop.

## AC-CL2-4: Cross-domain TOST (±3% acc, BH-FDR q=0.05)

| backbone | source→target | MSE mean | SDSC mean | TOST p (raw) | TOST p (BH) | equivalent? |
|---|---|---|---|---|---|---|
| SimMTM-Cls | xd_SleepEEG_Epilepsy | 0.9512 | 0.9450 | 0.006253 | 0.007504 | ✓ |
| SimMTM-Cls | xd_SleepEEG_Gesture | 0.7778 | 0.7972 | 0.1437 | 0.1437 |   |
| TFC | xd_SleepEEG_Epilepsy | 0.9297 | 0.9297 | 0 | 0 | ✓ |
| TS2Vec | xd_ECG_Epilepsy | 0.9480 | 0.9480 | 0 | 0 | ✓ |
| TS2Vec | xd_SleepEEG_Epilepsy | 0.9195 | 0.9195 | 0 | 0 | ✓ |
| TS2Vec | xd_SleepEEG_Gesture | 0.6250 | 0.6250 | 0 | 0 | ✓ |

**Cross-domain TOST**: 5/6 pairs equivalent at ±3% (BH-FDR q=0.05).

## AC-CL2-2: C-4 reconstruction metrics across backbones


### dataset-type: in_Epilepsy

| backbone | SDSC ↑ | ZCR_soft ↓ | MSE ↓ | PCC ↑ |
|---|---|---|---|---|
| TFC | 0.9339 | 0.0010 | 0.0092 | 0.5736 |
| TS2Vec | 0.9650 | 0.0002 | 0.0032 | 0.7996 |

### dataset-type: in_Gesture

| backbone | SDSC ↑ | ZCR_soft ↓ | MSE ↓ | PCC ↑ |
|---|---|---|---|---|
| TFC | 0.6522 | 0.5807 | 0.3807 | 0.7902 |
| TS2Vec | 0.6179 | 0.4762 | 0.4245 | 0.7511 |

### dataset-type: in_HAR

| backbone | SDSC ↑ | ZCR_soft ↓ | MSE ↓ | PCC ↑ |
|---|---|---|---|---|
| TFC | 0.6755 | 0.0963 | 0.2086 | 0.3341 |
| TS2Vec | 0.9350 | 0.0019 | 0.0090 | 0.9511 |

### dataset-type: xd_ECG_Epilepsy

| backbone | SDSC ↑ | ZCR_soft ↓ | MSE ↓ | PCC ↑ |
|---|---|---|---|---|
| TS2Vec | 0.9689 | 0.0002 | 0.0029 | 0.8169 |

### dataset-type: xd_SleepEEG_Epilepsy

| backbone | SDSC ↑ | ZCR_soft ↓ | MSE ↓ | PCC ↑ |
|---|---|---|---|---|
| TFC | 0.9568 | 0.0007 | 0.0050 | 0.7069 |
| TS2Vec | 0.9740 | 0.0002 | 0.0023 | 0.8404 |

### dataset-type: xd_SleepEEG_Gesture

| backbone | SDSC ↑ | ZCR_soft ↓ | MSE ↓ | PCC ↑ |
|---|---|---|---|---|
| TS2Vec | 0.4534 | 0.5070 | 0.5708 | 0.7500 |