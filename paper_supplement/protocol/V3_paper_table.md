# V3 Model Selection — Paper Section 5.2

Across 25 (dataset × backbone) combinations from the 200-cell grid, **selecting checkpoints by SDSC instead of MSE changes the chosen loss-mode in 40% of cases**.

When the decisions differ, the SDSC-chosen checkpoint incurs a small MSE cost (+0.62% on average) in exchange for a meaningful SDSC gain (+0.37% on average).

Combined with V1 (which shows SDSC measures structure where MSE is blind), this demonstrates that SDSC's measurement of reconstruction quality leads to **different and structure-favoring model-selection decisions**.


## Most-chosen losses

| selection criterion | top-3 most-chosen losses |
|---|---|
| mse-best | dilate (6), snr (6), hybrid (4) |
| sdsc-best | snr (6), dilate (5), hybrid (4) |
| hybrid-best | snr (7), dilate (5), hybrid (3) |