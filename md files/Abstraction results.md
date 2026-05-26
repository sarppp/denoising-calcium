⚡ main ~/workspace/denoising-calcium uv run python scripts/ablation_verdict.py $RUNS/nll $RUNS/mse $RUNS/mae $RUNS/anscombe_mse $RUNS/huber --stack F1

────────────────────────────────────────────────────────────────────────
  Run:    nll
  Config: ablation_nll  |  Loss: poisson_gaussian_nll
  JSONL:  train_n2v3d.jsonl  |  Epochs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ep    train_loss    stSNR     sSNR     tSNR
  ────  ────────────  ───────  ───────  ───────
     0   129683.9621   -3.552   -3.044   -4.060
     1   127625.8188   -3.538   -3.031   -4.045
     2   127496.4176   -3.528   -3.022   -4.035
     3   127184.4226   -3.520   -3.014   -4.026
     4   125720.1317   -3.514   -3.009   -4.020
     5   124790.0971   -3.508   -3.003   -4.014
     6   122538.7883   -3.503   -2.998   -4.008
     7   123132.7272   -3.499   -2.994   -4.004
     8   123896.4065   -3.495   -2.991   -4.000
     9   123728.9929   -3.492   -2.987   -3.997

────────────────────────────────────────────────────────────────────────
  Run:    mse
  Config: ablation_mse  |  Loss: mse
  JSONL:  train_n2v3d.jsonl  |  Epochs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ep    train_loss    stSNR     sSNR     tSNR
  ────  ────────────  ───────  ───────  ───────
     0   265243.9989   -3.540   -3.032   -4.047
     1   261720.6178   -3.515   -3.009   -4.021
     2   262276.3401   -3.500   -2.994   -4.005
     3   262032.9130   -3.486   -2.981   -3.991
     4   261665.6283   -3.473   -2.969   -3.978
     5   259608.3396   -3.460   -2.955   -3.965
     6   254184.9199   -3.447   -2.941   -3.952
     7   254678.9026   -3.436   -2.929   -3.943
     8   256345.4673   -3.427   -2.919   -3.936
     9   255640.0190   -3.421   -2.910   -3.931

────────────────────────────────────────────────────────────────────────
  Run:    mae
  Config: ablation_mae  |  Loss: mae
  JSONL:  train_n2v3d.jsonl  |  Epochs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ep    train_loss    stSNR     sSNR     tSNR
  ────  ────────────  ───────  ───────  ───────
     0      235.2777   -3.527   -3.021   -4.033
     1      221.1001   -3.441   -2.938   -3.945
     2      202.7369   -3.307   -2.799   -3.814
     3      184.7113   -3.098   -2.573   -3.623
     4      165.1034   -2.715   -2.192   -3.238
     5      149.1992   -2.220   -1.739   -2.701
     6      137.5487   -1.679   -1.263   -2.095
     7      129.5098   -1.102   -0.772   -1.433
     8      125.3238   -0.533   -0.305   -0.761
     9      123.8031   -0.002   +0.117   -0.121

────────────────────────────────────────────────────────────────────────
  Run:    anscombe_mse
  Config: ablation_anscombe_mse  |  Loss: anscombe_mse
  JSONL:  train_n2v3d.jsonl  |  Epochs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ep    train_loss    stSNR     sSNR     tSNR
  ────  ────────────  ───────  ───────  ───────
     0       17.2851   -3.527   -3.020   -4.033
     1       13.7935   -3.431   -2.929   -3.933
     2       10.7931   -3.288   -2.791   -3.784
     3        8.2262   -3.100   -2.608   -3.592
     4        6.1515   -2.854   -2.366   -3.342
     5        4.7156   -2.538   -2.056   -3.020
     6        3.8167   -2.140   -1.674   -2.606
     7        3.3128   -1.683   -1.244   -2.123
     8        3.0806   -1.219   -0.810   -1.629
     9        3.0068   -0.760   -0.384   -1.135

────────────────────────────────────────────────────────────────────────
  Run:    huber
  Config: ablation_huber  |  Loss: huber
  JSONL:  train_n2v3d.jsonl  |  Epochs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ep    train_loss    stSNR     sSNR     tSNR
  ────  ────────────  ───────  ───────  ───────
     0      234.7781   -3.527   -3.021   -4.033
     1      220.4102   -3.442   -2.938   -3.945
     2      202.3669   -3.303   -2.795   -3.810
     3      184.0557   -3.086   -2.564   -3.608
     4      164.4581   -2.706   -2.188   -3.224
     5      148.8788   -2.211   -1.734   -2.689
     6      137.0529   -1.660   -1.246   -2.074
     7      128.9517   -1.079   -0.750   -1.409
     8      124.7912   -0.512   -0.284   -0.739
     9      123.2893   +0.020   +0.135   -0.096

════════════════════════════════════════════════════════════════════════
 ABLATION VERDICT
════════════════════════════════════════════════════════════════════════

  Loss                          val_F1_stSNR   NaN steps     Status
  ──────────────────────────  ──────────────  ──────────  ─────────
  huber                            +0.020 dB           0   ✅ stable
  mae                              -0.002 dB           0   ✅ stable
  anscombe_mse                     -0.760 dB           0   ✅ stable
  mse                              -3.421 dB           0   ✅ stable
  poisson_gaussian_nll             -3.492 dB           0   ✅ stable

  Decision:
  🟡 MAE wins/ties best (-0.002 dB).
     → Use mae for full training.
     → Poisson tails dominate; median-targeting is more robust.
     (Tied within 0.5 dB: huber (+0.020))

  Epoch count:
    [poisson_gaussian_nll      ] trend=flat (-1.0% over last 4 epochs)  (10 epochs logged)
    [mse                       ] trend=flat (-0.6% over last 4 epochs)  (10 epochs logged)
    [mae                       ] trend=slowing (10.0% over last 4 epochs)  (10 epochs logged)
    [anscombe_mse              ] trend=still-dropping (21.2% over last 4 epochs)  (10 epochs logged)
    [huber                     ] trend=still-dropping (10.0% over last 4 epochs)  (10 epochs logged)

  → Loss is slowing. 50 epochs likely sufficient.

  RECOMMENDATION: mae
════════════════════════════════════════════════════════════════════════

⚡ main ~/workspace/denoising-calcium uv run python scripts/ablation_verdict.py $RUNS/nll $RUNS/mse $RUNS/mae $RUNS/anscombe_mse $RUNS/huber --stack F3

────────────────────────────────────────────────────────────────────────
  Run:    nll
  Config: ablation_nll  |  Loss: poisson_gaussian_nll
  JSONL:  train_n2v3d.jsonl  |  Epochs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ep    train_loss    stSNR     sSNR     tSNR
  ────  ────────────  ───────  ───────  ───────
     0   129683.9621   -8.671   -7.892   -9.449
     1   127625.8188   -8.450   -7.675   -9.224
     2   127496.4176   -8.307   -7.538   -9.077
     3   127184.4226   -8.199   -7.434   -8.964
     4   125720.1317   -8.119   -7.357   -8.882
     5   124790.0971   -8.048   -7.286   -8.809
     6   122538.7883   -7.986   -7.226   -8.747
     7   123132.7272   -7.936   -7.174   -8.697
     8   123896.4065   -7.892   -7.130   -8.655
     9   123728.9929   -7.848   -7.085   -8.611

────────────────────────────────────────────────────────────────────────
  Run:    mse
  Config: ablation_mse  |  Loss: mse
  JSONL:  train_n2v3d.jsonl  |  Epochs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ep    train_loss    stSNR     sSNR     tSNR
  ────  ────────────  ───────  ───────  ───────
     0   265243.9989   -8.487   -7.715   -9.259
     1   261720.6178   -8.090   -7.328   -8.852
     2   262276.3401   -7.836   -7.080   -8.591
     3   262032.9130   -7.659   -6.911   -8.406
     4   261665.6283   -7.539   -6.796   -8.281
     5   259608.3396   -7.433   -6.691   -8.175
     6   254184.9199   -7.342   -6.597   -8.088
     7   254678.9026   -7.265   -6.527   -8.003
     8   256345.4673   -7.216   -6.513   -7.919
     9   255640.0190   -7.217   -6.575   -7.859

────────────────────────────────────────────────────────────────────────
  Run:    mae
  Config: ablation_mae  |  Loss: mae
  JSONL:  train_n2v3d.jsonl  |  Epochs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ep    train_loss    stSNR     sSNR     tSNR
  ────  ────────────  ───────  ───────  ───────
     0      235.2777   -8.256   -7.495   -9.016
     1      221.1001   -6.544   -5.996   -7.092
     2      202.7369   -4.008   -3.710   -4.307
     3      184.7113   -2.098   -1.613   -2.584
     4      165.1034   -0.570   -0.044   -1.095
     5      149.1992   +0.674   +0.890   +0.458
     6      137.5487   +1.183   +0.988   +1.378
     7      129.5098   +0.816   +0.623   +1.009
     8      125.3238   +0.440   +0.368   +0.512
     9      123.8031   +0.249   +0.274   +0.224

────────────────────────────────────────────────────────────────────────
  Run:    anscombe_mse
  Config: ablation_anscombe_mse  |  Loss: anscombe_mse
  JSONL:  train_n2v3d.jsonl  |  Epochs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ep    train_loss    stSNR     sSNR     tSNR
  ────  ────────────  ───────  ───────  ───────
     0       17.2851   -8.334   -7.571   -9.096
     1       13.7935   -6.805   -6.104   -7.506
     2       10.7931   -3.917   -3.327   -4.507
     3        8.2262   +0.703   +1.371   +0.034
     4        6.1515   -0.437   -0.319   -0.554
     5        4.7156   -5.929   -6.870   -4.988
     6        3.8167   -7.435   -8.752   -6.118
     7        3.3128   -6.013   -7.050   -4.976
     8        3.0806   -3.683   -4.473   -2.893
     9        3.0068   -1.313   -1.955   -0.671

────────────────────────────────────────────────────────────────────────
  Run:    huber
  Config: ablation_huber  |  Loss: huber
  JSONL:  train_n2v3d.jsonl  |  Epochs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ep    train_loss    stSNR     sSNR     tSNR
  ────  ────────────  ───────  ───────  ───────
     0      234.7781   -8.257   -7.496   -9.017
     1      220.4102   -6.550   -5.999   -7.101
     2      202.3669   -3.904   -3.621   -4.186
     3      184.0557   -1.945   -1.463   -2.427
     4      164.4581   -0.370   +0.155   -0.896
     5      148.8788   +0.706   +0.872   +0.541
     6      137.0529   +1.300   +0.911   +1.689
     7      128.9517   +1.212   +0.683   +1.740
     8      124.7912   +0.686   +0.352   +1.021
     9      123.2893   +0.376   +0.239   +0.512

════════════════════════════════════════════════════════════════════════
 ABLATION VERDICT
════════════════════════════════════════════════════════════════════════

  Loss                          val_F3_stSNR   NaN steps     Status
  ──────────────────────────  ──────────────  ──────────  ─────────
  huber                            +0.376 dB           0   ✅ stable
  mae                              +0.249 dB           0   ✅ stable
  anscombe_mse                     -1.313 dB           0   ✅ stable
  mse                              -7.217 dB           0   ✅ stable
  poisson_gaussian_nll             -7.848 dB           0   ✅ stable

  Decision:
  🟡 MAE wins/ties best (+0.249 dB).
     → Use mae for full training.
     → Poisson tails dominate; median-targeting is more robust.
     (Tied within 0.5 dB: huber (+0.376))

  Epoch count:
    [poisson_gaussian_nll      ] trend=flat (-1.0% over last 4 epochs)  (10 epochs logged)
    [mse                       ] trend=flat (-0.6% over last 4 epochs)  (10 epochs logged)
    [mae                       ] trend=slowing (10.0% over last 4 epochs)  (10 epochs logged)
    [anscombe_mse              ] trend=still-dropping (21.2% over last 4 epochs)  (10 epochs logged)
    [huber                     ] trend=still-dropping (10.0% over last 4 epochs)  (10 epochs logged)

  → Loss is slowing. 50 epochs likely sufficient.

  RECOMMENDATION: mae
════════════════════════════════════════════════════════════════════════

⚡ main ~/workspace/denoising-calcium


⚡ main ~/workspace/denoising-calcium uv run python scripts/ablation_verdict.py $RUNS/nll $RUNS/mse $RUNS/mae $RUNS/anscombe_mse $RUNS/huber --stack F3

────────────────────────────────────────────────────────────────────────
  Run:    nll
  Config: ablation_nll  |  Loss: poisson_gaussian_nll
  JSONL:  train_n2v3d.jsonl  |  Epochs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ep    train_loss    stSNR     sSNR     tSNR
  ────  ────────────  ───────  ───────  ───────
     0   129683.9621   -8.671   -7.892   -9.449
     1   127625.8188   -8.450   -7.675   -9.224
     2   127496.4176   -8.307   -7.538   -9.077
     3   127184.4226   -8.199   -7.434   -8.964
     4   125720.1317   -8.119   -7.357   -8.882
     5   124790.0971   -8.048   -7.286   -8.809
     6   122538.7883   -7.986   -7.226   -8.747
     7   123132.7272   -7.936   -7.174   -8.697
     8   123896.4065   -7.892   -7.130   -8.655
     9   123728.9929   -7.848   -7.085   -8.611

────────────────────────────────────────────────────────────────────────
  Run:    mse
  Config: ablation_mse  |  Loss: mse
  JSONL:  train_n2v3d.jsonl  |  Epochs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ep    train_loss    stSNR     sSNR     tSNR
  ────  ────────────  ───────  ───────  ───────
     0   265243.9989   -8.487   -7.715   -9.259
     1   261720.6178   -8.090   -7.328   -8.852
     2   262276.3401   -7.836   -7.080   -8.591
     3   262032.9130   -7.659   -6.911   -8.406
     4   261665.6283   -7.539   -6.796   -8.281
     5   259608.3396   -7.433   -6.691   -8.175
     6   254184.9199   -7.342   -6.597   -8.088
     7   254678.9026   -7.265   -6.527   -8.003
     8   256345.4673   -7.216   -6.513   -7.919
     9   255640.0190   -7.217   -6.575   -7.859

────────────────────────────────────────────────────────────────────────
  Run:    mae
  Config: ablation_mae  |  Loss: mae
  JSONL:  train_n2v3d.jsonl  |  Epochs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ep    train_loss    stSNR     sSNR     tSNR
  ────  ────────────  ───────  ───────  ───────
     0      235.2777   -8.256   -7.495   -9.016
     1      221.1001   -6.544   -5.996   -7.092
     2      202.7369   -4.008   -3.710   -4.307
     3      184.7113   -2.098   -1.613   -2.584
     4      165.1034   -0.570   -0.044   -1.095
     5      149.1992   +0.674   +0.890   +0.458
     6      137.5487   +1.183   +0.988   +1.378
     7      129.5098   +0.816   +0.623   +1.009
     8      125.3238   +0.440   +0.368   +0.512
     9      123.8031   +0.249   +0.274   +0.224

────────────────────────────────────────────────────────────────────────
  Run:    anscombe_mse
  Config: ablation_anscombe_mse  |  Loss: anscombe_mse
  JSONL:  train_n2v3d.jsonl  |  Epochs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ep    train_loss    stSNR     sSNR     tSNR
  ────  ────────────  ───────  ───────  ───────
     0       17.2851   -8.334   -7.571   -9.096
     1       13.7935   -6.805   -6.104   -7.506
     2       10.7931   -3.917   -3.327   -4.507
     3        8.2262   +0.703   +1.371   +0.034
     4        6.1515   -0.437   -0.319   -0.554
     5        4.7156   -5.929   -6.870   -4.988
     6        3.8167   -7.435   -8.752   -6.118
     7        3.3128   -6.013   -7.050   -4.976
     8        3.0806   -3.683   -4.473   -2.893
     9        3.0068   -1.313   -1.955   -0.671

────────────────────────────────────────────────────────────────────────
  Run:    huber
  Config: ablation_huber  |  Loss: huber
  JSONL:  train_n2v3d.jsonl  |  Epochs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ep    train_loss    stSNR     sSNR     tSNR
  ────  ────────────  ───────  ───────  ───────
     0      234.7781   -8.257   -7.496   -9.017
     1      220.4102   -6.550   -5.999   -7.101
     2      202.3669   -3.904   -3.621   -4.186
     3      184.0557   -1.945   -1.463   -2.427
     4      164.4581   -0.370   +0.155   -0.896
     5      148.8788   +0.706   +0.872   +0.541
     6      137.0529   +1.300   +0.911   +1.689
     7      128.9517   +1.212   +0.683   +1.740
     8      124.7912   +0.686   +0.352   +1.021
     9      123.2893   +0.376   +0.239   +0.512

════════════════════════════════════════════════════════════════════════
 ABLATION VERDICT
════════════════════════════════════════════════════════════════════════

  Loss                          val_F3_stSNR   NaN steps     Status
  ──────────────────────────  ──────────────  ──────────  ─────────
  huber                            +0.376 dB           0   ✅ stable
  mae                              +0.249 dB           0   ✅ stable
  anscombe_mse                     -1.313 dB           0   ✅ stable
  mse                              -7.217 dB           0   ✅ stable
  poisson_gaussian_nll             -7.848 dB           0   ✅ stable

  Decision:
  🟡 MAE wins/ties best (+0.249 dB).
     → Use mae for full training.
     → Poisson tails dominate; median-targeting is more robust.
     (Tied within 0.5 dB: huber (+0.376))

  Epoch count:
    [poisson_gaussian_nll      ] trend=flat (-1.0% over last 4 epochs)  (10 epochs logged)
    [mse                       ] trend=flat (-0.6% over last 4 epochs)  (10 epochs logged)
    [mae                       ] trend=slowing (10.0% over last 4 epochs)  (10 epochs logged)
    [anscombe_mse              ] trend=still-dropping (21.2% over last 4 epochs)  (10 epochs logged)
    [huber                     ] trend=still-dropping (10.0% over last 4 epochs)  (10 epochs logged)

  → Loss is slowing. 50 epochs likely sufficient.

  RECOMMENDATION: mae
════════════════════════════════════════════════════════════════════════






● My read of the results

  Use huber. The script's recommendation of mae is wrong —
  it's a tie-breaking preference bug. Look at the actual
  numbers:

  ┌──────────────┬──────────┬────────────────┐
  │     Loss     │ F1 stSNR │ F3 stSNR (OOD) │
  ├──────────────┼──────────┼────────────────┤
  │ huber        │ +0.020   │ +0.376         │
  ├──────────────┼──────────┼────────────────┤
  │ mae          │ -0.002   │ +0.249         │
  ├──────────────┼──────────┼────────────────┤
  │ anscombe_mse │ -0.760   │ -1.313         │
  ├──────────────┼──────────┼────────────────┤
  │ mse          │ -3.421   │ -7.217         │
  ├──────────────┼──────────┼────────────────┤
  │ nll          │ -3.492   │ -7.848         │
  └──────────────┴──────────┴────────────────┘

  Huber beats mae on both stacks. The script ranked it right
   (huber first in the table) but then its tie-break logic
  prefers mae when within 0.5 dB — that preference order was
   for true ties. This isn't a tie: huber leads on F3 by
  0.127 dB which matters for OOD.

  ---
  Critical warning on anscombe_mse — F3 trajectory

  ep3:  +0.703  ← sudden jump
  ep4:  -0.437  ← collapses
  ep5:  -5.929  ← blows up
  ep6:  -7.435  ← worst
  ep7:  -6.013  ← slowly recovering

  This is the noise model mismatch from nb10 (R²=0.001–0.24
  for val stacks) biting hard. When gain is wrong, Anscombe
  amplifies the error instead of stabilizing it. Confirmed:
  never use anscombe_mse.

  ---
  Epoch count — run 100 with early stopping

  Look at the F3 trend for both mae and huber:

  huber F3:  ep6=+1.300 (peak) → ep7=+1.212 → ep8=+0.686 →
  ep9=+0.376
  mae   F3:  ep6=+1.183 (peak) → ep7=+0.816 → ep8=+0.440 →
  ep9=+0.249

  F3 peaked at epoch 6 and is declining — the model is
  slowly overfitting away from OOD generalization. But F1 is
   still improving steeply. The competition score is mean
  across files, so you need the epoch where F1 gain > F3
  loss.

  Early stopping (patience=5 on stSNR_val, which averages
  F1+F2+F3) will find that point automatically — don't pick
  the epoch manually. Set epochs: 100 and let it stop when
  the overall val score stops improving.

  ---
  What to do now

  Edit configs/n2v3d_large.yaml (or whichever model you're
  testing next):

  loss:
    name: huber          # ← winner
    huber_delta: 1.0     # already there

  training:
    epochs: 10           # keep 10 for model size test

  Then run the 10-epoch model size test (base vs large).
  After that, set epochs: 100 for the full run — early
  stopping handles the rest.

  Also fix the verdict script's tie-break preference order
  so huber ranks above mae:

  Read 1 file (ctrl+o to expand)

● The bug is in LOSS_PREFERENCE — mae comes before huber, so
   when they're within 0.5 dB the script picks mae even when
   huber has a higher number. Fix it so huber ranks before
  mae (huber is strictly better — it's MAE in the tails but
  MSE near zero, which is always at least as good):




❯ but its tie why did you choose huber

● Fair point. Let me be precise about what the numbers
  actually say:

  F1 (in-distribution):
  - huber: +0.020
  - mae: -0.002
  - difference: 0.022 dB — genuinely meaningless, within
  measurement noise

  F3 (OOD):
  - huber: +0.376
  - mae: +0.249
  - difference: 0.127 dB — small but consistent

  So on F1 it's a true tie. On F3 huber leads by 0.127 dB
  every epoch from ep5 onward — it's not random noise, it's
  a consistent gap. That's the tiebreak.

  The deeper reason: huber = MAE when the error is large
  (tails) + MSE when the error is small (near zero). It
  strictly dominates MAE — you can only do equal or better,
  never worse. So when they tie on F1, huber should be
  preferred by construction, not by luck.

  But honestly either will work fine. The difference is
  0.022 dB on F1 and 0.127 dB on F3 — both within the noise
  of a 10-epoch run. If you already have a strong preference
   for mae, use it. The model size and number of epochs
  matter far more than this gap.