# Error analysis notes (facts only)

- Test subjects: 00, 01, 15, 17; test-label MD5: 9e5ab0415363e4074ad14c729bab8f50
- Train metrics in history.csv are Keras per-epoch running averages (dropout active); val and test metrics are computed on the full set in inference mode.

## Class support (test)

| Class | Epochs | Share |
|---|---|---|
| S1 | 471 | 6.5% |
| S2 | 3652 | 50.6% |
| S3 | 912 | 12.6% |
| S4 | 564 | 7.8% |
| REM | 1613 | 22.4% |

## Top confused class pairs (true -> predicted, % of the true class)

**v2 1D-CNN**
- S1 -> REM: 52.2% (246/471)
- S3 -> S4: 34.2% (312/912)
- S2 -> REM: 21.2% (775/3652)
- S4 -> S3: 16.7% (94/564)
- S2 -> S3: 14.4% (525/3652)

**v2 SimpleRNN**
- S1 -> REM: 68.6% (323/471)
- S3 -> S4: 47.3% (431/912)
- S2 -> REM: 37.4% (1366/3652)
- REM -> S2: 20.0% (323/1613)
- S1 -> S2: 20.0% (94/471)

**v2 MLP (SimpleNN)**
- S1 -> REM: 47.6% (224/471)
- S4 -> S3: 28.0% (158/564)
- REM -> S1: 25.4% (410/1613)
- S2 -> REM: 21.9% (800/3652)
- S1 -> S2: 20.4% (96/471)

**v1 1D-CNN**
- S3 -> S4: 74.5% (679/912)
- S1 -> REM: 36.3% (171/471)
- REM -> S1: 27.4% (442/1613)
- S2 -> S3: 18.1% (661/3652)
- S2 -> REM: 17.3% (630/3652)

**v1 SimpleRNN**
- S3 -> S4: 52.0% (474/912)
- S2 -> S4: 34.7% (1267/3652)
- REM -> S3: 28.5% (460/1613)
- S2 -> S3: 27.9% (1018/3652)
- S1 -> REM: 27.6% (130/471)

**v1 MLP (SimpleNN)**
- S1 -> REM: 100.0% (471/471)
- S2 -> REM: 100.0% (3652/3652)
- S3 -> REM: 100.0% (912/912)
- S4 -> REM: 100.0% (564/564)
- S1 -> S2: 0.0% (0/471)

## Train vs validation gap

| Run | Model | Epochs run | Best epoch (val_loss) | Train acc | Val acc | Acc gap | Train loss | Val loss | Loss gap | Test acc | Val-test acc gap | Last-epoch acc gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 | 1D-CNN | 18 | 13 | 0.8463 | 0.7323 | +0.1140 | 0.5038 | 1.3439 | +0.8401 | 0.6685 | +0.0638 | +0.1598 |
| v2 | SimpleRNN | 10 | 10 | 0.3045 | 0.4328 | -0.1284 | 1.3235 | 1.3096 | -0.0139 | 0.4157 | +0.0171 | -0.1284 |
| v2 | MLP (SimpleNN) | 7 | 2 | 0.5465 | 0.5837 | -0.0372 | 0.9302 | 0.9390 | +0.0088 | 0.5817 | +0.0020 | +0.1984 |
| v1 | 1D-CNN | 11 | 6 | 0.4380 | 0.3933 | +0.0447 | 1.4863 | 1.9784 | +0.4921 | 0.4072 | -0.0140 | +0.0874 |
| v1 | SimpleRNN | 10 | 10 | 0.2019 | 0.2256 | -0.0237 | 1.5375 | 1.6018 | +0.0643 | 0.2054 | +0.0203 | -0.0237 |
| v1 | MLP (SimpleNN) | 6 | 1 | 0.1350 | 0.1991 | -0.0641 | 1.8487 | 1.6083 | -0.2405 | 0.2237 | -0.0245 | +0.1084 |

## v1 vs v2 differences

**1D-CNN**

| Item | v1 | v2 | v2 - v1 |
|---|---|---|---|
| Test accuracy | 0.4072 | 0.6685 | +0.2612 |
| Macro precision | 0.3861 | 0.6036 | +0.2175 |
| Macro recall | 0.4694 | 0.6522 | +0.1827 |
| Macro F1 | 0.3673 | 0.6059 | +0.2385 |
| Cohen's kappa | 0.2513 | 0.5405 | +0.2892 |
| F1 S1 | 0.2244 | 0.4045 | +0.1801 |
| F1 S2 | 0.4760 | 0.7353 | +0.2594 |
| F1 S3 | 0.2269 | 0.5201 | +0.2933 |
| F1 S4 | 0.4161 | 0.6748 | +0.2587 |
| F1 REM | 0.4934 | 0.6946 | +0.2012 |
| Learning rate (initial) | 1e-05 | 0.001 | |
| Normalisation | False | True | |
| ReduceLROnPlateau | False | True | |
| Early-stopping patience | 5 | 5 | |
| RNN downsample factor | 1 | 1 | |
| Epochs run | 11 | 18 | +7 |
| Total train time (s) | 784.2 | 1215.7 | |

**SimpleRNN**

| Item | v1 | v2 | v2 - v1 |
|---|---|---|---|
| Test accuracy | 0.2054 | 0.4157 | +0.2103 |
| Macro precision | 0.2448 | 0.3396 | +0.0947 |
| Macro recall | 0.2754 | 0.4319 | +0.1565 |
| Macro F1 | 0.1962 | 0.3423 | +0.1461 |
| Cohen's kappa | 0.0509 | 0.2274 | +0.1765 |
| F1 S1 | 0.1217 | 0.0197 | -0.1021 |
| F1 S2 | 0.1956 | 0.4244 | +0.2287 |
| F1 S3 | 0.1563 | 0.2695 | +0.1132 |
| F1 S4 | 0.2223 | 0.5070 | +0.2847 |
| F1 REM | 0.2852 | 0.4911 | +0.2058 |
| Learning rate (initial) | 1e-06 | 0.0001 | |
| Normalisation | False | True | |
| ReduceLROnPlateau | False | True | |
| Early-stopping patience | 5 | 5 | |
| RNN downsample factor | 10 | 10 | |
| Epochs run | 10 | 10 | +0 |
| Total train time (s) | 883.7 | 883.6 | |

**MLP (SimpleNN)**

| Item | v1 | v2 | v2 - v1 |
|---|---|---|---|
| Test accuracy | 0.2237 | 0.5817 | +0.3580 |
| Macro precision | 0.0447 | 0.5281 | +0.4833 |
| Macro recall | 0.2000 | 0.5570 | +0.3570 |
| Macro F1 | 0.0731 | 0.5336 | +0.4605 |
| Cohen's kappa | 0.0000 | 0.4153 | +0.4153 |
| F1 S1 | 0.0000 | 0.2074 | +0.2074 |
| F1 S2 | 0.0000 | 0.6915 | +0.6915 |
| F1 S3 | 0.0000 | 0.5793 | +0.5793 |
| F1 S4 | 0.0000 | 0.7005 | +0.7005 |
| F1 REM | 0.3656 | 0.4893 | +0.1237 |
| Learning rate (initial) | Adam default (0.001) | 0.001 | |
| Normalisation | False | True | |
| ReduceLROnPlateau | False | True | |
| Early-stopping patience | 5 | 5 | |
| RNN downsample factor | 1 | 1 | |
| Epochs run | 6 | 7 | +1 |
| Total train time (s) | 1682.3 | 1886.5 | |
