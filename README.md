# Credit Risk and Loan Approval Modelling

Segmenting LendingClub borrowers into four risk bands, then setting a different approval threshold for each band instead of one global cutoff. Recall on the highest risk segment reaches 82.6%, against 63.1% for a logistic baseline.

## Problem

Default prediction on consumer loans has two problems that a single accuracy number hides.

The first is imbalance. Defaults run at 15.6% in this sample, so a model that approves every application scores 84% accuracy and catches nothing. Recall on the default class is what matters, because every missed default is a direct write-off.

The second, and the more interesting one, is that a single decision threshold is wrong for a heterogeneous book. A prime borrower and a subprime borrower carry different base rates and different loss severity. Applying the same 0.5 cutoff to both means either rejecting good prime business or approving bad subprime business. There is no threshold that does neither.

So the approach here is: cluster first, then tune the threshold per cluster.

## Results

Recall on the default class, by borrower segment, on the held out test set:

| Segment | Default rate | n | Logistic regression | Basic DNN | Optimised DNN |
|---|---|---|---|---|---|
| Cluster 0, Prime | 9.0% | 2,177 | 0.123 | 0.133 | 0.036 |
| Cluster 1, Near-prime | 11.9% | 1,793 | 0.383 | 0.215 | 0.042 |
| Cluster 2, Sub-prime | 19.7% | 2,877 | 0.537 | 0.359 | **0.551** |
| Cluster 3, High-risk | 29.9% | 653 | 0.631 | 0.569 | **0.826** |

The optimised network is worse than the baseline on prime borrowers and much better on the risky ones, which is the right trade for a lender: the prime book is where losses are small and false rejections are expensive, and the high risk book is where a missed default actually hurts.

Recommended thresholds, derived from a sensitivity sweep of recall, precision and approval rate:

| Segment | Default rate | Threshold |
|---|---|---|
| Prime | 8.9% | 0.60 |
| Near-prime | 12.1% | 0.50 |
| Sub-prime | 19.8% | 0.40 |
| High-risk | 29.4% | 0.30 |

## Method

**Data.** LendingClub loans originated in 2012 and 2013, 50,000 rows, with realised outcomes. Only variables observable at origination are used. Anything recorded after the loan was issued is dropped, because including it leaks the outcome and produces a model that cannot be deployed.

**Features.** 18 selected: 14 numeric (loan amount, term, interest rate, instalment, employment length, annual income, DTI, delinquencies in the last 2 years, inquiries in the last 6 months, open accounts, public records, revolving balance, revolving utilisation, total accounts) plus grade, purpose, home ownership and verification status one-hot encoded into 24 columns. `revol_util` arrives as a percent string and is parsed to float; missing values in `emp_length` and `revol_util` are median imputed.

**Segmentation.** K-means on interpretable credit variables (DTI, revolving utilisation, annual income, recent inquiries, delinquencies, open and total accounts), fitted in R. Clusters are built from affordability and credit behaviour only, deliberately excluding the outcome, so the segmentation is available at decision time. The R Markdown for this stage is in [`r/borrower_segmentation.Rmd`](r/borrower_segmentation.Rmd).

**Models.** Three, trained on an identical 70/15/15 train, validation and test split so the comparison is fair:

1. Logistic regression, linear baseline.
2. Basic DNN, two hidden layers, ReLU, Adam, BCE loss.
3. Optimised DNN, three hidden layers, batch normalisation, dropout, AdamW, learning rate scheduling and early stopping on validation loss.

**Leakage control.** The scaler is fitted on the training split only. SMOTE is applied only to the standardised training set, never to validation or test, so synthetic minority samples are generated in the same feature space as the real data and never contaminate evaluation.

## Repository layout

```
notebooks/
  01_default_prediction_and_risk_band_thresholds.ipynb   Main pipeline: features, three models, per cluster evaluation, threshold policy
  02_earlier_iteration_three_segment.ipynb               Earlier iteration using a three segment split, kept for the comparison
r/
  borrower_segmentation.Rmd                              K-means segmentation and cluster profiling
data/
  early_2012_2013_loan_sample_with_outcome.csv           50,000 LendingClub loans with realised outcomes
```

## Running it

```bash
pip install -r requirements.txt
jupyter lab notebooks/01_default_prediction_and_risk_band_thresholds.ipynb
```

The R Markdown needs `factoextra`, `cluster`, `dplyr`, `psych` and `psychTools`. Knit it from RStudio with the CSV in the same directory.

## Honest limitations

ROC-AUC across all three models sits between 0.56 and 0.69. That is modest, and worth stating plainly. Origination-only features on a single vintage of LendingClub data have a genuine ceiling: there is no bureau score, no macroeconomic context, and no behavioural data after issue. The contribution here is the segmentation and thresholding policy rather than raw discriminative power, and the honest next step is external bureau data to lift recall on the high risk cluster.

## Stack

Python, PyTorch, scikit-learn, imbalanced-learn (SMOTE), pandas, NumPy, R, factoextra, cluster.

## Credit

Group project. The team collaborated on every stage: feature selection, segmentation, model architecture and the threshold policy. Project briefs and the written report are not included in this repository.
