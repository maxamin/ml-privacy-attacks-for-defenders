# Membership Inference: From Shadow Models to LiRA

Membership inference asks a yes-or-no question about one record: was it in the training set? It is the simplest privacy attack to state and the standard yardstick for every other privacy measurement in this book. If an attacker cannot tell members from non-members, most other attacks will also struggle; if they can, you have a measurable leak.

## Why membership leaks

Models fit their training data better than unseen data. A member usually gets lower loss and higher confidence on its true label than a similar non-member. Yeom, Giacomelli, Fredrikson and Jha (IEEE CSF 2018) formalised the link between this generalisation gap and membership advantage. Overfitting is not the only cause, though. Rare or outlier records are memorised even by models that generalise well on average, because the model has no other way to fit them.

## The attack family

| Attack | Signal | Needs | Strength |
|---|---|---|---|
| Loss or confidence threshold | Low loss means member | Target model scores only | Baseline; weak at low FPR |
| Shadow models (Shokri et al., 2017) | Classifier trained on shadow models' in/out outputs | Same-distribution data, many training runs | Historic reference point |
| Calibrated / difficulty-aware | Loss compared with the record's expected loss | Reference models or reference data | Much better on hard records |
| LiRA (Carlini et al., 2022) | Likelihood ratio of the record's score under "in" vs "out" Gaussians fitted from shadow models | Tens to hundreds of shadow models | Strong at low FPR |
| Label-only | Robustness of the predicted label to perturbations | Many queries | Works when scores are hidden |

**Shadow models.** Shokri, Stronati, Song and Shmatikov (IEEE S&P 2017) trained several "shadow" models on data the attacker controls, so the attacker knows which records were in or out of each one. An attack classifier then learns to map model outputs to "in" or "out", and is applied to the target model. The survey by Sun et al. (IEEE, 2020) summarises the key point: the attack needs only API access "and some data recording information", not internal details.

**LiRA.** Carlini, Chien, Nasr, Song, Terzis and Tramèr, "Membership Inference Attacks From First Principles" (IEEE S&P 2022, arXiv:2112.03570), changed how the field measures success. They argued that average accuracy and AUC hide what matters, and that attacks should be judged by true-positive rate (TPR) at very low false-positive rate (FPR), such as 0.1%. Their Likelihood Ratio Attack trains shadow models with and without each candidate record, fits a Gaussian to the record's scaled confidence in each case, and scores the target by the likelihood ratio. Per-record calibration is the key idea: a record that is always hard gets judged against its own baseline, not a global threshold.

## Large language models are different

For LLMs the picture is less clear. Duan et al., "Do Membership Inference Attacks Work on Large Language Models?" (COLM 2024, arXiv:2402.07841), evaluated attacks on models trained on the Pile from 160M to 12B parameters and found they "barely outperform random guessing for most settings". They attribute this to huge datasets seen for about one epoch and to fuzzy boundaries between member and non-member text. They also showed that several published successes came from a distribution shift (for example, members and non-members drawn from different dates), not from real membership signal.

The lesson for defenders is twofold. Pre-training membership on web text is weak evidence of anything. Fine-tuning on a small private set is a different regime, much closer to the small-model case, and should be tested.

## Metrics that mean something

- **TPR at fixed low FPR** (0.1%, 1%). This is what an attacker who must be confident achieves.
- **Full ROC curve on log-log axes**, so the low-FPR region is visible.
- **Membership advantage** (TPR − FPR), useful when comparing with differential-privacy bounds (Chapter 5).
- **Per-record scores**, so you can list the most exposed records and inspect why.

Always evaluate on members and non-members drawn from the *same* distribution, ideally by random split before training.

## Worked example

*Execution status: executed. The script below was run with Python 3.13 and NumPy 2.2 when this chapter was written; the output shown is what it printed. It uses synthetic data only.*

We train logistic regression on 100 features with 100, 1 000 and 10 000 rows and run a loss-threshold membership attack against equal numbers of members and fresh non-members.

```python
import numpy as np
rng = np.random.default_rng(0)
d = 100
w_true = rng.normal(size=d)
def make(n):
    X = rng.normal(size=(n, d))
    y = (X @ w_true + rng.normal(scale=4.0, size=n) > 0).astype(float)
    return X, y
def train(X, y, l2=0.01, steps=3000, lr=0.5):
    w = np.zeros(d)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
    return w
def loss(w, X, y):
    p = np.clip(1 / (1 + np.exp(-np.clip(X @ w, -30, 30))), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
for n in (100, 1000, 10000):
    Xin, yin = make(n); Xout, yout = make(n)
    w = train(Xin, yin)
    scores = np.r_[-loss(w, Xin, yin), -loss(w, Xout, yout)]   # low loss => guess "member"
    lab = np.r_[np.ones(n), np.zeros(n)][np.argsort(-scores)]
    tpr, fpr = np.cumsum(lab) / n, np.cumsum(1 - lab) / n
    auc = np.trapezoid(tpr, fpr)
    t1 = tpr[fpr <= 0.01].max()
    gap = ((Xin @ w > 0) == yin).mean() - ((Xout @ w > 0) == yout).mean()
    print(f"n_train={n:<6} gen_gap={gap:+.2f} attack_AUC={auc:.3f} TPR@1%FPR={t1:.3f}")
```

Output:

```text
n_train=100    gen_gap=+0.25 attack_AUC=0.774 TPR@1%FPR=0.040
n_train=1000   gen_gap=+0.07 attack_AUC=0.538 TPR@1%FPR=0.009
n_train=10000  gen_gap=+0.01 attack_AUC=0.506 TPR@1%FPR=0.009
```

**Reading it.** With 100 rows and 100 features the model fits the training set perfectly, the generalisation gap is 25 points, and the attack reaches AUC 0.77. More data closes the gap and the attack falls to chance. Note the TPR at 1% FPR: even the leaky model gives the attacker only 4% of members at that confidence. Whether that is acceptable depends on who those 4% are. In an earlier draft of this script, raising the L2 penalty did *not* lower attack AUC; regularisation is not a reliable privacy control by itself, so measure rather than assume.

To go further on real models, the open-source **ML Privacy Meter** (<https://github.com/privacytrustlab/ml_privacy_meter>) implements reference-model attacks, and the LiRA authors released code with their paper.

## Common pitfalls

- Reporting accuracy or AUC only. Carlini et al. showed attacks with similar AUC can differ by an order of magnitude at low FPR.
- Using non-members from a different source or time period, which measures distribution shift instead of membership.
- Training shadow models with a different architecture or recipe than the target, which weakens calibration.
- Treating weak LLM pre-training results as evidence that fine-tuned models are safe.
- Assuming hiding scores defeats the attack; label-only attacks trade queries for access.

## Exercises

1. Run the worked example and add a fourth row with `n=300`. Plot attack AUC against training-set size.
2. Replace the global loss threshold with a per-record calibrated score: train 16 reference models on random halves of a pool and score each record by its loss relative to its mean loss when excluded.
3. Run ML Privacy Meter against a small image classifier you train yourself and report TPR at 0.1% FPR.

## Further reading

- Carlini et al., "Membership Inference Attacks From First Principles", IEEE S&P 2022, arXiv:2112.03570
- Shokri et al., "Membership Inference Attacks Against Machine Learning Models", IEEE S&P 2017, arXiv:1610.05820
- Yeom et al., "Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting", IEEE CSF 2018, arXiv:1709.01604
- Duan et al., "Do Membership Inference Attacks Work on Large Language Models?", COLM 2024, arXiv:2402.07841
- Sun et al., "When Machine Learning Meets Privacy in 6G: A Survey", IEEE, 2020 (section on membership inference)
- ML Privacy Meter: <https://github.com/privacytrustlab/ml_privacy_meter>
