# Differential Privacy and DP-SGD in Practice

Every mitigation so far reduces leakage in some cases. Differential privacy (DP) is the one that comes with a proof. It bounds how much any single record can change what the model does, and so bounds every attack in Chapters 2 to 4 that depends on one record's presence. The price is accuracy, compute and the discipline to track a privacy budget honestly.

## The definition, briefly

A randomised algorithm *M* is (ε, δ)-differentially private if, for any two datasets that differ in one record and any set of outputs *S*, P[M(D₁) ∈ S] ≤ e^ε · P[M(D₂) ∈ S] + δ. The ENISA report *Privacy and Data Protection by Design – from policy to engineering* (Danezis et al., 2014) puts the intuition simply: noise is calibrated to the sensitivity of the query, and "the larger [ε], the less privacy". δ should be much smaller than 1/n, so that the "escape" probability cannot cover a single person.

For membership inference the link is direct. A DP guarantee bounds the attacker's achievable true-positive rate at each false-positive rate; with pure ε-DP, TPR ≤ e^ε · FPR. At ε = 8, a common published value, that bound allows an attacker a lot of room at moderate FPR. At ε = 1 it is tight. This is why the empirical attacks still matter even for DP models: they tell you where reality sits below the bound.

## Composition and budgets

DP composes: running several private computations on the same data adds up their privacy loss. Google's internal reference document *Differential privacy computations in data pipelines* (Google Anonymization team, 2021) describes how a whole pipeline can be treated "as a single DP operation with a certain epsilon and delta", with each sub-computation consuming part of the budget. It also introduces two engineering ideas that carry over to ML:

- **Privacy unit.** Decide whether the protected unit is a record, a user, or a user-day. A user with 500 support tickets gets 500 times less protection under record-level DP than under user-level DP.
- **Contribution bounding.** Cap how much one privacy unit can contribute (rows per user) before adding noise. DP-SGD's gradient clipping is the training-time version of the same idea.

Every retrain, hyperparameter search and evaluation on private data spends budget. Count them.

## DP-SGD

Abadi et al., "Deep Learning with Differential Privacy" (ACM CCS 2016, arXiv:1607.00133), introduced the method used in practice. Each training step:

1. Samples a batch (Poisson sampling, for the accounting to hold).
2. Computes a gradient **per example**.
3. **Clips** each per-example gradient to a maximum L2 norm *C*.
4. Sums the clipped gradients and adds Gaussian noise with standard deviation σ·C.
5. Updates the model with the noisy average.

A privacy accountant (the moments accountant in the original paper; Rényi DP or PRV accountants in current libraries) converts σ, the sampling rate, the number of steps and δ into an ε. Libraries: **Opacus** for PyTorch (<https://opacus.ai>), **TensorFlow Privacy**, and JAX implementations. Use the library's accountant; do not compute ε by hand.

## Trade-offs that bite

| Knob | Effect on privacy | Effect on utility | Note |
|---|---|---|---|
| Noise multiplier σ | Higher σ → lower ε | Slower, noisier training | Main dial |
| Clipping norm *C* | Neutral to ε | Too low biases gradients; too high adds noise | Tune on public data |
| Batch size / sampling rate | Larger batches amortise noise | Needs more compute | DP likes big batches |
| Epochs | More steps → higher ε | Better fit | Budget is spent per step |
| Pre-training on public data | None spent | Large gains | Fine-tune privately from a public base |

Utility loss is uneven. DP hurts rare classes and minority groups most, because their gradients are clipped and drowned in noise first. Report per-group accuracy, not just the average.

## Auditing a DP claim

A DP guarantee is only as good as the implementation. Jagielski, Ullman and Oprea, "Auditing Differentially Private Machine Learning: How Private is Private SGD?" (NeurIPS 2020, arXiv:2006.07709), used poisoned "canary" examples to compute empirical lower bounds on ε. Steinke, Nasr and Jagielski, "Privacy Auditing with One (1) Training Run" (NeurIPS 2023, arXiv:2305.08846), showed how to get a meaningful lower bound from a single training run by inserting or omitting many canaries at once. If your empirical lower bound exceeds the claimed ε, you have a bug, commonly in sampling, clipping or accounting.

## Worked example

*Execution status: executed. The script below was run with Python 3.13 and NumPy 2.2 when this chapter was written; output is what it printed. It implements the DP-SGD steps but does not compute ε; use Opacus or TensorFlow Privacy's accountant for that.*

```python
import numpy as np
rng = np.random.default_rng(3)
d = 20; w_true = rng.normal(size=d)
def make(n):
    X = rng.normal(size=(n, d)); return X, (X @ w_true + rng.normal(size=n) > 0).astype(float)
Xtr, ytr = make(2000); Xte, yte = make(2000)
def dpsgd(sigma, C=1.0, lr=0.5, batch=100, epochs=5):
    w = np.zeros(d)
    for _ in range(epochs * len(Xtr) // batch):
        idx = rng.choice(len(Xtr), batch, replace=False)     # (Poisson sampling in real accountants)
        X, y = Xtr[idx], ytr[idx]
        p = 1 / (1 + np.exp(-X @ w))
        g = (p - y)[:, None] * X                             # per-example gradients
        g /= np.maximum(1, np.linalg.norm(g, axis=1, keepdims=True) / C)   # clip each to norm C
        noisy = g.sum(0) + rng.normal(scale=sigma * C, size=d)            # add Gaussian noise once
        w -= lr * noisy / batch
    return ((Xte @ w > 0) == yte).mean()
for s in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
    print(f"noise_multiplier={s:<4} test_acc={np.mean([dpsgd(s) for _ in range(5)]):.3f}")
```

Output:

```text
noise_multiplier=0.0  test_acc=0.936
noise_multiplier=0.5  test_acc=0.936
noise_multiplier=1.0  test_acc=0.935
noise_multiplier=2.0  test_acc=0.937
noise_multiplier=4.0  test_acc=0.920
noise_multiplier=8.0  test_acc=0.893
```

**Reading it.** On this easy, low-dimensional task, noise up to σ = 2 costs nothing measurable, and σ = 8 costs about four points. Real tasks with many parameters and rare classes lose far more, which is why public pre-training followed by private fine-tuning is the standard recipe. Next step locally: wrap a small PyTorch model with Opacus's `PrivacyEngine`, set a target ε and δ, and record both accuracy and the accountant's reported ε for each run.

## Common pitfalls

- Quoting ε without δ, the privacy unit, the accountant used, or the number of training runs that touched the data.
- Using record-level DP when users contribute many records.
- Shuffled fixed batches with an accountant that assumes Poisson sampling, which invalidates the stated ε.
- Tuning hyperparameters on private data without counting those runs in the budget.
- Reporting average accuracy while minority groups degrade sharply.
- Treating a large ε (say, above 10) as meaningful protection without empirical audits.

## Exercises

1. Reproduce the worked example in PyTorch with Opacus and report ε at δ = 1e-5 for σ = 0.5, 1, 2 and 4.
2. Make one class rare (5% of the data) and plot its accuracy against σ next to the majority class.
3. Run the one-run audit of Steinke et al. on your Opacus model using 1 000 canaries and compare the empirical lower bound with the accountant's ε.

## Further reading

- Abadi et al., "Deep Learning with Differential Privacy", ACM CCS 2016, arXiv:1607.00133
- Dwork and Roth, *The Algorithmic Foundations of Differential Privacy*, Foundations and Trends in TCS, 2014 (cited by Chio and Freeman, 2018)
- Google Anonymization team, *Differential privacy computations in data pipelines* reference doc, 2021
- Danezis et al., *Privacy and Data Protection by Design – from policy to engineering*, ENISA, 2014
- Jagielski, Ullman, Oprea, "Auditing Differentially Private Machine Learning", NeurIPS 2020, arXiv:2006.07709
- Steinke, Nasr, Jagielski, "Privacy Auditing with One (1) Training Run", NeurIPS 2023, arXiv:2305.08846
- Opacus: <https://opacus.ai>; TensorFlow Privacy: <https://github.com/tensorflow/privacy>
