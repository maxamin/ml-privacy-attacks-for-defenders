# Verifying Unlearning and Running an ML Privacy Programme

People can ask to be forgotten. The ENISA report *Privacy and Data Protection by Design* (Danezis et al., 2014) lists Article 17 GDPR, the "right to erasure", among the obligations engineers must design for. Deleting a row from a database is easy. Removing its influence from a trained model is not, and claiming it has been removed without a test is the ML version of "we deleted your data" while the backups remain. This chapter covers unlearning, how to verify it, and how to fold every test in this book into a standing programme.

## Exact and approximate unlearning

**Exact unlearning** produces a model distributed exactly as if the record had never been in training. The only general method is retraining from scratch without it. **SISA** training (Bourtoule et al., "Machine Unlearning", IEEE S&P 2021, arXiv:1912.03817) makes this cheaper: split data into shards, train one sub-model per shard (further sliced with checkpoints), and aggregate. Deleting a record means retraining only its shard from the last checkpoint before the record appeared.

**Approximate unlearning** edits the existing model: gradient ascent on the forget set, fine-tuning on the retained set, influence-function updates, or targeted weight edits. These are fast but carry no guarantee. For LLMs, most "unlearning" methods are approximate, and many suppress outputs rather than remove information.

**Certified unlearning** methods sit between: for some convex models, noisy updates can be proven indistinguishable from retraining in a DP-like sense. They rarely apply to deep networks as deployed.

## What verification must show

The reference point is the retrained model. Good verification compares the unlearned model with models retrained without the forget set and asks whether an attacker can tell them apart on the forget records.

Hayes, Shumailov, Triantafillou, Khalifa and Papernot, "Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy" (arXiv:2403.01218, 2024), found that commonly used membership attacks for evaluating unlearning are too weak. Per-example attacks, calibrated against reference models as in LiRA (Chapter 2), reveal much more residual information than population-level thresholds, so many published unlearning methods look better than they are.

Three failure patterns to look for:

| Pattern | What you see | Why it fails |
|---|---|---|
| Residual memory | Forget-set loss still lower than unseen data | Influence not removed |
| Over-unlearning (Streisand effect) | Forget-set loss much *higher* than unseen data | The forget set now stands out; an attacker can detect it was deleted |
| Collateral damage | Retain-set accuracy drops | Unlearning harmed the model |
| Relearning | Small fine-tune restores forgotten behaviour | Information was hidden, not removed |

## Worked example

*Execution status: executed. The script below was run with Python 3.13 and NumPy 2.2 when this chapter was written; the output shown is what it printed. Synthetic data only.*

We train on 300 records, then try to forget 30. We compare the original model, exact retraining on the other 270, and a cheap approximate method (five steps of gradient ascent on the forget set). Each is tested with a loss-based membership score: forget records versus 30 fresh records never seen by any model.

```python
import numpy as np
rng = np.random.default_rng(4)
d = 100; w_true = rng.normal(size=d)
def make(n):
    X = rng.normal(size=(n, d)); return X, (X @ w_true + rng.normal(scale=4, size=n) > 0).astype(float)
def sig(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))
def train(X, y, w=None, steps=3000, lr=0.5, l2=0.01, sign=1):
    w = np.zeros(d) if w is None else w.copy()
    for _ in range(steps):
        w -= sign * lr * (X.T @ (sig(X @ w) - y) / len(y) + l2 * w)
    return w
def loss(w, X, y):
    p = np.clip(sig(X @ w), 1e-12, 1 - 1e-12); return -(y*np.log(p) + (1-y)*np.log(1-p))
def auc(a, b):                                   # P(score_a > score_b): forget vs never-seen
    return (a[:, None] > b[None, :]).mean()
X, y = make(300); Xf, yf = X[:30], y[:30]; Xr, yr = X[30:], y[30:]
Xu, yu = make(30)                                # fresh non-members, same distribution
orig = train(X, y)
retrain = train(Xr, yr)                          # exact unlearning: retrain without forget set
approx = train(Xf, yf, w=orig, steps=5, lr=0.5, sign=-1)   # 5 steps gradient ascent on forget set
for name, w in [("original", orig), ("retrain", retrain), ("approx-5-step", approx)]:
    lf, lu = loss(w, Xf, yf), loss(w, Xu, yu)
    print(f"{name:<14} mean_loss forget={lf.mean():.3f} unseen={lu.mean():.3f}  "
          f"MIA_AUC(forget vs unseen)={auc(-lf, -lu):.2f}  retain_acc={((Xr@w>0)==yr).mean():.2f}")
```

Output:

```text
original       mean_loss forget=0.145 unseen=0.420  MIA_AUC(forget vs unseen)=0.65  retain_acc=1.00
retrain        mean_loss forget=0.502 unseen=0.460  MIA_AUC(forget vs unseen)=0.54  retain_acc=1.00
approx-5-step  mean_loss forget=1.202 unseen=0.440  MIA_AUC(forget vs unseen)=0.48  retain_acc=0.94
```

**Reading it.** The original model clearly remembers the forget set (loss 0.145 against 0.420). Retraining brings the forget set in line with unseen data, as it should. The approximate method gets an AUC of 0.48, which a naive check would call "chance level, unlearning works". But its forget-set loss is 1.202, almost three times that of unseen data: the deleted records now stand out in the other direction, and retain accuracy fell six points. A one-sided "AUC below 0.5" test misses this. Compare the unlearned model's forget-set score distribution with the *retrained* model's distribution, two-sided, and with calibrated per-example attacks.

## Building the programme

Pull the chapters together into a repeatable process owned by a named team.

**Before training**
- Data inventory with privacy units and lawful basis; deduplicate and scrub.
- Decide DP or not, with a documented ε, δ, privacy unit and accountant if yes.
- Plant canaries (Chapter 3) and record them in a sealed list.
- Choose an unlearning strategy (SISA shards, scheduled full retrains, or approximate with verification) before the first deletion request arrives.

**Before release (gate)**
- Membership attack TPR at 0.1% and 1% FPR, with a calibrated attack (Chapter 2).
- Canary exposure for all planted canaries, against a written threshold.
- Extraction sampling on the release candidate (Chapter 3).
- Inversion and attribute tests for sensitive classes (Chapter 4).
- Empirical DP audit if a DP claim is made (Chapter 5).

**After release**
- Rerun the gate after each retrain or data-source change.
- Log and rate-limit sampling APIs; alert on extraction-shaped traffic.
- Process deletion requests with a verification report attached.
- Accept external reports through a disclosure channel and respond within a stated time.

## Research directions

- Membership and unlearning evaluation for LLMs that does not confuse distribution shift with memorisation.
- Unlearning verification that works from black-box access, so regulators and users can check claims.
- User-level DP for fine-tuning with practical utility.
- Auditing tools that give tight empirical ε lower bounds cheaply enough for CI.

## Common pitfalls

- Declaring unlearning successful because a weak attack scores near 0.5.
- Testing only that the model refuses to *say* the forgotten data; relearning attacks can restore it.
- Forgetting copies: checkpoints, distilled models, cached embeddings and evaluation sets.
- Making SISA shard boundaries depend on time, so one user's records spread across every shard.
- Running privacy tests once at launch and never again.

## Exercises

1. Extend the worked example with five retrained reference models and implement a two-sided test that flags both residual memory and over-unlearning.
2. Implement SISA with four shards on a small dataset and measure retraining cost for 10 random deletion requests against full retraining.
3. Write a one-page release-gate checklist for one of your models using the programme above, with thresholds and owners filled in.

## Further reading

- Bourtoule et al., "Machine Unlearning", IEEE S&P 2021, arXiv:1912.03817
- Hayes et al., "Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy", 2024, arXiv:2403.01218
- Danezis et al., *Privacy and Data Protection by Design – from policy to engineering*, ENISA, 2014
- Meta, *Llama 2 Responsible Use Guide*, 2023
- NIST AI 100-2, adversarial ML taxonomy: <https://csrc.nist.gov/pubs/ai/100/2/e2025/final>
- Sibling books: **ml-model-supply-chain-security**, **ai-red-teaming-security-copilots**
