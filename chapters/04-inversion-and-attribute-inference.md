# Model Inversion and Attribute Inference

Membership inference and extraction ask about specific records. Model inversion and attribute inference ask about *properties*: what a class looks like, or what hidden value a known person has. They are easy to confuse with ordinary statistical inference, which makes them hard to reason about. This chapter separates the cases and shows how to test for them.

## Model inversion

Fredrikson, Jha and Ristenpart, "Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures" (ACM CCS 2015), showed that with access to a face-recognition model's confidence scores, an attacker could reconstruct a recognisable image of a person given only their name (the class label). The method is optimisation: start from a blank input and adjust it to maximise the model's confidence in the target class, with a prior that keeps the input realistic. Meta's *Llama 2 Responsible Use Guide* (2023) describes the goal the same way: "model inversion attacks to reconstruct representative views of a subset of examples."

The phrase "representative views" is important. Inversion usually recovers a class prototype, not a specific training image. When each class is one person, as in face recognition, the prototype is a privacy harm in itself. When each class contains thousands of people, the prototype is closer to a population statistic.

White-box access makes inversion stronger. With weights, the attacker gets exact gradients. In federated learning, shared gradients can reveal individual training examples; this "gradient inversion" threat was demonstrated by Zhu, Liu and Han, "Deep Leakage from Gradients" (NeurIPS 2019). The 2020 IEEE survey by Sun et al. also notes that federated learning is vulnerable to membership inference, so "data never leaves the device" is not a privacy guarantee.

## Attribute inference

Attribute inference fills in a missing field. The attacker knows part of a record (age, postcode, some answers) and uses the model to predict a sensitive attribute (diagnosis, income, sexuality). The question defenders must answer is: does the model reveal more about *this person* because they were in the training set, or would the same prediction be made for anyone with those features?

The first case is a privacy leak tied to training membership. The second is ordinary inference from population correlations, sometimes called "statistical disclosure". It can still be harmful and may be a fairness or ethics issue, but differential privacy will not stop it, because DP protects individuals' contributions, not population-level patterns.

The older privacy-preserving data mining literature saw this coming. Aggarwal and Yu's *Privacy-Preserving Data Mining: Models and Algorithms* (Springer, 2008) notes that k-anonymity is "neither necessary nor sufficient to prevent attribute disclosure", and that linkage attacks combine released data with outside sources. Machine-learning models are another form of data release, and the same reasoning applies: removing names does not prevent inference.

| Property | Model inversion | Attribute inference | Membership inference |
|---|---|---|---|
| Input to attacker | Class label | Partial record | Full record |
| Output | Prototype input for the class | Hidden attribute value | In or out |
| Main access needed | Scores or gradients | Predictions on crafted inputs | Scores or loss |
| Stopped by DP? | Only per-record parts | Only the membership-linked part | Yes, bounded |

## Testing approach

1. **Inversion test.** For each sensitive class, run gradient-based inversion (white-box if you have weights; score-based black-box otherwise) and compare the result with the class mean of the training data and with the nearest individual training example. A result closer to one individual than to the class mean is a serious finding.
2. **Attribute test with a membership control.** Measure attribute-prediction accuracy separately for members and for comparable non-members. If accuracy is much higher on members, the model leaks individual information; if it is the same, the leak is population inference.
3. **Output hardening check.** Rerun the tests with the outputs you actually expose (for example rounded scores or top-1 label) to see how much the API design helps.

## Mitigations

- Expose the least output detail the product needs: top-1 labels, coarse confidence bands, no logits.
- Avoid models whose classes are individuals, or put them behind strong access control.
- Do not share raw gradients in federated settings without secure aggregation and DP.
- Use DP training to bound the membership-linked part of attribute leakage.
- For population inference, the controls are product and policy decisions (do not build the feature, restrict who can query), not training tricks.

## Worked example

*Execution status: executed. The script below was run with Python 3.13 and NumPy 2.2 when this chapter was written; the output shown is what it printed. It uses synthetic 8×8 "images", not faces.*

Four "people" each have a hidden prototype vector; we generate 50 noisy "photos" per person, train softmax regression, then invert each class from a blank input with gradient ascent and an L2 prior. We compare the reconstruction with the true prototypes.

```python
import numpy as np
rng = np.random.default_rng(2)
d, k, per = 64, 4, 50                       # 8x8 "images", 4 classes (people), 50 photos each
protos = rng.normal(size=(k, d))            # each person's true average face
X = np.vstack([p + 0.8 * rng.normal(size=(per, d)) for p in protos])
y = np.repeat(np.arange(k), per)
W = np.zeros((d, k)); b = np.zeros(k)
Y = np.eye(k)[y]
for _ in range(2000):                      # softmax regression
    Z = X @ W + b; P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    W -= 0.1 * (X.T @ (P - Y) / len(X) + 1e-3 * W); b -= 0.1 * (P - Y).mean(0)
def invert(target, steps=300, lr=0.1, lam=0.05):
    x = np.zeros(d)                         # attacker starts from a blank input
    for _ in range(steps):
        z = x @ W + b; p = np.exp(z - z.max()); p /= p.sum()
        grad = W @ (np.eye(k)[target] - p) - lam * x   # ascend log p(target|x) with L2 prior
        x += lr * grad
    return x
for t in range(k):
    x = invert(t)
    corr = [np.corrcoef(x, protos[j])[0, 1] for j in range(k)]
    print(f"target class {t}: corr with true prototype = {corr[t]:.2f}; best other = {max(c for j,c in enumerate(corr) if j!=t):.2f}")
```

Output:

```text
target class 0: corr with true prototype = 0.88; best other = -0.18
target class 1: corr with true prototype = 0.85; best other = -0.22
target class 2: corr with true prototype = 0.80; best other = -0.09
target class 3: corr with true prototype = 0.78; best other = -0.18
```

**Reading it.** Using only the trained weights and a class label, the attacker recovers a vector that correlates 0.78 to 0.88 with each person's true prototype. That is the face-recognition harm in miniature. It is a prototype, not any one photo; extend the script to compare the reconstruction with each individual training row and check that it is not closer to one of them than to the mean. Linear models make inversion easy; deep models need stronger image priors but the principle is the same.

## Common pitfalls

- Calling any accurate prediction of a sensitive attribute a "privacy attack" without the member versus non-member control.
- Assuming DP fixes population-level inference. It bounds individual contributions only.
- Testing inversion with full logits when the API exposes only labels, or the reverse.
- Sharing per-client gradients in federated learning on the grounds that "no raw data leaves the device".
- Ignoring embeddings and intermediate features exported to other teams; these invert more easily than final outputs.

## Exercises

1. Modify the worked example so the attacker sees only the top-1 label. Use random search or finite differences instead of gradients and compare reconstruction quality.
2. Build an attribute-inference test on a tabular dataset you own: hide one column, train a model with it as a feature, and compare attacker accuracy on members against non-members.
3. Add DP-SGD (Chapter 5) to the worked example's training loop and plot reconstruction correlation against noise multiplier.

## Further reading

- Fredrikson, Jha, Ristenpart, "Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures", ACM CCS 2015
- Aggarwal and Yu (eds.), *Privacy-Preserving Data Mining: Models and Algorithms*, Springer, 2008
- Sun et al., "When Machine Learning Meets Privacy in 6G: A Survey", IEEE, 2020
- Meta, *Llama 2 Responsible Use Guide*, 2023
- Zhu, Liu, Han, "Deep Leakage from Gradients", NeurIPS 2019, arXiv:1906.08935
- Sibling book: **adversarial-ml-security-classifiers**
