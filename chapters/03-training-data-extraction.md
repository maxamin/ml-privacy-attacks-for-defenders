# Training-Data Extraction from Language Models

Membership inference checks a record you already have. Extraction goes further: the attacker gets the model to emit training text it has memorised, without knowing it in advance. For generative language models this is the most direct privacy failure, because emitted text can contain names, phone numbers, keys and source code.

## What the research shows

Three lines of work define the field.

**Canaries and exposure.** Carlini, Liu, Erlingsson, Kos and Song, "The Secret Sharer" (USENIX Security 2019, arXiv:1802.08232), inserted random secrets such as "my social security number is ___" into training data and measured how highly the trained model ranked the true secret among all possible values. They defined **exposure** as log2 of the candidate-space size minus log2 of the secret's rank. A secret ranked first among 10 000 options has exposure of about 13.3 bits, the maximum. Exposure is the practical tool for defenders: plant canaries you control, then measure.

**Extraction from a production-scale model.** Carlini et al., "Extracting Training Data from Large Language Models" (USENIX Security 2021, arXiv:2012.07805), generated large numbers of samples from GPT-2, ranked them with membership signals (for example, the ratio of the model's perplexity to zlib compression size, or to a smaller model's perplexity), and confirmed verbatim memorised sequences including personal contact details. The method is generate-then-rank: sampling produces candidates, and a membership score picks the ones likely to be memorised.

**Aligned chat models.** Nasr et al., "Scalable Extraction of Training Data from (Production) Language Models" (arXiv:2311.17035, 2023), reported that alignment does not remove memorisation. They described a "divergence" prompt that caused a production chat model to leave its chat behaviour and emit memorised pre-training text, and they followed coordinated disclosure with the vendor before publishing. Treat the specific prompt as patched; the general lesson is that refusal training is a behavioural layer, not a data-removal layer.

Memorisation scales with repetition. Data that appears many times in the corpus is far more likely to be emitted verbatim, which is why deduplication is one of the most effective mitigations.

## Extraction surfaces beyond pre-training

| Surface | What leaks | Defender test |
|---|---|---|
| Pre-training corpus | Repeated web text, code, contact details | Sample-and-rank on your own checkpoints; canaries |
| Fine-tuning set | Customer tickets, chat logs, internal docs | Canaries in the fine-tuning set; prefix-completion tests |
| RAG index | Whole documents returned in context | Access-control tests on retrieval, not model tests |
| System prompt | Instructions, embedded keys | Prompt-leak tests; never put secrets in prompts |
| Embeddings | Approximate text via inversion | Treat embeddings as personal data |

RAG and system-prompt leaks are not memorisation, but users experience them the same way. Test them separately; see **llm-jailbreak-engineering-for-defenders** for prompt-leak testing.

## Mitigations and what they buy

- **Deduplicate** training data at document and substring level. Cheap, and it reduces verbatim regurgitation of repeated content.
- **Scrub** obvious secrets and identifiers before training (regex plus named-entity detection, plus secret scanners such as gitleaks for code corpora). Scrubbing misses context-dependent personal data.
- **Differential privacy** during fine-tuning (Chapter 5) bounds per-record influence. It is the only mitigation with a formal guarantee.
- **Output filters** that block exact matches against a corpus index (for example, a Bloom filter of n-grams). These reduce verbatim output, but can be bypassed with paraphrase or encoding requests and do not remove the data from the weights.
- **Rate limits and monitoring** on sampling APIs make large generate-and-rank campaigns visible.

## Worked example

*Execution status: executed. The script below was run with Python 3.13 when this chapter was written (about 5 seconds per run); the output shown is what it printed. It uses a toy character 4-gram model and synthetic text, not a neural network.*

We measure canary exposure as in "The Secret Sharer". The corpus has 5 000 random sentences plus 2 000 lines of the form "my pin is NNNN" with random PINs, so the format itself is common. We then insert the canary "my pin is 4721" 0, 1, 4 or 16 times and rank it among all 10 000 possible PINs by model likelihood.

```python
import math, random
from collections import Counter
random.seed(1)
words = "the a model data user record secure system log value number key report".split()
corpus = [" ".join(random.choice(words) for _ in range(12)) for _ in range(5000)]
corpus += [f"my pin is {random.randrange(10000):04d}" for _ in range(2000)]   # many other PIN-like lines
def exposure(repeats, secret="4721"):
    text = corpus + [f"my pin is {secret}"] * repeats
    chars = "\n".join(text)
    n = 4                                            # character 4-gram model, add-one smoothing
    grams = Counter(chars[i:i+n] for i in range(len(chars)-n+1))
    ctx = Counter(chars[i:i+n-1] for i in range(len(chars)-n+2))
    V = len(set(chars))
    def logp(s):
        s = "\n" + s
        return sum(math.log((grams[s[i:i+n]] + 1) / (ctx[s[i:i+n-1]] + V)) for i in range(len(s)-n+1))
    cands = [f"my pin is {k:04d}" for k in range(10000)]
    scores = sorted(((logp(c), c) for c in cands), reverse=True)
    rank = 1 + [c for _, c in scores].index(f"my pin is {secret}")
    return rank, math.log2(10000) - math.log2(rank)
for r in (0, 1, 4, 16):
    rank, ex = exposure(r)
    print(f"canary inserted {r:>2}x  rank={rank:>5}/10000  exposure={ex:5.2f} bits")
```

Output:

```text
canary inserted  0x  rank= 3026/10000  exposure= 1.72 bits
canary inserted  1x  rank=  460/10000  exposure= 4.44 bits
canary inserted  4x  rank=    2/10000  exposure=12.29 bits
canary inserted 16x  rank=    1/10000  exposure=13.29 bits
```

**Reading it.** With no insertion the canary sits mid-pack (exposure near zero, as expected by chance). One copy already lifts it into the top 5%. Four copies make it almost the top guess, and 16 copies give maximum exposure: an attacker who tries the model's top guess recovers the PIN. Repetition drives memorisation, which is why deduplication matters. For a real model, use the same procedure with the model's log-likelihood instead of the n-gram score; for large candidate spaces, Carlini et al. estimate rank by sampling rather than enumerating.

## Common pitfalls

- Planting canaries that look unlike real data. They should share the format of the secrets you worry about.
- Testing only exact-match regurgitation. Near-verbatim output with one changed digit is still a leak.
- Relying on output filters as the only control; the data is still in the weights and in any exported checkpoint.
- Forgetting that open-weight releases let attackers sample without rate limits or monitoring.
- Running extraction against a vendor's model without permission, or publishing extracted personal data. Report privately and redact.

## Exercises

1. Run the worked example with a 3-gram and a 6-gram model. How does model capacity change exposure for one inserted copy?
2. Add 20 canaries with different formats (email, API-key-shaped string, street address) to a small fine-tuning set for an open model you control, then measure each canary's rank with prefix completion.
3. Implement the zlib-ratio ranking from Carlini et al. (2021) over 1 000 samples from a small open model and manually inspect the top 20.

## Further reading

- Carlini et al., "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks", USENIX Security 2019, arXiv:1802.08232
- Carlini et al., "Extracting Training Data from Large Language Models", USENIX Security 2021, arXiv:2012.07805
- Nasr et al., "Scalable Extraction of Training Data from (Production) Language Models", 2023, arXiv:2311.17035
- Meta, *Llama 2 Responsible Use Guide*, 2023 (testing fine-tuned models for memorisation)
- gitleaks secret scanner: <https://github.com/gitleaks/gitleaks>
- Sibling books: **llm-jailbreak-engineering-for-defenders**, **code-model-data-poisoning**
