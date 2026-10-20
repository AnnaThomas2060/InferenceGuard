# InferenceGuard — Lean Canvas

**Owner:** Govind (Product) · **Last revised:** 2026-10-20 (Session 07) · **Status:** v2

Revised weekly. Changes are summarised in the "Lean canvas changes" section of that
week's report rather than tracked here, so this file always reads as current.

---

## 1. User and problem

**User.** Someone who pastes personal context into a general-purpose LLM — a student
asking for advice on a co-op, someone drafting a message about a workplace conflict,
someone asking a health or immigration question. They are not privacy experts. They
are aware enough to strip their name and email, and they assume that is sufficient.

**Problem.** It is not sufficient. A capable model recovers age, location, occupation
and education from ordinary detail that contains no identifier at all. Our own Session
04 baseline shows the shape of it: Presidio finds explicit spans in *"My co-op ends in
December, and I take the Green Line to campus"* — it tags the date and the transit line
— but nothing in its output addresses that "co-op" plus "campus" implies an
undergraduate, and that a Green Line commute narrows location to one metro area. The
user removed what looked identifying and left the inference intact.

The leak also compounds. Four messages that are individually unremarkable can describe
one person, and nothing in the user's interface shows that accumulating.

**What causes the pain today.** There is no moment in the workflow where anyone
inspects the text between typing and sending. The user's only signal is their own
intuition about what "counts" as personal, and that intuition is calibrated for
explicit identifiers.

## 2. Alternatives

| Alternative | What it does | Why it leaves the problem open |
|---|---|---|
| Doing nothing / self-censoring | User guesses what to omit | Guessing is exactly the failing skill; people redact names and keep the commute |
| Presidio and similar PII redactors | Tag and replace explicit spans | Built for identifiers, not implications. Zero coverage of the cues that carry the inference |
| Provider privacy controls (opt-out of training, chat history off) | Limit retention and reuse | Governs what the provider does with the text; the text still leaves the device fully intact |
| Local LLMs | Nothing leaves the machine | Solves it by giving up the frontier model. Our user wants the good model *and* the privacy |
| Manual rewriting | Works, when done well | Slow, and the user is back to guessing which details matter |

The gap we sit in: nobody rewrites for *inference* rather than identifiers, and nobody
does it locally.

## 3. Value proposition

> **We help people who paste personal context into an LLM reduce what a model can infer
> about them — not just what they explicitly wrote — before the text leaves their device,
> and without losing the detail their question depends on.**

The honest version of the pitch is the tradeoff, not the protection. Deleting everything
is perfect privacy and a useless product. What we are actually claiming is a better
privacy–utility curve than redaction, measured rather than asserted.

## 4. Distribution

Not solved, and we should stop describing it as if it were.

Today the only surface is a Colab notebook, which reaches nobody outside the team. The
plausible progression:

1. **Now** — web UI in the repo, paste-in / paste-out. Enough to run user sessions against.
2. **Realistic for this semester** — browser extension that intercepts the text box on
   the major chat UIs and offers a rewrite before send. This is where the product
   actually belongs: the protection has to sit in the moment of sending, not in a second
   tab the user has to remember.
3. **Beyond scope** — OS-level clipboard or keyboard integration.

**Open question for Session 05:** whether anyone will accept a second step before hitting
send. That is the distribution risk and it is cheaper to test than to build — the user
sessions should measure abandonment, not just rewrite quality.

## 5. Cost per user request

Local-first, so there is no per-request API cost. The cost is the user's own compute and
the patience it consumes — and as of Session 07 we have measured it rather than feared it.

Apple M4 Pro, 12 cores, CPU only. Per request, end to end:

| path | p50 | p95 | peak RSS |
|---|---:|---:|---:|
| heuristic rewriter | 53 ms | 80 ms | 1.0 GB |
| Qwen3-0.6B rewriter | 1,617 ms | 3,571 ms | 4.2 GB |

Per stage, the detector is effectively free — Presidio 0.06 ms, ModernBERT ~26 ms, the
utility check 0.1 ms. **The rewriter is the entire cost**: 1,561 ms of the 1,617.

**Marginal cash cost: ~$0.** That is still a real advantage over any hosted competitor,
and worth saying plainly: at zero marginal cost there is no usage cap to enforce and no
commercial reason to retain user text.

**But the latency finding is uncomfortable and we should not soften it.** 1.6 s median
and 3.6 s at the 95th percentile, holding 4.2 GB resident, on a high-end laptop, is not
something that sits easily in front of a send button. The fast path is 30x quicker and
produces exactly the output Session 06 participants rejected. **We do not yet have a
configuration that is both fast enough and good enough**, and that is now the central
product problem rather than a performance footnote.

Three things follow. A 4.2 GB resident footprint rules out the browser-extension route
on ordinary machines until it comes down. Latency has to be measured on mid-range
hardware, not only an M4 Pro, before we quote it to anyone. And the rewrite may have to
become opt-in per message rather than automatic, so the user pays the 1.6 s only when
they have decided the message is worth it.

(Evidence: `artifacts/latency/cpu-apple-m4-pro_2026-10-17_*.json`, PR #56. Measured on
base Qwen3-0.6B weights with no LoRA adapter, so these are generation-latency figures
only and say nothing about rewrite quality.)

## 6. North-star metric

**Held-out attribute leakage rate** — the share of protected target attributes that an
independent attacker, never used during training or tuning, still infers correctly.
Lower is better.

**Guardrail:** utility must hold. A leakage drop that comes with a rewrite people reject
is not progress, so the metric is only reported alongside rewrite acceptance and
semantic similarity.

*Session 04 status, stated honestly:* we cannot compute this yet. The held-out adversary
(Phi-4-mini-instruct) is not implemented, so the report uses a proxy — before/after risk
delta from the same estimator that guided the rewrite, which is a weaker claim and
should be labelled as one every week until the real attacker exists.

*Proposed change for Session 05:* the proxy reported in Session 04 was a single
attribute's delta. On the resume row of `data/logs/session04_usage_log.csv` that reads
0.852 while overall risk moved 0.01 and the rewrite left a name, an email and a phone
number in place. Any headline number we quote should be the worst attribute, not the
best one. `src/product/risk_bands.py` enforces this in code.

## 7. Ethics and privacy

- **Data rights.** SynthPAI (CC-BY-NC-SA-4.0), fully synthetic, no real individuals.
  Non-commercial terms suit a course project and would need revisiting for anything else.
- **No real personal data in user testing.** Participants work from synthetic scenarios
  in `docs/product/user_test_scenarios.md`. If a participant volunteers their own text
  anyway, it is processed and discarded — never committed, never in a report.
- **Bias.** Attribute inference is stereotype-shaped by construction: a model guessing
  occupation from phrasing is reproducing a correlation from its training data. Two
  consequences we accept as obligations — per-attribute error analysis rather than one
  averaged score, and output phrased as *"an attacker could infer"* and never *"you are"*.
  The second is enforced in `describe()` and tested.
- **The claim we must not make.** "Inference risk reduced", never "you are anonymous".
  A user who over-trusts the tool and shares more than they otherwise would is worse off
  than a user with no tool at all. This is the failure mode with the highest cost and it
  is a *product* failure, not a model failure.
- **Retention.** Session state is local, raw messages are not logged by default,
  pseudonym maps are cleared on reset.

## 8. Risk

**Biggest threat to viability:** that the protection does not survive contact with an
attacker we did not optimise against. Everything measured so far uses the same estimator
that drives the rewrite — a closed loop that can only flatter us. If leakage under
Phi-4-mini-instruct is close to the unprotected baseline, the core claim fails.

**Cheapest way to find out:** run the held-out attacker against the rewrites we already
have. No training, no new data, one evaluation script. A negative result found early is
reportable work; found late it is a collapse.

**Session 07 added a second-order version of this risk, and it is the more frightening
one.** We tried to escape the closed loop and found that the instrument meant to do it
had never worked: the adversary decoded its own prompt as the model's answer and fell
back to a regex without reporting it, and the success test counted `"CA"` as a correct
guess for `"Chicago"`. Three measurement tools were wrong at once, and all three erred in
the direction that flattered us.

So the risk is no longer only *that protection may not survive an unseen attacker*. It is
*that we may not be able to tell*. The mitigation is structural rather than analytical:
every evaluation result now records which mode actually produced it, `strict=True` raises
instead of degrading silently, and CI runs the suite on every pull request. A number
whose provenance is not recorded should not be reported.

**Second risk:** that nobody tolerates an extra step before sending. Cheapest test is
measuring abandonment in the scheduled user sessions rather than asking people whether
they would use it.

**Third risk:** the rewriter is not reliably capable, and we cannot yet train it out.
The SFT set is 27 pairs — not a dataset — so the model in the demo is base weights with a
regex scrub behind it, which means we are partly measuring a regex. Session 06
participants rejected its output for exactly the reasons that predicts: over-scrubbing,
truncation, and one case where it returned the original text unchanged. Mitigation is to
rebuild the SFT set with a teacher LLM, not to reach for a bigger base model, since a
bigger model worsens the latency finding above.
