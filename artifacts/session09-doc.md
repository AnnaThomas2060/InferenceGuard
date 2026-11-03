# Session 09 — Comprehension of the Revised Evaluation Interface

## Overview

Session 09 examines whether the revised InferenceGuard interface makes privacy results and evaluation information easier to understand following the findings of Session 08. The revised interface uses simpler labels, an original-versus-suggested text comparison, a plain-language outcome summary, and expandable sections for alternative methods and detailed metrics.

The current session includes one participant, P01, completing generative scenario S03 (medication follow-up). The test was conducted on November 2, 2026.

The evaluation focuses on:

- perceived clarity of the revised interface;
- comprehension of simplified labels and their visible explanations;
- interpretation of numerical evaluation results and method descriptions; and
- the relationship between participant acceptance and the actual content of the suggested text.

The central finding is that **the participant considered the revised interface clearer overall, but still did not fully understand most items in the method-comparison and detailed-evaluation sections**. Retention of original details was the only item in Detailed evaluation · numbers & methods explicitly reported as understandable.

These findings are exploratory observations from one participant. They do not establish a causal effect of the redesign or general usability performance.

## Study Materials

- **`01.json`**: exported input, system response, rewrite output, evaluation metrics, diagnostic information, and timing data for session `01`.
- **`session09_record.json`**: participant observations, task context, researcher coding, and protocol record for P01/S03.
- **Facilitator's supplementary feedback**: the participant did not fully understand the items shown in those screenshots except Retention of original details.

The structured record codes warning comprehension as `partial` and the unclear-content field as `unclear`. Session 09 recording link: https://drive.google.com/drive/folders/1UGIznnAc8UdfpSyk06daC_Ris45KfdkO?usp=drive_link
---

## Current Participants

| Participant | Scenario | Mode | ADR | Warning Comprehension | Rewrite Decision | Usefulness | Would Use |
|---|---|---|---:|---|---|---|---|
| P01 | Medication Follow-Up (S03) | Generative | 1.0 | Partial | Accept as is | 9/10| Yes |

ADR denotes anchor disclosure ratio. All recorded anchors were used: explicit age, medication names, multiple prescribers, a GP practice, and a named locality. No pre-submission evaluator intervention or protocol deviation was recorded. 

---

## Preliminary Findings

### 1. The Participant Reported Better Overall Clarity

The recorded adoption reason was: “It's clearer than last time.” P01 rated usefulness as 9, indicated willingness to use the system before sending a query, and accepted the suggestion without edits.

This supports a finding of improved perceived clarity from the participant's perspective. It does not establish which change helped, or whether the participant correctly understood each result.

The warning was summarized as “age is the main concern,” consistent with age being the system's primary attribute. The structured comprehension assessment nevertheless remained `partial`.

### 2. Simplified Labels and Visible Explanations Still Left Most Detailed Items Unclear

In supplementary feedback about the two screenshots, the participant reported incomplete understanding of the following items:

| Interface Item | Reported Understanding |
|---|---|
| UNDERSTAND YOUR EXPOSURE | Fully understandable | 
| REVIEW YOUR WORDS | Fully understandable | 
| Consistent identity aliases | Not fully understood |
| Basic personal-information masking | Not fully understood |
| Remaining privacy capacity | Not fully understood |
| Added risk from this message | Not fully understood |
| Fallback evaluation explanation | Not fully understood |
| Text similarity | Not fully understood |
| Support for rewritten content | Not fully understood |
| Retention of original details | Reported as understandable |
| Overall meaning-preservation score | Not fully understood |

This feedback should not be interpreted as complete failure to understand every item or as verified mastery of Retention of original details. No independent paraphrase or correctness assessment was supplied for each term.


### 3. A Task-Related Question May Be Easier to Understand Than an Abstract Metric

Retention of original details directly addresses a practical concern: whether the suggested version keeps information the user needs. Its relative comprehensibility suggests a candidate direction for future wording: concrete questions about the text and the user's task.

For example, Support for rewritten content could first ask whether the suggestion adds anything the original did not say, supported by a short example. Any displayed score must still explain its direction: a higher forward-entailment score suggests fewer unsupported additions, not more additions.

This is a design hypothesis arising from the feedback, not a tested explanation for why one label was easier to understand. Retention of original details was the only reviewed item reported as understandable. Future testing should examine whether concrete questions and examples make the other evaluation items easier to understand.


---

## Observed System Behavior and Metrics

The input sought help deciding whether to contact a clinic again about medications issued by different prescribers. The text included an explicit age, medication details, and a named GP-practice locality. This report evaluates the system's handling and communication of that text, not the medical appropriateness of the query or any treatment decision.

| Metric | Recorded Result | Interpretation for This Session |
|---|---|---|
| Overall Risk | 0.45, MEDIUM | Highest estimated input-attribute risk; not a validated identification probability. |
| Primary / secondary attributes | Age / occupation | Age and occupation both scored 0.45; age was returned as primary. |
| Attribute scores | Age 0.45; occupation 0.45; location 0.05; education 0.05 | The named locality received a low inferential score despite being detected by the explicit-information stage. |