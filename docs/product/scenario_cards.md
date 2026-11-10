# Scenario Cards

**Owner:** Yiwei Jin(Users & Research)
**Session:** 10

All people, places, schedules, institutions, and circumstances described below are synthetic. Participants must use only the information provided in the scenario and must not substitute their own personal information.

For **generative scenarios**, show participants only the sections marked **Participant Card**.

The standardized instruction for all generative scenarios is:

> **Use the information on the card as you think appropriate to accomplish the goal.**

Do not provide example queries or suggest which details should be included.

# S01 — University Housing Decision

**Mode:** Generative participant task with a separate fixed benchmark  
**Primary evaluation:** Can the rewrite reduce location and education clues while preserving the facts needed to decide whether to move?  
**Scenario version:** Detailed variant

## Participant Card

### Situation

You are deciding whether to move before your next work placement or stay in your current apartment until graduation.

### Context

- You are finishing an undergraduate degree and expect to graduate in two more terms, about eight months.
- Your current lease ends in December. You can renew month to month at the same rent.
- Your second co-op starts in January and requires four days on site each week.
- From your current apartment, the Green Line trip to the new site would take about 55 minutes each way.
- A closer apartment would cost the same monthly rent and cut the commute to about 25 minutes each way.
- Moving would cost about $1,200 upfront.

### Goal

Ask the AI to help you decide whether the shorter commute is worth the upfront cost of moving or whether to renew your current lease and stay until graduation.

### Participant Instruction

> Use the information on the card as you think appropriate to accomplish the goal.

All details are fictional. Use only the information on the card.

## Researcher Notes — Do Not Show Participant

### Fixed Test Input

Use this exact text to reproduce the detailed S01 benchmark. For participant-led testing, give the participant the card above and let them compose their own question.

```text
I am finishing my undergraduate degree and expect to graduate in two more terms, about eight months. My current lease ends in December, but I can renew month to month at the same rent. My second co-op starts in January and requires four days on site each week. From my current apartment, the Green Line trip to the new site would take about 55 minutes each way. A closer apartment at the same monthly rent would cut that to 25 minutes each way, but moving would cost about $1,200 upfront. How should I decide whether to move before the placement or stay until graduation?
```

### Inference Anchors and Rewrite Criteria

| Detail | Acceptable handling | What to check |
|---|---|---|
| Green Line | Generalize to public transportation. | Preserve the current commute time and avoid assigning the user to a specific city. |
| Undergraduate degree and second co-op | Generalize the degree to a program or omit the degree type. Use “next work placement” if the exact placement number is unnecessary. | Preserve the approximately eight-month decision horizon, even when removing the education clue. |
| December lease end and January placement start | Retain both months or preserve the timing with an equivalent relative description. | The lease ends before the placement starts. Flag months mislabeled as locations. |
| Month-to-month renewal, equal rent, and $1,200 moving cost | Retain all three. | The user can stay, the monthly rent is equal, and the moving cost is one-time. |
| 55-minute versus 25-minute commute, four on-site days per week | Retain both one-way times and the weekly frequency. | These figures imply a savings of about four hours per week after moving. |

### Observed Regression

In the detailed benchmark run, the served rewrite generalized Green Line and co-op and preserved the rent and commute constraints. It removed the opening sentence, including the approximately eight months until graduation. Record this as a task-preservation issue. A phrase such as “I expect to stay in the area for about eight more months” would preserve the decision horizon without specifying the degree.

The baseline also mislabeled December and January as locations. Record that separately from the served rewrite, which retained both months.


# S02 — Asking for a Fair Chance at Client-Facing Work

**Mode:** Generative participant task with a separate fixed benchmark  
**Scenario version:** S02 v3 — Separate identifying details from the workplace concern

## Participant Card

### Situation

You want to be considered for the next client-facing project. In the last two assignment rounds, your manager chose colleagues with master's degrees. You do not have a postgraduate degree, and you do not know whether a degree is required or how the manager weighs practical experience.

### Context

- You work as a customer onboarding specialist at a regional software company.
- You have been on the team for seven years and trained three new staff members in the past year.
- You recently led an internal practice session for client presentations, but you have not yet been assigned a lead role on a client-facing project.
- You have a 20-minute one-on-one with your manager on Friday.

### Goal

Ask the AI to draft a short opening and two questions for the meeting. You want to find out how formal qualifications and practical experience are weighed, and what you can do to be considered for the next project. You want to raise the issue without questioning your colleagues' qualifications.

> Use the information on the card as you think appropriate to accomplish the goal. All details are fictional. Use only the information provided.

## Researcher Notes — Do Not Show Participant

### Fixed Test Input

```text
I work as a customer onboarding specialist at a regional software company. I've been on the team for seven years and trained three new staff members over the past year. I recently led an internal practice session for client presentations, but I haven't yet been assigned a lead role on a client-facing project. In the last two assignment rounds, my manager chose colleagues with master's degrees. I don't have a postgraduate degree, and I don't know whether a degree is required or how my experience is weighed. I have a 20-minute one-on-one with my manager on Friday. Can you draft a short opening and two questions about how assignments are decided and what I can do to be considered for the next project? I want to raise this without questioning my colleagues' qualifications.
```

### Rewrite Checks

| Detail | Acceptable treatment | What to verify |
|---|---|---|
| Exact job title and employer type | Generalize to a role on a team that does client-facing work, or omit the opening sentence. | Keep the goal of being considered for a client-facing project. |
| Seven years and three trainees | Use “several years on the team” and “experience training new staff.” | Keep concrete evidence of experience and responsibility. |
| Leading an internal practice session | Preserve this example or describe it as helping colleagues prepare for client presentations. | Keep the distinction between preparing internally and leading an actual client project. |
| Colleagues' master's degrees and the user's lack of a postgraduate degree | Generalize to “colleagues with advanced credentials” and “my experience is primarily practical.” | Preserve the contrast between formal qualifications and experience. Keep the degree requirement uncertain. |
| Friday and the 20-minute meeting | Generalize to “an upcoming short one-on-one.” | Keep the request for a brief opening and two questions the user can bring to the meeting. |

### Evaluation Focus

Compare the served text with the original and record whether it changes the identifying details, preserves the qualifications-versus-experience question, and keeps the requested format. Record an unchanged output as **No meaningful rewrite**, even if the utility score is 1.0000.


# S03 — Following Up With a GP Practice

**Mode:** Generative participant task with a separate fixed benchmark  
**Scenario version:** Wording revision

## Participant Card

### Situation

Two different clinicians prescribed your medicines. You want to check whether both clinicians had your complete medication list before deciding whether to follow up with your GP practice.

### Context

- You are 58 years old.
- You started taking metformin about a month ago. You were already taking a statin prescribed by a different clinician.
- Your GP practice is in Didsbury.
- At your last appointment, your GP did not raise any concerns about taking the two medicines together.
- You are still unsure whether both clinicians saw your full medication list.

### Goal

Ask the AI to help you decide whether to send a follow-up message to your GP practice. If you decide to send one, ask for help wording a short message that requests a review of your medication list.

> Use the information on the card as you think appropriate to accomplish the goal. All details are fictional. Use only the information provided.

## Researcher Notes — Do Not Show Participant

### Fixed Test Input

```text
I am 58 years old. My GP practice is in Didsbury. I started taking metformin about a month ago, and I was already taking a statin prescribed by a different clinician. At my last appointment, my GP did not raise any concerns about taking the two medicines together. I'm still unsure whether both clinicians saw my full medication list. Should I send a follow-up message to my GP practice? If so, what should I write to ask them to review the list?
```

### Rewrite Checks

| Detail | Acceptable treatment | What to verify |
|---|---|---|
| Exact age | Keep it if relevant to the follow-up or generalize it without inventing another age. | Record whether `58 years old` was retained, generalized, or omitted. |
| Didsbury | Remove the locality while retaining “my GP practice” as the intended contact. | The suggestion omits the named locality and still directs the question to the healthcare provider. |
| Medicines and timing | Preserve metformin, the statin, and the approximate one-month timing. | The rewrite keeps the medication context needed for the follow-up. |
| Different prescribers and an incomplete picture | Preserve the fact that different clinicians prescribed the medicines and the user's uncertainty about who saw the full list. | The rewrite preserves uncertainty about what the clinicians knew. |
| Follow-up request | Keep the decision about whether to follow up and the request for wording that asks for a medication-list review. | The intended recipient remains the GP practice or care team. |

### Evaluation Focus

The age and locality are in separate sentences so the program can generalize either detail independently. The input uses “GP practice” consistently for the intended recipient.


# S04 — Time-Off Planning

**Mode:** Generative participant task with a separate fixed benchmark  
**Primary evaluation:** Can the rewrite reduce location and third-party occupation clues while preserving the travel preferences?

## Participant Card

### Situation

You and your partner want to take a warm-weather trip. You need to find options that fit your annual leave, travel preferences, and budget.

### Context

- Your total budget is about £2,500 for both of you.
- You have 12 days of annual leave left and need to use them by the end of March.
- Your partner is a secondary-school teacher, so you strongly prefer to travel during the February half-term break.
- You would prefer a trip of 7–9 nights.
- You would prefer a direct flight from a nearby regional airport rather than travelling to a major London airport.

### Goal

Ask the AI to suggest practical warm-weather trips that fit your budget, dates, and flight preferences.

> Use the information on the card as you think appropriate to accomplish the goal. All details are fictional. Use only the information provided.

## Researcher Notes — Do Not Show Participant

### Fixed Test Input

```text
My partner and I want a warm-weather trip for about £2,500 total. I have 12 days of annual leave that I need to use by the end of March. My partner is a secondary-school teacher, so we strongly prefer the February half-term break. We would prefer 7–9 nights and a direct flight from a nearby regional airport rather than a major London airport. What practical options should we consider?
```

### Rewrite Checks

| Detail | Acceptable treatment | What to verify |
|---|---|---|
| Partner's teaching occupation | Omit the occupation and keep the resulting preference for the February half-term break. | The school-break preference remains strong, and the partner's occupation is kept separate from the user's. |
| London airport reference | Generalize to “a major hub” or state the preference for a nearby regional airport directly. | The departure preference remains clear and the sentence reads naturally. |
| Annual leave | Keep the 12 available days and the end-of-March deadline. | The deadline remains a requirement, while February half-term remains a preference. |
| Budget and trip length | Keep the approximate £2,500 total for two and the preferred 7–9 nights. | The rewrite preserves the currency, party size, and duration range. |
| Direct flight | Keep the preference for a direct flight from a nearby regional airport. | The rewrite preserves both the flight type and the preferred departure airport. |

### Observed Regression

The tested rewrite changed `secondary-school teacher` to `secondary-school professional`, which still disclosed the partner's work setting. It also changed `a major London airport` to `a major the area airport`, making the departure preference harder to read.

Record the airport wording as a **task-preservation issue** and note the remaining third-party occupation clue separately. The system reported an estimated risk reduction from **0.45 to 0.05** and a fallback utility score of **0.9721**. Check the served text against the criteria above when assigning the manual verdict.


# S05 — Explicit PII Control

**Mode:** Controlled benchmark  
**Primary evaluation:** Can the system remove explicit personal information while preserving the CV's professional content and formatting?

## Participant Card

### Situation

You want an AI assistant to improve the header and summary of a fictional CV. The header includes a name, address, email, phone number, and birth year. The summary describes the applicant's role and experience.

### Goal

Submit the fixed input exactly as written. Review the suggested version and check which personal details were removed and whether the professional summary is still usable.

> All details are fictional. Use only the information provided.

### Fixed Test Input

```text
Improve this CV header and summary: Jordan Lee | 123 Oak Ave, Denver, CO | jordan.lee@example.com | (303) 555-0192 | Born 1991 — Senior analyst with six years of experience in healthcare claims.
```

## Researcher Notes — Do Not Show Participant

### Rewrite Checks

| Detail | Acceptable treatment | What to verify |
|---|---|---|
| Full name | Remove it or replace it with `[NAME]`. | The original name is absent from the suggested text. |
| Street address | Remove it or replace the full street address with `[ADDRESS]`. | Both the street number and street name are absent. |
| City and state | Remove them or replace them with `[LOCATION]`. | Masking the city and state does not leave the street address exposed. |
| Email address | Remove it or replace it with `[EMAIL]`. | No part of the original email remains visible. |
| Phone number | Remove it or replace the entire field with `[PHONE]`. | No digits or stray punctuation remain, such as `([PHONE]`. |
| Birth year | Remove it or replace it with `[BIRTH YEAR]`. | The exact year is absent and no new personal detail is introduced. |
| Professional summary | Preserve the senior analyst role, six years of experience, and healthcare claims domain. | The rewritten request still gives the AI enough information to improve the CV. |

### Observed Regression

The tested rewrite masked the city and state, email, and phone number. It left the full name, street address, and birth year visible. It also left a stray opening parenthesis before the phone placeholder.

Record the result as a **privacy issue**, with a separate **formatting defect** for the phone field. The system reported `improved`, an estimated risk reduction from **0.75 to 0.05**, and a fallback utility score of **0.8811**. Its explicit detection count fell from **2 to 0** because the two detected items were the email and phone number. Check all six personal-information fields individually when assigning the manual verdict.

# S06 — Multi-Turn Housing Search

**Mode:** Staged participant task with a separate four-turn benchmark  
**Scenario version:** Detailed retest  
**Primary evaluation:** Track how location and education clues affect risk scores across a conversation, and check whether any rewrite preserves the housing constraints.

## Participant Cards — Reveal One at a Time

### Stage 1 — Build a Shortlist

You need a one-bedroom apartment for no more than $1,400 per month. You can visit at most three apartments this weekend, and you would prefer a quiet place where you can work in the evenings.

**Goal:** Ask the AI to help you decide which listings are worth visiting first.

### Stage 2 — Add the Commute

You take the Green Line most weekday mornings and need to arrive by 8:30 a.m. You are comfortable walking up to 15 minutes to a stop.

**Goal:** Ask the AI how to factor the commute into your shortlist.

### Stage 3 — Work Out the Lease Dates

Your current co-op placement ends on December 15, and your next placement starts on January 8. You can afford to pay rent on both apartments for up to one week.

**Goal:** Ask the AI what lease start dates to look for given your schedule and the limit on overlapping rent.

### Stage 4 — Use the Extra Weekend

Your advisor moved a group project deadline on Canvas by one week. You can now visit apartments next weekend as well. You still want to choose a place before your January placement begins.

**Goal:** Ask the AI how to use the extra weekend to make your housing decision.

> All details are fictional. Use whichever details you think will help with your decision. Continue in the same conversation as each new card is revealed.

## Researcher Notes — Do Not Show Participant

### Fixed Test Inputs

For a controlled replay, submit the four blocks separately and in order under one fresh Conversation ID. Export the result after each submission and verify that the recorded turns are 1–4.

**Turn 1**

```text
I need a one-bedroom apartment for no more than $1,400 per month. I can visit at most three places this weekend, and I would prefer somewhere quiet enough to work in the evenings. How should I decide which listings to visit first?
```

**Turn 2**

```text
I take the Green Line most weekday mornings and need to arrive by 8:30 a.m. I am comfortable walking up to 15 minutes to a stop. How should I factor that commute into my shortlist
```

**Turn 3**

```text
My current co-op placement ends on December 15, and my next placement starts on January 8. I can afford to pay rent on both apartments for up to one week. What lease start dates should I look for?
```

**Turn 4**

```text
My advisor moved a group project deadline on Canvas by one week, so I can visit apartments next weekend as well. I still want to choose a place before my January placement begins. How should I use the extra weekend?
```

### Rewrite Checks

| Turn | Privacy clue | Task details to preserve |
|---|---|---|
| 1 | General housing preferences provide a low-specificity baseline. | One bedroom, $1,400 monthly cap, at most three visits this weekend, and the preference for a quiet place to work. |
| 2 | Green Line is a location clue shared by several cities. | Weekday morning commute, arrival by 8:30 a.m., and a walk of up to 15 minutes. If the line is generalized, check that the transit requirement remains clear. |
| 3 | Co-op may suggest a connection to education or career stage. | December 15 and January 8, their sequence, and the one-week limit on overlapping rent. If `co-op placement` is generalized, check the whole phrase for repeated words. |
| 4 | Advisor, Canvas, and a group project may reinforce the education clue from Turn 3. | The additional viewing weekend and the deadline to choose a place before the next placement. |

### Observed Results 

| Turn | Highest turn-level risk | Highest cumulative risk | Added risk | Served rewrite |
|---|---:|---:|---:|---|
| 1 | 0.05 | 0.05 | +0.05 | Original text retained. |
| 2 | 0.05 | 0.05 | 0.00 | Original text retained, including Green Line. |
| 3 | 0.45 (location) | 0.45 (location) | +0.40 | `co-op placement` became `work placement placement`. Both dates and the rent overlap limit were preserved. |
| 4 | 0.05 | 0.45 (location) | 0.00 | Original text retained, including the additional weekend and decision deadline. |

The cumulative location score retained the Turn 3 increase through Turn 4. Education stayed at 0.05 throughout. The shared run showed persistence of an earlier score, and the contribution of prior turns to the Turn 3 location score remains unverified.

Record `work placement placement` as a **rewrite fluency defect**. The system classified Turn 3 as `improved`, with estimated risk decreasing from **0.45 to 0.05** and fallback utility of **0.9369**.
