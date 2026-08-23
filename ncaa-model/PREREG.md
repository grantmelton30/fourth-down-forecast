# PRE-REGISTRATION LEDGER

Every judgment call in this build, flagged with whether it was fixed **before** the number
it could influence was seen. Created in Phase 0B.

A `preregistered: false` flag is not an accusation. Most of these decisions are defensible
and several are plainly correct. The flag records **how much a result resting on that
decision is worth**, which is a different question from whether the decision was reasonable.
A finding that survives only under choices made after seeing it is worth close to nothing;
the same finding under choices fixed in advance is worth a great deal.

The rule going forward: **a decision made after seeing its own outcome may not be used to
support a positive result.** It may still be used to kill one. Asymmetric on purpose —
that asymmetry is what stops a search from converging on a false positive.

---

## Scoring key

| flag | meaning |
|---|---|
| `true` | Fixed in writing before the affected quantity was computed. |
| `partial` | The trigger to revisit came from data, but the replacement value was fixed before the affected quantity was seen. |
| `false` | Decided after seeing a number the decision affects. |
| `outcome-blind` | Decided after seeing data, but the data seen is **provably independent** of `b`, CLV, or the win record. Weaker than `true`, much stronger than `false`. |

---

## The ledger

| id | decision | preregistered | affects | notes |
|---|---|---|---|---|
| **D1** | Drop `consensus` from `provider_priority` | `outcome-blind` | universe | Triggered by `GATE_OPENER_COVERAGE` failing at 16.8%. The quantity inspected was **opener coverage per provider** — a property of the data feed, computed before any projection existed. It cannot correlate with `b`. Correct regardless. |
| **D2** | Exclude seasons 2019–2020 | `true` | universe | The missing-at-random test and its four thresholds were written down before the comparison was run, and all four failed. |
| **D2a** | Override the "more conservative window governs" clause | **`false`** | **`b`, CLV, every headline** | **The highest-risk entry in this file.** The clause was overridden *after* seeing t = 2.844 (2021–2025) against t = 1.908 (2023–2025). The stated reasoning — that these are one effect at two sample sizes, and that 2023–2025 contaminates a tuning season into the holdout — is sound and was argued in writing. It remains a decision made with the t-statistics visible. **Every result that depended on it is now void anyway**: the close-anchored full-FBS test (Phase 0A) reads `b = −0.004` on all 2,985 games, so no window choice rescues anything. |
| **D3** | 12-cell hyperparameter grid, scored on held-out `t(b)` | `true` | `b` | Declared before the first sweep; all twelve cells reported, not the winner. Exemplary. |
| **D4** | Simulator never in the mean path | `true` | architecture | Transferred from the NFL null before the NCAA build started. |
| **D5** | Play-level efficiency so the garbage filter is real | `true` | features | Required by playbook §1.8, fixed before ingest. |
| **D6b** | Gain correction (rescale projection SD to market SD) | `outcome-blind` | `b`, `t` | Triggered by an SD ratio of 0.787 — a dispersion property, not an outcome. Its cost was then reported honestly and against interest (t fell 2.844 → 2.285), which is the behaviour a pre-registration regime is trying to produce. |
| **D6c** | `min_edge_points_total` 3.5 → 1.0 | `partial` | pick count, CLV | The *trigger* was data: at the fitted `b`, zero picks cleared on any slate — the filter was inert, not selective. The *replacement value* was derived from the arithmetic of the filter and fixed before any CLV number was computed. The weak point is that a threshold admitting ~20% of the slate is looser than the playbook's expected 8–15%, and that gap was never re-tested. |
| **D6d** | Exclude weeks 10, 13, 14, 15, 16 | **`false`** | CLV, pick count | Decided by `GATE_UNBIASED_BY_WEEK` after the gate had been run and the 1–6 record seen. The mitigation is real — the weeks were chosen by a bias measurement, invariant to shrinkage, and no outcome was consulted for the *selection* — but the decision to exclude at all followed a bad result. Note the same section correctly **retracted** an underpowered partition claim, which is the right behaviour. |
| **D6e** | Concentration caps, 15% team / 25% conference | **`false`** | CLV | The cap levels were set after measuring concentration on the 68 picks that already existed. Low impact: no team exceeded 15%, so the team cap never bound, and the conference cap cost 38% of the pick set for no CLV change. |
| **D7** | Spreads stay dark | `true` | scope | `b = +0.042, t = 0.80`. Emitting nothing on a null is the conservative direction. |

---

## Phase 0 additions

| id | decision | preregistered | notes |
|---|---|---|---|
| **P0-A** | Report the three D2a windows close-anchored | `true` | The windows were declared in D2a and are unchanged. Only the **anchor** moved, from the opener that Appendix A proved invalid as a measurement to the close. All three are reported whatever they say — that condition is what makes recomputing a declared window a correction rather than a search. |
| **P0-A2** | `REFERENCE_B = 0.20` as the alternative every null is judged against | `true` | Fixed at the effect size this build **already claimed** (`b = +0.195`), not chosen to make a null look adequate. Raising it would make weak samples appear adequately powered; it must not be tuned. |
| **P0-C** | Side-baseline adjustment to CLV — **attempted, failed, abandoned** | **`false`** | Invented after the valid control failed at 0.5846. It moved the control only to **0.5607**, still outside the 0.50 ± 0.03 gate, so it did not repair the metric and was not adopted. One post-hoc repair was attempted and reported; a second would be tuning a metric against its own control, which is the failure mode this phase exists to prevent. **CLV is deleted rather than adjusted.** |
| **P0-D** | `rmse_max_ratio = 1.0` | `true` | Fixed at the theoretical boundary (as accurate as the line you bet against), not at a level chosen to let the current model through — it fails at 1.035 totals / 1.104 spreads. |

---

## What Phase 0 changed about the record

Two entries in `NCAA_PLAYBOOK.md` Appendix A are corrected by Phase 0. Neither reverses the
null; one strengthens it and one repairs a metric that was wrongly condemned.

**0. GATE_CLV_CONTROL FAILS. CLV is deleted as a promotion metric.**
A projection with no information — `total_open + noise`, using only what is known at bet
time — earns **0.5846** raw and **0.5607** after the side-baseline correction, across 200
seeds. The gate requires 0.50 ± 0.03. Diagnosis: totals lines in this universe drift down
about a point, the `cover_prob` filter preferentially admits UNDER picks (OVER share 0.374),
and an UNDER pick collects CLV for free whenever the line drops. Correcting for the global
side base rate is not enough, because the filter also selects *which* games drop. CLV on
this pipeline measures the market's habits and the filter's tilt, not the pick. **It may not
be used to authorise anything.** The model's 0.7049 is not evidence and is retained only as
a record of what was computed.

**1. Appendix A Test 2 is invalid, and its conclusion about CLV does not follow.**
The zero-skill control was defined as `total_close + noise`. Since `side = sign(model −
total_open)` and CLV grades `sign(total_close − total_open)`, that control's side reduces to
`sign(line_move + noise)` — it was handed the exact quantity it is graded against. It scored
0.79–0.82 across 200 seeds, *higher than the model's 0.7049*, because it had lookahead, not
because CLV is meaningless. **A control given the closing line cannot control for a metric
that grades on the closing line.**

**2. The restricted-universe null was never adequately powered; the full-FBS one is.**
At n = 1,543 the close-anchored test carries 64.9% power against `b = 0.20` (MDE 0.239).
"No edge" there was absence of evidence. The full FBS window — declared in D2a, recomputed
on the corrected anchor — carries n = 2,985, **93.2% power**, and reads `b = −0.0040,
t = −0.07`. That is the properly powered close-anchored test the build never had, and it
settles the question: **evidence of absence, not absence of evidence.**

Appendix A **Test 1 is untouched** and was always the decisive one. The null stands, and
now stands on a stronger footing than when it was written.

---

## P1. FORWARD TRACKING RULE — totals, edge-capped. Declared 2026-08-19.

Declared **before the 2026 season has played a single game**, so every graded result under
it is genuinely out of sample. This is the only evidence source this build has not already
exhausted: five seasons of backtest have been sliced repeatedly, and slicing them again
cannot settle anything.

### The rule, stated so December cannot reinterpret it

* **Market:** totals only. Spreads are excluded — they measured 50.1% on edges >= 4, below
  breakeven, and the mechanism (team-specific news moves the spread and cancels in the
  total) is documented in D17.
* **Universe:** the `restricted` window — FBS vs FBS, excluding P5-vs-P5. Unchanged from the
  backtest that produced the baseline below.
* **Trigger:** bet when `0.5 <= |model_total - market_total| < 6.0`.
* **Side:** OVER if `model_total > market_total`, UNDER if below.
* **Excluded:** edge < 0.5 (no opinion) and edge >= 6.0 (the cap).
* **Stakes:** flat. One unit per qualifying game, no sizing, no Kelly, no discretion.
* **Grading:** at the number actually available when the bet is recorded, which is the real
  money question. The closing number is recorded alongside for comparability with the
  close-anchored baseline. Pushes are no action.
* **No mid-season changes.** Any adjustment to trigger, cap, universe or stake voids the
  test and starts a new one under a new id.

### What is being tested, and what may NOT be cited

Backtest baseline, 2021-2025, restricted, close-anchored: **54.0% over 1,063 bets**
(95% CI 51.0-57.0), against a 52.4% breakeven at -110.

**That number is not evidence and may not be cited as support.** Two reasons, both by this
ledger's own rule:

1. **The cap of 6.0 was chosen after seeing the table it was chosen from.** Flag: `false`.
   Win rate by cap was 53.6% (4), 54.0% (6), 53.3% (8), 53.3% (10), 53.1% (uncapped) --
   6 is the best-looking cell of five examined, and the differences between them are well
   inside noise. Per the ledger's standing rule, a decision made after seeing its own
   outcome may not support a positive result.
2. **Edge size carries no measured information.** A logistic fit of win rate on edge size
   across all 1,405 totals bets gives slope -0.012, **t = -0.70**. There is no detectable
   relationship, which means there is no principled place to cut and the cap is a judgment
   call rather than a finding. An earlier claim in this session that ">= 10 points should be
   ignored" rested on n=52 and was overstated; it is withdrawn.

Pre-registering the rule now is precisely what converts a post-hoc threshold into a
legitimate test: the choice is frozen, and the 2026 games it will be graded on did not exist
when it was made.

### Why this cannot be settled the usual way

To establish with 95% confidence that the true rate exceeds 52.4%, given an observed 53.1%,
requires roughly **19,500 bets -- about 14 seasons at this volume**. The margin under test
is 0.7 percentage points; no realistic sample resolves it. This tracking exercise therefore
**cannot prove the rule works.** What it can do is detect a rule that is clearly broken, and
accumulate honest out-of-sample record at a rate no backtest slice can fake.

### Pre-committed evaluation

* **First checkpoint:** end of the 2026 regular season, or 200 graded bets, whichever comes
  later.
* **Kill condition:** below **50.0%** at the checkpoint. That is roughly 1.2 standard errors
  under the 54% baseline at n=200 and, more importantly, below breakeven -- a rule losing
  money out of sample does not get a second season on the argument that the sample was
  small. Consistent with the ledger's asymmetry: a post-hoc choice may kill a result even
  though it may not support one.
* **Continue condition:** at or above 50.0%. Continuing is NOT a claim that it works; the
  arithmetic above says one season cannot establish that. It means the rule has not yet
  disqualified itself.
* **Reported either way**, in full, whatever it says.

### How it is recorded

`track_p1.py` implements the rule and owns the log (`data/p1_log.csv`):

    python track_p1.py record      # log this week's qualifying games, at today's number
    python track_p1.py grade       # settle finished bets
    python track_p1.py report      # running out-of-sample record

The log is **append-only**: `record` refuses to rewrite a game already present, and
settlement uses the line the bet was RECORDED at, not the close. Both properties are
pinned by tests, and both exist for the same reason -- a line remembered later is the same
class of error as the opener anchoring that produced this repo's only false positive.

`model_total` is the walk-forward OLS projection with weather applied, which is the exact
quantity the P1 baseline was measured on -- deliberately not the raw simulator mean, a
different and measurably worse estimate (16.412 vs 16.396 RMSE, D17). The rule constants
live in `track_p1.py` as module-level values with a test asserting they still match this
registration, so a silent edit cannot quietly redefine what is being tracked.

As of declaration, 2026 week 1 lines are already posted and **26 games would qualify**.


### CORRECTION 2026-08-21 — the tracker was logging the opener, not the available number

`track_p1.py` set `market_total_at_bet = total_open`. That is CFBD's `overUnderOpen`, the
**historical opener**, set whenever the book first hung the game. The number actually
available when a bet is recorded is `overUnder` (`total_close` in this repo), which only
becomes "the close" once the game kicks off. On the live 2026 week 1 slate the two differ on
31 of 51 priced games, mean 0.70 points and up to 4.

**This was a bug against this registration, not a change to it.** The rule above already
says "graded at the number actually available when the bet is recorded, which is the real
money question." The code was not doing that.

**It mattered.** Selected and priced at the opener, the same rule reads **52.07% and −7.1
units** across 2021–2025, against **54.13% and +39.1** at the number actually available. The
forward record was therefore testing a materially different and historically losing strategy.

**The 26 bets logged before the fix keep their recorded numbers** — the log is append-only
and a logged bet is never repriced. They carry `line_basis="opener"`; everything after
carries `"current"`. The two may not be pooled into one win rate, and `report` keeps them
apart. Treat the first 26 as a separate, already-compromised batch.

**Found by an external audit**, and confirmed here against the live slate before acting.

### Additive 2026-08-23: the opener is now recorded on every row

`market_total_open` joins `market_total_at_bet` and `market_total_close` on each logged bet,
so the full arc sits on the row itself — where the line opened, what was actually taken, and
where it closed.

**It changes nothing about the rule.** Trigger, side, universe, cap and stake are untouched,
and the opener is NEVER the settlement price: grading stays at the number available when the
bet was recorded. The 2026-08-21 correction exists precisely because grading at the opener
made the forward record test a different and historically losing strategy, and a test now
asserts the edge is computed off the bet line rather than the opener.

**Why it is worth having.** Measured on 2021-2025, betting the opener versus the close is
52.4% against 53.2% with only 29 of 1,071 bets resolving differently (McNemar p = 0.137) —
not significant, but that is a backtest average. Recording all three numbers makes the
question answerable on THIS season's actual bets: did they land before or after the market
moved, and did it help. Without it that requires joining an external archive after the fact.

**No NFL equivalent is possible.** nflverse publishes no opening line at all — verified,
there is no such column — so W1 rows carry the bet-time line and the close only. The
6-hourly market quote archive is the substitute there.

### What this registration still cannot claim

Also recorded 2026-08-21, from the same audit and verified independently: the backtest
behind P1 is **not statistically significant**. 577–489 is 54.13%, one-sided p = **0.133**
against the 52.38% break-even — and the ±6 cap was chosen from roughly 34 min/max
combinations, which pushes the multiplicity-adjusted figure to about **0.30**. Edge size
carries no information either: 0.5–2 pts returns 55.65%, 2–4 returns 52.19%, 4–6 returns
55.20%, and the trend test reads p = 0.72.

None of this kills P1 — the ledger's asymmetry allows a post-hoc analysis to kill a result,
and this one is not decisive enough to. It does mean **54.13% may not be described as an
edge anywhere**. It is a hypothesis with a forward test attached, and the forward test is
the only thing that can settle it.

### Status of the model itself, unchanged by this

`bets_allowed()` remains **False** -- five promotion gates fail (`GATE_UNBIASED_BY_WEEK`,
`GATE_SCALE`, `GATE_RMSE_TOTAL`, `GATE_RMSE_SPREAD`, `GATE_KEY_NUMBERS`, `GATE_CALIBRATED`).
This is a measurement protocol for tracking a declared rule, not an authorisation to stake
money, and it does not alter any gate or promotion decision.
