# START HERE — Football Spread & Total Simulator

Two playbooks. Each one is a complete, self-contained build spec designed to be
pasted into Claude Code as a single prompt with no follow-up questions required.

- `NFL_PLAYBOOK.md` — build this first. Cleaner data, fewer teams, faster iteration.
- `NCAA_PLAYBOOK.md` — build second. Same skeleton, different data layer and priors.

Both produce the same artifact: an Excel workbook, refreshed weekly, with one row per
game showing a model spread, a model total, the market number, the edge, a cover
probability, and a recommended stake. You read the row and place the bet on your app.

---

## How to run these

Open Claude Code in an empty directory and paste the entire contents of one playbook
as your first message. Nothing else. The playbook is written to answer every question
Claude Code would otherwise ask — data sources, file layout, function signatures,
parameter defaults, failure handling, and acceptance tests are all specified.

Build NFL first, confirm the acceptance gates pass, then build NCAA in a separate
directory. Do not try to build both at once.

---

## The architecture, and why it is this architecture

Four layers. Both sports use the same skeleton.

### Layer 1 — Team strength via ridge regression (not Elo)

Every team gets an offensive rating and a defensive rating on an EPA-per-play scale,
estimated by a single weighted ridge regression over all team-games in the window.

Why ridge instead of Elo, which is what most public models use:

- Elo consumes only final margin. It throws away the offense/defense split, so it can
  produce a spread but cannot produce a total. This build needs both.
- Elo updates sequentially and never revisits. Ridge solves the whole system at once,
  so a Week 3 opponent's strength gets retroactively corrected by what that opponent
  does in Week 10. That matters enormously early in a season.
- College football has 130+ FBS teams on wildly unbalanced schedules with almost no
  inter-conference connectivity. Elo and naive SRS are close to degenerate there. Ridge
  with a tuned penalty is the standard fix — the penalty shrinks poorly-connected teams
  toward the mean instead of letting them float on three data points.

Elo is still built, but only as a **baseline to beat** in the backtest. If the ridge
model cannot outperform Elo, something is wrong and the acceptance gate fails.

### Layer 2 — Drive-based Monte Carlo (not normal, not Poisson)

The simulator plays each game possession by possession. For each drive it draws an
outcome from a fitted multinomial — touchdown, field goal, no score, turnover — with
probabilities conditioned on the offense's rating, the defense's rating, and starting
field position. Points accumulate in real football increments.

Why not just draw the margin from a normal distribution around a predicted mean:

Football margins are lumpy. Roughly 15% of NFL games are decided by exactly 3 points
and about 9% by exactly 7 — those two numbers alone account for close to a quarter of
all outcomes. A smooth normal distribution spreads probability evenly across 2, 3, and
4 and therefore misprices every line sitting on or near a key number. Since the entire
question you are asking is "is this line on the right side of 3," a model that smooths
over 3 is answering a different question than the one you asked.

A drive simulator reproduces the key-number spikes for free, because it can only ever
produce scores in football increments. It also produces the correct correlation between
margin and total, the correct fat blowout tail (which is much fatter in college), and a
full distribution you can price alternate spreads and totals off of without refitting.

Poisson is also wrong here, for the same reason — it is a counting distribution for
events of size 1, and football events come in 3s, 7s, and 8s.

### Layer 3 — Market blend

This is the layer that separates a model that makes money from one that looks good in a
backtest, and it is the one most public builds skip.

The closing line at a sharp book is the best single estimate of the true number that
exists. Research consistently finds that the margin by which you beat the closing line
predicts long-run profitability better than your win-loss record does. A model built
from public data will not beat the closing line on its own. What it can do is add
information to the *opening* line before the market has fully processed it.

So the model output is never used raw. The final number is a weighted blend of the model
and the market, where the weight is **fit on out-of-sample backtest data**, not chosen
by hand. The fitting procedure regresses actual margin on both the model number and the
market number. The resulting coefficient on the model term tells you literally how much
independent information your model carries. If it comes back near zero, the model adds
nothing, and the correct output is "no bets," not a smaller bet.

The playbooks build that honesty in as a hard gate.

### Layer 4 — Bet selection and sizing

Edges are converted to cover probabilities off the simulated distribution, filtered
against a threshold, and sized by fractional Kelly with a hard cap. Anything that does
not clear the threshold is emitted as NO BET, and the workbook shows it greyed out
rather than hiding it, so you can see how selective the model is being.

---

## What "winners" realistically looks like

Worth stating plainly once, because it determines how the thresholds are set.

Break-even at standard -110 pricing is 52.38%. A good, well-built model applied to a
*filtered subset* of games lands somewhere around 53-55% against the spread. That is a
real, meaningful edge and it compounds. It is also close enough to a coin flip that you
will have losing months — a 54% bettor loses money over a 50-bet stretch reasonably
often just from variance.

Anything claiming 60%+ over a large sample is either overfit, mismeasured against stale
lines, or lying. The playbooks include a backtest that will tell you which of those you
have, and gates that refuse to emit picks if the model has not earned it.

The practical consequence: **the number of games you bet is small.** Both playbooks
default to flagging roughly 10-20% of the slate. Resist the urge to widen the threshold
to get more action — that is exactly how a real edge gets diluted back to break-even.

The metric to watch weekly is not your record. It is closing line value: did the number
move toward your side after you bet it. Over 20 bets your record is noise. Over 20 bets
your CLV is already informative.

---

## Data sources, all free

**NFL** — `nflreadpy` (the maintained successor to `nfl_data_py`, which is deprecated).
Play-by-play back to 1999, and critically, the schedules table carries `spread_line` and
`total_line` — historical closing lines maintained by Lee Sharpe. That means the entire
NFL backtest runs with zero paid API access.

**NCAA** — the CollegeFootballData API via the `cfbd` Python package. Free tier is
capped at 1,000 calls per month, which is plenty if you cache aggressively — the
playbook mandates a disk cache and a call counter that refuses to exceed budget. CFBD
carries play-by-play, drives, betting lines, returning production, SP+, and recruiting.

**Live lines for the current week** — The Odds API free tier. Optional; the workbook
works without it if you type this week's numbers into a CSV instead. The playbook
specifies both paths.

---

## What I deliberately left out, and why

You asked for depth, so here is the negative space. These are all things I listed in our
earlier conversation that I cut after looking harder:

- **Public betting percentages / reverse line movement.** The reliable versions of this
  data are paywalled, and the free versions are marketing. Using line movement as a
  model feature also means you are just following the market with a lag, which is not an
  edge — it is a slower copy of one. Line movement is kept, but only as a CLV
  diagnostic after the fact.
- **Head-to-head history.** Near-zero predictive value in football once you control for
  team strength. Rosters and coordinators turn over too fast, and the sample is one or
  two games.
- **Referee and umpire tendencies.** Real in baseball. Marginal and noisy in football
  once you have opponent-adjusted efficiency.
- **Generic "motivation" and "letdown spot" flags.** Unquantifiable as specified, and
  they backtest as noise. Two exceptions survive and are built in: NFL Week 18 teams
  with nothing to play for resting starters, and CFB bowl games with coaching changes or
  mass opt-outs.
- **Recruiting composite as a primary NCAA input.** Corrected above — it is now worth
  1-2%, not 20%.
- **Deep feature stacks generally.** Both models run on a small number of
  well-understood inputs. The failure mode in this domain is not underfitting.

---

## Order of operations

1. Build NFL. Run the acceptance gates.
2. If the NFL gates pass, build NCAA in a separate directory.
3. Paper trade both for four weeks. Log every pick and the closing line.
4. Check CLV. If CLV is positive, start sizing. If it is negative, the model is telling
   you it is not ready — go back to the blend weight and the threshold.

Step 3 is not optional padding. It costs you four weeks and it is the only honest test
that exists.
