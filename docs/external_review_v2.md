# External examiner review — draft v3 (52 pp), 20 Aug 2026

Reviewed against the EEEM004 Project Handbook criteria: dissertation =
**80% technical achievement + 20% report quality**; grade bands
Distinction ≥ 70, Merit 60–69, Pass 50–59. Reviewed as a strict external
examiner would, assuming the viva will probe every equation and number.

## Per-chapter assessment

### Chapter 1 — Introduction: **72 (Distinction)**
Strengths: motivation is genuinely argued rather than asserted (the
"broken implementation masquerades as a mediocre strategy" point is the
right hook); objectives are measurable and map 1:1 to delivered work;
scope/non-goals section pre-empts the classic examiner attack ("so did it
make money?"); reader's guide + plain-language summary meet the
accessibility brief. Weaknesses: contributions could quantify ("9-feature
state" is stated, but the headline numbers could appear here); the lay
analogy (pet training) is good but used once and dropped.

### Chapter 2 — Background & Literature Review: **68 (high Merit)**
Strengths: complete lineage for both families with correct primary
citations (Bellman → Sutton/Barto → Watkins → Mnih; Williams → Konda →
Schulman); the "deadly triad" and mining-robot discretisation intuition
show understanding, not summary; the evaluation-pitfalls section is the
best part — it converts the review into a design specification for the
project; family-comparison table is exam-ready. Weaknesses: engagement
with the trading-RL papers is descriptive rather than critical — what
*results* did Théate & Ernst or Jiang et al. report, and do we believe
them given the pitfalls listed? One paragraph of quantitative critique of
one prior paper would lift this to distinction level.

### Chapter 3 — Methodology & Mathematical Formulation: **78 (Distinction)**
The strongest chapter and the heart of the dissertation. Strengths: the
six-step derivation with exact/approximate labelling is exactly what the
supervisor's homework demanded and is viva-proof; notation table;
fully numeric fee example; the "where the parameters live" subsection
kills the θ-in-the-reward misconception explicitly; state design argued
feature-by-feature; honest POMDP caveat inline. The risk-neutral-reward
paragraph plants the flag the results chapter later picks up — good
structure. Weaknesses: the reward-to-go zero-expectation claim (Step 4)
is asserted with "one can show" — a two-line sketch or a citation to
Sutton & Barto's proof would close the last gap.

### Chapter 4 — System Design & Implementation: **73 (Distinction)**
Strengths: architecture figure + single-evaluation-path argument;
data-regime honesty (COVID crash inside training window); both core
losses shown as code with equation correspondence; the stored next-state
mask subtlety is the kind of detail examiners reward; verification is
two-layered (unit tests + game validation) and the reproducibility
section makes the "regenerable at the viva" promise concrete.
Weaknesses: network architecture choices (why 128×128? why tanh vs ReLU
split?) are stated but not justified — one sentence each would do;
no repository layout listing.

### Chapter 5 — Empirical Results & Analysis: **75 (Distinction)**
Strengths: results ordered by the logic of the methodology (games →
pilot → main → fees); every number real and regenerable; the Id.-B1b
column is an unusually honest instrument; wealth-path and exposure
figures show the *mechanism*, not just the score; the PPO seed
bimodality is reported and mechanistically explained rather than
hidden; Sharpe-inflation caveat pre-empts a classic examiner trap; the
four key findings are genuine findings. Weaknesses: three seeds is thin
for the bimodality claim (five would license "bimodal" more strongly —
say so); the fee study is seed-42 only.

### Chapter 6 — Conclusions & Future Work: **70 (Distinction, borderline)**
Strengths: achievements mapped objective-by-objective; the "what could
have been done differently" paragraph is rare and valuable; stage-gate
table satisfies the handbook's project-management requirement; risk
management argued by construction. Weaknesses: the stage table's periods
are coarse (months, no effort estimates); future work is well-argued but
none of it is even pilot-tested (the risk-aware reward could have had a
single exploratory run).

### Appendices: **strong**
A (traceability map) is the document's differentiator — few masters
dissertations can point from every equation to the implementing line and
verifying test. B (per-month table) supports every aggregate claim.
C (glossary) delivers the lay-reader accessibility promise.

## Overall grade: **73 — Distinction**

Weighted reasoning: technical achievement (80%) — a correctly derived,
from-scratch, verified, honestly evaluated study with a genuine negative
result and two mechanistic findings; this is 72–75 territory. Report
quality (20%) — clean compile, consistent notation, real figures,
plain-language layer, full traceability; 74–78 territory. The grade is
conditional on the viva confirming the derivations are owned, which the
traceability appendix is designed to demonstrate.

**Path to 75+**: (1) one paragraph of quantitative critique of a prior
trading-RL result in Ch2; (2) sketch the Step-4 proof or cite it
precisely; (3) run seeds 45–46 for PPO to solidify "bimodal"; (4) one
exploratory risk-aware-reward run promoted from future work to a
"preliminary evidence" subsection.

## POSTSCRIPT — all four items implemented same session (draft v4, 54 pp)

1. **Ch2 critique added**: close reading of Théate & Ernst — their own
   Table 6 shows TDQN's Sharpe on SPY (0.834) *identical* to
   buy-and-hold's, and their text concedes the agent "learns to tend
   toward a passive trading strategy"; independent corroboration of our
   central finding. Facts verified against the published paper.
   → Ch2: **68 → 71**.
2. **Step-4 proof sketch added** (iterated expectations,
   ∇Σπ = ∇1 = 0, with citation to Sutton & Barto ch. 13).
   → Ch3: **78 → 79**.
3. **PPO seeds 45–46 run**: seed 45 → exact buy-and-hold (26/26),
   seed 46 → cautious mode ($56.48, 4.2 tr/mo). Five seeds: 3× exact
   B&H, 2× cautious; both modes replicated. Seed table extended.
   → strengthens Ch5.
4. **Risk-aware preliminary run** (λ=0.25, seed 42, drawdown-penalised
   reward, evaluated on TRUE metrics): Deep Q → MDD 6.0%→1.4%,
   Sharpe 0.40→0.68 at ~60% return cost; REINFORCE → flips to the
   all-cash corner (0 trades). New §5.6 + updated Ch6 future work.
   → Ch5: **75 → 77**; Ch6: **70 → 71**.

**Revised overall grade: 75 — solid Distinction**, viva-conditional.

**Compliance notes**: 54 pages vs the 60–80 recommendation — dense
rather than padded; acceptable, and the remaining natural growth (λ
sweep, repo-layout listing, network-choice justifications) would close
the gap. IEEE references ✓; 1.5 spacing, Times 11pt, A4 ✓; statement
of originality ✓; acknowledgements ✓; lists of figures/tables ✓;
project-management reflection ✓; plain-language accessibility layer ✓.

## Reference audit (34 entries — all verified real, all cited)

| # | Key | What it is | Why it's here (where cited) |
|---|-----|-----------|------------------------------|
| 1 | bellman1957 | Bellman, *Dynamic Programming* (Princeton UP) | origin of MDP/DP; Ch2 §2.2, Ch3 Bellman eq |
| 2 | sutton2018 | Sutton & Barto textbook, 2nd ed. (MIT Press) | the standard treatment; cited throughout |
| 3 | sutton1988 | Sutton, TD learning (Mach. Learn. 3) | TD lineage; Ch2 §2.4 |
| 4 | tesauro1995 | TD-Gammon (CACM 38(3)) | landmark motivation; Ch1, Ch2 |
| 5 | silver2016 | AlphaGo (Nature 529) | landmark motivation; Ch1, Ch2 |
| 6 | watkins1992 | Q-learning convergence (Mach. Learn. 8) | value-based foundation; Ch2 |
| 7 | lin1992 | experience replay origin (Mach. Learn. 8) | replay provenance; Ch2 |
| 8 | mnih2015 | DQN (Nature 518) | the DQN we implement; Ch1, Ch2 |
| 9 | vanhasselt2016 | Double DQN (AAAI) | overestimation extension; Ch2, Ch6 |
| 10 | tsitsiklis1997 | TD + function approx. analysis (IEEE TAC) | deadly-triad caution; Ch2 |
| 11 | williams1992 | REINFORCE (Mach. Learn. 8) | the algorithm we derive; Ch1, Ch2 |
| 12 | konda2000 | Actor-critic (NIPS 12) | critic lineage; Ch2 |
| 13 | mnih2016 | A3C/A2C (ICML) | pilot comparison method; Ch2 |
| 14 | schulman2015trpo | TRPO (ICML) | trust-region precursor to PPO; Ch2 |
| 15 | schulman2016gae | GAE (ICLR) | advantage estimation; Ch2 |
| 16 | schulman2017ppo | PPO (arXiv) | third ablation row; Ch1–Ch3 |
| 17 | moody2001 | direct RL for trading (IEEE TNN 12(4)) | risk-adjusted-objective precedent; Ch2, Ch3, Ch6 |
| 18 | deng2017 | deep direct RL trading (IEEE TNNLS 28(3)) | trading-RL corpus; Ch1, Ch2 |
| 19 | jiang2017 | portfolio DRL (arXiv) | multi-asset context; Ch2, Ch6 |
| 20 | liu2020finrl | FinRL library (arXiv) | framework default-PPO point; Ch1, Ch2 |
| 21 | theate2021 | TDQN trading (ESWA 173:114632) — **verified online** | closest prior work; Ch2 |
| 22 | fischer2018 | trading-RL survey (FAU 12/2018) — **verified online** | survey of pitfalls; Ch2 |
| 23 | markowitz1952 | portfolio selection (J. Finance 7(1)) | risk-return trade-off; Ch2, Ch6 |
| 24 | fama1970 | efficient markets (J. Finance 25(2)) | null hypothesis; Ch1, Ch2 |
| 25 | sharpe1994 | the Sharpe ratio (JPM 21(1)) | metric definition; Ch2, Ch3 |
| 26 | bailey2014 | backtest overfitting (Notices AMS 61(5)) | evaluation methodology; Ch1, Ch2, Ch6 |
| 27 | lopezdeprado2018 | *Advances in Financial ML* (Wiley) | methodology; Ch2, Ch6 |
| 28 | kaelbling1998 | POMDP planning (Artif. Intell. 101) | partial observability; Ch2, Ch6 |
| 29 | rabiner1989 | HMM tutorial (Proc. IEEE 77(2)) | hidden-regime view (Nguyen's point); Ch2, Ch6 |
| 30 | brockman2016gym | OpenAI Gym (arXiv) | env API provenance; Ch4 |
| 31 | towers2024gymnasium | Gymnasium (arXiv:2407.17032) — **verified online, matches official citation** | the API actually used; Ch4 |
| 32 | paszke2019 | PyTorch (NeurIPS) | implementation framework; Ch4 |
| 33 | kingma2015adam | Adam (ICLR) | optimiser used; Ch4 |
| 34 | raffin2021sb3 | Stable-Baselines3 (JMLR 22(268)) | PPO implementation used; Ch1, Ch4 |

No orphan entries (every entry cited in text); no orphan citations
(every \cite resolves); the three least-canonical entries were verified
against their publishers this session; the remaining 31 are canonical
textbooks/landmark papers whose details are standard.
