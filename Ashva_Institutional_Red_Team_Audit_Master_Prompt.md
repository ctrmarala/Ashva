# ASHVA — INDEPENDENT INSTITUTIONAL RED-TEAM AUDIT MASTER PROMPT

You are conducting an independent, institutional-grade red-team audit of the **Ashva quantitative trading system**.

Your job is NOT to praise the architecture and NOT to redesign the system for the sake of sophistication.

Your job is to answer one practical question:

> **Can Ashva be trusted to research, backtest, paper trade, and eventually live-trade strategies without hidden flaws that materially distort research conclusions or create unacceptable financial/operational risk?**

At the same time, there is an equally important constraint:

> **Do NOT over-engineer Ashva. Do NOT impose institutional complexity merely because it is theoretically possible. Do NOT create so many gates, statistical requirements, abstractions, services, or controls that a genuinely useful alpha can never reach paper trading or live trading.**

The desired outcome is a **robust, practical trading research system**:
- scientifically honest,
- operationally safe,
- simple enough to maintain,
- capable of finding and trading real edges,
- and appropriately cautious without becoming paralyzed.

You are simultaneously acting as:

1. Senior Quant Researcher
2. Quantitative Statistics / Validation Specialist
3. Algorithmic Trading Engineer
4. Market Microstructure / Execution Specialist
5. Chief Risk Officer
6. Data Engineering / Data Forensics Specialist
7. Senior Python Architect
8. Database Architect
9. Production SRE / Reliability Engineer
10. Cybersecurity Engineer
11. Portfolio Risk Specialist
12. Broker / Exchange Integration Specialist
13. QA / Testing / Chaos Engineering Specialist
14. Model Risk / Research Governance Specialist
15. Independent Investment Committee

---

# 1. AUDIT PHILOSOPHY — BALANCE ROBUSTNESS WITH PRACTICALITY

Use the following principle throughout the audit:

## "Minimum Necessary Complexity"

For every recommendation, classify it as one of:

### MUST FIX
A defect that can:
- invalidate research,
- create materially incorrect P&L,
- cause uncontrolled financial loss,
- duplicate live orders,
- corrupt live state,
- compromise credentials,
- or make paper/live results materially untrustworthy.

### SHOULD FIX
A meaningful robustness issue that does not necessarily prevent continued research or paper trading.

### NICE TO HAVE
An engineering improvement that is useful but should NOT block alpha discovery or normal development.

### DO NOT FIX / ACCEPTABLE TRADE-OFF
A theoretical imperfection whose risk is small relative to the complexity required to eliminate it.

This distinction is extremely important.

Do NOT recommend:
- microservices merely because they are "institutional";
- distributed systems when a reliable single process is sufficient;
- elaborate event buses when simple queues/interfaces work;
- excessive database separation when simple, enforceable environment isolation works;
- statistically impossible thresholds that require thousands of trades before any strategy can be tested;
- complicated portfolio optimization when the system is still researching individual alphas;
- excessive monitoring before basic correctness is established.

For every major recommendation answer:

1. What problem does this solve?
2. How likely is that problem?
3. What is the financial/research impact?
4. What is the simplest reliable solution?
5. Does it need to block alpha research?
6. Does it need to block paper trading?
7. Does it need to block live trading?

Prefer the **simplest architecture that is demonstrably safe enough for the stage Ashva is in**.

---

# 2. INITIAL AUDIT MUST BE READ-ONLY

During the initial audit:

DO NOT:
- modify repository source files,
- refactor code,
- delete files,
- migrate databases,
- change production configuration,
- alter historical data,
- change strategy parameters.

You MAY:
- inspect all files,
- inspect git history,
- run existing tests,
- run existing research scripts,
- run static analysis,
- run security scans,
- create temporary test harnesses outside the repository,
- create temporary datasets,
- perform controlled experiments,
- reproduce bugs.

First discover and document problems.

Only after the audit is complete should you recommend fixes.

---

# 3. COMPLETE REPOSITORY COVERAGE — MANDATORY

First map the COMPLETE repository.

Identify every:

- source file
- strategy
- test
- script
- configuration
- database component
- schema
- migration
- data ingestion component
- feature/indicator implementation
- backtest component
- execution component
- broker/API integration
- paper trading component
- live trading component
- scheduler
- background worker
- logging component
- reporting component
- research/validation component
- experiment ledger
- deployment component
- dependency
- CI/CD component

Do not review only "important" files.

Every executable source file must be accounted for.

Create a coverage table:

| File | Lines | Functions/Classes | Reviewed | Tested | Findings | Severity | Status |

If anything cannot be fully reviewed, explicitly mark it:

**UNREVIEWED**

Do not claim a complete audit if files were skipped.

---

# 4. GIT HISTORY / RESEARCH EVOLUTION

Inspect git history, not just HEAD.

Look for:

- major architecture changes
- strategy additions
- strategy modifications
- changes made after seeing results
- backtester changes
- cost-model changes
- validator changes
- data changes
- risk-control changes
- live execution changes
- database changes

Determine whether any strategy was:

SPECIFIED
→ IMPLEMENTED
→ TESTED
→ MODIFIED AFTER RESULTS
→ RE-TESTED ON THE SAME DATA

If so, flag potential research selection bias.

Also search repository and git history for secrets.

Do not reproduce secrets.

If credentials have ever been committed, recommend rotation/revocation.

---

# 5. COMPLETE SYSTEM ARCHITECTURE

Map:

RAW DATA
→ DATA LAKE
→ CLEANING
→ FEATURES
→ STRATEGY
→ SIGNAL
→ ORDER INTENT
→ EXECUTION
→ POSITION
→ P&L
→ VALIDATION
→ REPORTING

Also map:

BACKTEST
PAPER
LIVE

Determine whether the three environments are actually isolated.

Do not accept conceptual isolation without verifying the code.

---

# 6. BACKTEST FORENSICS

Treat backtest results as financial evidence.

Trace representative trades all the way from raw market data to final P&L.

Audit:

- look-ahead bias
- future leakage
- rolling-window leakage
- shifted feature correctness
- daily aggregation leakage
- intrabar information leakage
- signal timing
- next-bar execution
- same-bar execution
- stop/target ordering
- impossible fills
- gap-through-stop
- gap-through-target
- slippage
- spread
- liquidity assumptions
- opening auction assumptions
- EOD execution
- partial fills
- position state
- quantity rounding
- zero-quantity handling
- capital limits
- transaction costs
- taxes
- corporate actions

Create adversarial examples where useful.

The most important question:

> Could the backtester know something at time T that a live trader would only know after time T?

If yes, classify severity according to impact.

---

# 7. BACKTEST VS PAPER VS LIVE CONSISTENCY

Determine whether the same conceptual trading lifecycle is used across:

BACKTEST
PAPER
LIVE

Ideally:

Strategy
→ Signal
→ Order Intent
→ Risk Check
→ Execution
→ Fill
→ Position
→ P&L

should have clear boundaries.

But do NOT require a giant abstraction hierarchy.

If the existing design is simple and reliable, preserve it.

Identify only the abstractions that are genuinely needed to prevent divergent behavior.

---

# 8. DATA FORENSICS

Audit:

- missing candles
- duplicate candles
- out-of-order candles
- timestamps
- timezone
- NSE trading calendar
- holidays
- special sessions
- partial sessions
- invalid OHLC
- abnormal volume
- stale data
- corporate actions
- splits
- bonuses
- symbol changes
- adjusted/unadjusted prices
- vendor changes

Also determine whether current historical data is sufficient for the strategies being tested.

Do NOT reject useful research simply because data is imperfect in a minor way.

Classify imperfections according to actual impact.

---

# 9. UNIVERSE / SURVIVORSHIP BIAS

Audit the stock universe.

Determine:

- why stocks were selected,
- whether selection uses today's information,
- whether survivorship bias exists,
- whether historical eligibility differs from current eligibility,
- whether the 14-stock universe is sufficient for discovery.

Recommend a better universe only if the current universe materially biases conclusions.

Do NOT insist on a massive universe before useful alpha discovery is possible.

Clearly distinguish:

**research universe**

from

**production trading universe**.

---

# 10. ALPHA-BY-ALPHA AUDIT

Review EVERY alpha currently present.

For each alpha determine:

- economic hypothesis
- mathematical specification
- implementation equivalence
- parameters
- parameter origin
- signal timing
- execution timing
- stop/target
- sizing
- costs
- sample size
- trade frequency
- PF
- Sharpe
- drawdown
- regime dependence
- symbol dependence
- time dependence
- cost sensitivity
- slippage sensitivity
- parameter sensitivity

Then independently determine:

- implementation correct?
- backtest trustworthy?
- evidence sufficient?
- likely overfit?
- current-regime relevance?
- suitable for paper?
- suitable for live?

Do NOT automatically reject a strategy because it has a small sample.

Instead state:

> "Evidence is insufficient to establish an edge."

That is different from:

> "The strategy is false."

Similarly, do NOT call a strategy "validated" merely because PF > 1.

---

# 11. POSITIVE ALPHAS GET THE HARDEST AUDIT

Pay special attention to strategies currently considered successful.

Try to DISPROVE them.

For each positive candidate ask:

- Is the edge driven by one stock?
- One month?
- One regime?
- A few trades?
- One unusually profitable trade?
- A favorable cost assumption?
- A favorable fill assumption?
- A parameter chosen after observing results?
- Current-universe selection?
- Recent luck?

Perform perturbation where practical.

Examples:

- slightly higher slippage
- slightly wider spread
- slightly different stop
- slightly different target
- delayed entry
- small parameter changes

Do not endlessly optimize.

The purpose is to determine whether the edge is fragile.

---

# 12. MULTIPLE TESTING / DATA SNOOPING

Reconstruct the research process.

Count or estimate:

- hypotheses
- strategies
- parameter variants
- symbols
- windows
- repeated tests
- discarded experiments
- post-hoc changes

Determine whether discovery and validation are actually separated.

Audit:

- experiment ledger
- strategy-family accounting
- DSR
- multiple testing
- false discovery risk

Independently verify the mathematics.

Do not accept "DSR exists in the code" as proof.

---

# 13. CPCV / PURGING / EMBARGO

Audit:

- CPCV
- train/test separation
- purging
- embargo
- rolling features
- point-in-time feature generation
- overlapping observations
- test-slice evaluation

Pay special attention to:

ATR
ADX
VWAP
20-day range
20-session volume baselines
relative strength
daily regimes

Determine whether features are truly point-in-time.

---

# 14. 540-DAY / MULTI-WINDOW FRAMEWORK

Audit:

60d
180d
365d
540d

and any recency weights such as:

50%
25%
15%
10%

Do NOT assume those weights are statistically optimal.

Check:

- small sample PF
- zero-trade windows
- PF sentinel values
- overlapping windows
- dependency
- effective sample size
- recent outliers
- regime stability
- current momentum

Particularly investigate cases such as:

PF_60d = extremely high
with only 2–3 trades.

A tiny recent sample must NOT be allowed to masquerade as strong evidence.

However:

Do NOT impose a universal minimum trade count so high that useful
early-stage alpha research becomes impossible.

Recommend a staged framework such as:

DISCOVERY
→ PROMISING
→ PAPER CANDIDATE
→ LIVE CANDIDATE

with progressively stronger evidence requirements.

---

# 15. STATISTICAL BALANCE

This is a critical requirement.

Do NOT over-correct for statistical purity.

Ashva is an alpha discovery system, not a peer-reviewed academic journal.

The audit must distinguish:

### Discovery evidence
"What looks interesting?"

### Validation evidence
"Does the effect survive reasonable testing?"

### Trading evidence
"Is there enough evidence to risk small capital?"

Do NOT require live-trading-level evidence before an alpha can be explored.

But do require meaningful evidence before capital is exposed.

Recommend practical thresholds based on:

- sample size
- trade frequency
- robustness
- regime stability
- costs
- drawdown
- implementation confidence

rather than arbitrary universal thresholds.

---

# 16. COST MODEL

Verify:

INTRADAY:
- STT
- brokerage
- GST
- exchange charges
- SEBI charges
- stamp duty
- slippage

DELIVERY/SWING:
- buy/sell STT
- brokerage
- GST
- stamp duty
- DP charges
- slippage

Separate:

FACTUAL CURRENT COST
ASSUMED COST
CONSERVATIVE BUFFER

Test reasonable cost sensitivity.

Do not reject an alpha solely because an unrealistically pessimistic
cost model was chosen.

---

# 17. POSITION SIZING / RISK

Audit:

- fixed risk
- stop distance
- quantity
- rounding
- capital cap
- zero quantity
- leverage
- margin
- gross exposure

Confirm the system cannot silently exceed its intended risk.

Then audit portfolio-level risk:

- simultaneous positions
- sector concentration
- correlated stocks
- strategy correlation
- market beta
- daily loss
- max drawdown

Do not require a sophisticated portfolio optimizer at this stage.

A simple, enforceable exposure/risk framework is preferable to a complex
model that nobody can trust.

---

# 18. BACKTEST / PAPER / LIVE ISOLATION

This is a MUST-FIX area if unsafe.

Determine whether:

BACKTEST
PAPER
LIVE

can safely operate simultaneously.

They should have separate:

- state
- positions
- orders
- credentials
- execution permissions

A practical solution may be:

ashva_research
ashva_paper
ashva_live

using separate databases OR separate schemas with strong permission
boundaries.

Do not insist on separate infrastructure if strong isolation can be
achieved simply and safely.

Critical checks:

- Can paper send a real broker order?
- Can research modify live state?
- Can cleanup scripts delete live trades?
- Can test fixtures touch live DB?
- Can order IDs collide?
- Can migrations damage live state?
- Can shared configuration accidentally enable live execution?

---

# 19. LIVE EXECUTION / IDEMPOTENCY

Audit:

- order submission
- acknowledgements
- retries
- timeouts
- duplicate prevention
- partial fills
- rejection
- cancellation
- modification
- broker reconciliation
- rate limits

Critical scenario:

Ashva submits an order.

Broker accepts it.

Network response is lost.

Ashva retries.

Can Ashva accidentally create two positions?

If yes, classify appropriately.

---

# 20. POSITION RECONCILIATION

A live system must trust the broker as the ultimate source of actual
position state.

Audit whether Ashva reconciles:

BROKER
vs
DATABASE
vs
OPEN ORDERS
vs
FILLS

Test mismatches.

Determine safe recovery behavior.

---

# 21. CRASH / FAILURE / CHAOS AUDIT

Consider at least:

1. Crash before order submission.
2. Crash after order submission.
3. Broker accepts but response is lost.
4. Network timeout.
5. Partial fill.
6. Gap through stop.
7. Duplicate candle.
8. Missing candle.
9. Out-of-order candle.
10. Database unavailable.
11. Process restart.
12. Two instances accidentally running.
13. Broker rejection.
14. Rate limit.
15. Incorrect system clock.
16. Feed freeze.
17. Open position during restart.

For each important scenario:

EXPECTED
CURRENT
RISK
FIX
REGRESSION TEST

Do not demand a complete distributed chaos platform.

Simple deterministic failure handling is enough if it is reliable.

---

# 22. SOFTWARE QUALITY

Audit:

- architecture
- coupling
- state management
- exception handling
- retries
- idempotency
- type safety
- validation
- configuration
- dead code
- duplication
- magic numbers
- global state
- concurrency
- database transactions
- performance
- maintainability

Do not recommend refactoring merely for style.

Only recommend architectural changes when they materially improve:

CORRECTNESS
SAFETY
RESEARCH VALIDITY
RELIABILITY
MAINTAINABILITY

---

# 23. DATABASE AUDIT

Audit:

- schema
- indexes
- constraints
- transactions
- concurrency
- migrations
- locking
- recovery
- cleanup
- environment separation
- audit history

Determine whether financial records can be accidentally overwritten or
deleted.

---

# 24. TESTING AUDIT

Evaluate whether tests prove:

- strategy correctness
- timing correctness
- cost correctness
- position correctness
- sizing correctness
- database correctness
- execution correctness
- failure behavior

Do not focus only on test count.

Find the most dangerous behaviors that are currently untested.

Where possible create temporary adversarial tests WITHOUT changing the
repository.

---

# 25. SECURITY AUDIT

Search repository and git history for:

- API keys
- broker credentials
- passwords
- tokens
- GitHub PATs
- private keys
- secrets
- .env values

Do not print secret values.

Recommend rotation if necessary.

Audit least privilege and environment isolation.

---

# 26. OBSERVABILITY

Determine whether the operator can answer:

- What positions exist?
- Which strategy created them?
- Why is the position open?
- What is the intended risk?
- What does the broker say?
- Which orders are pending?
- Is the market feed alive?
- Is the broker connection alive?
- Has a risk limit been hit?
- Is the system healthy?

Do not demand an elaborate enterprise observability stack.

Identify the MINIMUM monitoring needed for safe paper/live operation.

---

# 27. RESEARCH LIFECYCLE

Assess whether Ashva has a practical lifecycle:

IDEA
→ SPECIFICATION
→ IMPLEMENTATION
→ TEST
→ DISCOVERY
→ ROBUSTNESS CHECK
→ PAPER
→ LIVE
→ MONITOR
→ RETIRE

Recommend promotion gates.

But do NOT create gates that make alpha discovery impossible.

The system should be allowed to say:

"Interesting but insufficient evidence"

rather than only:

"Trade"
or
"Reject forever."

---

# 28. ALPHA EVIDENCE TIERS

Recommend a practical evidence model.

For example:

### TIER 0 — IDEA
Economic rationale only.

### TIER 1 — DISCOVERY
Initial backtest shows interesting behavior.

No capital.

### TIER 2 — RESEARCH CANDIDATE
Basic costs, robustness and leakage checks pass.

### TIER 3 — PAPER CANDIDATE
Enough evidence to test forward under realistic execution.

### TIER 4 — LIVE CANDIDATE
Current regime + historical robustness + operational safety support
small controlled capital.

### TIER 5 — PRODUCTION
Demonstrated live behavior and operational stability.

Adjust this model if your audit finds a better practical approach.

The key requirement:

**Do not make every alpha prove Tier 5 before it can reach Tier 3.**

---

# 29. "₹5 LAKH TOMORROW" TEST

Assume Ashva receives ₹5 lakh tomorrow.

Identify the 20 most plausible ways it could lose materially more money
than the backtest suggests.

For each:

PROBABILITY
SEVERITY
DETECTABILITY
RECOVERY DIFFICULTY
MINIMUM MITIGATION

Be practical.

---

# 30. FINAL READINESS MATRIX

Rate:

DATA
BACKTEST
QUANT RESEARCH
STATISTICS
COST MODEL
RISK
EXECUTION
DATABASE
SECURITY
PAPER TRADING
LIVE TRADING
OBSERVABILITY
RECOVERY

as:

GREEN
YELLOW
RED

Then provide:

Historical Research: GO / CONDITIONAL GO / NO-GO
Backtesting: GO / CONDITIONAL GO / NO-GO
Paper Trading: GO / CONDITIONAL GO / NO-GO
Small Capital Live: GO / CONDITIONAL GO / NO-GO
Larger Capital Live: GO / CONDITIONAL GO / NO-GO

---

# 31. FINAL REPORT STRUCTURE

Produce:

1. Executive Summary
2. Overall Audit Verdict
3. Architecture Diagram
4. Complete Repository Coverage
5. Critical Findings
6. High Findings
7. Medium Findings
8. Low Findings
9. Confirmed Bugs
10. Suspected Bugs
11. Quant Research Validity
12. Backtest Validity
13. Data Quality
14. Universe / Survivorship Bias
15. Statistical Validation
16. Multiple Testing / Data Snooping
17. CPCV / DSR / Monte Carlo
18. 540-Day / Recency Framework
19. Cost Model
20. Risk Management
21. Portfolio Risk
22. Execution
23. Backtest/Paper/Live Isolation
24. Database
25. Broker / Live Readiness
26. Crash / Recovery
27. Security
28. Software Quality
29. Testing
30. Observability
31. Alpha-by-Alpha Assessment
32. Positive Alpha Red-Team
33. Top 20 Financial Failure Modes
34. Required Fixes
35. Recommended Architecture
36. Recommended Research Lifecycle
37. Required Tests
38. Chaos Test Plan
39. Prioritized Roadmap
40. Final Go/No-Go

---

# 32. FINDING FORMAT

For every material finding:

ID:
SEVERITY:
PRIORITY: P0/P1/P2/P3
CATEGORY:
FILE:
LINE(S):
PROBLEM:
FACTUAL EVIDENCE:
WHY IT MATTERS:
FINANCIAL/RESEARCH IMPACT:
REPRODUCTION:
CURRENT BEHAVIOR:
EXPECTED BEHAVIOR:
SIMPLEST RELIABLE FIX:
REGRESSION TEST:
CONFIDENCE:

Clearly distinguish:

FACT
INFERENCE
SUSPECTED ISSUE
RECOMMENDATION

Do not present speculation as fact.

---

# 33. CRITICAL BALANCE RULE

Before recommending any change, ask:

> "Does this materially improve correctness, research validity, risk,
> or production safety?"

If NO:

Do not recommend it.

If YES:

Ask:

> "What is the simplest implementation that provides sufficient protection?"

Prefer that solution.

Do NOT turn Ashva into an unnecessarily complicated hedge-fund platform.

The goal is:

                    SIMPLE
                      +
                 CORRECT
                      +
                  ROBUST
                      +
              RESEARCH-CAPABLE
                      +
                 TRADEABLE

not:

                    COMPLEX
                      +
                  THEORETICALLY
                    PERFECT
                      +
                 UNUSABLE

---

# 34. FINAL INVESTMENT-COMMITTEE QUESTION

At the end, answer bluntly:

> "If this were your own ₹5 lakh, would you allow Ashva to trade it
> tomorrow?"

If NO:

List the smallest set of P0/P1 changes required to make the answer YES.

If YES:

Provide the evidence that justifies the decision.

Then answer one additional question:

> "Are any of your recommendations unnecessarily conservative or
> over-engineered for Ashva's current stage?"

If yes, explicitly remove or downgrade them.

The audit is successful only if it finds real weaknesses WITHOUT
destroying Ashva's ability to discover and trade genuine alpha.

TRY TO BREAK ASHVA.

BUT DO NOT BREAK THE RESEARCH PROCESS.
