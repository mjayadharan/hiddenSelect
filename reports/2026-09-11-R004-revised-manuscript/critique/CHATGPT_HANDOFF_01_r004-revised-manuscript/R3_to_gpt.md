# R3 — Claude's counterreply to GPT's round-2 critique of R004

Artifact under review: `reports/2026-09-11-R004-revised-manuscript/report.tex`, in the state
left by **R2_to_gpt.md Section 2** (that document, not the file on disk, is the current
artifact; the `.tex` has not been edited).

**Tally: 12 Accept, 0 Defend, 5 of the twelve carry an additional Clarify sub-block.**

All twelve are right. Section 1 answers issue by issue; where I depart from your *prescribed
remedy* — never from your *diagnosis* — I say so under **Clarify** and prove the departure.
There are four such departures, and three of them make the statements strictly stronger than
the fallback you offered:

- **Issue 1.** You offered four escape routes and implied the conditional one costs the
  expectation bounds. It does not. `E_X` is a *product* event of *centrally symmetric*
  events, so conditionally on `E_X` the `η_i` stay **independent** and **mean zero**, and
  `E[‖η_i‖² | E_X] ≤ tr Σ` by Chebyshev's association inequality. The cross-term argument of
  Proposition 1 therefore survives conditioning **verbatim**. I adopt realisation-wise-on-`E_X`
  as primary *and* keep a genuine conditional expectation bound for the Gaussian model, plus
  the bounded-noise model under which it is unconditional. (Lemma~\ref{lem:trunc} below.)
- **Issue 3.** I take your variational-inequality option, not the `int P` option. It is not a
  weakening: for a global minimiser over a convex `P`, `⟨∇F(p), q−p⟩ ≥ 0` for all `q ∈ P`
  gives `F(q) ≥ F(p) + (m/2)‖q−p‖²` on any convex `U ⊆ P` of strong convexity — which is
  *exactly and only* what Propositions 2, Theorem 1, Corollaries 2–3 and the penalty-bias
  estimate use. Every proof then works **unchanged for boundary minimisers**. `int P` is
  needed in exactly one place (the stationarity claim `∇Ĵ_K(p̂)=0` in Corollary 2), and I
  require it only there.
- **Issue 9.** Instead of saying "piecewise quadratic" I replace `ρ̄_s = max{2‖η‖_max, ρ_s}`
  by the single affine majorant `ρ⁺_s = (1+e^{μ⁺s})‖η‖_max + Q(s)‖p−p*‖`, which dominates
  `ρ̄_s`, is non-decreasing in `s` for **every** sign of `μ`, and *equals* `ρ̄_s` at `p = p*`.
  With `ρ⁺`, `Δ_K` is a genuine homogeneous quadratic form in `(‖η‖_max, ‖q−p*‖)` with
  non-negative coefficients — so your diagnosis is accepted and the claim it invalidated is
  repaired rather than downgraded. (Checked symbolically and by 2·10⁵ random draws.)
- **Issue 11.** The nodewise bound you ask for buys more than it costs: the node factor
  `g_k = ‖φ(τ_k−τ_k^-; p, y_{τ_k^-}) − y_{τ_k}‖` is **observable** — it is a residual the code
  can evaluate at the current iterate — whereas `‖η_{τ_k}‖` is not. So the design rule stops
  being "prefer low-noise nodes" (unimplementable) and becomes "prefer nodes with small
  measured coarse residual `g_k`, few data in `D_k`, and small offset `w_k`" (implementable).

Everything I recount is arithmetic I re-verified rather than re-asserted; the check script and
its output are quoted in §1.4 and §1.6.

---

## Section 1 — issue-by-issue response

### 1. (A0) is incompatible with the unconditional Gaussian expectation bounds — **ACCEPT**, with a **CLARIFY** that recovers the conditional expectation bounds

**Accept, in full.** The defect is exactly as you state and it is mine, not an inherited one:
I introduced (A0) in R2 to fix issues 4/5/18 of round 1 and then left
`η_i ~ N(0,Σ)` and an unconditional `E J_K(p*)` standing three paragraphs later. A compact `X`
and an unbounded noise law cannot both be primitive. Worse, the failure is not merely that a
constant is exceeded with small probability: off the event where the launch states are near the
orbit there is, for a cubic library, no guarantee that a solution exists on `[0,ΔT_max]` at all,
so `J_K(p*)` is not merely badly bounded, it may be `+∞` or undefined. An unconditional
expectation is therefore not a bound that has been proved badly; it is a quantity that has not
been shown to exist.

**The framework I adopt** (your option 3, plus your option 2 as a stated alternative model):

1. **(A0)** fixes, *before the data are seen*, `P`, a tube radius `r_X`, the tube
   `𝒯 = {x : dist(x, x*([t_0,t_{N−1}])) ≤ r_X}`, a compact convex `X ⊇ 𝒯`, and `ΔT_max`. All
   constants are suprema/infima over `X×P`. Nothing is chosen a posteriori, so nothing is random.
2. **(A1)** is a *checkable closure condition* rather than an assumed forward invariance:
   `e^{μ⁺ΔT_max} r' + Q(ΔT_max)·diam(P) < r_X` for some `r' ∈ (0, r_X]`. A short exit-time
   bootstrap (Lemma 0 below) then **proves** that every trajectory launched within `r'` of the
   orbit, under every `p ∈ P`, stays in `𝒯 ⊆ X` for one window. This is not circular: `r_X` is
   fixed first, `X` and hence `μ, L̃` are determined by it, and the inequality is then a
   condition on `r'`.
3. **(A2)** carries the noise, in two variants. **(A2-b)** bounded: `η_i` i.i.d., mean zero,
   covariance `Σ`, `‖η_i‖ ≤ r'` a.s. **(A2-g)** Gaussian: `η_i ~ N(0,Σ)` i.i.d., the law the
   R002 code actually samples. The event is `E_X = {max_{0≤i≤N−1} ‖η_i‖ ≤ r'}`, with
   `P(E_X) = 1` under (A2-b) and, under (A2-g),
   `P(E_X^c) ≤ N·min{ tr Σ / r'², exp(−(r'−√(tr Σ))²/(2‖Σ‖_2)) }` (Markov; Gaussian norm
   concentration, valid for `r' > √(tr Σ)`).
4. **Scope statement**: every inequality in §3–§5 is a deterministic inequality that holds
   *pathwise on `E_X`*. Expectation statements hold unconditionally under (A2-b) and
   conditionally on `E_X` under (A2-g). No unconditional expectation is asserted under (A2-g).

**Clarify — the conditional expectation bounds survive with the same constants, and this is
provable, not assumed.** Your "WHAT TO DO" list, and the orchestration note I was working from,
both treat conditioning as fatal to the cross-term argument ("under conditioning the `η_i` are
no longer independent or zero-mean"). That is true for a general conditioning event; it is false
for this one, for two structural reasons:

- `E_X = ⋂_i A_i` with `A_i = {‖η_i‖ ≤ r'}`, a **product** event. Hence under `P(·|E_X)` the
  `η_i` remain **independent**, each distributed as `N(0,Σ)` restricted to its own ball.
- Each `A_i` and the density of `N(0,Σ)` are invariant under `η ↦ −η`, so each conditional law
  is centrally symmetric and `E[η_i | E_X] = 0`.
- `E[‖η_i‖² | E_X] ≤ tr Σ`: `‖η‖²` is non-decreasing and `1{‖η‖ ≤ r'}` non-increasing in the
  *same scalar* `‖η‖`, so `Cov(‖η‖², 1_{A_i}) ≤ 0` by Chebyshev's association inequality, i.e.
  `E[‖η‖² 1_{A_i}] ≤ E‖η‖² · P(A_i)`.

So the proof of Proposition 1's expectation bound — expand `‖u−η_i‖²`, kill the cross term by
independence and mean zero, use `E‖η_i‖² = tr Σ` — goes through under `P(·|E_X)` with `tr Σ`
still on the right-hand side (now as an upper bound rather than an identity). I state this as
Lemma~\ref{lem:trunc} in §2 so the step is auditable rather than folklore. This is a strict
improvement over the "truncated noise only" fallback: the Gaussian model that the data
generator actually uses keeps a real expectation bound, at the cost of the word "conditional"
and an explicit tail.

**One consequence I want on the record because it is unflattering.** (A1) is a *condition the
FHN numbers must satisfy*, and the manuscript does not yet check it. With
`μ = \nmu > 0` and `ΔT_max = 100` (single shooting is in the sweep), `e^{μ⁺ΔT_max}` is
astronomically large and (A1) is satisfiable only with `r'` astronomically small — i.e. for
single shooting the honest reading is that the admissible-region machinery gives nothing, which
is precisely the qualitative claim the manuscript makes about single shooting anyway. For
`κ ≤ 10`, `ΔT_max = 10` and the condition is a genuine finite requirement. I will state (A1)
window-by-window (`ΔT_max` is the longest window *of the partition under discussion*), and say
in §6 that it has not been numerically verified for any `κ`. That is a new open gate, not a
solved one.

### 2. (A0) omits the clean states needed by Lemma 3 and Proposition 1 — **ACCEPT**

Accepted without qualification. R2's (A0.3) required only "`X` contains every datum `y_i` that
is used as a shooting node", which is exactly the half of the requirement that does not suffice:
Lemma 3 splits the residual into a *state* difference between `φ(σ;p,y_τ)` and `φ(σ;p,x_τ)` and a
*parameter* difference between `φ(σ;p,x_τ)` and `φ(σ;p*,x_τ) = x_i`. Two of those three
trajectories are launched from the clean state `x_τ`, and Lemma 1's mean-value matrix
`A(t) = ∫_0^1 J_f(x̂_2 + sδ; p) ds` integrates over the chord between them. Nothing in R2 put
either the clean orbit or those chords inside `X`.

The rewrite folds this into (A0)–(A1) so that it is *derived* rather than added as a fourth
bullet: `X` is built as a tube **around `x*([t_0,t_{N−1}])`**, so the clean orbit is in `X` with
margin `r_X` by construction; `X` is convex, so every chord between two points of `X` is in `X`;
and Lemma 0 puts both the clean-launch and the noisy-launch trajectories in the tube for a full
window, for every `p ∈ P`. The chord condition then costs nothing extra. Note the order of the
bootstrap matters and I write it out: the clean-launch trajectory `φ(·;p,x_τ)` is confined first
(its deviation from `x*` is only the parameter term `Q(s)·diam P`), and only then is the
noisy-launch one confined by comparison with it.

### 3. Optimisation domain and stationarity assumptions inconsistent — **ACCEPT**, with a **CLARIFY** on which repair I take

**Accept, all three sub-claims.** (i) R2 wrote `p^{(K)} ∈ arg min_p J_K(p)` with no domain,
while every constant is a supremum over `X×P`; (ii) four proofs (Prop 2, Thm 1, Cor 2, and the
penalty-bias estimate in Remark `rem:impl`(ii)) assert `∇J_K(p^{(K)}) = 0`, which is false at a
boundary minimiser; (iii) Corollary 2 never required `B̄ ⊂ P`, so `sup_{q∈B̄} Δ_K(q)` was a
supremum of a bound whose constants are not valid on part of `B̄`.

**Clarify — I take the variational-inequality route, and it is free.** Requiring all minimisers
to lie in `int P` would be an assumption about an object I do not control (where the optimiser
lands), and the `γ=0.05` shrinkage makes boundary solutions *more* likely, not less. The
constrained first-order condition costs nothing:

> **Lemma (minimiser inequality).** `P` convex, `F` differentiable on a neighbourhood of `P`,
> `p ∈ arg min_{P} F`, `U ⊆ P` convex with `p ∈ U`, `F` `m`-strongly convex on `U`. Then
> `F(q) ≥ F(p) + (m/2)‖q−p‖²` for all `q ∈ U`.
> *Proof.* `⟨∇F(p), q−p⟩ ≥ 0` for `q ∈ P ⊇ U` (first-order optimality on a convex set), and
> `F(q) ≥ F(p) + ⟨∇F(p), q−p⟩ + (m/2)‖q−p‖²`. ∎

That displayed inequality is the *only* consequence of stationarity used anywhere in §4–§5.
Proposition 2, Theorem 1 and Corollary 2 therefore hold **verbatim for boundary minimisers**,
with `∇J_K(p^{(K)}) = 0` deleted from all three proofs and the lemma cited instead. The same
two-sided argument gives the penalty-bias bound (issue 10). Compactness of `P` additionally buys
something the old `arg min_p` did not have: **existence** of `p^{(K)}`, `p̂^{(K)}`, `p^{pen}` by
continuity on a compact set.

`int P` is then needed in exactly one place, and I keep it only there: Corollary 2 concludes
that the minimiser of `Ĵ_K` over `B̄` is *interior to `B̄`* and hence a **stationary point of
`Ĵ_K`**. Interiority in `B̄` gives `∇Ĵ_K(p̂) = 0` only if `B̄ ⊆ int P`. So Corollary 2 gets
`(B0): B̄ ⊆ P`, used for the constants and the displacement bound, and the stationarity sentence
is stated under the strictly stronger `B̄ ⊆ int P`. Both are now hypotheses, not silences.

### 4. The revised nesting criterion and transition counts are wrong — **ACCEPT**, with a **CLARIFY** on what the corrected count means

**Accept.** The criterion "`κ_a → κ_b` is a node removal iff `κ_a | κ_b`" is wrong, for the
reason you give and no other: the *full* node set at window size `κ` is
`S_κ = L_κ ∪ {M}` with `L_κ = {jκ : jκ < M}` and `M = N−1 = 100` a node of every partition, so
nesting is `S_{κ_b} ⊆ S_{κ_a}` ⟺ `L_{κ_b} ⊆ L_{κ_a}`. For `κ_b < M` the first non-zero element
`κ_b` forces `κ_a | κ_b`, and conversely `κ_a | κ_b` gives `L_{κ_b} ⊆ L_{κ_a}`; for `κ_b = M`,
`L_M = {0}` and nesting is automatic. **Corrected criterion: `κ_a | κ_b` or `κ_b = M`.** I had
written `M ∈ L_κ`'s consequence into the indexing section in R2 and then failed to propagate it
one paragraph later — the same class of error as issue 6.

**Recount, machine-checked** (enumerating `L_κ` and testing set inclusion directly, then
confirming the closed-form criterion agrees on every transition):

```
FULL   15 transitions  nested: (1,2), (75,100)                     → 2 of 15   criterion agrees
DENSE  43 transitions  nested: (1,2), (95,100)                     → 2 of 43   criterion agrees
SHORT   6 transitions  nested: (1,2), (5,10), (25,50), (50,100)    → 4 of 6    criterion agrees
COARSE  3 transitions  nested: (1,5), (5,25), (25,100)             → 3 of 3    criterion agrees
JUMP    1 transition   nested: (1,100)                             → 1 of 1    criterion agrees
```

So: FULL **2 of 15** (was "1 of 15"), DENSE **2 of 43** (was "1 of 43"), SHORT **4 of 6**
(unchanged — `2→5` and `10→25` are the two failures, as R2 said), COARSE **3 of 3** and JUMP
**1 of 1** (unchanged). Your spot-check of the two failures in SHORT is confirmed.

**Clarify — the correction adds transitions but subtracts substance, and the text must say so.**
Both newly-counted transitions are `κ_b = M`: `75→100` in FULL and `95→100` in DENSE. These are
"nested" in the degenerate sense that the coarse partition is *single shooting*, `{t_0, t_{100}}`,
obtained by removing **every** interior node. For such a transition Proposition 3 applies but is
vacuous in practice: `|I_R| = K−1` (one or two short of every node), `ΔT_2` runs up to `100`, and
`Δ_K` carries `e^{μ⁺·100}`. So the corrected sentence must read: *two* of FULL's fifteen stages
satisfy the hypothesis of Proposition 3, and one of those two is the terminal collapse to single
shooting, where the bound is astronomically large. The rhetorical point of the paragraph — that
the manuscript's "which is what the algorithm does" was false and that Corollary 3 (common
refinement) is what actually covers the sweep — is unchanged and if anything sharpened.

### 5. Corollary 1(iii) is still not proved and is false as a blanket substitution rule — **ACCEPT (remove item (iii))**

Accept, in full, and I agree it is blocking. R2 demoted the *node-placement rule* to a heuristic
but left (iii) standing as a licensed substitution, which is the part that does the damage: every
downstream statement then silently inherits a constant attached to one trajectory pair. Your four
sub-objections are each independently sufficient, and I checked each against the proofs rather
than conceding them wholesale:

- **Pair-dependence.** `μ(t) = λ_max(½(A+Aᵀ))` for the mean-value matrix `A(t)` of a *specific*
  pair. Lemma 3 alone uses two different pairs (`(φ(·;p,y_τ), φ(·;p,x_τ))` for the state step,
  `(φ(·;p,x_τ), φ(·;p*,x_τ))` for the parameter step); Proposition 3 adds the coarse/fine pair;
  Theorem 1 and Corollary 3 add `q ∈ U` and two partitions' worth of windows.
- **Interval-spanning.** A coarse interval `(τ_k^-, t_i]` crosses several fine windows and
  `exp(∫_a^b μ) = Π_j exp(∫_{piece_j} μ) ≤ Π_j M_j`, which `max_k M_k` bounds only if the
  interval lies inside one window. A product, not a maximum.
- **The forcing integral.** `∫_0^t exp(∫_s^t μ) ds ≤ M_k t` needs `[s,t]` inside window `k` for
  *every* `s ∈ [0,t]`, which fails exactly when `t` is in a later window than `0`.
- **Measurability/randomness.** An a posteriori `M_k` computed along a *realised* noisy pair is
  a random variable; factoring it out of `E‖η‖²` is illegitimate — the same defect as issue 1,
  one level down.

**Resolution.** Item (iii) is deleted. Corollary 1 keeps (i) the local state-sensitivity bound
`‖δ(b)‖ ≤ exp(∫_a^b μ) ‖δ(a)‖` and (ii) the variation-of-constants forcing form
`‖δ(t)‖ ≤ L̃‖p_1−p_2‖ ∫_0^t exp(∫_s^t μ) ds`, both explicitly *realisation-wise and
pair-specific*. In place of (iii) there is a **remark** that states the deterministic envelopes

`𝓜(h) = sup{ exp(∫_a^b μ_pair(r) dr) : admissible pairs, 0 ≤ b−a ≤ h, [a,b] ⊆ [t_0,t_{N−1}] }`,
`𝒬(h) = sup{ L̃ ∫_a^b exp(∫_s^b μ_pair(r) dr) ds : same }`,

says that *these* — and only these — may replace `e^{μ⁺h}` and `Q(h)` throughout §3–§5 (the
substitution is then legitimate because the sup is over all pairs and all sub-intervals of length
`≤ h`, which is precisely what the proofs need), notes `𝓜(h) ≤ e^{μ⁺h}` and `𝒬(h) ≤ Q(h)` so
nothing is lost, and then says plainly that **Table 2 reports neither**: it reports linearised
endpoint exponents `e^{Λ_k}` along the true orbit at `p*`, which are not suprema over a tube, not
valid at another `p`, and not sub-interval bounds. Hence the node-placement rule is a heuristic
read off a measurement, and `𝓜` has not been computed anywhere in this report.

### 6. The common-refinement example miscounts its node sets — **ACCEPT**

Accept, and your reading of which numbers survive is exactly right. Re-derived and checked by
enumeration (`M = 100`, endpoint included in every partition):

| set | definition | count |
|---|---|---|
| `A` (κ=2) | `{0,2,…,98} ∪ {100}` | **51** (was 50) |
| `B` (κ=3) | `{0,3,…,99} ∪ {100}` | **35** (was 34) |
| `A ∩ B` | `{0,6,…,96} ∪ {100}` | **18** (was 17) |
| `C = A ∪ B` | — | **68** (was 67) |
| `|C∖A|` | removed to get back to `A` | **17** (unchanged) |
| `|C∖B|` | removed to get back to `B` | **33** (unchanged) |

The removed-node counts are unchanged because `|C∖A| = |C|−|A|` and both counts shifted by the
same one endpoint. The window quantities are unchanged and I re-verified them from the sorted
node lists: max gap of `C` is `2Δt` (the pattern `0,2,3,4,6,…` repeats with period 6 and gaps
`2,1,1,2`; the tail is `96,98,99,100`), so `ΔT_1^C = 2Δt` and `n^C_max = 2`; max gap of `A` is
`2Δt` (including `98→100`) so `ΔT^A = 2Δt`; max gap of `B` is `3Δt` so `ΔT^B = 3Δt`. The
headline of the remark — `|C∖A| + |C∖B| = 50` removed-node terms, more than a single clean
removal — is therefore unchanged.

I choose your **first** option: include `100` in every displayed set and count, and never quote a
launch-node count in the example. Mixing the two conventions is what produced both this and
issue 4, and having exactly one convention is worth the slightly uglier displays.

### 7. "As `Σ→0`, `J_K → J_K^*` uniformly" is not a well-defined deterministic claim — **ACCEPT**

Accept. The proof I wrote cites uniform continuity of `(t,x,p) ↦ φ(t;p,x)` on a compact set — a
deterministic fact about a deterministic limit — while the statement indexes the limit by a
covariance, which does not determine a pathwise sequence without a coupling. Restored to
`‖η‖_max → 0`, and I drop the `Σ → 0` phrasing entirely rather than repairing it with the
coupling `η_i = Σ^{1/2}ξ_i`: the coupled a.s. statement is true but is not used anywhere, and a
second probabilistic sentence in a proposition whose scope is now an event would invite exactly
the confusion of issue 1.

While rewriting I found that the compactness argument can be replaced by an explicit rate, which
is strictly better than what the statement claimed, so I state that instead: with
`β := Q(ΔT)·diam(P)` and `E := ‖η‖_max`, on `E_X`,

`sup_{p∈P} |J_K(p) − J_K^*(p)| ≤ (N−1)(1+e^{μ⁺ΔT}) E ( 2β + (1+e^{μ⁺ΔT}) E )`,

obtained from `|a²−b²| ≤ |a−b|(a+b)`, `|a−b| ≤ (1+e^{μ⁺ΔT})E` (Lemma 1 plus the residual noise)
and `b ≤ β` (Lemma 2). Linear in `E` for small `E`, uniform in `p ∈ P`, no compactness appeal.

### 8. `ρ_s` undefined at `μ = 0` — **ACCEPT**

Accept; a pure omission. Lemma 2 carried the convention and (rho) did not, while Lemma 3 and
Proposition 3 both advertise `μ = 0` coverage. In the rewrite the term is named once,

`Q(s) := (L̃/μ)(e^{μs} − 1)`, read as `L̃ s` when `μ = 0` (the continuous extension,
`lim_{μ→0}(e^{μs}−1)/μ = s`),

and `Q` is then used in Lemma 2, (rho), `ρ⁺`, Lemma 3, Proposition 3, Corollary 1(ii), (A1) and
the envelope remark, so the convention cannot go missing from one of them again. I also record
the two bounds the text uses without proof elsewhere: `Q(s) ≥ 0` for every sign of `μ`,
`Q(s) ≤ L̃ s` when `μ ≤ 0`, and `Q(s) ≤ L̃/|μ|` when `μ < 0`.

### 9. `Δ_K` is no longer a quadratic polynomial — **ACCEPT**, with a **CLARIFY** that repairs the claim instead of weakening it

**Accept the diagnosis.** With `ρ̄_s = max{2‖η‖_max, ρ_s}`, `Δ_K` is a product of two maxima of
affine forms, hence piecewise quadratic, and your identification of the switch is right:
`∂_s ρ_s = e^{μs}(μ‖η‖_max + L̃‖q−p*‖)` has a sign independent of `s`, so the active branch is
`ρ_s` when `μ‖η‖_max + L̃‖q−p*‖ ≥ 0` and `ρ_0 = 2‖η‖_max` otherwise — a switch that occurs only
for `μ < 0`, i.e. exactly in the dissipative case this revision advertises. "Coefficients …
and smaller" is also indefensible: it names one of three coefficients and describes the other two
by an adjective.

**Clarify — I take your second option, and it is better than "piecewise".** Define

`ρ⁺_s(p) := (1 + e^{μ⁺s}) ‖η‖_max + Q(s) ‖p − p*‖`.

Then, for every sign of `μ` and every `s ≥ 0`:

1. `ρ_s ≤ ρ⁺_s`, because `e^{μs} ≤ e^{μ⁺s}` (equality for `μ ≥ 0`; for `μ < 0`, `e^{μs} < 1 = e^0`);
2. `2‖η‖_max ≤ ρ⁺_s`, because `1 + e^{μ⁺s} ≥ 2` (as `μ⁺s ≥ 0`) and `Q(s) ≥ 0`;
   hence **`ρ̄_s ≤ ρ⁺_s`** and every bound proved with `ρ̄` remains true with `ρ⁺`;
3. `s ↦ ρ⁺_s` is **non-decreasing**: `∂_s[(1+e^{μ⁺s})‖η‖_max] = μ⁺e^{μ⁺s}‖η‖_max ≥ 0` and
   `∂_s Q(s) = L̃ e^{μs} ≥ 0` — the monotonicity that `ρ` lacks, now with no `max`;
4. `ρ⁺_s(p*) = (1+e^{μ⁺s})‖η‖_max = ρ̄_s(p*)`, so **nothing is lost at the truth**, which is
   where Proposition 3's displayed form and all of §6's numbers live;
5. `ρ⁺_s` is affine and homogeneous of degree 1 in `(‖η‖_max, ‖p−p*‖)` with non-negative,
   *deterministic* coefficients `A_s = 1+e^{μ⁺s}` and `B_s = Q(s)`.

(1)–(4) checked symbolically and on 2·10⁵ random draws of `(μ, L̃, ‖η‖_max, ‖p−p*‖, s, s′)` with
`μ` of both signs: zero violations. Consequently

`Δ_K(q) = c_0 (A_{ΔT_2} E + B_{ΔT_2} δ)(A_{ΔT_1+ΔT_2} E + B_{ΔT_1+ΔT_2} δ)`,
`c_0 = 2 n_max |I_R| e^{μ⁺ΔT_1}`, `E = ‖η‖_max`, `δ = ‖q−p*‖`,

is a genuine **homogeneous quadratic form with non-negative coefficients**, written out term by
term in the rewrite, and Theorem 1's sentence becomes true as stated rather than needing
"piecewise". Two further benefits I use: `sup_{q∈U} Δ_K(q) = Δ_K` evaluated at
`R_U := sup_{q∈U}‖q−p*‖` (monotone in `δ`, so the supremum in Theorem 1 and Corollary 2 is
*computable*, not just finite), and `Δ_K` is now manifestly `O(E²)` at `p = p*`, which is the
scaling claim the "what changed" paragraph makes. `ρ̄` is retired; it appears only in a one-line
remark saying `ρ̄ ≤ ρ⁺` and that `ρ̄` is the tighter but non-smooth envelope.

### 10. The penalty-bias bound lacks the assumptions needed to derive it — **ACCEPT**

Accept. `‖p^{pen} − p^{(K)}‖ ≤ γ/(m√n_p)` was stated in Remark `rem:impl`(ii) from "`J_K/N` is
`m`-strongly convex near its own minimiser `p^{(K)}`", which is exactly the circularity you
flagged for Theorem 1 in round 1 and which I then reproduced one remark later: the penalised
minimiser may sit outside that neighbourhood, and the argument needs strong monotonicity of
`∇(J_K/N)` along the segment joining the two. The bound is presented as **conditional**, with the
containment hypotheses named, and the derivation uses the constrained optimality conditions of
issue 3 so it too survives boundary minimisers:

With `F = J_K/N`, `G = (γ/n_p)Σ_j smoothℓ1(p_j)`, `p^{(K)} ∈ arg min_P F`,
`p^{pen} ∈ arg min_P (F+G)`, `U ⊆ P` convex containing **both and the segment between them**,
`F` `m`-strongly convex on `U`: add the two variational inequalities
(`⟨∇F(p^{(K)}), p^{pen}−p^{(K)}⟩ ≥ 0` and `⟨∇F(p^{pen})+∇G(p^{pen}), p^{(K)}−p^{pen}⟩ ≥ 0`) to get
`⟨∇F(p^{pen})−∇F(p^{(K)}), p^{pen}−p^{(K)}⟩ ≤ ‖∇G(p^{pen})‖·‖p^{pen}−p^{(K)}‖`, and use strong
monotonicity on `U` for the left side.

Your concession on the gradient factor is correct and I keep it with its proof:
`d/dx smoothℓ1(x) = 2σ(αx) − 1 = tanh(αx/2) ∈ (−1,1)`, so `‖∇G‖_2 ≤ (γ/n_p)√n_p = γ/√n_p`,
uniformly in `p` and independently of `α`. The `α = 500` sharpness therefore does not enter the
bias bound at all — worth saying, since the manuscript elsewhere treats `α` as if it mattered
quantitatively.

### 11. The low-noise-node design rule is not supported by the displayed bound — **ACCEPT**, with a **CLARIFY** that makes the rule implementable

**Accept.** The displayed `Δ_K` contains `|I_R|`, `n_max`, `ΔT_1`, `ΔT_2` and the *global*
`‖η‖_max`. Nothing in it varies with *which* nodes are in `I_R` at fixed cardinality, so "prefer
low-noise nodes" is read off a formula from which every node-specific quantity has already been
maximised away. The rule is real, but it lives in the proof, not in the statement — three
inequalities earlier, before `‖η_{τ_k}‖ → ‖η‖_max`, `|D_k| → n_max` and `Σ_{k∈I_R} → |I_R|`.

**Resolution: the nodewise bound becomes the primary statement of Proposition 3**, with the
collapsed form demoted to a corollary. Writing `h_k = τ_{k+1} − τ_k` (the fine window launched at
the removed node), `w_k = τ_k − τ_k^-` (the offset to its left retained neighbour), and

`g_k(p) = ‖ φ(w_k; p, y_{τ_k^-}) − y_{τ_k} ‖`  (the coarse residual **at** the removed node),

the proof gives, before any collapsing,

`|Ĵ_K(p) − J_K(p)| ≤ 2 Σ_{k∈I_R} |D_k| e^{μ⁺h_k} g_k(p) ρ⁺_{h_k+w_k}(p)`,
`g_k(p) ≤ e^{μ⁺w_k}‖η_{τ_k^-}‖ + ‖η_{τ_k}‖ + Q(w_k)‖p−p*‖ ≤ ρ⁺_{w_k}(p)`,

and `Δ_K` follows by `h_k ≤ ΔT_1`, `w_k ≤ ΔT_2`, `|D_k| ≤ n_max`, `ρ⁺_{w_k} ≤ ρ⁺_{ΔT_2}`. The
design rule is then a statement about the nodewise form and is labelled as such.

**Clarify — and the rule improves in the process.** "Prefer low-noise nodes" is not
implementable: `η_{τ_k}` is unobservable, and if it were observable the noise would be removable.
But `g_k(p)` **is** observable — it is one flow evaluation and one subtraction, both of which the
optimiser already performs, evaluated at the current iterate. So the nodewise bound supports an
*implementable* rule that the collapsed bound cannot support at all:

> remove the nodes minimising `|D_k| · e^{μ⁺h_k} · g_k(p)` at the current guess — i.e. nodes
> whose coarse residual is small, whose left neighbour is close, and which launch few data.

That is a computable node-selection criterion, and it is the first place in this manuscript where
the theory says something an implementation could act on. The honest caveat stays attached: it is
a criterion for making the *bound* small, not a proof that the coarse minimiser moves less, and
`Δ_K` bounds the cost gap, not the displacement, without the strong-convexity premise.

### 12. Two stale narrative claims — **ACCEPT (both)**

**"One-line modification."** Accept without reservation; it was already false when R2 was written
and I did not re-read the abstract against the diff. The revision adds a standing-assumption
block with a tube, a closure condition and an exit-time lemma, a Dini-derivative proof, two new
corollaries, a noise model with an event and a tail bound, a constrained-optimality lemma, a
monotone envelope, a re-indexing of every count from `N` to `N−1`, and a withdrawal of a
structural claim about the algorithm. "One line" describes only the `L → μ` substitution, which
is the smallest of these. Replaced with: *each change is local to one proof step, and the
assumptions under which each statement holds are now stated in full.*

**"The boundary of `P` made visible."** Accept. `P` is a set the analysis *posits*; the plateau is
produced by `abs(state) > 1e3 || isnan(state)` in `ms_loss`, an implementation threshold chosen to
keep the optimiser numerically alive. The implication that the analysis needs runs only one way:
where the flat penalty fires, there is no solution on the window inside `X`, so that `p ∉ P`. The
converse — that every `p ∉ P` triggers the plateau, or that the level set `{loss = 10³}` is
`∂P` — is unproved and almost certainly false (a `p` can be outside `X`-confinement, hence
outside `P`, while still producing states well under `10³`). The sentence is replaced by the
one-directional statement, and the same correction is applied to Remark `rem:impl`(iv), which
carried the same conflation in milder form ("the candidate `p` is outside the set `P`" — true —
"and there is no trajectory to bound" — true; but it must not be read as a characterisation).

---

## Section 2 — Updated artifact

Everything in **R2_to_gpt.md Section 2** that is not listed below is **unchanged** — in
particular blocks (f) Lemma 1, (g) Lemma 2 (except that its `μ=0` term is now written `Q(t)`),
(k) the definition of `Ĵ_K`, (q) the Figure 10 caption, and (r) the caption and gate-bullet edits
all stand as written there. Only the items below change this round.

### 2.1 Bullet list of changes (this round)

**Assumptions and conventions**

1. (A0) split into **(A0)** regularity + admissible region built as a *tube around the true
   orbit* (so the clean states and all chords are inside by construction), **(A1)** a checkable
   *closure/no-escape* condition with an exit-time lemma (Lemma 0) replacing assumed forward
   invariance, and **(A2)** the *noise model* with the event `E_X`, in a bounded variant (A2-b)
   and a Gaussian variant (A2-g) with an explicit tail bound. [issues 1, 2]
2. New **Lemma 0'** (`lem:trunc`): conditioning on `E_X` preserves independence, mean zero and
   `E‖η_i‖² ≤ tr Σ`, so the expectation bounds survive conditioning. [issue 1]
3. R2's strong-convexity assumption, previously labelled **(A2)**, is renamed **(A3)** to free
   the label for the noise model. Every reference to "(A2)" in Proposition 2 becomes "(A3)".
4. New **minimiser convention**: all minimisers are over `P`; existence by compactness; a
   **minimiser inequality lemma** (`lem:vi`) replaces every `∇J = 0` in §4–§5. [issues 3, 10]
5. `Q(s) := (L̃/μ)(e^{μs}−1)`, `:= L̃ s` at `μ=0`, named once and used everywhere; `ρ̄` retired
   in favour of the affine, monotone, quadratic-friendly `ρ⁺`. [issues 8, 9]

**Statements**

6. Lemma 3: `ρ̄ → ρ⁺`. [issue 9]
7. Proposition 1: realisation-wise on `E_X`; expectation unconditional under (A2-b) and
   conditional on `E_X` under (A2-g); explicit `‖η‖_max → 0` rate replacing the `Σ → 0` claim.
   [issues 1, 7]
8. Proposition 2: minimiser inequality instead of `∇J_K = 0`; expectation form conditional;
   (A2) → (A3). [issues 1, 3]
9. Proposition 3: **nodewise** statement primary, collapsed form as Corollary; `ρ⁺`; the design
   rule restated on the nodewise form with the observable `g_k`. [issues 9, 11]
10. Theorem 1: minimiser inequality; `Δ_K` written out as a quadratic form with its three
    coefficients; `sup_U Δ_K` evaluated at `R_U`. [issues 3, 9]
11. Corollary 2 (basin retention): `(B0) B̄ ⊆ P`, and `B̄ ⊆ int P` for the stationarity sentence
    only; minimiser inequality. [issue 3]
12. Corollary 3 (re-partition): `ρ⁺`; example counts corrected to 51 / 35 / 18 / 68. [issues 6, 9]
13. Corollary 1: item (iii) **deleted**; new remark on the envelopes `𝓜(h)`, `𝒬(h)` and on what
    Table 2 does and does not report. [issue 5]
14. Nesting paragraph: corrected criterion and corrected counts, with the degeneracy of the
    `κ_b = M` transitions stated. [issue 4]
15. Abstract: "one-line modification" deleted; blow-up/`∂P` sentence made one-directional in
    both places it occurs. [issue 12]

### 2.2 Corrected LaTeX

#### (A) §2.1 — standing assumptions [replaces R2 block (b)'s "Standing assumption (A0)" paragraph; the *Data and indexing*, *Partitions* and *Node removal* paragraphs of block (b) are unchanged]

```latex
\paragraph{Standing assumptions (A0)--(A2).}
All three are fixed \emph{before the data are seen}; nothing below is chosen a posteriori, so
none of $X$, $P$, $L$, $\tilde L$, $\mu$, $\mu_-$ is a random object.

\smallskip\noindent\textbf{(A0) Regularity and admissible region.}
Fix a compact convex parameter set $P\subset\mathbb R^{n_p}$ with $\pstar\in P$; a tube radius
$r_X>0$; the closed tube around the true orbit
\begin{equation}
\mathcal T=\big\{x\in\mathbb R^{d}:\ \operatorname{dist}\big(x,\,x^\star([t_0,t_{N-1}])\big)\le r_X\big\};
\label{eq:tube}
\end{equation}
a compact \emph{convex} $X\subset\mathbb R^{d}$ with $\mathcal T\subseteq X$; and a horizon
$\Delta T_{\max}>0$ at least as large as the longest window of any partition under discussion.
Assume $f$ is $C^1$ on an open neighbourhood of $X\times P$, and that for every $(x,p)\in X\times P$
the solution of $\dot\xi=f(\xi;p)$, $\xi(0)=x$, exists for as long as it remains in $X$, up to time
$\Delta T_{\max}$. All constants of \S\ref{sec:consts} are suprema or infima over $X\times P$.

\smallskip\noindent\textbf{(A1) Closure of the admissible region.}
With $\mu^{+}=\max(\mu,0)$ and $Q(s)=\frac{\tilde L}{\mu}(e^{\mu s}-1)$ (read as $\tilde L s$ when
$\mu=0$), there is $r'\in(0,r_X]$ with
\begin{equation}
e^{\mu^{+}\Delta T_{\max}}\,r' \;+\; Q(\Delta T_{\max})\,\operatorname{diam}(P)\;<\;r_X .
\label{eq:closure}
\end{equation}

\begin{lemma}[no escape from the tube]\label{lem:noescape}
Assume (A0)--(A1). Let $\tau\in[t_0,t_{N-1}]$, $x_\tau=x^\star(\tau)$, let $z\in\mathbb R^{d}$ with
$\norm{z-x_\tau}\le r'$, and let $0\le s\le\min\{\Delta T_{\max},\,t_{N-1}-\tau\}$. Then for every
$p\in P$
\[
\flow{s}{p}{x_\tau}\in\mathcal T\subseteq X
\qquad\text{and}\qquad
\flow{s}{p}{z}\in\mathcal T\subseteq X ,
\]
and consequently every chord between points of these trajectories lies in $X$.
\end{lemma}
\begin{proof}
Both statements are exit-time bootstraps; the first is needed to license the second.
\emph{Clean launch.} Let $T=\sup\{s\le\Delta T_{\max}:\ \flow{r}{p}{x_\tau}\in\mathcal T\ \forall r\le s\}$.
For $r<T$ both $\flow{r}{p}{x_\tau}$ and $x^\star(\tau+r)=\flow{r}{\pstar}{x_\tau}$ lie in
$\mathcal T\subseteq X$, and $X$ is convex, so Lemma~\ref{lem:param} applies and gives
$\norm{\flow{r}{p}{x_\tau}-x^\star(\tau+r)}\le Q(r)\norm{p-\pstar}\le Q(\Delta T_{\max})\operatorname{diam}(P)<r_X$
by \eqref{eq:closure}. If $T<\min\{\Delta T_{\max},t_{N-1}-\tau\}$ then by continuity the distance
to the orbit equals $r_X$ at $r=T$, contradicting the strict inequality; hence no exit.
\emph{Noisy launch.} Repeat with $\delta(r)=\flow{r}{p}{z}-x^\star(\tau+r)$, splitting
$\delta=[\flow{r}{p}{z}-\flow{r}{p}{x_\tau}]+[\flow{r}{p}{x_\tau}-x^\star(\tau+r)]$; the first
bracket is bounded by $e^{\mu r}\norm{z-x_\tau}\le e^{\mu^{+}\Delta T_{\max}}r'$ by
Lemma~\ref{lem:state} (legitimate on $[0,T)$, where both trajectories are in $X$ by the previous
step and by the definition of $T$), the second by $Q(\Delta T_{\max})\operatorname{diam}(P)$. Their
sum is $<r_X$ by \eqref{eq:closure}, and the same contradiction closes the bootstrap. The chord
claim is convexity of $X$.
\end{proof}

\smallskip\noindent\textbf{(A2) Noise model and the admissible event.}
$y_i=x_i+\eta_i$, $x_i=x^\star(t_i)$, with $\eta_0,\dots,\eta_{N-1}$ i.i.d., $\mathbb E\eta_i=0$ and
$\operatorname{Cov}(\eta_i)=\Sigma=\operatorname{diag}(\sigma_1^2,\dots,\sigma_d^2)$, so
$\mathbb E\norm{\eta_i}^2=\operatorname{tr}\Sigma$. Two variants are used:
\begin{enumerate}[nosep,label=(A2-\alph*)]
\item \emph{bounded}: $\norm{\eta_i}\le r'$ almost surely;
\item \emph{Gaussian}: $\eta_i\sim\mathcal N(0,\Sigma)$, the law the data generator of R002 samples.
\end{enumerate}
Define the \emph{admissible event}
\begin{equation}
E_X=\Big\{\max_{0\le i\le N-1}\norm{\eta_i}\le r'\Big\}.
\label{eq:EX}
\end{equation}
Under (A2-a), $\mathbb P(E_X)=1$. Under (A2-b), for $r'>\sqrt{\operatorname{tr}\Sigma}$,
\begin{equation}
\mathbb P(E_X^{c})\ \le\ N\min\Big\{\frac{\operatorname{tr}\Sigma}{r'^{\,2}},\
\exp\Big(-\frac{(r'-\sqrt{\operatorname{tr}\Sigma})^{2}}{2\norm{\Sigma}_2}\Big)\Big\}
\label{eq:tail}
\end{equation}
(Markov on $\norm{\eta_i}^2$; Gaussian concentration of the $\norm{\Sigma}_2^{1/2}$-Lipschitz map
$\xi\mapsto\norm{\Sigma^{1/2}\xi}$ about its mean, together with
$\mathbb E\norm{\eta_i}\le\sqrt{\operatorname{tr}\Sigma}$).

\readthis{scope, and it is the tightest constraint in this report.
\emph{(i)} On $E_X$ every shooting node satisfies $\norm{y_i-x_i}\le r'$, so
Lemma~\ref{lem:noescape} puts every launched trajectory --- noisy or clean, at every $p\in P$ ---
inside $\mathcal T\subseteq X$ for a full window, and every mean-value chord inside $X$ by
convexity. \emph{Every inequality in \S\ref{sec:lemmas}--\S\ref{sec:removal} is therefore a
deterministic inequality that holds realisation by realisation on $E_X$, and is asserted nowhere
else.}
\emph{(ii)} Expectations are unconditional under (A2-a) and \emph{conditional on $E_X$} under
(A2-b); Lemma~\ref{lem:trunc} is what makes the latter carry the same constants. No unconditional
expectation is claimed under (A2-b): off $E_X$ the constants of (A0) do not apply, a polynomial
right-hand side can blow up in finite time, and $\mathbb E J_K(\pstar)$ is not asserted to be
finite.
\emph{(iii)} \eqref{eq:closure} is a \emph{requirement on the FHN numbers, and it is not verified
anywhere in this report}. It is a genuine restriction: with $\mu=\nmu>0$ it forces $r'$ to be
exponentially small in $\Delta T_{\max}$, so for single shooting ($\Delta T_{\max}=100$) the
admissible event is effectively empty and the theory says nothing --- which is consistent with,
but not a proof of, the manuscript's qualitative claim about single shooting. This is recorded as
an open gate in \S\ref{sec:premises}.}

\begin{lemma}[conditioning on $E_X$ preserves the noise structure]\label{lem:trunc}
Assume (A2-b). Conditionally on $E_X$, the vectors $\eta_0,\dots,\eta_{N-1}$ are independent,
$\mathbb E[\eta_i\mid E_X]=0$, and $\mathbb E[\norm{\eta_i}^2\mid E_X]\le\operatorname{tr}\Sigma$.
\end{lemma}
\begin{proof}
$E_X=\bigcap_i A_i$ with $A_i=\{\norm{\eta_i}\le r'\}$ is a product event, so the conditional joint
law is the product of the conditional marginals: independence is preserved. Each $A_i$ and the
density of $\mathcal N(0,\Sigma)$ are invariant under $\eta\mapsto-\eta$, so each conditional law
is centrally symmetric and its mean is $0$. Finally $\norm{\eta_i}^2$ is non-decreasing and
$\mathbf 1_{A_i}$ non-increasing in the same scalar $\norm{\eta_i}$, so
$\operatorname{Cov}(\norm{\eta_i}^2,\mathbf 1_{A_i})\le0$ by Chebyshev's association inequality,
i.e.\ $\mathbb E[\norm{\eta_i}^2\mathbf 1_{A_i}]\le\operatorname{tr}\Sigma\cdot\mathbb P(A_i)$.
\end{proof}
```

#### (B) §2.1 — minimiser convention [new paragraph, placed after the cost definitions of R2 block (b)]

```latex
\paragraph{Minimisers, and the only optimality condition used.}
All minimisers are taken over the compact set $P$ of (A0):
\begin{equation}
p^{(K)}\in\arg\min_{p\in P}J_K(p),\qquad
\hat p^{(K)}\in\arg\min_{p\in P}\hatJ_K(p),\qquad
p^{\rm pen}\in\arg\min_{p\in P}\big(J_K/N+\text{penalty}\big),
\label{eq:minimisers}
\end{equation}
each of which exists because the objective is continuous and $P$ compact --- the draft's
$\arg\min_p$ over $\mathbb R^{n_p}$ guaranteed neither existence nor validity of the constants.

\begin{lemma}[minimiser inequality]\label{lem:vi}
Let $P$ be convex, $F$ differentiable on a neighbourhood of $P$, $p\in\arg\min_{P}F$, and let
$U\subseteq P$ be convex with $p\in U$ and $F$ $m$-strongly convex on $U$. Then
\begin{equation}
F(q)\ \ge\ F(p)+\tfrac m2\norm{q-p}^2\qquad\text{for all }q\in U.
\label{eq:vi}
\end{equation}
\end{lemma}
\begin{proof}
First-order optimality on a convex set gives $\langle\nabla F(p),q-p\rangle\ge0$ for every $q\in P$,
hence for every $q\in U$. Strong convexity on $U$ gives
$F(q)\ge F(p)+\langle\nabla F(p),q-p\rangle+\tfrac m2\norm{q-p}^2$. Add.
\end{proof}
\readthis{\eqref{eq:vi} is the \emph{only} consequence of optimality used in
Proposition~\ref{prop:perr}, Theorem~\ref{thm:main}, Corollaries~\ref{cor:basin} and
\ref{cor:repartition} and Remark~\ref{rem:impl}(ii). The draft, and the previous revision, wrote
$\nabla J_K(p^{(K)})=0$ in each of those proofs, which is false at a minimiser on $\partial P$ ---
and the $\ell_1$-type penalty makes boundary and near-boundary solutions the typical case, not an
edge case. With \eqref{eq:vi} every one of those proofs holds verbatim for boundary minimisers and
no minimiser is required to be interior. Interiority is needed in exactly one place, and is
assumed there explicitly: the \emph{stationarity} sentence of Corollary~\ref{cor:basin}.}
```

#### (C) §2.2 — the residual envelope [replaces eq. (rho) and its `\readthis` in R2 block (c); the rest of block (c) is unchanged]

```latex
Write $\mu^{+}=\max(\mu,0)$ and
\begin{equation}
Q(s)=\frac{\tilde L}{\mu}\big(e^{\mu s}-1\big),\qquad\text{read as }\ \tilde L\,s\ \text{ when }\mu=0,
\label{eq:Qdef}
\end{equation}
the continuous extension $\lim_{\mu\to0}(e^{\mu s}-1)/\mu=s$. For every sign of $\mu$, $Q(s)\ge0$
and $s\mapsto Q(s)$ is non-decreasing, with $Q(s)\le\tilde L s$ when $\mu\le0$ and
$Q(s)\le\tilde L/|\mu|$ when $\mu<0$. The residual abbreviations are
\begin{equation}
\rho_s(p)=\big(1+e^{\mu s}\big)\norm{\eta}_{\max}+Q(s)\norm{p-\pstar},
\qquad
\rho^{+}_s(p)=\big(1+e^{\mu^{+}s}\big)\norm{\eta}_{\max}+Q(s)\norm{p-\pstar},
\qquad s\ge0 .
\label{eq:rho}
\end{equation}
\begin{lemma}[properties of $\rho^{+}$]\label{lem:rhoplus}
For every $s\ge0$, every $p\in P$ and every sign of $\mu$:
\emph{(i)} $\rho_s\le\rho^{+}_s$ and $2\norm\eta_{\max}\le\rho^{+}_s$, hence
$\max\{2\norm\eta_{\max},\sup_{0\le r\le s}\rho_r\}\le\rho^{+}_s$;
\emph{(ii)} $s\mapsto\rho^{+}_s(p)$ is non-decreasing;
\emph{(iii)} $\rho^{+}_s$ is affine and homogeneous of degree $1$ in
$(\norm\eta_{\max},\norm{p-\pstar})$ with non-negative deterministic coefficients
$A_s=1+e^{\mu^{+}s}$ and $B_s=Q(s)$;
\emph{(iv)} $\rho^{+}_s(\pstar)=(1+e^{\mu^{+}s})\norm\eta_{\max}$.
\end{lemma}
\begin{proof}
(i) $e^{\mu s}\le e^{\mu^{+}s}$, and $1+e^{\mu^{+}s}\ge2$ because $\mu^{+}s\ge0$, while $Q(s)\ge0$;
the third claim follows since $\sup_{0\le r\le s}\rho_r=\max\{\rho_0,\rho_s\}$ (the sign of
$\partial_r\rho_r=e^{\mu r}(\mu\norm\eta_{\max}+\tilde L\norm{p-\pstar})$ does not depend on $r$)
and $\rho_0=2\norm\eta_{\max}$. (ii) $\partial_s\rho^{+}_s=\mu^{+}e^{\mu^{+}s}\norm\eta_{\max}
+\tilde L e^{\mu s}\norm{p-\pstar}\ge0$. (iii), (iv) are immediate.
\end{proof}
\readthis{$\rho_s$ is \emph{not} monotone in $s$: for $\mu<0$ and $p=\pstar$ it strictly
\emph{de}creases. Every step that bounds a shorter window's residual by a longer window's takes
$\rho^{+}$, never $\rho$. The previous revision used the exact envelope
$\bar\rho_s=\max\{2\norm\eta_{\max},\rho_s\}$; $\bar\rho_s\le\rho^{+}_s$, so $\bar\rho$ is tighter,
but it is a maximum of two affine forms and therefore makes $\Delta_K$ of
Proposition~\ref{prop:removal} only \emph{piecewise} quadratic, with the active branch switching
on the sign of $\mu\norm\eta_{\max}+\tilde L\norm{q-\pstar}$ when $\mu<0$.
$\rho^{+}$ is a single affine majorant, is monotone for every sign of $\mu$, and by
Lemma~\ref{lem:rhoplus}(iv) \emph{costs nothing at $p=\pstar$}, which is where every number in
\S\ref{sec:num} is evaluated. $\bar\rho$ is not used below.}
```

#### (D) Lemma 3 [replaces R2 block (h)]

```latex
\begin{lemma}[largest residual of a window; \rev{revised}]\label{lem:residual}
Assume (A0)--(A2) and work on the event $E_X$ of \eqref{eq:EX}. Let $\tau$ be a node, let
$s\in(0,\Delta T_{\max}]$ and let $t_i\in(\tau,\tau+s]$. Then, with $\sigma_i=t_i-\tau$ and every
$p\in P$,
\begin{equation}
\norm{\flow{\sigma_i}{p}{y_\tau}-y_i}\ \le\
e^{\mu\sigma_i}\norm{\eta_\tau}+Q(\sigma_i)\norm{p-\pstar}+\norm{\eta_i}
\ \le\ \rho_{\sigma_i}(p)\ \le\ \rho^{+}_{s}(p).
\label{eq:residual}
\end{equation}
\end{lemma}
\begin{proof}
On $E_X$, $\norm{\eta_\tau}\le r'$, so Lemma~\ref{lem:noescape} places $\flow{r}{p}{y_\tau}$,
$\flow{r}{p}{x_\tau}$ and $x^\star(\tau+r)$ in $\mathcal T\subseteq X$ for all $r\le\sigma_i$, and
all chords in $X$; the constants of (A0) therefore apply to every step below. Split
$\flow{\sigma_i}{p}{y_\tau}-y_i=[\flow{\sigma_i}{p}{y_\tau}-\flow{\sigma_i}{p}{x_\tau}]
+[\flow{\sigma_i}{p}{x_\tau}-x_i]-\eta_i$. Lemma~\ref{lem:state} bounds the first bracket by
$e^{\mu\sigma_i}\norm{\eta_\tau}$; since $x_i=\flow{\sigma_i}{\pstar}{x_\tau}$,
Lemma~\ref{lem:param} bounds the second by $Q(\sigma_i)\norm{p-\pstar}$. With
$\norm{\eta_\tau},\norm{\eta_i}\le\norm\eta_{\max}$ this is $\rho_{\sigma_i}(p)$, and
$\rho_{\sigma_i}\le\rho^{+}_{\sigma_i}\le\rho^{+}_{s}$ by Lemma~\ref{lem:rhoplus}(i)--(ii).
\readthis{the last step is \emph{not} monotonicity of $\rho$, which fails for $\mu<0$; it is
Lemma~\ref{lem:rhoplus}.}
\end{proof}
```

#### (E) Proposition 1 [replaces R2 block (i)]

```latex
\begin{proposition}[cost at the true parameter; \rev{revised}]\label{prop:cost}
Assume (A0)--(A2). Consider a partition with longest window $\Delta T\le\Delta T_{\max}$ and $n_k$
data in window $k$, $\sum_k n_k=N-1$.

\emph{(a) Realisation-wise, on $E_X$:} for every $p\in P$,
\begin{align}
J_K(p)&\le\sum_{k=1}^{K}\sum_{t_i\in(\tau_{k-1},\tau_k]}
 \Big(e^{\mu(t_i-\tau_{k-1})}\norm{\eta_{\tau_{k-1}}}
 +Q(t_i-\tau_{k-1})\norm{p-\pstar}+\norm{\eta_i}\Big)^2,\label{eq:JK_bound}\\
J_K(\pstar)&\le\norm{\eta}_{\max}^2\sum_{k=1}^{K}\sum_{j=1}^{n_k}\big(1+e^{\mu j\Delta t}\big)^2
 \ \le\ (N-1)\,\norm{\eta}_{\max}^2\big(1+e^{\mu^{+}\Delta T}\big)^2 .\label{eq:JK_true}
\end{align}
Moreover, with $\beta:=Q(\Delta T)\operatorname{diam}(P)$,
\begin{equation}
\sup_{p\in P}\big|J_K(p)-J_K^\star(p)\big|\ \le\
(N-1)\big(1+e^{\mu^{+}\Delta T}\big)\norm\eta_{\max}
\Big(2\beta+\big(1+e^{\mu^{+}\Delta T}\big)\norm\eta_{\max}\Big),
\label{eq:JK_unif}
\end{equation}
so $J_K\to J_K^\star$ uniformly on $P$ as $\norm\eta_{\max}\to0$, and $J_K^\star(\pstar)=0$.

\emph{(b) In expectation.} Under (A2-a),
\begin{equation}
\mathbb E\,J_K(\pstar)\le\operatorname{tr}\Sigma\sum_{k=1}^{K}\sum_{j=1}^{n_k}\big(1+e^{2\mu j\Delta t}\big)
 \ \le\ (N-1)\operatorname{tr}\Sigma\big(1+e^{2\mu^{+}\Delta T}\big).
\label{eq:JK_expect}
\end{equation}
Under (A2-b) the same two bounds hold for $\mathbb E\big[J_K(\pstar)\mid E_X\big]$, and
$\mathbb P(E_X^{c})$ is bounded by \eqref{eq:tail}. No unconditional expectation is claimed under
(A2-b).

\emph{(c) Single shooting.} On $E_X$,
$J_1(p)\le\sum_{i=1}^{N-1}\big(e^{\mu(t_i-t_0)}\norm{\eta_0}+Q(t_i-t_0)\norm{p-\pstar}+\norm{\eta_i}\big)^2$,
in which the initial-condition error is amplified over the whole record.
\end{proposition}
\begin{proof}
\emph{(a)} \eqref{eq:JK_bound} is \eqref{eq:residual} squared and summed. At $\pstar$ the middle
term vanishes and the $j$-th datum of a window sits at $t_i-\tau_{k-1}=j\Delta t$, giving the first
inequality of \eqref{eq:JK_true}; for the second, $e^{\mu j\Delta t}\le e^{\mu^{+}\Delta T}$ in both
cases ($\mu\ge0$: $\mu j\Delta t\le\mu\Delta T$; $\mu<0$: $e^{\mu j\Delta t}\le1=e^{\mu^{+}\Delta T}$),
and $\sum_k n_k=N-1$. For \eqref{eq:JK_unif}, write $a=\norm{\flow{\sigma}{p}{y_\tau}-y_i}$ and
$b=\norm{\flow{\sigma}{p}{x_\tau}-x_i}$ for the same index; then
$|a^2-b^2|\le|a-b|(a+b)$, $|a-b|\le e^{\mu\sigma}\norm{\eta_\tau}+\norm{\eta_i}
\le(1+e^{\mu^{+}\Delta T})\norm\eta_{\max}$ and $b\le Q(\sigma)\norm{p-\pstar}\le\beta$ by
Lemma~\ref{lem:param}, so $a+b\le2\beta+(1+e^{\mu^{+}\Delta T})\norm\eta_{\max}$; sum over the $N-1$
indices. The bound is uniform in $p\in P$ and vanishes linearly in $\norm\eta_{\max}$.

\emph{(b)} Expand $\norm{u-\eta_i}^2=\norm u^2-2\langle u,\eta_i\rangle+\norm{\eta_i}^2$ with
$u=\flow{t_i-\tau_{k-1}}{\pstar}{y_{\tau_{k-1}}}-x_i$, a function of $\eta_{\tau_{k-1}}$ alone. Since
$t_i\in(\tau_{k-1},\tau_k]$ we have $t_i\neq\tau_{k-1}$, so $\eta_i$ and $\eta_{\tau_{k-1}}$ are
distinct members of the i.i.d.\ family; independence and $\mathbb E\eta_i=0$ kill the cross term
\emph{termwise} (unaffected by $\eta_{\tau_k}$ appearing both as a residual noise in window $k$ and
as a launch noise in window $k+1$: those are different summands). Then
$\mathbb E\norm{\eta_i}^2=\operatorname{tr}\Sigma$ and
$\mathbb E\norm u^2\le e^{2\mu j\Delta t}\mathbb E\norm{\eta_{\tau_{k-1}}}^2$ by
Lemma~\ref{lem:state}, which under (A2-a) applies surely because $\mathbb P(E_X)=1$. Under (A2-b)
run the identical computation under $\mathbb P(\cdot\mid E_X)$: by Lemma~\ref{lem:trunc} the $\eta_i$
are still independent and mean zero, so the cross term still vanishes, and
$\mathbb E[\norm{\eta_i}^2\mid E_X]\le\operatorname{tr}\Sigma$ gives the same right-hand side.
\end{proof}
\emph{What changed.} (i) $L\to\mu$. (ii) The draft applied $e^{L\Delta T}$ to every datum in a
window; \eqref{eq:JK_true} keeps the offset $j\Delta t$, and the geometric sum
$\sum_{j\le n_k}e^{2\mu j\Delta t}$ saves a factor of order $\mu\Delta T$ relative to
$n_ke^{2\mu\Delta T}$ when $\mu>0$. (iii) $\norm\eta_{\max}\to\operatorname{tr}\Sigma$ in expectation.
On the FHN data the realised ratio
$\norm\eta_{\max}^2/\big(\tfrac1N\sum_i\norm{\eta_i}^2\big)$ is $\nnoiseSlack$ (a direct measurement
from the residual array; the isotropic Gaussian heuristic $1+2\log N/d$ gives $\cnoiseslackrule$, a
different quantity that the data happen to sit near and that does not apply to the anisotropic
$\Sigma$). (iv) $\mu^{+}$ and $N-1$ in the collapsed bounds: for $\mu\le0$ they read
$4(N-1)\norm\eta_{\max}^2$ and $2(N-1)\operatorname{tr}\Sigma$, whereas the draft's $e^{\mu\Delta T}$
form reversed the inequality. (v) The scope is now explicit: (a) is pathwise on $E_X$, (b) is
unconditional only under bounded noise. (vi) The draft's ``as $\Sigma\to0$'' has been replaced by
the deterministic limit $\norm\eta_{\max}\to0$ with the explicit rate \eqref{eq:JK_unif}; a
covariance does not index a pathwise sequence without a coupling.
```

#### (F) Proposition 2 [replaces R2 block (j)]

```latex
\begin{proposition}[displacement of the minimiser from the truth]\label{prop:perr}
Assume (A0)--(A2) and $p^{(K)}\in\arg\min_{P}J_K$ as in \eqref{eq:minimisers}. Suppose $J_K$ is
$m$-strongly convex, $m>0$, on a convex $U\subseteq P$ containing $p^{(K)}$ and $\pstar$. Then, on
$E_X$, realisation by realisation,
\begin{equation}
\norm{\pstar-p^{(K)}}\ \le\ \sqrt{\frac2m\Big(J_K(\pstar)-J_K(p^{(K)})\Big)}
\ \le\ \sqrt{\frac2m\Big((N-1)\norm\eta_{\max}^2\big(1+e^{\mu^{+}\Delta T}\big)^2-J_K(p^{(K)})\Big)}.
\label{eq:perr}
\end{equation}
If in addition
\begin{enumerate}[nosep,label=(A3)]
\item there exist a \emph{deterministic} $m>0$ and a \emph{deterministic} convex $U\subseteq P$ with
$\pstar\in U$ such that, almost surely on $E_X$, $J_K$ is $m$-strongly convex on $U$ and
$p^{(K)}\in U$, and $p^{(K)}$ is measurable,
\end{enumerate}
then, under (A2-a),
$\mathbb E\norm{\pstar-p^{(K)}}^2\le\tfrac2m\big((N-1)\operatorname{tr}\Sigma(1+e^{2\mu^{+}\Delta T})
-\mathbb E J_K(p^{(K)})\big)$, and under (A2-b) the same bound holds for
$\mathbb E\big[\norm{\pstar-p^{(K)}}^2\mid E_X\big]$ with
$\mathbb E[J_K(p^{(K)})\mid E_X]$ on the right. For single shooting,
$\norm{\pstar-p^{(1)}}\le\sqrt{\tfrac2m\big(\sum_{i=1}^{N-1}(e^{\mu(t_i-t_0)}\norm{\eta_0}
+\norm{\eta_i})^2-J_1(p^{(1)})\big)}$.
\end{proposition}
\begin{proof}
Lemma~\ref{lem:vi} with $F=J_K$, $p=p^{(K)}$, $q=\pstar\in U$ gives the first bound;
Proposition~\ref{prop:cost}(a) gives the second. Under (A3) the pointwise inequality holds on a set
of probability one (respectively, of full conditional probability given $E_X$) with a common
constant and a common neighbourhood, so expectations may be taken on both sides, and
Proposition~\ref{prop:cost}(b) supplies the right-hand side.
\end{proof}
\readthis{two independent caveats, and the assumption labels changed.
\emph{First}, what was called (A2) in the previous revision is now \emph{(A3)}; (A2) is the noise
model. \emph{Second}, (A3) is an assumption about the \emph{noise-dependent} objects $J_K$,
$p^{(K)}$, $m$ and $U$, and it is \emph{not} verified for the FHN data:
Figure~\ref{fig:hessian} shows the noisy data term is indefinite at $\pstar$ for every
$\kappa\ge2$, and Figure~\ref{fig:post}(a) shows the points a derivative-free optimiser returns are
positive definite in only $\nfracPD$ of the (seed, $\kappa$) cells. Without (A3) the expectation
form is a modelling statement, not a consequence of \eqref{eq:perr}. \emph{Third}, the proof no
longer uses $\nabla J_K(p^{(K)})=0$ --- which is false when the minimiser lies on $\partial P$ ---
but Lemma~\ref{lem:vi}. \emph{Fourth}, see Remark~\ref{rem:impl}(ii): this proposition compares
with $\pstar$ and therefore does not apply to the penalised objective actually optimised, whose
minimiser is displaced from $p^{(K)}$ by up to $\gamma/(m\sqrt{n_p})$ under the containment
hypotheses stated there.}
```

#### (G) Proposition 3 [replaces R2 block (l)]: nodewise statement primary, collapsed form as a corollary

```latex
\begin{proposition}[cost change under node removal, nodewise; \rev{revised}]\label{prop:removal}
Assume (A0)--(A2) and work on $E_X$. Let the coarse partition be a sub-partition of the fine one,
with removed indices $\mathcal I_R\subset\{1,\dots,K-1\}$. For $k\in\mathcal I_R$ put
\[
h_k=\tau_{k+1}-\tau_k,\qquad w_k=\tau_k-\tau_k^{-},\qquad
g_k(p)=\big\lVert\flow{w_k}{p}{y_{\tau_k^{-}}}-y_{\tau_k}\big\rVert,
\]
so $h_k$ is the fine window launched at the removed node, $w_k$ the offset to the nearest retained
node on its left, and $g_k(p)$ the \emph{coarse residual at the removed node}. Then for every
$p\in P$
\begin{equation}
\big|\hatJ_K(p)-J_K(p)\big|\ \le\
2\sum_{k\in\mathcal I_R}|D_k|\;e^{\mu^{+}h_k}\;g_k(p)\;\rho^{+}_{h_k+w_k}(p),
\label{eq:prop_removal_node}
\end{equation}
and each node factor obeys
\begin{equation}
g_k(p)\ \le\ e^{\mu^{+}w_k}\norm{\eta_{\tau_k^{-}}}+\norm{\eta_{\tau_k}}+Q(w_k)\norm{p-\pstar}
\ \le\ \rho^{+}_{w_k}(p).
\label{eq:gk}
\end{equation}
\end{proposition}
\begin{proof}
\emph{Which data change predictor.} In $J_K$ the datum $t_i\in(\tau_{k-1},\tau_k]$ is launched from
$\tau_{k-1}$; in $\hatJ_K$ from the last \emph{retained} node at or before $\tau_{k-1}$. So $t_i$
changes predictor iff its fine launching node is removed, i.e.\ exactly for
$t_i\in\bigcup_{k\in\mathcal I_R}D_k$, $D_k=\{t_i\in(\tau_k,\tau_{k+1}]\}$. The half-open convention
handles the two apparent off-by-one cases: with $\tau_k$ removed and $\tau_{k-1}$ retained, the
datum at time $\tau_k$ lies in $(\tau_{k-1},\tau_k]$, keeps its predictor, and is correctly
\emph{not} in $D_k$; with $\tau_{k-1}$ and $\tau_k$ both removed, the datum at $\tau_k$ lies in
$D_{k-1}$, does change predictor, and is correctly counted there. The $D_k$ are pairwise disjoint.

\emph{The per-datum estimate.} For $t_i\in D_k$ put $u_i=\flow{t_i-\tau_k^{-}}{p}{y_{\tau_k^{-}}}$,
$v_i=\flow{t_i-\tau_k}{p}{y_{\tau_k}}$, $a_i=\norm{u_i-y_i}$, $b_i=\norm{v_i-y_i}$. Then
$\hatJ_K-J_K=\sum_{k\in\mathcal I_R}\sum_{t_i\in D_k}(a_i^2-b_i^2)$ and
$|a_i^2-b_i^2|\le\norm{u_i-v_i}(a_i+b_i)$ by the reverse triangle inequality. By the flow property
$u_i=\flow{t_i-\tau_k}{p}{\flow{w_k}{p}{y_{\tau_k^{-}}}}$, so Lemma~\ref{lem:state} gives
$\norm{u_i-v_i}\le e^{\mu(t_i-\tau_k)}g_k(p)\le e^{\mu^{+}h_k}g_k(p)$ --- with $\mu^{+}$, because
$t_i-\tau_k\le h_k$ bounds the exponent only when $\mu\ge0$, while for $\mu<0$ the correct majorant
is $e^0=1$. Lemma~\ref{lem:residual} bounds $a_i$ (elapsed time $t_i-\tau_k^{-}\le h_k+w_k$) and
$b_i$ (elapsed time $\le h_k$) by $\rho^{+}_{h_k+w_k}(p)$, using
Lemma~\ref{lem:rhoplus}(ii). Summing over $D_k$ and over $k\in\mathcal I_R$ gives
\eqref{eq:prop_removal_node}. \eqref{eq:gk} is Lemma~\ref{lem:residual} applied to the single datum
$y_{\tau_k}$ launched from $y_{\tau_k^{-}}$ over elapsed time $w_k$. Lemma~\ref{lem:noescape} makes
every flow evaluation above admissible, since on $E_X$ every launch state is within $r'$ of the
orbit.
\end{proof}

\begin{corollary}[collapsed form]\label{cor:removal_global}
Under the hypotheses of Proposition~\ref{prop:removal}, with $n_{\max}=\max_k|D_k|$,
$\Delta T_1=\max_k(\tau_k-\tau_{k-1})$ and $\Delta T_2=\max_{k\in\mathcal I_R}w_k$,
\begin{equation}
\big|\hatJ_K(p)-J_K(p)\big|\ \le\
2\,n_{\max}\,|\mathcal I_R|\;e^{\mu^{+}\Delta T_1}\;\rho^{+}_{\Delta T_2}(p)\;
\rho^{+}_{\Delta T_1+\Delta T_2}(p)\ =:\ \Delta_K(p).
\label{eq:prop_removal}
\end{equation}
At $p=\pstar$ this is
$2n_{\max}|\mathcal I_R|\,e^{\mu^{+}\Delta T_1}\big(1+e^{\mu^{+}\Delta T_2}\big)
\big(1+e^{\mu^{+}(\Delta T_1+\Delta T_2)}\big)\norm\eta_{\max}^2$, quadratic in the noise.
\end{corollary}
\begin{proof}
$h_k\le\Delta T_1$, $w_k\le\Delta T_2$, $|D_k|\le n_{\max}$, $g_k\le\rho^{+}_{w_k}\le\rho^{+}_{\Delta T_2}$
and $\rho^{+}_{h_k+w_k}\le\rho^{+}_{\Delta T_1+\Delta T_2}$ by Lemma~\ref{lem:rhoplus}(ii); the sum
has $|\mathcal I_R|$ terms. The value at $\pstar$ is Lemma~\ref{lem:rhoplus}(iv).
\end{proof}
\readthis{\eqref{eq:prop_removal} is the form used in Theorem~\ref{thm:main} and
Corollaries~\ref{cor:basin}--\ref{cor:repartition}, because those need a bound \emph{uniform over
$q$ in a set}. It is also the form in which \emph{every node-specific quantity has been maximised
away}: it contains $|\mathcal I_R|$ but not \emph{which} nodes, and the global $\norm\eta_{\max}$
but not the noise at any particular node. The design rule below is therefore a statement about
\eqref{eq:prop_removal_node}, not about \eqref{eq:prop_removal}; the previous revision read it off
the collapsed bound, where it cannot be read.}
\emph{Design rule (from the nodewise bound).} The marginal contribution of removing $\tau_k$ is
at most $2|D_k|\,e^{\mu^{+}h_k}\,g_k(p)\,\rho^{+}_{h_k+w_k}(p)$: remove nodes that launch few data
($|D_k|$ small), sit close to their retained left neighbour ($w_k$ small, hence $\rho^{+}_{h_k+w_k}$
small), and above all have a small \emph{coarse residual} $g_k(p)$. \readthis{$g_k(p)$ is
\emph{observable} --- one flow evaluation and one subtraction at the current iterate, both of which
the optimiser already performs --- whereas the node noise $\norm{\eta_{\tau_k}}$ appearing in
\eqref{eq:gk} is not. So ``prefer low-noise nodes'' is not an implementable rule, but ``prefer nodes
with small measured $g_k$'' is, and \eqref{eq:gk} is the reason the two point the same way. The
caveat stands: this makes the \emph{bound} small, which is not the same as making the displacement
of the minimiser small; that step needs the strong-convexity premise of Theorem~\ref{thm:main}.}
\emph{What changed.} (i) $L\to\mu$, and $\mu\to\mu^{+}$ wherever an exponent is enlarged to a
window length. (ii) The draft bounded $a_i+b_i$ through $\norm{u_i+v_i-2y_i}$, injecting the size of
the data ($\cynormmax$ on FHN) into the constant; bounding $a_i+b_i$ by two residuals makes the
bound quadratic in $(\norm\eta,\norm{p-\pstar})$, as the left-hand side is, and removes a factor
$\norm y_{\max}/\norm\eta_{\max}\approx\nattractorOverNoise$. (iii) The draft collapsed the sum over
$t_i$ to $|\mathcal I_R|$ without the number of data per window; $n_{\max}$ restores it.
(iv) The draft's intermediate inequality carried $e^{2L(\Delta T_1+\Delta T_2)}$ while its statement
had $e^{L(\Delta T_1+\Delta T_2)}$. (v) $\rho\to\rho^{+}$, without which the steps bounding a shorter
window's residual by a longer window's are false for $\mu<0$. (vi) The statement is now nodewise,
with the collapsed bound demoted to Corollary~\ref{cor:removal_global}.
```

#### (H) Theorem 1 [replaces R2 block (m)]

```latex
\begin{theorem}[a posteriori, conditional displacement of the minimiser under node removal;
\rev{revised}]\label{thm:main}
Assume (A0)--(A2) and work on $E_X$. Let $p^{(K)}\in\arg\min_{P}J_K$ and
$\hat p^{(K)}\in\arg\min_{P}\hatJ_K$ as in \eqref{eq:minimisers}, and suppose $J_K$ is
$m$-strongly convex, $m>0$, on a convex $U\subseteq P$ containing both. Then
\begin{equation}
\norm{\hat p^{(K)}-p^{(K)}}\ \le\
\sqrt{\frac2m\Big(\Delta_K\big(p^{(K)}\big)+\Delta_K\big(\hat p^{(K)}\big)\Big)}
\ \le\ \sqrt{\frac4m\,\sup_{q\in U}\Delta_K(q)},
\label{eq:thm}
\end{equation}
with $\Delta_K$ from \eqref{eq:prop_removal}. Writing $E=\norm\eta_{\max}$, $\delta=\norm{q-\pstar}$,
$A_s=1+e^{\mu^{+}s}$, $B_s=Q(s)$ and $c_0=2n_{\max}|\mathcal I_R|e^{\mu^{+}\Delta T_1}$, each
$\Delta_K(q)$ is the homogeneous \emph{quadratic form} in $(E,\delta)$
\begin{equation}
\Delta_K(q)=c_0\Big[A_{\Delta T_2}A_{\Delta T_1+\Delta T_2}\,E^{2}
+\big(A_{\Delta T_2}B_{\Delta T_1+\Delta T_2}+A_{\Delta T_1+\Delta T_2}B_{\Delta T_2}\big)E\delta
+B_{\Delta T_2}B_{\Delta T_1+\Delta T_2}\,\delta^{2}\Big],
\label{eq:DeltaK_quad}
\end{equation}
whose three coefficients are non-negative and deterministic; consequently
$\sup_{q\in U}\Delta_K(q)$ is $\Delta_K$ evaluated at $\delta=R_U:=\sup_{q\in U}\norm{q-\pstar}$.
\end{theorem}
\begin{proof}
Lemma~\ref{lem:vi} with $F=J_K$, $p=p^{(K)}$, $q=\hat p^{(K)}\in U$ gives
$\tfrac m2\norm{\hat p^{(K)}-p^{(K)}}^2\le J_K(\hat p^{(K)})-J_K(p^{(K)})$. Insert
$\pm\hatJ_K(\hat p^{(K)})$ and $\pm\hatJ_K(p^{(K)})$:
\[
J_K(\hat p^{(K)})-J_K(p^{(K)})=
\big[J_K(\hat p^{(K)})-\hatJ_K(\hat p^{(K)})\big]
+\big[\hatJ_K(\hat p^{(K)})-\hatJ_K(p^{(K)})\big]
+\big[\hatJ_K(p^{(K)})-J_K(p^{(K)})\big].
\]
The middle bracket is $\le0$ because $\hat p^{(K)}$ minimises $\hatJ_K$ over $P$ and
$p^{(K)}\in P$; the outer brackets are bounded by Corollary~\ref{cor:removal_global}. The second
inequality of \eqref{eq:thm} replaces each $\Delta_K$ by its supremum over $U$.
\eqref{eq:DeltaK_quad} is the product of the two affine forms of Lemma~\ref{lem:rhoplus}(iii), and
monotonicity in $\delta$ (non-negative coefficients) gives the value of the supremum.
\end{proof}
\readthis{this is a \emph{conditional a posteriori} estimate, not a proof that guess propagation
stays in the right basin. It assumes what a displacement theorem is supposed to help establish ---
that $\hat p^{(K)}$ already lies in the old strong-convexity neighbourhood --- and in its first form
the right-hand side contains $\hat p^{(K)}$, so the inequality is implicit; the second form is
explicit but needs $U$ to be named, and $R_U$ finite. Corollary~\ref{cor:basin} supplies the
smallness condition under which the premise can be dispensed with, at the price of speaking about
the minimiser of $\hatJ_K$ \emph{restricted to a ball} rather than its global minimiser. The proof
no longer uses $\nabla J_K(p^{(K)})=0$, so the statement covers minimisers on $\partial P$.}
\emph{What changed.} The structure of the proof is the draft's; the bound inherits the improvements
of Proposition~\ref{prop:removal}, so the displacement now scales like the noise (through
$\sqrt{\Delta_K}\sim\norm\eta_{\max}$), not like $\sqrt{\text{noise}\times\text{attractor size}}$.
The draft mixed $L(p)$, $L(\pstar)$ and $L_K=\max\{L(p^{(K)}),L(\hat p^{(K)})\}$ in one formula;
with the constants taken over $X\times P$ the distinction disappears. The previous revision called
$\Delta_K$ ``a quadratic polynomial \dots with coefficients \dots and smaller'', which was wrong on
both counts with the envelope $\bar\rho$ (a maximum of affine forms makes $\Delta_K$ only piecewise
quadratic) and vague on the third coefficient; with $\rho^{+}$ the claim is true and
\eqref{eq:DeltaK_quad} writes all three coefficients out.
```

#### (I) Corollary 2 (basin retention) [replaces R2 block (n)]

```latex
\begin{corollary}[basin retention under node removal]\label{cor:basin}
Assume (A0)--(A2) and work on $E_X$. Let $r>0$ and $\bar B=\{q:\norm{q-p^{(K)}}\le r\}$. Suppose
\begin{enumerate}[nosep,label=(B\arabic*),start=0]
\item $\bar B\subseteq P$;
\item $J_K$ is $m$-strongly convex on $\bar B$ with $m>0$, and $p^{(K)}\in\arg\min_{P}J_K$;
\item $\displaystyle\bar\Delta:=\sup_{q\in\bar B}\Delta_K(q)<\frac{m r^2}{4}$.
\end{enumerate}
Then the minimum of $\hatJ_K$ over $\bar B$ is attained at an interior point $\hat p$ of $\bar B$,
and
\begin{equation}
\norm{\hat p-p^{(K)}}\ \le\ 2\sqrt{\frac{\bar\Delta}{m}}\ <\ r .
\label{eq:basin}
\end{equation}
If moreover $\bar B\subseteq\operatorname{int}P$, then $\hat p$ is a stationary point of $\hatJ_K$,
$\nabla\hatJ_K(\hat p)=0$, and hence a local minimiser of $\hatJ_K$ on $P$.
\end{corollary}
\begin{proof}
$\hatJ_K$ is continuous and $\bar B$ compact, so the minimum over $\bar B$ is attained. Let
$q\in\partial\bar B$, i.e.\ $\norm{q-p^{(K)}}=r$; by (B0) $q\in P$, so Lemma~\ref{lem:vi} with
$U=\bar B$ gives $J_K(q)\ge J_K(p^{(K)})+\tfrac m2r^2$. Corollary~\ref{cor:removal_global} gives
$|\hatJ_K-J_K|\le\bar\Delta$ on $\bar B$ (legitimate by (B0), which puts $\bar B$ inside the region
where the constants of (A0) hold), so
\[
\hatJ_K(q)\ \ge\ J_K(q)-\bar\Delta\ \ge\ J_K(p^{(K)})+\tfrac m2r^2-\bar\Delta
\ \ge\ \hatJ_K(p^{(K)})+\tfrac m2r^2-2\bar\Delta\ >\ \hatJ_K(p^{(K)})
\]
by (B2). Hence every boundary point of $\bar B$ has strictly larger $\hatJ_K$ than the interior
point $p^{(K)}$, so the minimiser $\hat p$ over $\bar B$ is interior to $\bar B$. For the bound,
$\hat p,p^{(K)}\in\bar B$ and, by Lemma~\ref{lem:vi} again,
\[
\tfrac m2\norm{\hat p-p^{(K)}}^2\le J_K(\hat p)-J_K(p^{(K)})
=\underbrace{\big[J_K(\hat p)-\hatJ_K(\hat p)\big]}_{\le\bar\Delta}
+\underbrace{\big[\hatJ_K(\hat p)-\hatJ_K(p^{(K)})\big]}_{\le0}
+\underbrace{\big[\hatJ_K(p^{(K)})-J_K(p^{(K)})\big]}_{\le\bar\Delta}\le2\bar\Delta,
\]
the middle bracket being $\le0$ because $\hat p$ minimises $\hatJ_K$ over $\bar B\ni p^{(K)}$.
Finally, if $\bar B\subseteq\operatorname{int}P$ then $\hat p$ is interior to $P$ as well, so the
unconstrained first-order condition applies and $\nabla\hatJ_K(\hat p)=0$.
\end{proof}
\readthis{what this does and does not say, and what the hypotheses cost.
\emph{(B0) is not decorative}: without $\bar B\subseteq P$ the quantity $\sup_{\bar B}\Delta_K$ is a
supremum of a bound whose constants are suprema over $X\times P$ and therefore need not apply on
part of $\bar B$. \emph{Interiority to $\bar B$ is not interiority to $P$}: the stationarity
sentence is the one place in \S\ref{sec:removal} where $\operatorname{int}P$ is genuinely needed,
and it is assumed there rather than everywhere. The corollary does \emph{not} bound the
\emph{global} minimiser of $\hatJ_K$: the coarse landscape acquires minima elsewhere as the windows
grow (Figures~\ref{fig:gp}, \ref{fig:land1d}), and no perturbation argument can exclude a lower
value far away. It says the coarse problem \emph{restricted to the ball} has its minimiser strictly
inside, close to $p^{(K)}$ --- which is the object guess propagation chases, since the next stage
starts \emph{at} $p^{(K)}$. A continuous descent path started at $p^{(K)}$ cannot leave $\bar B$,
because $\hatJ_K>\hatJ_K(p^{(K)})$ on the whole boundary shell; Nelder--Mead's iterates are not a
continuous path and can in principle jump the shell, so for the runs of \S\ref{sec:num} this is a
statement about the restricted problem, not a guarantee about the optimiser used. Condition (B2) is
a genuine smallness requirement: $\bar\Delta$ grows with $|\mathcal I_R|$, $n_{\max}$, $\Delta T_1$
and $\Delta T_2$, so it is the formal version of ``remove few nodes at a time''. By
\eqref{eq:DeltaK_quad}, $\bar\Delta=\Delta_K$ evaluated at $\delta=\norm{p^{(K)}-\pstar}+r$.}
```

#### (J) Corollary 3 (re-partition) and its example [replaces R2 block (o)]

```latex
\begin{corollary}[re-partition with simultaneous node additions and removals]\label{cor:repartition}
Assume (A0)--(A2) and work on $E_X$. Let $A$ and $B$ be two node sets, both containing $t_0$ and
$t_{N-1}$, and let $C=A\cup B$ be their common refinement. Write $J_A,J_B,J_C$ for the corresponding
costs \eqref{eq:JK}, let $\Delta T_1^{C}$ be the longest window of $C$, $n^{C}_{\max}$ the largest
number of data in a window of $C$, and for $S\in\{A,B\}$ let $\Delta T^{S}$ be the longest window of
$S$. Put
\begin{equation}
\Delta^{C\to S}(p)=2\,n^{C}_{\max}\,\big|C\setminus S\big|\;
e^{\mu^{+}\Delta T_1^{C}}\;\rho^{+}_{\Delta T^{S}}(p)\;\rho^{+}_{\Delta T_1^{C}+\Delta T^{S}}(p),
\qquad
\Delta_{A,B}=\Delta^{C\to A}+\Delta^{C\to B}.
\label{eq:repart}
\end{equation}
Then $\big|J_A(p)-J_B(p)\big|\le\Delta_{A,B}(p)$ for every $p\in P$. If moreover $J_A$ is
$m$-strongly convex on a convex $U\subseteq P$ containing $p_A\in\arg\min_{P}J_A$ and
$p_B\in\arg\min_{P}J_B$, then
\begin{equation}
\norm{p_B-p_A}\ \le\ \sqrt{\frac2m\Big(\Delta_{A,B}(p_A)+\Delta_{A,B}(p_B)\Big)} ,
\label{eq:repart_disp}
\end{equation}
and if instead the hypotheses of Corollary~\ref{cor:basin} hold with $J_A$, $J_B$ and
$\Delta_{A,B}$ in place of $J_K$, $\hatJ_K$ and $\Delta_K$, the minimiser of $J_B$ over
$\bar B(p_A,r)$ is interior to that ball and satisfies $\norm{\cdot-p_A}\le2\sqrt{\bar\Delta/m}<r$.
\end{corollary}
\begin{proof}
$A\subseteq C$ and $B\subseteq C$, so $A$ is obtained from the fine partition $C$ by removing the
nodes $C\setminus A$, and likewise for $B$; neither removed set contains the endpoints.
Corollary~\ref{cor:removal_global} applied twice with fine partition $C$ gives
$|J_A-J_C|\le\Delta^{C\to A}$ and $|J_C-J_B|\le\Delta^{C\to B}$, where $\Delta T_2$ for the removal
$C\to S$ is at most $\Delta T^{S}$, because the nearest retained node to the left of any removed
node lies in $S$ and successive $S$-nodes are at most $\Delta T^{S}$ apart. The triangle inequality
gives the first claim. For \eqref{eq:repart_disp}, run the proof of Theorem~\ref{thm:main} with
$(J_K,\hatJ_K,\Delta_K)$ replaced by $(J_A,J_B,\Delta_{A,B})$: the only properties used are the
pointwise two-sided bound, $J_B(p_B)\le J_B(p_A)$, and Lemma~\ref{lem:vi}. The last claim is
Corollary~\ref{cor:basin} verbatim with the same substitution.
\end{proof}
\begin{remark}[the re-partition bound on this algorithm]\label{rem:repart_numbers}
For the FULL schedule's stage $\kappa=2\to\kappa=3$ (a stage that is \emph{not} a node removal, see
\S\ref{sec:defs}), with $N=101$ and the common terminal node $t_{100}$ belonging to every partition:
\[
A=\{0,2,4,\dots,98\}\cup\{100\}\ (51\ \text{nodes}),\qquad
B=\{0,3,6,\dots,99\}\cup\{100\}\ (35\ \text{nodes}),
\]
\[
A\cap B=\{0,6,\dots,96\}\cup\{100\}\ (18\ \text{nodes}),\qquad
C=A\cup B\ (68\ \text{nodes}),
\]
so $|C\setminus A|=17$, $|C\setminus B|=33$. The gaps of $C$ are $1$ or $2$ data intervals (the
pattern $0,2,3,4,6$ repeats with period $6$; the tail is $96,98,99,100$), giving
$\Delta T_1^{C}=2\Delta t$ and $n^{C}_{\max}=2$; the gaps of $A$ are all $2\Delta t$ (including
$98\to100$) and the largest gap of $B$ is $3\Delta t$, so $\Delta T^{A}=2\Delta t$ and
$\Delta T^{B}=3\Delta t$. The bound is therefore driven by $|C\setminus A|+|C\setminus B|=50$
removed-node terms --- more than for a single clean removal. Corollary~\ref{cor:repartition}
restores a theorem for the algorithm as run; it does not make its stages cheap.
\readthis{every count above \emph{includes the terminal node} $t_{100}$, which is a node of every
partition. A previous version of this remark reported $50$, $34$, $17$ and $67$, which are the
\emph{launch}-node counts; mixing the two conventions in one example is what the indexing paragraph
of \S\ref{sec:defs} exists to prevent. The removed-node counts $17$ and $33$ are the same under
either convention, since $|C\setminus S|=|C|-|S|$ and both counts shift by the same endpoint.}
\end{remark}
```

#### (K) Corollary 1, item (iii) removed, with the envelope remark [replaces R2 block (p)'s corollary and the first part of its `\readthis`; the FHN heuristic paragraph and Remark `rem:weighted` are unchanged]

```latex
\begin{corollary}[per-window and per-subinterval exponents]\label{cor:window}
Assume (A0)--(A2) and work on $E_X$. Let
$\mu(t)=\lambda_{\max}\big(\tfrac12(A(t)+A(t)^\top)\big)$ for the mean-value matrix $A(t)$ of
\eqref{eq:proof1} associated with \emph{the specific pair of trajectories and the specific parameter
under consideration}. Then
\begin{enumerate}[nosep,label=(\roman*)]
\item (state sensitivity, local form) for $0\le a\le b$,
\begin{equation}
\norm{\delta(b)}\le\exp\Big(\int_a^b\mu(r)\dd r\Big)\norm{\delta(a)},
\qquad \delta=\flow{\cdot}{p}{x_1}-\flow{\cdot}{p}{x_2};
\label{eq:window}
\end{equation}
\item (parameter sensitivity, local form) with $\delta=\flow{\cdot}{p_1}{x_0}-\flow{\cdot}{p_2}{x_0}$
and $\delta(0)=0$,
\begin{equation}
\norm{\delta(t)}\ \le\ \tilde L\norm{p_1-p_2}\int_0^t\exp\Big(\int_s^t\mu(r)\dd r\Big)\dd s .
\label{eq:window_param}
\end{equation}
\end{enumerate}
Both are \emph{realisation-wise and pair-specific}: $\mu(\cdot)$ is built from the two trajectories
and the parameter named in the statement, and from no others.
\end{corollary}
\begin{proof}
(i) is \eqref{eq:proof1} integrated between $a$ and $b$ instead of $0$ and $t$. (ii) is the
Dini-derivative inequality $D^{+}\norm\delta\le\mu(t)\norm\delta+\tilde L\norm{p_1-p_2}$ of
Lemma~\ref{lem:param} integrated by variation of constants.
\end{proof}

\begin{remark}[what would be needed to substitute these into the propositions]\label{rem:envelopes}
Corollary~\ref{cor:window} does \emph{not} license replacing $e^{\mu^{+}s}$ by $e^{\Lambda_k}$, or by
$M_k$ of \eqref{eq:Mk}, anywhere in \S\ref{sec:lemmas}--\S\ref{sec:removal}. What would license a
substitution is a pair of \emph{deterministic envelopes}: for $h>0$,
\begin{equation}
\mathcal M(h)=\sup\Big\{\exp\Big(\int_a^b\mu_{\rm pair}(r)\dd r\Big)\Big\},\qquad
\mathcal Q(h)=\sup\Big\{\tilde L\int_a^b\exp\Big(\int_s^b\mu_{\rm pair}(r)\dd r\Big)\dd s\Big\},
\label{eq:envelopes}
\end{equation}
both suprema taken over \emph{all} admissible trajectory pairs --- launched anywhere within $r'$ of
the orbit, at any $p\in P$ --- and over \emph{all} sub-intervals $[a,b]\subseteq[t_0,t_{N-1}]$ with
$b-a\le h$. With these, Lemma~\ref{lem:residual}, Propositions~\ref{prop:cost} and
\ref{prop:removal}, Theorem~\ref{thm:main} and Corollaries~\ref{cor:basin}--\ref{cor:repartition}
hold with $e^{\mu^{+}s}\rightsquigarrow\mathcal M(s)$ and $Q(s)\rightsquigarrow\mathcal Q(s)$, and
nothing is lost, since $\mathcal M(h)\le e^{\mu^{+}h}$ and $\mathcal Q(h)\le Q(h)$.
\readthis{four reasons the \emph{per-window} quantities cannot play this role, each sufficient on
its own. \emph{(1) Pair dependence.} $\mu(\cdot)$ is attached to one pair and one parameter, while
the proofs use at least five: noisy versus clean launch, $p$ versus $\pstar$, coarse versus fine
prediction, different $q\in U$, and windows of two partitions and their common refinement.
\emph{(2) Interval spanning.} A coarse interval crosses several fine windows and
$\exp(\int_a^b\mu)$ is then a \emph{product} of per-window factors, which $\max_kM_k$ does not
bound. \emph{(3) The forcing integral.} $\int_0^t\exp(\int_s^t\mu)\dd s\le M_kt$ requires $[s,t]$ to
lie inside window $k$ for every $s\le t$, which fails as soon as $t$ is in a later window than $0$.
\emph{(4) Randomness.} An a posteriori $M_k$ computed along a realised noisy pair is a random
variable and cannot be factored out of $\mathbb E\norm\eta^2$.
And what \S\ref{sec:num} reports is neither $\mathcal M$ nor $M_k$: Table~\ref{tab:windows} gives
\emph{linearised endpoint} exponents $e^{\Lambda_k}$ computed from $J_f(x^\star(t);\pstar)$ along the
\emph{true} orbit at the \emph{true} parameter with windows aligned to $t_0$ --- the
infinitesimal-perturbation limit of one pair. These are not suprema over a tube of trajectories, not
valid at another $p$, and not sub-interval bounds ($M_k\ge\max\{1,e^{\Lambda_k}\}$, strictly
wherever $\mu(\cdot)$ changes sign inside the window, which on FHN happens twice per period,
Figure~\ref{fig:lemmas}c). $\mathcal M(h)$ is not computed anywhere in this report. The
node-placement rule below is therefore a \emph{heuristic} read off a measurement, and a previous
version of this corollary carried it as a licensed substitution (item (iii)), which is withdrawn.}
\end{remark}
```

#### (L) §2.3 — the nesting correction [replaces R2 block (e)]

```latex
\readthis{the sweep schedules are in general \emph{not} nested, so the stages of
Algorithm~\ref{alg:gp} are \emph{not} node removals. The node set at window size $\kappa$ is
$S_\kappa=L_\kappa\cup\{M\}$ with $L_\kappa=\{j\kappa:\ j\kappa<M\}$ the launch nodes and
$M=N-1=100$ the common terminal node, which belongs to every partition. Hence the stage
$\kappa_a\to\kappa_b$ is a node removal iff $S_{\kappa_b}\subseteq S_{\kappa_a}$, i.e.\ iff
$L_{\kappa_b}\subseteq L_{\kappa_a}$, and for increasing $\kappa_a<\kappa_b\le M$ this holds
\begin{equation}
\text{iff}\qquad \kappa_a\mid\kappa_b\quad\text{or}\quad\kappa_b=M
\label{eq:nesting}
\end{equation}
(for $\kappa_b<M$ the element $\kappa_b\in L_{\kappa_b}$ forces $\kappa_a\mid\kappa_b$, and
conversely divisibility gives inclusion; for $\kappa_b=M$, $L_M=\{0\}$ and inclusion is automatic).
Counting transitions with \eqref{eq:nesting}: FULL
$\{1,2,3,4,5,6,8,10,12,15,20,25,33,50,75,100\}$ has \emph{two} of $15$ nested ($1\to2$ and
$75\to100$); DENSE ($44$ stages) \emph{two} of $43$ ($1\to2$ and $95\to100$); SHORT
$\{1,2,5,10,25,50,100\}$ four of six ($1\to2$, $5\to10$, $25\to50$, $50\to100$; the failures are
$2\to5$ and $10\to25$); COARSE $\{1,5,25,100\}$ three of three and JUMP $\{1,100\}$ one of one.
\emph{Read this:} the second nested transition of FULL and of DENSE is in each case the terminal
$\kappa_b=M$ stage, i.e.\ the collapse to single shooting, which is a node removal only in the
degenerate sense that \emph{every} interior node is removed: there $|\mathcal I_R|=K-1$,
$\Delta T_2$ reaches $100$, and $\Delta_K$ of \eqref{eq:prop_removal} carries $e^{100\mu^{+}}$. So
Proposition~\ref{prop:removal} covers two of FULL's fifteen stages, one of which it covers
vacuously. And, awkwardly, the fully nested schedules are the ones that perform \emph{worst}
(Table~\ref{tab:T2}: DENSE $\nschedDense$, COARSE $\nschedCoarse$, JUMP $\nschedJump$). The general
case is covered by Corollary~\ref{cor:repartition}, which compares two arbitrary partitions through
their common refinement. The earlier version of this manuscript asserted that the sub-partition
hypothesis ``is what the algorithm does'', which was false and is withdrawn; the version before
this one stated the criterion as ``$\kappa_a$ divides $\kappa_b$'' and counted one nested transition
in FULL and one in DENSE, which omitted the $\kappa_b=M$ case of \eqref{eq:nesting}.}
```

#### (M) Abstract and the two blow-up sentences [replaces the "one-line modification" clause of the abstract, and the corresponding sentences in R2 blocks (b) and (d)]

```latex
% --- abstract: replaces the "one-line modification" clause ---
Each change is local to one proof step, and the assumptions under which each statement holds are
now stated in full: a fixed admissible region, a closure condition that confines every admissible
trajectory to a tube around the true orbit, a noise model with the event on which the deterministic
bounds hold, and a constrained optimality condition in place of a vanishing gradient.

% --- \S2.1 (A0) readthis: replaces "The analysis is a statement about $P$; the numerical
%     plateau is the boundary of $P$ made visible." ---
The analysis is a statement about $X\times P$. The implication runs one way only: where the
implemented loss returns its flat blow-up value, no solution of the window stays in $X$, so that
candidate is outside $P$. The converse is \emph{not} claimed --- the threshold $10^3$ in
\texttt{ms\_loss} is an implementation constant chosen to keep the optimiser numerically alive, the
level set $\{\text{loss}=10^3\}$ has not been related to $\partial P$, and a candidate can fail the
confinement condition \eqref{eq:closure} while producing states far below $10^3$. The plateau is
evidence that some candidates lie outside $P$, not a picture of $\partial P$.

% --- Remark~\ref{rem:impl}(iv): same correction, milder form ---
\item The blow-up replacement is outside the analysis entirely: where it fires, no solution of the
window remains in $X$ and the candidate $p$ is therefore outside the set $P$ of (A0), so there is no
trajectory to bound. This is a sufficient condition for $p\notin P$, not a characterisation of $P$
or of its boundary.
```

#### (N) Remark `rem:impl`(ii) — the penalty-bias bound, made conditional [replaces item (ii) of R2 block (d)'s Remark]

```latex
\item Proposition~\ref{prop:perr} does \emph{not} transfer. Its comparator $\pstar$ minimises
$J_K^\star$ but is not a minimiser of $J_K/N+\text{penalty}$. Write $F=J_K/N$ and
$G(p)=\frac{\gamma}{n_p}\sum_j\operatorname{smooth}\ell_1(p_j)$, and \emph{assume}:
\begin{enumerate}[nosep,label=(C\arabic*)]
\item $p^{(K)}\in\arg\min_{P}F$ and $p^{\rm pen}\in\arg\min_{P}(F+G)$ (both exist, $P$ compact);
\item there is a convex $U\subseteq P$ containing $p^{(K)}$, $p^{\rm pen}$ \emph{and the segment
joining them}, on which $F$ is $m$-strongly convex.
\end{enumerate}
Then
\begin{equation}
\norm{p^{\rm pen}-p^{(K)}}\ \le\ \frac{1}{m}\,\big\lVert\nabla G(p^{\rm pen})\big\rVert
\ \le\ \frac{\gamma}{m\sqrt{n_p}} .
\label{eq:penbias}
\end{equation}
\emph{Proof.} Adding the two first-order optimality inequalities
$\langle\nabla F(p^{(K)}),\,p^{\rm pen}-p^{(K)}\rangle\ge0$ and
$\langle\nabla F(p^{\rm pen})+\nabla G(p^{\rm pen}),\,p^{(K)}-p^{\rm pen}\rangle\ge0$ gives
$\langle\nabla F(p^{\rm pen})-\nabla F(p^{(K)}),\,p^{\rm pen}-p^{(K)}\rangle
\le\norm{\nabla G(p^{\rm pen})}\,\norm{p^{\rm pen}-p^{(K)}}$; strong monotonicity of $\nabla F$ on
$U$ (equivalent to $m$-strong convexity there, and requiring the segment to lie in $U$, whence
(C2)) bounds the left side below by $m\norm{p^{\rm pen}-p^{(K)}}^2$. For the gradient,
$\frac{\dd}{\dd x}\operatorname{smooth}\ell_1(x)=2\sigma(\alpha x)-1=\tanh(\alpha x/2)\in(-1,1)$, so
$\norm{\nabla G}_2\le(\gamma/n_p)\sqrt{n_p}=\gamma/\sqrt{n_p}$, uniformly in $p$ and independently
of $\alpha$. $\square$
\readthis{(C1)--(C2) are \emph{assumptions}, not facts about the FHN runs, and \eqref{eq:penbias} is
therefore conditional. ``$F$ is strongly convex near its own minimiser'' --- which is what the
previous revision assumed --- is \emph{not} enough: the penalised minimiser may lie outside that
neighbourhood, which is the same circularity flagged for Theorem~\ref{thm:main}. Note also that the
smoothing sharpness $\alpha=500$ does not appear in \eqref{eq:penbias} at all. This shrinkage bias
must be added to any statement about $\norm{\pstar-\cdot}$; it is the shrinkage visible in
Figure~\ref{fig:coef}.}
```

#### (O) §6 — one new gate bullet [added to the gate list of R2 block (r); the G7 and G8 bullets there are unchanged]

```latex
\item G9 (\emph{open, failing}): the closure condition \eqref{eq:closure} of (A1) has \emph{not}
been checked for any $\kappa$. It requires a tube radius $r_X$, the constants $\mu,\tilde L$ taken
over the resulting $X\times P$, a diameter for $P$, and a noise radius $r'$ with
$e^{\mu^{+}\Delta T}r'+Q(\Delta T)\operatorname{diam}(P)<r_X$; none of these has been instantiated
numerically, and with $\mu=\nmu>0$ the condition forces $r'$ to be exponentially small in the window
length, so it is expected to be unsatisfiable at $\kappa=100$ for any useful $r_X$. Every
deterministic statement of \S\ref{sec:lemmas}--\S\ref{sec:removal} is conditional on an event whose
probability this report does not bound numerically. Reported as a failure, not as a caveat.
```

---

## Section 3 — back to you

```
Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
```
