# R4 — Claude's counterreply to GPT's round-3 critique of R004

Artifact under review: `reports/2026-09-11-R004-revised-manuscript/report.tex`, in the state left by
**R2_to_gpt.md Section 2 as amended by R3_to_gpt.md Section 2**. Those two documents, not the file
on disk, are the current artifact; the `.tex` has still not been edited. This round amends them
again; **everything in R3_to_gpt.md's Section 2 that is not listed in Section 2 below stands
exactly as written there**, and everything in R2's Section 2 not superseded by R3 or by this
document likewise stands.

**Tally: 11 Accept, 0 Defend. Seven of the eleven carry an additional Clarify sub-block.**

All eleven are right, and issues 1 and 2 are right in the strongest sense: they identify a
*dependency* defect, not a constant defect. R3 fixed the semantics of (A0) and then left Lemmas 1
and 2 quantified over all of `X` and all `t ≤ ΔT_max`, which is exactly the statement that (A0) no
longer supports, and then used those statements inside the proof of the lemma that is supposed to
establish confinement. That is circular and I did not see it. The restructuring below makes the
order of logic explicit: **local lemmas → exit-time bootstrap → full-window corollary → everything
else**.

Where I depart from your prescribed remedy — never from your diagnosis — I say so under
**Clarify**. There are seven such departures; five of them make the statement strictly stronger
than what you asked for, one is a self-raised defect that your issue 5 uncovered but did not name,
and one is a scope correction to a retraction you asked for:

- **Issue 1 (self-raised addition).** Working through your restructuring exposed a gap you did not
  name: `ΔT_max` was fixed as "at least the longest window of any partition under discussion", but
  Proposition 3 integrates a *coarse* window of length `h_k + w_k`, which exceeds every fine window.
  Lemma 0 confines a trajectory for `ΔT_max` at most, so Proposition 3's own flow evaluations were
  outside the confinement horizon. (A0) is amended: `ΔT_max ≥ max_k (h_k + w_k)` over every removal
  considered.
- **Issue 3.** I retract the "costs nothing" sentence as you ask, and I retract it further than you
  ask: it is not merely that Figure 10 evaluates off-truth, it is that the *whole* comparison
  `ρ̄` vs `ρ⁺` is numerically invisible on FHN, because `μ = \nmu > 0` there and for `μ ≥ 0` the
  three envelopes `ρ`, `ρ̄`, `ρ⁺` are **identically equal**. The sentence was not just unsupported;
  it was arguing about a distinction that Section 6 cannot see. The distinction bites exactly where
  the revision advertises value — `μ < 0` — and there you are right that `ρ⁺` is strictly looser.
- **Issue 4.** I go one notch tighter than the bound you display. You write `ρ^env` at the two
  offsets; but Lemma 3 applied *at the actual offset* returns `ρ_σ` itself, not an envelope of it —
  no monotonicity and no envelope is needed for the per-datum terms at all. The primary bound
  therefore carries `ρ_{t_i−τ_k^-} + ρ_{t_i−τ_k}` with the bare `ρ`, and `ρ̄` appears only in the
  collapsed corollaries, where elapsed times really are enlarged. Your correction on `e^{μ(t_i−τ_k)}`
  with the *actual* `μ` is exactly right and is the reason no monotonicity is needed there either.
- **Issue 8.** I give your four numbers and add a fifth, defined separately as you suggest: the
  *actual coarse span* `Λ_R = max_{k∈I_R}(h_k + w_k)`, which is `100` for both terminal transitions
  while the collapsed argument is `150` (FULL) and `190` (DENSE). The gap between `100` and `150` is
  the price of the collapsed corollary, and the nodewise form of issue 4 does not pay it.
- **Issue 9.** Accepted, plus an explicit cost accounting: the greedy rule needs `O(|I_R|)` window
  integrations per step in the worst case (not one), because removing a node changes `τ_k^-`, `w_k`
  and `g_k` for every already-removed node in the same consecutive run.
- **Issue 10.** Accepted; the renumbering you offer as the alternative ("rename it G11") is the one
  I decline, because it would put a failing premise check inside a list whose members are all
  mechanical inheritance checks. It becomes **P1**, in a separate list, and the abstract says both
  numbers.
- **Issue 11.** Accepted, with one addition that makes the reading worse, not better: the obvious
  escape — enlarge `r_X` — does not work, because `μ` and `L̃` are suprema over `X ⊇ 𝒯(r_X)` and
  grow with `r_X` for a polynomial right-hand side, while `Q(ΔT_max)` grows exponentially in `μ`.
  So `Q(ΔT_max)·diam(P) < r_X` is a genuine competition that a bigger tube loses.

Arithmetic I quote is recomputed, not recalled; the evaluations of `Q` are listed in §1.11.

---

## Section 1 — issue-by-issue response

### 1. The new (A0) invalidates the unchanged statements of Lemmas 1 and 2 — **ACCEPT**, with a **CLARIFY** raising one further gap of the same kind

**Accept, in full, and this is the blocking one.** R3 removed forward invariance from (A0) and
replaced it by a *derived* confinement (Lemma 0), which was the right move; and then left R2's
Lemmas 1 and 2 quantified "for all `t ∈ [0,ΔT_max]`, `p ∈ P` and `x_1,x_2 ∈ X`". Under the new
(A0) those statements are **false**: nothing stops `φ(·;p,x_1)` from leaving `X` at some `t <
ΔT_max`, after which the mean-value matrix `A(t) = ∫_0^1 J_f(x̂_2 + sδ;p) ds` integrates the
Jacobian over a chord that is *not* in `X`, so `λ_max(sym A) ≤ μ` has no justification — and for a
cubic `f` the solution may not exist at all. Lemma 0 then invoked those false statements inside its
own bootstrap. The circularity is exactly as you describe: the confinement proof consumes the
global lemmas, which are true only given confinement.

**The restructuring I adopt is yours, literally.** Three levels, in this order:

1. **Definition (confinement time).** For `x_1,x_2 ∈ X`, `p_1,p_2 ∈ P`,
   `T_X((x_1,p_1),(x_2,p_2)) = sup{ t ≤ ΔT_max : both solutions exist on [0,t] and the chord
   [φ(s;p_1,x_1), φ(s;p_2,x_2)] ⊂ X for every s ≤ t }`. The chord contains its endpoints, so one
   condition covers both "trajectories in `X`" and "joining segment in `X`". `T_X ≥ 0` always, since
   `X` is convex and `x_1,x_2 ∈ X`.
2. **Lemma 1' and Lemma 2' are local**: their conclusions are asserted only for `t ≤ T_X` of the
   pair named in the statement. Their proofs are the R2 proofs restricted to `[0,T_X)` and extended
   to `t = T_X` by continuity; nothing else changes, because the R2 proofs used containment only
   through the mean-value step, which is now hypothesised at exactly the points where it is used.
3. **Lemma 0 uses only Lemma 1' and Lemma 2'**, and returns `T_X = min{ΔT_max, t_{N−1}−τ}` for the
   pairs that matter. The **full-window corollary** then restates Lemmas 1' and 2' without the
   proviso for clean launches and for `r'`-perturbed launches, and that corollary — not the
   lemmas — is what §3–§5 cite.

I have written all four out in §2, including Lemma 0's proof with the strict inequality made
load-bearing (`Q(ΔT_max)diam(P) < r_X` strictly, and `e^{μ⁺ΔT_max}r' + Q(ΔT_max)diam(P) < r_X`
strictly, so that the distance at a putative exit time is `< r_X` while `∂𝒯` requires `= r_X`).
Two details the proof now states rather than assumes: the solution extends *to* `T` because it
remains in the compact `𝒯 ⊆ X` on `[0,T)` and (A0) grants existence while in `X`; and the chord
condition at each step is discharged by **convexity of `X`** given that both endpoints lie in `𝒯`
— which is why `X` is convex and `𝒯` (which is not convex) is not asked to be.

**Clarify — the restructuring exposed a horizon gap of the same species, which I raise against
myself.** (A0) fixed `ΔT_max` "at least as large as the longest window of any partition under
discussion", and Lemma 0 confines a launched trajectory for at most `ΔT_max`. But Proposition 3's
coarse predictor is integrated from `τ_k^-` to `t_i`, an elapsed time `t_i − τ_k^- ≤ h_k + w_k`,
which is longer than any *fine* window and is not a window of the coarse partition either when a
run of consecutive nodes is removed. With `ΔT_max` set to the longest fine window, Proposition 3's
own flow evaluations sit outside the interval on which Lemma 0 confines anything, and Lemma 3 is
applied at an elapsed time exceeding its own hypothesis `s ≤ ΔT_max`. Amendment, in §2 block (A):

> `ΔT_max` is at least `max{ longest window of any partition under discussion, max_{k∈I_R}(h_k+w_k)
> over every node removal under discussion }`.

This is not free: it enlarges `ΔT_max` in the closure condition (A1), i.e. it makes (A1) *harder*,
and for the terminal transitions of issue 8 it pushes `ΔT_max` to `100` even though the fine
windows are `75` and `95`. I would rather report that than leave the gap.

### 2. The manuscript must distinguish the two — in fact four — scopes everywhere — **ACCEPT**

**Accept, and I adopt your four-way separation exactly as listed.** R3's scope declaration said
"every inequality in §3–§5 holds only on `E_X`", which over-claims in one direction and
under-claims in the other. It is wrong to stamp `E_X` on Lemma 1', which contains no noise at all:
its hypothesis is *trajectory containment*, a deterministic condition, and saying "on `E_X`"
substitutes a probabilistic event for a geometric one and thereby hides precisely the dependency
that issue 1 is about. And it is wrong to leave `E_X` implicit on a statement about `Ĵ_K` or
`J_K(p*)`, where the noise is the object.

The four scopes, tagged **(S1)–(S4)** in the manuscript, are:

- **(S1) Local deterministic flow estimates.** Hypothesis: the named trajectories and their joining
  chord remain in `X` on the interval in question (`t ≤ T_X`). No noise, no `E_X`, no (A1).
  Members: Lemma 1', Lemma 2', Corollary 1(i)–(ii).
- **(S2) Confinement under (A1).** Hypothesis: (A0)–(A1) and a launch within `r'` of the true orbit.
  Conclusion: `T_X` is the whole window, so (S1) applies on the full window. Still deterministic;
  the randomness enters only through *which* launches are within `r'`. Members: Lemma 0 and its
  corollary.
- **(S3) Noisy-cost statements on `E_X`.** Hypothesis: (A0)–(A2), pathwise on the event
  `E_X = {max_i ‖η_i‖ ≤ r'}`, which is what makes every shooting node an admissible launch for
  (S2). Deterministic inequalities, realisation by realisation. Members: Lemma 3, Proposition 1(a),
  Proposition 2's pathwise bound, Proposition 3 and its corollaries, Theorem 1, Corollaries 2–3.
- **(S4) Expectations.** Unconditional under the bounded model (A2-a), where `P(E_X) = 1`;
  conditional on `E_X` under the Gaussian model (A2-b), where Lemma `lem:trunc` supplies the same
  constants and `(eq:tail)` bounds `P(E_X^c)`. No unconditional expectation is claimed under
  (A2-b). Members: Proposition 1(b), Proposition 2's expectation form.

A "Scope of statements" paragraph carrying this list, plus a one-line table saying which result is
in which scope, is written out in §2 block (B). Every theorem environment in §3–§5 now opens with
its scope tag, so the reader never has to infer it.

### 3. Replacing `ρ̄` by `ρ⁺` violates the tightness requirement — **ACCEPT**, with a **CLARIFY** that widens the retraction

**Accept, and the reasoning is exactly right.** For `μ < 0`, `ρ̄_s = max{2‖η‖_max, ρ_s}` switches to
its constant branch precisely when `L̃‖p−p*‖ ≤ |μ|‖η‖_max` — I re-derived your switching condition
and it is the same one: with `Q(s) = (L̃/|μ|)(1 − e^{−|μ|s})` and `e^{μs} − 1 = −(1 − e^{−|μ|s})`,
`ρ_s − 2‖η‖_max = (1 − e^{−|μ|s})(L̃‖p−p*‖/|μ| − ‖η‖_max)`, so `ρ̄_s = 2‖η‖_max` iff
`L̃‖p−p*‖ ≤ |μ|‖η‖_max`. On that region `ρ⁺_s = 2‖η‖_max + Q(s)‖p−p*‖` is strictly larger for every
`p ≠ p*`, and the excess grows to `L̃‖p−p*‖/|μ|`. Trading that away to keep the word "quadratic" is
trading a bound for an adjective, and the revision's own success criterion forbids it.

**Resolution.** `ρ̄_s = max{2‖η‖_max, ρ_s}` is restored as the primary envelope everywhere
(Lemma 3, Proposition 3's corollaries, Theorem 1, Corollaries 2–3). `Δ_K` is described as
**piecewise quadratic**, with the switch located explicitly. Two properties that R3 obtained from
`ρ⁺` and that Theorem 1 and Corollary 2 actually use are *retained*, because `ρ̄` has them too and I
prove it rather than asserting it:

- `s ↦ ρ̄_s` is non-decreasing — by construction, since `ρ̄_s = sup_{0≤r≤s} ρ_r` (the sign of
  `∂_r ρ_r = e^{μr}(μ‖η‖_max + L̃‖p−p*‖)` is independent of `r`, so the sup over `[0,s]` is
  `max{ρ_0, ρ_s}` and `ρ_0 = 2‖η‖_max`). So every step that enlarges an elapsed time is still legal.
- `δ ↦ ρ̄_s` is non-decreasing (a max of two functions each non-decreasing in `δ`), so
  `sup_{q∈U} Δ_K(q)` is still *computable*: `Δ_K` evaluated at `δ = R_U := sup_{q∈U}‖q−p*‖`. This
  is the property Theorem 1's second inequality and Corollary 2's (B2) need, and it survives
  intact.

`ρ⁺` is demoted to a remark, stated as what it is: a smooth quadratic majorant `Δ_K ≤ Δ_K^+`, for
readers who want a closed-form quadratic (e.g. to solve `(B2)` for `r`), with the explicit warning
that it is strictly loose for `μ < 0` away from `p*`.

**Clarify — the retraction is wider than Figure 10.** You are right that
"costs nothing where every number in §6 is evaluated" is false because Figure 10 also evaluates at
a vector `0.06` from `p*` on six coefficients and at `p_wrong`. But the sentence is worse than
false; it is *non-cognitive on this data*. For `μ ≥ 0`,

`ρ_s − ρ_0 = (e^{μs} − 1)‖η‖_max + Q(s)‖p−p*‖ ≥ 0`, so `ρ̄_s = ρ_s`, and `μ⁺ = μ`, so `ρ⁺_s = ρ_s`:

the three envelopes are **identically equal**. The FHN global constant is `μ = \nmu > 0`. So every
number in §6 — at `p*`, at the off-truth perturbation, at `p_wrong` alike — is *insensitive to the
choice*, and the sentence was defending a choice using evidence that cannot distinguish the
alternatives. Where the choice does bite is `μ < 0`: the per-window and weighted-norm regime the
revision advertises (Table 2: `\nfracDneg` of windows have `Λ^D_k < 0`), and there `ρ̄` is strictly
tighter. The sentence is deleted and replaced by that statement, which is both true and the actual
reason `ρ̄` must be primary. Net: your diagnosis is accepted and the supporting argument is
replaced by a correct one that points the same way more strongly.

### 4. The nodewise Proposition 3 bound is still unnecessarily loose — **ACCEPT**, with a **CLARIFY** that goes one notch tighter than the form you display

**Accept, both losses.** R3's proof obtained `a_i ≤ ρ^+_{h_k+w_k}` and `b_i ≤ ρ^+_{h_k}` and then
wrote `a_i + b_i ≤ 2ρ^+_{h_k+w_k}`, discarding the second, shorter envelope for a factor of two; and
it replaced `e^{μ(t_i−τ_k)}` by `e^{μ^+h_k}` before summing over `D_k`, discarding the per-datum
offset. Both are exactly the kind of pre-emptive collapsing the revision claims not to do.

**Resolution — the primary statement of Proposition 3 keeps both.** With `h_k = τ_{k+1} − τ_k`,
`w_k = τ_k − τ_k^-` and `g_k(p) = ‖φ(w_k;p,y_{τ_k^-}) − y_{τ_k}‖`:

`|Ĵ_K(p) − J_K(p)| ≤ Σ_{k∈I_R} Σ_{t_i∈D_k} e^{μ(t_i−τ_k)} g_k(p) [ ρ_{t_i−τ_k^-}(p) + ρ_{t_i−τ_k}(p) ]`

and the two collapsed forms become corollaries:

- **Corollary (per node):** `Σ_{k∈I_R} |D_k| e^{μ^+h_k} g_k(p) [ ρ̄_{h_k+w_k}(p) + ρ̄_{h_k}(p) ]`;
- **Corollary (global):** `Δ_K(p) = n_max |I_R| e^{μ^+ΔT_1} ρ̄_{ΔT_2}(p) [ ρ̄_{ΔT_1+ΔT_2}(p) +
  ρ̄_{ΔT_1}(p) ]`, which is *at most* R3's `2 n_max |I_R| e^{μ^+ΔT_1} ρ^+_{ΔT_2} ρ^+_{ΔT_1+ΔT_2}`
  and is strictly smaller whenever `ρ̄_{ΔT_1} < ρ̄_{ΔT_1+ΔT_2}`.

Your remark on `e^{μ(t_i−τ_k)}` with the **actual** `μ` is right and is worth stating as the reason
the primary form needs no monotonicity anywhere: `‖u_i − v_i‖ ≤ e^{μ(t_i−τ_k)} g_k(p)` is Lemma 1'
applied at the *actual* elapsed time, so no exponent is enlarged and the sign of `μ` is irrelevant.
`μ^+` appears only in the per-node corollary, where `t_i − τ_k` is replaced by `h_k` and
monotonicity of `t ↦ e^{μt}` would be needed and fails for `μ < 0`.

**Clarify — one notch tighter than your display.** You write `ρ^env` at the two offsets. But the
per-datum terms do not need an envelope at all: Lemma 3 applied at node `τ_k^-` and datum `t_i`
returns

`a_i ≤ e^{μ(t_i−τ_k^-)}‖η_{τ_k^-}‖ + Q(t_i−τ_k^-)‖p−p*‖ + ‖η_i‖ ≤ ρ_{t_i−τ_k^-}(p)`

with the bare `ρ` at the actual elapsed time, and likewise `b_i ≤ ρ_{t_i−τ_k}(p)`. Envelopes are
needed only where an elapsed time is *enlarged*, i.e. in the two corollaries. Since
`ρ_σ ≤ ρ̄_σ ≤ ρ̄_s` for `σ ≤ s`, the corollaries follow from the primary form without any extra
step, and the primary form is strictly tighter than the `ρ^env` version for `μ < 0` — by the same
gap the whole of issue 3 is about. So the primary statement carries `ρ`, the corollaries carry `ρ̄`,
and `ρ⁺` appears in one remark. Written out in §2 block (G).

### 5. The deterministic-envelope remark does not cover every pair used by Proposition 3 — **ACCEPT**

**Accept.** R3 defined `𝓜(h)` and `𝒬(h)` as suprema over pairs "launched anywhere within `r'` of
the orbit, at any `p ∈ P`". Proposition 3's coarse predictor is
`u_i(s) = φ(s; p, φ(w_k;p,y_{τ_k^-}))`, whose initial state at the comparison time `τ_k` is a
*reachable* state, not a state within `r'` of the orbit: Lemma 0 places it within `r_X` of the
orbit, and `r' ≤ r_X` with (A1) forcing `r'` strictly smaller. So the displayed supremum does not
cover the pair the substitution is applied to, and the same objection hits Corollary 3 (the
re-partition pairs are launched from reachable states at two different partitions). The substitution
was unlicensed in exactly the place it was advertised.

**Resolution.** The supremum is taken over the class the proofs actually use, which is the largest
class on which `μ_pair` is even defined: pairs of solutions that **remain in `X`**, i.e. the
confined tube `𝒯` delivered by Lemma 0. Concretely, for `h > 0`,

- `𝓜(h) = sup{ exp(∫_a^b μ_ξ(r)dr) }` over all pairs `ξ_1,ξ_2` of solutions at a **common**
  `p ∈ P` on a sub-interval `[a,b] ⊆ [t_0,t_{N−1}]` with `b − a ≤ h`, whose joining chord lies in
  `X` throughout — equivalently, by Lemma 0, over all pairs of trajectories confined to `𝒯`;
- `𝒬(h) = sup{ L̃ ∫_a^b exp(∫_s^b μ_ξ(r)dr) ds }` over the same sub-intervals and over pairs at
  **two** parameters `p_1, p_2 ∈ P` (the mean-value matrix being taken at `p_1`), which is the
  configuration Lemma 2' is stated in and replaces R3's ambiguous "at any `p ∈ P`".

With those definitions, every pair used in Lemma 3, Propositions 1 and 3, Theorem 1 and
Corollaries 2–3 belongs to the class, *because Lemma 0 confines all of them to `𝒯 ⊆ X`* — which is
the only reason the substitution is licensed, and the remark now says so instead of asserting the
substitution. `𝓜(h) ≤ e^{μ^+h}` and `𝒬(h) ≤ Q(h)` still hold, since `μ_ξ(t) ≤ μ` pointwise on `X`.
The four reasons Table 2's `e^{Λ_k}` is none of these are unchanged and still stated. §2 block (K).

### 6. The blow-up implication is still false — **ACCEPT**

**Accept.** R3 kept a one-way implication — "where the flat penalty fires, no solution of the window
stays in `X`, hence `p ∉ P`" — and that direction is no better supported than the characterisation
it replaced. Your three gaps are each fatal on their own, and I add nothing to them: (i) the
observation is that a *numerical* Tsit5 trajectory exceeded `10^3` or returned NaN, which is
evidence about the integrator as much as about the exact flow; (ii) `X` has never been related to
`{‖x‖_∞ < 10^3}` — indeed `X` has never been instantiated numerically at all, which is the same
hole as issue 10's P1; (iii) the inference would in any case be conditional on (A1) holding and on
`E_X`, neither of which is verified for the realised experiment.

**Resolution.** The claim is withdrawn entirely, in both places it occurs (the (A0) `\readthis` and
Remark `rem:impl`(iv)), and replaced by the statement that survives with no premises at all:
*plateau evaluations lie outside the exact-flow analysis of this manuscript; the bound proved here
says nothing about them, in either direction.* The manuscript then lists what would be needed to
recover any inference — a verified `E_X`, a verified (A1), a relation `X ⊂ {‖x‖_∞ < 10^3}`, and a
controlled numerical-error bound — and says none is available. The `p ∉ P` sentence is deleted, not
weakened. §2 block (M).

### 7. Corollary 2's discussion of interiority is logically wrong — **ACCEPT**

**Accept, both halves.** (i) With `r > 0`, (B0) `B̄ ⊆ P` already gives `int B̄ ⊆ int P`: for
`q ∈ int B̄` there is `ε > 0` with `B(q,ε) ⊆ B̄ ⊆ P`, so `q ∈ int P`. The proof establishes that
`p̂` is interior to `B̄`; stationarity therefore follows from (B0) alone and R3's extra hypothesis
`B̄ ⊆ int P` is redundant. I had reasoned "interiority in `B̄` is not interiority in `P`", which is
true for a point of `∂B̄` and false for the interior point the proof actually produces. (ii) And the
consequence you draw is the one that matters rhetorically: `B̄ = B̄(p^{(K)}, r) ⊆ P` with `r > 0`
forces `p^{(K)} ∈ int P`, so Corollary 2 *cannot* cover a boundary minimiser, and the surrounding
text — which advertises that the variational-inequality repair frees every statement from
interiority — must not be read as covering this one.

**Resolution.** `B̄ ⊆ int P` is deleted from the statement; stationarity is concluded from (B0) plus
interiority of `p̂` in `B̄`, with the one-line argument spelled out. A new sentence says explicitly:
Proposition 2, Theorem 1 and Corollary 3's displacement bound hold for minimisers on `∂P`, because
they use only Lemma `lem:vi`; **Corollary 2 does not**, because (B0) with `r > 0` requires positive
clearance `r` between `p^{(K)}` and `∂P`, and this is a real restriction given that the `γ = 0.05`
shrinkage makes boundary and near-boundary solutions typical. So the basin corollary is the one
place where the interiority cost was not removed but merely quantified. §2 block (I).

### 8. The terminal-transition description still has wrong `ΔT_2` arithmetic — **ACCEPT**, with a **CLARIFY** defining the tighter span you invite

**Accept.** Re-derived from the definitions, node by node:

| | FULL `75→100` | DENSE `95→100` |
|---|---|---|
| launch set of the fine partition `L_κ = {jκ : jκ < M}` | `{0, 75}` | `{0, 95}` |
| fine node set `S_κ = L_κ ∪ {M}` | `{0, 75, 100}` | `{0, 95, 100}` |
| coarse node set (after removal) | `{0, 100}` | `{0, 100}` |
| removed set `I_R` | `{75}`, `|I_R| = 1 = K−1` | `{95}`, `|I_R| = 1 = K−1` |
| `ΔT_1 = max_k(τ_k − τ_{k−1})` over the **fine** partition | `max{75, 25} = 75` | `max{95, 5} = 95` |
| `w_k` for the removed node, so `ΔT_2` | `75 − 0 = 75` | `95 − 0 = 95` |
| `h_k` for the removed node | `100 − 75 = 25` | `100 − 95 = 5` |
| `|D_k|` (data in `(τ_k, τ_{k+1}]`, `Δt = 1`) | `25` | `5` |
| collapsed composite argument `ΔT_1 + ΔT_2` | `150` | `190` |
| **actual coarse span** `Λ_R = max_{k∈I_R}(h_k + w_k)` | `100` | `100` |

So R3's "`ΔT_2` reaches `100`" and "`Δ_K` carries `e^{100μ^+}`" are both wrong, in opposite
directions: `ΔT_2` is `75` (not `100`), while the *composite* argument of `Δ_K` is `150` (not
`100`), and the leading exponential of the collapsed bound is
`e^{μ^+ΔT_1}·e^{μ^+ΔT_2}·e^{μ^+(ΔT_1+ΔT_2)} = e^{300μ^+}` for FULL, not `e^{100μ^+}`. The
qualitative point — the terminal transition is the degenerate collapse to single shooting, where
Proposition 3 applies but says nothing useful — is unchanged and in fact understated by a factor of
`e^{200μ^+}`.

**Clarify — I define the tighter span separately, as you suggest.** The *actual* elapsed time in
Proposition 3's primary (nodewise) bound never exceeds `Λ_R = max_{k∈I_R}(h_k + w_k)`, which is
`100` for both terminal transitions while the collapsed corollary's argument is `150` / `190`. The
gap is entirely an artefact of collapsing `h_k` and `w_k` to independent maxima, and it is precisely
what issue 4's nodewise form avoids: on the terminal transitions the nodewise bound carries
`e^{μ(t_i−75)} ≤ e^{25μ^+}` and `ρ_{t_i} ≤ ρ_{100}`, against the collapsed `e^{75μ^+}` and
`ρ̄_{150}`. `Λ_R` is defined once in §2.1 and used in the terminal-transition paragraph and in the
`ΔT_max` amendment of issue 1 (where `ΔT_max ≥ Λ_R` is exactly the requirement). §2 block (L).

### 9. The "marginal contribution" node-selection language ignores interactions — **ACCEPT**, with a **CLARIFY** on what it costs

**Accept.** R3 wrote that "the marginal contribution of removing `τ_k` is at most
`2|D_k| e^{μ^+h_k} g_k(p) ρ^+_{h_k+w_k}(p)`" and invited the reader to rank nodes by it once. That is
wrong for exactly the reason you give: `τ_k^-` is the nearest *retained* node to the left, so
removing a node changes `w_k`, `τ_k^-` and hence `g_k` for every already-removed node in the same
consecutive run to its right. The displayed term is the contribution *given the current retained
set*, not a set-independent score, and the sum of pre-removal scores can badly misestimate the
bound of the removal set that is actually chosen — most severely for block removals, which is the
case Figure 10(b) runs.

**Resolution — the rule is stated on removal *sets*, with a sequential greedy implementation.**
For a removal set `R`, define `τ_k^-(R)` = nearest node of `S∖R` left of `τ_k`, `w_k(R) = τ_k −
τ_k^-(R)`, `g_k(R;p) = ‖φ(w_k(R);p, y_{τ_k^-(R)}) − y_{τ_k}‖`, and the score

`B(R;p) = Σ_{k∈R} Σ_{t_i∈D_k} e^{μ(t_i−τ_k)} g_k(R;p) [ ρ_{t_i−τ_k^-(R)}(p) + ρ_{t_i−τ_k}(p) ]`,

which is exactly the primary bound of Proposition 3 for that set. Greedy: `R_0 = ∅`, and
`R_{j} = R_{j−1} ∪ {argmin_c B(R_{j−1} ∪ {c}; p)}` over retained interior candidates `c`, with
`w_k`, `τ_k^-` and `g_k` **recomputed after each removal**. Alternatively, score complete proposed
removal sets by `B(R;p)` and compare them. Either way the object ranked is the bound of the set, not
a per-node constant.

**Clarify — and `g_k` is computable but not free, with a cost I can state.** R3 said `g_k` is
"one flow evaluation and one subtraction, both of which the optimiser already performs". The second
half is false: the fine objective integrates `[τ_k, τ_{k+1}]` from `y_{τ_k}`, whereas `g_k` needs the
integration of `[τ_k^-(R), τ_k]` from `y_{τ_k^-(R)}`, which is a *coarse* launch the fine objective
never performs, and which changes whenever `τ_k^-(R)` changes. Cost accounting: evaluating `B(R;p)`
costs `|R|` window integrations of total length `Σ_k w_k(R)` (the `ρ`'s are closed-form); one greedy
step costs one such integration per candidate, plus re-integration for every already-removed node
whose predecessor changed — `O(|R|)` in the worst case (a single consecutive run), `O(1)` when the
removals are isolated. The claim I keep is the one that survives: `g_k` is **observable** at the
current iterate, whereas `‖η_{τ_k}‖` is not, so a rule phrased on `g_k` is implementable and
"prefer low-noise nodes" is not. The caveat also stands: this minimises the *bound*, not the
displacement. §2 block (H).

### 10. The new gate number collides with the existing gate list — **ACCEPT**

**Accept, and the collision is real:** `report.tex` already has G9 (`report.md` contains every
section heading, table and figure caption of `report.tex`) and G10 (`report.pdf` exists, is newer
than `report.tex`, page count recorded), and R3's new closure-condition bullet was also labelled G9.
The abstract says "ten gates verify the inheritance".

**Resolution — and this is the one place where I decline your first option.** You offer "rename it
G11 or renumber the entire list". I decline both, for the reason your own "WHY" gives: G1–G10 are
mechanical **inheritance** checks that pass, and an unverified theoretical premise is not that kind
of object. Putting a permanently-failing item in that list would either corrupt the list's meaning
or invite someone to "fix" it. So:

- the ten gates G1–G10 keep their numbers, their statements and their count;
- the closure-condition item becomes **P1**, in a separate list headed *open theoretical-premise
  checks*, in §7 (Provenance and gates), stated as failing;
- the repo convention on gate polarity is respected: P1 is not a gate, so it does not enter
  `gates_summary.json`'s pass/fail tally, and §7 says why it is listed separately.

**Clarify — what the abstract says, verbatim.** The abstract's sentence becomes: *"ten gates verify
the inheritance (§7); one theoretical-premise check — the closure condition (A1) at
`ΔT_max = 100` — fails, and is reported there."* Both numbers, both outcomes, in the abstract, so
the failure is not discoverable only by reading §7. §2 block (N).

### 11. The closure-condition commentary misstates why the long-window case fails — **ACCEPT**, with a **CLARIFY** that the obvious escape is also closed

**Accept.** R3 wrote that (A1) "forces `r'` to be exponentially small", which locates the failure in
the noise radius alone. It is not there. (A1) is
`e^{μ^+ΔT_max} r' + Q(ΔT_max) diam(P) < r_X`, and since `r' > 0` this requires **both**

1. `Q(ΔT_max) · diam(P) < r_X` — a condition on the *parameter set*, containing no `r'` at all; and
2. `r' < e^{−μ^+ΔT_max}( r_X − Q(ΔT_max) diam(P) )` — the condition on the noise radius, which is
   *vacuous* (no positive `r'` exists) whenever (1) fails.

So for long windows (A1) can be unsatisfiable for every `r' > 0`, and shrinking the admissible noise
cannot repair it. Both conditions are now stated, in that order, with the remark that (1) is the
binding one.

Numerically, with the constants this report quotes (`μ = \nmu`, `L̃ = \nLtilde`, so
`Q(s) = (L̃/μ)(e^{μs} − 1)`; recomputed here, not recalled):

| `ΔT_max` | 1 | 2 | 5 | 10 | 75 | 95 | 100 |
|---|---|---|---|---|---|---|---|
| `Q(ΔT_max)` | `20.1` | `84.7` | `3.13·10³` | `1.09·10⁶` | `1.16·10³⁹` | `1.69·10⁴⁹` | `5.86·10⁵¹` |

Condition (1) is `diam(P) < r_X / Q(ΔT_max)`. At `ΔT_max = 1` that is `diam(P) < r_X/20`, a real but
conceivable requirement; at `ΔT_max = 10` it is `diam(P) < r_X/10⁶`; at `ΔT_max = 100` it is
`diam(P) < r_X/(5.9·10⁵¹)`, i.e. the parameter set must be a point for any tube of sane radius.
**The honest reading, and the one the manuscript will now carry:** the confinement machinery
certifies *short* windows only. That is consistent with the method's design — guess propagation
starts with short windows precisely because they are the tractable ones — but it explicitly does
**not** certify the final single-shooting stage, which is the stage the headline experiment is
about. The theory supports the beginning of the schedule and is silent at its end.

**Clarify — enlarging `r_X` does not rescue (1).** The obvious move is to read (1) as satisfiable by
taking `r_X` large. It is not, because `μ` and `L̃` are suprema of `λ_max(sym J_f)` and of the
parameter-derivative bound over `X × P`, and `X ⊇ 𝒯(r_X)`, so both constants are non-decreasing in
`r_X` — and for a polynomial (here cubic) right-hand side they *grow* with the tube. `Q(ΔT_max)`
then grows exponentially in `μ(r_X)·ΔT_max` while the right-hand side grows only linearly in `r_X`.
So (1) is a competition that a larger tube loses, and the only free direction is `ΔT_max`. One
caveat attached to the table above, which I would rather state than let a reader over-read: the
values `\nmu` and `\nLtilde` were measured in R002/R003 over *their* probe box, not over the
(never-instantiated) `X × P` of (A0), so the table is indicative of the order of magnitude and is
not itself a check of (A1) — which is exactly why P1 of issue 10 is reported as failing rather than
as "checked and failed". §2 blocks (A) and (N).

---

## Section 2 — Updated artifact

Everything in **R3_to_gpt.md Section 2** that is not listed below is **unchanged**, and everything
in **R2_to_gpt.md Section 2** not superseded by R3 or below is **unchanged**. In particular R3
blocks (B) minimiser convention and `lem:vi`, (E) Proposition 1, (F) Proposition 2, (N) the
penalty-bias remark, and the `lem:trunc` and (A2) parts of block (A) all stand as written there;
R2 blocks (c) constants, (k) the definition of `Ĵ_K`, (q) the Figure 10 caption and (r) the caption
and gate-bullet edits also stand. Only the items below change this round.

### 2.1 Bullet list of changes (this round)

**Structure and scope**

1. New **confinement time** `T_X` (Definition), and **Lemmas 1' and 2' restated as local
   statements** valid for `t ≤ T_X`; proofs are the R2 proofs restricted to `[0,T_X)` and extended
   by continuity. [issue 1]
2. **Lemma 0** (no escape) rewritten so that it uses *only* Lemmas 1' and 2', with the strict
   inequalities made load-bearing and the existence-up-to-`T` step stated. [issue 1]
3. New **full-window corollary** (`cor:confined`): under (A0)–(A1), clean launches and
   `r'`-perturbed launches have `T_X = min{ΔT_max, t_{N−1}−τ}`, so Lemmas 1'/2' hold on the whole
   window for them. §3–§5 cite this corollary, never the lemmas. [issue 1]
4. (A0) amended: `ΔT_max ≥ Λ_R := max_{k∈I_R}(h_k + w_k)` over every node removal considered, not
   just the longest fine window. `Λ_R` defined once in §2.1. [issues 1, 8]
5. New **"Scope of statements"** paragraph with tags **(S1)–(S4)** and a result-to-scope table;
   every theorem environment in §3–§5 opens with its scope tag. [issue 2]

**Statements**

6. `ρ̄_s = max{2‖η‖_max, ρ_s}` **restored as the primary envelope** (Lemma 3, Proposition 3's
   corollaries, Theorem 1, Corollaries 2–3); `Δ_K` described as **piecewise quadratic**, with the
   switch located; monotonicity of `ρ̄` in `s` and in `δ` proved, so `sup_U Δ_K = Δ_K(R_U)`
   survives. `ρ⁺` demoted to a remark giving the smooth quadratic majorant `Δ_K^+`. [issue 3]
7. **Proposition 3 primary bound keeps the per-datum offsets and both residual envelopes
   separately**, with the bare `ρ` at the actual elapsed times and the actual `μ` in
   `e^{μ(t_i−τ_k)}`; the `|D_k|`-collapsed and global forms become two corollaries. [issue 4]
8. **Node-selection rule** restated on removal *sets* with a sequential greedy implementation that
   recomputes `w_k`, `τ_k^-`, `g_k` after each removal, plus an explicit cost accounting for `g_k`.
   [issue 9]
9. **Envelope remark** rewritten: the suprema `𝓜(h)`, `𝒬(h)` are over all pairs of solutions
   confined to `X` (equivalently, by Lemma 0, to the tube `𝒯`), and `𝒬` explicitly over two
   parameters `p_1, p_2 ∈ P`. [issue 5]
10. **Plateau sentences withdrawn** in both places: no `p ∉ P` inference; plateau evaluations lie
    outside the exact-flow analysis, full stop, with the list of what would be needed to say more.
    [issue 6]
11. **Corollary 2**: `B̄ ⊆ int P` deleted (redundant); stationarity concluded from (B0) plus
    interiority of `p̂` in `B̄`; new sentence that the basin corollary requires positive clearance
    `r` from `∂P` while Proposition 2, Theorem 1 and Corollary 3 allow boundary minimisers.
    [issue 7]
12. **Terminal-transition paragraph** corrected: FULL `ΔT_1 = ΔT_2 = 75`, composite `150`;
    DENSE `ΔT_1 = ΔT_2 = 95`, composite `190`; actual span `Λ_R = 100` in both. [issue 8]
13. **(A1) commentary**: both necessary conditions stated, the parameter-diameter one first and
    identified as binding; the `Q(ΔT_max)` table; the note that enlarging `r_X` does not help; the
    "certifies short windows only" reading. [issue 11]
14. **Gates**: the closure-condition item is **not** a gate. G1–G10 unchanged in number and
    statement; new **P1** in a separate *open theoretical-premise checks* list; abstract states both
    the ten passing gates and the one failing premise check. [issue 10]
15. Consequential substitutions: every `ρ⁺` in R3 blocks (D) Lemma 3, (G) Proposition 3, (H)
    Theorem 1, (I) Corollary 2 and (J) Corollary 3 becomes `ρ̄`, and every citation of "Lemma 1 /
    Lemma 2" in §3–§5 becomes a citation of `cor:confined`. [issues 1, 3]

### 2.2 Corrected LaTeX

#### (A) §2.1 — (A0) horizon amendment, `Λ_R`, and (A1) with both necessary conditions [replaces the `ΔT_{\max}` clause of R3 block (A)'s (A0), and all of R3 block (A)'s (A1) paragraph and its `\readthis` item (iii)]

```latex
% --- inside (A0), replacing "and a horizon $\Delta T_{\max}>0$ at least as large as the
%     longest window of any partition under discussion" ---
and a horizon $\Delta T_{\max}>0$ with
\begin{equation}
\Delta T_{\max}\ \ge\ \max\Big\{\ \max_{\text{partitions under discussion}}\ \max_k(\tau_k-\tau_{k-1}),\
\ \Lambda_R\ \Big\},\qquad
\Lambda_R:=\max_{\text{removals under discussion}}\ \max_{k\in\mathcal I_R}\,(h_k+w_k),
\label{eq:DTmaxdef}
\end{equation}
where $h_k$ and $w_k$ are the fine window and the retained-neighbour offset of
Proposition~\ref{prop:removal}. \readthis{$\Lambda_R$ is \emph{not} bounded by the longest fine
window, and for a run of consecutively removed nodes it is not a window of the coarse partition
either: it is the elapsed time of the \emph{coarse predictor} from its retained launch node to the
last datum it predicts. Lemma~\ref{lem:noescape} confines a trajectory for at most $\Delta T_{\max}$,
and Lemma~\ref{lem:residual} is stated for elapsed times at most $\Delta T_{\max}$, so without
\eqref{eq:DTmaxdef} Proposition~\ref{prop:removal} would evaluate flows outside the interval on which
anything has been confined. This is not free: $\Delta T_{\max}$ enters \eqref{eq:closure}
exponentially, so enlarging it makes the closure condition strictly harder.}

% --- (A1), replacing R3's single-inequality version ---
\smallskip\noindent\textbf{(A1) Closure of the admissible region.}
With $\mu^{+}=\max(\mu,0)$ and $Q$ as in \eqref{eq:Qdef}, there is $r'\in(0,r_X]$ with
\begin{equation}
e^{\mu^{+}\Delta T_{\max}}\,r' \;+\; Q(\Delta T_{\max})\,\operatorname{diam}(P)\;<\;r_X .
\label{eq:closure}
\end{equation}
Since $r'>0$, \eqref{eq:closure} is satisfiable \emph{if and only if} both
\begin{equation}
\text{(A1-i)}\quad Q(\Delta T_{\max})\,\operatorname{diam}(P)\;<\;r_X
\qquad\text{and}\qquad
\text{(A1-ii)}\quad r'\;<\;e^{-\mu^{+}\Delta T_{\max}}\Big(r_X-Q(\Delta T_{\max})\operatorname{diam}(P)\Big).
\label{eq:closure_two}
\end{equation}
\readthis{(A1-i) is the binding one, and it contains no $r'$.
\emph{(i)} (A1-i) is a condition on the \emph{parameter set}: if it fails, no positive noise radius
whatever satisfies \eqref{eq:closure}, and the confinement machinery is unavailable for that
$\Delta T_{\max}$ at any noise level. Describing the long-window failure as ``$r'$ must be
exponentially small'' --- as an earlier version of this paragraph did --- misplaces it.
\emph{(ii)} Enlarging $r_X$ does not rescue (A1-i). $\mu$ and $\tilde L$ are suprema over
$X\times P$ with $X\supseteq\mathcal T(r_X)$, hence non-decreasing in $r_X$, and for a polynomial
right-hand side strictly increasing; $Q(\Delta T_{\max})$ then grows exponentially in
$\mu(r_X)\Delta T_{\max}$ while the right-hand side of (A1-i) grows only linearly in $r_X$.
\emph{(iii)} Orders of magnitude, using the constants measured in R002/R003
($\mu=\nmu$, $\tilde L=\nLtilde$, so $Q(s)=\frac{\tilde L}{\mu}(e^{\mu s}-1)$):
$Q(1)=20.1$, $Q(2)=84.7$, $Q(5)=3.13\cdot10^{3}$, $Q(10)=1.09\cdot10^{6}$,
$Q(75)=1.16\cdot10^{39}$, $Q(95)=1.69\cdot10^{49}$, $Q(100)=5.86\cdot10^{51}$. So (A1-i) reads
$\operatorname{diam}(P)<r_X/20$ at $\Delta T_{\max}=1$ and
$\operatorname{diam}(P)<r_X/(5.9\cdot10^{51})$ at $\Delta T_{\max}=100$.
\emph{(iv)} \textbf{Read this:} the confinement machinery of this section certifies \emph{short}
windows only. That is consistent with the design of guess propagation, which starts short by
construction; it is \emph{not} a certification of the terminal single-shooting stage, which is the
stage the headline experiment of \S\ref{sec:num} reports. The theory supports the beginning of the
schedule and is silent at its end.
\emph{(v)} These constants were measured over the probe box of R002/R003, not over the $X\times P$
of (A0), which has never been instantiated numerically. The table above is therefore indicative of
magnitude and is \emph{not} a check of (A1); see the premise check P1 in \S\ref{sec:prov}.}
```

#### (B) §2.1 — Scope of statements [new paragraph, placed immediately after (A2) and Lemma `lem:trunc`]

```latex
\paragraph{Scope of statements.}
Four kinds of statement appear below and they are not interchangeable. Each theorem environment in
\S\ref{sec:lemmas}--\S\ref{sec:removal} opens with its tag.
\begin{description}[nosep,leftmargin=2.2em]
\item[(S1) Local deterministic flow estimates.] Hypothesis: (A0), together with the requirement
that the two trajectories named in the statement, \emph{and the chord joining them}, remain in $X$
on the interval in question --- formally $t\le T_X$ of Definition~\ref{def:Tx}. No noise model, no
event, no (A1). \emph{Members:} Lemmas~\ref{lem:state} and \ref{lem:param},
Corollary~\ref{cor:window}(i)--(ii).
\item[(S2) Confinement under (A1).] Hypothesis: (A0)--(A1) and a launch within $r'$ of the true
orbit. Conclusion: $T_X$ is the whole window, so (S1) applies on the full window to every such pair.
Still deterministic. \emph{Members:} Lemma~\ref{lem:noescape} and Corollary~\ref{cor:confined}.
\item[(S3) Noisy-cost statements, pathwise on $E_X$.] Hypothesis: (A0)--(A2), and the event
$E_X=\{\max_i\norm{\eta_i}\le r'\}$ of \eqref{eq:EX}, which is exactly what makes every shooting
node an admissible launch for (S2). The conclusions are deterministic inequalities, asserted
realisation by realisation on $E_X$ and nowhere else. \emph{Members:} Lemma~\ref{lem:residual},
Proposition~\ref{prop:cost}(a), the pathwise bound of Proposition~\ref{prop:perr},
Proposition~\ref{prop:removal} and Corollaries~\ref{cor:removal_node}--\ref{cor:removal_global},
Theorem~\ref{thm:main}, Corollaries~\ref{cor:basin} and \ref{cor:repartition}.
\item[(S4) Expectations.] Unconditional under the bounded model (A2-a), where $\mathbb P(E_X)=1$;
\emph{conditional on $E_X$} under the Gaussian model (A2-b), with the same constants by
Lemma~\ref{lem:trunc} and $\mathbb P(E_X^{c})$ bounded by \eqref{eq:tail}. No unconditional
expectation is claimed under (A2-b). \emph{Members:} Proposition~\ref{prop:cost}(b) and the
expectation form of Proposition~\ref{prop:perr}.
\end{description}
\readthis{the distinction between (S1) and (S3) is the one a previous revision collapsed, by
declaring that ``every inequality in \S\ref{sec:lemmas}--\S\ref{sec:removal} holds on $E_X$''.
Lemmas~\ref{lem:state} and \ref{lem:param} contain no noise: their hypothesis is a \emph{geometric}
containment condition, and stamping a probabilistic event on them hides the dependency that
Lemma~\ref{lem:noescape} exists to establish --- it is that hiding which made an earlier version of
Lemma~\ref{lem:noescape} circular, since it invoked globally-quantified versions of the very lemmas
whose hypotheses its own conclusion supplies. The order is: (S1) is proved from (A0) alone; (S2)
uses (S1) to prove confinement; (S3) uses (S2) on the event where the launches are admissible;
(S4) integrates (S3).}
```

#### (C) §3 — confinement time, and Lemma 1' [replaces R2 block (f)]

```latex
\begin{definition}[confinement time of a pair]\label{def:Tx}
For $x_1,x_2\in X$ and $p_1,p_2\in P$ write $\xi_j(s)=\flow{s}{p_j}{x_j}$ and let
$[u,v]=\{u+\theta(v-u):\theta\in[0,1]\}$. Set
\[
T_X\big((x_1,p_1),(x_2,p_2)\big)
=\sup\Big\{t\in[0,\Delta T_{\max}]:\ \xi_1,\xi_2\ \text{exist on }[0,t]\ \text{and}\
[\xi_1(s),\xi_2(s)]\subset X\ \ \forall s\in[0,t]\Big\}.
\]
\end{definition}
\noindent
Three remarks. The chord contains its endpoints, so the displayed condition already requires
$\xi_1(s),\xi_2(s)\in X$. $T_X\ge0$ always, since $x_1,x_2\in X$ and $X$ is convex. And the
defining set is closed, so the condition holds on all of $[0,T_X]$: by (A0) each $\xi_j$ exists
while it remains in the compact set $X$, hence extends continuously to $T_X$ with
$\xi_j(T_X)\in X$, and $X$ closed and convex gives $[\xi_1(T_X),\xi_2(T_X)]\subset X$.

\begin{lemma}[flow sensitivity to the initial state, local; \rev{revised}]\label{lem:state}
\emph{Scope (S1).} Assume (A0). Let $p\in P$, $x_1,x_2\in X$, and let
$T=T_X((x_1,p),(x_2,p))$. Then for every $t\in[0,T]$,
\begin{equation}
e^{\mu_- t}\norm{x_1-x_2}\ \le\ \norm{\flow{t}{p}{x_1}-\flow{t}{p}{x_2}}\ \le\ e^{\mu t}\norm{x_1-x_2}.
\label{eq:lemma1}
\end{equation}
In particular, since $\mu\le L$ and $\mu_-\ge-L$, the draft's bounds
$e^{-Lt}\norm{x_1-x_2}\le\norm\cdot\le e^{Lt}\norm{x_1-x_2}$ hold on $[0,T]$.
\end{lemma}
\begin{proof}
Let $\hat x_j(t)=\flow{t}{p}{x_j}$ and $\delta=\hat x_1-\hat x_2$, so
$\dot\delta=f(\hat x_1;p)-f(\hat x_2;p)$ with $\delta(0)=x_1-x_2$. For $t<T$ the chord
$\hat x_2(t)+s\delta(t)$, $s\in[0,1]$, lies in $X$ by Definition~\ref{def:Tx}, so the mean-value form
$f(\hat x_1;p)-f(\hat x_2;p)=A(t)\delta$ with $A(t)=\int_0^1J_f(\hat x_2+s\delta;p)\dd s$ is valid and
every $J_f$ in the integrand is evaluated at a point of $X$; hence
\begin{equation}
\tfrac12\tfrac{\dd}{\dd t}\norm\delta^2=\langle\delta,A(t)\delta\rangle
 =\big\langle\delta,\tfrac12(A+A^\top)\delta\big\rangle\in\big[\mu_-\norm\delta^2,\ \mu\norm\delta^2\big]
\label{eq:proof1}
\end{equation}
on $[0,T)$. If $x_1=x_2$ both sides of \eqref{eq:lemma1} vanish. If $x_1\ne x_2$ then $\delta(t)\ne0$
on $[0,T)$: two solutions of the same initial-value problem with the same parameter that agree at one
time agree at all times by uniqueness (guaranteed by (A0)), so $\delta(t_\ast)=0$ would force
$\delta\equiv0$ and contradict $\delta(0)\ne0$. On $\{\delta\neq0\}$, $\norm\delta$ is differentiable
and \eqref{eq:proof1} gives $\mu_-\norm\delta\le\tfrac{\dd}{\dd t}\norm\delta\le\mu\norm\delta$;
Gr\"onwall in both directions gives \eqref{eq:lemma1} on $[0,T)$, and both sides are continuous at
$t=T$.
\end{proof}
\readthis{the quantifier. This lemma is \emph{not} asserted for all $t\le\Delta T_{\max}$: under (A0)
alone a trajectory launched in $X$ may leave $X$, after which the mean-value matrix integrates $J_f$
over a chord outside $X$ and $\lambda_{\max}(\operatorname{sym}A)\le\mu$ has no justification --- and
for a polynomial $f$ the solution may cease to exist. The previous revision stated
\eqref{eq:lemma1} for all $x_1,x_2\in X$ and all $t\le\Delta T_{\max}$ while simultaneously removing
forward invariance from (A0), which made the statement false and the proof of
Lemma~\ref{lem:noescape} circular. Full-window versions are Corollary~\ref{cor:confined}, obtained
\emph{after} Lemma~\ref{lem:noescape}.}
```

#### (D) §3 — Lemma 2' [replaces R2 block (g); only the statement and the containment sentences change, the Dini argument is verbatim]

```latex
\begin{lemma}[flow sensitivity to the parameters, local; \rev{revised}]\label{lem:param}
\emph{Scope (S1).} Assume (A0). Let $x_0\in X$, $p_1,p_2\in P$, and let
$T=T_X((x_0,p_1),(x_0,p_2))$. Then for every $t\in[0,T]$,
\begin{equation}
\norm{\flow{t}{p_1}{x_0}-\flow{t}{p_2}{x_0}}\ \le\ Q(t)\,\norm{p_1-p_2},
\qquad Q(t)=\frac{\tilde L}{\mu}\big(e^{\mu t}-1\big)
\label{eq:lemma2}
\end{equation}
with the conventions of \eqref{eq:Qdef}.
\end{lemma}
\begin{proof}
Let $\hat x_j(t)=\flow{t}{p_j}{x_0}$ and $\delta=\hat x_1-\hat x_2$, so $\delta(0)=0$ and
\[
\dot\delta=\underbrace{\big[f(\hat x_1;p_1)-f(\hat x_2;p_1)\big]}_{\text{state difference, equal parameter}}
+\underbrace{\big[f(\hat x_2;p_1)-f(\hat x_2;p_2)\big]}_{\text{parameter difference, equal state}} .
\]
For $t<T$ the chord $[\hat x_1(t),\hat x_2(t)]$ lies in $X$ by Definition~\ref{def:Tx}, so the first
bracket equals $A(t)\delta$ with $A$ as in \eqref{eq:proof1} and $\lambda_{\max}(\operatorname{sym}A)
\le\mu$; and $\hat x_2(t)\in X$, $p_1,p_2\in P$, so the second bracket has norm at most
$\tilde L\norm{p_1-p_2}$ by the definition of $\tilde L$ as a supremum over $X\times P$.
[The Dini-derivative argument of the previous revision, including the case $\delta(t)=0$ and the
comparison lemma, is unchanged, and is carried out on $[0,T)$.]
Hence $D^{+}\norm\delta\le\mu\norm\delta+\tilde L\norm{p_1-p_2}$ on $[0,T)$, and the comparison lemma
with $\norm{\delta(0)}=0$ gives \eqref{eq:lemma2} there; both sides are continuous at $t=T$.
\end{proof}
```

#### (E) §2.1 — Lemma 0, using only the local lemmas [replaces the `lem:noescape` of R3 block (A)]

```latex
\begin{lemma}[no escape from the tube]\label{lem:noescape}
\emph{Scope (S2).} Assume (A0)--(A1). Let $\tau\in[t_0,t_{N-1}]$, $x_\tau=x^\star(\tau)$, let
$z\in\mathbb R^{d}$ with $\norm{z-x_\tau}\le r'$, and put
$S=\min\{\Delta T_{\max},\,t_{N-1}-\tau\}$. Then for every $p\in P$ and every $s\in[0,S]$,
\[
\flow{s}{p}{x_\tau}\in\mathcal T\subseteq X
\qquad\text{and}\qquad
\flow{s}{p}{z}\in\mathcal T\subseteq X .
\]
\end{lemma}
\begin{proof}
Two exit-time bootstraps; the first is needed to license the second. Both use only
Lemmas~\ref{lem:state} and \ref{lem:param}, and only at times at which the relevant chord has
already been placed in $X$.

\emph{Step 1 (clean launch).} Let
$T=\sup\{t\in[0,S]:\ \flow{s}{p}{x_\tau}\in\mathcal T\ \forall s\le t\}$; $T\ge0$ since
$x_\tau\in\mathcal T$. Fix $s<T$. Then $\flow{s}{p}{x_\tau}\in\mathcal T\subseteq X$, and
$x^\star(\tau+s)=\flow{s}{\pstar}{x_\tau}\in\mathcal T\subseteq X$ because it lies on the orbit;
$X$ is convex, so the chord between them lies in $X$. Hence
$T\le T_X((x_\tau,p),(x_\tau,\pstar))$ and Lemma~\ref{lem:param} applies on $[0,T]$, giving
\begin{equation}
\operatorname{dist}\big(\flow{s}{p}{x_\tau},\,x^\star([t_0,t_{N-1}])\big)
\le\norm{\flow{s}{p}{x_\tau}-x^\star(\tau+s)}
\le Q(s)\norm{p-\pstar}\le Q(\Delta T_{\max})\operatorname{diam}(P)\ <\ r_X
\label{eq:clean_conf}
\end{equation}
for all $s\le T$, the last inequality being \emph{strict} by \eqref{eq:closure} (which subtracts a
positive $e^{\mu^{+}\Delta T_{\max}}r'$). Suppose $T<S$. The trajectory remains in the compact set
$\mathcal T\subseteq X$ on $[0,T)$, so by (A0) it extends to $T$ with $\flow{T}{p}{x_\tau}\in X$, and
\eqref{eq:clean_conf} holds at $s=T$ with strict inequality; by continuity of $s\mapsto
\operatorname{dist}(\flow{s}{p}{x_\tau},x^\star([t_0,t_{N-1}]))$ there is $\varepsilon>0$ with the
distance $<r_X$, hence the trajectory in $\mathcal T$, on $[0,T+\varepsilon]\cap[0,S]$ ---
contradicting the definition of $T$ as a supremum. Therefore $T=S$, and \eqref{eq:clean_conf} holds
on $[0,S]$.

\emph{Step 2 (perturbed launch).} Let
$T'=\sup\{t\in[0,S]:\ \flow{s}{p}{z}\in\mathcal T\ \forall s\le t\}$; $T'\ge0$ since
$\norm{z-x_\tau}\le r'\le r_X$. Fix $s<T'$. Then $\flow{s}{p}{z}\in\mathcal T\subseteq X$ and, by
Step 1, $\flow{s}{p}{x_\tau}\in\mathcal T\subseteq X$; $X$ convex puts the chord in $X$, so
$T'\le T_X((z,p),(x_\tau,p))$ and Lemma~\ref{lem:state} applies on $[0,T']$:
\[
\norm{\flow{s}{p}{z}-\flow{s}{p}{x_\tau}}\le e^{\mu s}\norm{z-x_\tau}\le e^{\mu^{+}\Delta T_{\max}}r' ,
\]
using $e^{\mu s}\le e^{\mu^{+}\Delta T_{\max}}$ for both signs of $\mu$. With \eqref{eq:clean_conf},
\[
\operatorname{dist}\big(\flow{s}{p}{z},\,x^\star([t_0,t_{N-1}])\big)
\ \le\ e^{\mu^{+}\Delta T_{\max}}r'+Q(\Delta T_{\max})\operatorname{diam}(P)\ <\ r_X
\]
by \eqref{eq:closure}, again \emph{strictly}. The same extension-and-continuity argument as in
Step 1 gives $T'=S$.
\end{proof}

\begin{corollary}[full-window estimates for admissible launches]\label{cor:confined}
\emph{Scope (S2).} Assume (A0)--(A1). Let $\tau\in[t_0,t_{N-1}]$, $S=\min\{\Delta T_{\max},
t_{N-1}-\tau\}$, and let $z,z_1,z_2$ satisfy $\norm{\cdot-x^\star(\tau)}\le r'$. Then for all
$p,p_1,p_2\in P$ and all $s\in[0,S]$: all the trajectories below lie in $\mathcal T\subseteq X$, every
chord between two of them lies in $X$, and
\begin{align}
e^{\mu_- s}\norm{z_1-z_2}\ \le\ \norm{\flow{s}{p}{z_1}-\flow{s}{p}{z_2}}\ &\le\ e^{\mu s}\norm{z_1-z_2},
\label{eq:conf_state}\\
\norm{\flow{s}{p_1}{z}-\flow{s}{p_2}{z}}\ &\le\ Q(s)\norm{p_1-p_2}.
\label{eq:conf_param}
\end{align}
The same conclusions hold with $z$ replaced by any state reachable as $\flow{\sigma}{p}{z}$ with
$\sigma+s\le S$, by the flow property.
\end{corollary}
\begin{proof}
Lemma~\ref{lem:noescape} places each trajectory in $\mathcal T\subseteq X$ for all $s\le S$; $X$ is
convex, so every chord lies in $X$; hence $T_X=S$ for each pair, and
Lemmas~\ref{lem:state} and \ref{lem:param} apply on the whole of $[0,S]$. The last sentence follows
because $\flow{s}{p}{\flow{\sigma}{p}{z}}=\flow{\sigma+s}{p}{z}$, which
Lemma~\ref{lem:noescape} has already confined.
\end{proof}
\readthis{\S\ref{sec:lemmas}--\S\ref{sec:removal} cite \emph{this corollary}, never
Lemmas~\ref{lem:state}/\ref{lem:param} directly; the lemmas are local and their hypothesis
$t\le T_X$ is discharged once and for all here. The clause about reachable states is what
Proposition~\ref{prop:removal} needs: its coarse predictor is launched from
$\flow{w_k}{p}{y_{\tau_k^{-}}}$, which is \emph{not} within $r'$ of the orbit --- it is within $r_X$
--- so it is admissible as a \emph{reachable} state and not as a launch.}
```

#### (F) §2.2 — the residual envelope, `ρ̄` primary [replaces R3 block (C)]

```latex
Write $\mu^{+}=\max(\mu,0)$ and
\begin{equation}
Q(s)=\frac{\tilde L}{\mu}\big(e^{\mu s}-1\big),\qquad\text{read as }\ \tilde L\,s\ \text{ when }\mu=0,
\label{eq:Qdef}
\end{equation}
the continuous extension $\lim_{\mu\to0}(e^{\mu s}-1)/\mu=s$. For every sign of $\mu$, $Q(s)\ge0$ and
$s\mapsto Q(s)$ is non-decreasing, with $Q(s)\le\tilde L s$ when $\mu\le0$ and
$Q(s)\le\tilde L/|\mu|$ when $\mu<0$. The residual abbreviations are
\begin{equation}
\rho_s(p)=\big(1+e^{\mu s}\big)\norm{\eta}_{\max}+Q(s)\norm{p-\pstar},
\qquad
\bar\rho_s(p)=\max\big\{2\norm\eta_{\max},\ \rho_s(p)\big\},\qquad s\ge0 .
\label{eq:rho}
\end{equation}
\begin{lemma}[properties of the envelope]\label{lem:rhobar}
For every $s\ge0$, every $p\in P$ and every sign of $\mu$:
\emph{(i)} $\bar\rho_s(p)=\sup_{0\le r\le s}\rho_r(p)$, so $\rho_\sigma\le\bar\rho_s$ whenever
$0\le\sigma\le s$;
\emph{(ii)} $s\mapsto\bar\rho_s(p)$ is non-decreasing;
\emph{(iii)} $\delta\mapsto\bar\rho_s$ is non-decreasing, where $\delta=\norm{p-\pstar}$, and
$E\mapsto\bar\rho_s$ is non-decreasing, where $E=\norm\eta_{\max}$;
\emph{(iv)} $\bar\rho_s(\pstar)=\big(1+e^{\mu^{+}s}\big)\norm\eta_{\max}$;
\emph{(v)} if $\mu\ge0$ then $\bar\rho_s=\rho_s$; if $\mu<0$ then $\bar\rho_s=2\norm\eta_{\max}$
exactly when $\tilde L\norm{p-\pstar}\le|\mu|\,\norm\eta_{\max}$, and $\bar\rho_s=\rho_s$ otherwise.
\end{lemma}
\begin{proof}
(i) $\partial_r\rho_r=e^{\mu r}\big(\mu\norm\eta_{\max}+\tilde L\norm{p-\pstar}\big)$ has a sign
independent of $r$, so $\sup_{0\le r\le s}\rho_r=\max\{\rho_0,\rho_s\}$ and $\rho_0=2\norm\eta_{\max}$.
(ii) is immediate from (i). (iii) A maximum of two functions non-decreasing in $\delta$ (resp.\ $E$)
is non-decreasing. (iv) At $p=\pstar$, $\bar\rho_s=\max\{2,1+e^{\mu s}\}\norm\eta_{\max}
=(1+e^{\mu^{+}s})\norm\eta_{\max}$. (v) For $\mu<0$, writing $Q(s)=\frac{\tilde L}{|\mu|}(1-e^{-|\mu|s})$
and $e^{\mu s}-1=-(1-e^{-|\mu|s})$ gives
$\rho_s-2\norm\eta_{\max}=(1-e^{-|\mu|s})\big(\tilde L\norm{p-\pstar}/|\mu|-\norm\eta_{\max}\big)$,
whose sign is that of $\tilde L\norm{p-\pstar}-|\mu|\norm\eta_{\max}$ and does not depend on $s>0$.
\end{proof}
\begin{remark}[the smooth quadratic majorant, and why it is not the primary bound]\label{rem:rhoplus}
Put $\rho^{+}_s(p)=\big(1+e^{\mu^{+}s}\big)\norm\eta_{\max}+Q(s)\norm{p-\pstar}$. Then
$\bar\rho_s\le\rho^{+}_s$ for every $s$ and $p$, with equality iff $\mu\ge0$ or $p=\pstar$, so every
bound below remains true with $\bar\rho$ replaced by $\rho^{+}$; $\rho^{+}$ is a single affine form in
$(\norm\eta_{\max},\norm{p-\pstar})$ with non-negative deterministic coefficients
$A_s=1+e^{\mu^{+}s}$, $B_s=Q(s)$, which makes $\Delta_K$ of \eqref{eq:prop_removal} a homogeneous
quadratic form rather than a piecewise quadratic one. \readthis{$\rho^{+}$ is a \emph{convenience},
not the bound. In the dissipative case $\mu<0$ with
$\tilde L\norm{p-\pstar}\le|\mu|\norm\eta_{\max}$, Lemma~\ref{lem:rhobar}(v) gives
$\bar\rho_s=2\norm\eta_{\max}$ while $\rho^{+}_s=2\norm\eta_{\max}+Q(s)\norm{p-\pstar}$, which is
strictly larger for every $p\ne\pstar$ and exceeds it by up to
$\tilde L\norm{p-\pstar}/|\mu|$. Since $\mu<0$ is exactly the regime this revision exists to
exploit, the primary statements below all carry $\bar\rho$, and $\Delta_K$ is described as
\emph{piecewise} quadratic --- the truthful description --- with the branch switching on the sign of
$\tilde L\norm{q-\pstar}-|\mu|\norm\eta_{\max}$. A previous revision made $\rho^{+}$ primary and
justified it by claiming the choice ``costs nothing where every number in
\S\ref{sec:num} is evaluated''; that claim was doubly wrong. Figure~\ref{fig:removal} evaluates at
$\pstar$ \emph{and} at an off-truth perturbation \emph{and} at $p_{\rm wrong}$; and more
fundamentally, the global FHN constant is $\mu=\nmu>0$, where by Lemma~\ref{lem:rhobar}(v)
$\rho=\bar\rho=\rho^{+}$ \emph{identically}, so \S\ref{sec:num} cannot distinguish the two choices at
all. The distinction lives where the per-window and weighted-norm analysis puts
$\mu_k<0$ (Table~\ref{tab:windows}: $\nfracDneg$ of windows have $\Lambda^D_k<0$), and there
$\bar\rho$ is strictly tighter.}
\end{remark}
```

*Consequential substitution:* in R3 block (D) (Lemma 3), the final inequality reads
`ρ_{σ_i} ≤ ρ̄_{σ_i} ≤ ρ̄_s` by Lemma `lem:rhobar`(i)–(ii), and every `ρ⁺` in R3 blocks (G), (H),
(I), (J) becomes `ρ̄`.

#### (G) Proposition 3, primary form with per-datum offsets, and its two corollaries [replaces R3 block (G) in full]

```latex
\begin{proposition}[cost change under node removal, nodewise; \rev{revised}]\label{prop:removal}
\emph{Scope (S3).} Assume (A0)--(A2) and work on $E_X$. Let the coarse partition be a sub-partition
of the fine one, with removed indices $\mathcal I_R\subset\{1,\dots,K-1\}$, and let $\Delta T_{\max}$
satisfy \eqref{eq:DTmaxdef}. For $k\in\mathcal I_R$ put
\[
h_k=\tau_{k+1}-\tau_k,\qquad w_k=\tau_k-\tau_k^{-},\qquad
D_k=\{t_i\in(\tau_k,\tau_{k+1}]\},\qquad
g_k(p)=\big\lVert\flow{w_k}{p}{y_{\tau_k^{-}}}-y_{\tau_k}\big\rVert ,
\]
where $\tau_k^{-}$ is the nearest \emph{retained} node to the left of $\tau_k$. Then for every
$p\in P$
\begin{equation}
\big|\hatJ_K(p)-J_K(p)\big|\ \le\
\sum_{k\in\mathcal I_R}\ \sum_{t_i\in D_k}
e^{\mu(t_i-\tau_k)}\;g_k(p)\;\Big[\rho_{t_i-\tau_k^{-}}(p)+\rho_{t_i-\tau_k}(p)\Big],
\label{eq:prop_removal_node}
\end{equation}
and each node factor obeys
\begin{equation}
g_k(p)\ \le\ e^{\mu w_k}\norm{\eta_{\tau_k^{-}}}+\norm{\eta_{\tau_k}}+Q(w_k)\norm{p-\pstar}
\ \le\ \rho_{w_k}(p).
\label{eq:gk}
\end{equation}
\end{proposition}
\begin{proof}
\emph{Which data change predictor.} In $J_K$ the datum $t_i\in(\tau_{k-1},\tau_k]$ is launched from
$\tau_{k-1}$; in $\hatJ_K$ from the last \emph{retained} node at or before $\tau_{k-1}$. So $t_i$
changes predictor iff its fine launching node is removed, i.e.\ exactly for
$t_i\in\bigcup_{k\in\mathcal I_R}D_k$. The half-open convention handles the two apparent off-by-one
cases: with $\tau_k$ removed and $\tau_{k-1}$ retained, the datum at time $\tau_k$ lies in
$(\tau_{k-1},\tau_k]$, keeps its predictor, and is correctly \emph{not} in $D_k$; with $\tau_{k-1}$
and $\tau_k$ both removed, the datum at $\tau_k$ lies in $D_{k-1}$, does change predictor, and is
correctly counted there. The $D_k$ are pairwise disjoint.

\emph{Admissibility of every flow evaluation.} On $E_X$ every node satisfies
$\norm{y_\tau-x^\star(\tau)}\le r'$, so Corollary~\ref{cor:confined} applies to each launch; the
coarse predictor is launched from the \emph{reachable} state $\flow{w_k}{p}{y_{\tau_k^{-}}}$ and is
covered by the last clause of that corollary, the total elapsed time $t_i-\tau_k^{-}\le h_k+w_k$
being at most $\Delta T_{\max}$ by \eqref{eq:DTmaxdef}. All chords lie in $X$ by convexity.

\emph{The per-datum estimate, with the actual offsets.} For $t_i\in D_k$ put
$u_i=\flow{t_i-\tau_k^{-}}{p}{y_{\tau_k^{-}}}$, $v_i=\flow{t_i-\tau_k}{p}{y_{\tau_k}}$,
$a_i=\norm{u_i-y_i}$, $b_i=\norm{v_i-y_i}$. Then
$\hatJ_K-J_K=\sum_{k\in\mathcal I_R}\sum_{t_i\in D_k}(a_i^2-b_i^2)$ and
$|a_i^2-b_i^2|\le\norm{u_i-v_i}\,(a_i+b_i)$. By the flow property
$u_i=\flow{t_i-\tau_k}{p}{\flow{w_k}{p}{y_{\tau_k^{-}}}}$, so \eqref{eq:conf_state} at the
\emph{actual} elapsed time $t_i-\tau_k$ gives
\[
\norm{u_i-v_i}\ \le\ e^{\mu(t_i-\tau_k)}\,g_k(p) ,
\]
with the actual $\mu$ and no enlargement of the exponent, hence no monotonicity requirement and no
$\mu^{+}$. Lemma~\ref{lem:residual} applied at node $\tau_k^{-}$ and datum $t_i$ gives
$a_i\le\rho_{t_i-\tau_k^{-}}(p)$, and applied at node $\tau_k$ and datum $t_i$ gives
$b_i\le\rho_{t_i-\tau_k}(p)$ --- both at their own elapsed times, so again no envelope is needed.
Summing over $D_k$ and over $k\in\mathcal I_R$ gives \eqref{eq:prop_removal_node}. \eqref{eq:gk} is
Lemma~\ref{lem:residual} applied to the single datum $y_{\tau_k}$ launched from $y_{\tau_k^{-}}$ over
elapsed time $w_k$.
\end{proof}

\begin{corollary}[per-node form]\label{cor:removal_node}
Under the hypotheses of Proposition~\ref{prop:removal},
\begin{equation}
\big|\hatJ_K(p)-J_K(p)\big|\ \le\
\sum_{k\in\mathcal I_R}|D_k|\;e^{\mu^{+}h_k}\;g_k(p)\;
\Big[\bar\rho_{h_k+w_k}(p)+\bar\rho_{h_k}(p)\Big].
\label{eq:prop_removal_pernode}
\end{equation}
\end{corollary}
\begin{proof}
For $t_i\in D_k$: $t_i-\tau_k\le h_k$, so $e^{\mu(t_i-\tau_k)}\le e^{\mu^{+}h_k}$ (with $\mu^{+}$,
because for $\mu<0$ the correct majorant of $e^{\mu(t_i-\tau_k)}$ over $t_i-\tau_k\in[0,h_k]$ is
$e^0=1$); $t_i-\tau_k^{-}\le h_k+w_k$ and $t_i-\tau_k\le h_k$, so
$\rho_{t_i-\tau_k^{-}}\le\bar\rho_{h_k+w_k}$ and $\rho_{t_i-\tau_k}\le\bar\rho_{h_k}$ by
Lemma~\ref{lem:rhobar}(i). The inner sum has $|D_k|$ terms.
\end{proof}

\begin{corollary}[collapsed form]\label{cor:removal_global}
Under the hypotheses of Proposition~\ref{prop:removal}, with $n_{\max}=\max_{k\in\mathcal I_R}|D_k|$,
$\Delta T_1=\max_k(\tau_k-\tau_{k-1})$ over the fine partition and
$\Delta T_2=\max_{k\in\mathcal I_R}w_k$,
\begin{equation}
\big|\hatJ_K(p)-J_K(p)\big|\ \le\
n_{\max}\,|\mathcal I_R|\;e^{\mu^{+}\Delta T_1}\;\bar\rho_{\Delta T_2}(p)\;
\Big[\bar\rho_{\Delta T_1+\Delta T_2}(p)+\bar\rho_{\Delta T_1}(p)\Big]\ =:\ \Delta_K(p).
\label{eq:prop_removal}
\end{equation}
At $p=\pstar$ this is
$n_{\max}|\mathcal I_R|\,e^{\mu^{+}\Delta T_1}\big(1+e^{\mu^{+}\Delta T_2}\big)
\big[\big(1+e^{\mu^{+}(\Delta T_1+\Delta T_2)}\big)+\big(1+e^{\mu^{+}\Delta T_1}\big)\big]
\norm\eta_{\max}^2$, quadratic in the noise. Moreover $\Delta_K$ is non-decreasing in
$\delta=\norm{p-\pstar}$, so $\sup_{q\in U}\Delta_K(q)=\Delta_K$ evaluated at
$\delta=R_U:=\sup_{q\in U}\norm{q-\pstar}$.
\end{corollary}
\begin{proof}
$h_k\le\Delta T_1$, $w_k\le\Delta T_2$, $|D_k|\le n_{\max}$,
$g_k\le\rho_{w_k}\le\bar\rho_{\Delta T_2}$ by \eqref{eq:gk} and Lemma~\ref{lem:rhobar}(i), and
$\bar\rho_{h_k+w_k}\le\bar\rho_{\Delta T_1+\Delta T_2}$, $\bar\rho_{h_k}\le\bar\rho_{\Delta T_1}$ by
Lemma~\ref{lem:rhobar}(ii); the sum has $|\mathcal I_R|$ terms. The value at $\pstar$ is
Lemma~\ref{lem:rhobar}(iv) and the monotonicity in $\delta$ is Lemma~\ref{lem:rhobar}(iii).
\end{proof}
\readthis{three losses, taken in this order and each reversible by moving one step back.
\emph{(a)} \eqref{eq:prop_removal_node} is the sharpest statement: actual $\mu$, actual per-datum
offsets, and the two residual envelopes kept \emph{separately}. A previous revision bounded
$a_i+b_i$ by $2\bar\rho_{h_k+w_k}$, discarding the shorter of the two for a factor of two, and
replaced $e^{\mu(t_i-\tau_k)}$ by $e^{\mu^{+}h_k}$ before summing; both losses are now confined to
the corollaries, where they are the point.
\emph{(b)} \eqref{eq:prop_removal} is the form used in Theorem~\ref{thm:main} and
Corollaries~\ref{cor:basin}--\ref{cor:repartition}, because those need a bound \emph{uniform over $q$
in a set}. It is also the form in which every node-specific quantity has been maximised away: it
contains $|\mathcal I_R|$ but not \emph{which} nodes, and the global $\norm\eta_{\max}$ but not the
noise at any particular node. The node-selection rule below is therefore a statement about
\eqref{eq:prop_removal_node}, not about \eqref{eq:prop_removal}.
\emph{(c)} By Lemma~\ref{lem:rhobar}(v), $\Delta_K$ is \emph{piecewise} quadratic in
$(\norm\eta_{\max},\delta)$, the branch switching on the sign of
$\tilde L\delta-|\mu|\norm\eta_{\max}$ when $\mu<0$ and being the single quadratic branch throughout
when $\mu\ge0$. Remark~\ref{rem:rhoplus} gives the smooth quadratic majorant for readers who want
one.}
```

#### (H) The node-selection rule, on removal sets [replaces the "Design rule" paragraph of R3 block (G)]

```latex
\paragraph{Node selection from the nodewise bound.}
For a proposed removal set $R$ (a subset of the interior fine nodes) write $\tau_k^{-}(R)$ for the
nearest node of $S\setminus R$ to the left of $\tau_k$, $w_k(R)=\tau_k-\tau_k^{-}(R)$,
$g_k(R;p)=\norm{\flow{w_k(R)}{p}{y_{\tau_k^{-}(R)}}-y_{\tau_k}}$, and
\begin{equation}
B(R;p)=\sum_{k\in R}\ \sum_{t_i\in D_k}
e^{\mu(t_i-\tau_k)}\,g_k(R;p)\,\Big[\rho_{t_i-\tau_k^{-}(R)}(p)+\rho_{t_i-\tau_k}(p)\Big],
\label{eq:score}
\end{equation}
which is \eqref{eq:prop_removal_node} for that set. Two usable rules: \emph{(i)} score complete
proposed removal sets by $B(R;p)$ at the current iterate $p$ and take the smallest; \emph{(ii)}
\emph{sequential greedy}: $R_0=\emptyset$ and
$R_j=R_{j-1}\cup\{\arg\min_{c}B(R_{j-1}\cup\{c\};p)\}$ over retained interior candidates $c$,
\emph{recomputing} $\tau_k^{-}$, $w_k$ and $g_k$ for the whole current set after each removal.
Qualitatively, \eqref{eq:score} prefers nodes that launch few data ($|D_k|$ small), sit close to
their retained left neighbour ($w_k$ small), and above all have a small \emph{coarse residual} $g_k$.
\readthis{two corrections to how a previous revision stated this.
\emph{(1) There is no set-independent per-node score.} $\tau_k^{-}$ is the nearest \emph{retained}
node, so removing one node changes $w_k$, $\tau_k^{-}$ and $g_k$ for every already-removed node in
the same consecutive run to its right. The displayed term is a contribution \emph{given the current
retained set}, not a marginal cost that can be ranked once; ranking once and removing the top
$|\mathcal I_R|$ can select a set whose bound is far from the sum of its pre-removal scores, and the
discrepancy is largest for block removals --- the case of Figure~\ref{fig:removal}(b).
\emph{(2) $g_k$ is computable but not free.} It is \emph{observable} at the current iterate --- one
window integration and one subtraction --- which is the property that matters, since the node noise
$\norm{\eta_{\tau_k}}$ of \eqref{eq:gk} is not observable and ``prefer low-noise nodes'' is therefore
not an implementable rule. But it is \emph{not} a quantity the fine objective has already computed:
the fine objective integrates $[\tau_k,\tau_{k+1}]$ from $y_{\tau_k}$, whereas $g_k$ needs
$[\tau_k^{-}(R),\tau_k]$ from $y_{\tau_k^{-}(R)}$, a coarse launch the fine objective never performs.
Evaluating $B(R;p)$ costs $|R|$ window integrations of total length $\sum_k w_k(R)$; one greedy step
costs one integration per candidate plus a re-integration for every already-removed node whose
predecessor changed, i.e.\ $O(|R|)$ in the worst case and $O(1)$ when removals are isolated.
\emph{(3)} The standing caveat: this minimises the \emph{bound} on the cost gap, which is not the
same as minimising the displacement of the minimiser; that step needs the strong-convexity premise of
Theorem~\ref{thm:main}.}
```

#### (I) Corollary 2 — interiority corrected [replaces the statement's last sentence, the corresponding proof sentence, and part of the `\readthis` of R3 block (I)]

```latex
% --- statement: replaces "If moreover $\bar B\subseteq\operatorname{int}P$, then ..." ---
Moreover $\hat p$ is a stationary point of $\hatJ_K$, $\nabla\hatJ_K(\hat p)=0$, and hence a local
minimiser of $\hatJ_K$ on $P$.

% --- proof: replaces the final sentence ---
Finally, $\hat p$ is interior to $\bar B$ and $\bar B\subseteq P$ by (B0), so there is
$\varepsilon>0$ with $B(\hat p,\varepsilon)\subseteq\bar B\subseteq P$; hence
$\hat p\in\operatorname{int}P$, the unconstrained first-order condition applies, and
$\nabla\hatJ_K(\hat p)=0$.

% --- \readthis: replaces the sentence "Interiority to $\bar B$ is not interiority to $P$ ..." ---
\emph{Interiority costs nothing extra here, but positive clearance from $\partial P$ does.} Since
$r>0$, (B0) makes $\bar B$ a full-dimensional closed ball inside $P$, so every point interior to
$\bar B$ is automatically interior to $P$ and the stationarity conclusion needs no hypothesis beyond
(B0); an earlier version of this corollary added $\bar B\subseteq\operatorname{int}P$, which is
redundant. The genuine cost is elsewhere and must be stated: (B0) with $r>0$ forces
$p^{(K)}\in\operatorname{int}P$ at distance at least $r$ from $\partial P$, so \emph{this corollary
does not cover a minimiser on $\partial P$}, and the surrounding claim that
Lemma~\ref{lem:vi} frees the theory from interiority assumptions must not be read as covering it.
Proposition~\ref{prop:perr}, Theorem~\ref{thm:main} and the displacement bound of
Corollary~\ref{cor:repartition} do hold verbatim for boundary minimisers, because they use only
\eqref{eq:vi}; the basin corollary requires positive clearance $r$ from $\partial P$, which is a real
restriction given that the $\gamma=0.05$ shrinkage makes boundary and near-boundary solutions
typical rather than exceptional.
```

#### (J) Theorem 1 — `Δ_K` piecewise quadratic, supremum still computable [replaces the `\eqref{eq:DeltaK_quad}` display and the sentences around it in R3 block (H)]

```latex
Writing $E=\norm\eta_{\max}$, $\delta=\norm{q-\pstar}$ and
$c_0=n_{\max}|\mathcal I_R|e^{\mu^{+}\Delta T_1}$, the bound of \eqref{eq:prop_removal} is
\begin{equation}
\Delta_K(q)=c_0\,\bar\rho_{\Delta T_2}(q)\Big[\bar\rho_{\Delta T_1+\Delta T_2}(q)
+\bar\rho_{\Delta T_1}(q)\Big],
\label{eq:DeltaK_quad}
\end{equation}
a \emph{piecewise} homogeneous quadratic in $(E,\delta)$ with non-negative deterministic
coefficients on each branch: by Lemma~\ref{lem:rhobar}(v) the branch is the single quadratic
$c_0\rho_{\Delta T_2}[\rho_{\Delta T_1+\Delta T_2}+\rho_{\Delta T_1}]$ whenever $\mu\ge0$ or
$\tilde L\delta>|\mu|E$, and the constant-in-$s$ branch $4c_0E^2$ when $\mu<0$ and
$\tilde L\delta\le|\mu|E$. On every branch $\Delta_K$ is non-decreasing in $\delta$, so
$\sup_{q\in U}\Delta_K(q)$ is $\Delta_K$ evaluated at $\delta=R_U:=\sup_{q\in U}\norm{q-\pstar}$:
the supremum in \eqref{eq:thm} is computable, not merely finite. A smooth quadratic majorant
$\Delta_K\le\Delta_K^{+}$, obtained by replacing each $\bar\rho$ with $\rho^{+}$, is given in
Remark~\ref{rem:rhoplus}; it is convenient (for instance for solving (B2) of
Corollary~\ref{cor:basin} for $r$ in closed form) and strictly loose for $\mu<0$ away from $\pstar$.
```

#### (K) The envelope remark — suprema over the confined tube [replaces the `\eqref{eq:envelopes}` display and the paragraph introducing it in R3 block (K)]

```latex
\begin{remark}[what would be needed to substitute these into the propositions]\label{rem:envelopes}
Corollary~\ref{cor:window} does \emph{not} license replacing $e^{\mu^{+}s}$ by $e^{\Lambda_k}$, or by
$M_k$ of \eqref{eq:Mk}, anywhere in \S\ref{sec:lemmas}--\S\ref{sec:removal}. What would license a
substitution is a pair of \emph{deterministic envelopes} taken over the class of pairs the proofs
actually use. Call a pair $(\xi_1,\xi_2)$ of solutions of $\dot\xi=f(\xi;p_j)$ on
$[a,b]\subseteq[t_0,t_{N-1}]$, with $p_1,p_2\in P$, \emph{confined} if
$[\xi_1(s),\xi_2(s)]\subset X$ for every $s\in[a,b]$, and write
$\mu_\xi(s)=\lambda_{\max}\big(\tfrac12(A_\xi+A_\xi^{\top})\big)$ for its mean-value matrix
$A_\xi(s)=\int_0^1J_f\big(\xi_2(s)+\theta(\xi_1(s)-\xi_2(s));p_1\big)\dd\theta$. For $h>0$ put
\begin{align}
\mathcal M(h)&=\sup\Big\{\exp\Big(\int_a^b\mu_\xi(r)\dd r\Big):\ \text{confined pairs with }p_1=p_2\in P,\
0\le b-a\le h\Big\},\label{eq:envelopeM}\\
\mathcal Q(h)&=\sup\Big\{\tilde L\int_a^b\exp\Big(\int_s^b\mu_\xi(r)\dd r\Big)\dd s:\
\text{confined pairs with }p_1,p_2\in P,\ 0\le b-a\le h\Big\}.\label{eq:envelopeQ}
\end{align}
With these, Lemma~\ref{lem:residual}, Propositions~\ref{prop:cost} and \ref{prop:removal},
Theorem~\ref{thm:main} and Corollaries~\ref{cor:basin}--\ref{cor:repartition} hold with
$e^{\mu^{+}s}\rightsquigarrow\mathcal M(s)$ and $Q(s)\rightsquigarrow\mathcal Q(s)$, and nothing is
lost, since $\mu_\xi(r)\le\mu$ pointwise on $X$ gives $\mathcal M(h)\le e^{\mu^{+}h}$ and
$\mathcal Q(h)\le Q(h)$.
\readthis{\emph{why the class must be ``confined'', not ``launched within $r'$ of the orbit''.} On
$E_X$, Lemma~\ref{lem:noescape} places every trajectory used in
\S\ref{sec:lemmas}--\S\ref{sec:removal} inside $\mathcal T\subseteq X$, and convexity of $X$ places
every joining chord in $X$ --- so every pair the proofs use is confined, which is exactly what
licenses the substitution. It is \emph{not} true that every such pair is launched within $r'$ of the
orbit: the coarse predictor of Proposition~\ref{prop:removal} is launched from
$\flow{w_k}{p}{y_{\tau_k^{-}}}$, a reachable state that Lemma~\ref{lem:noescape} places within $r_X$
of the orbit and not within $r'$, and the re-partition pairs of
Corollary~\ref{cor:repartition} are launched from two different partitions' nodes. A previous
version of this remark defined the suprema over pairs ``launched anywhere within $r'$ of the orbit,
at any $p\in P$'', which excludes those pairs and left the advertised substitution unlicensed in
precisely the place it was advertised; and ``at any $p\in P$'' is ambiguous for $\mathcal Q$, whose
pairs carry \emph{two} parameters. Both are fixed above.
\emph{Four reasons the per-window quantities cannot play this role}, each sufficient on its own.
\emph{(1) Pair dependence.} $\mu(\cdot)$ is attached to one pair and one parameter, while the proofs
use at least five: noisy versus clean launch, $p$ versus $\pstar$, coarse versus fine prediction,
different $q\in U$, and windows of two partitions and their common refinement.
\emph{(2) Interval spanning.} A coarse interval crosses several fine windows and $\exp(\int_a^b\mu)$
is then a \emph{product} of per-window factors, which $\max_kM_k$ does not bound.
\emph{(3) The forcing integral.} $\int_0^t\exp(\int_s^t\mu)\dd s\le M_kt$ requires $[s,t]$ to lie
inside window $k$ for every $s\le t$, which fails as soon as $t$ is in a later window than $0$.
\emph{(4) Randomness.} An a posteriori $M_k$ computed along a realised noisy pair is a random
variable and cannot be factored out of $\mathbb E\norm\eta^2$.
And what \S\ref{sec:num} reports is neither $\mathcal M$ nor $M_k$: Table~\ref{tab:windows} gives
\emph{linearised endpoint} exponents $e^{\Lambda_k}$ computed from $J_f(x^\star(t);\pstar)$ along the
\emph{true} orbit at the \emph{true} parameter with windows aligned to $t_0$ --- the
infinitesimal-perturbation limit of one pair. These are not suprema over a tube of trajectories, not
valid at another $p$, and not sub-interval bounds ($M_k\ge\max\{1,e^{\Lambda_k}\}$, strictly wherever
$\mu(\cdot)$ changes sign inside the window, which on FHN happens twice per period,
Figure~\ref{fig:lemmas}c). $\mathcal M(h)$ is not computed anywhere in this report. The
node-placement rule below is therefore a \emph{heuristic} read off a measurement, and a previous
version of this corollary carried it as a licensed substitution (item (iii)), which is withdrawn.}
\end{remark}
```

#### (L) §2.3 — the terminal transitions, with the exact quantities [replaces the `\emph{Read this:}` sentences of R3 block (L) from "the second nested transition" to "covers vacuously"]

```latex
\emph{Read this:} the second nested transition of FULL and of DENSE is in each case the terminal
$\kappa_b=M$ stage, i.e.\ the collapse to single shooting, which is a node removal only in the
degenerate sense that \emph{every} interior node is removed. The exact quantities, from the
definitions of Proposition~\ref{prop:removal} and Corollary~\ref{cor:removal_global}:
\begin{center}
\begin{tabular}{lcc}
\toprule
 & FULL $75\to100$ & DENSE $95\to100$\\
\midrule
fine node set $S_{\kappa}=L_\kappa\cup\{M\}$ & $\{0,75,100\}$ & $\{0,95,100\}$\\
coarse node set & $\{0,100\}$ & $\{0,100\}$\\
removed set $\mathcal I_R$ & $\{75\}$, $|\mathcal I_R|=1=K-1$ & $\{95\}$, $|\mathcal I_R|=1=K-1$\\
$\Delta T_1=\max_k(\tau_k-\tau_{k-1})$ & $\max\{75,25\}=75$ & $\max\{95,5\}=95$\\
$\Delta T_2=\max_{k\in\mathcal I_R}w_k$ & $75$ & $95$\\
$h_k$, $|D_k|$ & $25$, $25$ & $5$, $5$\\
collapsed argument $\Delta T_1+\Delta T_2$ & $150$ & $190$\\
actual coarse span $\Lambda_R=\max_{k}(h_k+w_k)$ & $100$ & $100$\\
\bottomrule
\end{tabular}
\end{center}
So $\Delta_K$ of \eqref{eq:prop_removal} carries $e^{\mu^{+}\Delta T_1}$ times two envelopes with
arguments $75$ and $150$, i.e.\ a leading factor $e^{300\mu^{+}}$ for FULL and $e^{380\mu^{+}}$ for
DENSE. Proposition~\ref{prop:removal} therefore covers two of FULL's fifteen stages, one of which it
covers vacuously. \readthis{two corrections to an earlier version of this paragraph, which reported
$\Delta T_2=100$ and a factor $e^{100\mu^{+}}$. $\Delta T_2$ is the offset of the removed node to its
retained left neighbour, which is $75$ (FULL) and $95$ (DENSE), not $100$; and the composite argument
$\Delta T_1+\Delta T_2$ is $150$ and $190$, so the earlier factor was understated, not overstated.
The \emph{actual} elapsed time in the primary bound \eqref{eq:prop_removal_node} never exceeds
$\Lambda_R=100$ in either case: the gap between $100$ and $150$ is entirely the price of collapsing
$h_k$ and $w_k$ to independent maxima in Corollary~\ref{cor:removal_global}. On these transitions the
nodewise bound carries $e^{\mu(t_i-75)}\le e^{25\mu^{+}}$ and residual arguments at most $100$,
against the collapsed $e^{75\mu^{+}}$ and $\bar\rho_{150}$.}
```

#### (M) The plateau sentences, withdrawn [replaces the two blow-up passages of R3 block (M)]

```latex
% --- \S2.1 (A0) readthis: replaces "The implication runs one way only: ... outside $P$." ---
The analysis is a statement about $X\times P$, and the numerical plateau lies outside it. Where the
implemented loss returns its flat blow-up value, the \emph{numerical} trajectory produced by the
integrator has exceeded $10^3$ in some component or returned NaN; this manuscript's bounds say
nothing about such an evaluation, in either direction. In particular it is \emph{not} claimed that
the exact flow left $X$, nor that the candidate is outside $P$: the threshold $10^3$ in
\texttt{ms\_loss} is an implementation constant chosen to keep the optimiser numerically alive, it
has never been related to $X$, the observation is about a Tsit5 trajectory and not about the exact
flow, and (A1) and the event $E_X$ are unverified for the realised experiment
(P1 in \S\ref{sec:prov}). Recovering \emph{any} inference from the plateau would require all four:
a verified $E_X$, a verified (A1), a proof that $X\subset\{\norm x_\infty<10^3\}$, and a controlled
bound on the integrator's error. None is available here.

% --- Remark~\ref{rem:impl}(iv): same withdrawal ---
\item The blow-up replacement is outside the analysis entirely: where it fires, the loss returned is
not $\sum_i\norm{\flow{\cdot}{p}{\cdot}-y_i}^2$ for any flow this manuscript bounds, so no statement
of \S\ref{sec:lemmas}--\S\ref{sec:removal} applies to that evaluation. No conclusion about $p$, $P$
or $\partial P$ is drawn from it; see the (A0) discussion in \S\ref{sec:defs}.
```

#### (N) §7 — gates unchanged, one separate premise check [replaces R3 block (O); the G1--G10 bullets of `report.tex` are unchanged]

```latex
% --- abstract: replaces "ten gates verify the inheritance (\S\ref{sec:prov})." ---
ten gates verify the inheritance (\S\ref{sec:prov}); one theoretical-premise check --- the closure
condition (A1) at $\Delta T_{\max}=100$ --- fails, and is reported there.

% --- \S\ref{sec:prov}, after the G1--G10 list and the "Outcome:" line ---
\paragraph{Open theoretical-premise checks.}
The following is \emph{not} a gate and is not counted among G1--G10: those ten are mechanical
inheritance checks on files, hashes, figures and regenerated numbers, all of which pass, and they are
the ones recorded in \texttt{gates\_summary.json}. The item below is a premise of the \emph{theory}
of \S\ref{sec:defs}--\S\ref{sec:removal} that this report has not established, and it is listed
separately so that a failing premise is not mistaken for a broken inheritance check, and so that no
future revision ``repairs'' it by weakening it.
\begin{itemize}[nosep]
\item \textbf{P1 (premise check --- \emph{fails}).} The closure condition (A1),
\eqref{eq:closure}--\eqref{eq:closure_two}, has not been verified for any $\kappa$. Verifying it
requires instantiating a tube radius $r_X$, recomputing $\mu$ and $\tilde L$ as suprema over the
resulting $X\times P$ (the values $\mu=\nmu$, $\tilde L=\nLtilde$ used in this report were measured
over R002's probe box, which is not $X\times P$), fixing $\operatorname{diam}(P)$, and exhibiting
$r'>0$ satisfying both (A1-i) and (A1-ii). None of these has been instantiated numerically. With
$\mu>0$ the binding condition is (A1-i), $Q(\Delta T_{\max})\operatorname{diam}(P)<r_X$, and the
order-of-magnitude table in \S\ref{sec:defs} indicates it is unsatisfiable at $\Delta T_{\max}=100$
for any tube of usable radius. Consequently every statement of
\S\ref{sec:lemmas}--\S\ref{sec:removal} is conditional on an event, $E_X$, whose probability this
report does not bound numerically, and the confinement machinery is claimed only for short windows.
Reported as a failure, not as a caveat.
\end{itemize}
```

---

## Section 3 — back to you

This is the penultimate round of a five-round loop; please flag which remaining items, if any, you
would block on versus accept as minor.

```
Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
```
