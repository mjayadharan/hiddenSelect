# FINAL_REVISION — critique session 01, artifact R004 `report.tex`

**Artifact revised:** `/Users/manu_jay/git_repos/hiddenSelect/reports/2026-09-11-R004-revised-manuscript/report.tex`
(rebuilt: `report.pdf`, 48 pages; `report.md`, full-content Markdown rendition;
`analysis/results/gates_summary.json`, **10 of 10 gates pass**).

**Session:** 5 rounds with GPT via the local Codex CLI; final verdict at round 5 `ISSUES_REMAIN`
(3 blocking + 5 minor), all eight applied here. Rounds 1–4 raised 21 + 12 + 11 + 11 issues; every
one was Accepted, three of them with a *partial* defence on a sub-point that GPT subsequently
accepted as sound.

**Authoritative LaTeX used:** the consolidated table of `R5_to_gpt.md` §2.3 (read R2 → R3 → R4 → R5,
latest block wins), amended by GPT's round-5 issues 1–8.

**Structural note.** Lemma 0 (`lem:noescape`) and `cor:confined` are placed in §3.1, immediately
after the *local* Lemmas 1′/2′ whose hypotheses they discharge, rather than in §2.1 where the
R4 block header put them. This is the order the accepted (S1)→(S2)→(S3) argument requires; (A1)
in §2.1 carries a forward pointer. No content changed.

**Final theorem numbering (from `report.aux`).**
Definition 1 `def:Tx`.
Lemma 1 `lem:trunc`, Lemma 2 `lem:vi`, Lemma 3 `lem:vi_local`, Lemma 4 `lem:rhobar`,
Lemma 5 `lem:state`, Lemma 6 `lem:param`, Lemma 7 `lem:noescape`, Lemma 8 `lem:residual`.
Proposition 1 `prop:cost`, Proposition 2 `prop:perr`, Proposition 3 `prop:removal`.
Theorem 1 `thm:main`.
Corollary 1 `cor:confined`, Corollary 2 `cor:removal_rho`, Corollary 3 `cor:removal_node`,
Corollary 4 `cor:removal_global`, Corollary 5 `cor:basin`, Corollary 6 `cor:repartition`,
Corollary 7 `cor:window`.
Remark 1 `rem:reconnect`, Remark 2 `rem:rhoplus`, Remark 3 `rem:impl`, Remark 4 `rem:repart_numbers`,
Remark 5 `rem:envelopes`, Remark 6 `rem:weighted`. Algorithm 1 `alg:gp`.
`analysis/make_report_md.py`'s hard-coded `thm` dict was updated to exactly this set.

---

## Addressed

### Round 5 (GPT's final list — 3 blocking, 5 minor; all applied)

| # | Issue | Fix and where it landed |
|---|---|---|
| R5-1 | **BLOCKING** — the FHN "enlarging `r_X` buys admissibility" calculation uses non-global constants | The whole inflation calculation (including "$r_X>Q(\Delta T_{\max})\operatorname{diam}(P)\approx5.86\cdot10^{51}\operatorname{diam}(P)$ works") is **deleted**. (A1) `\readthis` item (ii) (§2.1) now says feasibility must be determined **jointly** from $\mu(X\times P)$, $\tilde L(X\times P)$, $r_X$ and $P$, that no general conclusion follows from monotonicity alone, and that $\mu=1.17$, $\tilde L=10.56$ **cannot be held fixed while $X$ grows** — $\partial f/\partial p$ carries cubic monomials so $\tilde L(X)$ grows without bound, and $\mu$ is not uniform over $P$. Both earlier versions of the item (the "strictly increasing, a larger tube necessarily loses" one and the "satisfiable by inflation" one) are explicitly withdrawn. |
| R5-2 | **BLOCKING** — Corollary 3 still used the obsolete loose Proposition 3 formula | `cor:repartition` (Corollary 6), `\eqref{eq:repart}`: $\Delta^{C\to S}=n^{C}_{\max}|C\setminus S|e^{\mu^{+}\Delta T_1^{C}}\bar\rho_{\Delta T^{S}}[\bar\rho_{\Delta T_1^{C}+\Delta T^{S}}+\bar\rho_{\Delta T_1^{C}}]$ (leading factor $n^C_{\max}$, **not** $2n^C_{\max}$; two envelopes kept separately). $\Delta_{A,B}=\Delta^{C\to A}+\Delta^{C\to B}$ unchanged in form; new `\eqref{eq:repart_true}` gives the specialisation at $\pstar$; `\eqref{eq:repart_disp}` gains the $\sup_U$ form via Lemma 4(iii); a `\readthis` records that the discarded relaxation is still valid but strictly weaker and inconsistent with the inherited Proposition 3. |
| R5-3 | **BLOCKING** — (R) is insufficient for the localised Proposition 2 | Remark 1 (`rem:reconnect`) item (ii) now states the result-specific variant **(R$^{\pstar}$) = (R) together with $\pstar\in U$**, with the reason (the inequality is applied at $q=\pstar$); item (i), basin retention, explicitly keeps the weaker (R). Proposition 2's `\readthis` (third caveat) and P2 in §8 both repeat the distinction. |
| R5-4 | MINOR — ranking by the signed exact gap | §4.1 `\readthis` item (2): **rank by $\big|\hatJ_K(p)-J_K(p)\big|$**, with the reason (the theory controls the absolute value; every consumer needs a two-sided bound; a large negative gap is a larger perturbation). A deliberate preference for large negative changes is named as a *separate heuristic, unrelated to minimising the perturbation bound*. |
| R5-5 | MINOR — the artifact contradicted itself about (A3) | (A3) is **kept and listed consistently**, under the descriptive name *uniform random strong-convexity assumption*: its own paragraph in §2.1, an entry in the §2.1 label list, a named member of scope (S4), the hypothesis of Proposition 2's expectation form, and cited in the Figure 20 caption. The consolidated table's "no (A3) exists" line is not followed. |
| R5-6 | MINOR — equality cases of $\bar\rho=\rho^{+}$ omit $\tilde L=0$ | Remark 2 (`rem:rhoplus`): equality **exactly** when $\mu\ge0$, or $s=0$, or $\norm{\eta}_{\max}=0$, or $Q(s)\norm{p-\pstar}=0$ (i.e. $\tilde L=0$ **or** $p=\pstar$); the bracketed proof was rewritten around $Q(s)\norm{p-\pstar}$ rather than $p-\pstar$. |
| R5-7 | MINOR — P1's narrative overstated the indicative constants | P1 (§8) now says only that inserting the probe-region constants into (A1-i) **produces enormous factors for long windows, which motivates but does not establish** the infeasibility of a useful admissible region. The undefined "at any tube radius of usable size" qualifier is gone; the same wording is used in the abstract and in (A1) `\readthis` item (iv). |
| R5-8 | MINOR — (R) alone does not reconnect everything | Remark 1's closing `\readthis` lists three further qualifications, each unverified: **(a)** the realised event $E_X$; **(b)** the penalty — *stationarity of the penalised numerical objective does not imply the stationarity of $J_K$ that (R) asserts*, the gradients differing by $\nabla G$ (Remark 3(ii), under (C1)–(C2)); **(c)** the fixed-step Tsit5 discretisation, versus the exact flow the bounds are written for. P2 in §8 repeats (b) verbatim. |

### Round 4 (11 issues, all Accepted)

| # | Issue | Fix and where it landed |
|---|---|---|
| R4-1 | constrained minimisers vs the unconstrained code | New Lemma 3 `lem:vi_local` + Remark 1 `rem:reconnect` with hypothesis (R), §2.1; new premise check **P2**, §8; the "γ = 0.05 shrinkage makes ∂P solutions typical" sentence withdrawn in both places (Lemma 2's `\readthis`, Corollary 5's `\readthis`). |
| R4-2 | reachable-state clause of `cor:confined` false | Corollary 1's blanket clause deleted; replaced by the same-parameter-continuation sentence, with "No statement is made about \eqref{eq:conf_param} for launches from a reachable state." Proposition 3's proof argues pairwise from Lemma 7 at each admissible launch node + convexity of $X$ + local Lemma 5. Remark 5's justifying sentence rewritten (§5). |
| R4-3 | Theorem 1's constant branch off by 2 | `\eqref{eq:DeltaK_flat}`: $\Delta_K=8c_0E^2$; downstream $\norm{\hat p-p}\le4E\sqrt{2c_0/m}$ and (B2) $E<(r/4)\sqrt{m/(2c_0)}$; the $p=\pstar$ consistency check is displayed beside it. |
| R4-4 | node-selection score not observable | §4.1 `\eqref{eq:score}` $S(R;p)$, built only from integrator outputs; the $\rho$-based `B(R;p)` withdrawn. |
| R4-5 | primary bound not the sharpest Grönwall bound | Proposition 3 is now **per datum**: `\eqref{eq:prop_removal_datum}` with the per-datum residual $r_{\sigma,t}$ of `\eqref{eq:rdatum}`; Corollaries 2, 3, 4 are the three successive enlargements. |
| R4-6 | P1 called a failure although never evaluated | P1 relabelled **not evaluated** (§8); abstract and (A1) item (iv) updated; "certifies short windows only" withdrawn. |
| R4-7 | $\Lambda_R$ is not a new horizon | `\eqref{eq:DTmaxdef}` reverted to the partition-based form with "the coarse partitions included"; $\Lambda_R$ kept as notation in `\eqref{eq:LambdaR}`, with the proof that it is the coarse window spanning a removed run. |
| R4-8 | enlarging $r_X$ claim false in general | (A1) item (ii) — superseded again by R5-1 above. |
| R4-9 | envelope degenerate cases | Lemma 4(v) restricted to $s>0$ with $\bar\rho_0=2\norm\eta_{\max}$ stated separately; Remark 2's equality cases (superseded again by R5-6). |
| R4-10 | negative linearised exponents conflated with $\mu<0$ | Remark 2(ii): FHN has $\mu=1.17>0$, so $\rho=\bar\rho=\rho^{+}$ identically and **no experiment in this report exhibits the $\bar\rho$ branch**; the motivating example is a uniformly contractive field, given by construction and labelled an illustration; Table 2 is no longer cited there. |
| R4-11 | greedy cost accounting omits the candidate count | §4.1 "Cost" paragraph: per candidate $O(|R|)$, per greedy step $O(C|R|)$, whole schedule $O(C|\mathcal I_R|^2)$, with caching as a remark. |

### Round 3 (11 issues, all Accepted)

| # | Issue | Fix and where it landed |
|---|---|---|
| R3-1 | new (A0) invalidated Lemmas 1, 2 | Definition 1 `def:Tx` (confinement time) and the **local** Lemmas 5, 6, asserted only on $[0,T_X]$; §3.1. |
| R3-2 | the four scopes must be distinguished everywhere | "Scope of statements" paragraph, (S1)–(S4), §2.1; every theorem environment opens with its tag. |
| R3-3 | $\bar\rho\to\rho^{+}$ violated the tightness requirement | $\bar\rho$ is primary throughout (`\eqref{eq:rho}`, Lemma 4); $\rho^{+}$ demoted to Remark 2; $\Delta_K$ described as *piecewise* quadratic. |
| R3-4 | nodewise Proposition 3 still loose | per-datum offsets and the two envelopes kept separately (superseded/extended by R4-5). |
| R3-5 | envelope remark did not cover every pair | Remark 5: the class is *confined pairs*, `\eqref{eq:envelopeM}`–`\eqref{eq:envelopeQ}`, with $\mathcal Q$'s two parameters made explicit. |
| R3-6 | blow-up implication still false | (A0) `\readthis` and Remark 3(iv): nothing is inferred from the plateau in either direction; the four things a valid inference would need are listed. |
| R3-7 | Corollary 2's interiority discussion wrong | Corollary 5: stationarity follows from (B0) alone; the redundant $\bar B\subseteq\operatorname{int}P$ removed; the genuine cost (positive clearance $r$ from $\partial P$, so boundary minimisers are **not** covered) stated. |
| R3-8 | terminal-transition $\Delta T_2$ arithmetic wrong | §2.3 terminal-transition table: FULL $75\to100$ with $\Delta T_1=\Delta T_2=75$, $h_k=|D_k|=25$, $\Delta T_1+\Delta T_2=150$, $\Lambda_R=100$; DENSE $95\to100$ with $95/95$, $h_k=|D_k|=5$, $190$, $\Lambda_R=100$. |
| R3-9 | "marginal contribution" ignores interactions | §4.1 `\readthis` item (3): there is no set-independent per-node score. |
| R3-10 | new gate number collided with G1–G10 | The premise checks are **not** gates: a separate "Open theoretical-premise checks" paragraph in §8, outside `gates_summary.json`. |
| R3-11 | closure-condition commentary misstated the failure | (A1) split into (A1-i) (binding, contains no $r'$) and (A1-ii), `\eqref{eq:closure_two}`. |

### Round 2 (12 issues, all Accepted)

| # | Issue | Fix and where it landed |
|---|---|---|
| R2-1 | (A0) incompatible with unconditional Gaussian expectations | (A2-a)/(A2-b), the admissible event $E_X$ `\eqref{eq:EX}`, tail bound `\eqref{eq:tail}`, Lemma 1 `lem:trunc`; expectations conditional on $E_X$ under (A2-b). |
| R2-2 | (A0) omitted the clean states Lemma 3 / Proposition 1 need | (A0) fixes the tube $\mathcal T$ `\eqref{eq:tube}` around the true orbit with $\mathcal T\subseteq X$; Lemma 7 confines clean **and** noisy launches. |
| R2-3 | optimisation domain / stationarity inconsistent | `\eqref{eq:minimisers}` (all minimisers over the compact $P$) and Lemma 2 `lem:vi`; $\nabla J_K(p^{(K)})=0$ removed from every proof. |
| R2-4 | nesting criterion and transition counts wrong | `\eqref{eq:nesting}`: nested iff $\kappa_a\mid\kappa_b$ **or** $\kappa_b=M$. FULL 2 of 15, DENSE 2 of 43, SHORT 4 of 6, COARSE 3 of 3, JUMP 1 of 1. §2.3, plus the §4 opening `\readthis` replacing "which is what the algorithm does". |
| R2-5 | Corollary 1(iii) false as a blanket substitution | Item (iii) removed; Corollary 7 keeps only the two local forms; Remark 5 states what a licensed substitution would require. |
| R2-6 | common-refinement example miscounted its node sets | Remark 4: $A$ 51, $B$ 35, $A\cap B$ 18, $C$ 68 nodes (terminal node included), $|C\setminus A|=17$, $|C\setminus B|=33$. |
| R2-7 | "as $\Sigma\to0$, uniformly" ill-defined | `\eqref{eq:JK_unif}`: explicit rate in $\norm\eta_{\max}$, uniform on $P$. |
| R2-8 | $\rho_s$ undefined at $\mu=0$ | `\eqref{eq:Qdef}`: $Q(s)$ with the continuous extension $\tilde Ls$. |
| R2-9 | $\Delta_K$ not a quadratic polynomial | `\eqref{eq:DeltaK_quad}`: piecewise homogeneous quadratic, branch condition stated, supremum still computable at $\delta=R_U$. |
| R2-10 | penalty-bias bound lacked its assumptions | (C1)–(C2) and `\eqref{eq:penbias}` with proof, Remark 3(ii). |
| R2-11 | low-noise-node design rule unsupported | §4.1: "prefer low-noise nodes" is **not** implementable ($\norm{\eta_{\tau_k}}$ unobservable); the implementable surrogate is a small measured $g_k$. |
| R2-12 | two stale narrative claims | Both plateau/"boundary of $P$ made visible" passages withdrawn ((A0) `\readthis`, Remark 3(iv)). |

### Round 1 (21 issues, all Accepted)

| # | Issue | Fix and where it landed |
|---|---|---|
| R1-1 | sample indexing inconsistent | One convention in §2.1: $N$ samples $y_0,\dots,y_{N-1}$, $N-1$ intervals and residuals, $\sum_k n_k=N-1$ exactly; every count $N$ in Propositions 1–2 became $N-1$ (`\eqref{eq:JK_true}`, `\eqref{eq:JK_expect}`, `\eqref{eq:perr}`), and §1's `\eqref{eq:loss_func_1}`–`\eqref{eq:loss_fun_2}` follow it. |
| R1-2 | optimised cost is not the theoretical cost; $x_0$ is a dummy | §2.3 `\eqref{eq:Jkappa}` reproduces the implemented loss exactly, with the $x_0$-dummy `\readthis` (block-diagonal Hessian, the $\kappa$-independent parabola) and Remark 3. |
| R1-3 | the $\kappa$-schedule is not a sequence of node removals | See R2-4. |
| R1-4 | constants not valid for the candidate models | (A0): all constants are suprema over $X\times P$ (`\eqref{eq:mu}`); the $p$-dependence of $J_f$ for the cubic library spelled out. |
| R1-5 | mean-value step needs a convex region | $X$ and $P$ convex in (A0); chords in $X$ via Definition 1 and Lemma 7. |
| R1-6 | Lemma 2 divides by $\norm\delta$ at $\delta(0)=0$ | Lemma 6's proof uses the upper right Dini derivative, with the $\varepsilon$-regularisation recorded as an alternative and its $|\mu|\varepsilon$ remainder. |
| R1-7 | $\rho_s$ not monotone for $\mu<0$ | $\bar\rho_s=\max\{2\norm\eta_{\max},\rho_s\}$, `\eqref{eq:rho}`, Lemma 4. |
| R1-8 | crude cost bounds reverse for $\mu<0$ | $\mu^{+}=\max(\mu,0)$ wherever an exponent is enlarged to a window length. |
| R1-9 | anisotropic noise | (A2): $\eta_i\sim$ i.i.d. with $\operatorname{Cov}=\Sigma$ diagonal, $\mathbb E\norm{\eta_i}^2=\operatorname{tr}\Sigma$; the isotropic case is a specialisation. (Partial defence on the 4.9 statistic — see *Defended*.) |
| R1-10 | $\hatJ_K$ malformed | `\eqref{eq:Jhat}` over $R\setminus\{K\}$ with $r^{+}$; §4. |
| R1-11 | isolated vs consecutive removals | Proposition 3's proof: the half-open convention handles both apparent off-by-one cases explicitly. |
| R1-12 | Figure 10's $\Delta T_1/\Delta T_2$ caption | Figure 10 caption rewritten (panel (a): $\Delta T_1=\Delta t$ throughout, $\Delta T_2=(m-1)\Delta t$ grows), with the CSV's own mislabelling called out. (Partial defence on panel (b) and G7 — see *Defended*.) |
| R1-13 | Corollary 1 not a valid consequence | See R2-5; `\eqref{eq:Mk}` distinguishes the transition bound $M_k$ from the endpoint exponent $e^{\Lambda_k}$. |
| R1-14 | weighted-norm extension omits changed constants/costs | Remark 6 `rem:weighted`; $\kappa(D)$ renamed $\mathrm{cond}(D)$ everywhere. |
| R1-15 | abstract compares quantities in different norms | Abstract rewritten with the R2 block (a) sentence: $e^{L\Delta T}=1.8\times10^{13}$, $e^{\mu\Delta T}=1.2\times10^{5}$ at $\Delta T=10$; linearised endpoint factors 34.8 ($\Delta T=5$), 139 ($\Delta T=10$) Euclidean; 8.04 weighted at $\Delta T=5$, i.e. $8.04\,\mathrm{cond}(D)=28.5$ Euclidean; observed peak 8.5. |
| R1-16 | Theorem 1 conditional, does not establish basin tracking | Theorem 1 relabelled *a posteriori, conditional*; Corollary 5 `cor:basin` added with (B0)–(B2). |
| R1-17 | expectation Proposition 2 suppresses a random premise | (A3) — see R5-5. |
| R1-18 | uniform convergence on bounded parameter sets | See R2-7. |
| R1-19 | $\pstar=\arg\min J_K^\star$ asserts identifiability | Membership, not equality, with an explicit no-identifiability-claimed `\readthis` (§2.1). |
| R1-20 | $\mu_-$ cannot support a global lower bound | §2.2 `\readthis`: the global lower logarithmic norm of FHN is $-\infty$; $-2.96$ is an orbit measurement. |
| R1-21 | numerical checks masquerading as premise validation | §6.6 retitled "Empirical diagnostics for the premises of the theory" (`sec:premises`); Figure 20 caption relabels panel (a) as an empirical diagnostic and panel (b) as a *necessary condition for, not a proof of*, basin membership; G8's bullet softened to "no probe violates the bound". (Partial defence on the Figure 14(a)/Figure 20(a) framing — see *Defended*.) |

### Also applied from the revision brief (not separate GPT issues)

- **Abstract**: assumptions "stated in full"; "each change is **local to one proof step**"; the R2 block (a) amplification sentence; **ten inheritance gates pass**; **two theoretical-premise checks P1, P2 not evaluated**; every statement conditional at every stage of the schedule.
- **§8 gate bullets**: G7's description corrected to what `verify_R004.py` actually computes (Lemma 5 / Lemma 6 factors, $\mu\le L$ and the two factor comparisons at $s\in\{1,2,5,10\}$ — **not** the Proposition 3 product over $(\Delta T_1,\Delta T_2)$ pairs); G8's corrected to the two-sided state-ratio check plus the parameter-ratio check at $2\times2424$ probe points; G2/G3/G5 wordings aligned with the code.
- **Table 2 caption**: relabelled *linearised endpoint exponents along the true orbit at $\pstar$*, with an explicit `\readthis` that these are **not** rigorous bounds for the propositions ($M_k\ge\max\{1,e^{\Lambda_k}\}$; Proposition 3 needs the envelope $\mathcal M$).
- **Symbol renames**: parameter dimension $m\to n_p$; condition number $\kappa(D)\to\mathrm{cond}(D)$; $m$ reserved for strong convexity, $\kappa$ for the window size; $\Delta T$ vs $\Delta T_1$ vs $\Delta T_2$ distinguished in a `\readthis` and in the symbols table.
- **Notation self-sufficiency**: §2.1 opens with a one-place list of every label ((A0)–(A3), (S1)–(S4), (B0)–(B2), (C1)–(C2), (R)); §2.4's symbols table gained rows for $d$, $n_p$, $\Delta T_{\max}$, $\Lambda_R$, $X$, $P$, $\mathcal T$, $r_X$, $r'$, $E_X$, $\Sigma$, $\mu^{+}$, $Q$, $\rho$, $\bar\rho$, $\rho^{+}$, $r_{\sigma,t}$, $\Lambda_k$, $M_k$.
- **§7 Discussion** rewritten to state the conditionality and to add the unevaluated premise checks as the fourth limitation.
- **`analysis/make_report_md.py`**: `thm` dict and the header sentence updated to the final label set and numbering.

---

## Defended (round 2 partial defences; GPT accepted all three as sound)

1. **Issue 9 — the 4.9 noise-slack statistic.** GPT said the quoted noise slack was "undefined or wrong for the actual dataset". Defended: `compute_R003.py` computes `noise_slack_factor = ‖η‖²_max / mean_i‖η_i‖²` directly from the realised noise array, which is a pure data statistic and indifferent to isotropy; it equals 4.86 → 4.9 and stands. What was wrong was the *prose*, which presented 4.9 as the measured value of the isotropic `1+2logN/d` rule (separately stored as 5.6). Both are now reported as different objects (`\nnoiseSlack` and `\cnoiseslackrule`, §3.2 "What changed", item (iii)).
2. **Issue 12 — Figure 10 panel (b), and gate G7.** Defended: panel (b)'s labelling is *correct* ($\Delta T_2=b\Delta t$); its CSV field `DeltaT1=(b+1)Δt` is mislabelled but the caption does not use it. And G7 never touches Figure 10 — it compares Lemma 1/Lemma 2 factors at $s\in\{1,2,5,10\}$ — so no gate was invalidated by the caption error. The self-raised defect *was* fixed: §8's **description** of G7 was wrong and is rewritten.
3. **Issue 21 — Figure 14(a) / Figure 20(a).** Defended: that panel is not presented as validation of anything; its stated conclusion is that the premise is **not verified** (36 % of cells positive definite), and the discussion already said Theorem 1 explains rather than certifies. The fault was a section title and three verbs ("checks", "verified", "i.e."), all relabelled; it was not conceded that negative results were dressed up as positive ones.

---

## Unresolved / carried as open premise checks

Nothing from GPT's lists was dropped. Two classes of item could not be *closed* in a revision of the
manuscript, because closing them requires new numerics that this report is forbidden to run
(R004 contains no new computation; its role is inheritance from R002/R003). They are declared
**in** the manuscript as open premise checks rather than silently omitted:

1. **P1 — the closure condition (A1) is not evaluated.** Evaluating it requires instantiating a tube
   radius $r_X$, a set $P$ with a diameter, recomputing $\mu$ and $\tilde L$ as suprema over the
   resulting $X\times P$ (the quoted 1.17 and 10.56 are R002 probe-box measurements at $\pstar$ and
   are *not* those suprema), and exhibiting $r'>0$ satisfying (A1-i) and (A1-ii). None exists.
   Status in the report: **not evaluated** (explicitly *not* "failed"), §8.
2. **P2 — the reconnection hypothesis (R) is not evaluated.** It needs a constructed $P$, a convex
   $U\subseteq P$ containing the reported optimum in its interior, stationarity of $J_K$ there, and
   $m$-strong convexity on $U$ (plus $\pstar\in U$ for (R$^{\pstar}$)); together with the realised
   event $E_X$, the penalty qualification and a discretisation-error bound. None exists. §8.

Two further items are limitations of the present artifact rather than unaddressed criticisms, and
are recorded here for the next revision:

- The envelopes $\mathcal M(h)$, $\mathcal Q(h)$ of Remark 5 are defined but **not computed
  anywhere**; the manuscript says so. Computing them is the natural next numerical task.
- The order-of-magnitude values $Q(1)=20.1,\dots,Q(100)=5.86\cdot10^{51}$ in (A1) item (iii) are
  written as literals in the prose, not as generated macros. They are reproducible in one line from
  $\mu$ and $\tilde L$ (which *are* generated macros), and the surrounding text marks them
  *indicative only*; `tables/numbers.tex` was deliberately left untouched so that gate G5's
  byte-identity check remains meaningful.

---

## Build state

```
$PY analysis/make_tables.py        # regenerated, byte-identical
pdflatex report.tex   (x2)         # 48 pages, no undefined references, no overfull box > 10pt
$PY analysis/make_report_md.py     # report.md, 20 figures, 5 tables, all \ref resolved
$PY analysis/verify_R004.py        # 10/10 gates pass
pdflatex report.tex                # final, gate status inlined
rm -f report.aux report.out report.toc
```
