# R003 — Sharper error bounds for multiple shooting with the same machinery

*Original bounds of the manuscript, the proposed replacements, and how much each change buys.*
Folder `reports/2026-09-11-R003-tighter-error-bounds/`, 11 September 2026. Derived view of `report.tex` (the `.tex`/`.pdf` is authoritative).

## Summary

The manuscript *Sparse Optimization using Multi-Shooting and Guess Propagation* bounds everything by $e^{Lt}$ with $L$ a Lipschitz constant of the right-hand side. On the FitzHugh–Nagumo (FHN) system the measured amplification of an initial-state error never exceeds 8.5, while $e^{Lt}$ reaches $10^{13}$ at $t=10$ (report R002, Figures 8–10). Four changes, each a one-line modification of an existing proof, close most of that gap:

1. replace the Lipschitz constant $L$ by the *logarithmic norm* $\mu \le L$ (Lemmas 1, 2, Proposition 1, Theorem 1); on FHN this is $3.05 \to 1.17$ in the exponent, and $\mu$ is a *global* constant for FHN whereas $L$ grows with the region;
2. remove the attractor-size factor $C^{(2)}_k = \max\lVert y_{i+1}\rVert$ from Proposition 1, which makes its bound quadratic instead of linear in the small quantities (a factor of about 15 here);
3. state the noise terms in expectation instead of via $\lVert\eta\rVert_{\max}$ (a factor 4.9 on this data set), and sum the per-datum exponentials as a geometric series instead of taking $e^{L\Delta T}$ for all of them;
4. as a trajectory-dependent refinement, integrate the local logarithmic norm over each window; in a diagonally weighted norm most FHN windows are then *contracting* and the worst window has amplification $\le 8$ for every window length tested, which is the observed number.

What cannot be fixed with Grönwall-type arguments is also stated (§5). All numbers are recomputed from the R002 result files copied into `external_data/`; the eight gates in `analysis/verify_R003.py` all pass (§6). Table 1 and Figure 1 carry the comparison.

## Notation in one paragraph

$\dot x = f(x;p)$, $x\in\mathbb R^d$, true parameter $p^\star$, flow $\varphi_f(t;p,x_0)$; data $y_i = x_i+\eta_i$ at times $t_i$, $i=0,\dots,N$, spacing $\Delta t$; shooting nodes $\tau_k$ with window length $\Delta T=\max_k(\tau_k-\tau_{k-1})$; $J_K(p)=\sum_k\sum_{t_i\in[\tau_{k-1},\tau_k]}\lVert\varphi_f(t_i-\tau_{k-1};p,y_{\tau_{k-1}})-y_i\rVert^2$; $J_f=\partial f/\partial x$. $L=\sup\lVert J_f\rVert_2$ is the Lipschitz constant in $x$, $\tilde L=\sup\lVert\partial f/\partial p\rVert_2$ the one in $p$. The *logarithmic norm* (one-sided Lipschitz constant, Dahlquist) is

$$\mu := \sup_x \lambda_{\max}\Big(\tfrac12\big(J_f(x)+J_f(x)^\top\big)\Big), \qquad \text{equivalently}\qquad \langle f(x)-f(z),\,x-z\rangle \le \mu\lVert x-z\rVert^2. \tag{1}$$

Always $-L\le\mu\le L$; unlike $L$, $\mu$ can be zero or negative. FHN constants measured in R002 along the true orbit: $L=3.05$ ($5.34$ on the box $[-2.5,2.5]\times[-1,2]$), $\mu=1.17$, $\tilde L=10.56$, $\lVert\eta\rVert_{\max}=0.159$, $N=101$, $d=2$.

## 1. Lemma 1: sensitivity to the initial state

**Original.** $\lVert\varphi_f(t;p,x_1)-\varphi_f(t;p,x_2)\rVert\le e^{Lt}\lVert x_1-x_2\rVert$, proved from $\lVert\dot\delta\rVert\le L\lVert\delta\rVert$ for $\delta=\hat x_1-\hat x_2$.

**Proposed.** $\lVert\varphi_f(t;p,x_1)-\varphi_f(t;p,x_2)\rVert\le e^{\mu t}\lVert x_1-x_2\rVert$, and from below $\ge e^{\mu_- t}\lVert x_1-x_2\rVert$ with $\mu_-=\inf_x\lambda_{\min}(\tfrac12(J_f+J_f^\top))\ge -L$. The proof differs from the original in one line: differentiate the *square* of the norm instead of the norm,

$$\tfrac12\tfrac{d}{dt}\lVert\delta\rVert^2=\big\langle\delta,\,f(\hat x_1;p)-f(\hat x_2;p)\big\rangle=\Big\langle\delta,\Big(\int_0^1 J_f(\hat x_2+s\delta)\,ds\Big)\delta\Big\rangle\le\mu\lVert\delta\rVert^2, \tag{2}$$

then $\tfrac{d}{dt}\lVert\delta\rVert\le\mu\lVert\delta\rVert$ and Grönwall gives the claim. The middle equality is the mean-value form of $f(\hat x_1)-f(\hat x_2)$ and is exact. The original proof bounds $\langle\delta,\dot\delta\rangle$ by Cauchy–Schwarz, $\lVert\delta\rVert\lVert\dot\delta\rVert\le L\lVert\delta\rVert^2$, which discards the sign information in the inner product; (2) keeps it.

**Why it is better.** Table 1 (left) and Figure 1(a). At the window lengths the paper actually uses ($\Delta T=1,2$) the proposed bound sits within a factor 1.1–1.5 of the observed peak, where the original is off by a factor 7–65; at $\Delta T=10$ the gap shrinks from $10^{12}$ to $10^{4}$. Two further points specific to FHN. First, $\mu$ is a *global* constant: the FHN Jacobian is $\begin{pmatrix}1-v^2&-1\\0.08&-0.064\end{pmatrix}$, and $1-v^2\le 1$ for every $v$, so $\mu=1.17$ on all of $\mathbb R^2$ (attained at $v=0$), while $L$ grows like $v^2$ and is already $5.34$ on the R002 box. The proposed Lemma 1 therefore holds for perturbations of any size with the same constant; the original needs a neighbourhood. Second, the exponent is still positive, so the proposed bound still grows without limit while the true amplification turns around at $t\approx 3$ (the orbit is attracting); §4 and §5 address this.

**Table 1.** Original versus proposed bounds on FHN, evaluated at $t=\Delta T$, against the largest value observed over the 24 R002 probes for $t\le\Delta T$ (Lemma 1: $x_2=x_1+10^{-3}u$, $u$ a random unit vector, at four orbit phases; Lemma 2: $p_2=p^\star+10^{-3}u$, $u\in\mathbb R^{20}$). All bounds hold on every probe (gates G4–G7). Source: `analysis/results/lemma1_table.csv`, `lemma2_table.csv`.

| $t=\Delta T$ | L1 original $e^{L\Delta T}$ | L1 proposed $e^{\mu\Delta T}$ | L1 observed peak | L2 original $\frac{\tilde L}{L}(e^{L\Delta T}-1)$ | L2 proposed $\frac{\tilde L}{\mu}(e^{\mu\Delta T}-1)$ | L2 observed peak |
|---|---|---|---|---|---|---|
| 1 | 21.1 | 3.23 | 2.93 | 69.8 | 20.1 | 3.7 |
| 2 | 447 | 10.4 | 6.92 | $1.5\times10^{3}$ | 84.9 | 6.7 |
| 5 | $4.2\times10^{6}$ | 349 | 8.48 | $1.5\times10^{7}$ | $3.1\times10^{3}$ | 12.0 |
| 10 | $1.8\times10^{13}$ | $1.2\times10^{5}$ | 8.48 | $6.2\times10^{13}$ | $1.1\times10^{6}$ | 15.1 |

(L1 = Lemma 1, ratio $\lVert\varphi(t;x_1)-\varphi(t;x_2)\rVert/\lVert x_1-x_2\rVert$; L2 = Lemma 2, ratio $\lVert\varphi(t;p_1,x_0)-\varphi(t;p_2,x_0)\rVert/\lVert p_1-p_2\rVert$.)

## 2. Lemma 2: sensitivity to the parameters

**Original.** $\lVert\varphi_f(t;p_1,x_0)-\varphi_f(t;p_2,x_0)\rVert\le\frac{\tilde L}{L}\big(e^{Lt}-1\big)\lVert p_1-p_2\rVert$.

**Proposed.** $\lVert\varphi_f(t;p_1,x_0)-\varphi_f(t;p_2,x_0)\rVert\le\frac{\tilde L}{\mu}\big(e^{\mu t}-1\big)\lVert p_1-p_2\rVert$, read as $\tilde L\,t\,\lVert p_1-p_2\rVert$ when $\mu=0$ and as $\frac{\tilde L}{|\mu|}\big(1-e^{-|\mu|t}\big)\lVert p_1-p_2\rVert\le\frac{\tilde L}{|\mu|}\lVert p_1-p_2\rVert$ when $\mu<0$. Proof: split $\dot\delta=[f(\hat x_1;p_1)-f(\hat x_2;p_1)]+[f(\hat x_2;p_1)-f(\hat x_2;p_2)]$ exactly as in the manuscript, but take the inner product with $\delta$ first: $\tfrac12\tfrac{d}{dt}\lVert\delta\rVert^2\le\mu\lVert\delta\rVert^2+\tilde L\lVert\delta\rVert\lVert p_1-p_2\rVert$, hence $\tfrac{d}{dt}\lVert\delta\rVert\le\mu\lVert\delta\rVert+\tilde L\lVert p_1-p_2\rVert$, and the manuscript's integrating-factor step finishes it.

**Why it is better.** Table 1 (right), Figure 1(b). Beyond the constant-factor gain, the proposed form is qualitatively right in a way the original cannot be: for a dissipative system ($\mu<0$) it *saturates* at $\tilde L/|\mu|$, which is exactly the shape R002 Figure 9 shows (the observed sensitivity levels off at 15.1). A Lipschitz constant is non-negative by definition, so $\frac{\tilde L}{L}(e^{Lt}-1)$ can never predict saturation. On FHN in the Euclidean norm $\mu=1.17>0$, so saturation is not yet proved by this step alone; the weighted norm of §4 gets closer.

## 3. Proposition 1 and Theorem 1: cost change under node removal

These two results compare the cost $J_K$ on a fine partition with the cost $\hat{\mathcal J}_K$ after the nodes indexed by $\mathcal I_R$ are removed. Three independent changes apply.

**(a) Substitute $\mu$ for $L$.** Both results use Lemmas 1 and 2 only through the factors $e^{L\Delta T_1}$, $e^{L\Delta T_2}$ and $\frac{\tilde L}{L}(e^{L\Delta T_2}-1)$; each becomes its $\mu$ version verbatim.

**(b) Remove the attractor-size constant.** *Original:* the proof writes the per-datum difference of squared residuals as $\lVert u-v\rVert\,\lVert u+v-2y_{i+1}\rVert$ with $u,v$ the two model predictions, then bounds the second factor by $2C^{(1)}_k+2C^{(2)}_k$ where $C^{(2)}_k=\max\lVert y_{i+1}\rVert$ is the size of the data, $2.34$ on FHN. *Proposed:* write $a=\lVert u-y_{i+1}\rVert$, $b=\lVert v-y_{i+1}\rVert$ (the two residuals) and use $a^2-b^2=(a-b)(a+b)$ with $|a-b|\le\lVert u-v\rVert$ (reverse triangle inequality) and $a+b\le 2\max(a,b)$. Near $p^\star$ each residual is bounded by Lemmas 1–2 as $e^{\mu\Delta T_1}\lVert\eta\rVert_{\max}+\lVert\eta\rVert_{\max}+\frac{\tilde L}{\mu}(e^{\mu\Delta T_1}-1)\lVert p-p^\star\rVert$, so the constant $C_{\max}$ in

$$|\hat{\mathcal J}_K(p)-J_K(p)|\le 2|\mathcal I_R|\,C_{\max}\Big(e^{\mu(\Delta T_1+\Delta T_2)}\lVert\eta\rVert_{\max}+e^{\mu\Delta T_1}\big[\tfrac{\tilde L}{\mu}(e^{\mu\Delta T_2}-1)\lVert p-p^\star\rVert+\lVert\eta\rVert_{\max}\big]\Big)$$

is now $O(\lVert\eta\rVert+\lVert p-p^\star\rVert)$ instead of $O(\lVert y\rVert)$. *Why it is better:* the right-hand side becomes *quadratic* in the small quantities, as the left-hand side is (a difference of two squared residuals of size noise), instead of linear times the attractor radius; on FHN $\lVert y\rVert_{\max}/\lVert\eta\rVert_{\max}\approx 15$, so this is the largest constant-factor loss in Proposition 1 and it propagates unchanged into Theorem 1 through $C_{\max}$. It also makes the theorem's bound on $\lVert\hat p^{(K)}-p^{(K)}\rVert$ scale like the noise, not like $\sqrt{\text{noise}\times\text{attractor size}}$.

**(c) Two free tightenings of the summation.** (i) The proofs apply $e^{L\Delta T}$ to every datum in a window, but datum $t_i$ only sees $e^{L(t_i-\tau_{k-1})}$; summing the geometric series over the $n_k$ data in a window replaces $n_k e^{2L\Delta T}$ by $\frac{e^{2L\Delta T}-1}{e^{2L\Delta t}-1}\approx\frac{e^{2L\Delta T}}{2L\Delta t}$, a saving of order $L\Delta T$ for $n_k=\Delta T/\Delta t$. (ii) Every $\lVert\eta_i\rVert$ is replaced by $\lVert\eta\rVert_{\max}$. Under the manuscript's own Gaussian model, $\mathbb E\lVert\eta_i\rVert^2=d\sigma^2$ while $\lVert\eta\rVert_{\max}^2\approx\sigma^2(d+2\log N)$, a factor $1+2\log N/d$ ($5.6$ for $N=101$, $d=2$; measured $4.9$ on the R002 data set). Moreover the node noise $\eta_{\tau_{k-1}}$ and the datum noise $\eta_i$ are independent for $t_i\ne\tau_{k-1}$, so the cross term in $(e^{L\Delta T}\lVert\eta_{\tau_{k-1}}\rVert+\lVert\eta_i\rVert)^2$ vanishes in expectation, saving a further factor up to 2. A bound on $\mathbb E\,J_K(p^\star)$ is therefore both tighter and the natural object for the strong-convexity step that follows it. A typo to fix while there: equation (JK11) in the proof has $e^{2L(\Delta T_1+\Delta T_2)}$ while the statement of Proposition 1 has $e^{L(\Delta T_1+\Delta T_2)}$.

**Figure 1** (`figures/fig01_bounds.pdf`, `.png`). (a) Lemma 1 and (b) Lemma 2 on FHN: the original Lipschitz bound (solid vermilion), the proposed log-norm bound (dashed green), and the largest observed ratio over the 24 R002 probes at each $t$ (blue). Both bounds hold at every probe; the proposed one is $10^{8}$ closer at $t=10$ but still grows, while the observation turns around because the orbit is attracting. (c) The local rates along one FHN period: the local Lipschitz constant $\lVert\partial f/\partial x\rVert_2$ (dotted) is never below 1; the Euclidean log norm $\mu(x^\star(t))$ (green) is near 0 on the slow branches and peaks at 1.17 only during the two fast jumps; in the weighted norm $\lVert Dx\rVert$, $D=\mathrm{diag}(1,3.54)$, the local log norm $\mu_D$ (purple) is *negative* ($-0.064$) for 75 % of the period. Data: `analysis/results/envelopes.csv`, `local_lognorm.csv`, `local_lognorm_D.csv`.
*What the figure shows:* the three curves in (a) and (b) are ordered original bound > proposed bound > observation at every $t>0$, with the observation flattening after $t\approx 3$; in (c) growth is concentrated in two short spikes per period and the rest of the orbit is neutral (Euclidean) or contracting (weighted).

## 4. Refinement: per-window exponents and a weighted norm

**Original.** One constant $e^{L\Delta T}$ (or $e^{\mu\Delta T}$ after §1) for every window.

**Proposed.** The mean-value form in (2) holds pointwise in time, so with $\mu(t):=\lambda_{\max}\big(\tfrac12(A(t)+A(t)^\top)\big)$, $A(t)=\int_0^1 J_f(\hat x_2(t)+s\delta(t))\,ds$, the same argument gives the exact bound $\lVert\delta(t)\rVert\le\exp\big(\int_0^t\mu(s)\,ds\big)\lVert\delta(0)\rVert$. Per window this is $e^{\Lambda_k}$ with $\Lambda_k=\int_{\tau_{k-1}}^{\tau_k}\mu(s)\,ds$, and Lemmas 1–2, Proposition 1 and Theorem 1 go through with $e^{\mu\Delta T}$ replaced by $\max_k e^{\Lambda_k}$. For small perturbations $\mu(s)$ is the log norm of $J_f$ along the reference orbit (first-order in $\lVert\delta\rVert$; for a rigorous statement take the supremum over the segment between the two trajectories). The same proof also runs in any weighted norm $\lVert x\rVert_D=\lVert Dx\rVert$ with $\mu_D$ the log norm of $DJ_fD^{-1}$, and returns to the Euclidean norm at the cost of the condition number $\kappa(D)$. For FHN the choice $D=\mathrm{diag}(1,s)$, $s=\sqrt{|J_{12}|/|J_{21}|}=3.54$, makes the off-diagonals of $DJ_fD^{-1}$ equal and opposite, so its symmetric part is diagonal and $\mu_D(x)=\max(1-v^2,\,-0.064)$: negative whenever $|v|>1$, i.e. on both slow branches, with global supremum $1.00$.

**Why it is better.** Table 2. In the weighted norm the worst window over the whole record has amplification $\le 8.04$ for *every* window length tested (multiply by $\kappa(D)=3.54$ to convert to a Euclidean statement), against the observed peak 8.5 and the a-priori $e^{\mu\Delta T}$ of $1.2\times10^5$ at $\Delta T=10$; the median window contracts. The mean log norm over the record is 0.29 (Euclidean) and 0.077 (weighted), so over a period the orbit is nearly neutral, as it must be (the phase direction has Floquet exponent zero). Two consequences for the paper. First, this is the quantitative version of "the restart at each node resets the error": the per-window exponents are what the schematic R002 Figure 7 should be drawn with. Second, it gives a node-placement rule sharper than the one after Proposition 1 ("remove nodes with small $\eta_{\tau_k}$ and small $\Delta T_2$"): place and keep nodes just before the fast jumps, where $\int\mu$ accumulates, and remove them freely on the slow branches, where every window is contracting. The price is that $\Lambda_k$ is evaluated along the fitted orbit, so the bound is a posteriori rather than a closed-form constant.

**Table 2.** Per-window amplification along the true FHN orbit ($\Delta t=1$, 101 data, windows aligned to $t_0$). "Worst" and "median" are $\max_k$ and median$_k$ of $e^{\Lambda_k}$ over the windows of that length; "contracting" is the share of windows with $\Lambda^D_k<0$. Weighted-norm values convert to Euclidean statements by a factor at most $\kappa(D)=3.54$. Source: `analysis/results/window_exponents.csv`, `window_exponents_D.csv`.

| $\Delta T$ | windows | a priori $e^{\mu\Delta T}$ | Euclidean $e^{\Lambda_k}$ worst | Euclidean median | weighted $e^{\Lambda^D_k}$ worst | weighted median | contracting windows |
|---|---|---|---|---|---|---|---|
| 1 | 100 | 3.23 | 3.10 | 1.16 | 2.60 | 0.94 | 74 % |
| 2 | 50 | 10.4 | 8.56 | 1.41 | 5.89 | 0.88 | 70 % |
| 5 | 20 | 349 | 34.78 | 3.75 | 8.04 | 0.73 | 55 % |
| 10 | 10 | $1.2\times10^{5}$ | 138.79 | 29.25 | 5.89 | 3.20 | 30 % |

## 5. What this machinery cannot do

Any a-priori bound valid for every $f$ with a given $L$ (or $\mu$) is attained by the linear system $\dot x=Lx$ (or $\dot x=\mu x$) and is therefore exponential; no rearrangement of a Grönwall argument removes the exponential. The observed plateau at 8.5 in Figure 1(a) is a property of the FHN attractor, not of any Lipschitz-type constant. Proving it needs one of: a contraction metric with $\mu_P<0$ everywhere, which does not exist on a limit cycle because the phase direction is neutral (§4 gets $\mu_D<0$ on 75 % of the period, not all of it); a Lyapunov function for the attractor, giving boundedness of perturbations but not a rate; or a Floquet analysis of the variational equation about the periodic orbit, which bounds the transverse contraction rate but requires the orbit to be known. All three are more machinery than the manuscript uses. The honest summary is: §1–§3 make the constants of the right kind (they can saturate and are global for FHN) at no cost in rigour or length; §4 makes them quantitatively close along the actual orbit at the cost of being trajectory-dependent; the remaining gap is intrinsic to worst-case analysis.

## 6. Gates and reproduction

`analysis/verify_R003.py` writes `analysis/results/gates_summary.json`.

- G1: the seven input files match the SHA-256 hashes in `INPUTS.md`.
- G2: $L$, $\tilde L$, $\mu$ recomputed from the FHN library file agree with R002's `concept_meta.json` to $10^{-3}$ relative, and $\mu$ recomputed on the R002 box equals the analytic global value $\lambda_{\max}$ at $v=0$ to $10^{-6}$.
- G3: $\mu\le L$ and the weighted-norm supremum $\mu_D=1$ to $10^{-9}$.
- G4/G5: the original and proposed Lemma 1 upper bounds hold at all $24\times101$ probe points (and the lower bounds $e^{-Lt}$, $e^{\mu_- t}$ with $\mu_-=-2.96$).
- G6/G7: same for Lemma 2.
- G8: every number quoted in this text (Table 1, Table 2, the constants macros) is regenerated by `make_tables.py` from `results/` and matches the committed `tables/*.tex` byte for byte.

Outcome: **8/8 pass**. Regenerate with `compute_R003.py`, `make_tables.py`, `figures/make_figures.py`, `verify_R003.py`, then `pdflatex` twice (see `REPRODUCE.md`).
