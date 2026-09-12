# R1 handoff to GPT — R004 revised manuscript (round 1 of an adversarial critique loop)

You get **one shot per round, no follow-up questions**. Everything you need is in this
file. Section 1 is context, Section 2 is the artifact, Section 3 is your instructions.

---

# Section 1: Context bundle

## 1.1 What the project is

**Goal.** Sparse ODE model discovery by *simulation-based* fitting. Given noisy samples
$y_i$ of a trajectory of an unknown system $\dot{\mathbf x}=\mathbf F(\mathbf x;p)$, and a
library of candidate terms $\Theta=\{\Theta_1,\dots,\Theta_J\}$ so that
$\dot x_i=\sum_j p_{ij}\Theta_j(\mathbf x)$, recover the sparse coefficient matrix $p$ by
*forward-simulating* the candidate model and minimising a squared-residual cost. This is
not SINDy-style derivative regression; the model is integrated.

**Multiple shooting, without continuity constraints.** Single shooting (integrate once
from the first datum over the whole record) has a cost landscape that acquires local
minima and blow-up plateaux as the record lengthens. So the record is partitioned into
$K$ windows by shooting nodes $\tau_0<\dots<\tau_K$, and the model is restarted at each
node. **Crucially, the node states are the noisy data themselves, $y_{\tau_k}$ — they are
NOT optimisation variables, and no continuity is enforced at window ends.** This differs
from textbook multiple shooting, where node states are decision variables tied together
by continuity constraints. Consequences: (i) the only optimisation variables are $p$
(plus a free $x_0$ in the numerics); (ii) the node state carries the node's measurement
noise $\eta_{\tau_k}$ straight into the window, which is where the $e^{\mu\Delta T}\|\eta\|$
terms come from.

**Guess propagation.** The window size $\kappa$ (in data intervals) is swept *upward*
through a schedule, and the minimiser found at one stage is used as the starting point for
the next. Short windows give a smooth, nearly convex landscape but weak discrimination
between parameters; long windows discriminate sharply but are rugged. The sweep ends at
$\kappa=100$, i.e. full single shooting. The theory's job is to show that each node
removal moves the minimiser by a *bounded* amount, so the optimiser stays in the right
basin as the landscape roughens.

**What counts as success for this revision.** Bounds that are (a) **correct**, and (b) **as
tight as Grönwall-type machinery allows**, plus a manuscript whose theory and its
inherited numerical section are mutually consistent.

## 1.2 The revision: what changed from the 2025 draft, and why

The 2025 draft (`main.tex`, reproduced in §2.4 below) bounded everything by $e^{Lt}$ with
$L$ a Lipschitz constant. Report **R003** proposed the tightenings; **R004** (the artifact
under review) is the rewritten manuscript that folds them in and adds a numerical section.

**(a) $L\to\mu$ (logarithmic norm / one-sided Lipschitz constant, Dahlquist).**
$\mu=\sup_x\lambda_{\max}(\tfrac12(J_f+J_f^\top))$, so $-L\le\mu_-\le\mu\le L$ and $\mu$ may
be zero or negative where $L$ cannot. Motivation: on FitzHugh–Nagumo (FHN) the *measured*
amplification of an initial-state error never exceeds **8.5**, while $e^{Lt}$ reaches
$1.8\times10^{13}$ at $t=10$. The FHN Jacobian is
$\begin{pmatrix}1-v^2&-1\\0.08&-0.064\end{pmatrix}$; its only state dependence is $1-v^2\le1$,
so $\mu=1.17$ is a **global** constant on all of $\mathbb R^2$, whereas $L$ grows like $v^2$
(3.05 on the orbit, 5.34 on the box $[-2.5,2.5]\times[-1,2]$). The proof change is one
line: differentiate $\|\delta\|^2$ and keep the sign in $\langle\delta,A\delta\rangle$,
instead of applying Cauchy–Schwarz to $\langle\delta,\dot\delta\rangle$.

**(b) Comparator changed from $J^\star=J_1^\star$ to $J_K^\star$.** The draft claimed to
bound the distance of the windowed cost $J_K$ to the *noise-free single-shooting* cost
$J^\star$. Its actual decomposition inserts $\varphi_f(t_{i+1};p,x_0)$ where the algebra
needs $\varphi_f(t_{i+1}-\tau_{k-1};p,x_{\tau_{k-1}})$ — i.e. the draft's identity is wrong
as written, and what its argument really bounds is the distance to the noise-free
*windowed* cost $J_K^\star$. R004 therefore defines $J_K^\star$ and uses it. Both vanish at
$p^\star$, so downstream conclusions are unchanged.

**(c) Node-removal bound made quadratic.** The draft wrote the per-datum difference of
squared residuals as $\|u-v\|\cdot\|u+v-2y_{i+1}\|$ and bounded the second factor via
$C^{(2)}_k=\max\|y_{i+1}\|$ — the size of the *attractor* (2.34 on FHN). That makes the
bound *linear* in the small quantities times the attractor radius, while the left-hand
side is manifestly quadratic in them. R004 instead writes $a^2-b^2=(a-b)(a+b)$ with $a,b$
the two **residuals** and bounds $a+b$ by two residuals, so the bound is quadratic. On FHN
this removes a factor $\|y\|_{\max}/\|\eta\|_{\max}\approx15$.

**(d) Per-window data count $n_{\max}$ restored.** The draft collapsed the inner sum over
$t_i\in[\tau_k,\tau_k^+)$ to the factor $|\mathcal I_R|$ and silently dropped the number of
data per window. R004 puts $n_{\max}$ back.

**(e) Strong-convexity constant renamed $m$.** The draft called it $\mu$, which now clashes
with the logarithmic norm.

**(f) Expectation version of the cost-at-truth bound.** $\|\eta\|_{\max}$ is replaced by
$\mathbb E\|\eta_i\|^2=d\sigma^2$ where possible (a factor 4.9 on this data set; the rule of
thumb $\|\eta\|_{\max}^2\approx\sigma^2(d+2\log N)$ gives 5.6), and the cross term between
**node** noise $\eta_{\tau_{k-1}}$ and **datum** noise $\eta_i$ is claimed to vanish in
expectation because the two are independent for $t_i\ne\tau_{k-1}$ and $\mathbb E\eta_i=0$.
Also, the per-datum exponent keeps its actual offset $j\Delta t$ instead of $\Delta T$,
turning $n_k e^{2\mu\Delta T}$ into a geometric sum.

**(g) Trajectory-dependent refinement + weighted norm.** The mean-value identity holds
pointwise in time, so $e^{\mu t}$ can be replaced by $\exp\int_0^t\mu(s)\,ds$, i.e. a
per-window exponent $e^{\Lambda_k}$. And the same proof runs in $\|x\|_D=\|Dx\|$ with
$\mu_D$ the logarithmic norm of $DJ_fD^{-1}$. For FHN, $D=\mathrm{diag}(1,s)$ with
$s=\sqrt{|J_{12}|/|J_{21}|}=\sqrt{1/0.08}=3.54$ makes the off-diagonals of $DJ_fD^{-1}$
equal and opposite, so the symmetric part is diagonal and
$\mu_D(x)=\max(1-v^2,\,-0.064)$ — **negative whenever $|v|>1$**, i.e. on both slow branches
of the relaxation oscillation. $\kappa(D)=3.54$.

## 1.3 Measured FHN constants (all frozen in R002/R003, recomputed and gated)

| symbol | meaning | value |
|---|---|---|
| $L$ | Lipschitz constant in $x$, along the true orbit | 3.05 |
| $L$ (box) | same, on $[-2.5,2.5]\times[-1,2]$ | 5.34 |
| $\mu$ | logarithmic norm (global for FHN) | 1.17 |
| $\mu_-$ | lower logarithmic norm | $-2.96$ |
| $\tilde L$ | Lipschitz constant in $p$, along the orbit | 10.56 |
| $\|\eta\|_{\max}$ | largest noise norm on the data set | 0.159 |
| noise floor | $\tfrac1N\sum\|\eta_i\|^2$ | 0.0052 |
| $N$, $\Delta t$, $d$ | samples, spacing, state dimension | 101, 1, 2 |
| $s$, $\kappa(D)$ | weight, condition number | 3.54, 3.54 |
| FHN period | | 39.5 |
| $\|y\|_{\max}/\|\eta\|_{\max}$ | attractor-over-noise | $\approx 15$ |
| noise slack | $\|\eta\|_{\max}^2/(d\sigma^2)$, measured / rule-of-thumb | 4.9 / 5.6 |

**Observed (not bounds):** Lemma-1 peak amplification **8.5** (reached near the fast jump at
$t\approx3$, then contracts because the orbit is attracting); Lemma-2 peak parameter
sensitivity **15.1** (saturates).

## 1.4 Limitations the manuscript ALREADY acknowledges — do not spend your round on these

1. **Strong convexity is not verified at the optimiser's iterates.** Figure 20 (`fig:post`)
   shows the Nelder–Mead-returned point is positive definite in only 36 % of (seed, $\kappa$)
   cells. The manuscript says so explicitly, twice.
2. **The observed plateau of the amplification cannot be proved by Grönwall-type
   arguments.** It is a property of the attractor; proving it needs a contraction metric
   (impossible globally on a limit cycle — the phase direction is neutral), a Lyapunov
   function, or Floquet analysis. Stated in §5 and in the discussion.
3. **Per-window exponents $\Lambda_k$ are first-order and a posteriori** (evaluated along a
   trajectory, not a closed-form constant). Stated in §5.
4. **Discretisation error is ignored** — the analysis uses the exact flow $\varphi_f$ in
   place of the numerical integrator. Stated in §2.1.

If you think one of these is understated or mis-scoped, say so briefly, but the fact that
they are limitations is already conceded.

## 1.5 What we specifically want scrutinised

**Proof correctness, step by step**, in Lemmas 1–3, Propositions 1–3, Theorem 1 and
Corollary 1. In particular:

- Signs and directions of every inequality.
- The **mean-value form** $f(\hat x_1)-f(\hat x_2)=A(t)\delta$ with
  $A(t)=\int_0^1 J_f(\hat x_2+s\delta)\,ds$ — is it valid, and on what region? Does it need
  convexity of the domain / $C^1$ on the segment?
- The **Dini-derivative step**: going from $\tfrac12\tfrac{d}{dt}\|\delta\|^2\le\mu\|\delta\|^2$
  to $\tfrac{d}{dt}\|\delta\|\le\mu\|\delta\|$ divides by $\|\delta\|$. In Lemma 2,
  $\delta(0)=0$ exactly. Is the argument valid at $\delta=0$? Is the handling in Lemma 1
  ("wherever $\delta\ne0$, and $\delta\equiv0$ otherwise by uniqueness") sufficient?
- The case **$\mu=0$ or $\mu<0$** in Lemma 2 and everywhere $\mu$ appears in a denominator
  or in the direction of an inequality.
- Is **$\rho_s$ really monotone increasing in $s$**? Proposition 3's proof uses this
  explicitly. Check the $\mu<0$ case term by term.
- Does the **expectation cross-term** really vanish? Check the independence claim covers
  every $(i,k)$ pair actually summed over, including $t_i=\tau_k$ (the right endpoint of a
  window, which is itself the next node).
- **Proposition 3's window bookkeeping**: the definition of $\hat{\mathcal J}_K$ in
  eq. (Jhat) (indices, half-open intervals, use of $\tau_k^+$ for $k\notin\mathcal I_R$),
  the set $D_k$, the case of **consecutive removed nodes**, and **the datum sitting at the
  removed node $\tau_k$ itself** — does it change predictor, and is it counted?
- Is **Theorem 1's strong-convexity neighbourhood assumption stated correctly** (which
  point must the neighbourhood contain, and is the conclusion circular given that
  $\Delta_K(\hat p^{(K)})$ contains $\|\hat p^{(K)}-p^\star\|$)?
- Does the **definitions section (§2) actually define everything used**? Are there symbol
  clashes? (The revision renamed one symbol specifically to avoid a clash.)

**And separately: are any inherited numerical claims inconsistent with the revised theory
as written?** The numerical section is copied from earlier reports (R002/R003) and its
figure captions were written before the theory was rewritten. Check the figure captions'
use of $\Delta T_1$ / $\Delta T_2$ / $|\mathcal I_R|$ against §2's definitions; check
whether numbers quoted in the abstract and §5 are quoted in the norm they were computed
in; check whether the cost actually optimised in the numerics (eq. Jkappa) is the cost the
theory analyses (eq. JK).

---

# Section 2: The artifact under review

Notation note: LaTeX macros have been expanded so you see numbers, not macro names.
`\flow{t}{p}{x}` is written `\varphi_f(t;\,p,\,x)`, `\norm{x}` as `\lVert x\rVert`,
`\pstar` as `p^\star`, `\hatJ` as `\hat{\mathcal J}`. Colour markup (`\rev{}`) has been
stripped to plain braces; `\readthis{}` renders as "**Read this:**".

## 2.0 Abstract

```latex
We study the discovery of sparse polynomial governing equations $\dot{\mathbf x}=\mathbf F(\mathbf x;p)$ from noisy trajectory samples by fitting the parameters $p$ of a candidate library through simulation. Fitting by a single forward simulation from the first datum (single shooting) produces a cost landscape whose local minima and blow-up regions multiply with the length of the record. We instead restart the simulation at shooting nodes placed on the data (multiple shooting without continuity constraints) and \emph{propagate the guess}: the minimiser found with short windows becomes the starting point for longer windows, ending with a full single-shooting fit. The analysis bounds the sensitivity of the flow to the initial state and to the parameters, the cost at the true parameter, the change of the cost when shooting nodes are removed, and the resulting displacement of the minimiser. This revision replaces the Lipschitz constant $L$ of the earlier draft by the logarithmic norm $\mu\le L$ throughout, removes an attractor-size constant from the node-removal bound so that it is quadratic in the small quantities, states the noise terms in expectation, corrects a missing per-window count, and adds a trajectory-dependent refinement in a weighted norm; each change is a one-line modification of the original proof. A numerical section on the FitzHugh--Nagumo system checks every bound against measurement (the amplification of an initial-state error never exceeds $8.5$, where the earlier bound gave $10^{13}$ and the revised one gives $10^{5}$ a priori and $\le 8$ per window along the orbit), maps the cost landscape as the window grows (local minima per slice from $1.25$ to $7.9$), and reports the headline experiment over $8$ optimiser seeds: with guess propagation the median parameter error at full single shooting is $0.32$ against $1.65$ without it (paired difference $-1.14$, 95\,\% bootstrap CI $[-1.43,-0.77]$, better in 7 of 8 seeds), with the same behaviour on Lotka--Volterra and, with a gradient optimiser, on Lorenz. The three hand-drawn figures of the draft are replaced by figures computed from the data. All numbers are inherited from reports R002 and R003, whose result files are copied here with hashes; ten gates verify the inheritance.
```

## 2.1 Sections 2–5 of `report.tex`, verbatim (macros expanded)

```latex
\section{Notation and definitions}\label{sec:defs}
Everything used in the statements, captions and tables is defined here, including what is inherited from the earlier draft and from reports R002 and R003; the provenance column says where each item was fixed.

\subsection{Model, flow, data and costs}
Consider $\dot x=f(x;p)$ with $x\in\mathbb R^d$ and $p\in\mathbb R^m$ (the earlier draft wrote $p\in\mathbb R$; nothing below uses a scalar parameter). $p^\star$ is the true parameter and $x^\star(t)$ the true trajectory from $x_0$. The \emph{flow} $\varphi_f(t;\,p,\,x_0)=X(t)$ is the solution of the initial-value problem at time $t$; it satisfies $\varphi_f(t_1+t_2;\,p,\,x_0)=\varphi_f(t_2;\,p,\,\varphi_f(t_1;\,p,\,x_0))$. In the analysis the flow replaces the numerical integrator, i.e.\ discretisation error is ignored. Data are $y_i=x_i+\eta_i$ with $x_i=x^\star(t_i)$ and $\eta_i\sim\mathcal N(0,\sigma^2 I_d)$ independent, $i=0,\dots,N$, spacing $\Delta t$; $\lVert \eta\rVert_{\max}=\max_i\lVert \eta_i\rVert$. Shooting nodes $t_0=\tau_0<\dots<\tau_K=t_N$ are a subset of the data times; $\tau(i)$ denotes the last node at or before $t_i$, $\Delta T=\max_k(\tau_k-\tau_{k-1})$ the longest window, $n_k$ the number of data in window $k$ and $n_{\max}=\max_k n_k$. The costs are
\begin{align}
J_K(p)&=\sum_{k=1}^{K}\sum_{t_i\in(\tau_{k-1},\tau_k]}\lVert \varphi_f(t_i-\tau_{k-1};\,p,\,y_{\tau_{k-1}})-y_i\rVert^2 &&\text{(multiple shooting, $K$ windows),}\label{eq:JK}\\
J_1(p)&=\sum_{i=1}^{N}\lVert \varphi_f(t_i-t_0;\,p,\,y_0)-y_i\rVert^2 &&\text{(single shooting),}\label{eq:J1}\\
J_K^\star(p)&=\sum_{k=1}^{K}\sum_{t_i\in(\tau_{k-1},\tau_k]}\lVert \varphi_f(t_i-\tau_{k-1};\,p,\,x_{\tau_{k-1}})-x_i\rVert^2 &&\text{(noise-free windowed cost).}\label{eq:JKstar}
\end{align}
$J_K^\star$ starts every window from the \emph{true} state and compares with the true state, so $J_K^\star(p^\star)=0$ and $p^\star=\arg\min J_K^\star$ for every partition. \textbf{Read this:} {the earlier draft compared $J_K$ with the noise-free \emph{single-shooting} cost $J^\star=J_1^\star$; the decomposition used in its proof actually bounds the distance to $J_K^\star$, which is the comparator used here. Both vanish at $p^\star$, so the conclusions are unchanged.} $p^{(K)}=\arg\min_p J_K(p)$. For node removal, $\mathcal I_R\subset\{1,\dots,K-1\}$ indexes the removed nodes, $\hat{\mathcal J}_K$ is the cost of the coarser partition, $\hat p^{(K)}$ its minimiser, and for $k\in\mathcal I_R$, $\tau_k^-$ and $\tau_k^+$ are the nearest remaining nodes to the left and right of $\tau_k$; $\Delta T_1=\max_k(\tau_k-\tau_{k-1})$ is the fine window length and $\Delta T_2=\max_{k\in\mathcal I_R}(\tau_k-\tau_k^-)$. $D_k=\{t_i\in(\tau_k,\tau_{k+1}]\}$ is the fine window launched at the removed node $\tau_k$.

\subsection{Constants of the right-hand side}
$J_f(x)=\partial f/\partial x$. The Lipschitz constants are $L=\sup_x\lVert J_f(x)\rVert_2$ (in $x$, for fixed $p$) and $\tilde L=\sup\lVert \partial f/\partial p\rVert_2$ (in $p$). The \emph{logarithmic norm} (one-sided Lipschitz constant, Dahlquist) and its lower counterpart are
\begin{equation}
\mu=\sup_x\lambda_{\max}\!\Big(\tfrac12\big(J_f(x)+J_f(x)^\top\big)\Big),\qquad \mu_-=\inf_x\lambda_{\min}\!\Big(\tfrac12\big(J_f(x)+J_f(x)^\top\big)\Big),
\label{eq:mu}
\end{equation}
equivalently $\mu_-\lVert x-z\rVert^2\le\langle f(x)-f(z),x-z\rangle\le\mu\lVert x-z\rVert^2$. Always $-L\le\mu_-\le\mu\le L$; unlike $L$, $\mu$ can be zero or negative. The suprema are over the region containing the trajectories considered; for FHN the value of $\mu$ is global (\S\ref{sec:lemma1}). For a weighted norm $\lVert x\rVert_D=\lVert Dx\rVert$ with $D$ invertible, $\mu_D$ is the logarithmic norm of $DJ_fD^{-1}$ and $\kappa(D)=\lVert D\rVert\lVert D^{-1}\rVert$ its condition number. The \emph{local} logarithmic norm along a trajectory is $\mu(t)=\lambda_{\max}(\tfrac12(A(t)+A(t)^\top))$ for the matrix $A(t)$ of the mean-value form, and $\Lambda_k=\int_{\tau_{k-1}}^{\tau_k}\mu(s)\,\mathrm{d} s$ is the window exponent. The abbreviation used in the bounds is
\begin{equation}
\rho_s(p)=\big(1+e^{\mu s}\big)\lVert \eta\rVert_{\max}+\frac{\tilde L}{\mu}\big(e^{\mu s}-1\big)\lVert p-p^\star\rVert,\qquad s\ge0,
\label{eq:rho}
\end{equation}
the largest residual a window of length $s$ can produce at parameter $p$ (Lemma~\ref{lem:residual}). $m>0$ denotes a strong-convexity constant of a cost near its minimiser (the draft wrote $\mu$ for it; renamed to avoid the clash with the logarithmic norm).

\subsection{Numerical protocol (R002)}
The target is FHN, $\dot v=v-\tfrac13v^3-w+0.5$, $\dot w=\tfrac1{12.5}(v+0.7-0.8w)$, written over the cubic library $\Theta=\{1,w,w^2,w^3,v,vw,vw^2,v^2,v^2w,v^3\}$ ($J=10$, $p\in\mathbb R^{20}$, 7 non-zeros). Data: fixed-step Tsit5 at $0.01$ from $(1,1)$ to $T=156$, every 100th point kept, cropped to $t\in[50,150]$ and shifted to $[0,100]$ ($N=101$, $\Delta t=1$), Gaussian noise of standard deviation $0.05\times$ each component's own standard deviation; $\lVert \eta\rVert_{\max}=0.159$, noise floor $\tfrac1N\sum\lVert \eta_i\rVert^2=0.0052$. The optimised cost is the normalised, sparsity-penalised version of \eqref{eq:JK},
\begin{equation}
J_\kappa(z)=\frac1N\Big[\lVert x_0-y_0\rVert^2+\sum_{i=1}^{N-1}\lVert \hat x(t_{i+1};p,y_{\tau(i)})-y_{i+1}\rVert^2\Big]+\frac{\gamma}{20}\sum_{j=1}^{20}\operatorname{smooth}\ell_1(p_j),\qquad z=[x_0;p]\in\mathbb R^{22},
\label{eq:Jkappa}
\end{equation}
where $\kappa$ is the \emph{window size} in data intervals ($K=\lceil(N-1)/\kappa\rceil$ windows; $\kappa=1$ is one window per interval, $\kappa=100$ is single shooting), each interval is integrated with $S=10$ Tsit5 sub-steps, $\operatorname{smooth}\ell_1(x)=\alpha^{-1}[\log(1+e^{-\alpha x})+\log(1+e^{\alpha x})]$ with $\alpha=500$, and $\gamma=0.05$. If a state exceeds $10^3$ or becomes NaN the loss returns $10^3$ (\emph{flat blow-up penalty}; a plateau with zero gradient); the \emph{graded} penalty returns $10^3(1+\tfrac{N-1-i}{N-1})+J_{\rm partial}/N$ when the blow-up occurs at index $i$. The theory checks use $\gamma=0$.

\emph{Arms.} \textbf{GP}: sweep $\kappa$ upward through the FULL schedule $\{1,2,3,4,5,6,8,10,12,15,20,25,33,50,75,100\}$, each stage started from the previous minimiser. \textbf{best-GP}: as GP but started from the lowest-cost minimiser seen so far. \textbf{Control} (no propagation): every stage started from the same random seed vector. Optimiser: Nelder--Mead, 2500 iterations per stage, unless stated; BFGS/L-BFGS use the exact automatic-differentiation gradient. Seeds $s=1,\dots,8$: $z_{\rm seed}=[y_0;0.1\,\xi]$, $\xi\sim\mathcal N(0,I_{20})$, giving $\lVert p_0-p^\star\rVert\in[1.50,1.84]$. Arms are compared at equal budgets (same iterations, schedule and seeds) unless a table says otherwise. Other schedules: SHORT $\{1,2,5,10,25,50,100\}$ (variations), DENSE (44 stages), COARSE $\{1,5,25,100\}$, JUMP $\{1,100\}$.

\emph{Metrics.} $J_\kappa$ at the returned minimiser (values $\ge10^3$ flag the plateau); the \emph{parameter error} $\lVert p-p^\star\rVert$ (Euclidean, 20 coefficients; the identification metric and the primary one here); the \emph{recovery score} in $[0,1]$: each coefficient earns credit 1 if it is correctly zero ($|p_j|<0.05$ when $p^\star_j=0$) or correctly non-zero within 25\,\% of the truth, averaged over the 20 coefficients; the \emph{paired difference} of $\lVert p-p^\star\rVert$ between two arms sharing a seed, reported as the mean with a 95\,\% percentile bootstrap CI over 2000 resamples of the seeds (the resampling unit is the optimiser seed), with \emph{cellwise} counts of seeds in which one arm is strictly better.

\subsection{Symbols and frozen constants}
\begin{center}\small
\begin{tabular}{p{0.16\textwidth}p{0.42\textwidth}p{0.14\textwidth}p{0.2\textwidth}}
\toprule
symbol & meaning & value & where frozen\\
\midrule
$N$, $\Delta t$ & samples, spacing & $101$, $1$ & R002 data recipe\\
$\kappa$, $K$ & window size (data intervals), number of windows & swept, $\lceil100/\kappa\rceil$ & R002 schedules\\
$\Delta T$ & window length $\kappa\Delta t$ & swept & draft, \S\ref{sec:defs}\\
$S$, $\delta t$ & integrator sub-steps per interval, step & $10$, $0.1$ & R002 driver\\
$\gamma$, $\alpha$ & sparsity weight, smooth-$\ell_1$ sharpness & $0.05$, $500$ & R002 driver\\
$\lVert \eta\rVert_{\max}$ & largest noise norm & $0.159$ & R002 \texttt{fhn\_data\_meta}\\
$L$ & Lipschitz constant in $x$ along the orbit / on $[-2.5,2.5]\times[-1,2]$ & $3.05$ / $5.34$ & R002 \texttt{concept\_meta}\\
$\mu$, $\mu_-$ & logarithmic norm and lower counterpart along the orbit & $1.17$, $-2.96$ & R002 / R003\\
$\tilde L$ & Lipschitz constant in $p$ along the orbit & $10.56$ & R002 \texttt{concept\_meta}\\
$D$, $\kappa(D)$ & weighting $\mathrm{diag}(1,s)$, $s=\sqrt{|J_{12}|/|J_{21}|}$ & $s=3.54$ & R003\\
$m$ & strong-convexity constant near a minimiser & assumed & draft (as ``$\mu$'')\\
blow-up threshold & $|x|>10^3$ or NaN & $10^3$ & R002 loss\\
\bottomrule
\end{tabular}
\end{center}
\textbf{Read this:} {$L$ and $\mu$ are measured along the true orbit, not assumed. The numerical section shows that the amplification of an initial-state error on the FHN limit cycle never exceeds $8.5$; bounds carrying $e^{L\Delta T}$ are loose by orders of magnitude for $\Delta T\gtrsim2$, bounds carrying $e^{\mu\Delta T}$ by much less, and the per-window exponents of \S\ref{sec:refine} are close to the measurement.}

\section{Convergence analysis of multiple shooting}\label{sec:theory}
Throughout, $f$ is continuously differentiable with the constants of \S\ref{sec:defs} finite on the region of interest. Statements marked (revised) differ from the earlier draft; each is followed by the change and its effect.

\subsection{Sensitivity of the flow to the initial state}\label{sec:lemma1}
\begin{lemma}[flow sensitivity to the initial state; revised]\label{lem:state}
For all $t\ge0$, $p$ and $x_1,x_2$,
\begin{equation}
e^{\mu_- t}\lVert x_1-x_2\rVert\ \le\ \lVert \varphi_f(t;\,p,\,x_1)-\varphi_f(t;\,p,\,x_2)\rVert\ \le\ e^{\mu t}\lVert x_1-x_2\rVert.
\label{eq:lemma1}
\end{equation}
In particular, since $\mu\le L$ and $\mu_-\ge-L$, the draft's bounds $e^{-Lt}\lVert x_1-x_2\rVert\le\lVert \cdot\rVert\le e^{Lt}\lVert x_1-x_2\rVert$ hold.
\end{lemma}
\begin{proof}
Let $\hat x_j(t)=\varphi_f(t;\,p,\,x_j)$ and $\delta=\hat x_1-\hat x_2$, so $\dot\delta=f(\hat x_1;p)-f(\hat x_2;p)$ with $\delta(0)=x_1-x_2$. By the mean-value form, $f(\hat x_1;p)-f(\hat x_2;p)=A(t)\delta$ with $A(t)=\int_0^1J_f(\hat x_2+s\delta)\,\mathrm{d} s$, hence
\begin{equation}
\tfrac12\tfrac{\,\mathrm{d}}{\,\mathrm{d} t}\lVert \delta\rVert^2=\langle\delta,A(t)\delta\rangle=\big\langle\delta,\tfrac12(A+A^\top)\delta\big\rangle\in\big[\mu_-\lVert \delta\rVert^2,\ \mu\lVert \delta\rVert^2\big].
\label{eq:proof1}
\end{equation}
Thus $\mu_-\lVert \delta\rVert\le\tfrac{\,\mathrm{d}}{\,\mathrm{d} t}\lVert \delta\rVert\le\mu\lVert \delta\rVert$ wherever $\delta\ne0$ (and $\delta\equiv0$ otherwise by uniqueness), and Gr\"onwall's inequality gives \eqref{eq:lemma1}.
\end{proof}
\emph{What changed.} The draft bounded $\lVert \dot\delta\rVert\le L\lVert \delta\rVert$ and integrated $\tfrac{\,\mathrm{d}}{\,\mathrm{d} t}\ln\lVert \delta\rVert$. That step applies Cauchy--Schwarz to $\langle\delta,\dot\delta\rangle$ and discards the sign information in the inner product; \eqref{eq:proof1} keeps it. On FHN the Jacobian is $\begin{psmallmatrix}1-v^2&-1\\0.08&-0.064\end{psmallmatrix}$ and $1-v^2\le1$ for every $v$, so $\mu=1.17$ on all of $\mathbb R^2$ (attained at $v=0$), whereas $L$ grows like $v^2$ and is $3.05$ on the orbit but $5.34$ on the box. The revised lemma therefore holds for perturbations of any size with one constant. At $t=\Delta T=2$ the bound falls from $e^{L\Delta T}=447$ to $e^{\mu\Delta T}=10.4$ against a measured peak of $6.9$ (Table~\ref{tab:lemmas}).

\subsection{Sensitivity of the flow to the parameters}
\begin{lemma}[flow sensitivity to the parameters; revised]\label{lem:param}
For all $t\ge0$, $x_0$ and $p_1,p_2$,
\begin{equation}
\lVert \varphi_f(t;\,p_1,\,x_0)-\varphi_f(t;\,p_2,\,x_0)\rVert\le\frac{\tilde L}{\mu}\big(e^{\mu t}-1\big)\lVert p_1-p_2\rVert,
\label{eq:lemma2}
\end{equation}
where the right-hand side is read as $\tilde L\,t\lVert p_1-p_2\rVert$ when $\mu=0$ and as $\frac{\tilde L}{|\mu|}(1-e^{-|\mu|t})\lVert p_1-p_2\rVert\le\frac{\tilde L}{|\mu|}\lVert p_1-p_2\rVert$ when $\mu<0$.
\end{lemma}
\begin{proof}
With $\hat x_j(t)=\varphi_f(t;\,p_j,\,x_0)$ and $\delta=\hat x_1-\hat x_2$, $\delta(0)=0$, split
$\dot\delta=\big[f(\hat x_1;p_1)-f(\hat x_2;p_1)\big]+\big[f(\hat x_2;p_1)-f(\hat x_2;p_2)\big]$: the first bracket is the state difference at equal parameters, the second the parameter difference at equal state. Taking the inner product with $\delta$, the first bracket contributes at most $\mu\lVert \delta\rVert^2$ by \eqref{eq:proof1} and the second at most $\tilde L\lVert \delta\rVert\lVert p_1-p_2\rVert$ by the Lipschitz property in $p$, so $\tfrac{\,\mathrm{d}}{\,\mathrm{d} t}\lVert \delta\rVert\le\mu\lVert \delta\rVert+\tilde L\lVert p_1-p_2\rVert$. Multiplying by the integrating factor $e^{-\mu t}$ and integrating from $0$ to $t$ gives \eqref{eq:lemma2}.
\end{proof}
\emph{What changed.} Only the first bracket's bound, $L\to\mu$, as in Lemma~\ref{lem:state}. The consequence is qualitative: for a dissipative system ($\mu<0$) the revised bound \emph{saturates} at $\tilde L/|\mu|$, which is the shape the measurement has (Figure~\ref{fig:lemmas}b), whereas $\frac{\tilde L}{L}(e^{Lt}-1)$ can never saturate because a Lipschitz constant is non-negative.

\begin{lemma}[largest residual of a window]\label{lem:residual}
Let $\tau$ be a node and $t_i\in(\tau,\tau+s]$. Then
\begin{equation}
\lVert \varphi_f(t_i-\tau;\,p,\,y_\tau)-y_i\rVert\ \le\ e^{\mu(t_i-\tau)}\lVert \eta_\tau\rVert+\lVert \varphi_f(t_i-\tau;\,p,\,x_\tau)-x_i\rVert+\lVert \eta_i\rVert\ \le\ \rho_s(p).
\label{eq:residual}
\end{equation}
\end{lemma}
\begin{proof}
Write $\varphi_f(t_i-\tau;\,p,\,y_\tau)-y_i=\big[\varphi_f(t_i-\tau;\,p,\,y_\tau)-\varphi_f(t_i-\tau;\,p,\,x_\tau)\big]+\big[\varphi_f(t_i-\tau;\,p,\,x_\tau)-x_i\big]-\eta_i$ and apply the triangle inequality; Lemma~\ref{lem:state} bounds the first bracket by $e^{\mu(t_i-\tau)}\lVert \eta_\tau\rVert$. For the second bracket note $x_i=\varphi_f(t_i-\tau;\,p^\star,\,x_\tau)$, so Lemma~\ref{lem:param} bounds it by $\frac{\tilde L}{\mu}(e^{\mu(t_i-\tau)}-1)\lVert p-p^\star\rVert$; with $t_i-\tau\le s$ and $\lVert \eta_\tau\rVert,\lVert \eta_i\rVert\le\lVert \eta\rVert_{\max}$ this is $\rho_s(p)$ of \eqref{eq:rho}.
\end{proof}

\subsection{The windowed cost near the truth}
\begin{proposition}[cost at the true parameter; revised]\label{prop:cost}
For any partition with longest window $\Delta T$ and $n_k$ data in window $k$,
\begin{align}
J_K(p)&\le\sum_{k=1}^{K}\sum_{t_i\in(\tau_{k-1},\tau_k]}\Big(e^{\mu(t_i-\tau_{k-1})}\lVert \eta_{\tau_{k-1}}\rVert+\lVert \varphi_f(t_i-\tau_{k-1};\,p,\,x_{\tau_{k-1}})-x_i\rVert+\lVert \eta_i\rVert\Big)^2,\label{eq:JK_bound}\\
J_K(p^\star)&\le\lVert \eta\rVert_{\max}^2\sum_{k=1}^{K}\sum_{j=1}^{n_k}\big(1+e^{\mu j\Delta t}\big)^2\ \le\ N\lVert \eta\rVert_{\max}^2\big(1+e^{\mu\Delta T}\big)^2,\label{eq:JK_true}\\
\mathbb E\,J_K(p^\star)&\le d\sigma^2\sum_{k=1}^{K}\sum_{j=1}^{n_k}\big(1+e^{2\mu j\Delta t}\big)\ \le\ N d\sigma^2\big(1+e^{2\mu\Delta T}\big).\label{eq:JK_expect}
\end{align}
As $\eta\to0$, $J_K(p)\to J_K^\star(p)$ uniformly on bounded sets of $p$, and $J_K^\star(p^\star)=0$. For single shooting the same argument gives $J_1(p)\le\sum_{i=1}^N\big(e^{\mu(t_i-t_0)}\lVert \eta_0\rVert+\lVert \varphi_f(t_i-t_0;\,p,\,x_0)-x_i\rVert+\lVert \eta_i\rVert\big)^2$, in which the initial-condition error is amplified over the whole record.
\end{proposition}
\begin{proof}
\eqref{eq:JK_bound} is Lemma~\ref{lem:residual} squared and summed. At $p^\star$ the middle term vanishes; the $j$-th datum of a window sits at $t_i-\tau_{k-1}=j\Delta t$, which gives \eqref{eq:JK_true}, and $j\Delta t\le\Delta T$ the crude form. For \eqref{eq:JK_expect} expand $\lVert u-\eta_i\rVert^2=\lVert u\rVert^2-2\langle u,\eta_i\rangle+\lVert \eta_i\rVert^2$ with $u=\varphi_f(t_i-\tau;\,p^\star,\,y_\tau)-x_i$, which depends on $\eta_\tau$ only; for $t_i\ne\tau$ the noises are independent and $\mathbb E\eta_i=0$, so the cross term has zero mean, $\mathbb E\lVert \eta_i\rVert^2=d\sigma^2$, and $\mathbb E\lVert u\rVert^2\le e^{2\mu j\Delta t}\mathbb E\lVert \eta_\tau\rVert^2$ by Lemma~\ref{lem:state}.
\end{proof}
\emph{What changed.} Three things. (i) $L\to\mu$. (ii) The draft applied the worst-case factor $e^{L\Delta T}$ to every datum in the window; \eqref{eq:JK_true} keeps the actual offset $j\Delta t$, and the geometric sum $\sum_{j\le n_k}e^{2\mu j\Delta t}\approx e^{2\mu\Delta T}/(2\mu\Delta t)$ saves a factor of order $\mu\Delta T$ relative to $n_ke^{2\mu\Delta T}$. (iii) The draft replaced every $\lVert \eta_i\rVert$ by $\lVert \eta\rVert_{\max}$; under the Gaussian model of \S\ref{sec:defs}, $\mathbb E\lVert \eta_i\rVert^2=d\sigma^2$ while $\lVert \eta\rVert_{\max}^2\approx\sigma^2(d+2\log N)$, a factor $1+2\log N/d$ ($4.9$ measured on the FHN data), and the cross term between node noise and datum noise vanishes in expectation, saving a further factor up to 2. \eqref{eq:JK_expect} is the natural input to the next step.

\subsection{Bound on the error $\lVert p^\star-p^{(K)}\rVert$}
\begin{proposition}[displacement of the minimiser from the truth]\label{prop:perr}
Suppose $J_K$ is strongly convex with constant $m>0$ on a convex neighbourhood of $p^{(K)}$ containing $p^\star$, i.e.\ $J_K(q)\ge J_K(p^{(K)})+\tfrac m2\lVert q-p^{(K)}\rVert^2$ there. Then
\begin{equation}
\lVert p^\star-p^{(K)}\rVert\ \le\ \sqrt{\frac2m\Big(J_K(p^\star)-J_K(p^{(K)})\Big)}\ \le\ \sqrt{\frac2m\Big(N\lVert \eta\rVert_{\max}^2\big(1+e^{\mu\Delta T}\big)^2-J_K(p^{(K)})\Big)},
\label{eq:perr}
\end{equation}
and, replacing $J_K(p^\star)$ by its expectation, $\mathbb E\lVert p^\star-p^{(K)}\rVert^2\le\tfrac2m\big(Nd\sigma^2(1+e^{2\mu\Delta T})-\mathbb E J_K(p^{(K)})\big)$. For single shooting, $\lVert p^\star-p^{(1)}\rVert\le\sqrt{\tfrac2m\big(\sum_{i=1}^N(e^{\mu(t_i-t_0)}\lVert \eta_0\rVert+\lVert \eta_i\rVert)^2-J_1(p^{(1)})\big)}$.
\end{proposition}
\begin{proof}
The gradient vanishes at the minimiser, so the strong-convexity inequality at $q=p^\star$ gives the first bound; Proposition~\ref{prop:cost} gives the second. (The draft's first version of this bound, via a second-order Taylor expansion with Hessian eigenvalue $\lambda_{\min}$, is the same statement with $m=\lambda_{\min}$ and the factor $2$ absorbed.)
\end{proof}
The bound is sharp in kind, not in size: the first inequality in \eqref{eq:perr} is an identity for a quadratic cost, so the whole slack is in the estimate of $J_K(p^\star)$, i.e.\ in the exponential. \textbf{Read this:} {the strong-convexity premise must hold at the \emph{minimiser}; the numerical section shows that at $p^\star$ the noisy cost is indefinite for every $\kappa\ge2$ (Figure~\ref{fig:hessian}) and that the premise is not verified at the iterates Nelder--Mead returns (Figure~\ref{fig:post}).}

\section{Propagating the best guess across shooting-node partitions}\label{sec:removal}
We now bound how the optimum moves when shooting nodes are removed. Let the fine partition have nodes $\{\tau_k\}_{k=0}^K$ and remove the nodes indexed by $\mathcal I_R$; the coarse cost is
\begin{equation}
\hat{\mathcal J}_K(p)=\sum_{k\notin\mathcal I_R}\ \sum_{t_i\in(\tau_k,\tau_k^{+}]}\lVert \varphi_f(t_i-\tau_k;\,p,\,y_{\tau_k})-y_i\rVert^2,
\label{eq:Jhat}
\end{equation}
i.e.\ every datum in a fine window $D_k$ launched at a removed node $\tau_k$ is now predicted from the nearest remaining node to its left, $\tau_k^-$, through the removed node (Figure~\ref{fig:removal}). We assume the removed nodes' neighbours are a subset of the fine nodes (the coarse partition is a sub-partition), which is what the algorithm does.

\begin{proposition}[cost change under node removal; revised]\label{prop:removal}
With $\rho_s$ as in \eqref{eq:rho}, for every $p$,
\begin{equation}
\big|\hat{\mathcal J}_K(p)-J_K(p)\big|\ \le\ 2\,n_{\max}\,|\mathcal I_R|\;e^{\mu\Delta T_1}\;\rho_{\Delta T_2}(p)\;\rho_{\Delta T_1+\Delta T_2}(p)\ =:\ \Delta_K(p).
\label{eq:prop_removal}
\end{equation}
At $p=p^\star$ this is $2n_{\max}|\mathcal I_R|\,e^{\mu\Delta T_1}(1+e^{\mu\Delta T_2})(1+e^{\mu(\Delta T_1+\Delta T_2)})\lVert \eta\rVert_{\max}^2$, quadratic in the noise.
\end{proposition}
\begin{proof}
Only the data in $D_k$, $k\in\mathcal I_R$, change predictor. For $t_i\in D_k$ let $u_i=\varphi_f(t_i-\tau_k^-;\,p,\,y_{\tau_k^-})$ (coarse prediction), $v_i=\varphi_f(t_i-\tau_k;\,p,\,y_{\tau_k})$ (fine prediction), $a_i=\lVert u_i-y_i\rVert$ and $b_i=\lVert v_i-y_i\rVert$. Then
\[
\hat{\mathcal J}_K(p)-J_K(p)=\sum_{k\in\mathcal I_R}\sum_{t_i\in D_k}\big(a_i^2-b_i^2\big),\qquad |a_i^2-b_i^2|=|a_i-b_i|\,(a_i+b_i)\le\lVert u_i-v_i\rVert\,(a_i+b_i)
\]
by the reverse triangle inequality. By the flow property $u_i=\varphi_f(t_i-\tau_k;\,p,\,\varphi_f(\tau_k-\tau_k^-;\,p,\,y_{\tau_k^-}))$, so Lemma~\ref{lem:state} gives $\lVert u_i-v_i\rVert\le e^{\mu(t_i-\tau_k)}\,g_k\le e^{\mu\Delta T_1}g_k$ with $g_k=\lVert \varphi_f(\tau_k-\tau_k^-;\,p,\,y_{\tau_k^-})-y_{\tau_k}\rVert$ the coarse residual at the removed node, and Lemma~\ref{lem:residual} gives $g_k\le\rho_{\Delta T_2}(p)$. The residuals $a_i$ and $b_i$ are residuals of windows of length at most $\Delta T_1+\Delta T_2$ and $\Delta T_1$, so $a_i+b_i\le2\rho_{\Delta T_1+\Delta T_2}(p)$ ($\rho_s$ is increasing in $s$). Summing over the at most $n_{\max}$ data of each of the $|\mathcal I_R|$ windows gives \eqref{eq:prop_removal}.
\end{proof}
\emph{What changed.} (i) $L\to\mu$. (ii) The draft bounded $a_i+b_i$ through $\lVert u_i+v_i-2y_i\rVert\le\dots+2\max\lVert y_i\rVert$, which introduces the size of the data ($C^{(2)}_k=\max\lVert y_{i+1}\rVert$, equal to $2.34$ on FHN) into the constant $C_{\max}$; the bound was therefore linear in the small quantities times the attractor radius. Here $a_i+b_i$ is bounded by two residuals, so the bound is quadratic in $(\lVert \eta\rVert,\lVert p-p^\star\rVert)$, as the left-hand side is; on FHN this removes a factor $\lVert y\rVert_{\max}/\lVert \eta\rVert_{\max}\approx15$. (iii) The draft's sum over $t_i\in[\tau_k,\tau_k^+)$ was collapsed to the factor $|\mathcal I_R|$ without the number of data per window; $n_{\max}$ restores it. (iv) The draft's intermediate inequality carried $e^{2L(\Delta T_1+\Delta T_2)}$ while its statement had $e^{L(\Delta T_1+\Delta T_2)}$; the exponent here is $e^{\mu\Delta T_1}$ times the exponents inside the two $\rho$ factors, consistently. The design rule of the draft survives unchanged: remove few nodes at a time, prefer low-noise nodes, keep $\Delta T_2$ small (Figure~\ref{fig:prop1}).

\begin{theorem}[displacement of the minimiser under node removal; revised]\label{thm:main}
Let $p^{(K)}=\arg\min J_K$ and $\hat p^{(K)}=\arg\min\hat{\mathcal J}_K$, and suppose $J_K$ is strongly convex with constant $m>0$ on a convex neighbourhood of $p^{(K)}$ that contains $\hat p^{(K)}$. Then
\begin{equation}
\lVert \hat p^{(K)}-p^{(K)}\rVert\ \le\ \sqrt{\frac2m\Big(\Delta_K\big(p^{(K)}\big)+\Delta_K\big(\hat p^{(K)}\big)\Big)},
\label{eq:thm}
\end{equation}
with $\Delta_K$ from \eqref{eq:prop_removal}; each $\Delta_K(q)$ is a quadratic polynomial in $\lVert \eta\rVert_{\max}$ and $\lVert q-p^\star\rVert$ with coefficients $e^{\mu\Delta T_1}(1+e^{\mu\Delta T_2})(1+e^{\mu(\Delta T_1+\Delta T_2)})$ and smaller.
\end{theorem}
\begin{proof}
Strong convexity at the minimiser ($\nabla J_K(p^{(K)})=0$) gives $\tfrac m2\lVert \hat p^{(K)}-p^{(K)}\rVert^2\le J_K(\hat p^{(K)})-J_K(p^{(K)})$. Insert $\pm\hat{\mathcal J}_K(\hat p^{(K)})$ and $\pm\hat{\mathcal J}_K(p^{(K)})$:
\[
J_K(\hat p^{(K)})-J_K(p^{(K)})=\big[J_K(\hat p^{(K)})-\hat{\mathcal J}_K(\hat p^{(K)})\big]+\big[\hat{\mathcal J}_K(\hat p^{(K)})-\hat{\mathcal J}_K(p^{(K)})\big]+\big[\hat{\mathcal J}_K(p^{(K)})-J_K(p^{(K)})\big].
\]
The middle bracket is $\le0$ because $\hat p^{(K)}$ minimises $\hat{\mathcal J}_K$; the outer brackets are bounded by Proposition~\ref{prop:removal}.
\end{proof}
\emph{What changed.} The structure of the proof is the draft's; the bound inherits the three improvements of Proposition~\ref{prop:removal}. In particular the displacement now scales like the noise (through $\sqrt{\Delta_K}\sim\lVert \eta\rVert_{\max}$), not like $\sqrt{\text{noise}\times\text{attractor size}}$. The draft also mixed $L(p)$, $L(p^\star)$ and $L_K=\max\{L(p^{(K)}),L(\hat p^{(K)})\}$ in one formula; with $\mu$ a single constant on the region of interest the distinction disappears.

\section{Refinement: trajectory-dependent exponents and a weighted norm}\label{sec:refine}
The bounds above are a priori: they hold for every right-hand side with the same constants and are attained by the linear system $\dot x=\mu x$, so they are necessarily exponential. Along an attracting orbit the actual growth is far smaller, and the same proof yields a sharper, trajectory-dependent statement.

\begin{corollary}[per-window exponents]\label{cor:window}
The identity \eqref{eq:proof1} holds pointwise in time, so with the local logarithmic norm $\mu(t)$ of the mean-value matrix $A(t)$,
\begin{equation}
\lVert \varphi_f(t;\,p,\,x_1)-\varphi_f(t;\,p,\,x_2)\rVert\le\exp\!\Big(\int_0^t\mu(s)\,\mathrm{d} s\Big)\lVert x_1-x_2\rVert,
\label{eq:window}
\end{equation}
and Lemmas~\ref{lem:state}--\ref{lem:residual}, Propositions~\ref{prop:cost}, \ref{prop:removal} and Theorem~\ref{thm:main} hold with $e^{\mu\Delta T}$ replaced by $\max_ke^{\Lambda_k}$, $\Lambda_k=\int_{\tau_{k-1}}^{\tau_k}\mu(s)\,\mathrm{d} s$. The same proof runs in any weighted norm $\lVert x\rVert_D=\lVert Dx\rVert$ with $\mu_D$ in place of $\mu$, and a bound in $\lVert \cdot\rVert_D$ implies the Euclidean bound with an extra factor $\kappa(D)$.
\end{corollary}
For small perturbations $\mu(s)$ is the logarithmic norm of $J_f$ along the reference orbit (first order in $\lVert \delta\rVert$; for a rigorous statement take the supremum over the segment joining the two trajectories). On FHN the choice $D=\mathrm{diag}(1,s)$ with $s=\sqrt{|J_{12}|/|J_{21}|}=3.54$ makes the off-diagonal entries of $DJ_fD^{-1}$ equal and opposite, so its symmetric part is diagonal and $\mu_D(x)=\max(1-v^2,-0.064)$: \emph{negative} whenever $|v|>1$, i.e.\ on both slow branches of the relaxation oscillation, with global supremum $1$. Table~\ref{tab:windows} evaluates the window exponents along the true orbit: in the weighted norm the worst window has amplification at most $8.04$ for every window length tested (times $\kappa(D)=3.54$ for a Euclidean statement) against the measured peak of $8.5$, and the median window contracts. Two consequences. First, this is the quantitative version of ``the restart at each node resets the error''. Second, it gives a node-placement rule sharper than the one after Proposition~\ref{prop:removal}: keep nodes just before the fast jumps, where $\int\mu$ accumulates, and remove them freely on the slow branches, where every window is contracting. The price is that $\Lambda_k$ is evaluated along the fitted orbit, so the bound is a posteriori. What no Gr\"onwall-type argument can deliver is the observed \emph{plateau} of the amplification (Figure~\ref{fig:lemmas}a): that is a property of the attractor and would require a contraction metric (impossible globally on a limit cycle, whose phase direction is neutral), a Lyapunov function, or a Floquet analysis of the variational equation.
```

## 2.2 Section 6 (numerical results), condensed

Section 6 is inherited wholesale from reports R002 (522 optimisation runs, 18 gates) and
R003 (bound evaluation, 8 gates); the figures and tables are byte-identical copies and
every prose number is regenerated from copied result files. R004 itself contains **no new
numerics**. What follows is the prose of each subsection, the two theory-checking tables,
and the three key figure captions.

### 6.1 The bounds against measurement

**Table `lemmas.tex` — draft versus revised bounds on FHN**, evaluated at $t=\Delta T$,
against the largest value observed over 24 probes for $t\le\Delta T$ (Lemma 1 probes:
$x_1$ a datum at four phases of the orbit, $x_2=x_1+10^{-3}u$, six random unit directions;
Lemma 2 probes: $p_2=p^\star+10^{-3}u$, $u\in\mathbb R^{20}$).

| $t=\Delta T$ | L1 original $e^{L\Delta T}$ | L1 proposed $e^{\mu\Delta T}$ | L1 observed peak | L2 original $\frac{\tilde L}{L}(e^{L\Delta T}-1)$ | L2 proposed $\frac{\tilde L}{\mu}(e^{\mu\Delta T}-1)$ | L2 observed peak |
|---|---|---|---|---|---|---|
| 1  | 21.1 | 3.23 | 2.93 | 69.8 | 20.1 | 3.7 |
| 2  | 447 | 10.4 | 6.92 | $1.5\times10^{3}$ | 84.9 | 6.7 |
| 5  | $4.2\times10^{6}$ | 349 | 8.48 | $1.5\times10^{7}$ | $3.1\times10^{3}$ | 12.0 |
| 10 | $1.8\times10^{13}$ | $1.2\times10^{5}$ | 8.48 | $6.2\times10^{13}$ | $1.1\times10^{6}$ | 15.1 |

**Table `windows.tex` — per-window amplification $e^{\Lambda_k}$ along the true FHN orbit**
($\Delta t=1$, windows aligned to $t_0$), Corollary 1: worst and median window, in the
Euclidean norm and in the weighted norm $\|Dx\|$; "contracting" is the share of windows
with $\Lambda^D_k<0$. Caption states: weighted values convert to Euclidean statements by a
factor at most $\kappa(D)=3.54$.

| $\Delta T$ | windows | a priori $e^{\mu\Delta T}$ | Euclidean worst | Euclidean median | weighted worst | weighted median | contracting |
|---|---|---|---|---|---|---|---|
| 1  | 100 | 3.23 | 3.10 | 1.16 | 2.60 | 0.94 | 74 % |
| 2  | 50  | 10.4 | 8.56 | 1.41 | 5.89 | 0.88 | 70 % |
| 5  | 20  | 349  | 34.78 | 3.75 | 8.04 | 0.73 | 55 % |
| 10 | 10  | $1.2\times10^{5}$ | 138.79 | 29.25 | 5.89 | 3.20 | 30 % |

**Figure 7 caption (`fig:lemmas`, bounds vs measurement), verbatim:**

```latex
\caption{\textbf{Lemmas~\ref{lem:state} and \ref{lem:param} on FHN.} (a) Amplification $\lVert \varphi(t;p^\star,x_1)-\varphi(t;p^\star,x_2)\rVert/\lVert x_1-x_2\rVert$, largest value over 24 probes ($x_1$ a datum at four phases of the orbit, $x_2=x_1+10^{-3}u$ for six random unit directions; blue), against the draft's bound $e^{Lt}$ (vermilion) and the revised $e^{\mu t}$ (green dashed). (b) The same for the parameter sensitivity $\lVert \varphi(t;p_1,x_0)-\varphi(t;p_2,x_0)\rVert/\lVert p_1-p_2\rVert$ ($p_2=p^\star+10^{-3}u$, $u\in\mathbb R^{20}$) against $\frac{\tilde L}{L}(e^{Lt}-1)$ and $\frac{\tilde L}{\mu}(e^{\mu t}-1)$. Both bounds hold at every probe; the revised one is $10^8$ closer at $t=10$ but still grows, while the measurement peaks at $8.5$ near the fast jump and then contracts because the orbit is attracting. (c) Local rates along one FHN period ($39.5$ time units): the local Lipschitz constant $\lVert \partial f/\partial x\rVert_2$ (dotted) never drops below 1; the Euclidean logarithmic norm $\mu(x^\star(t))$ (green) is near zero on the slow branches and reaches $1.17$ only during the two fast jumps; in the weighted norm the local $\mu_D$ (purple) is negative for $75\%$ of the period. Inherited from R003 Figure~1.}\label{fig:lemmas}
```

**Figure 10 caption (`fig:prop1`, Proposition 3 check), verbatim:**

```latex
\caption{\textbf{Proposition~\ref{prop:removal}: node removal changes the cost by a bounded amount.} Starting from the finest partition ($K=100$, $\gamma=0$): (a) keep every $m$-th node so that $|\mathcal I_R|=100-\lceil100/m\rceil$ nodes are removed ($\Delta T_2=\Delta t$ fixed, $\Delta T_1=m\Delta t$ grows); (b) remove one contiguous block of $b$ nodes after node 30 ($|\mathcal I_R|=b$, $\Delta T_2=b\Delta t$ grows). Shown for $p=p^\star$ (green), a vector $0.06$ from $p^\star$ on six coefficients (blue) and $p_{\rm wrong}$ of Figure~\ref{fig:single} (vermilion). $|\hat{\mathcal J}-J|$ grows with $|\mathcal I_R|$ and with $\Delta T_2$, and much faster away from $p^\star$, as the proposition says; the rate $e^{L\Delta T_2}$ of the draft (grey dashed in (b)) is far steeper than the observed growth, which saturates once the removed block spans a full fast excursion. Inherited from R002 Figure~10.}\label{fig:prop1}
```

### 6.2 The cost landscape as the window grows

Three figures, no prose. Summary of what they report: (Fig. 8) two-dimensional slices of
$J_\kappa$ in the $(v,v^3)$ coefficient plane — as $\kappa$ grows 1→100 the basin around the
truth narrows by an order of magnitude and the blow-up plateau expands (any positive cubic
coefficient makes $\dot v\sim cv^3$ blow up in finite time). (Fig. 9) one-dimensional
slices along random unit directions $u_d\in\mathbb R^{20}$: local minima per slice, mean
over 12 directions, rise from **1.25** at $\kappa=1$ to **7.9** at $\kappa=100$; the finite
region shrinks to $|s|\lesssim0.5$. (Fig. `fig:hessian`) curvature of the data term at
$p^\star$: $\lambda_{\max}$ grows from 36 ($\kappa=1$) to $1.2\times10^6$ ($\kappa=100$),
$\|\nabla J_\kappa(p^\star)\|$ from 0.04 to 96, and the number of negative Hessian
eigenvalues at $p^\star$ is $\ge1$ for **every $\kappa\ge2$** (up to 6 of 20), so "the
strong-convexity premise of Proposition 2 and Theorem 1 can only hold near the
*minimiser*, not near $p^\star$".

### 6.3 Guess propagation versus no propagation: the headline experiment

> Protocol: 8 optimiser seeds × 3 arms × the 16-stage FULL schedule, Nelder–Mead with 2500
> iterations per stage, $\gamma=0.05$, $S=10$, flat penalty. Table T1 gives the seed medians
> and the paired statistics per stage; Figures 11–14 show them.

Selected rows of Table T1 (seed medians; full 16-row table in `tables/T1_headline.md`):

| $\kappa$ | $J$ GP | $J$ ctrl | $\|p-p^\star\|$ GP | $\|p-p^\star\|$ ctrl | score GP | score ctrl | GP−ctrl | CI lo | CI hi | #GP better | #GP on plateau |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1   | 0.02569 | 0.02569 | 0.8641 | 0.8641 | 0.40 | 0.40 | 0 | 0 | 0 | 0 | 3 |
| 5   | 0.01906 | 0.1259  | 0.2218 | 1.543  | 0.775 | 0.20 | −1.133 | −1.654 | −0.6627 | 7 | 1 |
| 15  | 0.01414 | 0.5398  | 0.3348 | 1.574  | 0.725 | 0.275 | −1.109 | −1.327 | −0.7739 | 7 | 1 |
| 100 | 0.01273 | 1.506   | 0.3216 | 1.649  | 0.80  | 0.25 | −1.138 | −1.426 | −0.7719 | 7 | 1 |

Accompanying figure findings: all arms coincide at $\kappa=1$ (same start); from $\kappa=2$
GP drops to $\|p-p^\star\|\approx0.22$–$0.38$ and ends single shooting at 0.32 (score 0.80),
while the control climbs back to the seed distance (1.65 at $\kappa=100$; 25 % of its seeds
end on the plateau). One GP seed starts on the plateau at $\kappa=1$ and never moves. **The
parameter error is not monotone**: smallest at $\kappa=5$ (0.22), drifting up to 0.32 while
the cost keeps falling — the noise-induced bias. GP−control CI excludes zero at every
$\kappa\ge2$; GP and best-GP are statistically indistinguishable; GP strictly better in 7
of 8 seeds, tied in the eighth.

### 6.4 Variations (verbatim prose)

> Table 4 summarises the variation experiments of R002 (SHORT schedule unless stated; the
> schedule rows change the number of stages by design, and the optimiser rows do different
> work per iteration, so those comparisons are not at equal budget). Three readings matter
> for the method. Removing few nodes per step matters: DENSE ends at 0.30, COARSE at 0.52
> and JUMP at 0.76, which is the design rule of Proposition 3 in practice. The graded
> blow-up penalty removes the "stuck on the plateau" failure (GP 0.27, no seed on the
> plateau). With the exact gradient, BFGS with guess propagation is the best and cheapest
> arm (0.21, score 0.95), while BFGS without propagation fails because the gradient
> vanishes on the plateau. GP is robust up to 10 % noise (0.31) and fails at 20 % (1.88);
> the control fails even at zero noise.

### 6.5 Other systems (verbatim prose)

> The implementation is dimension- and degree-agnostic; Table 5 and Figures 17–18 apply the
> same sweep to Lotka–Volterra ($\dot x=x-0.5xy$, $\dot y=-0.8y+0.3xy$; quadratic library,
> 12 coefficients, $N=101$, $\Delta t=0.25$, 5 % noise, Nelder–Mead, 6 seeds) and Lorenz
> ($\sigma=10$, $\rho=28$, $\beta=8/3$; quadratic library, 30 coefficients, $N=101$,
> $\Delta t=0.02$, 2 % noise, 4 seeds; BFGS and Nelder–Mead). GP recovers Lotka–Volterra
> (0.29, score 0.83; the control ends at 1.41 with 83 % of its seeds on the plateau) and,
> with BFGS, Lorenz (3.9 on coefficients of size up to 28, score 0.83); GP with Nelder–Mead
> does not converge in 33 dimensions (18.1, score 0.23), so a derivative-free sweep does
> not scale.

### 6.6 The premises of the theory, checked where they are assumed

**Figure 20 caption (`fig:post`, premises check), verbatim:**

```latex
\caption{\textbf{Premises checked at the minimisers.} (a) Smallest eigenvalue of the data-term Hessian at the GP minimiser returned by Nelder--Mead at every stage, 7 seeds (plateau seed excluded; blue = positive definite, vermilion = at least one negative eigenvalue): the returned point is positive definite in only $36\%$ of the (seed, $\kappa$) cells, with on average less than one of 20 eigenvalues slightly negative elsewhere, i.e.\ Nelder--Mead stops short of a stationary point in 22 dimensions and the strong-convexity premise of Theorem~\ref{thm:main} is \emph{not verified} at its iterates (the BFGS iterates would be the place to test it). (b) The cost of the next stage along the segment from $p^{(\kappa_i)}$ to $p^{(\kappa_{i+1})}$ (seed 2), relative to its value at the carried guess: it decreases monotonically along every segment, i.e.\ the carried guess lies inside the basin of the next minimiser, the mechanism Theorem~\ref{thm:main} describes. (c) Cross-evaluation: the minimiser of $\kappa_i$ evaluated at every other $\kappa_j$; the near-diagonal structure shows that a minimiser is good only for nearby window sizes, which is why the schedule must be gradual. Inherited from R002 Figure~37.}\label{fig:post}
```

## 2.3 Gates declared in §7 (for reference)

G1 hash/byte identity of copied inputs with R002/R003; G2 figures byte-identical to
sources; G3 every referenced figure exists and no retired figure is referenced; G4 verbatim
tables byte-identical; G5 `numbers.tex` and `T2_compact.tex` regenerate byte-identically;
G6 prose numbers agree with R002's `table_numbers.json` and R003's `summary.json` to printed
precision; **G7 Proposition 3 is consistent with the draft where they overlap — the exponent
factors of the revised bound are $\le$ the draft's for every $\Delta T_1,\Delta T_2\in\{1,2,5,10\}$
with the FHN constants**; **G8 the revised Lemma 1 / Lemma 2 bounds hold at all $2\times2424$
probe points of R002**; G9 `report.md` contains every heading, table and figure caption;
G10 `report.pdf` exists and is newer than `report.tex`.

## 2.4 Original 2025 draft statements (for comparison)

Verbatim from `main.tex` of `Multishooting_and_sparse_optimization.zip`. Statements only;
proofs omitted except where noted. Note the draft writes $p\in\mathbb R$ (scalar) and uses a
parameter-dependent Lipschitz constant $L(p)$.

**Original Lemma 1 (sensitivity to the initial state):**

```latex
\begin{lemma}\label{lipchitz_lemma_1}
If the RHS function \( f \) in IVP \eqref{eq:model} is Lipschitz, then for \( p \in \mathbb{R} \), there exists \( L(p) \) such that:
\begin{equation}
e^{-L(p)t} \|x_1 - x_2\| \leq \|\varphi_f(t; p, x_1) - \varphi_f(t; p, x_2)\| \leq e^{L(p)t} \|x_1 - x_2\|
\label{eq:lemma_bounds}
\end{equation}
\end{lemma}
```

(Its proof runs: $\|\dot\delta\|\le L(p)\|\delta\|$, therefore
$\left\|\frac{\dot\delta(t)}{\delta(t)}\right\|\le L(p)\Rightarrow\left|\frac{d}{dt}\ln|\delta(t)|\right|\le L(p)$,
then integrate.)

**Original Lemma 2 (sensitivity to the parameters):**

```latex
\begin{lemma} \label{lipchitz_bound_lemma_2}
    Suppose \(f\) is uniformly Lipschitz in both \(x\) and \(p\); i.e.
\[
\|f(x_1;p_1)-f(x_2;p_2)\|\le \tilde L\|p_1-p_2\|,\quad
\|f(x_1;p)-f(x_2;p)\|\le L(p)\|x_1-x_2\|.
\]
Then
\begin{equation}\label{eq:flow_param_diff}
\|\varphi_f(t;p_1,x_0)-\varphi_f(t;p_2,x_0)\|
\le \frac{\tilde L}{L(p)}\bigl(e^{L(p)t}-1\bigr)\,\|p_1-p_2\|.
\end{equation}
\end{lemma}
```

**Original Proposition 1 (cost change under node removal):**

```latex
\begin{proposition} \label{prop:cost_fun_diff_1}
   Let $J_K$ and $\hat{\mathcal{J}}_K$ be cost functions associated with the partitions $\{\tau_k\}_k$ and  \( \{ \tau_k \}_{k \notin \mathcal{I}_R} \) as discussed above.  Let $p^*$ the true optimal parameter and let f be Lipchitz such that
   \[
\|f(x_1;p_1)-f(x_2;p_2)\|\le \tilde L\|p_1-p_2\|,\quad
\|f(x_1;p)-f(x_2;p)\|\le L(p)\|x_1-x_2\|.
\]
Let $\Delta T_1 = \max_k(\tau_k - \tau_{k-1})$, $\Delta T_2 = \max_k(\tau_k-\tau_k^-)$ . Then $\exists$ a  constant $C_{\max}>0$   such that

\begin{equation}
\|\hat{\mathcal{J}}_K(p)-J_K(p)\|
\le 2|\mathcal I_R|\,C_{\max}\Bigl(
e^{L(p)(\Delta T_1+\Delta T_2)}\|\eta\|_{\max}
+e^{L(p)\Delta T_1}\bigl[\tfrac{\tilde L}{L(p)}(e^{L(p)\Delta T_2}-1)\|p-p^*\|
+\|\eta\|_{\max}\bigr]\Bigr).
\label{eq:14}
\end{equation}
Moreover,  $C_{\max}=\max\{C_k^{(1)}, C_k^{(2)}\}$ where  $C_k^{(1)} = \max_{t_i \in [\tau_k^-, \tau_k]} \left\{ \|\varphi_f(t_{i+1} - \tau_k^-; p, y_{\tau_k^-}) - y_{\tau_k}\|,\; \|\varphi_f(t_{i+1} - \tau_k; p, y_{\tau_k}) - y_{\tau_k}\| \right\}$ and $C_k^{(2)} = \max_{t_i \in [\tau_k^-, \tau_k]} \|y_{i+1}\|$.
\end{proposition}
```

The draft's coarse cost, for reference:

```latex
\hat{\mathcal{J}}_K(p) = \sum_{k=1,k \notin \mathcal{I}_R}^K \sum_{t_i \in [\tilde{\tau}_{k-1}, \tilde{\tau}_k)} \left\| \varphi_f(t_i - \tau_{k}^-; p, y_{\tau_{k}^-}) - y_{i+1} \right\|^2
```

**Original Theorem (displacement of the minimiser under node removal):**

```latex
\begin{theorem}
       Let  \(p^{(K)}\) and \(\hat p^{(K)}\) be the optimal parameters to the cost functions $J_K$ and $\hat{\mathcal{J}}_K$ associated with the partitions $\{\tau_k\}_k$ and  \( \{ \tau_k \}_{k \notin \mathcal{I}_R} \) respectively.  Let $p^*$ the true optimal parameter and let f be Lipchitz such that
   \[
\|f(x_1;p_1)-f(x_2;p_2)\|\le \tilde L\|p_1-p_2\|,\quad
\|f(x_1;p)-f(x_2;p)\|\le L(p)\|x_1-x_2\|.
\]
Under the assumption of strong convexity in a local neighborhood of \(p^{(K)}\), there exists positive constants $ \mu, \tilde L, L_K,C_{max}$ such that the following bound holds
\begin{align}
\label{eq:JK17}
\|\hat{p}^{(K)} - p^{(K)}\|
&\le \frac{2}{\sqrt{\mu}} \, \bigg[
|\mathcal{I}_R|\, C_{\max} \Big(
   \|\eta\|_{\max} \big( e^{L_K(\Delta T_1 + \Delta T_2)} + 1 \big) \nonumber \\
&\quad + e^{L(p) \Delta T_1} \cdot \frac{\tilde L}{L(p)} \big( e^{L_K \Delta T_2} - 1 \big)
   \big( \|p^{(K)} - p^*\| + \|\hat{p}^{(K)} - p^*\| \big)
\Big) \bigg]^{1/2}.
\end{align}
\end{theorem}
```

**Original comparator and cost-at-truth / parameter-error bounds** (context for revision
decisions (b) and (e)); the draft's $J^\star$ is the noise-free *single-shooting* cost:

```latex
J^*(p) = \sum_{i=1}^{N} \left\| \varphi_f(t_{i+1}; p, x_0) - x_i \right\|^2

% draft decomposition (the step R004 says is wrong):
\varphi_f(t_{i+1} - \tau_{k-1}; p, y_{\tau_{k-1}}) - y_{i+1}
= \left( \varphi_f(t_{i+1} - \tau_{k-1}; p, y_{\tau_{k-1}}) - \varphi_f(t_{i+1} - \tau_{k-1}; p, x_{\tau_{k-1}}) \right)
  + \varphi_f(t_{i+1}; p, x_0) - x_{i+1} + \eta_{i+1}

% draft's Taylor / Hessian version:
\|p^* - p^{(K)}\|^2 \leq \frac{1}{\lambda_{\min}} \left( N \|\eta_{\max}\|^2 (1 + e^{L(p^*)\Delta T})^2 - J_K(p^{(K)}) \right)

% draft's strong-convexity version ("there exists $\mu>0$"):
\|p^* - p^{(K)}\|
\leq \sqrt{ \frac{2}{\mu} \left( N \|\eta_{\max}\|^2 (1 + e^{L(p^*) \Delta T})^2 - J_K(p^{(K)}) \right) }

% draft's single-shooting version:
\|p^* - p^{(1)}\|
\leq \sqrt{ \frac{2}{\mu} \left(\sum_{i=1}^N e^{L(p^*)(t_{i}-t_0)} \|\eta_{i-1}\|^2 + \|\eta_{i}\|^2 - J_1(p^{(1)}) \right) }
```

---

# Section 3: Critique prompt

```
You are an adversarial reviewer. Be critical. Be argumentative.
Find every hole: missing steps, wrong algebra, untested assumptions,
edge cases not addressed, implicit dependencies, claims without
evidence, off-by-one errors, sign errors, dimensional errors. Don't
be polite — if something is wrong, say so. Concision over hedging.

For each issue, state:
  - WHAT is wrong (specific, not vague — name the line or symbol)
  - WHY it matters (what breaks downstream if uncorrected)
  - WHAT to do (concrete fix, or what evidence would close the gap)

Number your issues. After all issues, end your response with exactly
one of these lines, no other text after it:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

Use APPROVED only when there are no issues you would block on.
Minor nitpicks alone do not justify ISSUES_REMAIN — call them out
but still verdict APPROVED. Use ISSUES_REMAIN whenever any of your
issues are genuinely blocking.
```
