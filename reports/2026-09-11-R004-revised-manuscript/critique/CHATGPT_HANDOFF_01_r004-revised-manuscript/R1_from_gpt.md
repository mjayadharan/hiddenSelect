1. **The sample indexing is inconsistent and breaks every count.**  
   **WHAT:** Data are declared as \(i=0,\dots,N\), which means \(N+1\) samples, but the manuscript says \(N=101\) samples on \(t=0,\dots,100\). Equation (Jkappa) sums \(i=1,\dots,N-1\), predicts \(y_{i+1}\), therefore skips \(y_1\) and requires nonexistent \(y_N\). Meanwhile \(K=\lceil(N-1)/\kappa\rceil\) treats \(N\) as the number of samples.  
   **WHY:** The costs, window counts, \(n_k\), factors \(N\), and claimed correspondence with the numerical implementation cannot all be correct simultaneously.  
   **WHAT TO DO:** Use one convention throughout. The natural one is \(i=0,\dots,N-1\), \(\tau_K=t_{N-1}\), residual indices \(1,\dots,N-1\), and in (Jkappa) \(i=0,\dots,N-2\). Then replace exact counts by \(N-1\), or explicitly retain \(N\) only as a loose upper bound.

2. **The cost claimed to be optimized is not the theoretical cost, and \(x_0\) is currently a dummy variable.**  
   **WHAT:** In (Jkappa), every trajectory is launched from \(y_{\tau(i)}\), while the free \(x_0\) appears only in \(\|x_0-y_0\|^2\). Thus its optimum is trivially \(x_0=y_0\), and it never affects a simulation. Also \(\hat x(t_{i+1};p,y_{\tau(i)})\) omits the required elapsed time \(t_{i+1}-\tau(i)\). The numerical objective additionally contains normalization, smooth-\(\ell_1\) regularization, numerical integration, and a nonsmooth blow-up replacement absent from \(J_K\).  
   **WHY:** As written, the manuscript does not specify the objective actually used by the code. Proposition 2 does not apply to the penalized objective at \(p^\star\), since the penalty generally does not vanish there. Hessians of the data term are not Hessians of the optimized objective.  
   **WHAT TO DO:** Reproduce the implemented loss exactly, including how \(x_0\) launches the first window, correct the elapsed-time and index notation, and state separately which results extend to the penalized objective. Proposition 3 can extend because a partition-independent penalty cancels, but Proposition 2 cannot be imported unchanged.

3. **The actual \(\kappa\)-schedule is not a sequence of node removals.**  
   **WHAT:** The theory requires each coarse partition to be a subpartition of the preceding fine partition and says this “is what the algorithm does.” It is false. For example, the \(\kappa=2\) nodes are \(0,2,4,6,\dots\), whereas the \(\kappa=3\) nodes are \(0,3,6,\dots\): node \(3\) is added while nodes \(2,4\) are removed. Most transitions in the FULL schedule are non-nested.  
   **WHY:** Proposition 3 and Theorem 1 do not bound the displacement between the consecutive objectives used by guess propagation. This severs the manuscript’s central link between theory and the headline experiment.  
   **WHAT TO DO:** Either use a genuinely nested schedule, or prove a repartition bound allowing simultaneous node additions and removals. A comparison through a common refinement is possible, but its two perturbation terms and resulting displacement bound must be written explicitly.

4. **The constants used in the proofs are not valid for the candidate models to which the results are applied.**  
   **WHAT:** Lemmas 1–3 claim bounds for all \(p\), and Proposition 3 claims its bound for every \(p\), but \(\mu=1.17\), \(L=3.05\), and \(\tilde L=10.56\) were evaluated for the true FHN vector field along the true orbit. The 20-parameter cubic library has a \(p\)-dependent state Jacobian; altered coefficients and altered trajectories need not satisfy those constants. Some bounded candidate parameters produce finite-time blow-up.  
   **WHY:** The reported constants cannot certify the bounds at \(p^{(K)}\), \(\hat p^{(K)}\), \(p_{\rm wrong}\), or even the parameter-perturbation probes. Empirically passing a finite probe grid is not a proof.  
   **WHAT TO DO:** Introduce a specified parameter set \(P\) and a forward-invariant or uniformly valid state region \(X\), then take suprema over \(x\in X,p\in P\), including all line segments needed by the proofs. Alternatively restore \(\mu(p)\), \(\tilde L(P,X)\), and application-specific constants. State clearly that the current numerical constants are empirical local estimates if that is all they are.

5. **The mean-value step needs assumptions not presently stated.**  
   **WHAT:**  
   \[
   f(x_1;p)-f(x_2;p)=\left(\int_0^1J_f(x_2+s(x_1-x_2);p)\,ds\right)(x_1-x_2)
   \]
   is valid only when \(f(\cdot;p)\) is \(C^1\) on an open set containing the entire joining segment. A “region containing the trajectories” need not contain those chords. The parameter Lipschitz estimate derived from \(\partial f/\partial p\) similarly needs the parameter segment between \(p_1,p_2\).  
   **WHY:** Without segment containment, neither \(A(t)\) nor the stated supremum bounds are justified.  
   **WHAT TO DO:** Require a convex region—or explicitly the union of all relevant state and parameter segments—on which \(f\) is \(C^1\) and the constants are finite.

6. **Lemma 2 divides by \(\|\delta\|\) exactly where \(\delta(0)=0\).**  
   **WHAT:** The inner-product inequality only yields the displayed differential inequality after division by \(\|\delta\|\). In Lemma 2, \(\delta(0)=0\), and unlike Lemma 1, parameter forcing permits later zeros without \(\delta\equiv0\). The uniqueness remark from Lemma 1 does not repair Lemma 2.  
   **WHY:** The proof as written is invalid at the initial point and any subsequent zero.  
   **WHAT TO DO:** Use the upper right Dini derivative,
   \[
   D^+\|\delta\|\le \mu\|\delta\|+\tilde L\|p_1-p_2\|,
   \]
   or regularize with \((\|\delta\|^2+\varepsilon^2)^{1/2}\) and pass to the limit. Lemma 1’s uniqueness argument is adequate only after saying that two equal-parameter solutions with distinct initial conditions cannot meet.

7. **\(\rho_s\) is not monotone for \(\mu<0\), so Lemma 3 and Proposition 3 are false in an explicitly advertised case.**  
   **WHAT:** Writing \(E=\|\eta\|_{\max}\) and \(r=\|p-p^\star\|\),
   \[
   \rho_s'=e^{\mu s}\big(\mu E+\tilde Lr\big).
   \]
   For \(p=p^\star\) and \(\mu<0\), \(\rho_s\) strictly decreases. Therefore replacing an offset \(r_0\le s\) by \(s\) is in the wrong direction. Likewise,
   \[
   e^{\mu(t_i-\tau_k)}\le e^{\mu\Delta T_1}
   \]
   is false when \(\mu<0\).  
   **WHY:** The dissipative case is presented as a qualitative advantage of the revision, yet the residual and node-removal bounds fail precisely in that case.  
   **WHAT TO DO:** Define the monotone envelope
   \[
   \bar\rho_s=\sup_{0\le r\le s}\left[(1+e^{\mu r})E+
   \frac{\tilde L}{\mu}(e^{\mu r}-1)\|p-p^\star\|\right]
   =\max\{2E,\rho_s\},
   \]
   with the continuous \(\mu=0\) extension. Replace \(e^{\mu\Delta T_1}\) by \(e^{\max(\mu,0)\Delta T_1}\), or retain the actual offsets.

8. **The crude cost bounds also reverse inequalities when \(\mu<0\).**  
   **WHAT:** From \(j\Delta t\le\Delta T\), the manuscript concludes
   \[
   (1+e^{\mu j\Delta t})^2\le(1+e^{\mu\Delta T})^2,
   \]
   and similarly for \(e^{2\mu j\Delta t}\). Both inequalities reverse for \(\mu<0\).  
   **WHY:** The second inequalities in (JK_true) and (JK_expect), and every Proposition 2 bound that uses them, are false for negative \(\mu\).  
   **WHAT TO DO:** Keep the valid per-datum sums. A simple uniform bound for \(\mu<0\) is \(4N\|\eta\|_{\max}^2\) and \(2Nd\sigma^2\), subject to corrected sample counts.

9. **The Gaussian expectation formula does not match the numerical noise model.**  
   **WHAT:** Theory assumes \(\eta_i\sim N(0,\sigma^2I_d)\), but the numerics assign each component a standard deviation equal to \(0.05\) times that component’s own trajectory standard deviation. That is generally anisotropic covariance, not \(\sigma^2I_d\).  
   **WHY:** The claims \(\mathbb E\|\eta_i\|^2=d\sigma^2\) and the quoted “noise slack” are undefined or wrong for the actual dataset unless the states were normalized first.  
   **WHAT TO DO:** Use \(\eta_i\sim N(0,\Sigma)\) and \(\mathbb E\|\eta_i\|^2=\operatorname{tr}\Sigma\), then recompute the slack. The cross term itself does vanish: the right endpoint \(t_i=\tau_k\) is still distinct from the launching node \(\tau_{k-1}\), and its later reuse as another node does not affect linearity of expectation.

10. **The coarse-cost definition is formally malformed.**  
    **WHAT:** In (Jhat), the range of \(k\) is omitted; if it includes \(K\), \(\tau_K^+\) is undefined. Worse, Section 2 defines \(\tau_k^+\) only for removed nodes, while (Jhat) uses it for nonremoved nodes.  
    **WHY:** The object bounded in Proposition 3 is not actually defined.  
    **WHAT TO DO:** Let \(R=\{0,\dots,K\}\setminus\mathcal I_R\), define \(\tau_r^+\) as the next node in \(R\), and sum over \(r\in R\setminus\{K\}\).

11. **The consecutive-node bookkeeping is salvageable, but the manuscript does not explain the crucial endpoint case.**  
    **WHAT:** For an isolated removed node \(\tau_k\), the datum at \(\tau_k\) retains its old predictor and should not be counted. For consecutive removals, the datum at a later removed node does change predictor and is counted as the right endpoint of \(D_{k-1}\). The present claim “only data in \(D_k\), \(k\in\mathcal I_R\), change predictor” is correct only after this argument and after fixing (Jhat).  
    **WHY:** Without the explanation, the half-open/closed conventions look like an off-by-one omission and are impossible to audit.  
    **WHAT TO DO:** Add the isolated-versus-consecutive argument explicitly and state that the \(D_k\) are disjoint fine windows, with \(|D_k|\le n_{\max}\).

12. **Figure 10’s \(\Delta T_1,\Delta T_2\) caption contradicts the definitions.**  
    **WHAT:** Starting from the finest partition and keeping every \(m\)-th node, the fine-window length is \(\Delta T_1=\Delta t\), not \(m\Delta t\). The maximum distance from a removed node to the remaining node on its left is approximately \((m-1)\Delta t\), not fixed at \(\Delta t\).  
    **WHY:** The figure’s interpretation of which factor is being varied is backwards. It cannot be presented as a check of Proposition 3 under the stated definitions.  
    **WHAT TO DO:** Correct the caption and any plotting labels/code derived from those quantities. Re-evaluate Gate G7 using the quantities actually represented by each experiment.

13. **Corollary 1 is not a valid consequence of the preceding proofs.**  
    **WHAT:** A time-dependent logarithmic norm gives
    \[
    \|\delta(t)\|\le e^{\int_0^t\mu(r)dr}\|\delta(0)\|,
    \]
    but Lemma 2 becomes
    \[
    \|\delta(t)\|\le \tilde L\|p_1-p_2\|
    \int_0^t\exp\!\left(\int_s^t\mu(r)\,dr\right)ds.
    \]
    There is no blanket substitution of \(e^{\mu\Delta T}\) by \(e^{\Lambda_k}\), especially for \((e^{\mu t}-1)/\mu\). Moreover, when \(\mu(t)\) changes sign, the full-window exponent \(e^{\Lambda_k}\) need not bound prefix or subinterval amplification. A window may contract overall after a large transient expansion. Proposition 3 requires precisely such subinterval bounds.  
    **WHY:** The trajectory-dependent refinement, its residual bounds, and its claimed node-placement rule are not proved.  
    **WHAT TO DO:** Define application-specific transition bounds
    \[
    M_k=\sup_{\tau_{k-1}\le a\le b\le\tau_k}
    \exp\!\left(\int_a^b\mu(r)\,dr\right),
    \]
    and use the correct variation-of-constants integral for parameter sensitivity. Each pair of trajectories and each parameter value needs its own segment-based \(\mu(r)\), or a rigorous supremum over a tube.

14. **The weighted-norm extension omits changed constants and changed costs.**  
    **WHAT:** In the \(D\)-norm, parameter forcing requires
    \[
    \tilde L_D=\sup\|D\,\partial_pf\|_{2\leftarrow2},
    \]
    noise becomes \(\|D\eta\|\), and \(\mathbb E\|D\eta\|^2=\sigma^2\operatorname{tr}(D^\top D)\) under isotropic noise. If the cost itself uses \(D\)-norm residuals, its minimizer differs from that of the Euclidean cost. If \(D\)-bounds are merely converted back to Euclidean residual bounds, squared-cost estimates acquire squared conversion factors.  
    **WHY:** “The same proof runs” and “an extra factor \(\kappa(D)\)” are insufficient and can be numerically wrong depending on which object is being bounded.  
    **WHAT TO DO:** State separate weighted lemmas, define \(\tilde L_D\) and weighted noise quantities, and trace the norm-equivalence factors through residuals, squared costs, and the final square root.

15. **The abstract compares quantities computed in different norms.**  
    **WHAT:** The measured peak \(8.5\) is Euclidean. The advertised “\(\le8\) per window” is the weighted endpoint exponent \(8.04\). Its stated Euclidean conversion is \(8.04\,\kappa(D)\approx28.5\), not \(8\). The direct Euclidean exponent is \(34.78\) at \(\Delta T=5\) and \(138.79\) at \(\Delta T=10\).  
    **WHY:** The abstract’s headline tightening is false as a Euclidean comparison. Additionally, the \(8.04\) value is only a first-order, reference-orbit endpoint estimate, not a rigorous finite-perturbation “at most” bound.  
    **WHAT TO DO:** Say “weighted linearized endpoint factor \(8.04\), corresponding to the Euclidean bound \(28.5\),” and qualify the 8.5 as the maximum observed over 24 probes, not something that “never” occurs beyond them.

16. **Theorem 1 is formally conditional but does not establish the claimed basin-tracking result.**  
    **WHAT:** It assumes the strong-convexity neighbourhood around \(p^{(K)}\) already contains \(\hat p^{(K)}\), the very fact a displacement theorem is supposed to help establish. Its right side also contains \(\Delta_K(\hat p^{(K)})\), hence \(\|\hat p^{(K)}-p^\star\|\), so it is an implicit inequality. Without a known parameter-radius bound or a smallness condition, it need not yield any finite a priori displacement control.  
    **WHY:** The theorem cannot justify that guess propagation remains inside the next basin. It only bounds two minimizers after assuming the new minimizer lies in the old convex neighbourhood.  
    **WHAT TO DO:** Assume strong convexity on a known ball, bound the perturbation on that ball, and impose a boundary/smallness condition proving the coarse minimizer remains inside it. Alternatively label the current theorem an a posteriori conditional estimate and stop claiming it establishes basin retention.

17. **The expectation version of Proposition 2 suppresses a random strong-convexity assumption.**  
    **WHAT:** \(J_K\), \(p^{(K)}\), and potentially its strong-convexity constant and valid neighbourhood depend on the noise. Taking expectations requires the pointwise premise to hold almost surely with a common deterministic \(m>0\), plus measurability and neighbourhood inclusion. None is stated.  
    **WHY:** One cannot simply “replace \(J_K(p^\star)\) by its expectation” if the inequality’s premise varies or fails across noise realizations.  
    **WHAT TO DO:** State an almost-sure uniform-\(m\) assumption, or retain a realization-wise bound and avoid the unconditional expectation claim.

18. **The claimed uniform convergence on bounded parameter sets is false without uniform existence.**  
    **WHAT:** Polynomial candidate systems can blow up in finite time for bounded coefficients. Continuity of the flow does not imply uniform convergence on every bounded set of \(p\) when the flow may cease to exist or approach a blow-up boundary.  
    **WHY:** \(J_K^\star\) and \(J_K\) may not even be finite on the claimed set.  
    **WHAT TO DO:** Restrict to a compact parameter set on which all required solutions exist up to \(\Delta T\) and remain in a common compact state region. Then prove uniformity there.

19. **\(p^\star=\arg\min J_K^\star\) asserts unproved identifiability.**  
    **WHAT:** \(J_K^\star(p^\star)=0\) proves only \(p^\star\in\arg\min J_K^\star\). Finite samples on one trajectory can leave library coefficients non-identifiable or admit multiple zero-cost models.  
    **WHY:** Uniqueness is central if parameter recovery is claimed rather than trajectory interpolation.  
    **WHAT TO DO:** Replace equality by membership unless a persistence-of-excitation or identifiability condition is supplied.

20. **The reported \(\mu_-\) cannot support the globally stated lower bound.**  
    **WHAT:** For true FHN, \(1-v^2\to-\infty\), so the global lower logarithmic norm is \(-\infty\), not \(-2.96\). The latter is only an orbit-restricted measurement. Lemma 1 nevertheless says “for all \(x_1,x_2\).”  
    **WHY:** The lower sensitivity bound is not certified for arbitrary perturbations, even though the upper \(\mu=1.17\) happens to be global for the true FHN field.  
    **WHAT TO DO:** Restrict the lower bound to a specified bounded region containing all joining segments, or omit the finite global \(\mu_-\) claim.

21. **Several numerical statements are observations masquerading as validation of theorem premises.**  
    **WHAT:** Passing 2,424 probe points does not verify a supremum-based bound. Monotonic decrease along one sampled straight segment does not prove that a point lies in a Nelder–Mead basin. Checking the data-term Hessian at an approximate penalized optimizer does not check strong convexity of the actual objective on a neighbourhood.  
    **WHY:** These figures are useful diagnostics, but they do not close the theoretical assumptions they are said to “check.”  
    **WHAT TO DO:** Relabel them as empirical diagnostics. For an actual premise check, bound the Hessian throughout a neighbourhood, include the regularizer, verify stationarity, and use constants valid on the relevant trajectory tube and parameter set.

VERDICT: ISSUES_REMAIN