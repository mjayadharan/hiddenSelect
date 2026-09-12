1. **BLOCKING — Restricting every minimizer to \(P\) changes the optimization problem relative to the code.**  
   **WHAT:** The revised theory defines
   \[
   p^{(K)}\in\arg\min_{p\in P}J_K(p),
   \]
   but the reported Nelder–Mead and BFGS experiments optimize unconstrained coefficient vectors. No compact constraint \(P\) is enforced by the implementation, and \(P\) has not even been instantiated.  
   **WHY:** The theoretical minimizers need not be the numerical minimizers. Statements about displacement, basin retention, boundary minimizers, and penalty bias therefore do not explain the reported optimizer trajectories without an additional containment result. The claim that shrinkage makes \(\partial P\)-solutions typical is especially unsupported: the code has no knowledge of \(\partial P\).  
   **WHAT TO DO:** Explicitly separate constrained theoretical minimizers from the unconstrained numerical outputs. To reconnect them, construct a concrete \(P\), verify that every relevant numerical minimizer and connecting neighbourhood lies in \(\operatorname{int}P\), and show it is also a minimizer of the restricted problem. Otherwise withdraw claims that these theorems analyze the numerical minimizers.

2. **BLOCKING — The reachable-state clause of Corollary `cor:confined` is false as written.**  
   **WHAT:** It says the same state- and parameter-sensitivity conclusions hold after replacing \(z\) by any reachable state \(\varphi(\sigma;p;z)\). Flow composition confines continuation under the same parameter \(p\), but it does not confine
   \[
   \varphi(s;p_1,\varphi(\sigma;p_0;z))
   \quad\text{and}\quad
   \varphi(s;p_2,\varphi(\sigma;p_0;z))
   \]
   for arbitrary \(p_0,p_1,p_2\). The reachable state may be within \(r_X\), not within the \(r'\) launch radius required by Lemma 0.  
   **WHY:** The parameter-sensitivity part of the clause is unproved. Proposition 3 then cites this clause to justify its coarse/fine comparison.  
   **WHAT TO DO:** Delete the blanket reachable-state sentence. In Proposition 3, argue directly: the coarse continuation and the fresh fine launch are each already known from Lemma 0 to remain in \(X\), so the local Lemma 1 applies to that particular pair. Only same-parameter continuation follows directly from the flow property.

3. **BLOCKING — Theorem 1’s constant-branch algebra is wrong by a factor of two.**  
   **WHAT:** When \(\mu<0\) and \(\tilde L\delta\le|\mu|E\), all three envelopes in
   \[
   \Delta_K=c_0\bar\rho_{\Delta T_2}
   \left(\bar\rho_{\Delta T_1+\Delta T_2}+\bar\rho_{\Delta T_1}\right)
   \]
   equal \(2E\). Therefore
   \[
   \Delta_K=c_0(2E)(2E+2E)=8c_0E^2,
   \]
   not \(4c_0E^2\).  
   **WHY:** The theorem’s explicit piecewise formula contradicts its own collapsed bound and underestimates it.  
   **WHAT TO DO:** Replace \(4c_0E^2\) by \(8c_0E^2\) everywhere and check any derived smallness condition using that branch.

4. **BLOCKING — The proposed “usable” node-selection score is not observable.**  
   **WHAT:** Although \(g_k(R;p)\) is computable, \(B(R;p)\) also contains
   \[
   \rho_s(p)=(1+e^{\mu s})\|\eta\|_{\max}
             +Q(s)\|p-p^\star\|.
   \]
   Both \(\|\eta\|_{\max}\) and \(p^\star\) are unknown in an actual discovery problem. Thus the complete score cannot be evaluated at the current iterate as claimed.  
   **WHY:** Making one factor observable does not make the proposed greedy rule implementable.  
   **WHAT TO DO:** Either require explicit prior upper bounds on noise and \(\|p-p^\star\|\), or use the observable pre-collapse expression
   \[
   e^{\mu(t_i-\tau_k)}g_k(p)\big(a_i+b_i\big),
   \]
   where the current fine residual \(b_i\) and proposed coarse residual \(a_i\) can both be evaluated. State the resulting simulation cost honestly.

5. **BLOCKING — The primary bound is still not the sharpest Grönwall bound claimed.**  
   **WHAT:** The “sharpest statement” retains actual time offsets but replaces the individual noise norms by the global \(\|\eta\|_{\max}\) inside both \(\rho\) factors. Lemma 3 already supplies the tighter datum-specific quantity
   \[
   r_{\tau,i}(p)=e^{\mu(t_i-\tau)}\|\eta_\tau\|
     +Q(t_i-\tau)\|p-p^\star\|+\|\eta_i\|.
   \]
   **WHY:** The project explicitly requires bounds as tight as Grönwall machinery permits. Global maximization belongs in a collapsed corollary, not the primary per-datum result.  
   **WHAT TO DO:** Use \(r_{\tau_k^-,i}(p)+r_{\tau_k,i}(p)\) in the primary theoretical bound. Then derive the \(\rho\), \(\bar\rho\), per-node, and global forms as successive simplifications.

6. **BLOCKING — P1 is called a failure even though it was never evaluated.**  
   **WHAT:** The abstract says the closure check at \(\Delta T_{\max}=100\) “fails,” while P1 says \(X\), \(P\), \(r_X\), \(\mu(X\times P)\), and \(\tilde L(X\times P)\) were never instantiated and that the displayed numbers are merely indicative. It also says (A1) has not been verified for any \(\kappa\).  
   **WHY:** “Unverified” and “failed” are not interchangeable. The manuscript cannot conclude failure from constants computed on a different region. Nor can it say the theory “supports the beginning” of the schedule when (A1) is unverified even there.  
   **WHAT TO DO:** Label P1 “NOT EVALUATED” or “UNVERIFIED.” Say the estimates suggest long-window infeasibility. Until a valid \(X\times P\) calculation is performed, describe every stage—including short windows—as conditional, not certified.

7. **MINOR — \(\Lambda_R\) is not a new horizon beyond the coarse partitions.**  
   **WHAT:** For a consecutive removed block, the largest \(h_k+w_k\) occurs at the last removed node and equals the affected coarse-window length from the retained node on the left to the retained node on the right. Thus \(\Lambda_R\) is a coarse-window length. The earlier requirement that \(\Delta T_{\max}\) cover every partition under discussion already covers it if the coarse partition is included.  
   **WHY:** The text announces a new horizon gap that does not actually exist and incorrectly says \(h_k+w_k\) is not a coarse window.  
   **WHAT TO DO:** Keep \(\Lambda_R\) as useful notation for the affected coarse span if desired, but remove the false claim and state that the original “all fine and coarse partitions” horizon already suffices.

8. **MINOR — The claim that enlarging \(r_X\) cannot rescue (A1-i) is false in general.**  
   **WHAT:** Suprema are non-decreasing as \(X\) grows, but they need not be strictly increasing for a polynomial field. Constant and linear polynomial systems are immediate counterexamples; even the true FHN upper logarithmic norm is global and independent of \(r_X\). Non-decreasing constants also do not by themselves prove that \(Q(r_X)\) outgrows \(r_X\).  
   **WHY:** The manuscript turns a system-dependent warning into a theorem.  
   **WHAT TO DO:** Say enlarging \(r_X\) may fail because the constants must be recomputed and can grow rapidly. Make any stronger statement only after proving it for the specified library and parameter set.

9. **MINOR — Several envelope claims mishandle degenerate cases.**  
   **WHAT:** For \(\mu<0\), Lemma `rhobar`(v) says \(\bar\rho_s=2E\) exactly under the threshold condition, but at \(s=0\) it equals \(2E\) for every parameter distance. The remark says \(\bar\rho_s=\rho_s^+\) iff \(\mu\ge0\) or \(p=p^\star\), but equality also holds when \(E=0\) and at \(s=0\).  
   **WHY:** The “if and only if” claims are false at legitimate edge cases.  
   **WHAT TO DO:** State the branch characterization for \(s>0\), and include \(E=0\) and \(s=0\) in the equality cases.

10. **MINOR — Negative linearized window exponents are being conflated with a negative uniform \(\mu\).**  
    **WHAT:** The \(\bar\rho\) improvement applies when the uniform one-sided Lipschitz constant used in the residual proof satisfies \(\mu<0\). Table 2 instead reports some negative integrated, weighted, reference-orbit endpoint exponents. FHN’s global Euclidean \(\mu\) is positive, and even the global weighted supremum is positive. The local exponents are explicitly not licensed substitutions into the cost bounds.  
    **WHY:** The remark claims the \(\bar\rho\)/\(\rho^+\) distinction “lives” in a numerical regime that the theory has just said is not interchangeable with uniform \(\mu<0\).  
    **WHAT TO DO:** Motivate \(\bar\rho\) using genuinely contractive systems with uniform \(\mu<0\). Do not cite Table 2 as an instance of that branch.

11. **MINOR — The greedy cost accounting omits the number of candidates.**  
    **WHAT:** Testing one candidate can require \(O(|R|)\) recomputations in the worst case. Testing every retained candidate in one greedy step can therefore cost \(O(C|R|)\) window integrations for \(C\) candidates, not merely \(O(|R|)\).  
    **WHY:** The proposed implementation is made to sound substantially cheaper than the stated algorithm.  
    **WHAT TO DO:** Give per-candidate and per-greedy-step costs separately, noting possible caching.

VERDICT: ISSUES_REMAIN