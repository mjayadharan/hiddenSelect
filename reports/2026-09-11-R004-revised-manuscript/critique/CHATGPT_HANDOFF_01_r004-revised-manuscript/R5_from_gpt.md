1. **BLOCKING — The new FHN claim about enlarging \(r_X\) uses constants that are not global.**  
   **WHAT:** Block (H5) says that once \(X\) is sufficiently large, both \(\mu=\nmu\) and \(\tilde L=\nLtilde\) become global, so choosing
   \[
   r_X>Q(\Delta T_{\max})\operatorname{diam}(P)
   \]
   makes (A1-i) satisfiable. Only the true-field upper logarithmic norm is global in \(x\) at \(p^\star\). For the cubic candidate library,
   \[
   \frac{\partial f}{\partial p}
   \]
   contains monomials through degree three, so \(\tilde L(X)\) grows without bound as \(X\) expands. Moreover, \(\mu=1.17\) is not uniform over \(P\); candidate coefficients alter the Jacobian and may introduce positive unbounded growth.  
   **WHY:** The conclusion that an enormous tube buys admissibility is unsupported and generally false for the actual candidate family. It also contradicts the manuscript’s earlier, correct warning that the measured constants apply only at \(p^\star\) along the probe region.  
   **WHAT TO DO:** Delete the FHN-specific “satisfiable by inflating the tube” calculation. State only that feasibility must be determined jointly from \(\mu(X\times P)\), \(\tilde L(X\times P)\), \(r_X\), and \(P\). The values \(1.17\) and \(10.56\) cannot be held fixed while \(X\) is enlarged.

2. **BLOCKING — The common-refinement bound still uses the obsolete loose Proposition 3 formula.**  
   **WHAT:** Corollary 3 remains inherited from R3 and defines
   \[
   \Delta^{C\to S}
   =2n_{\max}^C|C\setminus S|e^{\mu^+\Delta T_1^C}
     \bar\rho_{\Delta T^S}\bar\rho_{\Delta T_1^C+\Delta T^S}.
   \]
   R5’s corrected collapsed node-removal bound is instead
   \[
   n_{\max}^C|C\setminus S|e^{\mu^+\Delta T_1^C}
   \bar\rho_{\Delta T^S}
   \left[
     \bar\rho_{\Delta T_1^C+\Delta T^S}
     +\bar\rho_{\Delta T_1^C}
   \right].
   \]
   The old expression remains a valid but weaker upper bound because the second bracket is at most twice its longer envelope.  
   **WHY:** General repartition is the theorem that actually covers almost all schedule transitions. Leaving its principal bound at the discarded factor-two relaxation violates the stated tightness objective and the claimed “consequential inheritance” of the corrected Proposition 3.  
   **WHAT TO DO:** Replace \(\Delta^{C\to S}\) by the corrected bracketed expression, then update \(\Delta_{A,B}\), its truth specialization, and any downstream displacement constants.

3. **BLOCKING — Hypothesis (R) is insufficient for the localized Proposition 2 claim.**  
   **WHAT:** (R) requires \(p_{\rm num}\in\operatorname{int}U\), stationarity there, and strong convexity on \(U\), but does not require \(p^\star\in U\). Proposition 2 applies the local minimizer inequality specifically at \(q=p^\star\).  
   **WHY:** Without \(p^\star\in U\), neither
   \[
   J_K(p^\star)\ge J_K(p_{\rm num})
      +\frac m2\|p^\star-p_{\rm num}\|^2
   \]
   nor the advertised parameter-displacement bound follows.  
   **WHAT TO DO:** Add \(p^\star\in U\) to the conditions for the localized Proposition 2. It need not be part of the weaker hypothesis used solely for basin retention, so state result-specific variants if desired.

4. **MINOR — Ranking by the “exact gap” needs an absolute value or a different objective.**  
   **WHAT:** The observable quantity is
   \[
   \hat J_K(p)-J_K(p)=\sum(a_i^2-b_i^2),
   \]
   which is signed. The perturbation theory controls its absolute value. Choosing the “smallest” signed gap could favor a very large negative perturbation rather than a small change between objectives.  
   **WHY:** The proposed practical rule is ambiguous and can optimize the opposite quantity from the theorem.  
   **WHAT TO DO:** If the aim is partition continuity, rank using
   \[
   |\hat J_K(p)-J_K(p)|.
   \]
   If large negative changes are deliberately preferred, state that this is a separate heuristic unrelated to minimizing the perturbation bound.

5. **MINOR — The artifact contradicts itself about assumption (A3).**  
   **WHAT:** The consolidated table says “no (A3) exists; the label is not used,” but authoritative R3 block (F), Proposition 2, explicitly introduces an assumption labelled (A3), invokes it in the proof, and discusses it in the following `\readthis`.  
   **WHY:** The consolidated state—the document intended to drive the actual edit—is not internally reliable.  
   **WHAT TO DO:** Retain and list (A3), preferably under a descriptive name such as “uniform random strong-convexity assumption,” or rename it consistently in Proposition 2 and the scope table.

6. **MINOR — The claimed equality cases for \(\bar\rho=\rho^+\) still omit \(\tilde L=0\).**  
   **WHAT:** For \(\mu<0\), \(s>0\), \(E>0\), and \(p\ne p^\star\), the remark claims strict inequality. If \(\tilde L=0\), however, \(Q(s)=0\), so
   \[
   \rho_s^+=2E=\bar\rho_s
   \]
   regardless of \(p-p^\star\).  
   **WHY:** The “exactly when” statement is false for a parameter-independent field, which is permitted by the assumptions.  
   **WHAT TO DO:** Add \(Q(s)\|p-p^\star\|=0\), equivalently \(\tilde L=0\) or \(p=p^\star\), to the equality characterization.

7. **MINOR — P1’s narrative still overstates what the indicative constants establish.**  
   **WHAT:** P1 correctly says (A1) was not evaluated, but then says the indicative values show it “fails for long windows at any tube radius of usable size.” “Usable size” is undefined, and after issue 1 there is no valid extrapolation of \(\tilde L=10.56\) or \(\mu=1.17\) to enlarged tubes.  
   **WHY:** The text retreats from “failed” to “not evaluated” and then effectively reintroduces failure through an undefined qualifier.  
   **WHAT TO DO:** Say only that inserting the probe-region constants produces enormous factors for long windows and motivates—but does not establish—infeasibility of a useful admissible region.

8. **MINOR — Hypothesis (R) alone does not reconnect all aspects of the numerical experiment.**  
   **WHAT:** The reported point comes from a discretized, penalized objective, while (R) assumes exact stationarity and strong convexity of the exact-flow, unpenalized \(J_K\). The noisy-cost results also require the realized event \(E_X\).  
   **WHY:** Saying “what reconnects them is (R)” is incomplete even though P2 later admits that (R) is unverified.  
   **WHAT TO DO:** Say reconnection requires (R), \(E_X\), and the separately stated penalty/discretization qualifications. In particular, stationarity of the penalized numerical objective does not imply the stationarity appearing in (R).

VERDICT: ISSUES_REMAIN