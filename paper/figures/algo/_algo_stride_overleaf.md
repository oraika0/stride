\documentclass{article}
\usepackage[margin=1.5in]{geometry}
\usepackage{amsmath,amssymb,bm}
\usepackage{algpseudocode}
\usepackage{float}

% ruled 外框:上下橫線 + 標題下橫線(不用 algorithm 套件，避免與 \newfloat 衝突)
\floatstyle{ruled}
\newfloat{algorithm}{t}{lop}
\floatname{algorithm}{Algorithm}

% Input / Output 標籤
\renewcommand{\algorithmicrequire}{\textbf{Input:}}
\renewcommand{\algorithmicensure}{\textbf{Output:}}
\algnewcommand\Initialize{\item[\textbf{Initialize:}]}

\begin{document}
\pagestyle{empty}

\begin{algorithm}[t]
\caption{Training STRIDE for SDN traffic engineering}
\label{alg:stride-training}
\small
\begin{algorithmic}[1]
\Require discount factor $\gamma$, importance-ratio clip threshold $\rho_{\max}$, reverse steps $M$, batch size $bs$, sequence length $L$, actor learning rate $\alpha_\theta$, critic learning rate $\alpha_\phi$, encoder learning rate $\alpha_\omega$, target soft-update coefficient $\tau$, control delay $t_d$, total monitoring periods $T$
\Statex\hspace*{-\leftmargin}\makebox[\dimexpr\linewidth+\leftmargin\relax][l]{\hrulefill}
\Initialize encoder $\omega$, policy $\theta$, critic $\phi$, target networks $\bar{\omega}$ and $\bar{\phi}$, empty replay buffer $\mathcal{D}$, per-pair candidate-link masks $\{\mathbf{b}_i\}_{i\in\mathcal{I}}$, initial hidden states $\{\mathbf{h}_{i,0}^{\mathrm{pair}}\}_{i\in\mathcal{I}}$
\Statex\hspace*{-\leftmargin}\makebox[\dimexpr\linewidth+\leftmargin\relax][l]{\hrulefill}
\For{$t = 1, 2, \ldots, T$}
    \State Observe link state $\mathbf{s}_t$
    \State Pass $(\mathbf{s}_t, \{\mathbf{b}_i\}_{i\in\mathcal{I}}, \{\mathbf{h}_{i,t-1}^{\mathrm{pair}}\}_{i\in\mathcal{I}})$ through the link encoder to obtain $\{\mathbf{h}_{i,t}^{\mathrm{pair}}\}_{i\in\mathcal{I}}$
    \State Initialize $x_{M,i} = [\mathrm{MASK}]$ for all $i$
    \For{$m = M, M-1, \ldots, 1$}
        \State Pass $(x_m, \{\mathbf{h}_{i,t}^{\mathrm{pair}}\}_{i\in\mathcal{I}}, \{\hat{\mathbf{e}}_{0,i}^{(m)}\}_{i\in\mathcal{I}}, m)$ through the decoder to obtain $\{\tilde{\mathbf{p}}^{(m)}_i\}_{i\in\mathcal{I}}$ in Eq.~(14)
        \State Compute the reverse transition distributions $\{\psi^{(m)}_i\}_{i\in\mathcal{I}}$ from $\{\tilde{\mathbf{p}}^{(m)}_i\}_{i\in\mathcal{I}}$ in Eq.~(15)
        \State Sample $x_{m-1}$ from $\{\psi^{(m)}_i\}_{i\in\mathcal{I}}$
    \EndFor
    \State Set the joint action $\mathbf{a}_t = x_0$ and record the behavior-policy probabilities $\{\pi_{\beta}(a_{i,t}\mid\mathbf{s}_t)\}_{i\in\mathcal{I}}$
    \State Store $\mathbf{a}_t$ into the routing-path repository
    \If{$t \geq t_d + 2$}
        \State Compute the reward vector $\mathbf{r}_t$ for the delay-aligned effective action in Eq.~(22)
        \State Store the delay-corrected transition $(\mathbf{s}_t, \mathbf{a}_t, \mathbf{r}_t, \mathbf{s}_{t+1}, \{\pi_{\beta}(a_{i,t}\mid\mathbf{s}_t)\}_{i\in\mathcal{I}})$ into $\mathcal{D}$
    \EndIf
    \If{$|\mathcal{D}| \geq bs$}
        \State Sample a mini-batch of $bs/L$ experience sequences of $L$ consecutive timesteps from $\mathcal{D}$
        \State Recompute the current-policy probabilities $\{\pi_{\theta}(a_{i,t}\mid\mathbf{s}_t)\}_{i\in\mathcal{I}}$
        \State Compute the clipped importance ratios $\{\rho_{i,t}\}_{i\in\mathcal{I}}$ in Eq.~(23)
        \State Compute the TD targets $\{y_{i,t}\}_{i\in\mathcal{I}}$ in Eq.~(25)
        \State Compute the advantages $\{\delta_{i,t}\}_{i\in\mathcal{I}}$ in Eq.~(26)
        \State Update the critic $\phi$ and encoder $\omega$ by minimizing $\mathcal{L}_{\mathrm{critic}}$ in Eq.~(27)
        \State Update the actor $\theta$ and encoder $\omega$ by minimizing $\mathcal{L}_{\mathrm{actor}}$ in Eq.~(28)
        \State Soft-update the target networks: $\bar{\omega} \leftarrow \tau\omega + (1-\tau)\bar{\omega}$,\ \ $\bar{\phi} \leftarrow \tau\phi + (1-\tau)\bar{\phi}$
    \EndIf
    \State Wait for the next monitoring period
\EndFor
\Statex\hspace*{-\leftmargin}\makebox[\dimexpr\linewidth+\leftmargin\relax][l]{\hrulefill}
\Statex \textbf{Output:} trained STRIDE model
\end{algorithmic}
\end{algorithm}

\end{document}
