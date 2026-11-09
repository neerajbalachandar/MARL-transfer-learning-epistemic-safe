# Algorithm mapping and reference audit

The mapping below separates paper mechanisms from this repository's adaptations. Links point to the reference material; the implementation should not be described as faithful where the listed gaps remain.

| Method | Reference mechanism | Current code | Status and adaptation |
|---|---|---|---|
| MAPPO | Centralized training with decentralized policies and PPO updates | BenchMARL's pinned MAPPO implementation, configured in `transfer/experiment.py` | Baseline uses continuous VMAS actions and task-specific goal layouts. It is not the `marlbenchmark/on-policy` codebase. |
| MAPTF | Peer transfer with state representation and successor-representation options | `transfer/maptf.py`, `experiments/transfer_maptf.py` | Pilot only: frozen peers, imitation/KL-style regularization, online peer selection and batch termination. No SRO update equations or environment-persistent options. |
| BPR-inspired reuse | Belief over source-policy hypotheses, updated from observed transfer evidence | `transfer/methods.py`, `experiments/transfer_bayesian_reuse.py` | Factorized likelihood baseline and a joint categorical success/contact plus conditional-Gaussian return model. The transfer copies source features and action means, resets the target action scale to 1, and uses 4 PPO epochs at learning rate `1e-5`. This avoids the non-finite actor updates seen when reusing the source's nearly zero action scales on another task. The joint model is a tractable approximation, not a validated generative model. Not Bayes-ToMoP. |
| PORTAL-style curriculum | Choose candidate tasks by current-policy difficulty and target-task representation similarity | `transfer/methods.py`, `experiments/transfer_portal.py` | Fixed-budget pilot. Similarity is an aligned Gaussian-component mean-distance proxy. No full GMM KL, competence stopping, or paper-equivalent component adaptation. |
| Epistemic knowledge | A proposition is known when true in every indistinguishable possible world | `transfer/epistemic.py` | Exact only for the finite worlds supplied by caller; no VMAS state discretizer or sensor model yet. |
| Geometric safety filter | Filter nominal actions against a discrete barrier constraint | `transfer/safety.py` | Nominal one-step global action scaling utility. Not integrated with MAPPO and not a formal safety guarantee. |

## Reference source locations

- MAPTF paper: [NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/8d9a6e908ed2b731fb96151d9bb94d49-Abstract.html); code: [tianpeiyang/MAPTF_code](https://github.com/tianpeiyang/MAPTF_code), especially `alg/option/option_ma_sro.py`, `alg/muti_ptf_ppo/ppo.py`, and `alg/sharing_multi_ppo/`.
- MAPPO reference: [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy). This repository instead relies on BenchMARL's pinned package; see the recorded dependency revision in each manifest.
- BPR: [Bayesian Policy Reuse](https://arxiv.org/abs/1505.00284). The transfer runner implements source-policy hypothesis updates but not opponent modelling or Theory of Mind.
- PORTAL: [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/29524); reference code is under [TJU-DRL-LAB transfer repository](https://github.com/TJU-DRL-LAB/transfer-and-multi-task-reinforcement-learning), in `Multi-agent Transfer RL/Transfer across tasks/PORTAL/OO_task_similarity.py`.
- VMAS: [repository](https://github.com/facebookresearch/vmas); navigation reward and completion semantics are taken from the repository's pinned local checkout.
- BenchMARL: [repository](https://github.com/facebookresearch/BenchMARL); validation callbacks use its evaluation-end callback interface.
- EPIRL candidate specified in the task: [yoon-gu/epirl](https://github.com/yoon-gu/epirl). Its README describes epidemiology simulation tooling, so it does not supply the intended epistemic-logic navigation implementation. A separate AAMAS paper titled *A Simple Integration of Epistemic Logic and Reinforcement Learning* is [here](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p686.pdf); it is not evidence that this navigation code has reproduced that method.

## VMAS adaptation

The navigation scenario has three holonomic agents and continuous force actions. The pinned scenario uses two physics substeps and the simulator's semi-implicit Euler integration. Scenario collision reward applies near contact using signed surface distance; physical overlap and contact are tracked separately. Completion requires every agent to be within its own radius-based goal threshold at the same time. See `evaluation/metrics.py`, `evaluation/collision_metrics.py`, and the local pinned VMAS `scenarios/navigation.py` and `simulator/core.py` for exact code.

The PORTAL observation comparison uses the current fixed-size local observation vectors. The MAPTF and epistemic papers' assumptions do not transfer automatically to this continuous-action partially observed setup; equivalent updates and uncertainty representations must be derived and tested before claiming reproduction.
