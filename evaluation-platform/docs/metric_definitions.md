# Metric definitions

The registry in `config/metric_registry.yaml` defines the VMAS pilot's primary metric as the fraction of episodes where all agents simultaneously reach their assigned goals. It is dimensionless, in `[0,1]`, higher is better, and uses `environment_frames` as its x-axis unit. Its normalized learning-curve budget is the pilot default of 600,000 frames; records beyond that budget are clipped only for AUC calculation, not removed from exports.

Team return is retained as a secondary native metric and is only comparable inside the same benchmark semantics. Episode length is a count of native policy decisions, not simulator physics substeps. VMAS environment frames and policy decisions remain separate.

ZSC-Eval JSON scalar keys are preserved exactly. The adapter labels them as native scalar exports unless the key is the BR-Div selection determinant emitted by the upstream selection script. No opaque scalar is silently relabeled as BR-Prox, success, or reward.
