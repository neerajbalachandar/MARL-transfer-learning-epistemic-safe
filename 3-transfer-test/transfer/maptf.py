"""A compact MAPTF-style peer imitation update for the BenchMARL MAPPO loop.

This pilot uses a learned reward-bandit value for peer selection and a small
reward-change termination classifier. It is an explicit approximation of the
paper's option-value/SRO updates, not a faithful reproduction of those updates.
"""

import torch
import torch.nn.functional as F

from benchmarl.experiment.experiment import Experiment
from transfer.experiment import assert_policy_compatible
from transfer.methods import continuous_policy_kl, freeze_policy


class MAPTFExperiment(Experiment):
    def __init__(
        self, *args, source_policy, transfer_weight=0.1, adaptive=True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert_policy_compatible(source_policy, self.policy)
        self.source_policy = freeze_policy(source_policy).to(self.config.train_device)
        self.transfer_weight = float(transfer_weight)
        self.adaptive = bool(adaptive)
        self.option_values = torch.zeros((3, 3), device=self.config.train_device)
        self.term_weight = torch.zeros((1, 2), device=self.config.train_device, requires_grad=True)
        self.term_bias = torch.zeros(1, device=self.config.train_device, requires_grad=True)
        self.term_optimizer = torch.optim.SGD(
            [self.term_weight, self.term_bias], lr=0.02
        )
        self.previous_return = None
        self.selected_sources = {}
        self.option_log = []
        self.option_duration = [0] * 3
        self.active_sources = [None] * 3
        self.option_steps = 0
        for parameter in self.source_policy.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _policy_params(policy, data):
        output = policy(data.clone())
        return output.get(("agents", "loc")), output.get(("agents", "scale"))

    def _peer_transfer_loss(self, data):
        observation = data.get(("agents", "observation"))
        current_loc, current_scale = self._policy_params(self.policy, data)
        losses = []
        selected = []
        source_count = current_loc.shape[-2]
        reward = data.get(("agents", "reward"), None)
        if reward is None:
            reward = data.get(("next", "agents", "reward"))
        reward_signal = float(reward.detach().mean().item())

        for receiver in range(source_count):
            peers = [peer for peer in range(source_count) if peer != receiver]
            option = int(self.option_values[receiver, peers].argmax().item()) if self.adaptive else 0
            peer = peers[option]

            # Call-and-return is approximated at batch boundaries. A small
            # classifier learns whether a lower return should end the option.
            obs_summary = observation[..., receiver, :].detach().abs().mean()
            reward_change = 0.0 if self.previous_return is None else reward_signal - self.previous_return
            features = torch.tensor(
                [[float(obs_summary), reward_change]],
                device=self.config.train_device,
                dtype=current_loc.dtype,
            )
            stop_probability = torch.sigmoid(F.linear(features, self.term_weight, self.term_bias))
            if self.adaptive and self.previous_return is not None and stop_probability.item() > 0.5:
                option = 1 - option
                peer = peers[option]
            should_stop = float(self.previous_return is not None and reward_change < 0)
            if self.adaptive:
                term_loss = F.binary_cross_entropy(
                    stop_probability.clamp(1e-6, 1 - 1e-6),
                    torch.tensor([[should_stop]], device=self.config.train_device, dtype=current_loc.dtype),
                )
                self.term_optimizer.zero_grad()
                term_loss.backward()
                self.term_optimizer.step()

            # Attribute the observed team return to the selected peer option.
            if self.adaptive:
                old_value = self.option_values[receiver, peer]
                self.option_values[receiver, peer] = old_value + 0.1 * (reward_signal - old_value)

            repeated = observation[..., receiver : receiver + 1, :].expand_as(observation)
            source_data = data.clone()
            source_data.set(("agents", "observation"), repeated)
            with torch.no_grad():
                source_loc, source_scale = self._policy_params(self.source_policy, source_data)
            losses.append(
                continuous_policy_kl(
                    source_loc[..., peer, :],
                    source_scale[..., peer, :],
                    current_loc[..., receiver, :],
                    current_scale[..., receiver, :],
                )
            )
            selected.append(peer)
            if self.active_sources[receiver] != peer:
                if self.active_sources[receiver] is not None:
                    self.option_log.append(
                        {
                            "receiver": receiver,
                            "source": self.active_sources[receiver],
                            "duration_updates": self.option_duration[receiver],
                        }
                    )
                self.active_sources[receiver] = peer
                self.option_duration[receiver] = 0
            self.option_duration[receiver] += 1
            option_record = {"receiver": receiver, "source": peer}
            if self.adaptive:
                option_record["termination_probability"] = float(stop_probability.detach().item())
            self.option_log.append(option_record)

        self.previous_return = reward_signal
        self.selected_sources[self.total_frames] = selected
        self.option_steps += 1
        return torch.stack(losses).mean()

    def _optimizer_loop(self, group):
        data = self.replay_buffers[group].sample().to(self.config.train_device)
        loss_values = self.losses[group](data)
        if group == "agents" and self.transfer_weight:
            loss_values["loss_objective"] = (
                loss_values["loss_objective"]
                + self.transfer_weight * self._peer_transfer_loss(data)
            )
        training_data = loss_values.detach()
        loss_values = self.algorithm.process_loss_vals(group, loss_values)
        for loss_name, loss_value in loss_values.items():
            optimizer = self.optimizers[group].get(loss_name)
            if optimizer is None:
                continue
            loss_value.backward()
            grad_norm = self._grad_clip(optimizer)
            training_data.set(
                f"grad_norm_{loss_name}",
                torch.tensor(grad_norm, device=self.config.train_device),
            )
            optimizer.step()
            optimizer.zero_grad()
        self.replay_buffers[group].update_tensordict_priority(data)
        if self.target_updaters[group] is not None:
            self.target_updaters[group].step()
        callback_loss = self._on_train_step(data, group)
        if callback_loss is not None:
            training_data.update(callback_loss)
        return training_data
