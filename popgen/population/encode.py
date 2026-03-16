"""Reusable encoders for population-level evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import litellm
import numpy as np
import torch
from torch import nn

from popgen.utils.config import LitellmConfig


class Encoder(ABC):
    """Base encoder with single-item and batched interfaces."""

    @abstractmethod
    def encode(self, input: Any) -> np.ndarray:
        """Encode one input item."""

    @abstractmethod
    def batch_encoder(self, inputs: list[Any]) -> np.ndarray:
        """Encode multiple inputs."""


class TextEncoder(Encoder):
    """Text embedder backed by LiteLLM embedding endpoints."""

    def __init__(self, embedding_model: str | LitellmConfig, batch_size: int = 32, num_workers: int = 8):
        if isinstance(embedding_model, str):
            embedding_model = LitellmConfig(model_name=embedding_model)
        self.embedding_config = embedding_model
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

    def _to_text(self, item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict) and "description" in item:
            return str(item["description"])
        return str(item)

    def encode(self, input: Any) -> np.ndarray:
        embeddings = self.batch_encoder([input])
        return embeddings[0]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = litellm.embedding(
            model=self.embedding_config.model_name,
            input=batch,
            **self.embedding_config.get_completion_kwargs(),
        )
        return [item["embedding"] if isinstance(item, dict) else item.embedding for item in response.data]

    def batch_encoder(self, inputs: list[Any]) -> np.ndarray:
        if len(inputs) == 0:
            return np.zeros((0, 0), dtype=float)

        texts = [self._to_text(item) for item in inputs]
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

        if len(batches) <= 1 or self.num_workers <= 1:
            all_embeddings: list[list[float]] = []
            for batch in batches:
                all_embeddings.extend(self._embed_batch(batch))
        else:
            from pqdm.threads import pqdm

            results = pqdm(batches, self._embed_batch, n_jobs=min(len(batches), self.num_workers), disable=True)
            all_embeddings = [emb for batch_embs in results for emb in batch_embs]

        return np.asarray(all_embeddings, dtype=float)


class TrajectoryEncoder(Encoder):
    """Trajectory encoder wrapper around train_cl models."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        action_pad_idx: int | None = None,
        agent_name: str = "ego_agent",
        group_name: str | None = "ego_team",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.action_pad_idx = int(action_pad_idx) if action_pad_idx is not None else int(getattr(model, "n_actions", 0))
        self.agent_name = agent_name
        self.group_name = group_name

    def _flatten_numeric(self, value: Any) -> list[float]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return self._flatten_numeric(value.tolist())
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, bool):
            return [float(value)]
        if isinstance(value, (int, float)):
            if np.isnan(value):
                return []
            return [float(value)]
        if isinstance(value, dict):
            out: list[float] = []
            for item in value.values():
                out.extend(self._flatten_numeric(item))
            return out
        if isinstance(value, (list, tuple)):
            out: list[float] = []
            for item in value:
                out.extend(self._flatten_numeric(item))
            return out
        return []

    def _extract_agent_payload(self, step: dict[str, Any]) -> dict[str, Any] | None:
        candidate_agent_names = [self.agent_name]
        if self.agent_name == "ego_agent":
            candidate_agent_names.append("agent0")
        candidate_group_names: list[str] = []
        if self.group_name is not None:
            candidate_group_names.append(self.group_name)
            if self.group_name == "ego_team":
                candidate_group_names.extend(["ego", "agents", "agent_blue"])
            elif self.group_name == "opponent_team":
                candidate_group_names.extend(["opponent", "agent_red"])

        agents = step.get("agents")
        if isinstance(agents, dict):
            for gname in candidate_group_names:
                if isinstance(agents.get(gname), dict):
                    for aname in candidate_agent_names:
                        payload = agents[gname].get(aname)
                        if isinstance(payload, dict):
                            return payload

            for aname in candidate_agent_names:
                payload = agents.get(aname)
                if isinstance(payload, dict):
                    return payload

            for maybe_group in agents.values():
                if not isinstance(maybe_group, dict):
                    continue
                for aname in candidate_agent_names:
                    payload = maybe_group.get(aname)
                    if isinstance(payload, dict):
                        return payload

        for gname in candidate_group_names:
            if isinstance(step.get(gname), dict):
                for aname in candidate_agent_names:
                    payload = step[gname].get(aname)
                    if isinstance(payload, dict):
                        return payload

        for key, value in step.items():
            if key in {"step", "env"} or not isinstance(value, dict):
                continue
            for aname in candidate_agent_names:
                payload = value.get(aname)
                if isinstance(payload, dict):
                    return payload

        return None

    def _coerce_action(self, action: Any) -> int:
        flat = self._flatten_numeric(action)
        if not flat:
            return 0
        return int(round(float(flat[0])))

    def _trajectory_to_arrays(self, trajectory: Any) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(trajectory, dict) and "states" in trajectory and "actions" in trajectory:
            states = np.asarray(trajectory["states"], dtype=np.float32)
            actions = np.asarray(trajectory["actions"], dtype=np.int64)
            if states.ndim == 1:
                states = states[:, None]
            if actions.ndim != 1:
                actions = actions.reshape(-1)
            t = min(states.shape[0], actions.shape[0])
            return states[:t], actions[:t]

        if isinstance(trajectory, np.ndarray):
            states = np.asarray(trajectory, dtype=np.float32)
            if states.ndim == 1:
                states = states[:, None]
            actions = np.zeros((states.shape[0],), dtype=np.int64)
            return states, actions

        if not isinstance(trajectory, list):
            trajectory = [trajectory]

        states_rows: list[list[float]] = []
        actions_rows: list[int] = []

        for step in trajectory:
            state_val: Any = None
            action_val: Any = None

            if isinstance(step, dict):
                payload = self._extract_agent_payload(step)
                if payload is not None:
                    state_val = payload.get("state")
                    action_val = payload.get("action")
                else:
                    state_val = step.get("state")
                    action_val = step.get("action")
            else:
                state_val = step

            state_flat = self._flatten_numeric(state_val)
            if not state_flat:
                continue
            states_rows.append(state_flat)
            actions_rows.append(self._coerce_action(action_val))

        if not states_rows:
            return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        max_dim = max(len(row) for row in states_rows)
        states = np.zeros((len(states_rows), max_dim), dtype=np.float32)
        for i, row in enumerate(states_rows):
            states[i, : len(row)] = np.asarray(row, dtype=np.float32)

        actions = np.asarray(actions_rows, dtype=np.int64)
        return states, actions

    def encode(self, input: Any) -> np.ndarray:
        embeddings = self.batch_encoder([input])
        return embeddings[0]

    def batch_encoder(self, inputs: list[Any]) -> np.ndarray:
        if len(inputs) == 0:
            return np.zeros((0, 0), dtype=float)

        trajectories = [self._trajectory_to_arrays(item) for item in inputs]
        lengths = [int(actions.shape[0]) for _, actions in trajectories]
        max_t = max(max(lengths), 1)
        state_dim = max(states.shape[1] if states.ndim == 2 else 1 for states, _ in trajectories)

        states_batch = np.zeros((len(inputs), max_t, state_dim), dtype=np.float32)
        actions_batch = np.full((len(inputs), max_t), self.action_pad_idx, dtype=np.int64)
        mask_batch = np.zeros((len(inputs), max_t), dtype=bool)

        for i, (states, actions) in enumerate(trajectories):
            t = min(states.shape[0], actions.shape[0], max_t)
            if t == 0:
                continue
            d = min(states.shape[1], state_dim)
            states_batch[i, :t, :d] = states[:t, :d]
            actions_batch[i, :t] = actions[:t]
            mask_batch[i, :t] = True

        with torch.no_grad():
            states_t = torch.as_tensor(states_batch, device=self.device)
            actions_t = torch.as_tensor(actions_batch, device=self.device)
            mask_t = torch.as_tensor(mask_batch, device=self.device)
            embeddings = self.model(states_t, actions_t, mask_t)

        return embeddings.detach().cpu().numpy().astype(float)
