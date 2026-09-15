# Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
# Copyright (C) 2026, aiter contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
"""Runtime routing for aiter communication kernels.

Declarative, table-driven replacement for the hardcoded dispatch constants:

- Layer 2 (kernel dispatch): ``resolve_use_1stage`` decides the fused
  AR+RMSNorm kernel variant from a rule table instead of inline if-chains.
- Layer 3 (AR backend): ``backend_order`` / ``backend_gate`` provide the
  all-reduce backend priority and per-backend byte windows read by the
  vLLM-side communicator.

The table lives at ``configs/runtime_routing.yaml`` inside the aiter package
(or any path given by ``AITER_RUNTIME_ROUTING``). Missing table, parse
failure, or the per-layer kill switch (``AITER_ROUTING_KERNEL=0`` /
``AITER_ROUTING_AR=0``) all fall back to built-in defaults that are
bit-identical to the pre-routing hardcoded behavior — routing never
fail-fasts a serving process.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROUTING_ENV = "AITER_RUNTIME_ROUTING"
_KERNEL_SWITCH = "AITER_ROUTING_KERNEL"
_AR_SWITCH = "AITER_ROUTING_AR"

# Built-in defaults == the pre-routing hardcoded constants, kept as the
# fallback for every layer so an absent table changes nothing.
_DEFAULT_KERNEL_RULES: list[dict[str, Any]] = [
    {
        "name": "1stage_ws2",
        "when": {"world_size": 2},
        "select": {"use_1stage": True},
    },
    {
        "name": "1stage_fullmesh_small",
        "when": {"fully_connected": True, "world_size": [2, 3, 4], "max_total_bytes": 262144},
        "select": {"use_1stage": True},
    },
    {
        "name": "1stage_fullmesh_8",
        "when": {"fully_connected": True, "world_size": [5, 6, 7, 8], "max_total_bytes": 131072},
        "select": {"use_1stage": True},
    },
    {"name": "2stage", "when": {}, "select": {"use_1stage": False}},
]
_DEFAULT_AR_ORDER = ["AITER_CUSTOM", "PYNCCL"]
_DEFAULT_AR_GATES: dict[str, dict[str, Any]] = {
    "AITER_CUSTOM": {"min_bytes": 0, "max_bytes": 67108864},
    "PYNCCL": {"fallback": True},
}

# Shared hard limits from the kernel launchers themselves (unchanged by routing).
_MAX_HIDDEN_PACKS = 1024
_MAX_TOKEN_NUM = 80


def _load_table() -> dict[str, Any]:
    path = os.environ.get(_ROUTING_ENV, "").strip()
    if not path:
        # aiter/dist/routing.py -> repo-root configs/ (sibling of the aiter pkg)
        path = str(
            Path(__file__).resolve().parent.parent.parent / "configs" / "runtime_routing.yaml"
        )
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            table = yaml.safe_load(f) or {}
        if not isinstance(table, dict):
            raise ValueError(f"routing table {path} is not a mapping")
        return table
    except Exception as e:  # noqa: BLE001 - any table problem means defaults
        logger.warning("runtime routing table unusable (%s); using built-in defaults", e)
        return {}


def _switched_off(name: str) -> bool:
    return os.environ.get(name, "1").strip() in ("0", "false", "False")


@lru_cache(maxsize=1)
def _kernel_rules() -> list[dict[str, Any]]:
    if _switched_off(_KERNEL_SWITCH):
        return _DEFAULT_KERNEL_RULES
    rules = _load_table().get("kernel_dispatch", {}).get("fused_ar_rms", {}).get("rules")
    if not rules:
        return _DEFAULT_KERNEL_RULES
    return rules


@lru_cache(maxsize=1)
def _ar_table() -> tuple[list[str], dict[str, dict[str, Any]]]:
    if _switched_off(_AR_SWITCH):
        return _DEFAULT_AR_ORDER, _DEFAULT_AR_GATES
    cfg = _load_table().get("ar_backends", {})
    order = cfg.get("order") or _DEFAULT_AR_ORDER
    gates = cfg.get("gates") or _DEFAULT_AR_GATES
    return list(order), gates


def _match_condition(cond: Any, actual: Any) -> bool:
    """Scalar equality, list membership, or null-means-absent for spec fields."""
    if isinstance(cond, list):
        return actual in cond
    return actual == cond


def resolve_use_1stage(
    hidden_dim: int,
    token_num: int,
    total_bytes: int,
    world_size: int,
    fully_connected: bool,
    pack_size: int = 8,
    *,
    _defaults=None,
) -> bool:
    """Layer-2 router: pick the fused AR+RMS kernel variant.

    ``hidden_dim % pack_size == 0`` and the launcher hard limits
    (``hidden // pack <= 1024``, ``token_num <= 80``) always apply — the table
    can only further restrict, never widen, what the kernels support.
    """
    # Kernel launcher hard limits (correctness, not policy). pack_size<=0
    # marks a dtype the 1-stage kernel cannot serve (e.g. fp32 activations).
    if pack_size <= 0 or hidden_dim % pack_size != 0 or hidden_dim // pack_size > _MAX_HIDDEN_PACKS:
        return False
    if token_num > _MAX_TOKEN_NUM:
        return False

    facts = {
        "world_size": world_size,
        "fully_connected": fully_connected,
        "total_bytes": total_bytes,
        "hidden_packs": hidden_dim // pack_size,
        "token_num": token_num,
    }
    for rule in _defaults if _defaults is not None else _kernel_rules():
        when = rule.get("when", {})
        ok = True
        for key, cond in when.items():
            if key in ("max_total_bytes", "max_hidden_packs", "max_token_num"):
                fact_key = {
                    "max_total_bytes": "total_bytes",
                    "max_hidden_packs": "hidden_packs",
                    "max_token_num": "token_num",
                }[key]
                if facts[fact_key] > cond:
                    ok = False
                    break
            elif key == "require_hidden_pack_aligned":
                if cond and hidden_dim % pack_size != 0:
                    ok = False
                    break
            else:
                if not _match_condition(cond, facts.get(key)):
                    ok = False
                    break
        if ok:
            return bool(rule.get("select", {}).get("use_1stage", False))
    return False


def backend_order() -> list[str]:
    """Layer-3 router: all-reduce backend dispatch order."""
    return _ar_table()[0]


def fused_ar_rms_token_gate() -> int | None:
    """Layer-2 router: max token count for which the fused AR+RMS pass may
    apply (compile-range gate for the vLLM fusion pass). ``None`` = no
    additional gate beyond the per-call rules."""
    if _switched_off(_KERNEL_SWITCH):
        return None
    return _load_table().get("kernel_dispatch", {}).get("fused_ar_rms", {}).get(
        "max_tokens_gate"
    )


def backend_gate(backend: str, nbytes: int) -> bool:
    """Layer-3 router: does ``backend`` accept a tensor of ``nbytes``?

    Unconditional backends (``fallback: true`` or no gate entry) always pass;
    windowed backends pass iff ``min_bytes < nbytes <= max_bytes``.
    """
    _, gates = _ar_table()
    gate = gates.get(backend)
    if gate is None or gate.get("fallback"):
        return True
    lo = gate.get("min_bytes", 0)
    hi = gate.get("max_bytes")
    if nbytes <= lo:
        return False
    return hi is None or nbytes <= hi
