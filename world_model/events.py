"""Lightweight offline event annotations derived from recorded signals."""

from __future__ import annotations

import numpy as np


def derive_event_tags(
    contact_active: np.ndarray,
    tangential_force: np.ndarray,
    success: np.ndarray,
    normal_force: np.ndarray | None = None,
    slip_force_threshold: float = 0.25,
) -> dict[str, np.ndarray]:
    active = np.asarray(contact_active, dtype=bool)
    any_contact = active.any(axis=1)
    previous = np.concatenate(([False], any_contact[:-1]))
    onset = any_contact & ~previous
    release = ~any_contact & previous

    tangent_norm = np.linalg.norm(np.nan_to_num(tangential_force), axis=-1)
    slip = (tangent_norm > slip_force_threshold).any(axis=1) & any_contact
    if normal_force is not None:
        normal = np.maximum(np.asarray(normal_force), 1e-6)
        slip |= ((tangent_norm / normal) > 0.8).any(axis=1) & any_contact

    regrasp = np.zeros_like(any_contact)
    release_indices = np.flatnonzero(release)
    onset_indices = np.flatnonzero(onset)
    for idx in onset_indices:
        if np.any((release_indices < idx) & (release_indices >= idx - 10)):
            regrasp[idx] = True

    success_bool = np.nan_to_num(np.asarray(success), nan=0.0).astype(bool)
    prior_success = np.concatenate(([False], success_bool[:-1]))
    success_transition = success_bool & ~prior_success
    drop = release & ~success_bool
    return {
        "contact_onset": onset,
        "contact_release": release,
        "slip_event": slip,
        "regrasp_event": regrasp,
        "drop_event": drop,
        "success_transition": success_transition,
    }
