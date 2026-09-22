"""Explicit scenario registration; no import-time auto-discovery."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from .contracts import EnvironmentProfile
from .ports import EnvironmentAdapter


AdapterFactory = Callable[..., EnvironmentAdapter]


class ScenarioRegistry:
    """Maps stable scenario ids to profile declarations and adapter factories.

    The registry deliberately stays in-process for the first version. A future
    package-entry-point plugin layer can sit above this API without changing
    workflow contracts.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[EnvironmentProfile, AdapterFactory]] = {}

    def register(self, profile: EnvironmentProfile, factory: AdapterFactory) -> None:
        if profile.id in self._entries:
            raise ValueError(f"scenario already registered: {profile.id}")
        self._entries[profile.id] = (profile, factory)

    def profile(self, scenario_id: str) -> EnvironmentProfile:
        try:
            return self._entries[scenario_id][0]
        except KeyError as exc:
            raise KeyError(f"unknown scenario: {scenario_id}") from exc

    def create(self, scenario_id: str, **kwargs: Any) -> EnvironmentAdapter:
        profile, factory = self._entries[scenario_id]
        adapter = factory(**kwargs)
        if adapter.profile != profile:
            raise ValueError("factory returned an adapter for a different profile")
        return adapter

    def __contains__(self, scenario_id: object) -> bool:
        return scenario_id in self._entries

    def __iter__(self) -> Iterator[EnvironmentProfile]:
        return iter(profile for profile, _ in self._entries.values())
