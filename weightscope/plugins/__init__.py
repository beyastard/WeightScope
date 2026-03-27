"""
WeightScope Plugin System
=========================

Plugins extend WeightScope by adding new Gradio tabs (or other UI elements)
without touching the core codebase.

Auto-discovery
--------------
Any Python package placed inside the ``plugins/`` directory that:

  1. Contains a ``plugin.py`` module, AND
  2. That module exposes a class inheriting from ``BasePlugin``

…will be discovered and registered automatically at startup.

Manual registration
-------------------
You can also register a plugin programmatically::

    from weightscope.plugins import registry
    from my_plugin import MyPlugin
    registry.register(MyPlugin())

Plugin contract
---------------
See ``weightscope/plugins/base.py`` for the full ``BasePlugin`` ABC.

---

Copyright (C) 2026 Bryan K Reinhart & BeySoft

This file is part of WeightScope.

WeightScope is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

WeightScope is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public
License along with WeightScope. If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import List

from .base import BasePlugin


class PluginRegistry:
    """Holds all active plugin instances and exposes them to the app builder."""

    def __init__(self) -> None:
        self._plugins: List[BasePlugin] = []

    def register(self, plugin: BasePlugin) -> None:
        """Register a *plugin* instance."""
        if not isinstance(plugin, BasePlugin):
            raise TypeError(f"{plugin!r} must be a BasePlugin subclass instance")
        self._plugins.append(plugin)
        print(f"  🔌 Plugin registered: {plugin.name} v{plugin.version}")

    def discover(self, plugins_dir: Path) -> None:
        """
        Walk *plugins_dir*, import every ``plugin.py`` found, and register any
        ``BasePlugin`` subclasses defined inside it.
        """
        if not plugins_dir.exists():
            return

        for entry in sorted(plugins_dir.iterdir()):
            plugin_file = entry / "plugin.py" if entry.is_dir() else (
                entry if entry.suffix == ".py" and entry.stem != "__init__" else None
            )
            if plugin_file is None or not plugin_file.exists():
                continue

            module_name = f"_ws_plugin_{entry.stem}"
            try:
                spec   = importlib.util.spec_from_file_location(module_name, plugin_file)
                module = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
                sys.modules[module_name] = module
                spec.loader.exec_module(module)                          # type: ignore[union-attr]

                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BasePlugin)
                        and attr is not BasePlugin
                    ):
                        self.register(attr())

            except Exception as exc:
                print(f"  ⚠️  Failed to load plugin '{entry.name}': {exc}")

    @property
    def plugins(self) -> List[BasePlugin]:
        return list(self._plugins)

    def mount_all(self, demo) -> None:
        """Call ``mount(demo)`` on every registered plugin."""
        for plugin in self._plugins:
            try:
                plugin.mount(demo)
            except Exception as exc:
                print(f"  ⚠️  Plugin '{plugin.name}' failed to mount: {exc}")


# Module-level singleton – import this everywhere you need plugin access
registry = PluginRegistry()

__all__ = ["BasePlugin", "PluginRegistry", "registry"]
