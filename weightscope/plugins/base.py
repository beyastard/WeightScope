"""
WeightScope Plugin Base Class
=============================

To write a plugin:

  1. Create a directory inside ``plugins/`` (e.g. ``plugins/my_analysis/``).
  2. Add a ``plugin.py`` file that subclasses ``BasePlugin``.
  3. Implement ``name``, ``version``, ``description``, and ``mount()``.
  4. Restart WeightScope - your tab appears automatically.

Minimal example
---------------

.. code-block:: python

    # plugins/my_analysis/plugin.py
    import gradio as gr
    from weightscope.plugins.base import BasePlugin

    class MyAnalysisPlugin(BasePlugin):
        name        = "My Custom Analysis"
        version     = "0.1.0"
        description = "Adds a custom analysis tab."

        def mount(self, demo: gr.Blocks) -> None:
            with gr.Tab("🔧 My Analysis"):
                gr.Markdown("## Hello from MyAnalysisPlugin!")
                btn = gr.Button("Click me")
                out = gr.Textbox()
                btn.click(fn=lambda: "It works!", outputs=out)

Accessing shared state
----------------------
The ``mount()`` method receives the top-level ``gr.Blocks`` instance.
Shared Gradio ``gr.State`` objects (``current_df``, ``current_metadata``,
``current_model_id``) are injected via ``inject_state()`` before ``mount()``
is called – use them as inputs to your event handlers if needed.

.. code-block:: python

    def mount(self, demo: gr.Blocks) -> None:
        with gr.Tab("📐 My Tab"):
            out = gr.JSON()
            gr.Button("Analyze").click(
                fn=self._analyze,
                inputs=[self.state["current_df"]],
                outputs=[out],
            )

    def _analyze(self, df):
        if df is None:
            return {"error": "No model loaded"}
        return {"rows": len(df)}

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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import gradio as gr


class BasePlugin(ABC):
    """Abstract base class for all WeightScope plugins."""

    #: Human-readable plugin name (shown in logs)
    name: str = "Unnamed Plugin"

    #: Semantic version string
    version: str = "0.0.0"

    #: One-line description
    description: str = ""

    def __init__(self) -> None:
        # Populated by the app builder via inject_state()
        self.state: Dict[str, Any] = {}

    def inject_state(self, state: Dict[str, Any]) -> None:
        """
        Called by ``app_builder`` before ``mount()`` to provide access to
        shared ``gr.State`` components (``current_df``, ``current_metadata``,
        ``current_model_id``).
        """
        self.state = state

    @abstractmethod
    def mount(self, demo: gr.Blocks) -> None:
        """
        Add UI elements to *demo* (a ``gr.Blocks`` context that is already
        open).  This method is called *inside* a ``with demo:`` block, so you
        can call ``gr.Tab(…)``, ``gr.Row(…)``, etc. directly.
        """
        ...

    def __repr__(self) -> str:
        return f"<Plugin {self.name!r} v{self.version}>"
