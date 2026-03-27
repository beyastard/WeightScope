#!/usr/bin/env python3
"""
WeightScope: SafeTensors Model Analyzer v0.2.1 - Entry Point for application
A research tool for analyzing model weight distributions, compression potential, and pruning candidates.

Supports: Any .safetensors model (Llama, Mistral, Qwen, Stable Diffusion, Whisper, BERT, etc.)

New Features in v0.2.1:
- Clip & Normalize simulation tab
- Query presets with Custom mode
- Fixed scatter plot filtering (independent masks + stratified sampling)
- Accurate parameter count (tensor-based, not config-based)

Run with:
    python app.py

Or set environment variables to override defaults:
    WEIGHTSCOPE_HOST=0.0.0.0 WEIGHTSCOPE_PORT=8080 python app.py

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

import warnings
warnings.filterwarnings("ignore")

from weightscope         import APP_NAME, APP_VERSION
from weightscope.config  import SERVER_HOST, SERVER_PORT, SAVE_STATE_DIR
from weightscope.utils   import get_available_ram_gb, get_total_ram_gb
from weightscope.ui      import build_app


def main() -> None:
    print(f"🔍 {APP_NAME}  v{APP_VERSION}")
    print(f"📁 Cache directory : {SAVE_STATE_DIR.absolute()}")
    print(f"💻 RAM             : {get_available_ram_gb():.1f} GB available "
          f"/ {get_total_ram_gb():.1f} GB total")
    print("-" * 60)

    demo = build_app()
    demo.launch(
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        show_error=True,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
