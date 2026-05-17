#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from CanalTextoChefZeroWaste import CanalTextoChefZeroWaste


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    data = base / "datos"
    chat = CanalTextoChefZeroWaste(
        fileVectors=str(data / "ChefZeroWaste.vec"),
        fileVoc=str(data / "ChefZeroWaste.voc"),
    )
    chat.run()
