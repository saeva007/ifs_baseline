#!/usr/bin/env python3
"""Tests for strict node-local Static-RNN data staging."""

from __future__ import annotations

import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import stage_lowvis_local_cache as mod


@contextmanager
def workspace_temp_dir():
    path = Path(__file__).resolve().parent / f".tmp_local_cache_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class StageLowVisLocalCacheTest(unittest.TestCase):
    def test_stages_exact_names_expected_by_training_loader(self) -> None:
        with workspace_temp_dir() as root:
            data = root / "data"
            cache = root / "cache"
            data.mkdir()
            for index, name in enumerate(mod.REQUIRED_FILES):
                (data / name).write_bytes(bytes([index + 1]) * (index + 2))

            report = mod.stage_dataset(data, cache, "demo", reserve_bytes=0)
            self.assertEqual(report["copied_files"], 4)
            for name in mod.REQUIRED_FILES:
                source = data / name
                target = mod.cache_path(source, cache, "demo")
                self.assertEqual(target.read_bytes(), source.read_bytes())

            reused = mod.stage_dataset(data, cache, "demo", reserve_bytes=0)
            self.assertEqual(reused["copied_files"], 0)
            self.assertEqual(reused["reused_files"], 4)

    def test_insufficient_space_is_a_hard_failure(self) -> None:
        with workspace_temp_dir() as root:
            data = root / "data"
            cache = root / "cache"
            data.mkdir()
            cache.mkdir()
            for name in mod.REQUIRED_FILES:
                (data / name).write_bytes(b"1234")

            usage = type("Usage", (), {"free": 0})()
            with patch.object(mod.shutil, "disk_usage", return_value=usage):
                with self.assertRaisesRegex(OSError, "insufficient local space"):
                    mod.stage_dataset(data, cache, "demo", reserve_bytes=0)


if __name__ == "__main__":
    unittest.main()
