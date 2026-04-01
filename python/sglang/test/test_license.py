"""
Unit tests for the offline license manager.

Usage:
    python -m pytest python/sglang/test/test_license.py -v
"""

import base64
import datetime as dt
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def _load_license_module():
    this_dir = os.path.dirname(__file__)
    python_dir = os.path.abspath(os.path.join(this_dir, "..", ".."))
    module_path = os.path.join(python_dir, "sglang", "srt", "license.py")

    module_name = "_sglang_srt_license_for_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_license = _load_license_module()
LicenseEnforcementError = _license.LicenseEnforcementError
LicenseManager = _license.LicenseManager


class TestLicenseManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.private_key = Ed25519PrivateKey.generate()

        self.public_key_path = os.path.join(self.temp_dir.name, "license.pub.pem")
        public_key = self.private_key.public_key()
        with open(self.public_key_path, "wb") as f:
            f.write(
                public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )

        self.license_path = os.path.join(self.temp_dir.name, "license.json")
        self.state_path = os.path.join(self.temp_dir.name, "license.state.json")

    def _make_args(self):
        return types.SimpleNamespace(
            license_file=self.license_path,
            license_public_key_path=self.public_key_path,
            license_state_file=self.state_path,
            license_warning_days=7,
            license_reload_interval_seconds=1,
            license_reminder_interval_seconds=1,
        )

    def _make_manager(
        self,
        *,
        now,
        monotonic_value=0.0,
        machine_fingerprint="fp-host-a",
        boot_id="boot-a",
    ):
        return LicenseManager(
            self._make_args(),
            time_provider=lambda now=now: now,
            monotonic_provider=lambda value=monotonic_value: value,
            machine_fingerprint_provider=lambda fingerprint=machine_fingerprint: fingerprint,
            boot_id_provider=lambda boot_id=boot_id: boot_id,
        )

    def _write_license(self, payload):
        doc = {
            "payload": payload,
            "signature_algorithm": "ed25519",
            "signature": base64.b64encode(
                self.private_key.sign(_license._canonicalize_payload(payload))
            ).decode("ascii"),
        }
        with open(self.license_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=True, sort_keys=True)

    def test_state_transitions_and_enforcement(self):
        payload = {
            "license_id": "lic-001",
            "customer": "acme",
            "product": "sglang",
            "machine_fingerprint": "fp-host-a",
            "issue_no": 1,
            "grace_days": 7,
            "not_before": "2026-03-01T00:00:00Z",
            "not_after": "2026-03-30T00:00:00Z",
        }
        self._write_license(payload)

        cases = [
            ("active", dt.datetime(2026, 3, 10, tzinfo=dt.timezone.utc), True),
            ("warning", dt.datetime(2026, 3, 24, tzinfo=dt.timezone.utc), True),
            ("grace", dt.datetime(2026, 4, 2, tzinfo=dt.timezone.utc), True),
            ("expired", dt.datetime(2026, 4, 7, 0, 0, 1, tzinfo=dt.timezone.utc), False),
        ]
        for expected_state, now, allowed in cases:
            with self.subTest(state=expected_state):
                manager = self._make_manager(now=now)
                status = manager.get_status(force=True)
                self.assertEqual(status.state, expected_state)
                self.assertEqual(status.allowed, allowed)
                self.assertTrue(status.machine_fingerprint_present)
                self.assertTrue(status.machine_fingerprint_match)
                if allowed:
                    manager.ensure_request_allowed("/generate")
                else:
                    with self.assertRaises(LicenseEnforcementError):
                        manager.ensure_request_allowed("/generate")
                    manager.ensure_request_allowed("/license/status")

    def test_invalid_signature_blocks_requests(self):
        payload = {
            "license_id": "lic-002",
            "machine_fingerprint": "fp-host-a",
            "issue_no": 2,
            "not_before": "2026-03-01T00:00:00Z",
            "not_after": "2026-04-01T00:00:00Z",
        }
        self._write_license(payload)
        with open(self.license_path, "r+", encoding="utf-8") as f:
            doc = json.load(f)
            doc["payload"]["machine_fingerprint"] = "fp-host-b"
            f.seek(0)
            json.dump(doc, f)
            f.truncate()

        manager = self._make_manager(
            now=dt.datetime(2026, 3, 10, tzinfo=dt.timezone.utc),
        )
        status = manager.get_status(force=True)
        self.assertEqual(status.state, "invalid")
        self.assertEqual(status.reason_code, "verification_failed")
        with self.assertRaises(LicenseEnforcementError):
            manager.ensure_request_allowed("/generate")

    def test_hot_reload_and_issue_rollback_protection(self):
        first_payload = {
            "license_id": "lic-003",
            "machine_fingerprint": "fp-host-a",
            "issue_no": 3,
            "not_before": "2026-03-01T00:00:00Z",
            "not_after": "2026-03-20T00:00:00Z",
        }
        self._write_license(first_payload)

        manager = self._make_manager(
            now=dt.datetime(2026, 3, 10, tzinfo=dt.timezone.utc),
            monotonic_value=10.0,
        )
        self.assertEqual(manager.get_status(force=True).issue_no, 3)

        second_payload = dict(first_payload)
        second_payload["issue_no"] = 4
        second_payload["not_after"] = "2026-05-01T00:00:00Z"
        self._write_license(second_payload)
        manager.refresh(force=True)
        self.assertEqual(manager.get_status().issue_no, 4)
        self.assertEqual(manager.get_status().state, "active")

        self._write_license(first_payload)
        manager.refresh(force=True)
        rollback_status = manager.get_status()
        self.assertEqual(rollback_status.state, "invalid")
        self.assertEqual(rollback_status.reason_code, "issue_rollback_detected")

    def test_machine_fingerprint_mismatch_blocks_requests(self):
        payload = {
            "license_id": "lic-004",
            "machine_fingerprint": "fp-host-a",
            "issue_no": 1,
            "not_before": "2026-03-01T00:00:00Z",
            "not_after": "2026-04-01T00:00:00Z",
        }
        self._write_license(payload)

        manager = self._make_manager(
            now=dt.datetime(2026, 3, 10, tzinfo=dt.timezone.utc),
            machine_fingerprint="fp-host-b",
        )
        status = manager.get_status(force=True)
        self.assertEqual(status.state, "invalid")
        self.assertEqual(status.reason_code, "machine_fingerprint_mismatch")
        self.assertFalse(status.machine_fingerprint_match)
        with self.assertRaises(LicenseEnforcementError):
            manager.ensure_request_allowed("/generate")

    def test_clock_rollback_detected_with_monotonic_drift(self):
        payload = {
            "license_id": "lic-005",
            "machine_fingerprint": "fp-host-a",
            "issue_no": 5,
            "not_before": "2026-03-01T00:00:00Z",
            "not_after": "2026-04-01T00:00:00Z",
        }
        self._write_license(payload)

        first_manager = self._make_manager(
            now=dt.datetime(2026, 3, 10, 12, 0, tzinfo=dt.timezone.utc),
            monotonic_value=1000.0,
        )
        first_status = first_manager.get_status(force=True)
        self.assertEqual(first_status.state, "active")

        second_manager = self._make_manager(
            now=dt.datetime(2026, 3, 10, 12, 10, tzinfo=dt.timezone.utc),
            monotonic_value=2200.0,
        )
        second_status = second_manager.get_status(force=True)
        self.assertEqual(second_status.state, "invalid")
        self.assertEqual(second_status.reason_code, "clock_rollback_detected")
        self.assertIn("monotonic", second_status.message.lower())


if __name__ == "__main__":
    unittest.main()
