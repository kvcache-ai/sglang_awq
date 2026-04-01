from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from prometheus_client import Gauge

logger = logging.getLogger(__name__)

LICENSE_EXEMPT_PATH_PREFIXES = ("/health", "/metrics", "/license/status")
LICENSE_STATE_VALUE = {
    "disabled": 0,
    "active": 1,
    "warning": 2,
    "grace": 3,
    "expired": 4,
    "invalid": 5,
}
LICENSE_STATE_GAUGE = Gauge(
    "sglang_license_state",
    "Numeric license state. disabled=0, active=1, warning=2, grace=3, expired=4, invalid=5",
)
LICENSE_SECONDS_TO_EXPIRY_GAUGE = Gauge(
    "sglang_license_seconds_until_expiry",
    "Seconds until license expiry. Negative after expiry.",
)
LICENSE_SECONDS_TO_GRACE_END_GAUGE = Gauge(
    "sglang_license_seconds_until_grace_end",
    "Seconds until grace period ends. Negative after grace end.",
)
BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"
MACHINE_ID_SOURCES = (
    ("product_uuid", "/sys/class/dmi/id/product_uuid"),
    ("board_serial", "/sys/class/dmi/id/board_serial"),
    ("product_serial", "/sys/class/dmi/id/product_serial"),
    ("chassis_serial", "/sys/class/dmi/id/chassis_serial"),
)


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _isoformat(value: Optional[dt.datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_datetime(
    value: Any, *, field_name: str, end_of_day_for_date_only: bool
) -> dt.datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty ISO8601 string")

    text = value.strip()
    if "T" not in text:
        date_value = dt.date.fromisoformat(text)
        if end_of_day_for_date_only:
            return dt.datetime.combine(
                date_value,
                dt.time(23, 59, 59, 999999, tzinfo=dt.timezone.utc),
            )
        return dt.datetime.combine(
            date_value,
            dt.time(0, 0, 0, 0, tzinfo=dt.timezone.utc),
        )

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include timezone information")
    return parsed.astimezone(dt.timezone.utc)


def _canonicalize_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _decode_signature(signature: str) -> bytes:
    try:
        return base64.b64decode(signature.encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError("signature must be base64 encoded") from exc


def _load_public_key(public_key_path: str) -> Ed25519PublicKey:
    with open(public_key_path, "rb") as f:
        key = serialization.load_pem_public_key(f.read())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("license public key must be an Ed25519 public key")
    return key


def _load_private_key(private_key_path: str) -> Ed25519PrivateKey:
    with open(private_key_path, "rb") as f:
        key = serialization.load_pem_private_key(f.read(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("license private key must be an Ed25519 private key")
    return key


def _extract_payload_and_signature(
    raw_doc: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    if not isinstance(raw_doc, dict):
        raise ValueError("license file must contain a JSON object")

    signature = raw_doc.get("signature")
    if not isinstance(signature, str) or not signature.strip():
        raise ValueError("license signature is missing")

    algorithm = raw_doc.get("signature_algorithm", "ed25519")
    if algorithm != "ed25519":
        raise ValueError(f"unsupported license signature_algorithm: {algorithm}")

    if "payload" in raw_doc:
        payload = raw_doc["payload"]
        if not isinstance(payload, dict):
            raise ValueError("license payload must be a JSON object")
        return payload, signature, algorithm

    payload = {
        key: value
        for key, value in raw_doc.items()
        if key not in {"signature", "signature_algorithm"}
    }
    return payload, signature, algorithm


def _hash_license_document(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def _read_identifier_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = f.read().strip()
    except OSError:
        return None
    if not value or value in {"None", "none", "unknown"}:
        return None
    return value


def collect_machine_identity() -> dict[str, str]:
    identity: dict[str, str] = {}
    for key, path in MACHINE_ID_SOURCES:
        value = _read_identifier_file(path)
        if value:
            identity[key] = value
    if not identity:
        raise RuntimeError(
            "No stable machine identifiers were found. Checked: "
            + ", ".join(path for _, path in MACHINE_ID_SOURCES)
        )
    return identity


def generate_machine_fingerprint(identity: Optional[dict[str, str]] = None) -> str:
    identity = identity or collect_machine_identity()
    return hashlib.sha256(_canonicalize_payload(identity)).hexdigest()


def build_machine_fingerprint_payload() -> dict[str, Any]:
    identity = collect_machine_identity()
    return {
        "created_at": _isoformat(utc_now()),
        "machine_fingerprint": generate_machine_fingerprint(identity),
        "identity_sources": sorted(identity.keys()),
    }


def get_boot_id() -> Optional[str]:
    return _read_identifier_file(BOOT_ID_PATH)


def build_signed_license_document(
    payload: dict[str, Any], private_key_path: str
) -> dict[str, Any]:
    private_key = _load_private_key(private_key_path)
    signature = private_key.sign(_canonicalize_payload(payload))
    return {
        "payload": payload,
        "signature_algorithm": "ed25519",
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def _coerce_non_negative_int(value: Any, *, field_name: str, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


@dataclasses.dataclass(frozen=True)
class LicenseStatus:
    configured: bool
    state: str
    allowed: bool
    message: str
    reason_code: Optional[str] = None
    license_id: Optional[str] = None
    customer: Optional[str] = None
    product: Optional[str] = None
    issue_no: Optional[int] = None
    machine_fingerprint_present: bool = False
    machine_fingerprint_match: Optional[bool] = None
    not_before: Optional[dt.datetime] = None
    not_after: Optional[dt.datetime] = None
    warning_start: Optional[dt.datetime] = None
    grace_until: Optional[dt.datetime] = None
    checked_at: Optional[dt.datetime] = None
    last_loaded_at: Optional[dt.datetime] = None
    license_hash: Optional[str] = None

    def days_until_expiry(self) -> Optional[float]:
        if self.checked_at is None or self.not_after is None:
            return None
        return (self.not_after - self.checked_at).total_seconds() / 86400.0

    def seconds_until_expiry(self) -> Optional[float]:
        if self.checked_at is None or self.not_after is None:
            return None
        return (self.not_after - self.checked_at).total_seconds()

    def seconds_until_grace_end(self) -> Optional[float]:
        if self.checked_at is None or self.grace_until is None:
            return None
        return (self.grace_until - self.checked_at).total_seconds()

    def to_headers(self) -> dict[str, str]:
        headers = {
            "X-SGLang-License-State": self.state,
            "X-SGLang-License-Configured": str(self.configured).lower(),
        }
        if self.reason_code:
            headers["X-SGLang-License-Reason"] = self.reason_code
        if self.license_id:
            headers["X-SGLang-License-Id"] = self.license_id
        if self.issue_no is not None:
            headers["X-SGLang-License-Issue-No"] = str(self.issue_no)
        headers["X-SGLang-License-Machine-Bound"] = str(
            self.machine_fingerprint_present
        ).lower()
        if self.machine_fingerprint_match is not None:
            headers["X-SGLang-License-Machine-Match"] = str(
                self.machine_fingerprint_match
            ).lower()
        if self.not_after is not None:
            headers["X-SGLang-License-Expires-At"] = _isoformat(self.not_after)
        if self.grace_until is not None:
            headers["X-SGLang-License-Grace-Ends-At"] = _isoformat(self.grace_until)
        if self.warning_start is not None:
            headers["X-SGLang-License-Warning-Starts-At"] = _isoformat(
                self.warning_start
            )
        days_left = self.days_until_expiry()
        if days_left is not None:
            headers["X-SGLang-License-Days-Left"] = f"{days_left:.3f}"
        return headers

    def to_dict(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "state": self.state,
            "allowed": self.allowed,
            "message": self.message,
            "reason_code": self.reason_code,
            "license_id": self.license_id,
            "customer": self.customer,
            "product": self.product,
            "issue_no": self.issue_no,
            "machine_fingerprint_present": self.machine_fingerprint_present,
            "machine_fingerprint_match": self.machine_fingerprint_match,
            "not_before": _isoformat(self.not_before),
            "not_after": _isoformat(self.not_after),
            "warning_start": _isoformat(self.warning_start),
            "grace_until": _isoformat(self.grace_until),
            "checked_at": _isoformat(self.checked_at),
            "last_loaded_at": _isoformat(self.last_loaded_at),
            "days_until_expiry": self.days_until_expiry(),
            "seconds_until_expiry": self.seconds_until_expiry(),
            "seconds_until_grace_end": self.seconds_until_grace_end(),
        }


class LicenseEnforcementError(RuntimeError):
    def __init__(self, detail: str, status: LicenseStatus, status_code: int = 403):
        super().__init__(detail)
        self.detail = detail
        self.license_status = status
        self.status_code = status_code


class LicenseManager:
    def __init__(
        self,
        server_args: Any,
        *,
        time_provider: Callable[[], dt.datetime] = utc_now,
        monotonic_provider: Callable[[], float] = time.monotonic,
        machine_fingerprint_provider: Callable[[], str] = generate_machine_fingerprint,
        boot_id_provider: Callable[[], Optional[str]] = get_boot_id,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self.server_args = server_args
        self.time_provider = time_provider
        self.monotonic_provider = monotonic_provider
        self.machine_fingerprint_provider = machine_fingerprint_provider
        self.boot_id_provider = boot_id_provider
        self.logger = logger_ or logger

        self.license_file = getattr(server_args, "license_file", None)
        self.public_key_path = getattr(server_args, "license_public_key_path", None)
        self.state_file = getattr(server_args, "license_state_file", None)
        self.warning_days = getattr(server_args, "license_warning_days", 7)
        self.reload_interval_seconds = getattr(
            server_args, "license_reload_interval_seconds", 60
        )
        self.reminder_interval_seconds = getattr(
            server_args, "license_reminder_interval_seconds", 3600
        )
        self.clock_rollback_tolerance_seconds = 300.0
        self.default_grace_days = 7

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_file_fingerprint: Optional[tuple[int, int]] = None
        self._last_refresh_monotonic = 0.0
        self._last_logged_key: Optional[tuple[str, Optional[str], Optional[str]]] = None
        self._last_log_monotonic = 0.0

        self._status = self._build_disabled_status()
        self.refresh(force=True)

    @property
    def enabled(self) -> bool:
        return bool(self.license_file)

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_background_refresh,
            name="sglang-license-refresh",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def is_exempt_path(self, path: Optional[str]) -> bool:
        if not path:
            return False
        return any(path.startswith(prefix) for prefix in LICENSE_EXEMPT_PATH_PREFIXES)

    def attach_headers(self, response: Any, status: Optional[LicenseStatus] = None) -> Any:
        status = status or self.get_status()
        for key, value in status.to_headers().items():
            response.headers[key] = value
        return response

    def get_status(self, *, force: bool = False) -> LicenseStatus:
        self.refresh(force=force)
        with self._lock:
            return self._status

    def get_status_dict(self, *, force: bool = False) -> dict[str, Any]:
        return self.get_status(force=force).to_dict()

    def ensure_request_allowed(self, path: Optional[str] = None) -> LicenseStatus:
        status = self.get_status()
        if self.is_exempt_path(path):
            return status
        if not status.allowed:
            raise LicenseEnforcementError(status.message, status=status)
        return status

    def refresh(self, *, force: bool = False) -> None:
        with self._lock:
            if not self.enabled:
                self._status = self._build_disabled_status()
                self._update_metrics(self._status)
                return

            now_monotonic = self.monotonic_provider()
            if not force:
                if (
                    now_monotonic - self._last_refresh_monotonic
                    < self.reload_interval_seconds
                    and not self._license_file_changed()
                ):
                    return

            self._last_refresh_monotonic = now_monotonic
            self._status = self._load_status()
            self._update_metrics(self._status)
            self._maybe_log_status(self._status, now_monotonic)

    def _build_disabled_status(self) -> LicenseStatus:
        now = self.time_provider()
        return LicenseStatus(
            configured=False,
            state="disabled",
            allowed=True,
            message="License enforcement is disabled.",
            checked_at=now,
            last_loaded_at=now,
        )

    def _load_status(self) -> LicenseStatus:
        now = self.time_provider()
        if not self.public_key_path:
            return self._invalid_status(
                now,
                "public_key_missing",
                "License enforcement is configured but no public key path is set.",
            )

        if not os.path.exists(self.license_file):
            self._last_file_fingerprint = None
            return self._invalid_status(
                now,
                "file_missing",
                f"License file not found: {self.license_file}",
            )

        try:
            stat = os.stat(self.license_file)
            self._last_file_fingerprint = (stat.st_mtime_ns, stat.st_size)
            with open(self.license_file, "rb") as f:
                raw_bytes = f.read()
            raw_doc = json.loads(raw_bytes.decode("utf-8"))
            payload, signature_b64, _algorithm = _extract_payload_and_signature(raw_doc)
            public_key = _load_public_key(self.public_key_path)
            signature = _decode_signature(signature_b64)
            public_key.verify(signature, _canonicalize_payload(payload))
            status = self._evaluate_payload(
                payload=payload,
                raw_bytes=raw_bytes,
                checked_at=now,
            )
        except Exception as exc:
            detail = str(exc).strip() or type(exc).__name__
            return self._invalid_status(
                now,
                "verification_failed",
                f"License verification failed: {detail}",
            )

        if status.reason_code is None:
            self._persist_runtime_state(status)
        return status

    def _evaluate_payload(
        self,
        *,
        payload: dict[str, Any],
        raw_bytes: bytes,
        checked_at: dt.datetime,
    ) -> LicenseStatus:
        not_before = _parse_datetime(
            payload.get("not_before"),
            field_name="not_before",
            end_of_day_for_date_only=False,
        )
        not_after = _parse_datetime(
            payload.get("not_after"),
            field_name="not_after",
            end_of_day_for_date_only=True,
        )
        if not_after < not_before:
            raise ValueError("not_after must be later than or equal to not_before")

        issue_no = _coerce_non_negative_int(
            payload.get("issue_no"),
            field_name="issue_no",
            default=0,
        )
        grace_days = _coerce_non_negative_int(
            payload.get("grace_days"),
            field_name="grace_days",
            default=self.default_grace_days,
        )
        expected_machine_fingerprint = payload.get("machine_fingerprint")
        if (
            not isinstance(expected_machine_fingerprint, str)
            or not expected_machine_fingerprint.strip()
        ):
            raise ValueError("machine_fingerprint must be a non-empty string")
        expected_machine_fingerprint = expected_machine_fingerprint.strip()
        try:
            current_machine_fingerprint = self.machine_fingerprint_provider()
        except Exception as exc:
            return self._invalid_bound_status(
                checked_at,
                payload,
                raw_bytes,
                not_before,
                not_after,
                issue_no,
                grace_days,
                "machine_fingerprint_unavailable",
                f"Unable to compute local machine_fingerprint: {exc}",
                machine_fingerprint_match=None,
            )
        if current_machine_fingerprint != expected_machine_fingerprint:
            return self._invalid_bound_status(
                checked_at,
                payload,
                raw_bytes,
                not_before,
                not_after,
                issue_no,
                grace_days,
                "machine_fingerprint_mismatch",
                "License machine_fingerprint does not match this machine.",
                machine_fingerprint_match=False,
            )

        runtime_state = self._load_runtime_state()
        highest_seen_issue_no = runtime_state.get("highest_seen_issue_no", -1)
        if issue_no < highest_seen_issue_no:
            return self._invalid_bound_status(
                checked_at,
                payload,
                raw_bytes,
                not_before,
                not_after,
                issue_no,
                grace_days,
                "issue_rollback_detected",
                "License issue_no is older than the highest previously accepted issue_no.",
                machine_fingerprint_match=True,
            )

        rollback_message = self._detect_clock_rollback(
            runtime_state=runtime_state,
            checked_at=checked_at,
        )
        if rollback_message is not None:
            return self._invalid_bound_status(
                checked_at,
                payload,
                raw_bytes,
                not_before,
                not_after,
                issue_no,
                grace_days,
                "clock_rollback_detected",
                rollback_message,
                machine_fingerprint_match=True,
            )

        warning_start = not_after - dt.timedelta(days=self.warning_days)
        grace_until = not_after + dt.timedelta(days=grace_days)
        license_hash = _hash_license_document(raw_bytes)
        base_kwargs = {
            "configured": True,
            "license_id": payload.get("license_id"),
            "customer": payload.get("customer"),
            "product": payload.get("product"),
            "issue_no": issue_no,
            "machine_fingerprint_present": True,
            "machine_fingerprint_match": True,
            "not_before": not_before,
            "not_after": not_after,
            "warning_start": warning_start,
            "grace_until": grace_until,
            "checked_at": checked_at,
            "last_loaded_at": checked_at,
            "license_hash": license_hash,
        }

        if checked_at < not_before:
            return LicenseStatus(
                state="invalid",
                allowed=False,
                message="License is not valid yet.",
                reason_code="not_yet_valid",
                **base_kwargs,
            )
        if checked_at < warning_start:
            return LicenseStatus(
                state="active",
                allowed=True,
                message="License is valid.",
                **base_kwargs,
            )
        if checked_at < not_after:
            days_left = (not_after - checked_at).total_seconds() / 86400.0
            return LicenseStatus(
                state="warning",
                allowed=True,
                message=f"License will expire in {days_left:.2f} days.",
                reason_code="expiring_soon",
                **base_kwargs,
            )
        if checked_at < grace_until:
            grace_days_left = (grace_until - checked_at).total_seconds() / 86400.0
            return LicenseStatus(
                state="grace",
                allowed=True,
                message=f"License has expired and is in grace period for another {grace_days_left:.2f} days.",
                reason_code="grace_period",
                **base_kwargs,
            )
        return LicenseStatus(
            state="expired",
            allowed=False,
            message="License has expired and the grace period has ended.",
            reason_code="grace_ended",
            **base_kwargs,
        )

    def _invalid_status(
        self,
        checked_at: dt.datetime,
        reason_code: str,
        message: str,
    ) -> LicenseStatus:
        return LicenseStatus(
            configured=True,
            state="invalid",
            allowed=False,
            message=message,
            reason_code=reason_code,
            checked_at=checked_at,
            last_loaded_at=checked_at,
        )

    def _invalid_bound_status(
        self,
        checked_at: dt.datetime,
        payload: dict[str, Any],
        raw_bytes: bytes,
        not_before: dt.datetime,
        not_after: dt.datetime,
        issue_no: int,
        grace_days: int,
        reason_code: str,
        message: str,
        machine_fingerprint_match: Optional[bool],
    ) -> LicenseStatus:
        return LicenseStatus(
            configured=True,
            state="invalid",
            allowed=False,
            message=message,
            reason_code=reason_code,
            license_id=payload.get("license_id"),
            customer=payload.get("customer"),
            product=payload.get("product"),
            issue_no=issue_no,
            machine_fingerprint_present=bool(payload.get("machine_fingerprint")),
            machine_fingerprint_match=machine_fingerprint_match,
            not_before=not_before,
            not_after=not_after,
            warning_start=not_after - dt.timedelta(days=self.warning_days),
            grace_until=not_after + dt.timedelta(days=grace_days),
            checked_at=checked_at,
            last_loaded_at=checked_at,
            license_hash=_hash_license_document(raw_bytes),
        )

    def _license_file_changed(self) -> bool:
        if not self.enabled:
            return False
        try:
            stat = os.stat(self.license_file)
        except OSError:
            return self._last_file_fingerprint is not None
        fingerprint = (stat.st_mtime_ns, stat.st_size)
        return fingerprint != self._last_file_fingerprint

    def _load_runtime_state(self) -> dict[str, Any]:
        state_file = self._resolved_state_file()
        if state_file is None or not os.path.exists(state_file):
            return {}
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            if isinstance(state, dict):
                return state
        except Exception:
            self.logger.warning("Failed to load license runtime state file: %s", state_file)
        return {}

    def _detect_clock_rollback(
        self,
        *,
        runtime_state: dict[str, Any],
        checked_at: dt.datetime,
    ) -> Optional[str]:
        tolerance_seconds = max(0.0, self.clock_rollback_tolerance_seconds)
        rollback_tolerance = dt.timedelta(seconds=tolerance_seconds)

        last_verified_at_str = runtime_state.get("last_verified_at")
        if last_verified_at_str:
            last_verified_at = _parse_datetime(
                last_verified_at_str,
                field_name="last_verified_at",
                end_of_day_for_date_only=False,
            )
            if checked_at + rollback_tolerance < last_verified_at:
                return (
                    "System clock appears to have moved backwards beyond the allowed "
                    "tolerance compared with the last accepted verification time."
                )
        else:
            last_verified_at = None

        if last_verified_at is None:
            return None

        last_verified_boot_id = runtime_state.get("last_verified_boot_id")
        last_verified_monotonic = runtime_state.get("last_verified_monotonic")
        if not last_verified_boot_id or last_verified_monotonic is None:
            return None

        try:
            current_boot_id = self.boot_id_provider()
        except Exception:
            return None
        if not current_boot_id or current_boot_id != last_verified_boot_id:
            return None

        try:
            current_monotonic = float(self.monotonic_provider())
            last_monotonic = float(last_verified_monotonic)
        except Exception:
            return None

        elapsed_monotonic = current_monotonic - last_monotonic
        if elapsed_monotonic <= tolerance_seconds:
            return None

        elapsed_wall = (checked_at - last_verified_at).total_seconds()
        if elapsed_wall + tolerance_seconds < elapsed_monotonic:
            return (
                "System clock appears inconsistent with monotonic elapsed time on "
                "the current boot, which suggests manual rollback."
            )
        return None

    def _persist_runtime_state(self, status: LicenseStatus) -> None:
        state_file = self._resolved_state_file()
        if state_file is None:
            return
        state = self._load_runtime_state()
        state["highest_seen_issue_no"] = max(
            int(state.get("highest_seen_issue_no", -1)),
            int(status.issue_no or 0),
        )
        if status.checked_at is not None:
            state["last_verified_at"] = _isoformat(status.checked_at)
        if status.license_hash is not None:
            state["last_license_hash"] = status.license_hash
        try:
            state["last_verified_monotonic"] = float(self.monotonic_provider())
        except Exception:
            state.pop("last_verified_monotonic", None)
        try:
            boot_id = self.boot_id_provider()
        except Exception:
            boot_id = None
        if boot_id:
            state["last_verified_boot_id"] = boot_id
        else:
            state.pop("last_verified_boot_id", None)
        tmp_path = f"{state_file}.tmp"
        state_dir = os.path.dirname(state_file)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        os.replace(tmp_path, state_file)

    def _resolved_state_file(self) -> Optional[str]:
        if not self.enabled:
            return None
        if self.state_file:
            return self.state_file
        return f"{self.license_file}.state.json"

    def _run_background_refresh(self) -> None:
        interval = max(1, int(self.reload_interval_seconds))
        while not self._stop_event.wait(interval):
            try:
                self.refresh(force=True)
            except Exception:
                self.logger.exception("Unexpected error while refreshing license state")

    def _maybe_log_status(self, status: LicenseStatus, now_monotonic: float) -> None:
        key = (status.state, status.reason_code, status.message)
        should_log = (
            key != self._last_logged_key
            or now_monotonic - self._last_log_monotonic >= self.reminder_interval_seconds
        )
        if not should_log:
            return

        if status.state in {"invalid", "expired"}:
            self.logger.error("[license] %s", status.message)
        elif status.state in {"warning", "grace"}:
            self.logger.warning("[license] %s", status.message)
        elif status.state == "active" and self._last_logged_key != key:
            self.logger.info("[license] %s", status.message)

        self._last_logged_key = key
        self._last_log_monotonic = now_monotonic

    def _update_metrics(self, status: LicenseStatus) -> None:
        LICENSE_STATE_GAUGE.set(LICENSE_STATE_VALUE.get(status.state, 5))
        seconds_to_expiry = status.seconds_until_expiry()
        seconds_to_grace_end = status.seconds_until_grace_end()
        LICENSE_SECONDS_TO_EXPIRY_GAUGE.set(
            seconds_to_expiry if seconds_to_expiry is not None else 0.0
        )
        LICENSE_SECONDS_TO_GRACE_END_GAUGE.set(
            seconds_to_grace_end if seconds_to_grace_end is not None else 0.0
        )


def _write_json(path: str, data: dict[str, Any], pretty: bool) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)
            f.write("\n")
        else:
            json.dump(data, f, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _print_or_write_json(
    data: dict[str, Any], *, output: Optional[str], pretty: bool
) -> None:
    if output:
        _write_json(output, data, pretty)
        return
    if pretty:
        print(json.dumps(data, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        print(json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=True))


def _cmd_fingerprint(args: argparse.Namespace) -> int:
    payload = build_machine_fingerprint_payload()
    _print_or_write_json(payload, output=args.output, pretty=args.pretty)
    return 0


def _cmd_generate_keypair(args: argparse.Namespace) -> int:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    with open(args.private_key, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    with open(args.public_key, "wb") as f:
        f.write(
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
    return 0


def _cmd_sign(args: argparse.Namespace) -> int:
    with open(args.payload, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("payload JSON must be an object")
    document = build_signed_license_document(payload, args.private_key)
    _print_or_write_json(document, output=args.output, pretty=args.pretty)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline SGLang license tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fingerprint = subparsers.add_parser(
        "fingerprint",
        help="Export the current machine_fingerprint and the source identifiers used to build it.",
    )
    fingerprint.add_argument("--output", type=str, default=None)
    fingerprint.add_argument("--pretty", action="store_true")
    fingerprint.set_defaults(func=_cmd_fingerprint)

    generate_keypair = subparsers.add_parser(
        "generate-keypair",
        help="Generate an Ed25519 keypair for signing and verifying licenses.",
    )
    generate_keypair.add_argument("--private-key", type=str, required=True)
    generate_keypair.add_argument("--public-key", type=str, required=True)
    generate_keypair.set_defaults(func=_cmd_generate_keypair)

    sign = subparsers.add_parser(
        "sign",
        help="Sign a license payload JSON file with an Ed25519 private key.",
    )
    sign.add_argument("--payload", type=str, required=True)
    sign.add_argument("--private-key", type=str, required=True)
    sign.add_argument("--output", type=str, default=None)
    sign.add_argument("--pretty", action="store_true")
    sign.set_defaults(func=_cmd_sign)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
