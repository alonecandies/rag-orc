"""ONNX Runtime hardening for the embedding layer.

Why this module exists
----------------------
On macOS, a process that exits while ONNX Runtime sessions are still alive can
abort during teardown rather than exiting cleanly. The crash report is
unmistakable::

    Abort trap: 6 / SIGABRT
      abort
      __cxa_throw
      std::__1::recursive_mutex::lock()
      onnxruntime_pybind11_state.so
        Microsoft::Applications::Events::DebugEventSource::DispatchEvent
        Microsoft::Applications::Events::LogManagerImpl::DispatchEvent

That is ONNX Runtime's bundled Microsoft telemetry client (the 1DS SDK) trying to
dispatch an event on a background thread whose ``recursive_mutex`` has already
been destroyed by static destructors. Locking a destroyed mutex throws
``std::system_error``, the throw crosses a ``noexcept`` boundary, and the process
aborts. Nothing in the *application* is wrong when this happens — the work has
already completed — but the exit code is 134 and macOS files a crash report, so
it looks like a failure to every CI system and every operator.

It is most likely when the interpreter shuts down abruptly (a killed pipeline, a
SIGPIPE from ``| head``, a container stop) while sessions exist, and more likely
the more sessions there are: the default configuration loads three (dense,
sparse, reranker).

Two mitigations, applied in order of directness:

1. **Disable the telemetry subsystem itself.** It is the exact code in the stack
   trace, it sends usage data to Microsoft, and nothing here needs it. Disabled
   via the environment variable *before* ``onnxruntime`` is imported (the C++
   side reads it during static initialization, so setting it afterwards is too
   late) and via the Python API as a second attempt for builds that ignore it.
2. **Release sessions before static destruction.** An ``atexit`` handler drops
   the cached models while the interpreter is still alive, so ONNX tears down its
   threads in a defined order instead of racing the C++ runtime's own exit.

Both are cheap, neither changes inference behaviour, and together they turn a
crash-on-exit into a clean exit.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import sys
from typing import Any

import structlog

log = structlog.get_logger(__name__)

__all__ = ["configure_onnx_runtime", "register_shutdown_hook"]

#: Read by ONNX Runtime's C++ static initializer, so it must be set before the
#: extension module is imported.
_TELEMETRY_ENV = "ORT_DISABLE_TELEMETRY"

_configured = False
_hooks_registered: set[Any] = set()


def configure_onnx_runtime() -> None:
    """Disable ONNX telemetry and quiet its logger. Idempotent.

    Called at import time of the FastEmbed provider, which is the first thing in
    this library that can cause ``onnxruntime`` to load. If the module is somehow
    already imported, the environment variable is too late — hence the second,
    API-level attempt.
    """
    global _configured
    if _configured:
        return
    _configured = True

    # Respect an explicit operator choice: someone who has deliberately set this
    # (either way) should not be overridden by a library.
    if _TELEMETRY_ENV not in os.environ:
        os.environ[_TELEMETRY_ENV] = "1"

    if "onnxruntime" not in sys.modules:
        # The common path: nothing to do, the env var will be honoured on import.
        return

    try:
        import onnxruntime
    except ImportError:  # pragma: no cover - onnxruntime ships with fastembed
        return

    disable = getattr(onnxruntime, "disable_telemetry_events", None)
    if callable(disable):
        try:
            disable()
        except Exception as exc:  # noqa: BLE001 - never fail an import over this
            log.debug("onnx_telemetry_disable_failed", error=str(exc)[:120])

    # ONNX's default verbosity emits warnings during teardown too; 3 = ERROR.
    set_severity = getattr(onnxruntime, "set_default_logger_severity", None)
    if callable(set_severity):
        try:
            set_severity(3)
        except Exception as exc:  # noqa: BLE001
            log.debug("onnx_logger_severity_failed", error=str(exc)[:120])


def register_shutdown_hook(release: Any) -> None:
    """Register ``release`` to drop cached model sessions at interpreter exit.

    Idempotent **per callable**, not globally. It used to be guarded by a single
    module-level flag, which made it idempotent in the wrong sense: the first
    caller registered and every later one was silently dropped. Three modules in
    this package cache native sessions — FastEmbed's ONNX models, the
    late-chunking torch models and the sentence-transformers models — and only
    whichever imported first was ever released, while the module docstring above
    promised all of them.

    ``release`` is passed in rather than imported to keep this module free of a
    circular dependency on its callers.

    The handler is deliberately silent about failures. It runs during shutdown,
    when logging handlers may already be closed, and an exception raised from an
    ``atexit`` callback is printed to stderr as an unhandled error — producing
    exactly the noisy, alarming exit this module exists to prevent.
    """
    if release in _hooks_registered:
        return
    _hooks_registered.add(release)

    def _release() -> None:
        with contextlib.suppress(Exception):
            release()

    atexit.register(_release)
