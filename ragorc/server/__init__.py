"""The HTTP service.

    uvicorn ragorc.server.app:app --host 0.0.0.0 --port 8000

or, with explicit configuration:

    from ragorc.server import create_app
    app = create_app(Settings(environment="prod"))

Importing this package is free. :mod:`ragorc.server.app` imports FastAPI, uvicorn
and sse-starlette *inside* the functions that need them, and the module-level
``app`` attribute is materialized on first access — so ``from ragorc.server import
create_app`` costs nothing until it is called, and :class:`~ragorc.server.app.RagService`
can be imported by the CLI, which needs the composition but not a web framework.

To exercise the HTTP layer without a datastore, override the one dependency every
handler resolves the service through:

    app = create_app(settings)
    app.dependency_overrides[service_dependency] = lambda: my_stub
    client = TestClient(app)  # no lifespan, so no pools and no ONNX session
"""

from __future__ import annotations

from ragorc.server.app import RagService, create_app, service_dependency

__all__ = ["RagService", "create_app", "service_dependency"]
