"""Offline tests use real DeepEval with only its HTTP transport replaced."""

import os

import httpx
import pytest

# These must be set before DeepEval is imported during collection.
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
for name in ("TYPESAFE_API_KEY", "OPENAI_API_KEY", "CONFIDENT_API_KEY"):
    os.environ.pop(name, None)


@pytest.fixture
def mock_http(monkeypatch):
    client_class, async_class = httpx.Client, httpx.AsyncClient
    clients = []

    def install(handler):
        transport = httpx.MockTransport(handler)

        def sync_client(**kwargs):
            client = client_class(transport=transport, **kwargs)
            clients.append(client)
            return client

        def async_client(**kwargs):
            client = async_class(transport=transport, **kwargs)
            clients.append(client)
            return client

        monkeypatch.setattr(httpx, "Client", sync_client)
        monkeypatch.setattr(httpx, "AsyncClient", async_client)
        return clients

    return install


@pytest.fixture(autouse=True)
def block_real_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Offline tests attempted a real HTTP request")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked)
