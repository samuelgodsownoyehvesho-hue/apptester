"""Reachability probe for model provider endpoints.

A provider being "free" is irrelevant if the network cannot reach it. This
tests each candidate host for DNS, TCP, and a completed TLS handshake, because
they fail differently and point at different fixes:

* DNS failure      -> name resolution or offline
* TCP failure      -> routing, firewall, or the host is down
* TLS failure      -> usually interception or SNI-based blocking
* HTTP 401/403     -> reachable, credentials are the remaining problem

Run with ``uv run python scripts/net_check.py``.
"""

from __future__ import annotations

import socket
import ssl
import sys

import httpx

#: Hosts worth knowing about, grouped by whether we currently depend on them.
HOSTS: list[tuple[str, str]] = [
    ("gemini", "generativelanguage.googleapis.com"),
    ("aistudio", "aistudio.google.com"),
    ("nvidia-nim", "integrate.api.nvidia.com"),
    ("nvidia-build", "build.nvidia.com"),
    ("openai", "api.openai.com"),
    ("openrouter", "openrouter.ai"),
    ("sambanova", "api.sambanova.ai"),
    ("huggingface", "huggingface.co"),
    ("ollama-dl", "ollama.com"),
    # Known-good controls: both were reached successfully during setup.
    ("control-pypi", "pypi.org"),
    ("control-github", "github.com"),
]

TIMEOUT = 10.0


def probe_tcp(host: str, port: int = 443) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=TIMEOUT):
            return True, "connected"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def probe_tls(host: str) -> tuple[bool, str]:
    ctx = ssl.create_default_context()
    try:
        with (
            socket.create_connection((host, 443), timeout=TIMEOUT) as sock,
            ctx.wrap_socket(sock, server_hostname=host) as tls,
        ):
            return True, f"{tls.version()}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def probe_http(host: str) -> str:
    try:
        resp = httpx.get(f"https://{host}/", timeout=TIMEOUT, follow_redirects=True)
        # Any HTTP status means the transport worked and the host is reachable.
        return f"HTTP {resp.status_code}"
    except Exception as exc:
        return f"{type(exc).__name__}"


def main() -> int:
    print(f"python {sys.version.split()[0]} / {ssl.OPENSSL_VERSION}")
    print(f"timeout {TIMEOUT:.0f}s per probe\n")

    header = f"{'label':16} {'dns':16} {'tcp':10} {'tls':8} {'http':22}"
    print(header)
    print("-" * len(header))

    reachable: list[str] = []
    for label, host in HOSTS:
        try:
            ip = socket.gethostbyname(host)
        except Exception as exc:
            print(f"{label:16} {type(exc).__name__:16} {'-':10} {'-':8} {'-':22}")
            continue

        tcp_ok, _ = probe_tcp(host)
        tls_ok, tls_note = probe_tls(host)
        http_note = probe_http(host) if tls_ok else "(tls failed)"
        if tls_ok:
            reachable.append(label)

        tls_display = tls_note if tls_ok else "FAIL"
        print(
            f"{label:16} {ip:16} {'ok' if tcp_ok else 'FAIL':10} "
            f"{tls_display:8} {http_note:22}"
        )

    print(f"\nTLS reachable: {', '.join(reachable) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
