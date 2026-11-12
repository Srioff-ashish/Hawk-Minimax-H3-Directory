"""Outbound fetches of user-supplied URLs only reach the public internet.

The API runs on a public tunnel next to an unauthenticated ComfyUI on 127.0.0.1, and on cloud VMs that answer a
metadata service on 169.254.169.254. A URL import (``/v1/assets/from-url``, a manhwa chapter) that followed a link
there would hand whatever those return to the caller. ``public_only`` is an httpx request hook: httpx calls it for
the first request and again for every redirect, so a public page that redirects inward is refused too.

A host that does not resolve is let through: the real connection fails on its own, and test transports that never
touch the network keep working.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket

import httpx


class BlockedAddress(httpx.HTTPError):
    """The URL points at this machine or a private network."""


def is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


async def _addresses(host: str) -> list[str]:
    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        pass
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError):
        return []
    return sorted({info[4][0].split("%", 1)[0] for info in infos})


async def check_url(url: str | httpx.URL) -> None:
    """Raise BlockedAddress when the URL's host is, or resolves to, a non-public address."""
    parsed = httpx.URL(str(url))
    host = (parsed.host or "").strip("[]")
    if not host:
        raise BlockedAddress(f"{url} has no host.")
    if host.lower() in ("localhost", "localhost.localdomain") or host.lower().endswith(".localhost"):
        raise BlockedAddress(f"{host} is this machine; only public addresses can be fetched.")
    for address in await _addresses(host):
        if not is_public(address):
            raise BlockedAddress(f"{host} points at {address}, a private or local address; only public "
                                 "addresses can be fetched.")


async def public_only(request: httpx.Request) -> None:
    """httpx request hook: refuse any request (first or redirected) to a non-public address."""
    await check_url(request.url)


def hooks() -> dict:
    return {"request": [public_only]}
