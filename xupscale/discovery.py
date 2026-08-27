import logging
import socket

from zeroconf import ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

from .config import Config

log = logging.getLogger("xupscale.discovery")

SERVICE_TYPE = "_xremote._tcp.local."


class Discovery:
    """Advertises this node as a second, distinct XRemote cast target so it
    shows up alongside the real receiver in the phone app's target list.

    TXT record keys mirror XRemoteDiscovery.createParamsMap() in the real
    XRemoteDesktop app so the client's parser accepts them unmodified.

    Uses zeroconf's asyncio API deliberately - the synchronous Zeroconf
    class spins its own event-loop thread only when instantiated OUTSIDE a
    running loop; called from inside our already-running asyncio loop (same
    thread) it deadlocks waiting on itself.
    """

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._azc: AsyncZeroconf | None = None
        self._info: ServiceInfo | None = None

    async def start(self) -> None:
        cfg = self._cfg
        cast_name = cfg.cast_name + cfg.cast_name_suffix
        name = f"{cast_name}.{SERVICE_TYPE}"

        properties = {
            "app.name": "XRemotexServer",
            "app.codename": "xremote_pc",
            "app.version": "3.0",
            "app.version_code": "30",
            "app.api_version": "10",
            "app.device_type": "WINDOWS",
            "app.cast_name": cast_name,
            "app.allow_cast": "true",
            "app.sync_name": cast_name,
            "app.allow_sync": "false",
        }

        self._info = ServiceInfo(
            SERVICE_TYPE,
            name,
            addresses=[socket.inet_aton(self._cfg.node_ip)],
            port=cfg.listen_port,
            properties=properties,
            server=f"{cast_name.replace(' ', '-')}.local.",
        )

        self._azc = AsyncZeroconf()
        await self._azc.async_register_service(self._info)
        log.info("mDNS: advertising %r on %s:%s", cast_name, cfg.node_ip, cfg.listen_port)

    async def stop(self) -> None:
        if self._azc and self._info:
            await self._azc.async_unregister_service(self._info)
            await self._azc.async_close()
            log.info("mDNS: service unregistered")
