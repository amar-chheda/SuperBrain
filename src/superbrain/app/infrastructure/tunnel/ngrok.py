"""Optional ngrok tunnel for local Telegram webhook development.

Started automatically at lifespan if SUPERBRAIN_TELEGRAM_BOT_TOKEN is set
and SUPERBRAIN_TELEGRAM_WEBHOOK_URL is not explicitly configured.
Set SUPERBRAIN_TELEGRAM_WEBHOOK_URL to a real URL in production to skip ngrok entirely.
"""

import httpx
import structlog
from pyngrok import conf, ngrok

from superbrain.settings import Settings

log = structlog.get_logger(__name__)

_WEBHOOK_PATH = "/bot/telegram/webhook"


async def start(settings: Settings, port: int = 8000) -> str | None:
    """Start an ngrok tunnel and register the Telegram webhook.

    Returns the public tunnel URL, or None if the bot token is not configured.
    Skips ngrok if SUPERBRAIN_TELEGRAM_WEBHOOK_URL is already set (production mode).
    """
    if not settings.telegram_bot_token:
        return None

    public_url = settings.telegram_webhook_url
    if not public_url:
        public_url = _open_tunnel(settings, port)

    if public_url:
        await _register_webhook(settings.telegram_bot_token, public_url)

    return public_url


def stop() -> None:
    """Disconnect all active ngrok tunnels."""
    try:
        ngrok.kill()
    except Exception:
        pass


def _open_tunnel(settings: Settings, port: int) -> str | None:
    try:
        if settings.ngrok_authtoken:
            conf.get_default().auth_token = settings.ngrok_authtoken

        # Check if any ngrok process (managed or external) already has a tunnel
        # for this port by querying the ngrok local REST API directly.
        existing = _find_existing_tunnel(port)
        if existing:
            log.info("ngrok.tunnel_reused", public_url=existing, port=port)
            return existing

        tunnel = ngrok.connect(port, "http")
        url: str = tunnel.public_url.replace("http://", "https://")
        log.info("ngrok.tunnel_opened", public_url=url, port=port)
        return url
    except Exception as exc:
        log.warning("ngrok.tunnel_failed", error=str(exc))
        return None


def _find_existing_tunnel(port: int) -> str | None:
    """Query the ngrok local API to find an existing tunnel for the given port.

    ngrok binds its API to 4040 by default but falls back to 4041, 4042, etc.
    when multiple processes are running.
    """
    import urllib.request
    import json

    for api_port in range(4040, 4045):
        try:
            with urllib.request.urlopen(f"http://localhost:{api_port}/api/tunnels", timeout=1) as resp:
                data = json.loads(resp.read())
                for tunnel in data.get("tunnels", []):
                    addr = tunnel.get("config", {}).get("addr", "")
                    if f":{port}" in addr or addr == str(port):
                        public_url: str = tunnel["public_url"].replace("http://", "https://")
                        return public_url
        except Exception:
            continue
    return None


async def _register_webhook(token: str, public_url: str) -> None:
    webhook_url = f"{public_url}{_WEBHOOK_PATH}"
    api_url = f"https://api.telegram.org/bot{token}/setWebhook"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(api_url, json={"url": webhook_url})
            data = resp.json()
            if data.get("ok"):
                log.info("telegram.webhook_registered", url=webhook_url)
            else:
                log.warning("telegram.webhook_failed", response=data)
    except Exception as exc:
        log.warning("telegram.webhook_error", error=str(exc))
