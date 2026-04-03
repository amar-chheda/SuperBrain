"""Local notifier adapter for development/testing."""

import logging

from superbrain.app.application.ports import TelegramClient

logger = logging.getLogger(__name__)


class LoggingTelegramNotifier(TelegramClient):
    """Log outbound digest notifications instead of sending to Telegram."""

    def send_message(self, chat_id: str, text: str) -> None:
        """Log a notification payload."""

        logger.info("digest_notification", extra={"chat_id": chat_id, "text": text})
