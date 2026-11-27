import asyncio
import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
import sys

from announcementExtraction import get_latest_announcement_content

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_PATH = SCRIPT_DIR / "announcements.md"
# REFRESH_INTERVAL = 60 * 60  # 60 minutes
REFRESH_INTERVAL = 1

logger = logging.getLogger(__name__)

refresh_lock = asyncio.Lock()


def calculate_hash(content: str) -> str:
    """Calculate SHA-256 hash of announcement content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def read_cached_hash() -> str | None:
    """Read the content hash from existing cache frontmatter."""
    if not CACHE_PATH.exists():
        return None

    try:
        content = CACHE_PATH.read_text(encoding="utf-8")

        match = re.search(
            r"^content_hash:\s*([a-f0-9]{64})\s*$",
            content,
            re.MULTILINE,
        )

        return match.group(1) if match else None

    except OSError:
        logger.exception("Failed to read announcement cache")
        return None


def atomic_write(content: str) -> None:
    """Atomically replace the announcement cache."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp_path = CACHE_PATH.with_suffix(".tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_path, CACHE_PATH)

    finally:
        if temp_path.exists():
            temp_path.unlink()


async def refresh_announcements() -> None:
    """Fetch announcements and update cache only when content changes."""

    async with refresh_lock:
        try:
            # Fetch latest announcement content.
            content = await asyncio.to_thread(
                get_latest_announcement_content
            )

            if not content or not content.strip():
                logger.warning("Announcement fetch returned empty content")
                return

            # Calculate content hash.
            new_hash = calculate_hash(content)

            # Compare against existing cache.
            existing_hash = read_cached_hash()

            if existing_hash == new_hash:
                logger.info("Announcements unchanged; skipping cache update")
                return

            # Generate frontmatter.
            extracted_at = datetime.now(timezone.utc).isoformat()

            cached_content = (
                "---\n"
                f"extracted_at: {extracted_at}\n"
                f"content_hash: {new_hash}\n"
                "---\n\n"
                f"{content}"
            )

            # Atomically replace cache.
            atomic_write(cached_content)

            logger.info("Announcement cache successfully updated")

        except Exception:
            logger.exception("Announcement refresh failed")


async def announcement_refresh_worker() -> None:
    """Refresh immediately, then repeat every 60 minutes."""

    while True:
        await refresh_announcements()
        await asyncio.sleep(REFRESH_INTERVAL)


async def start_announcement_refresh() -> asyncio.Task:
    """Start the background refresh task."""

    return asyncio.create_task(
        announcement_refresh_worker(),
        name="announcement-refresh-worker",
    )


def get_cached_announcements_content() -> str:
    """Return cached announcement content without YAML frontmatter."""

    if not CACHE_PATH.exists():
        return ""

    try:
        content = CACHE_PATH.read_text(encoding="utf-8")

        # Remove YAML frontmatter.
        match = re.match(
            r"\A---\s*\n.*?\n---\s*\n?",
            content,
            re.DOTALL,
        )

        if match:
            content = content[match.end():]

        return content.strip()

    except OSError:
        logger.exception("Failed to read announcement cache")
        return ""

async def main():
    worker = await start_announcement_refresh()

    # Keep the application running
    await worker

if __name__ == "__main__":
    # asyncio.run(refresh_announcements())

    # print(get_cached_announcements_content()[:100])
    asyncio.run(main())