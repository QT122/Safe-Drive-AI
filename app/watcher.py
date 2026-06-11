"""File system watcher for real-time image processing."""
import os
import time
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class ImageWatcher(FileSystemEventHandler):
    """Watches a directory for new/modified .jpg images and sends them to a callback."""

    def __init__(self, callback, extensions: tuple = (".jpg", ".jpeg", ".png")):
        self.callback = callback
        self.extensions = extensions
        self.last_processed: str | None = None
        self._running = False

    def _should_process(self, path: str) -> bool:
        return (
            not os.path.isdir(path)
            and path.lower().endswith(self.extensions)
            and path != self.last_processed
        )

    def on_created(self, event):
        if not event.is_directory and self._should_process(event.src_path):
            self._process(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and self._should_process(event.src_path):
            self._process(event.src_path)

    def _process(self, path: str):
        self.last_processed = path
        for attempt in range(10):
            try:
                self.callback(path)
                return
            except (OSError, IOError):
                time.sleep(0.2)
        print(f"Failed to read image after 10 attempts: {os.path.basename(path)}")


def start_watching(folder: str | Path, callback, block: bool = True):
    """Start a watchdog observer on the given folder.

    Args:
        folder: Directory to watch for new images.
        callback: Function called with the file path when a new image arrives.
        block: If True, blocks the current thread. If False, starts in background.
    Returns:
        The Observer instance (so the caller can stop it).
    """
    folder = str(folder)
    os.makedirs(folder, exist_ok=True)

    observer = Observer()
    handler = ImageWatcher(callback)
    observer.schedule(handler, folder, recursive=False)
    observer.start()
    print(f"Watching folder: {folder}")

    if not block:
        return observer

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    return observer
