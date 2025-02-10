import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import requests

from trubrics.config import (
    DEFAULT_FLUSH_BATCH_SIZE,
    DEFAULT_FLUSH_INTERVAL,
    DEFAULT_FLUSH_PERIODIC_CHECK,
    MAX_FLUSH_BATCH_SIZE,
    MIN_FLUSH_INTERVAL,
)
from trubrics.logger import trubrics_logger


class Trubrics:
    def __init__(
        self,
        api_key: str,
        host: str = "https://app.trubrics.com/api/ingestion",
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        flush_batch_size: int = DEFAULT_FLUSH_BATCH_SIZE,
        flush_periodic_check: float = DEFAULT_FLUSH_PERIODIC_CHECK,
        logger: logging.Logger = trubrics_logger,
    ):
        f"""
        Initialize the Trubrics client.
        Args:
            api_key (str): The API key for the Trubrics account.
            host (str): The host URL for the Trubrics API.
            flush_interval (int): The interval in seconds after which events should be flushed. Minimum possible value is {MIN_FLUSH_INTERVAL}.
            flush_batch_size (int): The number of events to flush at a time. Max possible value is {MAX_FLUSH_BATCH_SIZE}.
            flush_periodic_check (float): The interval in seconds between periodic flush checks.
            logger (logging.Logger): The logger to use for logging.

        """
        self.host = host
        self.api_key = api_key
        self.queue: list[dict] = []
        self.last_flush_time = datetime.now(timezone.utc)
        self.is_flushing = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.logger = logger

        self._init_flush_parameters(
            flush_interval, flush_batch_size, flush_periodic_check
        )
        self._thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self._thread.start()

    def _init_flush_parameters(
        self, flush_interval: float, flush_batch_size: int, flush_periodic_check: float
    ):
        if flush_interval < MIN_FLUSH_INTERVAL:
            self.logger.warning(
                f"Flush interval {flush_interval} is too low. Setting it to minimum allowed value of {MIN_FLUSH_INTERVAL}."
            )
            flush_interval = MIN_FLUSH_INTERVAL
        if flush_batch_size > MAX_FLUSH_BATCH_SIZE:
            self.logger.warning(
                f"Flush batch size {flush_batch_size} is too high. Setting to maximum allowed value of {MAX_FLUSH_BATCH_SIZE}."
            )
            flush_batch_size = MAX_FLUSH_BATCH_SIZE
        if flush_periodic_check > flush_interval:
            self.logger.warning(
                f"Periodic flush check interval {flush_periodic_check} is higher than defined \
                    flush interval period. Setting it to Flush interval period {flush_interval}."
            )
            flush_periodic_check = flush_interval

        self.flush_interval = flush_interval
        self.flush_batch_size = flush_batch_size
        self.flush_periodic_check = flush_periodic_check

    def track(
        self,
        user_id: str,
        event: str,
        properties: dict | None = None,
        timestamp: datetime | None = None,
    ):
        """
        Track an event.
        Args:
            user_id (str): The ID of the user.
            event (str): The name of the event.
            properties (dict | None): Additional properties to track.
            timestamp (datetime | None): The timestamp of event. If None, the current time in UTC is used. If not a datetime object, the event is ignored.
        """

        event_dict = {
            "user_id": user_id,
            "event": event,
            "properties": properties,
            "timestamp": (
                timestamp.isoformat()
                if timestamp
                else datetime.now(timezone.utc).isoformat()
            ),
        }
        with self._lock:
            self.queue.append(event_dict)
            self.logger.info(
                f"Event `{event}` by user `{user_id}` has been added to queue."
            )

    def track_llm(
        self,
        user_id: str,
        prompt: str,
        assistant_id: str,
        generation: str,
        properties: dict | None = None,
        timestamp: datetime | None = None,
        latency: int = 1,
    ):
        """
        Track an LLM prompt and generation.
        Args:
            user_id (str): The ID of the user.
            prompt (str): The prompt given to the LLM.
            assistant_id (str): The ID of the assistant.
            generation (str): The generated response from the LLM.
            properties (dict | None): Additional properties to track.
            timestamp (datetime | None): The timestamp of the generation event. If None, the current time in UTC is used.
            latency (int): The latency in milliseconds between the prompt and the generation. Defaults to 1.
        """
        generation_timestamp = timestamp or datetime.now(timezone.utc)
        prompt_timestamp = generation_timestamp - timedelta(milliseconds=latency)

        self.track(
            event="Prompt",
            user_id=user_id,
            properties={"$text": prompt, **(properties or {})},
            timestamp=prompt_timestamp,
        )

        self.track(
            event="Generation",
            user_id=user_id,
            properties={
                "$text": generation,
                "$assistant_id": assistant_id,
                "$prompt": prompt,
                "latency(ms)": latency,
                **(properties or {}),
            },
            timestamp=generation_timestamp,
        )

    def flush(self):
        """Flush the event queue."""
        events = []
        with self._lock:
            queue_len = len(self.queue)
            if queue_len and not self.is_flushing:
                self.logger.info(f"Flushing {queue_len} events.")

                self.is_flushing = True
                events = self.queue[:]
                self.queue.clear()

        if events:
            for batch_id in range(0, len(events), self.flush_batch_size):
                batch = events[batch_id : batch_id + self.flush_batch_size]
                success = self._post(batch)

                if not success:
                    self.logger.warning(
                        f"Retrying flush of batch {batch_id} of {len(batch)} events."
                    )
                    time.sleep(5)
                    self._post(batch)

            self.last_flush_time = datetime.now(timezone.utc)
            self.logger.info(f"Flush of {len(events)} events completed.")

        with self._lock:
            self.is_flushing = False

    def close(self):
        """Stop the background thread and perform a final flush of the event queue."""
        self._stop_event.set()
        self._thread.join()
        self.flush()
        self.logger.info("Background thread stopped and final flush completed.")

    def _post(self, events: list[dict]):
        with requests.Session() as session:
            try:
                response = session.post(
                    f"{self.host}/publish_events",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self.api_key,
                    },
                    json=events,
                )
                response.raise_for_status()
                self.logger.info(f"{len(events)} events sent to Trubrics.")
                return True
            except requests.exceptions.HTTPError as e:
                error_message = response.text if response.status_code != 200 else str(e)
                try:
                    error_message = json.loads(error_message).get(
                        "detail", error_message
                    )
                except json.JSONDecodeError:
                    pass
                self.logger.error(
                    f"Error flushing {len(events)} events: {error_message}"
                )
                return False
            except Exception as e:
                self.logger.error(
                    f"Unexpected error flushing {len(events)} events: {e}"
                )
                return False

    def _periodic_flush(self):
        while not self._stop_event.is_set():
            time.sleep(self.flush_periodic_check)

            queue_len = len(self.queue)
            now = datetime.now(timezone.utc)
            time_since_last_flush = (now - self.last_flush_time).total_seconds()
            if (
                queue_len >= self.flush_batch_size
                or time_since_last_flush >= self.flush_interval
            ):
                self.logger.info(f"queue_len {queue_len}")
                self.logger.info(f"time_since_last_flush {time_since_last_flush}")
                self.flush()
