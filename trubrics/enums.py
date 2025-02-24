from enum import Enum


class EventTypes(Enum):
    event = "event"
    llm_event = "llm_event"


class IngestionEndpoints(Enum):
    events = "publish_events"
    llm_events = "publish_llm_events"
