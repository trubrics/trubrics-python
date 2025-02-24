import logging
import os

from trubrics import Trubrics

trubrics = Trubrics(api_key=os.environ["TRUBRICS_API_KEY"], host="http://127.0.0.1:8001/api/ingestion")
trubrics.logger.setLevel(logging.DEBUG)


trubrics.track(event="Page view", user_id="user_id", properties={"page": "home"})
trubrics.track(event="Page view", user_id="user_id", properties={"page": "events"})

trubrics.track_llm(
    user_id="user_id",
    prompt="What is Trubrics?",
    assistant_id="gpt4o",
    generation="Trubrics is a product analytics platform for AI applications.",
    latency=2,
)

trubrics.close()
