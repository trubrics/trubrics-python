import logging

trubrics_logger = logging.getLogger("trubrics")
if not any(isinstance(h, logging.StreamHandler) for h in trubrics_logger.handlers):
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    trubrics_logger.addHandler(handler)
    trubrics_logger.setLevel(logging.INFO)
