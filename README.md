# Trubrics

A Python client for tracking and analyzing events and LLM interactions with Trubrics.

## Overview

Trubrics is a Python client that provides event tracking capabilities with a focus on LLM (Large Language Model) interactions. It features an efficient queuing system with automatic background flushing of events to the Trubrics API.

## Installation

Install using pip:
``` bash
pip install trubrics
```

Or using uv:

``` bash
pip install uv
uv pip install trubrics
```

## Key Features

- Event tracking with custom properties
- Automatic background event flushing
- Thread-safe implementation
- Configurable flush intervals and batch sizes

## Usage

### Basic Setup

``` python
from trubrics import Trubrics

client = Trubrics(
    api_key="your-api-key",
    flush_interval=10,  # seconds
    flush_batch_size=20,        # events
)
```

### Tracking Events

``` python
# Track a simple event
client.track(
    user_id="user123",
    event="button_click",
    properties={"button_type": "submit"},
)

# Track LLM interactions
client.track_llm(
    user_id="user123",
    prompt="What is the capital of France?",
    assistant_id="gpt-4",
    generation="The capital of France is Paris.",
    properties={"model": "gpt-4"},
    latency=150  # milliseconds
)
```

### Closing the Client

``` python
# Ensure all events are flushed before shutting down
client.close()
```

### Logging

The client uses python's `logging` library to log messages, by default at the `ERROR` level.

To adjust the verbosity of the default logs, specify the log level:
``` python
trubrics = Trubrics(api_key="your-api-key")
trubrics.logger.setLevel(logging.DEBUG)
```

Or specify your own logger completely:
``` python
trubrics = Trubrics(api_key="your-api-key", logger=your_cool_logger)
```

## Configuration Options

- api_key: Your Trubrics API key
- flush_interval: Time in seconds between automatic flushes (default: 10)
- flush_batch_size: Number of events that trigger a flush (default: 20)

## Development

### Requirements

The project uses the following main dependencies:
- requests==2.32.3
- certifi==2025.1.31
- charset-normalizer==3.4.1
- idna==3.10
- urllib3==2.3.0

### Setting up Development Environment

Using Make commands:

``` bash
make setup_uv_venv
make install_dev_requirements  # Install development dependencies
make install_requirements     # Install production dependencies
```

## Publishing

The package is automatically published to PyPI when a new version tag is pushed. To publish a new version:
- Create a PR from the main branch for the desired changes
- Update the version in `pyproject.toml`, and add changelog entry in `CHANGELOG.md`
- Once the PR is merged, create a tag in the repo `trubrics-python` based on the version in `pyproject.toml`, and push it
    ``` bash
    git tag -a v1.0.0 -m "Release version v1.0.0"
    git push origin v1.0.0
    ```
- Once the tag is pushed, a Github action `release.yml` will be triggered. It will:
  - Create a release in the repo `trubrics-python` based on the `CHANGELOG.md` entry
  - build the package with `uv`
  - publish it to PyPI
