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
    flush_at=20,       # events
    is_verbose=False
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

## Configuration Options

- api_key: Your Trubrics API key
- flush_interval: Time in seconds between automatic flushes (default: 10)
- flush_at: Number of events that trigger a flush (default: 20)
- is_verbose: Enable detailed logging (default: False)

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
1. Update version in pyproject.toml
2. Create and push a new tag:

``` bash
git tag v1.0.0
git push origin v1.0.0
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
