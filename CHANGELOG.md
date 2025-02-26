# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.8.3] - 2025-02-26
### Changed
- Change logging level to debug for flushing events and adding event to queue

### Added
- LLM events are now routed to the new publis_llm_events endpoint

## [1.8.2] - 2025-02-10
### Fixed
- Added missing flushing limit values, and setup automated flush if limit is reached
### Changed
- Change LLM tracking latency to seconds and remove default latency from event properties

## [1.8.1] - 2025-02-07
### Fixed
- Fixed error logging when flushing events

## [1.8.0] - 2025-02-06
### Added
- Update logging, removing `is_verbose` parameter
- Removed CONTRIBUTING.md, as info is now in README.md
- Brought back repository to PyPI, and uses uv to build and publish
- Changed name to "trubrics" for easier installation

## [1.6.2] - 2023-10-24
### Added
- Option to user text_area for textual feedback collection in Streamlit. Thanks @hamdan-27

## [1.6.1] - 2023-09-12
### Fixed
- Now compatible for pydantic>=1.5 (v1 and v2)

## [1.6.0] - 2023-09-08
[streamlit-feedback](https://github.com/trubrics/streamlit-feedback/stargazers) has been revamped for tighter integration with st chat elements
### Changed
- fix dependency with `streamlit-feedback==0.1.2`
- `single_submit` has been removed from `.st_feedback()`, and replaced with `disable_with_score`

### Added
- new streamlit chatbot streaming example

### Fixed
- streamlit llm examples

## [1.5.2] - 2023-09-04
### Fixed
- `user_response` is dict rather than `Response` for feedback_type="text"
- `collector.st_feedback(..., save_to_trubrics=False)` returns a python dict
- pydantic dependency updated and `.dict()` updated to `.model_dump()`

## [1.5.1] - 2023-08-29
### Changed
- Rename `model_config` with `config_model` for pydantic

## [1.5.0] - 2023-08-29
⚠️ This version has breaking changes.

### Depreciated
- All validations have been depreciated.

### Added
- New hierarchy for organising `Projects` in Trubrics
- New `log_prompts` method for tracing user prompts and model generations

### Changed
- Updated API (Python SDK & Streamlit) for collecting feedback

## [1.4.5] - 2023-08-11
### Changed
- Added option to skip success or error message upon saving feedback to Trubrics

## [1.4.4] - 2023-08-08
### Changed
- Replaced streamlit python feedback components with React components

## [1.4.3] - 2023-07-17
### Changed
- Move all validations dependencies to "pip install trubrics[validations]"

## [1.4.2] - 2023-07-5
### Changed
- Make UTC default for saving feedback responses

### Fixed
- Use default factory for Feedback object `created_on`

## [1.4.1] - 2023-06-30
### Changed
- Refactored all feedback docs to fit new trubrics feedback API
- Reorganised all examples

### Fixed
- Fixed flask example app
- Fixed titanic example app
- Fixed llm example app

## [1.4.0] - 2023-06-21
Refactor of Feedback collector to fit to new Trubrics user insights platform

## [1.3.7] - 2023-05-07
### Changed
- Added trubrics_platform_auth into titanic example app
- Upgrade streamlit>=1.18.0

## [1.3.6] - 2023-05-03
### Fixed
- Fix `Unauthenticated` error in Trubrics platform auth with refresh function parameter

## [1.3.5] - 2023-04-25
### Fixed
- Fixed `trubrics run` with new .json file corresponding to new `Trubric` data model

## [1.3.4] - 2023-04-20
### Added
- Functionality to fail a Trubric run (cli or notebook) based on the severity of validations
- New integration with MlFlow 🎉 - you can now:
  - Validate an mlflow run with Trubrics with `mlflow.evaluate(evaluators="trubrics")`
  - Save all validation results to the MLflow UI
  - Write custom python functions to validate your data or models with MLflow

### Changed
- Changed data model of `Trubric` object
- Tutorials for classification and regression models added to docs, ready to run in google colab
- Removed notebook run in the docs CI

## [1.3.3] - 2023-04-05
### Fixed
- Users can now trubrics init with environment variables
- Clearer trubrics init documentation

## [1.3.1.2] - 2023-03-31

### Fixed
- Users can now trubrics init without manual prompts

## [1.3.2] - 2023-03-28

### Added
- New methods of `FeedbackCollector` to allow for the use of standalone Trubrics UI components. E.g. `collector.st_faces_ui()`
- Open question feedback option to collect with feedback types "issue" & "faces"
- Disable on click functionality for a smoother user experience with feedback types
- `Feedback` pydantic model returned from `st_feedback()` method

### Changed
- Updated data model for the Feedback object
- Add a note to the demo app explaining the experiment features
- Changed order of feedback and validations in README
- `Feedback` components are now decoupled from the data context

## [1.3.1] - 2023-03-18

### Added
- Custom type for streamlit FeedbackCollector
- Unit tests for streamlit FeedbackCollector
- Example code snippets for Demo app

## [1.3.0] - 2023-03-17

### Added
- A brand new, shiny FeedbackCollector for streamlit 🎉. Highlights:
  - A new demo app on Titanic, with options for auth directly in the CLI
  - New FeedbackCollector object that stores metadata of your models and dataset versions
  - New auth component for Trubrics platform
  - New st_feedback() component with multiple types available
  - Updated docs

### Fixed
- Flex dependencies versions
- Moved Streamlit, Gradio and Dash to extra_dependencies
- Moved feedback integrations to an integrations/ dir

## [1.2.9] - 2023-03-03

### Fixed

- Display up to 50 projects from Trubrics in `trubrics init`
- Add @lru_cache to get Idtoken upon each write
- Hide locals in typer print (for sensitive password on error)

## [1.2.8] - 2023-02-27

### Changed

- Refactored TrubricRun to include methods to generate a Trubric
- Update CLI docs and gifs

## [1.2.7] - 2023-02-26

### Fixed

- Added .json to python package

## [1.2.6] - 2023-02-24

### Changed

- Refactored CLI with rich prints
- Moved trubric_run_context path argument to `trubrics run` from `trubrics init`

## [1.2.5] - 2023-02-20

### Changed

- Change collaborators in feedback from email to display name
- Add archived projects filter in trubrics init

## [1.2.4] - 2023-02-20

### Added

- Python \_\_version\_\_ number in cli and \_\_init\_\_.py
- Add save_ui param to trubrics example-app in CLI

## [1.2.3] - 2023-02-16

### Fixed

- Added project_id to `trubrics init`

## [1.2.0] - 2023-02-15

### Changed

- `trubrics init` refactoring with authentication with Trubrics
- `trubrics run` refactoring to store validations to firestore (Trubrics DB)
- `Feedback` and `Trubric` have adapted data model
- Cleaned notebooks examples/ folder
- Updated docs and README with Trubrics platform references

### Added

- FeedbackCollector now has authentication for Trubrics users
- `Feedback` and `Trubric` have `save_ui()` methods for new DB

### Fixed

- Bumped streamlit version

## [1.1.1] - 2022-11-14

### Changed

- Allow for list types to be saved in result dict validation output
- Restricted extra_fields of `Trubric` pydantic model
- Changed `trubric_name` field to `name` in `Trubric` pydantic model

## [1.1.0] - 2022-11-02

### Added

- Getting started video to readme
- Features field for DataContext
- Integration with Gradio for collecting feedback
- Integration with Dash for collecting feedback

### Changed

- Adopted a functional approach to FeedbackCollector
- Separated feedback collector to collect and experiment functions
- Updated feedback collector readme and docs

## [1.0.2] - 2022-10-14

### Fixed

- Fixed github action to display changelog to release tag

## [1.0.1] - 2022-10-13

### Fixed

- Fixed save feedback log
- FeedbackCollector has simplified feedback form

## [1.0.0] - 2022-10-06

### Added

- New CHANGELOG.md
- New CONTRIBUTING.md

### Removed

- Getting started video from README and docs. Soon to be replaced with updated video

## [0.2.3] - 2022-10-03

### Added

- New metrics, cli & data_context docs

### Changed

- Restructure readme with examples that run
- Restructure docs to follow readme key features
- Complete unit tests for ModelValidator
- Separate out "contexts" from context.py into their respective folders

### Fixed

- PyPI packaged example data is now readable
