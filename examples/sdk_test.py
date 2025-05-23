from trubrics import Trubrics

trubrics = Trubrics(
    api_key="LuggKCVUi-Ef3hEuQukoJUF2P-Dx98apzoUY3g41js04it5v-v2DC4BfR14SOXPT",
    host="https://app.trubrics.com/api/ingestion",
)

trubrics.track(
    event="Sign Up",
    user_id="Britney",
    properties={
        "country": "USA",
        "company": "Acme",  # Add more properties as needed
    },
)

trubrics.track_llm(
    user_id="sdk_test",
    prompt="What is Trubrics?",
    assistant_id="gpt4o",
    generation="Trubrics is a product analytics platform for AI applications.",
    latency=2,
)

trubrics.close()
