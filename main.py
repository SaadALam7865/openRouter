from dotenv import load_dotenv
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

# Load .env file
load_dotenv()
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

if not openrouter_api_key:
    raise ValueError('OPENROUTER_API_KEY is not set in the .env file')

# Setup the OpenRouter client
external_client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url='https://api.openrouter.ai/v1',
)

# Choose a supported model (safe one like mistral or gemini)
model = OpenAIChatCompletionsModel(
    model='google/gemini-1.5-flash',
    openai_client=external_client
)

# Configure agent run
config = RunConfig(
    model=model,
    model_provider='external_client',
    tracing_disabled=True,
)

# Create the agent
agent = Agent(
    name='writer agent',
    instructions='You are a writer agent. Generate stories, poems, or any creative content based on the provided input.'
)

# Run the agent synchronously
response = Runner.run_sync(
    agent,
    input='write a short story about a robot learning to love.',
    run_config=config
)

# Print the output
print('Agent Output:\n', response.final_output if hasattr(response, 'final_output') else response)
