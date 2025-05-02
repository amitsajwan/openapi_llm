import os
from openai import AzureOpenAI

# --- Configuration ---
# Load from environment variables (recommended)
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION") # e.g., "2024-02-15-preview"
deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") # Your deployment name

# OR, uncomment and set directly (less secure for production)
# azure_endpoint = "YOUR_AZURE_OPENAI_ENDPOINT"
# api_key = "YOUR_AZURE_OPENAI_API_KEY"
# api_version = "YOUR_API_VERSION" # e.g., "2024-02-15-preview"
# deployment_name = "YOUR_CHAT_DEPLOYMENT_NAME" # e.g., "gpt-4" or "gpt-35-turbo"

if not all([azure_endpoint, api_key, api_version, deployment_name]):
    raise ValueError(
        "Please set the environment variables AZURE_OPENAI_ENDPOINT, "
        "AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, and "
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME."
    )

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version=api_version,
)

# --- Method 1: Instruction in System Message ---
print("\n--- Method 1: Instruction in System Message ---")
messages_method1 = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Please think step-by-step. Show your reasoning clearly in a 'Scratchpad' section before providing the 'Final Answer'."
    },
    {
        "role": "user",
        "content": "A farmer has 15 sheep. All but 8 die. How many sheep does the farmer have left?"
    }
]

try:
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages_method1,
        temperature=0.2, # Lower temperature for more predictable reasoning
        max_tokens=400
    )
    print("Assistant's Response:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"An error occurred: {e}")


# --- Method 2: Instruction in User Message ---
print("\n--- Method 2: Instruction in User Message ---")
messages_method2 = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant." # More generic system message
    },
    {
        "role": "user",
        "content": "Solve the following math problem. Show your work step-by-step in a 'Reasoning' section and then state the 'Final Answer'. Problem: If Lisa is 10 years old and her brother Tom is twice her age, how old will Tom be in 5 years?"
    }
]

try:
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages_method2,
        temperature=0.2,
        max_tokens=400
    )
    print("Assistant's Response:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"An error occurred: {e}")


# --- Method 3: Structured Formatting Request in User Message ---
print("\n--- Method 3: Structured Formatting Request in User Message ---")
user_prompt_structured = """
Problem: A car travels at 60 km/h for 2 hours and then at 80 km/h for 1.5 hours. Calculate the total distance traveled.

Instructions:
1. Calculate the distance covered in the first part.
2. Calculate the distance covered in the second part.
3. Calculate the total distance.
4. Present your calculations clearly under a 'Scratchpad' heading.
5. Provide only the final numerical answer under a 'Total Distance' heading.

Please follow this exact format.
"""
messages_method3 = [
    {
        "role": "system",
        "content": "You are an AI assistant that follows formatting instructions precisely."
    },
    {
        "role": "user",
        "content": user_prompt_structured
    }
]

try:
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages_method3,
        temperature=0.1, # Very low temperature to encourage format adherence
        max_tokens=500
    )
    print("Assistant's Response:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"An error occurred: {e}")


# --- Method 4: Chain-of-Thought (Few-Shot Example) ---
print("\n--- Method 4: Chain-of-Thought (Few-Shot Example) ---")
messages_method4 = [
    {
        "role": "system",
        "content": "You are a math problem solver. Show your reasoning step-by-step."
    },
    # Example 1
    {
        "role": "user",
        "content": "John has 5 apples. He buys 3 bags with 4 apples each. How many apples?"
    },
    {
        "role": "assistant",
        "content": """
        Scratchpad:
        - John starts with 5 apples.
        - He buys 3 bags.
        - Each bag has 4 apples.
        - Apples bought = 3 bags * 4 apples/bag = 12 apples.
        - Total apples = initial apples + apples bought = 5 + 12 = 17 apples.
        Final Answer: John has 17 apples.
        """
    },
    # Actual Question
    {
        "role": "user",
        "content": "A bakery made 100 cookies. They sold 3 boxes of 12 cookies and 2 boxes of 10 cookies. How many cookies are left?"
    }
]

try:
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages_method4,
        temperature=0.2,
        max_tokens=500
    )
    print("Assistant's Response:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"An error occurred: {e}")
