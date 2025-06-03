# if you dont use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()

#Step1: Setup GROQ API key
import os

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")

#Step2: Convert image to required format
import base64

image_path="./skin_rash.jpg"
image_file=open(image_path, "rb")
encoded_image=base64.b64encode(image_file.read()).decode('utf-8')

#Step3: Setup Multimodal LLM 
from groq import Groq

# First, get medical context
initial_prompt = """You are an experienced dermatologist. Based on the following query, provide a brief medical context and specific aspects to look for in the image:
Query: {query}

Please provide:
1. Relevant medical conditions to consider
2. Key visual indicators to look for
3. Important questions to ask the patient
4. Potential risk factors to assess

Keep your response concise and focused on dermatological aspects."""

client = Groq()
# Get medical context first
context_messages = [
    {
        "role": "user",
        "content": initial_prompt.format(query="Is there something wrong with my face like acne?")
    }
]

context_completion = client.chat.completions.create(
    messages=context_messages,
    model="llama2-70b-4096"
)

medical_context = context_completion.choices[0].message.content

# Now use the context with the vision model
query = f"""As a dermatologist, analyze this image considering the following medical context:
{medical_context}

Original query: Is there something wrong with my face like acne?"""

model = "meta-llama/llama-4-scout-17b-16e-instruct"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": query
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                },
            },
        ],
    }
]

chat_completion = client.chat.completions.create(
    messages=messages,
    model=model
)

print("Medical Context:")
print(medical_context)
print("\nFinal Analysis:")
print(chat_completion.choices[0].message.content)