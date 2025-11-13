import openai
import os

# എൻവയോൺമെന്റ് വേരിയബിളുകൾ ലോഡ് ചെയ്യുക
from dotenv import load_dotenv
load_dotenv()

# എൻവയോൺമെന്റ് വേരിയബിളിൽ നിന്ന് API കീ ലോഡ് ചെയ്യുക
api_key = os.getenv("OPENAI_API_KEY")
print(f"API key from .env: {api_key}")

# API കീ സജ്ജമാക്കുക
if api_key:
    openai.api_key = api_key
    print("OpenAI API key set successfully.")
else:
    # എൻവയോൺമെന്റിൽ നിന്ന് ലഭിച്ചില്ലെങ്കിൽ നേരിട്ട് സജ്ജമാക്കുക
    openai.api_key = "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"
    print("OpenAI API key set directly.")

try:
    # ഒരു ചെറിയ ടെസ്റ്റ് റിക്വസ്റ്റ് - പുതിയ മോഡൽ ഉപയോഗിച്ച്
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, this is a test."}
        ],
        max_tokens=5
    )
    print("API key is working!")
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print(f"API key error: {e}")

try:
    # DALL-E ടെസ്റ്റ്
    response = openai.Image.create(
        prompt="A cute cat in the style of Studio Ghibli",
        n=1,
        size="1024x1024"
    )
    print("DALL-E API is working!")
    print("Image URL:", response['data'][0]['url'])
except Exception as e:
    print(f"DALL-E API error: {e}")