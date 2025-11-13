from dotenv import load_dotenv
import os

# .env ഫയലിൽ നിന്ന് എൻവയോൺമെന്റ് വേരിയബിളുകൾ ലോഡ് ചെയ്യുക
load_dotenv()

# OpenAI API കീ സജ്ജമാക്കുക
openai.api_key = os.getenv("OPENAI_API_KEY")