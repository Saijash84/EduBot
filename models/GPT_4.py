# gpt4_model.py
import openai

openai.api_key = 'your-api-key'

def generate_text_response(prompt):
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text
