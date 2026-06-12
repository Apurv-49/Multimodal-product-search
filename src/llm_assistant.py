import requests

class ProductAssistant:
    def __init__(self, api_url=None, token=None):
        # Default initialization for local deployment or inference API wrappers
        self.api_url = api_url or "https://api-inference.huggingface.co/models/google/gemma-2b-it"
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def generate_response(self, user_query: str, retrieved_items: list) -> str:
        context_str = ""
        for i, item in enumerate(retrieved_items, 1):
            context_str += f"[{i}] Item ID: {item.get('id')} | Brand: {item.get('brand')} | Category: {item.get('category')} | Color: {item.get('color')} | Gender: {item.get('gender')}\n"

        prompt = f"""You are an advanced, helpful AI E-Commerce Personal Assistant.
Given the following context of retrieved items inside our store platform, provide an informative, professional response to the customer's query.

Retrieved Products Context:
{context_str}

Customer Query: {user_query}

Your Expert Response:"""

        # Fallback template logic if no direct token/endpoint API configuration is added
        if not self.headers:
            return f"Mocked Gemma Response: I found {len(retrieved_items)} matching your request. Based on your profile, the best pick is the {retrieved_items[0].get('brand', 'Premium')} {retrieved_items[0].get('category', 'item')}."

        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 250, "temperature": 0.7}}
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            return response.json()[0]['generated_text'].split("Your Expert Response:")[-1].strip()
        except Exception as e:
            return f"Error connecting to LLM endpoint: {str(e)}"
