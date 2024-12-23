from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="your-api-key",
)

import json

import requests

OPENROUTER_API_KEY = "your-api-key"

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        # "HTTP-Referer": f"{YOUR_SITE_URL}",  # Optional, for including your app on openrouter.ai rankings.
        # "X-Title": f"{YOUR_APP_NAME}",  # Optional. Shows in rankings on openrouter.ai.
    },
    data=json.dumps(
        {
            "model": "google/gemini-flash-1.5",  # Optional
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                            },
                        },
                    ],
                }
            ],
        }
    ),
)
print(response.json())
