import os, json, requests
api_key = os.getenv("NVIDIA_NIM_API_KEY")

prompt = "Hello, world!"
url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 1024,
    "temperature": 0.2
}
resp = requests.post(url, headers=headers, json=data)
print("Status Code:", resp.status_code)
print("Response:", resp.text)
