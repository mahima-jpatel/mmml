from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://mathvista.openai.azure.com/",
    api_key="8p2zkw7QFCN8ySKpe4Ipe9CunMhItcfiPhpOK5qMuE6cpogADPCuJQQJ99BCACYeBjFXJ3w3AAABACOGUbyW",
    api_version="2024-02-15-preview"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
