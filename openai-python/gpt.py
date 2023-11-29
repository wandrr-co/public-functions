from openai import OpenAI

def chat_with_gpt(key,context,prompt, model):
    client = OpenAI(api_key = key)
    model = model
    completion = client.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": context},
        {"role": "user", "content": prompt}
      ]
    )
    return completion.choices[0].message.content
