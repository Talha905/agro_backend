# import google.generativeai as genai

# # Replace with your key
# api_key = "key"

# try:
#     genai.configure(api_key=api_key)
#     print("Available Models:")
#     models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
#     for model in models:
#         print(f" - {model}")
# except Exception as e:
#     print(f"Error: {e}")