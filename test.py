# test_gemini.py
import google.generativeai as genai
genai.configure(api_key="AIzaSyAXYXTHiNKJFgcn2jwnRtmme8F723Z6P6o")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Hello")
print(response.text)