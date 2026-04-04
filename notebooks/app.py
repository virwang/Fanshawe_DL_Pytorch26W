import streamlit as st
from groq import Groq
   
# --- 1. Basic Configuration ---
st.set_page_config(page_title="AI Food Guide", page_icon="🍣")

# Sidebar for API Key input to ensure privacy
with st.sidebar:
    st.title("🛠️ Settings")
    # set colab key as default, otherwise set api key as None
    api_key = st.text_input("Enter Groq API Key", 
                            value=api_key_from_colab if api_key_from_colab else "",
                            type="password")
    if not api_key:
        st.info("Get your key at [console.groq.com](https://console.groq.com/)")  


# --- 2. Llama Core Function ---
def ask_llama_chef(food_name, api_key):
    if not api_key:
        return "❌ Please enter your API Key in the sidebar!"
    
    try:
        client = Groq(api_key=api_key)
        prompt = f"""
        You are an expert Japanese travel guide serving a tourist from Canada.
        The vision model has identified this dish as "{food_name}".
        Please provide information in English: 1. Taste & Texture, 2. Travel Trivia, 3. Advice for Travelers.
        Tone: Humorous and friendly. Keep it under 150 words.
        """
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error occurred: {str(e)}"

# --- 3. UI Interface ---
st.title("🍣 AI Japanese Food Guide")
mock_predicted_label = st.selectbox(
    "Select a recognition result to test:",
    ["Sushi", "Ramen", "Takoyaki", "Tempura", "Okonomiyaki"]
)

if st.button("View Food Guide"):
    with st.spinner("Thinking..."):
        guide_text = ask_llama_chef(mock_predicted_label, api_key)
        st.chat_message("assistant", avatar="👨‍🍳").write(guide_text)
