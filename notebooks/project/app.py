from torchvision import models, transforms
import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from groq import Groq
from google.colab import userdata
import os

# selected food list (must match training classes and order)
FOOD_CLASSES = ['apple_pie',
 'baklava',
 'bibimbap',
 'ceviche',
 'cheesecake',
 'chicken_curry',
 'club_sandwich',
 'donuts',
 'dumplings',
 'edamame',
 'escargots',
 'falafel',
 'french_fries',
 'fried_rice',
 'grilled_salmon',
 'guacamole',
 'gyoza',
 'hamburger',
 'hot_dog',
 'hummus',
 'ice_cream',
 'lasagna',
 'macarons',
 'miso_soup',
 'mussels',
 'nachos',
 'oysters',
 'pad_thai',
 'pancakes',
 'pho',
 'pizza',
 'sashimi',
 'spaghetti_bolognese',
 'spring_rolls',
 'steak',
 'strawberry_shortcake',
 'sushi',
 'tacos',
 'tiramisu',
 'waffles']

# --- 2. load trained model ---
@st.cache_resource
def load_vision_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # recreate the model architecture (must match training)

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1024), 
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, len(FOOD_CLASSES))
    )

    # load the weights and move model to device
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, device

# Llama culture translator function
def ask_llama_chef(food_name, user_origin, api_key):
    if not api_key:
        return "❌ Please enter your API Key in the sidebar!"
    
    try:
        client = Groq(api_key=api_key)
        # person from user_origin is looking at food_name, explain it in a culturally relevant way
        prompt = f"""
        Context: You are a culinary cultural expert. A person from {user_origin} is looking at a dish called "{food_name}".
        
        Task: 
        1. Describe the taste and texture using analogies that someone from {user_origin} would easily understand.
        2. Briefly explain the history of this dish.
        3. Explain any unique cultural "fun facts" (e.g., if it's like Swiss surströmming or blue cheese).
        
        Tone: Friendly and storytelling. Language: English. Keep it under 150 words.
        """
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# --- 4. UI  ---
st.set_page_config(page_title="Flavor Bridge", page_icon="🌉")
st.title("🌉 Flavor Bridge")
st.markdown("### *Crossing Cultures, One Plate at a Time*")

# side bar settings
with st.sidebar:
    st.header("⚙️ Configuration")
    # user input for cultural background
    user_home = st.text_input("Where are you from?", "Canada")
    
    # API Key 
    try:
        default_key = userdata.get("GROQ_API_KEY")
    except:
        default_key = ""
    api_key = st.text_input("Groq API Key", value=default_key, type="password")

#file upload
uploaded_file = st.file_uploader("Upload a food photo...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button("Analyze & Translate Culture"):
    
        # Image classification
        # model_path = "/content/drive/MyDrive/FlavorBridge_Project/food_best.pth" 
        model_path ="food_best.pth" # local path for Colab
    
        model, device = load_vision_model(model_path)
        
        if model:
            # image preprocessing (must match training transformations)
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            input_tensor = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                food_name = FOOD_CLASSES[pred.item()].replace("_", " ")
            
            st.success(f"Detected: **{food_name.title()}**")
            
    
            #LLM story generation
            with st.spinner(f"Llama is translating the flavor for someone from {user_home}..."):
                explanation = ask_llama_chef(food_name, user_home, api_key)
                st.info(explanation)
        else:
            st.error("Model file not found! Check your Google Drive path.")