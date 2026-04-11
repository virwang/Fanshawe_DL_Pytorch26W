import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from groq import Groq
import os

# basic page config
st.set_page_config(page_title="Flavor Bridge: Identify Dishes and Unlock Their Stories", page_icon=":plate_with_cutlery:", layout="centered")

# make sure this list matches the order of classes in your training dataset
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


# load model and cache it to avoid reloading on every interaction
@st.cache_resource
def load_my_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # recreate the model architecture (must match training)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(FOOD_CLASSES))
    
    # load weights and move model to device
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

# image preprocessing function to match training transformations
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

# Llama core function to generate cultural story based on food name and user origin
def ask_llama_chef(food_name, user_origin, api_key):
    try:
        client = Groq(api_key=api_key)
        prompt = f"""
        Context: A traveler from {user_origin} is seeing "{food_name}" for the first time.
        Task: Explain this dish to them.
        1. Compare the taste/texture to foods common in {user_origin}.
        2. Briefly explain its history/origin.
        3. Give one tip on how to eat it properly.
        Tone: Enthusiastic, cultural, and helpful. Keep it under 150 words.
        """
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"LLM Error: {str(e)}"

# UI interface
st.title("🍴 PlateAI: Cultural Food Guide")
st.markdown("Upload a photo of a dish, and I'll tell you what it is and its story!")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Groq API Key", type="password")
    user_home = st.text_input("Where are you from?", "Canada")
    model_file = "/content/drive/MyDrive/food_best.pth" # content/drive/MyDrive/food_best.pth

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Dish', use_container_width=True)
    
    if st.button("Analyze This Dish"):
        if not api_key:
            st.error("Please enter Groq API Key in the sidebar!")
        else:
            with st.spinner("Classifying food..."):
            
                # print("Loading model and classifying image...")
                model, device = load_my_model(model_file)
                input_tensor = preprocess_image(img).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, pred = torch.max(outputs, 1)
                    food_name = FOOD_CLASSES[pred.item()]
                
                st.success(f"I think this is: **{food_name.replace('_', ' ').title()}**")
            
            with st.spinner("Generating cultural story..."):
                #LLM story generation
                story = ask_llama_chef(food_name, user_home, api_key)
                st.info(story)