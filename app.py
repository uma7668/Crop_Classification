import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model("best_model.keras") 


class_names = ['Aji pepper plant',
 'Almonds plant',
 'Amaranth plant',
 'Apples plant',
 'Artichoke plant',
 'Avocados plant',
 'Bananas plant',
 'Barley plant',
 'Beets plant',
 'Black pepper plant',
 'Blueberries plant',
 'Bok choy plant',
 'Brazil nuts plant',
 'Broccoli plant',
 'Brussels sprout plant',
 'Buckwheat plant',
 'Cabbages and other brassicas plant',
 'Camucamu plant',
 'Carrots and turnips plant',
 'Cashew nuts plant',
 'Cassava plant',
 'Cauliflower plant',
 'Celery plant',
 'Cherimoya plant',
 'Cherry plant',
 'Chestnuts plant',
 'Chickpeas plant',
 'Chili peppers and green peppers plant',
 'Cinnamon plant',
 'Cloves plant',
 'Cocoa beans plant',
 'Coconuts plant',
 'Coffee (green) plant',
 'Collards plant',
 'Cotton lint plant',
 'Cranberries plant',
 'Cucumbers and gherkins plant',
 'Dates plant',
 'Dry beans plant',
 'Dry peas plant',
 'Durian plant',
 'Eggplants (Aubergines) plant',
 'Endive plant',
 'Fava bean plant',
 'Figs plant',
 'Flax fiber and tow plant',
 'Flaxseed (Linseed) plant',
 'Fonio plant',
 'Garlic plant',
 'Ginger plant',
 'Gooseberries plant',
 'Grapes plant',
 'Groundnuts (Peanuts) plant',
 'Guarana plant',
 'Guavas plant',
 'Habanero pepper plant',
 'Hazelnuts plant',
 'Hemp plant',
 'Hen eggs (shell weight) plant',
 'Horseradish plant',
 'Jackfruit plant',
 'Jute plant',
 'Kale plant',
 'Kohlrabi plant',
 'Leeks plant',
 'Lemons and limes plant',
 'Lentils plant',
 'Lettuce and chicory plant',
 'Lima bean plant',
 'Longan plant',
 'Lupins plant',
 'Lychee plant',
 'Maize (Corn) plant',
 'Mandarins, clementines, satsumas plant',
 'Mangoes, mangosteens, guavas plant',
 'Maracuja(Passionfruit) plant',
 'Millet plant',
 'Mint plant',
 'Mung bean plant',
 'Mustard greens plant',
 'Mustard seeds plant',
 'Navy bean plant',
 'Oats plant',
 'Oil palm fruit plant',
 'Okra plant',
 'Olives plant',
 'Onions (dry) plant',
 'Oranges plant',
 'Oregano plant',
 'Papayas plant',
 'Parsley plant',
 'Peaches and nectarines plant',
 'Peas (Green) plant',
 'Persimmons plant',
 'Pine nuts plant',
 'Pineapples plant',
 'Pinto bean plant',
 'Pistachios plant',
 'Plantains plant',
 'Pomegranates plant',
 'Potatoes plant',
 'Pumpkins, squash and gourds plant',
 'Quinoa plant',
 'Radishes and similar roots plant',
 'Rambutan plant',
 'Rapeseed (Canola) plant',
 'Raspberries plant',
 'Rice (Paddy) plant',
 'Rosemary plant',
 'Rubber (natural) plant',
 'Rye plant',
 'Saffron plant',
 'Sage plant',
 'Scallions plant',
 'Sorghum plant',
 'Soursop plant',
 'Soybeans plant',
 'Spinach plant',
 'Starfruit plant',
 'Strawberries plant',
 'Sugar beet plant',
 'Sugar cane plant',
 'Sunflower seeds plant',
 'Sweet potatoes plant',
 'Swiss chard plant',
 'Tamarind plant',
 'Taro (cocoyam) plant',
 'Tea plant',
 'Teff plant',
 'Thyme plant',
 'Tomatoes plant',
 'Triticale plant',
 'Turmeric plant',
 'Turnip greens plant',
 'Vanilla beans plant',
 'Walnuts plant',
 'Watermelons plant',
 'Wheat plant',
 'Yams plant']  


st.title("🌱 Crop Classification App")
st.write("Upload a crop image and I'll tell you what it is!")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    
if st.button("Predict"):
        
        img = image.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Show results
        st.success(f"Prediction: **{predicted_class}**")

        st.info(f"Confidence: **{confidence:.2f}%**")
