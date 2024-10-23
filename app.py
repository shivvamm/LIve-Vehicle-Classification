import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from collections import Counter
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
import requests
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time


st.set_page_config(
    page_title="Object Detector Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.sidebar.title("Traffic Monitoring AI")


@st.cache_resource
def fetch_images(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    driver.get(url)

    image_elements = driver.find_elements(By.CSS_SELECTOR, ".card img")
    image_urls = [img_elem.get_attribute('src') for img_elem in image_elements]

    driver.quit()
    return image_urls


weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model(threshold):
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=threshold)
    model.eval()
    return model

def make_prediction(img, model):
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(
        img_tensor,
        boxes=prediction["boxes"],
        labels=prediction["labels"],
        colors=["Green" if label == "person" else "red" for label in prediction["labels"]],
        width=1
    )
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_with_bboxes_np


threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.01)


camera_urls = {
    "Woodlands": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/woodlands.html#trafficCameras",
    "SLE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/sle.html#trafficCameras",
    "TPE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/tpe.html#trafficCameras",
    "LTM": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/ltm.html#trafficCameras",
    "KJE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/kje.html#trafficCameras",
    "BKE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/bke.html#trafficCameras",
    "CTE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/cte.html#trafficCameras",
    "KPE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/kpe.html#trafficCameras",
    "PIE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/pie.html#trafficCameras",
    "AYE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/aye.html#trafficCameras",
    "MCE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/mce.html#trafficCameras",
    "ECP": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/ecp.html#trafficCameras",
    "STG": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/stg.html#trafficCameras"
}


selected_area = st.sidebar.selectbox("Select Traffic Area", list(camera_urls.keys()))


st.title(f"Object Detection for {selected_area}")
st.markdown(f"Fetching images from {selected_area} camera...")
st.markdown("**Note:** The images will update every 1 minute.")


if 'area_data' not in st.session_state:
    st.session_state.area_data = {}


if selected_area not in st.session_state.area_data:
    st.session_state.area_data[selected_area] = {
        'predictions': [],
        'table_data': [],
        'last_update': 0
    }


current_time = time.time()


if st.button("Refresh Images"):
    st.session_state.area_data[selected_area]['last_update'] = 0 


if current_time - st.session_state.area_data[selected_area]['last_update'] > 60:  
    image_urls = fetch_images(camera_urls[selected_area])
    st.session_state.area_data[selected_area]['last_update'] = current_time  


    all_predictions = []
    if image_urls:

        num_images = len(image_urls)
        cols = st.columns(min(num_images, 4))  

    with st.spinner("Processing images, please wait..."):
        for i, img_url in enumerate(image_urls):
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))

            model = load_model(threshold)
            prediction = make_prediction(img, model)
            img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2, 0, 1), prediction)

            
            with cols[i % 4]:  
                st.header(f"Image {i + 1}: Object Detection Results")
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111)
                plt.imshow(img_with_bbox)
                plt.xticks([], [])
                plt.yticks([], [])
                ax.spines[["top", "bottom", "right", "left"]].set_visible(True)
                st.pyplot(fig, use_container_width=True)

            
            for label in prediction["labels"]:
                all_predictions.append({"Image": f"Image {i + 1}", "Label": label})

    
    st.session_state.area_data[selected_area]['predictions'] = all_predictions

    
    if all_predictions:
        
        image_object_counts = []
        for i in range(1, len(image_urls) + 1):
            
            current_image_preds = [pred['Label'] for pred in all_predictions if pred['Image'] == f"Image {i}"]
            
            object_count = Counter(current_image_preds)
            for label, count in object_count.items():
                image_object_counts.append({"Image": f"Image {i}", "Label": label, "Count": count})

        
        df_summary = pd.DataFrame(image_object_counts)

        
        vehicle_categories = ['car', 'bus', 'motorcycle', 'truck', 'train', 'bicycle', 'scooter']


        df_vehicles = df_summary[df_summary['Label'].isin(vehicle_categories)]

        
        st.session_state.area_data[selected_area]['table_data'] = df_vehicles


if not st.session_state.area_data[selected_area]['table_data'].empty:
    st.header("Combined Object Detection Table for All Images")


    image_filter = st.selectbox("Filter by Image", options=["All"] + [f"Image {i}" for i in range(1, len(image_urls) + 1)])

    
    if image_filter != "All":
        filtered_data = st.session_state.area_data[selected_area]['table_data'][st.session_state.area_data[selected_area]['table_data']['Image'] == image_filter]
    else:
        filtered_data = st.session_state.area_data[selected_area]['table_data']

    st.dataframe(filtered_data,use_container_width=True)