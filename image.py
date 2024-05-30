import streamlit as st
from fer import FER
import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot(emotions) -> None:
    """Plots the emotion percentages using box plots.
    Args: emotions (list of dicts)
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.ion()
    fruits = list(emotions.keys())
    counts = list(emotions.values())
    bar_labels = list(emotions.keys())
    bar_colors = ['tab:red', 'tab:brown', 'tab:blue',
                  'tab:orange', 'tab:olive', 'tab:purple', 'tab:green']

    ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

    ax.set_ylabel('percentage')
    ax.set_title('emotions by percentage')
    ax.legend(title='Colours')

    with columns[1]:
        st.pyplot(fig)


def detect(image) -> None:
    """Detects emotions in a given image with facial expression. The emotion
    values range from 0 to 1.
    Args: image (Streamlit's UploadedFile): the uploaded image
    """
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    detector = FER()
    detector.detect_emotions(img)
    if detector.emotions:
        plot(detector.emotions[0].get('emotions'))
    else:
        with columns[1]:
            st.error(
                "Sorry, error.. but it`s just one debug away from shining again",
                icon="🚨")


def render_image(image):
    """Renders a given image on the UI.
    Args: image (Streamlit's UploadedFile): the uploaded image
    """
    with columns[0]:
        st.image(image, caption="my flatteringly expressive face")


def run():
    """Uploads a image selected by the user. Calls the image renderer to 
    render the uploaded image. And then invokes the detector to detect
    the emotions in the image.
    """
    with columns[0]:
        image = st.file_uploader("Choose an emotional pic")

        if image is not None:
            render_image(image=image)
            detect(image=image)


if __name__ == "__main__":
    # page / UI set up
    st.set_page_config(page_title="RL - Lie to Me",
                       page_icon="🦸",
                       layout="wide")
    st.title("🦸 Lie To Me")
    st.caption("<p style='color:red'> An Emotion Detector - Part One \
                powered by FER, OpenCV and Python <p>",
               unsafe_allow_html=True)
    columns = st.columns([0.4, 0.6])

    run()
