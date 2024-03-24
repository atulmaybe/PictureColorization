# importing required libraries

import streamlit as st
import pandas as pd
import os
from io import StringIO
from PIL import Image
from PIL import Image

from main import *
# adding a file uploader
st.title('COLOR IT')
file = st.file_uploader("Please choose a file")


def save_uploaded_file(uploadedfile):
    with open(os.path.join("test", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved file :{} in tempDir".format(uploadedfile.name))


if file is not None:
    file_details = {"FileName": file.name, "FileType": file.type}
    save_uploaded_file(file)
    col1, col2 = st.columns(2)
    with col1:
        st.header("Gray Image")
        image = Image.open('test/' + str(file.name))
        st.image(image)
    with col2:
        getColorImage('test/'+str(file.name))
        st.header("Color Image")
        image = Image.open('result/result.png')
        st.image(image)
