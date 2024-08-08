import streamlit as st
import os
from streamlit import session_state as ss
import torch
import torchvision as tv
from torchvision import transforms as tr
from random import randint
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from models import Autoencoder, VariationalAutoencoder

use_variational = True
model_name = "variationalae" if use_variational else "autoencoder"

if "autoencoder" not in ss:
    dirname = os.path.dirname(os.path.realpath(__file__))+"/.."
    net_state_dict = torch.load(dirname+f"/models/{model_name}")
    ss.autoencoder = VariationalAutoencoder(2, "cpu") if use_variational else Autoencoder(2)
    ss.autoencoder.load_state_dict(net_state_dict)
if "fashion_data" not in ss:
    dirname = os.path.dirname(os.path.realpath(__file__))+"/.."
    img_transf = tr.Compose([tr.Resize((32,32)),
                             tr.ToTensor()])
    ss.fashion_data = tv.datasets.FashionMNIST(dirname+"/datasets",
                                                train = False,
                                                transform=img_transf,
                                                download=True)
    train_data = tv.datasets.FashionMNIST(dirname+"/datasets",
                                            train = True,
                                            transform=img_transf,
                                            download=True)
    ss.embeds_x = []
    ss.embeds_y = []
    ss.embeds_z = []
    for sample in train_data:
        embeddings = ss.autoencoder.encoder(sample[0].unsqueeze(1)).detach().tolist()[0]
        ss.embeds_x.append(embeddings[0])
        ss.embeds_y.append(embeddings[1])
        ss.embeds_z.append(sample[1])
    x_range = max(ss.embeds_x) - min(ss.embeds_x)
    y_range = max(ss.embeds_y) - min(ss.embeds_y)
    offlims = 0.25
    ss.x_lims = [min(ss.embeds_x) - offlims*x_range, max(ss.embeds_x) + offlims*x_range]
    ss.y_lims = [min(ss.embeds_y) - offlims*y_range, max(ss.embeds_y) + offlims*y_range]

submenu = st.selectbox("Submenu", ("Sample dataset", "Generate"))

if submenu == "Sample dataset":
    if st.button("Run"):
        sample = randint(0, len(ss.fashion_data))
        x = ss.fashion_data[sample][0]
        middle = ss.autoencoder.encoder(x.unsqueeze(1))
        y = ss.autoencoder.decoder(middle)
        st.write(f"Embeddings: {middle.detach().tolist()[0]}")
        cleft, cright = st.columns(2)
        with cleft:
            img = Image.fromarray(np.array(x[0,:,:]*256, dtype=np.uint8))
            cleft.image(img, "Input image", use_column_width=True)
        with cright:
            img = Image.fromarray(np.array(y.detach()[0,0,:,:]*256, dtype=np.uint8))
            cright.image(img, "Model output", use_column_width=True)

elif submenu == "Generate":
    cols = st.columns(2)
    x = [0,0]
    with cols[0]:
        x[0] = st.slider("x value", value=sum(ss.x_lims)/2, min_value=ss.x_lims[0], max_value=ss.x_lims[1])
        x[1] = st.slider("y value", value=sum(ss.y_lims)/2, min_value=ss.y_lims[0], max_value=ss.y_lims[1])
    with cols[1]:
        embeds = ss.autoencoder.decoder(torch.tensor([x]))
        img = Image.fromarray(np.array(embeds.detach()[0,0,:,:]*256, dtype=np.uint8))
        st.image(img, "Model output", use_column_width=True)
    plt.title("Embeddings space")
    labels = list(ss.fashion_data.class_to_idx.keys())
    plt.scatter(ss.embeds_x, ss.embeds_y, c=ss.embeds_z, s=0.5, vmin = 0, vmax = len(labels))
    cb = plt.colorbar()
    plt.scatter([x[0]], [x[1]], s=10, c="r")
    cb.set_ticks(ticks=list(range(len(labels))), labels=labels)
    st.pyplot(plt)