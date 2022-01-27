#!/usr/bin/env python
# coding: utf-8

# # Finetuning our model
# 
# In the previous notebook we [built a simple fashion search engine using Docarray](https://colab.research.google.com/github/alexcg1/neural-search-notebooks/blob/main/fashion-search/1_build_basic_search/basic_search.ipynb)
# 
# Now we'll finetune our model to deliver better results!
# 
# ## Important note
# 
# This code won't run well in a notebook, since we need to run Finetuner on our local machine. Please:
# 
# - Download this notebook as a Python file to your local machine (*File* > *Download* > *Download .py*)
# - Install finetuner in a virtual environment (`pip install finetuner`)
# - Run this script from that directory
# 
# If you don't follow the above instructions the script will fail since it can't run the Finetuner GUI from within a notebook.

# ## Setup

# In[ ]:


get_ipython().system('pip install torchvision~=0.11')


# In[ ]:


get_ipython().system('pip install git+https://github.com/jina-ai/finetuner # Change to stable release later')


# In[ ]:


from docarray import Document, DocumentArray


# ## Load images
# 
# This is just the same process we followed in the last notebook

# In[ ]:


DATA_DIR = "./data"
DATA_PATH = f"{DATA_DIR}/images/*.jpg"
MAX_DOCS = 1000

# Toy data - If data dir doesn't exist, we'll get data of ~800 fashion images from here
TOY_DATA_URL = "https://github.com/alexcg1/neural-search-notebooks/blob/main/docarray/fashion-search/data.zip?raw=true"


# In[ ]:


# Download images if they don't exist
import os

if not os.path.isdir(DATA_DIR) and not os.path.islink(DATA_DIR):
    print(f"Can't find {DATA_DIR}. Downloading toy dataset")
    get_ipython().system('wget "$TOY_DATA_URL" -O data.zip')
    get_ipython().system("unzip -q data.zip # Don't print out every darn filename")
    get_ipython().system('rm -f data.zip')
else:
    print(f"Nothing to download. Using {DATA_DIR} for data")


# In[ ]:


docs = DocumentArray.from_files(DATA_PATH, size=MAX_DOCS)
print(f"{len(docs)} Documents in DocumentArray")


# In[ ]:


def preproc(doc):
    return (
        doc.load_uri_to_image_tensor(80, 60)
        .set_image_tensor_normalization()
        .set_image_tensor_channel_axis(-1, 0)
    )  # No need for changing channel axes line if you are using tf/keras


docs.apply(preproc)


# ## Load model
# 
# Again, we're playing the same old song, loading a model just like we did last time.

# In[ ]:


import torchvision

# model = torchvision.models.resnet50(pretrained=True)
model = torchvision.models.resnet18(pretrained=True)


# ## Finetune model
# 
# Here's where the new stuff kicks in!
# 
# We'll:
# 
# - Set some basic parameters
# - Install a module to see progress
# - Finetune our model, focusing on the embedding layer *just* before the classification layer

# In[ ]:


# Basic setup
EPOCHS = 1         # higher = more time, better finetuning
BATCH_SIZE = 10    # higher = use more memory
DEVICE = "cpu"     # if gpu, use "cuda", else "cpu"
LAYER_NAME = "adaptiveavgpool2d_67" # This will vary based on the model you use


# In[ ]:


# See progress bar in notebook
get_ipython().system('pip install ipywidgets')
import ipywidgets


# In[ ]:


import finetuner as ft

tuned_model = ft.fit(
    model=model,
    train_data=docs,
    loss='TripletLoss',
    epochs=EPOCHS,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    to_embedding_model=True,
    input_size=(3, 80, 60),
    layer_name=LAYER_NAME, # layer before fc as feature extractor
    freeze=False,
)


# ## Save model

# In[ ]:


import torch

torch.save(model, "tuned-model")

