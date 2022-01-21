import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import finetuner
import torchvision
from docarray import DocumentArray
import torch
import os


MAX_DOCS = 1000

model = torchvision.models.resnet50(pretrained=True)

# Continue training from last time if possible
if os.path.isfile("./tuned-model"):
    print("Loading tuned model")
    model.load_state_dict(torch.load("./tuned-model"))
else:
    print("Loading default model")


print("Loading docs")
docs = DocumentArray.from_files("./data/images/*.jpg", size=MAX_DOCS, to_dataturi=True)

for doc in docs:
    doc.load_uri_to_image_blob(
        height=80, width=60
    ).set_image_blob_normalization().set_image_blob_channel_axis(-1, 0)


finetuner.fit(
    model,
    train_data=docs,
    interactive=True,
    to_embedding_model=True,
    freeze=False,
    input_size=(3, 80, 60),
)
