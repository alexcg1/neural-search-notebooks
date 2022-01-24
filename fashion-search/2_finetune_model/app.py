import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)  # less clutter on CLI
import finetuner
import torchvision
from docarray import DocumentArray, Document
import torch
import os


def preproc(d: Document):
    return (
        d.load_uri_to_image_tensor()  # load
        .set_image_tensor_shape(
            (80, 60)
        )  # ensure all images right size (dataset image size _should_ be (80, 60))
        .set_image_tensor_normalization()  # normalize color
        .set_image_tensor_channel_axis(-1, 0)
    )  # switch color axis for the PyTorch model later


MAX_DOCS = 1000
DATA_PATH = "./data/images/*.jpg"
MODEL_FILE_NAME = "tuned-model"  # filename for tuned model

model = torchvision.models.resnet50(pretrained=True)  # same model as last time

# Continue training from last time if possible
if os.path.isfile(MODEL_FILE_NAME):
    print("Loading tuned model")
    model.load_state_dict(torch.load(MODEL_FILE_NAME))
else:
    print("Loading default model")


print("Loading docs")
docs = DocumentArray.from_files(DATA_PATH, size=MAX_DOCS)
docs.apply(preproc)

# for doc in docs:
    # doc.load_uri_to_image_blob(
        # height=80, width=60
    # ).set_image_blob_normalization().set_image_blob_channel_axis(-1, 0)


finetuner.fit(
    model,
    train_data=docs,
    interactive=True,
    to_embedding_model=True,
    freeze=False,
    input_size=(3, 80, 60),
)
