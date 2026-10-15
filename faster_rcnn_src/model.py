import torch

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


CLASS_NAMES = [
    "Pothole",
    "Crack",
    "Manhole"
]

NUM_CLASSES = len(CLASS_NAMES) + 1


def create_model():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    model = fasterrcnn_resnet50_fpn(
        weights=weights
    )

    in_features = (
        model.roi_heads.box_predictor
        .cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    return model


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


if __name__ == "__main__":
    device = get_device()

    print("=" * 50)
    print("MODEL TEST")
    print("=" * 50)

    print("Using device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    model = create_model()

    model = model.to(device)

    print("Number of object classes:", len(CLASS_NAMES))
    print("Number of model classes:", NUM_CLASSES)
    print("Model created successfully.")