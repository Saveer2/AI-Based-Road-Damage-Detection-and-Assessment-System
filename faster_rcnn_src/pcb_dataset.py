from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# TODO: Update these class names to match your road damage dataset classes
CLASS_NAMES = [
    "pothole",
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack"
]

NUM_CLASSES = len(CLASS_NAMES) + 1

transform = transforms.ToTensor()


def normalize_name(filename):
    name = filename.stem
    if name.endswith("_256"):
        name = name[:-4]
    elif name.endswith("_600"):
        name = name[:-4]
    return name


def yolo_to_xyxy(
    center_x,
    center_y,
    box_width,
    box_height,
    image_width,
    image_height
):
    center_x = center_x * image_width
    center_y = center_y * image_height

    box_width = box_width * image_width
    box_height = box_height * image_height

    x1 = center_x - box_width / 2
    y1 = center_y - box_height / 2
    x2 = center_x + box_width / 2
    y2 = center_y + box_height / 2

    return [x1, y1, x2, y2]


class RoadDamageDataset(Dataset):

    def __init__(self, dataset_dir, split="train"):
        self.dataset_dir = Path(dataset_dir)
        self.split = split

        self.images_dir = self.dataset_dir / split / "images"
        self.labels_dir = self.dataset_dir / split / "labels"

        self.image_files = sorted(
            [
                file
                for file in self.images_dir.iterdir()
                if file.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp"
                }
            ]
        )

        self.label_files = {
            normalize_name(file): file
            for file in self.labels_dir.iterdir()
            if file.suffix.lower() == ".txt"
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = Image.open(image_path).convert("RGB")

        image_width = image.width
        image_height = image.height

        image_name = normalize_name(image_path)
        label_path = self.label_files[image_name]

        boxes = []
        labels = []

        with open(label_path, "r") as file:
            for line in file:
                values = line.strip().split()
                if not values:
                    continue

                class_id = int(values[0])
                center_x = float(values[1])
                center_y = float(values[2])
                box_width = float(values[3])
                box_height = float(values[4])

                box = yolo_to_xyxy(
                    center_x,
                    center_y,
                    box_width,
                    box_height,
                    image_width,
                    image_height
                )

                boxes.append(box)
                labels.append(class_id + 1)

        image = transform(image)

        boxes = torch.tensor(
            boxes,
            dtype=torch.float32
        )

        labels = torch.tensor(
            labels,
            dtype=torch.int64
        )

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    # Updated path to point to processed_dataset
    dataset_dir = (
        Path(__file__).resolve().parent.parent
        / "processed_dataset"
    )

    dataset = RoadDamageDataset(
        dataset_dir,
        split="train"
    )

    print("=" * 50)
    print("DATASET")
    print("=" * 50)

    print("Number of images:", len(dataset))

    image, target = dataset[0]

    print()
    print("SINGLE SAMPLE")
    print("-" * 50)

    print("Image type:", type(image))
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)

    print()
    print("Boxes:")
    print(target["boxes"])

    print("Boxes dtype:")
    print(target["boxes"].dtype)

    print()
    print("Labels:")
    print(target["labels"])

    print("Labels dtype:")
    print(target["labels"].dtype)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    print()
    print("=" * 50)
    print("DATALOADER")
    print("=" * 50)

    print("Batch size:", 4)

    images, targets = next(iter(dataloader))

    print()
    print("BATCH TEST")
    print("-" * 50)

    print("Number of images in batch:", len(images))
    print("Number of targets in batch:", len(targets))

    for i in range(len(images)):
        print()
        print("Image", i + 1)
        print("Shape:", images[i].shape)

        print("Boxes:")
        print(targets[i]["boxes"])

        print("Labels:")
        print(targets[i]["labels"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("=" * 50)
    print("DEVICE")
    print("=" * 50)

    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print()
    print("=" * 50)
    print("MODEL")
    print("=" * 50)

    print("Number of object classes:", len(CLASS_NAMES))
    print("Number of model classes:", NUM_CLASSES)

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    model = fasterrcnn_resnet50_fpn(
        weights=weights
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    model = model.to(device)

    print("Model created successfully.")
    print("Model device:", device)

    model.eval()

    images_for_model = [
        image.to(device)
        for image in images
    ]

    print()
    print("=" * 50)
    print("MODEL INFERENCE TEST")
    print("=" * 50)

    with torch.no_grad():
        predictions = model(images_for_model)

    print("Number of predictions:", len(predictions))

    for i, prediction in enumerate(predictions):
        print()
        print("Image", i + 1)

        print("Predicted boxes shape:")
        print(prediction["boxes"].shape)

        print("Predicted labels shape:")
        print(prediction["labels"].shape)

        print("Predicted scores shape:")
        print(prediction["scores"].shape)

    print()
    print("=" * 50)
    print("ALL TESTS COMPLETED")
    print("=" * 50)