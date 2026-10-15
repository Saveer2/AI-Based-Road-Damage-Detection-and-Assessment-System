from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASS_NAMES = ["Pothole", "Crack", "Manhole"]
NUM_CLASSES = len(CLASS_NAMES) + 1

transform = transforms.ToTensor()

def normalize_name(filename):
    name = filename.stem
    if name.endswith("_256"):
        name = name[:-4]
    elif name.endswith("_600"):
        name = name[:-4]
    return name

def yolo_to_xyxy(center_x, center_y, box_width, box_height, image_width, image_height):
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
        
        # Points directly to processed_dataset/images/train and processed_dataset/labels/train
        self.images_dir = self.dataset_dir / "images" / split
        self.labels_dir = self.dataset_dir / "labels" / split

        self.image_files = sorted(
            [
                file for file in self.images_dir.iterdir()
                if file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
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
        image_width, image_height = image.width, image.height

        image_name = normalize_name(image_path)
        label_path = self.label_files.get(image_name)

        boxes = []
        labels = []

        if label_path and label_path.exists():
            with open(label_path, "r") as file:
                for line in file:
                    values = line.strip().split()
                    if not values:
                        continue
                    class_id = int(values[0])
                    center_x, center_y = float(values[1]), float(values[2])
                    box_width, box_height = float(values[3]), float(values[4])

                    box = yolo_to_xyxy(center_x, center_y, box_width, box_height, image_width, image_height)
                    boxes.append(box)
                    labels.append(class_id + 1)

        image = transform(image)
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))