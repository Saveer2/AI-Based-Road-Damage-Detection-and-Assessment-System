from pathlib import Path

# Updated path to point to your processed_dataset directory
dataset_dir = Path(__file__).resolve().parent.parent / "processed_dataset"

splits = ["train", "val", "test"]

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

def normalize_name(filename):
    """
    Removes the final resolution suffix
    such as _256 or _600 if present.
    """
    name = filename.stem
    if name.endswith("_256"):
        name = name[:-4]
    elif name.endswith("_600"):
        name = name[:-4]
    return name


for split in splits:
    # Updated path structure to match images/split and labels/split
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    if not images_dir.exists() or not labels_dir.exists():
        print(f"\nSkipping {split.upper()} - directory not found.")
        continue

    image_stems = {
        normalize_name(file)
        for file in images_dir.iterdir()
        if file.suffix.lower() in image_extensions
    }

    label_stems = {
        normalize_name(file)
        for file in labels_dir.iterdir()
        if file.suffix.lower() == ".txt"
    }

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    print(f"\n{split.upper()} MATCHING")
    print("-" * 30)
    print("Images without labels:", len(missing_labels))
    print("Labels without images:", len(missing_images))

    if missing_labels:
        print("Example missing labels:", list(missing_labels)[:5])

    if missing_images:
        print("Example missing images:", list(missing_images)[:5])


# Updated class names for your road damage dataset
class_names = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole"
}

class_counts = {
    class_id: 0 
    for class_id in class_names
}

for split1 in splits:
    labels_dir = dataset_dir / "labels" / split1
    if not labels_dir.exists():
        continue
        
    for label_file in labels_dir.iterdir():
        if label_file.suffix.lower() != ".txt":
            continue

        with open(label_file, "r") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                values = line.split()
                class_id = int(values[0])
                if class_id in class_counts:
                    class_counts[class_id] += 1

print("\nCLASS DISTRIBUTION")
print("-" * 30)

for class_id, count in class_counts.items():
    print(
        f"{class_id}: "
        f"{class_names[class_id]:20s} "
        f"{count}"
    )