from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD

# Updated import to use RoadDamageDataset
from road_damage_dataset import RoadDamageDataset, collate_fn
from model import create_model, get_device


NUM_EPOCHS = 25
BATCH_SIZE = 1
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for batch_index, (images, targets) in enumerate(dataloader):
        images = [
            image.to(device)
            for image in images
        ]

        targets = [
            {
                "boxes": target["boxes"].to(device),
                "labels": target["labels"].to(device)
            }
            for target in targets
        ]

        optimizer.zero_grad()

        loss_dict = model(images, targets)

        losses = sum(
            loss for loss in loss_dict.values()
        )

        losses.backward()
        optimizer.step() 

        loss_value = losses.item()
        total_loss += loss_value

        print(
            f"Epoch {epoch} | "
            f"Batch {batch_index + 1}/{len(dataloader)} | "
            f"Loss: {loss_value:.4f}"
        )

    average_loss = total_loss / len(dataloader)
    return average_loss


def validate_model(model, dataloader, device):
    model.train()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = [
                image.to(device)
                for image in images
            ]

            targets = [
                {
                    "boxes": target["boxes"].to(device),
                    "labels": target["labels"].to(device)
                }
                for target in targets
            ]

            loss_dict = model(images, targets)

            losses = sum(
                loss for loss in loss_dict.values()
            )

            total_loss += losses.item()

    average_loss = total_loss / len(dataloader)
    return average_loss


if __name__ == "__main__":
    print("=" * 60)
    print("ROAD DAMAGE DETECTION - TRAINING")
    print("=" * 60)

    device = get_device()
    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )
        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )

    # Project paths updated for road damage dataset and rcnn1 folder
    project_dir = Path(__file__).resolve().parent.parent

    dataset_dir = project_dir / "processed_dataset"

    output_dir = project_dir / "rcnn1"
    output_dir.mkdir(
        exist_ok=True
    )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(
        exist_ok=True
    )

    history_file = output_dir / "training_history.csv"

    print()
    print("Dataset:", dataset_dir)

    train_dataset = RoadDamageDataset(
        dataset_dir,
        split="train"
    )

    val_dataset = RoadDamageDataset(
        dataset_dir,
        split="val"
    )

    print(
        "Training images:",
        len(train_dataset)
    )

    print(
        "Validation images:",
        len(val_dataset)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    print()
    print("Creating model...")

    model = create_model()
    model = model.to(device)

    print("Model ready.")

    params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    print()
    print("Optimizer: SGD")
    print("Learning rate:", LEARNING_RATE)
    print("Momentum:", MOMENTUM)
    print("Weight decay:", WEIGHT_DECAY)
    print("Epochs:", NUM_EPOCHS)

    with open(
        history_file,
        "w",
        newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss"
            ]
        )

    best_val_loss = float("inf")

    print()
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        print()
        print("-" * 60)
        print(
            f"EPOCH {epoch}/{NUM_EPOCHS}"
        )
        print("-" * 60)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch
        )

        print()
        print(
            f"Training loss: {train_loss:.4f}"
        )

        print()
        print("Running validation...")

        val_loss = validate_model(
            model,
            val_loader,
            device
        )

        print(
            f"Validation loss: {val_loss:.4f}"
        )

        with open(
            history_file,
            "a",
            newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    val_loss
                ]
            )

        checkpoint_path = (
            checkpoint_dir
            / f"checkpoint_epoch_{epoch}.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss
            },
            checkpoint_path
        )

        print(
            "Checkpoint saved:",
            checkpoint_path
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Renamed to best.pt inside rcnn1 folder
            best_model_path = (
                output_dir / "best.pt"
            )

            torch.save(
                model.state_dict(),
                best_model_path
            )

            print(
                "Best model updated:",
                best_model_path
            )

        if torch.cuda.is_available():
            allocated = (
                torch.cuda.memory_allocated(0)
                / (1024 ** 3)
            )
            reserved = (
                torch.cuda.memory_reserved(0)
                / (1024 ** 3)
            )

            print(
                "GPU memory allocated:",
                round(allocated, 2),
                "GB"
            )
            print(
                "GPU memory reserved:",
                round(reserved, 2),
                "GB"
            )

    final_model_path = (
        output_dir / "final_model.pth"
    )

    torch.save(
        model.state_dict(),
        final_model_path
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        "Best validation loss:",
        round(best_val_loss, 4)
    )

    print(
        "Final model:",
        final_model_path
    )

    print(
        "Training history:",
        history_file
    )