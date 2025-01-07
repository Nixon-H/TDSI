import torchaudio
torchaudio.set_audio_backend("ffmpeg")

import torch
from torch.optim import Adam
from pathlib import Path
from src.allModels.models import AudioSealDetector, AudioSealWM, MsgProcessor
from src.allModels.SEANet import SEANetDecoder, SEANetEncoderKeepDimension
from src.utils.data_prcocessing import get_dataloader
from src.losses.loss import compute_perceptual_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configuration
num_epochs = 50  # Initial number of epochs
batch_size = 1
audio_length = 8000
learning_rate = 1e-3
nbits = 32
latent_dim = 128
patience = 10  # Patience for early stopping
min_delta = 0.01  # Minimum improvement to reset early stopping counter

# Data paths
train_data_dir = Path("/content/TDSI/data/train").resolve()
validate_data_dir = Path("/content/TDSI/data/validate").resolve()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize SEANet encoder and decoder
encoder = SEANetEncoderKeepDimension(
    channels=1,
    dimension=latent_dim,
    n_filters=32,
    n_residual_layers=3,
    ratios=[8, 5, 4, 2],
    output_dim=latent_dim,
).to(device)

decoder = SEANetDecoder(
    channels=1,
    dimension=latent_dim,
    n_filters=32,
    n_residual_layers=3,
    ratios=[8, 5, 4, 2],
).to(device)

msg_processor = MsgProcessor(
    nbits=32,  # Number of bits for the watermark message
    hidden_size=latent_dim,
).to(device)

# Initialize generator and detector
generator = AudioSealWM(
    encoder=encoder,
    decoder=decoder,
    msg_processor=msg_processor,
).to(device)

detector = AudioSealDetector(
    channels=1,
    dimension=latent_dim,
    n_filters=32,
    n_residual_layers=3,
    ratios=[8, 5, 4, 2],
    output_dim=latent_dim,
    nbits=nbits,
).to(device)

# Optimizers and schedulers
optimizer_g = Adam(generator.parameters(), lr=learning_rate, weight_decay=1e-4)
optimizer_d = Adam(detector.parameters(), lr=learning_rate, weight_decay=1e-4)

# Schedulers
scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=5, verbose=True)
scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=5, verbose=True)

# DataLoaders
try:
    train_loader = get_dataloader(
        data_dir=train_data_dir,
        batch_size=batch_size,
        sample_rate=audio_length,
        shuffle=True,
        num_workers=0,
    )

    validate_loader = get_dataloader(
        data_dir=validate_data_dir,
        batch_size=batch_size,
        sample_rate=audio_length,
        shuffle=False,
        num_workers=0,
    )
except FileNotFoundError as e:
    print(f"Error initializing DataLoaders: {e}")
    exit(1)

# Dataset size check
if len(train_loader.dataset) == 0 or len(validate_loader.dataset) == 0:
    print("Error: Empty datasets.")
    exit(1)

print(f"Train dataset size: {len(train_loader.dataset)}")
print(f"Validation dataset size: {len(validate_loader.dataset)}")

# Training process
def train(
    generator,
    detector,
    train_loader,
    val_loader,
    optimizer_g,
    optimizer_d,
    scheduler_g,
    scheduler_d,
    num_epochs,
    patience,
    min_delta,
):
    best_val_loss = float('inf')
    early_stop_counter = 0
    dynamic_epochs = num_epochs

    for epoch in range(dynamic_epochs):
        # Training phase
        generator.train()
        detector.train()
        train_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs_g = generator(inputs)
            outputs_d = detector(inputs)

            # Compute loss
            loss_g = compute_perceptual_loss(outputs_g, labels)
            loss_d = compute_perceptual_loss(outputs_d, labels)
            total_loss = loss_g + loss_d

            # Backward pass
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            total_loss.backward()
            optimizer_g.step()
            optimizer_d.step()

            train_loss += total_loss.item()

        # Validation phase
        generator.eval()
        detector.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs_g = generator(inputs)
                outputs_d = detector(inputs)
                loss_g = compute_perceptual_loss(outputs_g, labels)
                loss_d = compute_perceptual_loss(outputs_d, labels)
                val_loss += loss_g.item() + loss_d.item()

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)

        print(f"Epoch {epoch + 1}/{dynamic_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Step the scheduler
        scheduler_g.step(val_loss)
        scheduler_d.step(val_loss)

        # Check for improvement
        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            early_stop_counter = 0  # Reset counter if improvement
            print("Validation loss improved. Saving checkpoint.")
            torch.save(generator.state_dict(), f"./checkpoints/generator_epoch_{epoch + 1}.pth")
            torch.save(detector.state_dict(), f"./checkpoints/detector_epoch_{epoch + 1}.pth")
        else:
            early_stop_counter += 1
            print(f"No improvement. Early stopping counter: {early_stop_counter}/{patience}")

        # Early stopping check
        if early_stop_counter >= patience:
            print("Early stopping triggered. Training terminated.")
            break

        # Dynamically extend training epochs if needed
        if epoch == dynamic_epochs - 1 and early_stop_counter == 0:
            dynamic_epochs += patience
            print(f"Extending training by {patience} epochs. New total epochs: {dynamic_epochs}")

    print("Training completed.")

# Run training
try:
    train(
        generator=generator,
        detector=detector,
        train_loader=train_loader,
        val_loader=validate_loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
        num_epochs=num_epochs,
        patience=patience,
        min_delta=min_delta,
    )
except Exception as e:
    print(f"Training error: {e}")
    exit(1)
