import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# ===========================
# One-Hot Vector Encoding Functions
# ===========================
def one_hot_vector(a):
    '''Create a one-hot vector (OHV) for a given hexadecimal digit.'''
    ret = np.zeros(16)
    hex_map = "0123456789abcdef"
    if a in hex_map:
        ret[hex_map.index(a)] = 1
    return ret

def create_CAN_image(s):
    '''Create the CAN image (3x16 matrix) for a given CAN ID.'''
    # Chop off the first character and use the remaining 3 characters
    s = s[1:]  # Remove the first character

    # Now we should have 3 hexadecimal digits (after removing the first character)
    a = one_hot_vector(s[0])  # One-hot vector for the first digit
    b = one_hot_vector(s[1])  # One-hot vector for the second digit
    c = one_hot_vector(s[2])  # One-hot vector for the third digit
    
    return np.array([a, b, c])  # 3x16 matrix (48 values)

def create_64batch_Discriminator(can_images):
    '''Create a 64x48 batch for discriminator training (flattened 3x16 images).'''
    return np.vstack([img.reshape(1, 48) for img in can_images])  # Flatten to 48 values per image

# ===========================
# Load Data
# ===========================
print("GAN Intrusion Detection System for CAN Bus case study. Started.")
ids_normal_list = []
ids_intrusion_list = []

# Load NORMAL IDS (Real Data)
with open("NORMAL_IDS.txt", "r") as f:
    ids_normal_list = [line.strip() for line in f.readlines() if line.strip()]

# Load DOS IDS (Intrusion Data)
with open("DOS_IDS.txt", "r") as f:
    ids_intrusion_list = [line.strip() for line in f.readlines() if line.strip()]

# Convert to CAN Images
normal_CAN_images = [create_CAN_image(can_id) for can_id in ids_normal_list]
intrusion_CAN_images = [create_CAN_image(can_id) for can_id in ids_intrusion_list]

# Prepare batches (ensure they are 64-multiples)
normal_batches = [torch.tensor(create_64batch_Discriminator(normal_CAN_images[i:i+64]), dtype=torch.float32)
                  for i in range(0, len(normal_CAN_images) - (len(normal_CAN_images) % 64), 64)]
intrusion_batches = [torch.tensor(create_64batch_Discriminator(intrusion_CAN_images[i:i+64]), dtype=torch.float32)
                     for i in range(0, len(intrusion_CAN_images) - (len(intrusion_CAN_images) % 64), 64)]

# ===========================
# Generator Architecture (Based on Paper)
# ===========================
class Generator(nn.Module):
    def __init__(self, input_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),  # First fully connected layer
            nn.ReLU(),
            nn.Linear(256, 512),        # Second fully connected layer
            nn.ReLU(),
            nn.Linear(512, 48),         # Output layer (3x16 flattened to 48)
            nn.Sigmoid()                 # Sigmoid activation (range [0, 1] for binary output)
        )
    
    def forward(self, z):
        # Apply sigmoid and round to ensure binary output
        output = self.model(z).reshape(-1, 3, 16)  # Reshape into 3x16 (CAN image)
        return torch.round(output)  # Ensure output is binary (0 or 1)

# ===========================
# Discriminator Architecture (Based on Paper)
# ===========================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),                   # Flatten input (48-dimensional vector)
            nn.Linear(48, 512),             # First fully connected layer (48 input)
            nn.ReLU(),
            nn.Linear(512, 256),            # Second fully connected layer
            nn.ReLU(),
            nn.Linear(256, 1),              # Output layer (binary classification)
            nn.Sigmoid()                    # Sigmoid for output probability
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize models
generator = Generator(input_dim=100)
discriminator = Discriminator()

# Optimizers and loss function
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
criterion = nn.BCELoss()

# ===========================
# Train the GAN
# ===========================
def old_train_gan(generator, discriminator, normal_batches, epochs=10000, batch_size=64):
    '''Train the GAN using normal CAN data.'''
    d_losses = []
    g_losses = []
    fake_data = None  # Initialize fake_data for visualization later

    for epoch in range(epochs):
        # Select a random batch
        idx = np.random.randint(0, len(normal_batches))
        real_data = normal_batches[idx]

        # Generate fake CAN images
        noise = torch.randn(batch_size, 100)  # Random noise
        fake_data = generator(noise)

        # Labels for real and fake data
        real_labels = torch.ones((batch_size, 1))
        fake_labels = torch.zeros((batch_size, 1))

        # Train the discriminator
        discriminator.train()
        d_optimizer.zero_grad()
        real_loss = criterion(discriminator(real_data), real_labels)
        fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Train the generator
        generator.train()
        g_optimizer.zero_grad()
        g_loss = criterion(discriminator(fake_data), real_labels)
        g_loss.backward()
        g_optimizer.step()

        # Store loss values
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        # Print progress
        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    return d_losses, g_losses, fake_data


def train_gan(generator, discriminator, normal_batches, epochs=10000, batch_size=64):
    '''Train the GAN using normal CAN data.'''
    d_losses = []
    g_losses = []
    fake_data = None  # Initialize fake_data for visualization later

    for epoch in range(epochs):
        # Select a random batch
        idx = np.random.randint(0, len(normal_batches))
        real_data = normal_batches[idx]

        # Generate fake CAN images
        noise = torch.randn(batch_size, 100)  # Random noise
        fake_data = generator(noise)

        # Labels for real and fake data with label smoothing
        real_labels = torch.full((batch_size, 1), 0.9)  # Smoothed real labels
        fake_labels = torch.full((batch_size, 1), 0.1)  # Smoothed fake labels

        # Train the discriminator
        discriminator.train()
        d_optimizer.zero_grad()  # Zero the discriminator's gradients
        real_loss = criterion(discriminator(real_data), real_labels)  # loss for real data
        fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)  # loss for fake data
        d_loss = real_loss + fake_loss  # Total loss for the Discriminator
        d_loss.backward()  # Backpropagation for the Discriminator
        d_optimizer.step()  # Update Discriminator's weights

        # Train the generator
        generator.train()
        g_optimizer.zero_grad()  # Zero the generator's gradients
        g_loss = criterion(discriminator(fake_data), real_labels)  # Fake data should be classified as real
        g_loss.backward()  # Backpropagation for the Generator
        g_optimizer.step()  # Update Generator's weights

        # Store loss values for later plotting
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        # Print progress every 1000 epochs
        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    return d_losses, g_losses, fake_data

# Train the GAN
d_losses, g_losses, fake_data = train_gan(generator, discriminator, normal_batches)

# ===========================
# Evaluate Model (Intrusion and Normal Classification)
# ===========================
def evaluate_model_with_percentages(discriminator, normal_batches, intrusion_batches):
    '''Evaluate the GAN's ability to classify real vs. fake CAN IDs and detect intrusions.'''

    # Track correct classifications
    intrusion_true_positive = 0
    normal_true_negative = 0
    intrusion_total = 0
    normal_total = 0

    # Collect true labels and predictions for ROC Curve
    true_labels = []
    predictions = []

    # Evaluate on normal data (real)
    discriminator.eval()
    for batch in normal_batches:
        real_labels = torch.ones(batch.size(0), 1)  # Normal data is real
        output = discriminator(batch)
        predictions.append(output.detach().numpy())
        true_labels.append(real_labels.detach().numpy())

        # Count True Negatives (TN)
        normal_total += batch.size(0)
        normal_true_negative += np.sum(output.detach().numpy() < 0.5)  # Classify as normal if score < 0.5

    # Evaluate on intrusion data (fake)
    for batch in intrusion_batches:
        fake_labels = torch.zeros(batch.size(0), 1)  # Intrusion data is fake
        output = discriminator(batch)
        predictions.append(output.detach().numpy())
        true_labels.append(fake_labels.detach().numpy())

        # Count True Positives (TP)
        intrusion_total += batch.size(0)
        intrusion_true_positive += np.sum(output.detach().numpy() >= 0.5)  # Classify as intrusion if score >= 0.5

    # Flatten lists
    true_labels = np.concatenate(true_labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    # Calculate percentage of correctly classified intrusion and normal data
    intrusion_classification_rate = (intrusion_true_positive / intrusion_total) * 100
    normal_classification_rate = (normal_true_negative / normal_total) * 100

    print(f"Intrusion Classification Rate: {intrusion_classification_rate:.2f}%")
    print(f"Normal Classification Rate: {normal_classification_rate:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(true_labels, (predictions > 0.5))
    print(f"Confusion Matrix:\n{cm}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Example of evaluating the model
evaluate_model_with_percentages(discriminator, normal_batches, intrusion_batches)

