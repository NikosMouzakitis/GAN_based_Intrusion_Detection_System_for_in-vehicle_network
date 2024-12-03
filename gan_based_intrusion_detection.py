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
print(normal_CAN_images[0])
plt.figure(1)
plt.imshow(create_64batch_Discriminator(normal_CAN_images[0:64]),cmap="binary")
plt.title("A discriminator's batch")
plt.show()


# Prepare batches (ensure they are 64-multiples)
normal_batches = [torch.tensor(create_64batch_Discriminator(normal_CAN_images[i:i+64]), dtype=torch.float32)
                  for i in range(0, len(normal_CAN_images) - (len(normal_CAN_images) % 64), 64)]
intrusion_batches = [torch.tensor(create_64batch_Discriminator(intrusion_CAN_images[i:i+64]), dtype=torch.float32)
                     for i in range(0, len(intrusion_CAN_images) - (len(intrusion_CAN_images) % 64), 64)]


# ===========================
# Generator Architecture (Based on Paper)
# ===========================
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            # Input: latent_dim (100) → Output: 256x4x3
            nn.Linear(latent_dim, 256 * 4 * 3),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 3)),  # Reshape to 4x3 spatial dimensions with 256 channels
            
            # Layer 1: Transposed Convolution (Upsampling) → Output: 128x8x6
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Layer 2: Transposed Convolution (Upsampling) → Output: 64x16x12
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Layer 3: Transposed Convolution (Upsampling) → Output: 32x32x24
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Layer 4: Transposed Convolution (Upsampling) → Output: 1x64x48 (Final CAN image)
            nn.ConvTranspose2d(32, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()  # Output values scaled to [-1, 1]
        )
        
        # Apply Xavier Normal Initialization to weights
        self._initialize_weights()

    def forward(self, z):
        """
        Forward pass to generate the image.
        Only at the final step, normalize the output to [0, 1] and apply threshold to get binary output.
        """
        output = self.model(z)
        # Apply normalization and thresholding only to the final output:
        output = (output + 1) / 2  # Normalize from [-1, 1] to [0, 1]
        output = torch.where(output >= 0.5, torch.tensor(1.0), torch.tensor(0.0))  # Threshold to get binary values
        return output

    def _initialize_weights(self):
        # Initialize weights using Xavier Normal initialization
        for m in self.model.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# ===========================
# Discriminator Architecture (Based on Paper)
# ===========================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # First Linear layer expects a 3072-dimensional vector (flattened image)
            nn.Linear(3072, 1536),  # Input size of 3072 (flattened from 64x48), output 1536
            nn.ReLU(),
            nn.Linear(1536, 768),   # Reduce to 768
            nn.ReLU(),
            nn.Linear(768, 1),      # Output layer (single value: real or fake)
            nn.Sigmoid()            # Sigmoid for binary classification (0 or 1)
        )
    
    def forward(self, x):
        # Flatten the input only for the first layer (64x48 -> 3072)
        x = x.view(x.size(0), -1)  # Flatten each image to 3072 features (batch_size, 3072)
        return self.model(x)  # Pass through the model





# Example usage for generating a single fake CAN image
latent_dim = 100
generator = Generator(latent_dim=latent_dim)

# Generate a single fake CAN image
random_noise = torch.randn(1, latent_dim)  # Single random noise vector
fake_image = generator(random_noise)  # Output shape is [1, 64, 48], values are 0 or 1

# Remove batch and channel dimensions
fake_image = fake_image.squeeze(0).squeeze(0).detach().numpy()  # Output shape [64, 48]

# Visualize the binary fake CAN image
plt.figure(figsize=(8, 6))
plt.imshow(fake_image, cmap="binary")  # Visualize the binary 64x48 fake CAN image
plt.title("Fake CAN Image (Generated by Generator)")
plt.colorbar()  # Add a color bar to see intensity values (0 or 1)
plt.show()





import torch
import torch.optim as optim
import torch.nn as nn

# Initialize models
generator = Generator(latent_dim=100)
first_discriminator = Discriminator()  # First Discriminator for known attacks
second_discriminator = Discriminator()  # Second Discriminator for unknown attacks

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d1 = optim.Adam(first_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d2 = optim.Adam(second_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification

# Training loop
num_epochs = 100
ide = 0
for epoch in range(num_epochs):
    print("epoch "+str(ide))
    ide+=1
    for normal_data, attack_data in zip(normal_batches, intrusion_batches):  # Assuming batches are prepared
       # print("RUN---")
        batch_size = normal_data.size(0)

        # Step 1: Train the First Discriminator (Known Attacks)
        first_discriminator.train()
        optimizer_d1.zero_grad()


        # Real labels for normal data (1) --> Only one label for the entire batch
        real_labels = torch.ones(1, 1)  # Shape: (1, 1) for the entire batch
        # Fake labels for attack data (0) --> Only one label for the entire batch
        fake_labels = torch.zeros(1, 1)  # Shape: (1, 1) for the entire batch


        normal_data = normal_data.reshape(1,-1)
        attack_data = attack_data.reshape(1,-1)

         # Train on normal data
        output_real = first_discriminator(normal_data)  # Passes through forward(), which flattens internally
        loss_real = criterion(output_real, real_labels)

        # Train on attack data
        output_fake = first_discriminator(attack_data)  # Passes through forward(), which flattens internally
        loss_fake = criterion(output_fake, fake_labels)

        #print(f"First Discriminator - Real Data: {output_real.item()} (Real label: 1)")
        #print(f"First Discriminator - Fake Data: {output_fake.item()} (Fake label: 0)")


        # Backprop and update First Discriminator
        loss_d1 = loss_real + loss_fake
        loss_d1.backward()
        optimizer_d1.step()

        # Step 2: Train the Generator and Second Discriminator (Unknown Attacks)
        noise = torch.randn(1, 100)  # Latent vector for Generator
        fake_images = generator(noise)  # Generate fake images
       # fake_images = fake_images.squeeze(0).squeeze(0).detach().numpy()  # Output shape [64, 48]

        optimizer_g.zero_grad()
        optimizer_d2.zero_grad()
        #print(normal_data.shape)
       #print(fake_images.shape)

        # Train the Second Discriminator with real and fake images
        output_real_d2 = second_discriminator(normal_data)  # Pass through forward(), which flattens internally
        #print(output_real_d2)
        output_fake_d2 = second_discriminator(fake_images)  # Pass through forward(), which flattens internally

        loss_real_d2 = criterion(output_real_d2, real_labels)
        loss_fake_d2 = criterion(output_fake_d2, fake_labels)

        # Backprop and update Second Discriminator
        loss_d2 = loss_real_d2 + loss_fake_d2
        loss_d2.backward()
        optimizer_d2.step()

        # Train the Generator to fool the Second Discriminator
        output_fake_g = second_discriminator(fake_images)  # Pass through forward(), which flattens internally
        loss_g = criterion(output_fake_g, real_labels)  # Generator wants to fool D into thinking it's real
        loss_g.backward()
        optimizer_g.step()

    # Print progress for each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] | D1 Loss: {loss_d1.item()} | D2 Loss: {loss_d2.item()} | G Loss: {loss_g.item()}")










'''
# ===========================
# Evaluate Model (Intrusion and Normal Classification)
# ===========================
def evaluate_model_with_percentages(discriminator, normal_batches, intrusion_batches):
    #Evaluate the GAN's ability to classify real vs. fake CAN IDs and detect intrusions.

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

'''
