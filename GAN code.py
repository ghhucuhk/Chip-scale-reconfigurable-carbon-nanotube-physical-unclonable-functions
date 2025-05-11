import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# -------------------------
# 1. Set Random 
# -------------------------
def set_seed(seed=186):
    """
    Sets random seeds for reproducibility. Change this number for 10-time different tests.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# -------------------------
# 2. Device Configuration
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# 3. Data Loading and Preprocessing
# -------------------------
file_path = '/data/needed_data_formatted.xlsx'
df = pd.read_excel(file_path, header=None, skiprows=1, nrows=20000)  # Skip header, read 20,000 rows
binary_data = df.iloc[:, 0].tolist()  # Extract first column

def binary_strings_to_tensor(data):
    """
    Converts a list of binary strings to a PyTorch tensor.
    Ensures all entries are valid binary strings and scales them to [-1, 1].
    """
    valid_data = []
    for idx, line in enumerate(data, start=2):  # Start=2 to account for skipped header
        if isinstance(line, str) and all(bit in '01' for bit in line.strip()):
            valid_data.append([int(bit) for bit in line.strip()])
        else:
            print(f"Invalid binary string skipped at row {idx}: {line}")  # Log invalid entries
    tensor = torch.Tensor(valid_data).float()
    tensor = tensor * 2 - 1  # Scale from [0,1] to [-1,1]
    return tensor

binary_data_tensor = binary_strings_to_tensor(binary_data)
print(f"Total valid rows processed: {binary_data_tensor.size(0)} / {len(binary_data)}")

# -------------------------
# 4. Model Definitions
# -------------------------

def weights_init_normal(m):
    """
    Applies Xavier normal initialization to linear layers.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    """
    Generator network with expanded capacity.
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 108),
            nn.Tanh()  # Output scaled to [-1,1]
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """
    Discriminator (Critic) network with reduced dropout and no final activation.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(108, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Lower dropout to 0.3
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Lower dropout to 0.3
            nn.Linear(1024, 1)
            # No activation
        )
    
    def forward(self, x):
        return self.model(x)

# -------------------------
# 5. Gradient Penalty
# -------------------------
def gradient_penalty(discriminator, real_data, fake_data):
    """
    Computes the gradient penalty for WGAN-GP.
    """
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1).to(real_data.device)
    epsilon = epsilon.expand_as(real_data)
    
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated.requires_grad_(True)
    
    prob_interpolated = discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

# -------------------------
# 6. Utility Functions
# -------------------------

def save_generated_data(generated_data, iteration):
    """
    Saves generated binary data to an Excel file.
    """
    generated_data = (generated_data + 1) / 2
    generated_binary = (generated_data > 0.5).int().numpy()
    generated_binary_strings = [''.join(str(bit) for bit in line) for line in generated_binary]
    output_file_path = f'/results/GAN code/x_th-generated_data_iteration_{iteration}.xlsx'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    generated_df = pd.DataFrame({'Generated Data': generated_binary_strings})
    generated_df.to_excel(output_file_path, index=False)
    print(f"Generated data for iteration {iteration} saved to {output_file_path}")

def plot_losses(generator_losses, discriminator_losses, iteration):
    """
    Plots and saves the generator and discriminator losses.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(generator_losses, label='Generator Loss')
    plt.plot(discriminator_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Losses up to Epoch {iteration}')
    plt.legend()
    plt.grid()
    os.makedirs('/results/GAN code', exist_ok=True)
    plt.savefig(f'/results/GAN code/x_th-loss_plot_epoch_{iteration}.svg')
    plt.close()

def visualize_generated_samples(generator, iteration, num_samples=10):
    """
    Generates and prints samples from the generator for inspection.
    """
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, 128).to(device)
        generated_data = generator(noise).cpu()
        generated_binary = (generated_data > 0).float()
        print(f"Generated Samples at Iteration {iteration}:\n", generated_binary)
    generator.train()

def calculate_hamming_distance(sample1, sample2):
    return (sample1 != sample2).sum()

def calculate_normalized_hamming_distance(sample1, sample2):
    return calculate_hamming_distance(sample1, sample2) / sample1.size(0)

def calculate_correlation_coefficient(sample1, sample2):
    sample1_np = sample1.cpu().numpy()
    sample2_np = sample2.cpu().numpy()
    if np.std(sample1_np) == 0 or np.std(sample2_np) == 0:
        return 0.0
    correlation, _ = pearsonr(sample1_np, sample2_np)
    return correlation

def evaluate_metrics(generator, discriminator, validation_data, num_samples=4000):
    """
    Evaluates average normalized Hamming distance, average correlation coefficient,
    and prediction accuracies (Discriminator Real Accuracy, Discriminator Fake Accuracy,
    Weighted Average, Generator Accuracy).
    """
    generator.eval()
    discriminator.eval()
    
    with torch.no_grad():
        # Real Samples
        real_data = validation_data
        # Removed additional noise to real_data
        real_scores = discriminator(real_data)
        real_preds = (real_scores > 0).float()
        real_acc = real_preds.mean().item()
        
        # Fake Samples
        noise = torch.randn(validation_data.size(0), 128).to(device)
        fake_data = generator(noise)
        # Removed additional noise to fake_data
        fake_scores = discriminator(fake_data)
        fake_preds = (fake_scores <= 0).float()
        fake_acc = fake_preds.mean().item()
        
        # Weighted Average Accuracy
        disc_weighted_acc = 0.5 * real_acc + 0.5 * fake_acc
        
        # Generator Accuracy: proportion of fake samples recognized as fake
        generator_acc = fake_acc
        
        # Hamming Distance & Correlation
        hd_total, cc_total, count = 0.0, 0.0, 0
        batch_size = num_samples
        for i in range(0, validation_data.size(0), batch_size):
            batch_real = validation_data[i:i+batch_size]
            noise = torch.randn(batch_real.size(0), 128).to(device)
            generated = generator(noise)
            # Convert to binary
            generated_binary = (generated > 0).float()
            
            hd = (batch_real != generated_binary).sum(dim=1).float() / 108
            hd_total += hd.sum().item()
            
            for j in range(batch_real.size(0)):
                cc = calculate_correlation_coefficient(batch_real[j], generated_binary[j])
                cc_total += cc
            count += batch_real.size(0)
        
        average_normalized_hd = hd_total / count
        average_cc = cc_total / count
    
    generator.train()
    discriminator.train()
    
    return {
        'Wasserstein Distance': real_scores.mean().item() - fake_scores.mean().item(),
        'Average Normalized Hamming Distance': average_normalized_hd,
        'Average Correlation Coefficient': average_cc,
        'Discriminator Real Accuracy': real_acc,
        'Discriminator Fake Accuracy': fake_acc,
        'Discriminator Weighted Average Accuracy': disc_weighted_acc,
        'Generator Accuracy': generator_acc
    }

# -------------------------
# 7. Training Function
# -------------------------
def train_wgan_gp(binary_data_tensor, num_epochs=150, batch_size=64, lambda_gp=1, n_critic=5):
    """
    Trains the WGAN-GP model with the specified parameters.
    - Reduced lambda_gp to 1 for a less strict gradient penalty.
    - Removed additional noise from real/fake data inside the training loop.
    - Reduced dropout in the Discriminator to 0.3
    - Removed weight_decay from the Discriminator's optimizer.
    """
    # Split data into training and validation sets
    train_indices = list(range(16000))
    validation_indices = list(range(16000, 20000))
    
    train_dataset = TensorDataset(binary_data_tensor[train_indices])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_data = binary_data_tensor[validation_indices].to(device)
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Apply weight initialization
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    # Define optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0006, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999), weight_decay=0)
    
    # Lists to keep track of losses
    generator_losses = []
    discriminator_losses = []
    
    # Lists to keep track of metrics
    metrics_history = []
    
    for epoch in range(1, num_epochs + 1):
        generator.train()
        discriminator.train()
        
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        
        for i, real_batch in enumerate(train_loader, 1):
            real_data = real_batch[0].to(device)
            current_batch_size = real_data.size(0)
            
            ## ---------------------
            ## Train Discriminator
            ## ---------------------
            optimizer_d.zero_grad()
            
            # Generate fake data
            noise = torch.randn(current_batch_size, 128).to(device)
            fake_data = generator(noise).detach()
            
            # Discriminator outputs with NO noise injection
            real_score = discriminator(real_data)
            fake_score = discriminator(fake_data)
            
            # Gradient penalty
            gp = gradient_penalty(discriminator, real_data, fake_data)
            
            # Wasserstein loss for Discriminator
            d_loss = fake_score.mean() - real_score.mean() + lambda_gp * gp
            d_loss.backward()
            optimizer_d.step()
            d_loss_epoch += d_loss.item()
            
            ## -----------------
            ## Train Generator
            ## -----------------
            if i % n_critic == 0:
                optimizer_g.zero_grad()
                noise = torch.randn(current_batch_size, 128).to(device)
                fake_data = generator(noise)
                # Generator wants to maximize real_score => minimize negative
                g_loss = -discriminator(fake_data).mean()
                g_loss.backward()
                optimizer_g.step()
                g_loss_epoch += g_loss.item()
        
        # Average losses for the epoch
        avg_d_loss = d_loss_epoch / len(train_loader)
        g_updates = len(train_loader) // n_critic if len(train_loader) >= n_critic else 1
        avg_g_loss = g_loss_epoch / g_updates
        discriminator_losses.append(avg_d_loss)
        generator_losses.append(avg_g_loss)
        
        # Evaluate metrics
        metrics = evaluate_metrics(generator, discriminator, validation_data, num_samples=4000)
        metrics_history.append(metrics)
        
        # Print metrics
        print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")
        print(f"Wasserstein Distance: {metrics['Wasserstein Distance']:.4f}")
        print(f"Average Normalized Hamming Distance: {metrics['Average Normalized Hamming Distance']:.4f}")
        print(f"Average Correlation Coefficient: {metrics['Average Correlation Coefficient']:.4f}")
        print(f"Discriminator Real Accuracy: {metrics['Discriminator Real Accuracy']*100:.2f}%")
        print(f"Discriminator Fake Accuracy: {metrics['Discriminator Fake Accuracy']*100:.2f}%")
        print(f"Discriminator Weighted Average Accuracy: {metrics['Discriminator Weighted Average Accuracy']*100:.2f}%")
        print(f"Generator Accuracy: {metrics['Generator Accuracy']*100:.2f}%\n")
        
        # Plot losses
        plot_losses(generator_losses, discriminator_losses, epoch)
        
        # Visualize generated samples and save data every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                noise = torch.randn(10, 128).to(device)
                sample_fake = generator(noise).cpu()
            visualize_generated_samples(generator, epoch)
            save_generated_data(sample_fake, epoch)
    
    print("Training Completed!")
    
    # Final Evaluation
    final_metrics = metrics_history[-1]
    print(f"\nFinal Metrics after {num_epochs} epochs:")
    print(f"Wasserstein Distance: {final_metrics['Wasserstein Distance']:.4f}")
    print(f"Average Normalized Hamming Distance: {final_metrics['Average Normalized Hamming Distance']:.4f}")
    print(f"Average Correlation Coefficient: {final_metrics['Average Correlation Coefficient']:.4f}")
    print(f"Discriminator Real Accuracy: {final_metrics['Discriminator Real Accuracy']*100:.2f}%")
    print(f"Discriminator Fake Accuracy: {final_metrics['Discriminator Fake Accuracy']*100:.2f}%")
    print(f"Discriminator Weighted Average Accuracy: {final_metrics['Discriminator Weighted Average Accuracy']*100:.2f}%")
    print(f"Generator Accuracy: {final_metrics['Generator Accuracy']*100:.2f}%")
    
    # Save final generated data
    with torch.no_grad():
        noise = torch.randn(64, 128).to(device)
        fake_data_final = generator(noise).cpu()
    visualize_generated_samples(generator, "Final")
    save_generated_data(fake_data_final, "Final")

# -------------------------
# 8. Start Training
# -------------------------
train_wgan_gp(binary_data_tensor, num_epochs=150, batch_size=64, lambda_gp=1, n_critic=5)
