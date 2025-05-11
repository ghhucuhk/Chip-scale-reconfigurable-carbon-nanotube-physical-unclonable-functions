import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import hamming
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm  # for coolwarm color map

# --------------------------------------------------------------------------------
# 1. Define all seeds we want to test
# --------------------------------------------------------------------------------
seeds = [42, 51, 64, 77, 89, 91, 103, 114, 120, 133]

# Path to data
file_path = "/data/needed_data_formatted.xlsx"

# Directory for results
output_dir = "/results/XGBoost"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------------------------------------
# 2. Load and prepare data (common across all runs)
# --------------------------------------------------------------------------------

# Read Excel
data = pd.read_excel(file_path, header=None)

# Extract the binary responses from the specified rows and column
responses = data.iloc[1:20001, 0].copy()

# Convert the binary strings into lists of integers
response_bits = responses.apply(lambda x: [int(bit) for bit in x])

# Split the responses into query-response pairs (alternate rows)
query_responses = response_bits.iloc[::2].reset_index(drop=True)  # Odd-index rows as queries
actual_responses = response_bits.iloc[1::2].reset_index(drop=True)  # Even-index rows as responses

# Expand the query and response bits into separate columns
query_df = pd.DataFrame(query_responses.tolist(), columns=[f'Query_{i}' for i in range(108)])
response_df = pd.DataFrame(actual_responses.tolist(), columns=[f'Response_{i}' for i in range(108)])

# Combine queries and responses into a single DataFrame
full_df = pd.concat([query_df, response_df], axis=1)

# Define feature columns (queries) and target columns (responses)
feature_cols = [f'Query_{i}' for i in range(108)]
target_cols = [f'Response_{i}' for i in range(108)]

# --------------------------------------------------------------------------------
# 3. Prepare storage for PMFs across all seeds (NEW)
# --------------------------------------------------------------------------------
# We'll store 10 PMFs for n-HD and 10 PMFs for CC (one per seed).
# We use 40 bins in the range of [0,1] for n-HD, and 40 bins in [-1,1] for CC.
# We'll fix the bin edges so we can reference them after the loop.
n_hd_bins = np.linspace(0, 1, 41)    # 40 bins from 0 to 1
cc_bins   = np.linspace(-1, 1, 41)   # 40 bins from -1 to 1

hamming_pmf_array = []     # list of length 10, each is a PMF array for n-HD
correlation_pmf_array = [] # list of length 10, each is a PMF array for CC

# --------------------------------------------------------------------------------
# 4. Loop over all seeds and repeat the experiment
# --------------------------------------------------------------------------------
for seed in seeds:
    # -------------------------
    # (A) Set the random seed for reproducibility
    # -------------------------
    random.seed(seed)
    np.random.seed(seed)

    # -------------------------
    # (B) Split dataset with the given random_state
    # -------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        full_df[feature_cols],
        full_df[target_cols],
        train_size=0.8,
        random_state=seed,
        shuffle=True
    )

    # -------------------------
    # (C) Training and Evaluating XGBoost Models
    # -------------------------
    accuracies = []
    trained_models = []

    for i, target in enumerate(target_cols):
        print(f"[Seed {seed}] Training model for {target} ({i+1}/108)")
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        
        eval_set = [(X_val, y_val[target])]
        model.fit(
            X_train, y_train[target],
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=False
        )
        
        predictions = model.predict(X_val)
        acc = accuracy_score(y_val[target], predictions)
        accuracies.append(acc)
        trained_models.append(model)

    average_accuracy = np.mean(accuracies)
    print(f"\n[Seed {seed}] Average Prediction Accuracy with XGBoost: {average_accuracy:.4f}")

    # -------------------------
    # (D) Visualization of Prediction Accuracy
    # -------------------------
    predicted_responses = pd.DataFrame()
    for i, target in enumerate(target_cols):
        predicted_responses[target] = trained_models[i].predict(X_val)

    # We'll create a 10×10 matrix comparing the first 10 rows of X_val to the first 10 rows of predicted_responses
    num_pairs = 10
    accuracy_matrix = np.zeros((num_pairs, num_pairs))

    for i in range(num_pairs):
        query_sample = X_val.iloc[i].values  # actual bits
        for j in range(num_pairs):
            predicted_sample = predicted_responses.iloc[j].values
            accuracy_matrix[i, j] = np.mean(query_sample == predicted_sample)

    accuracy_matrix_flipped = np.flipud(accuracy_matrix)

    # Plot predictive accuracy heatmap (0 to 1) with 'RdBu'
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(
        accuracy_matrix_flipped,
        vmin=0,
        vmax=1,
        annot=True,
        cmap='RdBu',
        square=True,
        fmt=".2f",
        cbar_kws={'shrink': 0.8}
    )
    ax.set_xticks(np.arange(num_pairs) + 0.5)
    ax.set_xticklabels(np.arange(num_pairs), rotation=0)
    ax.set_yticks(np.arange(num_pairs) + 0.5)
    ax.set_yticklabels(np.arange(num_pairs), rotation=0)

    plt.title(f'Predictive Accuracy Heatmap (10×10) [Seed {seed}]')
    plt.xlabel('Predicted QR Pairs')
    plt.ylabel('Actual QR Pairs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'prediction_heatmap_seed{seed}.svg'), format='svg')
    plt.show()

    # -------------------------
    # (E) Hamming Distance and Correlation
    # -------------------------
    random_indices = random.sample(range(len(X_val)), 20)
    predicted_sample = predicted_responses.iloc[random_indices].values
    actual_sample = y_val.iloc[random_indices].values

    # Compute normalized hamming distance and correlation coefficient
    hamming_distances = [hamming(actual_sample[i], predicted_sample[i]) for i in range(len(random_indices))]
    correlation_coefficients = [
        np.corrcoef(actual_sample[i], predicted_sample[i])[0, 1] if np.std(actual_sample[i]) > 0 and np.std(predicted_sample[i]) > 0 else 0
        for i in range(len(random_indices))
    ]

    # Create 10×10 matrices for Hamming distance & correlation
    hamming_matrix = np.zeros((num_pairs, num_pairs))
    corr_matrix = np.zeros((num_pairs, num_pairs))

    for i in range(num_pairs):
        actual_bits = X_val.iloc[i].values
        for j in range(num_pairs):
            predicted_bits = predicted_responses.iloc[j].values
            hamming_matrix[i, j] = np.mean(actual_bits != predicted_bits)
            # Handle cases where correlation is undefined
            if np.std(actual_bits) > 0 and np.std(predicted_bits) > 0:
                corr_matrix[i, j] = np.corrcoef(actual_bits, predicted_bits)[0, 1]
            else:
                corr_matrix[i, j] = 0  # Assign 0 if standard deviation is zero

    hamming_matrix_flipped = np.flipud(hamming_matrix)
    corr_matrix_flipped = np.flipud(corr_matrix)

    # White→Red gradient for normalized Hamming distance
    red_gradient = LinearSegmentedColormap.from_list(
        "RedGradient", 
        [(200/255, 0, 0),(1, 1, 1)]
    )
    # White→Blue gradient for correlation
    blue_gradient = LinearSegmentedColormap.from_list(
        "BlueGradient", 
        [(35/255, 114/255, 169/255),(1, 1, 1)]
    )

    # Plot Normalized Hamming Distance Heatmap
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(
        hamming_matrix_flipped,
        vmin=0,
        vmax=1,
        annot=False,
        cmap=red_gradient,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    ax.set_xticks(np.arange(num_pairs) + 0.5)
    ax.set_xticklabels(np.arange(num_pairs), rotation=0)
    ax.set_yticks(np.arange(num_pairs) + 0.5)
    ax.set_yticklabels(np.arange(num_pairs), rotation=0)

    plt.title(f'Normalized Hamming Distance Heatmap (10×10) [Seed {seed}]')
    plt.xlabel('Predicted QR Pairs')
    plt.ylabel('Actual QR Pairs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'hamming_heatmap_seed{seed}.svg'), format='svg')
    plt.show()

    # Plot Correlation Coefficient Heatmap
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(
        corr_matrix_flipped,
        vmin=-1,
        vmax=1,
        annot=False,
        cmap=blue_gradient,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    ax.set_xticks(np.arange(num_pairs) + 0.5)
    ax.set_xticklabels(np.arange(num_pairs), rotation=0)
    ax.set_yticks(np.arange(num_pairs) + 0.5)
    ax.set_yticklabels(np.arange(num_pairs), rotation=0)

    plt.title(f'Correlation Coefficient Heatmap (10×10) [Seed {seed}]')
    plt.xlabel('Predicted QR Pairs')
    plt.ylabel('Actual QR Pairs')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'corr_heatmap_seed{seed}.svg'), format='svg')
    plt.show()

    # -----------------------
    # (F) Histograms (No Gaussian Fit)
    # -----------------------
    def plot_histogram(data, bins, title, xlabel, ylabel, color, filename):
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        pmf = hist / np.sum(hist)  # Convert histogram to PMF

        plt.figure(figsize=(8, 6))
        plt.bar(bin_edges[:-1], pmf, width=np.diff(bin_edges), align="edge",
                color=color, edgecolor="black", alpha=0.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(filename, format="svg")
        plt.show()
        return pmf  # Return the PMF if we want to store it

    # Plot histogram for Normalized Hamming Distance (no Gaussian fit)
    pmf_hd = plot_histogram(
        hamming_distances,
        bins=n_hd_bins,  # 40 bins from 0 to 1
        title=f"PMF of Normalized Hamming Distance [Seed {seed}]",
        xlabel="Normalized Hamming Distance",
        ylabel="Probability Mass Function (PMF)",
        color="#C80000",
        filename=os.path.join(output_dir, f'n-HD_seed{seed}.svg')
    )

    # Plot histogram for Correlation Coefficient (no Gaussian fit)
    pmf_cc = plot_histogram(
        correlation_coefficients,
        bins=cc_bins,  # 40 bins from -1 to 1
        title=f"PMF of Correlation Coefficient [Seed {seed}]",
        xlabel="Correlation Coefficient",
        ylabel="Probability Mass Function (PMF)",
        color="#2372A9",
        filename=os.path.join(output_dir, f'CC_seed{seed}.svg')
    )

    # -------------------------
    # (G) Summary of Results
    # -------------------------
    print(f"\n[Seed {seed}] Summary of Model Accuracies per Bit:")
    for i, acc in enumerate(accuracies):
        print(f"{target_cols[i]}: {acc:.4f}")

    print(f"\n[Seed {seed}] Overall Average Accuracy: {average_accuracy:.4f}")

    hd_mean = np.mean(hamming_distances)
    hd_std = np.std(hamming_distances)
    cc_mean = np.mean(correlation_coefficients)
    cc_std = np.std(correlation_coefficients)

    print(f"[Seed {seed}] Normalized Hamming Distance Mean: {hd_mean:.4f}, Std: {hd_std:.4f}")
    print(f"[Seed {seed}] Correlation Coefficient Mean: {cc_mean:.4f}, Std: {cc_std:.4f}")
    print("-" * 70)

    # -----------------------------------------------------------
    # (H) NEW: Store the PMFs for the final 3D plots
    # -----------------------------------------------------------
    hamming_pmf_array.append(pmf_hd)       # shape (40,)
    correlation_pmf_array.append(pmf_cc)   # shape (40,)

# --------------------------------------------------------------------------------
# 5. After all seeds are processed, draw 3D PMFs of n-HD & CC for all 10 seeds
# --------------------------------------------------------------------------------
# We'll have hamming_pmf_array: shape (10,40)
# We'll have correlation_pmf_array: shape (10,40)
random_seed = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]  # For labeling the y-axis

# --- 3D PMF for Normalized Hamming Distance ---
fig = plt.figure(figsize=(20, 8), facecolor="white")
ax = fig.add_subplot(111, projection='3d', facecolor="white")

colors = cm.coolwarm(np.linspace(0, 1, len(random_seed)))  # color gradient

# Light grey dashed grid lines
ax.xaxis._axinfo['grid'].update(color="lightgrey", linestyle="dashed")
ax.yaxis._axinfo['grid'].update(color="lightgrey", linestyle="dashed")
ax.zaxis._axinfo['grid'].update(color="lightgrey", linestyle="dashed")

# x-axis bin centers for n-HD
x_centers_hd = (n_hd_bins[:-1] + n_hd_bins[1:]) / 2.0

for i, pmf in enumerate(hamming_pmf_array):
    x = x_centers_hd  # 40 bin centers in [0,1]
    y = np.ones_like(x) * (random_seed[i] * 1.0)  # space along y-axis by seed
    z = pmf
    # bar width is ~ (bin width), adjusted for higher resolution
    bin_width = (n_hd_bins[1] - n_hd_bins[0]) * 0.8  # 0.02
    ax.bar(
        x, z, zs=y, zdir='y',
        width=bin_width,
        color=colors[i], alpha=0.8, edgecolor="black"
    )

ax.set_xlabel("Normalized Hamming Distance")
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])  # custom ticks
ax.set_ylabel("Random Seed")
ax.set_zlabel("PMF")
ax.set_xlim(0, 1)
ax.set_ylim(min(random_seed)*1.0 - 10, max(random_seed)*1.0 + 10)  # small padding
ax.set_title("3D PMF of Normalized Hamming Distance (10 seeds)")
ax.view_init(elev=25, azim=-30)
#ax.w_xaxis.pane.fill = False
#ax.w_yaxis.pane.fill = False
#ax.w_zaxis.pane.fill = False
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spaced_3d_pmf_hamming_distance_white_bg.svg"), format="svg")
plt.show()

# --- 3D PMF for Correlation Coefficients ---
fig = plt.figure(figsize=(20, 8), facecolor="white")
ax = fig.add_subplot(111, projection='3d', facecolor="white")

# Light grey dashed grid lines
ax.xaxis._axinfo['grid'].update(color="lightgrey", linestyle="dashed")
ax.yaxis._axinfo['grid'].update(color="lightgrey", linestyle="dashed")
ax.zaxis._axinfo['grid'].update(color="lightgrey", linestyle="dashed")

# x-axis bin centers for CC
x_centers_cc = (cc_bins[:-1] + cc_bins[1:]) / 2.0

for i, pmf in enumerate(correlation_pmf_array):
    x = x_centers_cc  # 40 bin centers in [-1,1]
    y = np.ones_like(x) * (random_seed[i] * 1.0)
    z = pmf
    # bar width is ~ (bin width), adjusted for higher resolution
    bin_width = (cc_bins[1] - cc_bins[0]) * 0.8  # 0.04
    ax.bar(
        x, z, zs=y, zdir='y',
        width=bin_width,
        color=colors[i], alpha=0.8, edgecolor="black"
    )

ax.set_xlabel("Correlation Coefficient")
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_ylabel("Random Seed")
ax.set_zlabel("PMF")
ax.set_xlim(-1, 1)
ax.set_ylim(min(random_seed)*1.0 - 10, max(random_seed)*1.0 + 10)  # small padding
ax.set_title("3D PMF of Correlation Coefficient (10 seeds)")
ax.view_init(elev=25, azim=-30)
#ax.w_xaxis.pane.fill = False
#ax.w_yaxis.pane.fill = False
#ax.w_zaxis.pane.fill = False
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spaced_3d_pmf_correlation_coefficient_white_bg.svg"), format="svg")
plt.show()