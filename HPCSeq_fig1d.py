import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import pdfTextSet

# Number of scenes
n_scenes = 10
scenes = list(range(n_scenes))  # Scene labels: 0 to 9

# 1. Create Structured Transition Matrix
# Each scene transitions to the next scene with probability 1
structured_matrix = np.zeros((n_scenes, n_scenes))
for i in range(n_scenes):
    next_scene = (i + 1) % n_scenes  # Wrap around to 0 after the last scene
    structured_matrix[i, next_scene] = 1

# 2. Create Random Transition Matrix
# Each scene transitions to any other scene with equal probability (1/9)
random_matrix = np.full((n_scenes, n_scenes), 1/9)
np.fill_diagonal(random_matrix, 0)  # Probability 0 to stay in the same scene

# 3. Define Color Limits
vmin = 0
vmax = 1

# 4. Plotting the Combined Heatmaps
fig, axs = plt.subplots(1, 2, figsize=(16, 7))

# Structured Transition Heatmap
sns.heatmap(
    structured_matrix,
    ax=axs[0],
    cmap='coolwarm',
    cbar=True,
    annot=True,
    fmt=".2f",
    xticklabels=scenes,
    yticklabels=scenes,
    vmin=vmin,
    vmax=vmax
)
axs[0].set_title('Structured Transition Matrix')
axs[0].set_xlabel('To Scene')
axs[0].set_ylabel('From Scene')

# Random Transition Heatmap
sns.heatmap(
    random_matrix,
    ax=axs[1],
    cmap='coolwarm',
    cbar=True,
    annot=True,
    fmt=".2f",
    xticklabels=scenes,
    yticklabels=scenes,
    vmin=vmin,
    vmax=vmax
)
axs[1].set_title('Random Transition Matrix')
axs[1].set_xlabel('To Scene')
axs[1].set_ylabel('From Scene')

plt.tight_layout()

# 5. Save the Combined Heatmaps as a Single PNG File
combined_filename = "transition_matrices.pdf"
plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
print(f"Combined heatmaps saved as '{combined_filename}'.")

# 6. Save Each Heatmap Separately
# --------------------------------

# Save Structured Transition Matrix
fig_struct, ax_struct = plt.subplots(figsize=(8, 7))
sns.heatmap(
    structured_matrix,
    ax=ax_struct,
    cmap='coolwarm',
    cbar=True,
    annot=False,
    fmt=".2f",
    xticklabels=scenes,
    yticklabels=scenes,
    vmin=vmin,
    vmax=vmax
)
ax_struct.set_title('Structured Transition Matrix')
ax_struct.set_xlabel('To Scene')
ax_struct.set_ylabel('From Scene')
plt.tight_layout()
structured_filename = "structured_transition_matrix.pdf"
fig_struct.savefig(structured_filename, dpi=300, bbox_inches='tight')
plt.close(fig_struct)  # Close the figure to free memory
print(f"Structured transition matrix heatmap saved as '{structured_filename}'.")

# Save Random Transition Matrix
fig_rand, ax_rand = plt.subplots(figsize=(8, 7))
sns.heatmap(
    random_matrix,
    ax=ax_rand,
    cmap='coolwarm',
    cbar=True,
    annot=False,
    fmt=".2f",
    xticklabels=scenes,
    yticklabels=scenes,
    vmin=vmin,
    vmax=vmax
)
ax_rand.set_title('Random Transition Matrix')
ax_rand.set_xlabel('To Scene')
ax_rand.set_ylabel('From Scene')
plt.tight_layout()
random_filename = "random_transition_matrix.pdf"
fig_rand.savefig(random_filename, dpi=300, bbox_inches='tight')
plt.close(fig_rand)  # Close the figure to free memory
print(f"Random transition matrix heatmap saved as '{random_filename}'.")

