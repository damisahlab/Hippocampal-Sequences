import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap

# Define the size of the matrix
N = 10

# Initialize a 10x10 matrix
matrix = np.zeros((N, N), dtype=int)

# Compute the distance from the diagonal with circular wraparound
for i in range(N):
    for j in range(N):
        distance = min(abs(i - j), N - abs(i - j))
        matrix[i, j] = distance - 0.75

# Mask the diagonal (distance = 0)
mask = (matrix <-1)
matrix_masked = np.ma.masked_where(mask, matrix)

# Define specific colors for each distance using viridis_r
# We'll extract distinct colors from viridis_r for distances 1 to 5
viridis_r = plt.cm.get_cmap('viridis_r', 5)  # 5 discrete colors
listed_colors = viridis_r(np.arange(viridis_r.N))

# Create a ListedColormap
cmap = ListedColormap(listed_colors)

# Define boundaries for discrete color mapping
boundaries = np.arange(0.5, 6.5, 1)  # [0.5, 1.5, 2.5, ..., 5.5]
norm = colors.BoundaryNorm(boundaries, cmap.N)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))

# Display the heatmap with discrete normalization
cax = ax.imshow(matrix_masked, cmap=cmap, norm=norm)

# Add a colorbar with ticks for each distance
cbar = fig.colorbar(cax, ax=ax, ticks=[1, 2, 3, 4, 5])
cbar.set_label('Distance from Diagonal', fontsize=12)
cbar.set_ticks([1, 2, 3, 4, 5])
cbar.set_ticklabels(['1', '2', '3', '4', '5'])  # Optional: Explicit tick labels

# Set axis labels and title
ax.set_xlabel('Column', fontsize=12)
ax.set_ylabel('Row', fontsize=12)
ax.set_title('10x10 Heatmap with Circular Wraparound', fontsize=14, pad=20)

# Configure tick marks
ax.set_xticks(np.arange(N))
ax.set_yticks(np.arange(N))

# Optional: Add grid lines for better readability
ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(which='minor', bottom=False, left=False)

# Optional: Label each cell with its distance value
'''
for i in range(N):
    for j in range(N):
        if not mask[i, j]:
            text = ax.text(j, i, matrix[i, j],
                           ha="center", va="center", color="black", fontsize=10)
'''
# Adjust layout for better spacing
plt.tight_layout()

# Save the plot as a PDF
plt.savefig('viridisDist.pdf', format='pdf')

# Additionally, save as SVG for better compatibility with Illustrator
plt.savefig('viridisDist.svg', format='svg')

# Optionally, display the plot to verify

