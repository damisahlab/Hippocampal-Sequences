from utils import pdfTextSet
import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from ripser import ripser
#from persim import plot_diagrams
from matplotlib.collections import LineCollection
from utils import ripserHelper as rh

# Set random seed for reproducibility
np.random.seed(42)

RING_SIZE=1.5
RING_SIZE=1
# Generate points on a ring embedded in higher dimension and add noise
num_points = 15  # Adjusted from 50 to 15 as per your adaptation
num_points = 20  # Adjusted from 50 to 15 as per your adaptation
#num_points = 30  # Adjusted from 50 to 15 as per your adaptation
dim = 5  # higher-dimensional embedding
angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
circle_x = RING_SIZE*np.cos(angles)
circle_y = RING_SIZE*np.sin(angles)
points = np.zeros((num_points, dim))
points[:, 0] = circle_x
points[:, 1] = circle_y

# Add Gaussian noise to introduce spurious features
noise_level = 0.25  # Final noise_level as per your adaptation
#noise_level = 0.3  # Final noise_level as per your adaptation
points += np.random.normal(scale=noise_level, size=points.shape)

# Compute persistent homology with ripser (up to H1)
res = ripser(points, maxdim=1)
diagrams = res['dgms']

# Extract H1 intervals
H1_intervals = diagrams[1]

# Define radii at which we "grow" balls for illustration
radii = [0.05, 0.15, 0.25, 0.35, 0.45]
#radii = np.array([0.2, 0.4, 0.6, 0.8, 1.0]) - 0.1
radii = 1.5*np.array([0.05, 0.15, 0.25, 0.35, 0.45])
radii=np.linspace(0.075,0.55,20)
radii=np.linspace(0.075,0.6,50)
radii=np.linspace(0.1,0.6,50)
radii=np.linspace(0.1,0.5,50)
radii = radii[::-1]  # Now: [0.45, 0.35, 0.25, 0.15, 0.05]

########################################
# Define Separate Colormaps and Normalizations
########################################

# Plot 1: All Radii Overlaid
# Use 'viridis' colormap for better perceptual uniformity
#cmap_radii = plt.cm.jet_r
cmap_radii = plt.cm.Greys

# Normalize based on radii values for consistent mapping in Plot 1
norm_radii = plt.Normalize(min(radii), max(radii))  # 0.05 to 0.45

# Generate colors for radii based on normalization
colors_radii = cmap_radii(norm_radii(radii))

# Plot 3: H1 Barcode
# Use 'plasma' colormap for distinct color mapping
#cmap_barcode = plt.cm.jet_r
cmap_barcode = plt.cm.Greys

# (Normalization will be defined later based on filtration scale)

########################################
# Plot 1: All Radii Overlaid
########################################
fig1, ax_top = plt.subplots(figsize=(6,6))
ax_top.set_aspect('equal')
ax_top.set_xlim(1.5*RING_SIZE*(-1.5), 1.5*RING_SIZE*(1.5))
ax_top.set_ylim(1.5*RING_SIZE*(-1.5), 1.5*RING_SIZE*1.5)
#ax_top.set_xticks([])
#ax_top.set_yticks([])
ax_top.set_title("All Radii Overlaid", fontsize=10)

# Plot the points in black
ax_top.scatter(points[:,0], points[:,1], color='black', s=20, zorder=3)

# Highlight approximate ring structure in black
guide = Circle((0,0), 1.0, fill=False, edgecolor='black', linewidth=2, alpha=0.8)
ax_top.add_patch(guide)

# Plot circles with colors corresponding to radii
for i, r in enumerate(radii):
    color = colors_radii[i]
    for p in points[:,:2]:
        circ = Circle((p[0], p[1]), r, color=color, alpha=0.8, linewidth=1)
        ax_top.add_patch(circ)

# Add a colorbar for radii
sm1 = plt.cm.ScalarMappable(cmap=cmap_radii, norm=norm_radii)
sm1.set_array([])  # Only needed for matplotlib < 3.1
cbar1 = fig1.colorbar(sm1, ax=ax_top, orientation='vertical', fraction=0.046, pad=0.04)
cbar1.set_label('Radius', fontsize=9)

plt.tight_layout()
# Save to PDF and SVG
fig1.savefig("all_radii_overlaid.pdf")
fig1.savefig("all_radii_overlaid.svg")
plt.close(fig1)

########################################
# Plot 2: Persistence Diagrams
########################################
fig2, ax_dgm = plt.subplots(figsize=(4,4))
ax_dgm.set_title("Persistence Diagrams", fontsize=10)
rh.plot_diagrams_local(diagrams,  ax=ax_dgm)
ax_dgm.set_xlabel("Birth")
ax_dgm.set_ylabel("Death")

plt.tight_layout()
# Save to PDF and SVG
fig2.savefig("persistence_diagrams.pdf")
fig2.savefig("persistence_diagrams.svg")
plt.close(fig2)

########################################
# Plot 3: H1 Barcode
########################################
fig3, ax_bar = plt.subplots(figsize=(4,3))  # Adjusted figsize for better aspect ratio
ax_bar.set_title("H1 Barcode (Color Reflects Filtration Scale)", fontsize=10)
ax_bar.set_xlabel("Filtration Scale")
ax_bar.set_yticks([])

if len(H1_intervals) > 0:
    # Sort intervals by persistence (longest first)
    pers = H1_intervals[:,1] - H1_intervals[:,0]
    sort_idx = np.argsort(-pers)  # descending order by persistence
    H1_intervals_sorted = H1_intervals[sort_idx]

    # Compute actual filtration scale range
    min_filtration = H1_intervals_sorted[:,0].min()
    max_filtration = H1_intervals_sorted[:,1].max()

    # Update normalization based on actual filtration scale
    norm_scale = plt.Normalize(min_filtration, max_filtration)
    OFFSET=0.9
    norm_scale = plt.Normalize(OFFSET+np.min(radii)-0.3,OFFSET+np.max(radii)+0.05)

    # Use a distinct colormap for the barcode
    cmap_cont = cmap_barcode  # 'plasma' colormap

    # Optional: Print filtration scale range for verification
    print(f"Filtration Scale Range: {min_filtration} to {max_filtration}")

    y_offset = 0
    for i, (birth, death) in enumerate(H1_intervals_sorted):
        # Discretize the interval into segments
        n_segments = 100
        x_values = np.linspace(birth, death, n_segments)
        y_values = np.ones_like(x_values) * y_offset

        # Create line segments
        segments = np.array([x_values[:-1], y_values[:-1], x_values[1:], y_values[1:]]).T.reshape(-1,2,2)

        # For coloring, map the filtration scale to colors using the 'plasma' colormap
        midpoints = (x_values[:-1] + x_values[1:]) / 2.0
        segment_colors = cmap_cont(norm_scale(midpoints))

        # Create a LineCollection with those segments and colors
        lc = LineCollection(segments, colors=segment_colors, linewidth=4)
        ax_bar.add_collection(lc)

        # Label features in black
        ax_bar.text(death + 0.02, y_offset, f"Feature {len(H1_intervals)-i}", va='center', color='black', fontsize=9)

        y_offset += 0.5

    ax_bar.set_ylim(-0.5, y_offset)
    #ax_bar.set_xlim(min_filtration, max_filtration)
    ax_bar.set_xlim(0.75,1.8)

    # Add a colorbar for filtration scale
    sm3 = plt.cm.ScalarMappable(cmap=cmap_cont, norm=norm_scale)
    sm3.set_array([])
    cbar3 = fig3.colorbar(sm3, ax=ax_bar, orientation='vertical', fraction=0.046, pad=0.04)
    cbar3.set_label('Filtration Scale', fontsize=9)

else:
    # No H1 features
    ax_bar.text(0.5,0.5,"No H1 Features",ha='center',va='center')

plt.tight_layout()
# Save to PDF and SVG
fig3.savefig("h1_barcode.pdf")
fig3.savefig("h1_barcode.svg")
plt.close(fig3)


