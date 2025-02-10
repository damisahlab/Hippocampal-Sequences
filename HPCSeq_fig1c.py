import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from matplotlib.transforms import Affine2D
from matplotlib.colors import Normalize, BoundaryNorm
import matplotlib.cm as cm
from utils import pdfTextSet

# Function to create the Structured Order figure
def create_structured_order(ax, colors):
    """
    Arrange 10 rectangles in a circular pattern with 10 rows,
    each containing 1 rectangle. Scenes are arranged in clockwise
    order with increasing numbers, rotated by +90 degrees so that
    Scene 1 is at the top. Scene 1 overlays Scene 10 but not Scene 2.
    """
    num_rows = 10
    radius = 1.6  # Radius of the circle
    angle_between_rows = 360 / num_rows  # Degrees

    rect_width = 0.6
    rect_height = 1.2

    # Define the plotting order: [Scene 10, Scene 1, Scenes 2-9]
    # Zero-based indexing: Scene 1 -> 0, Scene 10 -> 9
    plot_order = [9, 0] + list(range(1, 9))  # [Scene 10, Scene 1, Scenes 2-9]

    for idx, scene in enumerate(plot_order):
        # Rotate the circle by +90 degrees: start at 90 degrees
        angle_deg = 90 - scene * angle_between_rows  # Clockwise order

        # Fixed spacing_deg as per user's adaptation
        spacing_deg = 20  # degrees; adjust as needed for visual spacing

        # Calculate angle for the single rectangle in the row
        theta = np.deg2rad(angle_deg)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        # Create a rectangle centered at (x, y) with rotation
        trans = Affine2D().rotate_deg(angle_deg) + Affine2D().translate(x, y) + ax.transData
        rect = patches.Rectangle(
            (-rect_width / 2, -rect_height / 2),  # Centered at origin
            rect_width,
            rect_height,
            linewidth=1,
            edgecolor='black',
            facecolor=colors[scene],  # Each scene has a unique color
            transform=trans,
            zorder=scene + 1  # Optional: can be adjusted if needed
        )
        ax.add_patch(rect)

    # Set limits and aspect
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')
    ax.set_title('Structured Scene Order')
    ax.axis('off')

# Function to create the Random Order figure
def create_random_order(ax, random_colors, num_rects=20):
    """
    Arrange 20 rectangles in a straight line with colors provided.
    - First 10 rectangles: Scenes 1 through 10 in colorbar order.
    - Next 10 rectangles: Another random permutation of Scenes 1 through 10.
    Rectangles are wider than tall, overlapping horizontally by 80%,
    and have a slight vertical offset to fit more in a tighter space.
    Each successive rectangle appears on top of the previous ones.
    """
    rect_width = 1.0   # Wider horizontally
    rect_height = 0.6  # Shorter vertically
    overlap = 0.8       # 80% overlap
    vertical_offset = 0.1  # Small vertical shift per rectangle

    spacing = rect_width * (1 - overlap)  # 0.2

    # Calculate starting position
    total_width = num_rects * spacing + rect_width * (1 - overlap)
    start_x = -total_width / 2
    y = 0.5

    for i in range(num_rects):
        x = start_x + i * spacing
        current_y = y - (i / 15.0)  # Adjusted to fit within y-limits

        rect = patches.Rectangle(
            (x - rect_width / 2, current_y - rect_height / 2),
            rect_width,
            rect_height,
            linewidth=1,
            edgecolor='black',
            facecolor=random_colors[i],
            zorder=i  # Ensure each new rectangle is on top
        )
        ax.add_patch(rect)

    # Set limits and aspect
    ax.set_xlim(start_x - rect_width, start_x + num_rects * spacing + rect_width)
    ax.set_ylim(-1.5, 1)  # Increased lower limit to -1.5 to accommodate rectangles
    ax.set_aspect('equal')
    ax.set_title('Random Scene Order')
    ax.axis('off')

def main():
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Get tab10 colors
    cmap = plt.get_cmap('tab10')
    tab10_colors = [cmap(i) for i in range(10)]

    # Create Structured Order Figure
    fig_structured, ax_structured = plt.subplots(figsize=(6,6))
    create_structured_order(ax_structured, tab10_colors)
    
    # Add colorbar to Structured Order figure with centered labels
    # Define boundaries and normalization for discrete color mapping
    boundaries = np.arange(-0.5, 10.5, 1)  # 10 bins for 10 scenes
    norm = BoundaryNorm(boundaries, cmap.N)

    sm = cm.ScalarMappable(cmap='tab10', norm=norm)
    sm.set_array([])  # Only needed for older matplotlib versions

    # Add colorbar with ticks at integer positions
    cbar = fig_structured.colorbar(sm, ax=ax_structured, ticks=range(10), pad=0.1)
    cbar.set_label("Scene identity")

    # Set tick labels to be centered on each color
    cbar.set_ticks(range(10))
    cbar.set_ticklabels([f"{i+1}" for i in range(10)])  # Labeling scenes as 1-10

    # Save the Structured Order figure
    structured_filename = "structured_scene_order.pdf"
    fig_structured.savefig(structured_filename, bbox_inches='tight', dpi=300)
    print(f"Structured scene order figure saved as '{structured_filename}'.")

    # Create Random Order Figure with specific scene arrangement
    # First 10 scenes: 1 through 10 in colorbar order
    # Next 10 scenes: Another random permutation of 1 through 10
    scenes = list(range(10))  # Scenes 0-9 correspond to Scenes 1-10
    first_order = scenes.copy()  # Scenes 0-9 in order
    second_order = random.sample(scenes, 10)  # Random permutation of scenes 0-9
    random_colors = [tab10_colors[i] for i in first_order + second_order]

    fig_random, ax_random = plt.subplots(figsize=(10,10))  # Adjusted figure size to 10x10
    create_random_order(ax_random, random_colors, num_rects=20)
    
    # Save the Random Order figure
    random_filename = "random_scene_order.pdf"
    fig_random.savefig(random_filename, bbox_inches='tight', dpi=300)
    print(f"Random scene order figure saved as '{random_filename}'.")

    # Optionally, display the figures
    # Uncomment the following line if you want to display the figures
    # plt.show()

if __name__ == "__main__":
    main()

