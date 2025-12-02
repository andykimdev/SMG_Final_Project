import matplotlib.pyplot as plt
import numpy as np

# 1. Define the Data
data = {
    'Breast Cancer': 1084,
    'Non-Small Cell Lung Cancer': 1053,
    'Esophagogastric Cancer': 622,
    'Colorectal Cancer': 594,
    'Glioblastoma': 592,
    'Endometrial Cancer': 586,
    'Ovarian Epithelial Tumor': 585,
    'Head and Neck Cancer': 523,
    'Glioma': 514,
    'Renal Clear Cell Carcinoma': 512,
    'Thyroid Cancer': 500,
    'Prostate Cancer': 494,
    'Melanoma': 448,
    'Bladder Cancer': 411,
    'Hepatobiliary Cancer': 372,
    'Renal Non-Clear Cell Carcinoma': 348,
    'Cervical Cancer': 297,
    'Sarcoma': 255,
    'Leukemia': 200,
    'Pancreatic Cancer': 184,
    'Pheochromocytoma': 147,
    'Thymic Epithelial Tumor': 123,
    'Adrenocortical Carcinoma': 92,
    'Pleural Mesothelioma': 87,
    'Non-Seminomatous Germ Cell Tumor': 86,
    'Ocular Melanoma': 80,
    'Seminoma': 63,
    'Mature B-Cell Neoplasms': 48,
    'Cholangiocarcinoma': 36,
    'Miscellaneous Neuroepithelial Tumor': 31
}

labels = list(data.keys())
sizes = list(data.values())
total = sum(sizes)

# 2. Create the Plot
fig, ax = plt.subplots(figsize=(14, 10))

# Generate the pie chart
# Note: autopct is REMOVED here to prevent the overlap
wedges, texts = ax.pie(
    sizes, 
    startangle=90,
    labels=None, # We will add labels manually
)

# 3. Manually place Combined Labels (Name + %)
threshold = 2.0  # Only label slices > 2%
for i, p in enumerate(wedges):
    # Calculate angles
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    
    # Calculate coordinates for the text
    # 0.75 puts it nicely in the middle of the slice (0 to 1 radius)
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    
    # Calculate percentage
    percent = (sizes[i] / total) * 100
    
    # Skip tiny slices
    if percent < threshold:
        continue
    
    # Combine Label and Percent into one string with a line break (\n)
    label_text = f"{labels[i]}\n({percent:.1f}%)"
    
    # Rotate text to align with the slice
    rotation = ang if x > 0 else ang + 180
    
    # Add the text
    ax.text(x*0.75, y*0.75, label_text, fontsize=8, 
            ha='center', va='center', color='white', weight='bold', rotation=rotation)

# 4. Create the Side Legend (for completeness)
sorted_indices = np.argsort(sizes)[::-1]
sorted_labels = [f"{labels[i]}: {sizes[i]} ({(sizes[i]/total)*100:.1f}%)" for i in sorted_indices]
sorted_handles = [wedges[i] for i in sorted_indices]

ax.legend(
    sorted_handles, 
    sorted_labels,
    title="Cancer Types",
    loc="center left",
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=9
)

ax.set_title("Cancer Type Distribution (Combined Labels)")
plt.tight_layout()
plt.show()