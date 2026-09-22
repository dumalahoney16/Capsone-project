"""
create_sample_data.py
==============================================================================
Synthetic Retinal Fundus Sample Generator for Student Experimentation.

Purpose:
--------
Generates representative synthetic fundus images across all 5 Diabetic
Retinopathy severity classes and saves them into the respective subdirectories
under `dataset/`:
    - dataset/No_DR/          (Class 0: Normal retina, healthy vessels, optic disc)
    - dataset/Mild/           (Class 1: Isolated microaneurysms)
    - dataset/Moderate/       (Class 2: Multiple microaneurysms, mild exudates)
    - dataset/Severe/         (Class 3: Blot hemorrhages, cotton wool spots)
    - dataset/Proliferate_DR/ (Class 4: Neovascularization, severe hemorrhages)

Why this is useful:
-------------------
Downloading the full APTOS 2019 Kaggle dataset (~8.8 GB) can be slow or
impractical on low-bandwidth student connections. This script enables students
to verify the full pipeline (training, evaluation, Grad-CAM, and Streamlit app)
immediately on their local machine before or alongside using the full Kaggle data.

Usage:
------
    python src/create_sample_data.py
==============================================================================
"""

import os
import math
import random
from PIL import Image, ImageDraw, ImageFilter

CLASSES = {
    "No_DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Proliferate_DR": 4
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
SAMPLES_PER_CLASS = 16  # Generates 16 samples per class (80 total) for fast verification


def generate_synthetic_fundus(class_idx: int, size: int = 224) -> Image.Image:
    """
    Creates a synthetic retinal fundus photograph of dimension (size, size).
    Simulates:
      - Dark circular ocular aperture with black background
      - Orange-red retinal fundus background with radial vignetting
      - Optic disc (yellowish circular structure)
      - Retinal vascular tree branching out from optic disc
      - Severity-dependent pathological lesions (microaneurysms, exudates, hemorrhages)
    """
    # 1. Base black image
    img = Image.new("RGB", (size, size), (5, 5, 5))
    draw = ImageDraw.Draw(img)

    center_x = size // 2
    center_y = size // 2
    radius = int(size * 0.46)

    # 2. Draw circular fundus mask with warm orange-red retinal gradient
    for r in range(radius, 0, -2):
        factor = r / radius
        # Central fundus is brighter reddish-orange; periphery is darker
        red = int(170 + 45 * (1.0 - factor))
        green = int(55 + 35 * (1.0 - factor))
        blue = int(20 + 20 * (1.0 - factor))
        draw.ellipse(
            [center_x - r, center_y - r, center_x + r, center_y + r],
            fill=(red, green, blue)
        )

    # 3. Draw Optic Disc (yellowish-cream ellipse on nasal side)
    od_x = center_x - int(radius * 0.45)
    od_y = center_y + random.randint(-15, 15)
    od_radius = int(size * 0.08)
    draw.ellipse(
        [od_x - od_radius, od_y - od_radius, od_x + od_radius, od_y + od_radius],
        fill=(245, 215, 140)
    )
    # Optic cup (central depression)
    draw.ellipse(
        [od_x - od_radius // 2, od_y - od_radius // 2, od_x + od_radius // 2, od_y + od_radius // 2],
        fill=(255, 235, 175)
    )

    # 4. Draw Retinal Blood Vessels (branching dark reddish-brown curves)
    vessel_color = (105, 20, 10)
    for _ in range(6):
        curr_x, curr_y = od_x, od_y
        angle = random.uniform(-math.pi / 2, math.pi / 2)
        step_len = random.randint(4, 7)
        for _ in range(12):
            next_x = curr_x + int(step_len * math.cos(angle))
            next_y = curr_y + int(step_len * math.sin(angle))
            # Keep within fundus circle
            if (next_x - center_x) ** 2 + (next_y - center_y) ** 2 < (radius - 10) ** 2:
                draw.line([(curr_x, curr_y), (next_x, next_y)], fill=vessel_color, width=random.choice([1, 2]))
                curr_x, curr_y = next_x, next_y
                angle += random.uniform(-0.35, 0.35)

    # 5. Add Simulated Lesions based on Diabetic Retinopathy Severity Grade
    # --------------------------------------------------------------------
    # Class 0: No DR -> No lesions
    # Class 1: Mild DR -> A few tiny red dots (microaneurysms)
    if class_idx >= 1:
        num_ma = random.randint(4, 10) if class_idx == 1 else random.randint(8, 20)
        for _ in range(num_ma):
            lx = center_x + random.randint(-int(radius * 0.7), int(radius * 0.7))
            ly = center_y + random.randint(-int(radius * 0.7), int(radius * 0.7))
            if (lx - center_x) ** 2 + (ly - center_y) ** 2 < (radius - 15) ** 2:
                draw.ellipse([lx - 1, ly - 1, lx + 1, ly + 1], fill=(120, 0, 0))

    # Class 2: Moderate DR -> Hard exudates (bright yellow flecks) + more microaneurysms
    if class_idx >= 2:
        num_exudates = random.randint(5, 15) if class_idx == 2 else random.randint(15, 30)
        for _ in range(num_exudates):
            lx = center_x + random.randint(-int(radius * 0.6), int(radius * 0.6))
            ly = center_y + random.randint(-int(radius * 0.6), int(radius * 0.6))
            if (lx - center_x) ** 2 + (ly - center_y) ** 2 < (radius - 15) ** 2:
                draw.rectangle([lx, ly, lx + random.randint(2, 4), ly + random.randint(2, 4)], fill=(245, 240, 150))

    # Class 3: Severe DR -> Blot hemorrhages (larger dark-red patches) + cotton-wool spots (fluffy white)
    if class_idx >= 3:
        num_hemorrhages = random.randint(4, 10)
        for _ in range(num_hemorrhages):
            lx = center_x + random.randint(-int(radius * 0.65), int(radius * 0.65))
            ly = center_y + random.randint(-int(radius * 0.65), int(radius * 0.65))
            if (lx - center_x) ** 2 + (ly - center_y) ** 2 < (radius - 15) ** 2:
                r_hem = random.randint(3, 6)
                draw.ellipse([lx - r_hem, ly - r_hem, lx + r_hem, ly + r_hem], fill=(95, 5, 5))

        # Soft cotton-wool spots
        for _ in range(3):
            lx = center_x + random.randint(-int(radius * 0.5), int(radius * 0.5))
            ly = center_y + random.randint(-int(radius * 0.5), int(radius * 0.5))
            draw.ellipse([lx - 4, ly - 3, lx + 4, ly + 3], fill=(220, 220, 210))

    # Class 4: Proliferative DR -> Neovascularization (tangle of fine abnormal vessels) + extensive hemorrhages
    if class_idx == 4:
        for _ in range(4):
            neo_x = center_x + random.randint(-int(radius * 0.5), int(radius * 0.5))
            neo_y = center_y + random.randint(-int(radius * 0.5), int(radius * 0.5))
            for _ in range(8):
                dx = random.randint(-8, 8)
                dy = random.randint(-8, 8)
                draw.line([(neo_x, neo_y), (neo_x + dx, neo_y + dy)], fill=(130, 15, 15), width=2)
                neo_x += dx // 2
                neo_y += dy // 2

    # Smooth the image slightly to mimic optical fundus camera blur
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    return img


def main():
    print("=" * 70)
    print("Diabetic Retinopathy - Synthetic Fundus Sample Generator")
    print("=" * 70)
    print(f"Target Directory: {OUTPUT_DIR}")
    print(f"Generating {SAMPLES_PER_CLASS} sample images per class...\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_created = 0

    for class_name, class_idx in CLASSES.items():
        folder_path = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(folder_path, exist_ok=True)

        for i in range(1, SAMPLES_PER_CLASS + 1):
            img = generate_synthetic_fundus(class_idx=class_idx, size=224)
            filename = f"sample_{class_name.lower()}_{i:03d}.png"
            file_path = os.path.join(folder_path, filename)
            img.save(file_path, "PNG")
            total_created += 1

        print(f"  [+] Created {SAMPLES_PER_CLASS:02d} images in: dataset/{class_name}")

    print("\n" + "=" * 70)
    print(f"Success! Total {total_created} sample fundus images generated.")
    print("The project is now ready for training, evaluation, Grad-CAM, and Streamlit.")
    print("Note: To use the real APTOS 2019 dataset, place Kaggle images in these folders.")
    print("=" * 70)


if __name__ == "__main__":
    main()
