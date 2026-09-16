# quick_test_6classes.py
import cv2
import numpy as np
import os

print("Creating test images with simulated damages...")

test_dir = "test_6classes"
os.makedirs(test_dir, exist_ok=True)

# 6 damage types
damage_types = [
    'dent',
    'scratch', 
    'crack',
    'glass_shatter',
    'lamp_broken',
    'tire_flat'
]

colors = {
    'dent': (0, 165, 255),      # Orange
    'scratch': (0, 0, 255),     # Red
    'crack': (255, 0, 0),       # Blue
    'glass_shatter': (255, 255, 0), # Cyan
    'lamp_broken': (255, 0, 255),   # Magenta
    'tire_flat': (0, 255, 0)    # Green
}

# Create test images
for i in range(len(damage_types)):
    # Create car image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img.fill(200)  # Light gray background
    
    # Draw car outline
    cv2.rectangle(img, (100, 150), (540, 350), (100, 100, 100), -1)  # Body
    cv2.rectangle(img, (200, 100), (440, 150), (100, 100, 100), -1)  # Roof
    cv2.circle(img, (200, 380), 30, (50, 50, 50), -1)  # Wheel
    cv2.circle(img, (440, 380), 30, (50, 50, 50), -1)  # Wheel
    
    # Add specific damage
    damage_type = damage_types[i]
    color = colors[damage_type]
    
    if damage_type == 'dent':
        cv2.ellipse(img, (320, 200), (60, 40), 0, 0, 360, color, -1)
    elif damage_type == 'scratch':
        cv2.line(img, (150, 250), (350, 250), color, 3)
        cv2.line(img, (350, 250), (490, 220), color, 3)
    elif damage_type == 'crack':
        points = np.array([[300, 120], [320, 110], [340, 125], [360, 115]], np.int32)
        cv2.polylines(img, [points], False, color, 2)
    elif damage_type == 'glass_shatter':
        cv2.circle(img, (320, 125), 25, color, 2)
        cv2.line(img, (295, 125), (345, 125), color, 1)
        cv2.line(img, (320, 100), (320, 150), color, 1)
    elif damage_type == 'lamp_broken':
        cv2.circle(img, (120, 200), 20, color, -1)
        cv2.line(img, (100, 180), (140, 220), (255, 255, 255), 2)
        cv2.line(img, (140, 180), (100, 220), (255, 255, 255), 2)
    elif damage_type == 'tire_flat':
        cv2.ellipse(img, (200, 380), (30, 20), 0, 0, 360, color, -1)
        cv2.line(img, (170, 380), (230, 380), (255, 255, 255), 2)
    
    # Add label
    cv2.putText(img, damage_type.replace('_', ' ').title(), (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(img, f"Test {i+1}", (50, 440), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save
    img_path = os.path.join(test_dir, f"{damage_type}.jpg")
    cv2.imwrite(img_path, img)
    print(f"Created: {img_path}")

print(f"\n✅ Created {len(damage_types)} test images in '{test_dir}/'")
print("\nNow analyze them:")
print(f"  python analyze_6classes.py --images {test_dir}/ --conf 0.1")