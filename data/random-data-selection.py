import random
import shutil
import os

# Generate 99 random numbers between 1 and 859
numbers = [random.randint(1, 859) for _ in range(100)]

print("Generated IDs:", numbers)

# Ensure the output folder exists
os.makedirs("selected_randoms-100", exist_ok=True)

for num in numbers:
    src = f"daily/ID_{num}.csv"
    dst = f"selected_randoms-100/ID_{num}.csv"

    # Copy file if it exists
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied: {src} → {dst}")
    else:
        print(f"Missing file: {src}")

