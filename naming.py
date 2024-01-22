import os

folder_path = "C:\\Users\\GTS\\Desktop\\deep learning\\Archaeological Sites Project\\ajloun_images"  # Change this to the actual path of your folder

# Get the list of image files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', 'webp'))]

# Rename the files with sequential numbers
for i, old_name in enumerate(image_files, start=1):
    # Build the new name with a padded number
    new_name = f"{i:03d}.jpg"  # Adjust the extension as needed
    
    # Create the full paths for the old and new names
    old_path = os.path.join(folder_path, old_name)
    new_path = os.path.join(folder_path, new_name)
    
    # Check if the new name already exists
    if os.path.exists(new_path):
        print(f"File '{new_path}' already exists. Skipping...")
    else:
        # Rename the file
        os.rename(old_path, new_path)

print("Image renaming completed.")
