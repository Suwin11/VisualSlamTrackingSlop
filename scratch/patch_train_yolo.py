with open('train_yolo_obstacles.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = """    # Check and Copy Best Model
    best_pt = os.path.join("runs", "train_fibo", "obstacle_wall", "weights", "best.pt")
    if os.path.exists(best_pt):"""

new_block = """    # Check and Copy Best Model
    cand_paths = [
        os.path.join(str(getattr(results, "save_dir", "")), "weights", "best.pt"),
        os.path.join("runs", "train_fibo", "obstacle_wall", "weights", "best.pt"),
        os.path.join(os.path.dirname(os.path.dirname(data_yaml)), "runs", "detect", "runs", "train_fibo", "obstacle_wall", "weights", "best.pt"),
        r"C:\\Users\\suwin\\Documents\\FIBO\\YOLO\\runs\\detect\\runs\\train_fibo\\obstacle_wall\\weights\\best.pt",
    ]
    best_pt = next((p for p in cand_paths if os.path.exists(p)), None)
    if best_pt is not None:"""

assert old_block in content, 'old_block not found'
content = content.replace(old_block, new_block, 1)

with open('train_yolo_obstacles.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('train_yolo_obstacles.py updated successfully.')
