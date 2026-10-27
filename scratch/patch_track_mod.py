with open('yolo_mask_track.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_mod = '''    model_name = cfg["yolo_and_planner"]["yolo_model"]
    model_path = model_name
    if not os.path.exists(model_path):
        candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", model_name)
        if os.path.exists(candidate):
            model_path = candidate

    print(f"[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)'''

new_mod = '''    model_name = cfg["yolo_and_planner"]["yolo_model"]
    cand_paths = [
        model_name,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", model_name),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", model_name),
        os.path.join("models", model_name),
        os.path.join("pc", "models", model_name),
    ]
    model_path = next((p for p in cand_paths if os.path.exists(p)), model_name)
    print(f"[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    print(f"[INFO] Active Detection Classes: {model.names}")'''

assert old_mod in content, 'old_mod not found in yolo_mask_track.py'
content = content.replace(old_mod, new_mod, 1)

with open('yolo_mask_track.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('yolo_mask_track.py model loader updated successfully.')
