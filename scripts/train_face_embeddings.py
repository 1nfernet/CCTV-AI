"""
train_face_embeddings.py – Scan data/faces/ and generate known face encodings.

Directory layout expected:

    data/faces/
        Sister/
            photo1.jpg
            photo2.jpg
        Brother/
            photo1.jpg
        ...

Each sub-folder name becomes the person's label.

Usage:
    python scripts/train_face_embeddings.py
"""

import os
import sys
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import load_config, resolve_path


def main() -> None:
    import face_recognition
    from tqdm import tqdm

    config = load_config()
    face_cfg = config.get("face_recognition", {})

    faces_dir = resolve_path(face_cfg.get("known_faces_dir", "data/faces"))
    output_file = resolve_path(face_cfg.get("encodings_file", "models/face/known_faces.pkl"))

    print(f"📂 Scanning faces directory: {faces_dir}")

    known_encodings: list = []
    known_names: list[str] = []

    if not faces_dir.exists():
        print("❌ Faces directory does not exist. Create it and add sub-folders.")
        return

    person_dirs = [d for d in faces_dir.iterdir() if d.is_dir()]
    if not person_dirs:
        print("⚠️  No person sub-folders found in data/faces/.")
        print("   Create folders named after each person and add their photos.")
        return

    for person_dir in person_dirs:
        name = person_dir.name
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.jpeg")) + list(person_dir.glob("*.png"))

        if not images:
            print(f"  ⏭  {name}: no images found, skipping")
            continue

        print(f"  👤 Processing {name} ({len(images)} images)…")
        for img_path in tqdm(images, desc=f"   {name}", leave=False):
            image = face_recognition.load_image_file(str(img_path))
            encs = face_recognition.face_encodings(image)
            if encs:
                known_encodings.append(encs[0])
                known_names.append(name)
            else:
                print(f"     ⚠️  No face found in {img_path.name}")

    if not known_encodings:
        print("❌ No face encodings generated. Check your images.")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as fh:
        pickle.dump({"encodings": known_encodings, "names": known_names}, fh)

    print(f"\n✅ Saved {len(known_encodings)} encoding(s) for {len(set(known_names))} person(s)")
    print(f"   Output: {output_file}")


if __name__ == "__main__":
    main()
