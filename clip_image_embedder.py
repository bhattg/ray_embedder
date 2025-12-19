import os
import ray
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

############################################
# Config
############################################
IMAGENET_ROOT = "/data/gbhatt2/imagenet_1k_train"  # or val
OUTPUT_DIR = "/data/gbhatt2/imagenet_1k_train_embeddings_clip"  # or val

MODEL_NAME = "openai/clip-vit-large-patch14"
BATCH_SIZE = 1024
NUM_GPUS_PER_WORKER = 4
NUM_CPUS_PER_WORKER = 8

############################################
# Utils: list ImageNet samples
############################################
def list_imagenet_samples(root):
    samples = []
    for cls in sorted(os.listdir(root)):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append({
                    "class": cls,
                    "image_path": os.path.join(cls_path, fname)
                })
    return samples

############################################
# Ray GPU embedding worker
############################################
class CLIPImageEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.model = CLIPModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16
        ).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def __call__(self, batch):
        # Load images
        images = []
        for path in batch["image_path"]:
            img = Image.open(path).convert("RGB")
            images.append(img)

        # Processor (images only — no text)
        inputs = self.processor(
            images=images,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract image embeddings
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features.float().cpu().numpy()

        return {
            "class": batch["class"],
            "image_path": batch["image_path"],
            "embedding": list(image_features)
        }

############################################
# Main
############################################
def main():
    ray.init()

    print("Listing ImageNet samples...")
    samples = list_imagenet_samples(IMAGENET_ROOT)
    print(f"Found {len(samples)} images")

    ds = ray.data.from_items(samples)

    ds = ds.map_batches(
        CLIPImageEmbedder,
        batch_size=BATCH_SIZE,
        batch_format="pandas",
        concurrency= (1, 4),
        num_gpus=1,
        num_cpus=8,
    )

    print("Writing Parquet...")
    ds.write_parquet(
        OUTPUT_DIR,
        compression="zstd"
    )

    print("Done.")

if __name__ == "__main__":
    main()
