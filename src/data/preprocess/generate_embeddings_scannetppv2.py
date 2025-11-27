from transformers import AutoTokenizer, SiglipTextModel
import os
import torch


def main():

    metadata_root = "data/scannetppv2/metadata"
    classes_file = os.path.join(metadata_root, "top100.txt")

    embed_location = os.path.join(metadata_root, "text_embeddings.pth")

    with open(classes_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f]
    ckpt = "google/siglip2-base-patch16-512"

    model = SiglipTextModel.from_pretrained(ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    batch_size = 32
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(classes), batch_size):
            batch_classes = classes[i:i+batch_size]
            inputs = tokenizer(
                batch_classes,
                padding="max_length",
                truncation=False,
                max_length=64,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            pooled_output = outputs.pooler_output.cpu()  # move to CPU
            all_embeddings.append(pooled_output)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(all_embeddings, embed_location)


if __name__ == '__main__':
    main()
