from torch.utils.data import Dataset, DataLoader
import torch.cuda
from data.dataloader import SceneSplatDataset
from model.pointconv import PointConvDensityClsSsg
from tqdm import tqdm
import warnings
from torchinfo import summary

N_EPOCHS = 10


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    scene_names = ["13c3e046d7"]
    scene_root = "/mnt/g/Projects/RICS/PostProcess/data/scenes"
    feature_root = "/mnt/g/Projects/RICS/PostProcess/data/features"
    text_embeddings_root = "/mnt/g/Projects/RICS/PostProcess/data/text_embeddings"

    dataset = SceneSplatDataset(
        scene_names, scene_root, feature_root, text_embeddings_root, device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = PointConvDensityClsSsg(feature_dim=dataset.feature_dim)

    N = 1670470
    M = 6589
    x_dict = {
        "coord": torch.rand(1, 3, N),
        "feature": torch.rand(1, 768, N),
        "voxel_feature": torch.rand(1, 768, M),
        "voxel_coord": torch.rand(1, 3, M),
        "voxel_to_gauss": []
    }

    summary(model, input_data=(x_dict,))  # 1670470
    model.to(device)

    train(dataloader, model)


def train(dataloader: DataLoader, model: PointConvDensityClsSsg):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-08,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.7)

    ignore_index = dataloader.dataset.ignore_index

    model.train()
    print("Start training")

    for epoch in range(N_EPOCHS):
        print(f"Epoch: {epoch}")
        for batch_id, data in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):
            optimizer.zero_grad()

            new_feat = model(data)

            B, F, N = new_feat.shape
            mask_valid = data["valid_mask"]        # [N]

            valid_feat_list = []
            for b in range(B):
                # select valid entries for this batch
                mask_no_ignore = data["segment_raw"][b,
                                                     :, 0] != ignore_index  # [N]
                mask = mask_valid & mask_no_ignore     # combine masks [N]
                mask = mask.squeeze(0)
                valid_feat_b = new_feat[b].view(N, F)[mask]   # [num_valid, F]
                valid_feat_list.append(valid_feat_b)

            # pad or stack as needed
            # shape depends on validity pattern
            valid_feat = torch.stack(valid_feat_list)
            valid_feat = valid_feat.permute(0, 2, 1)
            gt_feat = data["gaussian_text_embedding"]
            print(valid_feat.shape)
            print(gt_feat.shape)
            exit()
        scheduler.step()


if __name__ == '__main__':
    main()
