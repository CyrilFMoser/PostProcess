from torch.utils.data import Dataset, DataLoader
import torch.cuda
from data.dataloader import SceneSplatDataset
from model.pointconv import PointConvDensityClsSsg
from tqdm import tqdm
import warnings
from torchsummary import summary

N_EPOCHS = 10


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    scene_names = ["13c3e046d7"]
    scene_root = "/mnt/g/Projects/RICS/PostProcess/data/scenes"
    feature_root = "/mnt/g/Projects/RICS/PostProcess/data/features"

    dataset = SceneSplatDataset(
        scene_names, scene_root, feature_root, device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print("setup dataloader")

    model = PointConvDensityClsSsg(feature_dim=dataset.feature_dim)

    summary(model, input_size=[(3, 100), (768, 100)])

    print("initialized model")
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

    model.train()
    print("Start training")

    for epoch in range(N_EPOCHS):
        print(f"Epoch: {epoch}")
        for batch_id, data in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):
            optimizer.zero_grad()

            xyz = data["coord"].permute(0, 2, 1)
            feat = data["feature"].permute(0, 2, 1)
            new_feat = model(xyz, feat)
            print("Finished inference run")
            exit()
        scheduler.step()


if __name__ == '__main__':
    main()
