from torch.utils.data import Dataset, DataLoader
import torch.cuda
from data.dataloader.scannetppv2_dataloader import ScanNetPPV2DataSet
from tqdm import tqdm
import warnings
from torchinfo import summary
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from external.PointTransformerV3.model import PointTransformerV3

N_EPOCHS = 10


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    scene_names = ["13c3e046d7_mini"]
    scene_root = "data/scenesplat/scannetppv2/scenes/train"
    feature_root = "data/scenesplat/scannetppv2/features/train"
    metadata_root = "data/scannetppv2/metadata"
    label_root = "data/scannetppv2/gaussianlabels"

    dataset = ScanNetPPV2DataSet(
        scene_names, scene_root, feature_root, metadata_root,label_root, device)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, collate_fn=dataset.collate_fn)

    k = dataset.num_classes 

    model = build_model(k)    

    # summary(model, input_data=(x_dict,))  # 1670470
    model.to(device)

    train(dataloader, model,k)


def train(dataloader: DataLoader, model,k):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-08,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.7)

    loss = nn.CrossEntropyLoss()

    model.train()
    num_params = sum(p.numel()
                     for p in model.parameters() if p.requires_grad)
    writer = SummaryWriter()


    print(f"Start training with {num_params} number of parameters")
    for epoch in range(N_EPOCHS):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9)
        epoch_loss = 0.0
        for batch_id, data in pbar:
            optimizer.zero_grad()
            new_feat = model(data)

            valid_mask = data["valid_mask"]
            labels = data["label"]

            valid_new_feat = new_feat.feat[valid_mask]
            valid_labels = labels[valid_mask]

            output = loss(valid_new_feat,valid_labels)

            output.backward()
            optimizer.step()

            epoch_loss += output.item()

            # Show running average loss in tqdm
            pbar.set_description(f"Epoch {epoch} | Loss {epoch_loss / (batch_id+1):.4f}")
            writer.add_scalar("train/loss", output.item(), epoch * len(dataloader) + batch_id)


        scheduler.step()

def build_model(k:int):
    

    model = PointTransformerV3(
        in_channels=k,
        stride=             (2, 2),
        enc_depths=         (2, 2, 2),
        enc_channels=       (100, 200, 400),
        enc_num_head=       (2, 4, 8),
        enc_patch_size=     (1024, 1024, 1024),
        dec_depths=         (2, 2),
        dec_channels=       (100, 200),
        dec_num_head=       (4, 8),
        dec_patch_size=     (1024, 1024),
        enable_flash=False)
    return model


if __name__ == '__main__':
    main()
