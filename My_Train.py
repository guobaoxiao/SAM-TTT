import numpy as np
import matplotlib.pyplot as plt
import os
join = os.path.join
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import monai
import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn import BatchNorm2d
# from torchvision.models.mobilenetv2 import InvertedResidual
from NewD import newdropout

class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            ):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
                              dilation=dilation, groups=groups, bias=False)
        self.bn = BatchNorm2d(nOut)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

# set seeds
torch.manual_seed(2024)
np.random.seed(2024)

class NpzDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.npz_files = sorted(os.listdir(self.data_root))
        self.npz_data = [np.load(join(data_root, f)) for f in self.npz_files]
        # this implementation is ugly but it works (and is also fast for feeding data to GPU)
        # if your server has enough RAM
        # as an alternative, you can also use a list of npy files and load them one by one
        self.ori_gts = np.vstack([d['gts'] for d in self.npz_data])
        self.img_embeddings = np.vstack([d['img_embeddings'] for d in self.npz_data])
        self.boundary = np.vstack([d['boundary'] for d in self.npz_data])
        print(f"img_embeddings.shape={self.img_embeddings.shape}, ori_gts.shape={self.ori_gts.shape}, "
              f"boundary.shape={self.boundary.shape}")

    def __len__(self):
        return self.ori_gts.shape[0]

    def __getitem__(self, index):
        img_embed = self.img_embeddings[index]
        gt2D = self.ori_gts[index]
        boundary = self.boundary[index]
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        # convert img embedding, mask, bounding box to torch tensor
        return torch.tensor(img_embed).float(), torch.tensor(gt2D[None, :, :]).long(), torch.tensor(bboxes).float(),\
               torch.tensor(boundary[None, :, :]).long()

    # %% test dataset class and dataloader
npz_tr_path = 'dataset/COD_train'
work_dir = './work_dir_cod'
task_name = 'SAM-TTT'
# prepare SAM model
model_type = 'vit_b'
checkpoint = 'work_dir/SAM/sam_vit_b_01ec64.pth'
torch.cuda.device_count()
device = 'cuda:0'
model_save_path = join(work_dir, task_name)
os.makedirs(model_save_path, exist_ok=True)
sam_model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)

sam_model.train()
# Set up the optimizer, hyperparameter tuning will improve performance here
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

num_epochs = 300
losses = []
best_loss = 1e10
train_dataset = NpzDataset(npz_tr_path)
mask_threshold = 0.0
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
for epoch in range(num_epochs+1):
    epoch_loss = 0
    # train
    for step, (image_embedding, gt2D, boxes, boundary) in enumerate(tqdm(train_dataloader)):
        # do not compute gradients for image encoder(Just calculate it in pre grey rgb2D.py)
        # and prompt encoder
        with torch.no_grad():
            # convert box to 1024x1024 grid
            box_np = boxes.numpy()
            # boundary_np = boundary.numpy()
            sam_trans = ResizeLongestSide(sam_model.image_encoder.img_size)
            box = sam_trans.apply_boxes(box_np, (gt2D.shape[-2], gt2D.shape[-1]))
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            # boun_conv = nn.Conv2d(1, 1, (3, 3),
            #                       bias=False, padding=1).to(device)
            boundary = torch.as_tensor(boundary, dtype=torch.float, device=device)
            # boundary = boun_conv(boundary)
            image_embedding = torch.as_tensor(image_embedding, dtype=torch.float, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)
            # get prompt embeddings
            sparse_embeddings_box, dense_embeddings_box = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None
            )

            sparse_embeddings_boundary, dense_embeddings_boundary = sam_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=boundary
            )

        # conv_dropwaste = newdropout(channel=256).to(device)
        conv_dropwaste = nn.Sequential(
            ConvBNReLU(256, 256 * 2, ksize=1, pad=0),
            ConvBNReLU(256 * 2, 256 * 2),
            ConvBNReLU(256 * 2, 256 * 2),
            ConvBNReLU(256 * 2, 256 * 2),
            ConvBNReLU(256 * 2, 256 * 2),
            nn.Conv2d(256 * 2, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
        ).to(device)
        hybrid_embedding_1 = conv_dropwaste(image_embedding)
        high_frequency_1 = sam_model.DWT(hybrid_embedding_1)
        dense_embeddings_1, sparse_embeddings_1 = sam_model.ME(dense_embeddings_boundary, dense_embeddings_box,
                                                           high_frequency_1, sparse_embeddings_box, route=1)

        # route = 2, mamba
        high_frequency_2 = sam_model.DWT(image_embedding)
        dense_embeddings_2, sparse_embeddings_2 = sam_model.ME(dense_embeddings_boundary, dense_embeddings_box,
                                                           high_frequency_2, sparse_embeddings_box, route=2)

        dense_embeddings, sparse_embeddings = sam_model.routefuse(dense_embeddings_1, sparse_embeddings_1,
                                                                  dense_embeddings_2, sparse_embeddings_2)

        # predicted masks
        mask_predictions, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        loss = seg_loss(mask_predictions, gt2D.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= step
    losses.append(epoch_loss)
    print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
    # save the latest model checkpoint
    if epoch >= 200 and epoch % 10 == 0:
        torch.save(sam_model.state_dict(), join(model_save_path, str(epoch) + 'sam_model.pth'))
    # save the best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(sam_model.state_dict(), join(model_save_path, 'sam_model_best.pth'))
# plot loss
plt.plot(losses)
plt.title('Dice + Cross Entropy Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.show() # comment this line if you are running on a server
plt.savefig(join(model_save_path, 'train_loss.png'))
plt.close()