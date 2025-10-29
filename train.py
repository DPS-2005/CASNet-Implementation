import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import CASNet

IMAGE_SIZE = 128

class Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

dataset = Dataset("TrainingImages", transform)
test_dataset = Dataset("BSD68", transform)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_nf = next(iter(dataloader)).to(device).shape[1]
model = CASNet(32, img_nf, 8).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 30

test_img = test_dataset[0].unsqueeze(0).to(device)

plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

os.makedirs("plots", exist_ok=True)

for epoch in range(1, num_epochs + 1):
    total_loss = 0
    model.train()

    for imgs in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}"):
        imgs = imgs.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, cs_ratio=0.5)
        loss = criterion(outputs, imgs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Total Loss: {total_loss:.4f}")

    # sample test
    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            output : torch.Tensor = model(test_img, cs_ratio=0.5)
            loss = criterion(output, test_img)
            output_img = output.squeeze().detach().to("cpu", non_blocking=True).numpy()
            target_img = test_img.squeeze().detach().to("cpu", non_blocking=True).squeeze().numpy()
            ax[0].cla()
            ax[0].imshow(target_img, cmap="gray")
            ax[0].set_title("Original")
            ax[1].cla()
            ax[1].imshow(output_img, cmap="gray")
            ax[1].set_title(f"Output--Loss: {loss:.5f}")
            path = f"plots/epoch_{epoch:03d}.png"
            plt.savefig(path)
            plt.pause(0.01)

