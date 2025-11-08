import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
from model import CASNet

IMAGE_SIZE = 256
SAVE_DIR = "Test_Result"

class Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(self.files[idx])

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

test_dataset = Dataset("BSD68", transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_nf = 1
model = CASNet(32, img_nf, 1).to(device)    
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

os.makedirs(SAVE_DIR, exist_ok=True)

def norm(x):
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x

def psnr(img1, img2):
    mse = torch.nn.functional.mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(float("inf"))
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
avg_psnr = 0
with torch.no_grad():
    for img, name in test_dataloader:   
        img = img.to(device)
        saliency, y, x0 = model(img, 0.1, True)
        output = model(img, 0.1)
        psnr_val = psnr(norm(output), norm(img))
        avg_psnr += psnr_val
        base_name = os.path.splitext(name[0])[0]
        out_dir = os.path.join(SAVE_DIR, base_name, f"psnr_{psnr_val}")
        os.makedirs(out_dir, exist_ok=True)
        save_image(norm(img), os.path.join(out_dir, f"{base_name}_input.png"))
        save_image(norm(saliency), os.path.join(out_dir, f"{base_name}_saliency.png"))
        save_image(norm(y), os.path.join(out_dir, f"{base_name}_undersampled.png"))
        save_image(norm(x0), os.path.join(out_dir, f"{base_name}_initialized.png"))
        save_image(norm(output), os.path.join(out_dir, f"{base_name}_reconstructed.png"))

print(f"avg_psnr: {avg_psnr/len(test_dataset):.2f}")
