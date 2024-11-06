from models import myresnet2
import torch
from torchvision.transforms import transforms
import torch.nn as nn
from mydataset import Nutrition_RGBD
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from collections import OrderedDict

test_transform = transforms.Compose([
    transforms.Resize((320, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
data_root = "/icislab/volume1/swj/nutrition/nutrition5k/nutrition5k_dataset"

nutrition_rgbd_ims_root = os.path.join(data_root, 'imagery')
nutrition_test_txt = os.path.join(data_root, 'imagery', 'rgbd_test_processed.txt')  # depth_color.png
nutrition_test_rgbd_txt = os.path.join(data_root, 'imagery', 'rgb_in_overhead_test_processed.txt')  # rbg.png

testset = Nutrition_RGBD(nutrition_rgbd_ims_root, nutrition_test_rgbd_txt, nutrition_test_txt, transform=test_transform)

test_loader = DataLoader(testset,
                         batch_size=4,
                         shuffle=False,
                         num_workers=1,
                         pin_memory=True
                         )

device = torch.device("cuda:5" if torch.cuda.is_available() else 'cpu')

net_rgb = myresnet2.resnet101(rgbd=True)
net_depth = myresnet2.resnet101(rgbd=True)
net_cat = myresnet2.Resnet101_concat(4 )

checkpoint_path = '/icislab/volume1/swj/nutrition/saved/new/regression_nutrition_rgbd_resnet101_resnet101_food2k_pretrained_fpn_multifusion_w_5scales_new/ckpt_best.pth'
models_state_dict = torch.load(checkpoint_path)

net_rgb.load_state_dict(models_state_dict["net"])

new_state_dict_depth = OrderedDict()
for k, v in models_state_dict['net_d'].items():
    name = k[7:] if k.startswith('module') else k
    new_state_dict_depth[name] = v
missing_keys, _ = net_depth.load_state_dict(new_state_dict_depth)

new_state_dict_cat = OrderedDict()
for k, v in models_state_dict['net_cat'].items():
    name = k[7:] if k.startswith('module') else k
    new_state_dict_cat[name] = v
net_cat.load_state_dict(new_state_dict_cat)

net_rgb.to(device)
net_depth.to(device)
net_cat.to(device)

net_rgb.eval()
net_depth.eval()
net_cat.eval()

criterion = nn.L1Loss()

epoch_iterator = tqdm(test_loader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
# epoch_iterator=test_loader
test_loss = 0
calories_loss = 0
mass_loss = 0
fat_loss = 0
carb_loss = 0
protein_loss = 0

with torch.no_grad():
    for batch_idx, x in enumerate(epoch_iterator):
        inputs = x[0].to(device)
        total_calories = x[2].to(device).float()
        total_mass = x[3].to(device).float()
        total_fat = x[4].to(device).float()
        total_carb = x[5].to(device).float()
        total_protein = x[6].to(device).float()
        inputs_rgbd = x[7].to(device)

        p2, p3, p4, p5 = net_rgb(inputs)
        outputs_rgbd = net_depth(inputs_rgbd)
        d2, d3, d4, d5 = outputs_rgbd
        outputs = net_cat([p2, p3, p4, p5], [d2, d3, d4, d5])

        calories_total_loss = total_calories.shape[0] * criterion(outputs[0],
                                                                  total_calories) / total_calories.sum().item()
        mass_total_loss = total_calories.shape[0] * criterion(outputs[1], total_mass) / total_mass.sum().item()
        fat_total_loss = total_calories.shape[0] * criterion(outputs[2], total_fat) / total_fat.sum().item()
        carb_total_loss = total_calories.shape[0] * criterion(outputs[3], total_carb) / total_carb.sum().item()
        protein_total_loss = total_calories.shape[0] * criterion(outputs[4], total_protein) / total_protein.sum().item()

        loss = calories_total_loss + mass_total_loss + fat_total_loss + carb_total_loss + protein_total_loss

        test_loss += loss.item()
        calories_loss += calories_total_loss.item()
        mass_loss += mass_total_loss.item()
        fat_loss += fat_total_loss.item()
        carb_loss += carb_total_loss.item()
        protein_loss += protein_total_loss.item()

        epoch_iterator.set_description(
                    "loss=%2.5f | calorieloss=%2.5f | massloss=%2.5f| fatloss=%2.5f | carbloss=%2.5f | proteinloss=%2.5f " % (test_loss/(batch_idx+1), calories_loss/(batch_idx+1), mass_loss/(batch_idx+1), fat_loss/(batch_idx+1), carb_loss/(batch_idx+1),protein_loss/(batch_idx+1))
                )