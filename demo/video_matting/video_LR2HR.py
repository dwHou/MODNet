import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from src.models.modnet import MODNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

print('Load pre-trained MODNet...')
pretrained_ckpt = './pretrained/modnet_webcam_portrait_matting.ckpt'
modnet = MODNet(backbone_pretrained=False)
modnet = nn.DataParallel(modnet).cuda()
modnet.load_state_dict(torch.load(pretrained_ckpt))
modnet.eval()

print('Init WebCam...')
cap = cv2.VideoCapture("./demo/input/align2.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

bg_np = cv2.imread("./demo/input/bg.png")
bg_np = cv2.resize(bg_np, (1194, 672), cv2.INTER_AREA)
background_np = bg_np[:, 0:960, :]

# 960x672
# 960x832(660)

cv2.imwrite("./demo/input/background.png", background_np)
print('Start matting...')

rval = True
filename = 1

while rval:

    rval, frame_np = cap.read()
    frame_np = frame_np[14:846, 440:1400, :]
    # 960x832 → LR mask 480x416

    frame_LR_np = cv2.resize(frame_np, (480, 416), cv2.INTER_AREA)
    frame_LR_PIL = Image.fromarray(frame_LR_np)
    frame_LR_tensor = torch_transforms(frame_LR_PIL)
    frame_LR_tensor = frame_LR_tensor[None, :, :, :].cuda()

    with torch.no_grad():
        _, _, matte_tensor = modnet(frame_LR_tensor, inference=True)

    matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
    matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
    matte_head = cv2.resize(matte_np, (960, 832), cv2.INTER_AREA)


    kernel1 = np.ones((8, 8), np.uint8)
    kernel2 = np.ones((16, 16), np.uint8)
    dilation_head = cv2.dilate(matte_head*255, kernel1)
    erosion_head = cv2.erode(dilation_head, kernel2)
    matte_head = erosion_head / 255

    x = matte_head * frame_np

    matte_zeros = np.zeros(background_np.shape)
    matte_zeros[12:672, :, :] = matte_head[:660, :, :] # 480 x 384

    x_zeros = np.zeros(background_np.shape)
    x_zeros[12:672, :, :] = x[:660, :, :]

# 992x672
# 960x800(660)


    fg_np = x_zeros + (1 - matte_zeros) * background_np
    print('matte_np {}, fg_np {}, frame_np {}'.format(matte_np.shape, fg_np.shape, frame_np.shape))
    view_np = np.uint8(fg_np)

    path = './erosion_HR/' + str(filename) + '.png'
    cv2.imwrite(path, view_np)
    filename += 1

cap.release()
print('Exit...')

