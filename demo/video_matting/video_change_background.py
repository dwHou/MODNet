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

cap = cv2.VideoCapture("/apdcephfs/share_139366/devonnhou/matting/MODNet/demo/input/align_result.mp4")
bg = cv2.imread("/apdcephfs/share_139366/devonnhou/matting/MODNet/demo/input/background.png")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

print('Start matting...')

rval = True
filename = 1

while rval:

    rval, frame_np = cap.read()
    # frame_np = frame_np[14:846, 440:1400, :]
    # 960x832  to  LR mask 480x416

    frame_LR_np = cv2.resize(frame_np, (256, 256), cv2.INTER_AREA)
    #print(frame_np.shape)
    frame_LR_PIL = Image.fromarray(frame_LR_np)
    frame_LR_tensor = torch_transforms(frame_LR_PIL)
    frame_LR_tensor = frame_LR_tensor[None, :, :, :].cuda()

    with torch.no_grad():
        _, _, matte_tensor = modnet(frame_LR_tensor, inference=True)

    matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
    matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
    matte_np = cv2.resize(matte_np, (512, 512), cv2.INTER_AREA)

    matte_np = np.uint8(matte_np*255)
 

    frame_zero = bg.copy()
    print(frame_zero.shape)
    frame_zero[254:766, 260:772, :] = frame_np

    matte_zero = np.zeros(bg.shape) 
    matte_zero[254:766, 260:772, :] = matte_np1 / 255


    outimg = (matte_zero) * frame_zero + (1 - matte_zero) * bg 
    outpath = '/apdcephfs/share_139366/devonnhou/matting/MODNet/demo/output/align_result/' + str("%05d" % filename) + '.png'


    cv2.imwrite(outpath, outimg)

    print(filename)
    filename += 1

cap.release()
print('Exit...')

