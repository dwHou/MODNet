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
cap = cv2.VideoCapture("./demo/input/realobm.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print('Start matting...')

# if cap.isOpened():
  #  rval, frame_np = cap.read()
# else:
  #  rval = False

rval = True
filename = 1

while rval:

    rval, frame_np = cap.read()
    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
    # frame_np = cv2.resize(frame_np, (512, 512), cv2.INTER_AREA)
    frame_np = cv2.resize(frame_np, (910, 512), cv2.INTER_AREA)
    frame_np = frame_np[:, 120:792, :]

    frame_PIL = Image.fromarray(frame_np)
    frame_tensor = torch_transforms(frame_PIL)
    frame_tensor = frame_tensor[None, :, :, :].cuda()

    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        _, _, matte_tensor = modnet(frame_tensor, inference=True)

    matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
    matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
    
    # 膨胀+腐蚀，使得边缘不那么硬，融入更自然。
    kernel1 = np.ones((8, 8), np.uint8)
    kernel2 = np.ones((12,12), np.uint8)
    dilation = cv2.dilate(matte_np*255, kernel1)
    erosion = cv2.erode(dilation, kernel2)
    matte_np = erosion/255
    
    
    green = np.zeros(frame_np.shape)
    green[:, :, 1] = 255.0
    
    fg_np = matte_np * frame_np + (1 - matte_np) * green
    # fg_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
     
    print('matte_np {}, fg_np {}, frame_np {}'.format(matte_np.shape, fg_np.shape, frame_np.shape))

    view_np = np.uint8(matte_np*255.0)
    # view_np = np.uint8(np.concatenate((frame_np, fg_np), axis=1))
    view_np = cv2.cvtColor(view_np, cv2.COLOR_RGB2BGR)

    path = './demo/out_real/' + str(filename) + '.png'
    cv2.imwrite(path, view_np)
    # cv2.imshow('MODNet - WebCam [Press \'Q\' To Exit]', view_np)
    filename += 1
    # if cv2.waitKey(1) & 0xFF == ord('q'):
      #  break
cap.release()
print('Exit...')
