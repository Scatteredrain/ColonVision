import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
from lib.pvt import PVT_TSN
import torchvision.transforms as transforms
from PIL import Image



class segmentation():

    Model = PVT_TSN()
    testsize = 352

    def rgb_loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    def load_model(self):



        save_path: str = './Pre_Results/'

        pth_path = './weights/PVT-SSP+TSN44-best.pth'
        # test_txt_file = 'E:/python-practice/DaChuang_eval_code/test_all.txt'
        # save_mask = 'store_true'
        # show_gt_and_pre = 'store_true'

        # color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)



        if torch.cuda.device_count() >= 1:
            self.Model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(pth_path).items()})
            torch.cuda.set_device(0)
            self.Model.cuda()
            self.Model.eval()

        else:
            self.Model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(pth_path,map_location=torch.device('cpu')).items()})
            self.Model.eval()

        print('Done loading model')

    def segment(self,path):

        gt_transform = transforms.ToTensor()
        image0 = self.rgb_loader(path)

        transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])



        image_cls = transform(image0)
        image = image_cls.unsqueeze(0)
        ori_image = gt_transform(image0)


        if torch.cuda.device_count() >= 1:
            image = image.cuda()
            P1, P2, _, _ = self.Model(image, step=3, cur_epoch= 5)
            res = F.interpolate(P1 + P2, size=ori_image.shape[1:3], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()

            res_cls = F.interpolate(P1 + P2, size=self.testsize, mode='bilinear',
                                    align_corners=False).sigmoid().data.cpu().squeeze()

            print('model running on GPU')

        else:
            P1, P2, _, _ = self.Model(image, step=3, cur_epoch=5)
            res = F.interpolate(P1 + P2, size=ori_image.shape[1:3], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.numpy().squeeze()

            res_cls = F.interpolate(P1 + P2, size=self.testsize, mode='bilinear',
                                    align_corners=False).sigmoid().data.cpu().squeeze()

            print('model running on CPU')

        ori_image = ori_image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        ori_image = cv2.cvtColor(np.asarray(ori_image), cv2.COLOR_RGB2BGR)
        ori_image_copy = np.copy(ori_image)

        res = (res - res.min()) / (res.max() - res.min() + 1e-8)  # [0,1]之间
        res[res >= 0.5] = 255.0
        res[res < 0.5] = 0
        res = res.astype('uint8')

        res_cls = (res_cls - res_cls.min()) / (res_cls.max() - res_cls.min() + 1e-8)  # [0,1]之间
        res_cls[res_cls >= 0.5] = 255.0
        res_cls[res_cls < 0.5] = 0
        res_cls=res_cls.unsqueeze(0)
        img_cls = torch.cat((image_cls,res_cls),0)


        # 显示图像
        # img1 = np.array(image0)
        # res = np.expand_dims(res,axis=2)
        # res1 = (res/255).astype('uint8')
        # img_4_final = img1 * res1


        # img_4 = np.concatenate((img1,res),axis=2)
        # if img_4.shape[2] == 4:
        #     alpha = img_4[:, :, 3]
        #     alpha = np.expand_dims(alpha, axis=2)
        #     alpha = (alpha/255).astype('uint8')
        #     img_4_final = img_4[:,:,:3] * alpha


        # cv2.imshow('img_rgba', img_4_final)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # plt.imshow(img_4_final)
        # plt.show()



        contours, _ = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ori_image, contours, -1, (0, 255, 0), 2)


        # ori_image1 = cv2.hconcat([ori_image_copy,ori_image])
        # cv2.imshow('image',ori_image1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return ori_image_copy, ori_image, img_cls


