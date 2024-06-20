import torch
from torchvision import models, transforms
import torch.nn as nn


class ClsHead():

    device = 0
    model = models.resnet152(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(2048, 2),
        nn.Softmax(dim=1)
    )

    def load_model_cls(self):
        if torch.cuda.is_available():
            state_dict = torch.load('./weights/Loo_1_17.pth')
        else:
            state_dict = torch.load('./weights/Loo_1_17.pth',map_location=torch.device('cpu'))

        new_dict = state_dict.copy()
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in new_dict.items()})

        with torch.no_grad():
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = nn.DataParallel(self.model)
            model.to(self.device)
            model.eval()
            print('Done loading model_cls')

    def detectpolyp(self,img_cls):

        inputs = img_cls.unsqueeze(0).to(self.device)
        outputs = self.model(inputs)
        ret, predictions = torch.max(outputs.data, 1)

        if torch.cuda.is_available():
            ret = ret.cpu()
            predictions = predictions.cpu()

        ret = float(ret.numpy())
        predictions=int(predictions.numpy())
        # print(predictions,ret)

        return predictions,ret








