import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from model import Model
from parameter import get_parameters
import torchvision.transforms as transforms


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, _ in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class TextRecognition(nn.Module):
    def __init__(self, opt):
        super(TextRecognition, self).__init__()
        self.opt = opt
        self.converter = AttnLabelConverter(opt.sensitive_character)
        opt.num_class = len(self.converter.character)

        self.model = Model(opt).eval()
        self.model.load_state_dict(torch.load(
            opt.model_path, map_location='cpu'))
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, img):
        batch_size = img.size(0)

        self.text_for_pred = torch.LongTensor(
            batch_size, self.opt.batch_max_length + 1).fill_(0)
        self.length_for_pred = torch.IntTensor(
            [self.opt.batch_max_length] * batch_size)

        preds = self.model(img, self.text_for_pred, is_train=False)

        _, preds_index = preds.max(2)
        preds_str = self.converter.decode(preds_index, self.length_for_pred)
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)

        pred_list, confidence_score_list = [], []

        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]

            pred_max_prob = pred_max_prob[:pred_EOS]
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]

            pred_list.append(pred)
            confidence_score_list.append(confidence_score)

        return pred_list, confidence_score_list


def preprocess(img_path):

    img = Image.open(img_path).convert('RGB')
    imgH, imgW = 32, 100

    img = img.resize((imgW, imgH), Image.BICUBIC)

    toTensor = transforms.ToTensor()
    img = toTensor(img)
    img.sub_(0.5).div_(0.5)
    img = img.unsqueeze(0)

    return img


if __name__ == '__main__':

    opt = get_parameters()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img1 = preprocess('demo_image/demo_1.png')
    img2 = preprocess('demo_image/demo_3.png')
    img3 = preprocess('demo_image/demo_5.png')
    img4 = preprocess('demo_image/demo_7.png')

    img = torch.cat([img1, img2, img3, img4], dim=0)
    gray_img = (img[:, 0, :, :] * 299 + img[:, 1, :, :]
                * 587 + img[:, 2, :, :] * 114) / 1000
    gray_img.unsqueeze_(1)

    text_recognition = TextRecognition(opt).to(device)
    print(text_recognition.parameters)

    pred_list, confidence_score_list = text_recognition(gray_img.to(device))

    for pred, confidence_score in zip(pred_list, confidence_score_list):
        print('pred:  {}, confidence_score:  {}'.format(pred, confidence_score))
