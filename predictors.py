import cv2
import numpy as np
import torch

from parameter import get_parameters
from render_standard_text import make_standard_text
from style_text_model import Generator
from text_recognition import TextRecognition


class StyleTextRenderPredictor():
    def __init__(self, opt, model_path, data_shape=[64, None]):

        self.model_path = model_path
        self.data_shape = data_shape
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = self.load_model()

        self.text_recognition = TextRecognition(opt).to(self.device)

    def load_model(self):
        G = Generator().to(self.device)

        # 多GPU机器训练的模型需要做map_location
        checkpoint = torch.load(self.model_path, map_location=self.device)
        G.load_state_dict(checkpoint)

        # 预测时应该调整为eval模式，否则图像质量非常差
        G.eval()
        return G

    def predict_single_image(self, style_img, text_corpus, font_path):
        h, w = style_img.shape[:2]
        text_img = make_standard_text(
            font_path, text_corpus, style_img.shape[:2])

        tensor_style_img = self.preprocess(style_img)

        gray_img = (tensor_style_img[:, 0, :, :] * 114 + tensor_style_img[:, 1, :, :]
                    * 587 + tensor_style_img[:, 2, :, :] * 299) / 1000
        gray_img = gray_img.unsqueeze(1)

        pred, confidence_score = self.text_recognition(
            gray_img.to(self.device))
        print('pred:  {}, confidence_score:  {}'.format(
            pred[0], confidence_score[0]))

        tensor_text_img = self.preprocess(text_img)

        o_t, o_b, o_f = self.generator(
            [tensor_text_img.to(self.device), tensor_style_img.to(self.device)])

        gray_img = (o_f[:, 0, :, :] * 114 + o_f[:, 1, :, :]
                    * 587 + o_f[:, 2, :, :] * 299) / 1000
        gray_img = gray_img.unsqueeze(1)
        pred, confidence_score = self.text_recognition(
            gray_img.to(self.device))
        print('pred:  {}, confidence_score:  {}'.format(
            pred[0], confidence_score[0]))

        to_shape = (w, h)
        o_t = self.postprocess(o_t, to_shape)
        o_b = self.postprocess(o_b, to_shape)
        o_f = self.postprocess(o_f, to_shape)

        return {
            "i_t": text_img,
            "o_t": o_t,
            "o_b": o_b,
            "o_f": o_f
        }

    def preprocess(self, img):
        h, w = img.shape[:2]

        ratio = self.data_shape[0] / h
        predict_h = self.data_shape[0]
        predict_w = round(int(w * ratio) / 8) * 8
        predict_scale = (predict_w, predict_h)  # w first for cv2
        new_img = cv2.resize(img, predict_scale)

        if new_img.dtype == np.uint8:
            new_img = new_img.astype(np.float32) / 127.5 - 1

        new_img = torch.from_numpy(np.expand_dims(new_img, axis=0))
        transpose_vector = [0, 3, 1, 2]
        new_img = new_img.permute(transpose_vector)
        return new_img

    def postprocess(self, tensor, to_shape):
        img = tensor.data.cpu()
        transpose_vector = [0, 2, 3, 1]
        img = img.permute(transpose_vector).numpy()
        img = ((img[0] + 1.) * 127.5)
        img = img.astype(np.uint8)
        img = cv2.resize(img, to_shape)
        return img


if __name__ == '__main__':
    opt = get_parameters()
    predictor = StyleTextRenderPredictor(
        opt, model_path='pretrained/G_new_font_wgan.pth')

    style_img = cv2.imread('demo_image/demo_1.png')

    result = predictor.predict_single_image(style_img=style_img,
                                            text_corpus='YANG', font_path='fonts/zh_standard.ttc')

    cv2.imwrite('output/fake.png', result['o_f'])
