import numpy as np
import cv2

import paddle

paddle.disable_static()
import paddle.vision.transforms as T
import gradio as gr


def imgg(image):
    pre_model = paddle.vision.models.resnet50(pretrained=True, num_classes=2)
    pre_model.set_state_dict(paddle.load('acc0.95.model'))
    pre_model.eval()
    normalize = T.Normalize(mean=0, std=1)
    image = np.array(image)
    # image = np.array(Image.open(image_path))    # H, W, C
    image = image.transpose([2, 0, 1])[:3]  # C, H, W
    a = {'0': '只不是奶龙', '1': '上古巨兽在此'}

    # 图像变换
    features = cv2.resize(image.transpose([1, 2, 0]), (256, 256)).transpose([2, 0, 1]).astype(np.float32)
    features = normalize(features)

    features = paddle.to_tensor([features])
    pre = list(np.array(pre_model(features)[0]))
    # print(pre)
    max_item = max(pre)
    print(max_item)
    pre = pre.index(max_item)
    pre = a.get(str(pre))
    hanzi = pre
    return hanzi


def image_mod(image):
    return imgg(image)


demo = gr.Interface(
    image_mod,
    gr.Image(type="pil"),
    "text",
    flagging_options=["correct", "incorrect", 'error'])

if __name__ == "__main__":
    demo.launch()
