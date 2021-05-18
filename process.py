import numpy as np

from getimg import file_name
from getimg import load_allimg
from getimg import CAPTCHA_HEIGHT, CAPTCHA_WIDTH, CAPTCHA_LEN, CAPTCHA_LIST
import random
import cv2
# 图片转为黑白，3维转1维
def convert2gray(img):
    if len(img.shape)>2:
        img = np.mean(img, -1)
    return img

# 验证码文本转为向量
def text2vec(text,captcha_len=CAPTCHA_LEN, captcha_list=CAPTCHA_LIST):
    text_len = len(text)
    if text_len > captcha_len:
        raise ValueError("验证码超过4位啦！")
    vector = np.zeros(captcha_len * len(captcha_list))
    for i in range(text_len): vector[captcha_list.index(text[i]) + i * len(captcha_list)] = 1
    return vector


# 验证码向量转为文本
def vec2text(vec, captcha_list=CAPTCHA_LIST, size=CAPTCHA_LEN):
    vec_idx = vec
    text_list = [captcha_list[v] for v in vec_idx]
    return ''.join(text_list)


# 返回特定shape图片
def wrap_gen_captcha_text_and_image(shape=(CAPTCHA_HEIGHT, CAPTCHA_WIDTH, 3)):
        t_list = []
        t = file_name("./yzm") # 这里填写验证码数据集的路径
        for i in t:
            index = i.rfind('.')
            name = i[:index]
            t_list.append(name)
        # print(t_list)
        im = load_allimg()

        im_list = []
        for i in range(0, len(im)):
            if im[i].shape == shape:
                im_list.append(im[i])
        # print(len(im_list))
        # print(len(t_list))
        return t_list, im_list


# 获取训练图片组
def next_batch(batch_count=60, width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT):
    batch_x = np.zeros([batch_count, width * height])
    batch_y = np.zeros([batch_count, CAPTCHA_LEN * len(CAPTCHA_LIST)])
    text, image = wrap_gen_captcha_text_and_image()
    for i in range(batch_count):
        text_a = random.choice(text)
        image_a = image[text.index(text_a)]
        image_a = convert2gray(image_a)
        # 将图片数组一维化 同时将文本也对应在两个二维组的同一行
        batch_x[i, :] = image_a.flatten()/ 255
        batch_y[i, :] = text2vec(text_a)
    # 返回该训练批次
    return batch_x, batch_y

if __name__ == '__main__':
    x,y = next_batch(batch_count=1)
    print(x,'\n\n',y)



