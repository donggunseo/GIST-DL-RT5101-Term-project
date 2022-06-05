import streamlit as st
from PIL import Image
from video2image import video2img, detect
import os
from inference import inference

# file_path = '../data/test_video.mp4'
# image_list = video2img(file_path, 10)
# os.makedirs('../data/test_cropped', exist_ok=True)
# for k, image in image_list:
#     detect(image, k)
img_list = sorted(os.listdir('../data/test_cropped'))
img_list = [os.path.join('../data/test_cropped', x) for x in img_list]
res = inference(img_list)
img_list_f = os.listdir('../data/test_cropped')
st.title("흡연, 잡았다 요놈")
st.write("딥러닝 기반 흡연 감지 모델")
img_list_path = []
img_list_filename_list = []
for i, result in enumerate(res):
    if result==0:
        img_list_path.append(img_list[i])
        img_list_filename_list.append(img_list_f[i])



times = []
for i, name in enumerate(img_list_filename_list):
    frame = name[:-4]
    time = int(frame) / 30
    time = (int(time/60), int(time%60))
    times.append(time)

## Slider
level = st.slider("흡연 사진들", 1, len(img_list_path))


img = Image.open(img_list_path[level - 1])
st.image(img, width=400)
st.error("He smokes at 13:{m}:{t}!!".format(m=times[level-1][0], t = times[level - 1][1]))
