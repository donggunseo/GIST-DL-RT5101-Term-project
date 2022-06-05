# Smoke Classification (GIST 2022-1 DL class)
## Environment Setting
```bash
pip install -r requirements.txt
```
## Directory
```bash
Team4_Termproject
├── code
│   ├── README.md
│   ├── crop_train.py
│   ├── data.py
│   ├── inference.py
│   ├── streamlit_app.py
│   ├── train.py
│   ├── utils.py
│   └── video2image.py
├──opencv_detection
│   ├── yolov3.cfg
│   └── yolov3.weights
├──pt
│   └── pytorch.bin
├──README.md
├──requirements.txt
└── data
    ├── test_cropped
    ├── Train_image_cropped
    ├── Train_image_original
    └── test_video.mp4
```
## Prepare Train dataset
We cropped automatically focused on person itself
```bash
python crop_train.py
```

## Train Smoke classification model
```bash
python train.py
```

## Show Result from recorded video(CCTV)
```bash
streamlit run strealit_app.py
```