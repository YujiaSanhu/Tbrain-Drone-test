## [YOLOv5](https://github.com/ultralytics/yolov5)無人機影像辨識


#### 0.環境配置

安裝必要的python package和配置相關環境

```
# python3.9
# torch==1.13.0
# torchvision==0.14.0

安裝完python及pytorch後即可使用
```

### 1.下載檔案和模型

先到[Google雲端](https://drive.google.com/file/d/1fEzqibY4f4cPhFUk-V3eVRywVwaG8esJ/view?usp=share_link)下載模型，之後下載github內的檔案後再將模型移入。

#### 2.影像偵測辨識`detect.py`

開啟此檔案後，按下執行即可開始進行辨識。

 ```python
def parse_opt():
    ...
    將從雲端上下載的訓練完的模型安裝到指定路徑後，調整裡面參數進行辨識。
    ...
    
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'R_4_2_x.pt', help='model path or triton URL')   #預設模型名稱為「R_4_2_x.pt」
    parser.add_argument('--source', type=str, default='./datasets/test/', help='file/dir/URL/glob/screen/0(webcam)')    #要辨識的圖檔路徑
    #parser.add_argument('--source', type=str, default='2', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')    #使用coco128訓練集
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280,1920], help='inference size h,w')    #解析度設為1920 x 1280
    parser.add_argument('--conf-thres', type=float, default=0.35, help='confidence threshold')    #信心指數設為0.35
    parser.add_argument('--iou-thres', type=float, default=0.35, help='NMS IoU threshold')    #IoU設為0.35
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')   #最大偵測張數
 ```
