## [YOLOv5](https://github.com/ultralytics/yolov5)無人機影像辨識


#### 0.环境配置

安装必要的python package和配置相关环境

```
# python3.9
# torch==1.13.0
# torchvision==0.14.0

安裝完python及pytorch後即可使用
```

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

#### 2.创建标注文件

可以使用LabelImg,Labme,[Labelbox](https://labelbox.com/), [CVAT](https://github.com/opencv/cvat)来标注数据，对于目标检测而言需要标注bounding box即可。然后需要将标注转换为和**darknet format**相同的标注形式，每一个图像生成一个`*.txt`的标注文件（如果该图像没有标注目标则不用创建`*.txt`文件）。创建的`*.txt`文件遵循如下规则：

- 每一行存放一个标注类别
- 每一行的内容包括`class x_center y_center width height`
- Bounding box 的坐标信息是归一化之后的（0-1）
- class label转化为index时计数是从0开始的

```python
def convert(size, box):
    '''
    将标注的xml文件标注转换为darknet形的坐标
    '''
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
```

每一个标注`*.txt`文件存放在和图像相似的文件目录下，只需要将`/images/*.jpg`替换为`/lables/*.txt`即可（这个在加载数据时代码内部的处理就是这样的，可以自行修改为VOC的数据格式进行加载）

例如：

```
datasets/score/images/train/000000109622.jpg  # image
datasets/score/labels/train/000000109622.txt  # label
```
如果一个标注文件包含5个person类别（person在coco数据集中是排在第一的类别因此index为0）：

<img width="500" align="center" alt="Screen Shot 2020-04-01 at 11 44 26 AM" src="./readmepic/readme2/pic/78174482-307bb800-740e-11ea-8b09-840693671042.png">

#### 3.组织训练集的目录

将训练集train和验证集val的images和labels文件夹按照如下的方式进行存放

<img width="500" align="center" alt="Screen Shot 2020-04-01 at 11 44 26 AM" src="./readmepic/readme2/pic/datalist.png">

至此数据准备阶段已经完成，过程中我们假设算法工程师的数据清洗和数据集的划分过程已经自行完成。

#### 4.选择模型backbone进行模型配置文件的修改

在项目的`./models`文件夹下选择一个需要训练的模型，这里我们选择[yolov5x.yaml](https://github.com/ultralytics/yolov5/blob/master/models/yolov5x.yaml),最大的一个模型进行训练，参考官方README中的[table](https://github.com/ultralytics/yolov5#pretrained-checkpoints),了解不同模型的大小和推断速度。如果你选定了一个模型，那么需要修改模型对应的`yaml`文件

```yaml

# parameters
nc: 3  # number of classes   <------------------  UPDATE to match your dataset
depth_multiple: 1.33  # model depth multiple
width_multiple: 1.25  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# yolov5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 1-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 2-P2/4
   [-1, 3, Bottleneck, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 4-P3/8
   [-1, 9, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 6-P4/16
   [-1, 9, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]], # 8-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 6, BottleneckCSP, [1024]],  # 10
  ]

# yolov5 head
head:
  [[-1, 3, BottleneckCSP, [1024, False]],  # 11
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1, 0]],  # 12 (P5/32-large)

   [-2, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 1, Conv, [512, 1, 1]],
   [-1, 3, BottleneckCSP, [512, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1, 0]],  # 17 (P4/16-medium)

   [-2, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 1, Conv, [256, 1, 1]],
   [-1, 3, BottleneckCSP, [256, False]],
   [-1, 1, nn.Conv2d, [na * (nc + 5), 1, 1, 0]],  # 22 (P3/8-small)

   [[], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]

```

#### 5.Train

```bash
# Train yolov5x on score for 300 epochs

$ python3 train.py --img-size 640 --batch-size 16 --epochs 300 --data ./data/score.yaml --cfg ./models/score/yolov5x.yaml --weights weights/yolov5x.pt

```


#### 6.Visualize

开始训练后，查看`train*.jpg`图片查看训练数据，标签和数据增强，如果你的图像显示标签或数据增强不正确，你应该查看你的数据集的构建过程是否有问题

<img width="1000" align="center" alt="Screen Shot 2020-04-01 at 11 44 26 AM" src="./readmepic/readme2/pic/train_batch0.jpg">

一个训练epoch完成后，查看`test_batch0_gt.jpg`查看batch 0 ground truth的labels


<img width="1000" align="center" alt="Screen Shot 2020-04-01 at 11 44 26 AM" src="./readmepic/readme2/pic/test_batch0_gt.jpg">

查看`test_batch0_pred.jpg`查看test batch 0的预测

<img width="1000" align="center" alt="Screen Shot 2020-04-01 at 11 44 26 AM" src="./readmepic/readme2/pic/test_batch0_pred.jpg">

训练的losses和评价指标被保存在Tensorboard和`results.txt`log文件。`results.txt`在训练结束后会被可视化为`results.png`

```python
>>> from utils.utils import plot_results
>>> plot_results()
# 如果你是用远程连接请安装配置Xming: https://blog.csdn.net/akuoma/article/details/82182913
```

<img width="1000" align="center" alt="Screen Shot 2020-04-01 at 11 44 26 AM" src="./readmepic/readme2/pic/results.png">

#### 7.推断

```python
$ python3 detect.py --source file.jpg  # image 
                            file.mp4  # video
                            ./dir  # directory
                            0  # webcam
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp stream
                            http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http stream
````


```python
# inference  /home/myuser/xujing/EfficientDet-Pytorch/dataset/test/ 文件夹下的图像
$ python3 detect.py --source /home/myuser/xujing/EfficientDet-Pytorch/dataset/test/ --weights weights/best.pt --conf 0.1

$ python3 detect.py --source ./inference/images/ --weights weights/yolov5x.pt --conf 0.5

# inference  视频
$ python3 detect.py --source test.mp4 --weights weights/yolov5x.pt --conf 0.4
```

<img width="1000" align="center" alt="Screen Shot 2020-04-01 at 11 44 26 AM" src="./readmepic/readme2/pic/t1.jpg">

<img width="1000" align="center" alt="Screen Shot 2020-04-01 at 11 44 26 AM" src="./readmepic/readme2/pic/20200514_p6_5_247_one.jpg">

#### 8.YOLOv5的TensorRT加速

[请到这里来](./README_v3.md)


**Reference**

[1].https://github.com/ultralytics/yolov5

[2].https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
