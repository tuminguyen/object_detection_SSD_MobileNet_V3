# Object dection with SSD_MobileNet_V3

This repo using pretrained model from SSD_MobileNet_V3, which is train on COCO dataset.

![demo-video2gif](demo.gif)


## Requirements
- Python 3+
- OpenCV
```
pip install opencv-python
```

## Usage

_Argurments parser_
```python
'--img', '-i':  
    type=str
    description='image path'
'--thresh', '-t': 
    type=float
    description='detection confidence threshold'
    default=0.6
'--incl', '-incl':
    description='list of interested objects, all lowercase'
```

_Detect on single image_
```
# using --img
python main.py  --img image_path

# Example:
python main.py --img test.jpg

# or using -i image_path
python main.py -i test.jpg
```

_Detect on stream_

```
# Using all objects in coco names
python main.py
```
or 
```
# Using specific interested objects
python main.py --incl object_name1 object_name_n

# Example:
python main.py --incl mouse keyboard cup

# Or
python main.py -incl mouse keyboard cup
```

_Set detection threshold_
```
# All detection with confidence that less than thresh value will not be counted
python  main.py --thresh 0.55

# Or 
python main.py -t 0.55
```

You can find other pretrained models from [here](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API)
