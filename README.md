# Object dection with SSD_MobileNet_V3

This repo using pretrained model from SSD_MobileNet_V3, which is train on COCO dataset.


## Usage

_Detect on single image_
```
python main.py  --img image_path

# Example:
python main.py --img test.jpg
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
```
