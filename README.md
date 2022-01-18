# object-detection-randoms

Used [coco-labels](https://github.com/amikelive/coco-labels) to generate [coco-labels.txt](coco_labels.txt)

## How to run
```
pip install -r requirements.txt
python3 main.py
```

If you have proper dependencies installed to use CUDA, go to [`./core/objectDetector.py`](core/objectDetector.py) and comment line 13

```python
# in ./core/objectDetector.py
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

