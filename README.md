# object-detection-randoms
## How to run
1. install dependencies
    ```
    pip install -r requirements.txt
    ```
2. 1. To run Image Detection
        ```
        python3 imageDetection.py
        ```
    2. To run Video Detection
        ```
        python3 videoDetection.py
        ```


## Notes
If you have proper dependencies installed to use CUDA, go to [`./core/objectDetector.py`](core/objectDetector.py) and comment line 13

```python
# in ./core/objectDetector.py
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## Special Thanks

1. Special Thanks to [coco-labels](https://github.com/amikelive/coco-labels), I used this repository to generate [coco-labels.txt](coco_labels.txt)

