![alt tag](https://media3.giphy.com/media/IDIu4F2htizT2/giphy.gif)
![alt tag](https://media3.giphy.com/media/IDIu4F2htizT2/giphy.gif)

Install homebrew
===

```/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"```

Python configuration
===

Install Python 3
```brew install python3```

Create a virtual environment
===

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Up and running boys!

TODO
===

Install dependencies (TF and such)
Install OpenCV3 (http://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/)

> Link OpenCV3 to the Python env
```
cd venv/lib/python3.6/site-packages
nano opencv.pth
```

Paste this
```
/usr/local/opt/opencv3/lib/python3.6/site-packages/
```

Save and run by doing
```python detect.py ssd_mobilenet_v1_coco_11_06_2017```


