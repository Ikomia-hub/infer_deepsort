<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_deepsort/main/icons/logo.png" alt="Algorithm icon">
  <h1 align="center">infer_deepsort</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_deepsort">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_deepsort">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_deepsort/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_deepsort.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Run DeepSort tracking algorithm for video analysis. In most cases, tracking algorithms should be connected to object detection algorithm.

Simple Online and Realtime Tracking (SORT) is a pragmatic approach to multiple object tracking with a focus on simple, effective algorithms. This algorithm improves performance of SORT by introducing deep association metric to reduce object identity switches.

![Example image](https://raw.githubusercontent.com/Ikomia-hub/infer_deepsort/feat/new_readme/icons/example.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display
import cv2

# Init your workflow
wf = Workflow()

# Add object detection algorithm
detector = wf.add_task(name="infer_yolo_v7", auto_connect=True)

# Add DeepSORT tracking algorithm
tracking = wf.add_task(name="infer_deepsort", auto_connect=True)

stream = cv2.VideoCapture(0)
while True:
    # Read image from stream
    ret, frame = stream.read()

    # Test if streaming is OK
    if not ret:
        continue

    # Run the workflow on current frame
    wf.run_on(array=frame)

    # Get results
    image_out = tracking.get_output(0)
    obj_detect_out = tracking.get_output(1)

    # Display
    img_res = cv2.cvtColor(image_out.get_image_with_graphics(obj_detect_out), cv2.COLOR_BGR2RGB)
    display(img_res, title="DeepSORT", viewer="opencv")

    # Press 'q' to quit the streaming process
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the stream object
stream.release()
# Destroy all windows
cv2.destroyAllWindows()

```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

```python
# Add DeepSORT tracking algorithm
tracking = wf.add_task(name="infer_deepsort", auto_connect=True)

tracking.set_parameters({
    "categories": "all",
    "conf_thres": "0.5",
})
```

- **categories** (str, default="all"): categories of objects you want to track. Use a comma separated string to set multiple categories (ex: "dog,person,car").
- **conf_thresh** (float, default=0.5): object detection confidence.

***Note***: parameter key and value should be in **string format** when added to the dictionary.

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
# Add DeepSORT tracking algorithm
tracking = wf.add_task(name="infer_deepsort", auto_connect=True)

stream = cv2.VideoCapture(0)
while True:
    # Read image from stream
    ret, frame = stream.read()

    # Test if streaming is OK
    if not ret:
        continue

    # Run the workflow on current frame
    wf.run_on(array=frame)

    # Iterate over outputs
    for output in tracking.get_outputs():
        # Print information
        print(output)
        # Export it to JSON
        output.to_json()
```

DeepSORT algorithm generates 2 outputs:

1. Forwaded original image (CImageIO)
2. Object detection output (CObjectDetectionIO)
