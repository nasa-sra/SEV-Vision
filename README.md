# SEV Vision code

- Building: run `pip install -r requirements.txt` to install the necessary packages.
- Recommended Python version: 3.11.15

# Description

*Note: Full documentation is available on the SEV Autonomous Google Drive Folder ("Road Segmentation Documentation").*

This repository contains code for training and evaluating a road segmentation model used for the SEV Autonomous project.

The goal of the project is to autonomously drive the SEV from Building 9 to the Rockyard. This route contains several challenges for a computer vision model. Some streets are backed by sidewalks and buildings; other streets are rocky and transition into grass along the edges. Any computer vision model used for SEV Autonomous must be able to handle paved urban streets with parked cars and sidewalks. It must also be able to handle thin roads backed only by vegetation and telephone wires.

For more information on how this vision model works in these various scenarios, please consult the Google Drive folder. We proceed here to outline how to use the provided code.

# Usage

## Training

**IMPORTANT NOTE: In this section, we explain how to manually train a vision model to effectively classify all the roads from Building 9 to the Rockyard. However, we have already done this training. You can find our model in the SEV Autonomous Google Drive folder, in the Vision folder.**

This computer vision model is based on NVIDIA SegFormer, provided by Hugging Face. The model is modified slightly to include a fourth channel in the input image for depth. It is then trained on the Virtual KITTI 2 dataset to give the model an understanding of the new depth channel, and also to help it with rural roads.

To train the model, run `train_segformer.py`. You will likely need data from VKITTI 2. Place the color images and depth maps from the KITTI dataset into `kitti_eval/images` and `kitti_eval/depth`, respectively. It is recommended to use the filename format `rgb_#####.jpg` and `depth_#####.png`, respectively. For training purposes, we used `vkitti_2.0.3_rgb/Scene20/clone/frames/rgb/Camera_0` and the corresponding depth maps. This includes 836 images; we used the first 600 for training and the remaining 236 for validation. Future work could include using more of the enormous VKITTI 2 dataset for training.

You will also need ground-truth segmentation masks for training purposes. These are provided in the VKITTI 2 dataset; however, it is crucial to convert the VKITTI 2 masks into Cityscapes masks. The SegFormer model we use was originally trained by NVIDIA on Cityscapes, and its outputs are the Cityscapes labels. To effectively train the model, it needs Cityscape labels to compare its own predictions against. The script `kitti_eval/masks_conversion.py` can convert the KITTI masks to Cityscapes masks.

To train, first adjust the training parameters in `train_segformer.py`. We used 20 epochs, a batch size of 8 (this can be reduced or increased depending on your available VRAM -- 8 was optimal for the 16 GB of VRAM we had available), an encoder stage 0 learning rate of 1e-5, a decoder learning rate of 2e-5, and 100 warmup steps. Then, simply run `python train_segformer.py`.

## Compiling to TensorRT

As an optimization step -- and because PyTorch often does not play nicely with ROS -- you can compile the PyTorch vision model into a TensorRT model. First, export your Torch model to an ONNX file (`python export_onnx.py --model-path <INSERT MODEL PATH> --output <INSERT OUTPUT PATH.onnx>`). We recommend the `--static-shape` and `--fp16` flags so the exported model only accepts input images of one shape (specified with `--width` and ``--height`) and uses 16-bit floating point arithmetic for optimization purposes. **IMPORTANT NOTE: The .onnx file for our pre-trained model is also available in the Vision folder, located in the SEV Autonomous folder of the Google Drive.**

The next step is compiling that ONNX file to a TensorRT engine. This must be done on every machine you wish to run the model on, since TensorRT engines are GPU-specific. This is done with the `trtexec` command, available after installing TensorRT on your device. When we first compiled our model into TensorRT, it was actually much slower than its PyTorch counterpart. The following flags helped significantly:
- `--noTF32`
- `--builderOptimizationLevel=5`
- `--stronglyTyped` (so the 16-bit floats from the ONNX file are used in the TensorRT engine)
- `--memPoolSize=workspace:16384` (or as much VRAM as you have available; 16384 worked on a 16-GB machine)

## Processing RealSense video

As explained in the Google Drive documentation, we took a video of the drive from Building 9 to the Rockyard with an Intel RealSense D455 camera. This recording is enormous in raw form (~25 GB); it is available on the Autonomous team's Jetson machine (`Building9ToRockyard.db3`).

After obtaining this video (or capturing your own), you can run the vision model on it in real-time with `python process_video_realtime.py`. To use a TensorRT model, use the `--backend trt` flag; to use a PyTorch model, use the ``--backend torch`` flag. You can specify the TensorRT engine path with ``--trt-engine`` or the PyTorch model path with ``--model-path``. To benchmark the model and print results to the terminal, include ``--timing``.

In the end, this vision model is used to mark all non-drivable pixels as such. Those pixels are then projected into 3D space using the RealSense depth data, and are marked as "keep-out zones" for the SEV controller. To view these final keep-out zones, use the ``--display-drivable`` flag.