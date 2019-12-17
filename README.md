# DENSELY-ANNOTATED-VIDEO-SEGMENTATION using OSVOS: One-Shot Video Object Segmentation

We have increased the accuracy of OSVOS by ~4.18% implementing BubbleNets and Image Augmentation in the online training/testing part.

### Installation:
1. Clone the repository
   ```Shell
   git clone https://github.com/zeeem/UAlberta-Multimedia-Master-Program-DENSELY-ANNOTATED-VIDEO-SEGMENTATION.git
   ```
2. Install if necessary the required dependencies:
   
   - Python 3
   - Tensorflow r1.0 or higher (`pip install tensorflow-gpu`) along with standard [dependencies](https://www.tensorflow.org/install/install_linux)
   - Other python dependencies: PIL (Pillow version), numpy, scipy 1.2, matplotlib, openCV
   
3. Download the model and Data files from [here](https://drive.google.com/file/d/1PPPsyiLB3gr1TJL9PZXtC8YsYUL8mC2k/view?usp=sharing) (3GB) and unzip it (It should create a folder named 'data, DAVIS, model and methods') under the main directory and replace if needed.


### to get the best frame for each video seq, run the BubbleNets
	!python bubblenets_select_frame.py
You will get the best frame suggestion for the videos from BubbleNets to use in the `argv3`.


### online training and testing
1. If needed, edit in file `osvos_demo.py` the 'User defined parameters' (eg. gpu_id, train_model, etc).

2. Run 
	`!python osvos_demo.py 'scooter-black' 'base' '00000'`
	
	
#### here, 
	argv1: sequence_name = "scooter-black",
	argv2: base/with_aug/bubblenets (test type),
	argv3: for using the first frame: value is "00000" and 
		for best frame from bubbleNets: value should be the frame number i.e. "00032".

### for argv1, you can find the video names below:
	blackswan,
	car-shadow,
	bmx-trees,
	breakdance,
	camel,
	car-roundabout,
	cows,
	dog,
	goat,
	horsejump-high,
	paragliding-launch,
	parkour,
	scooter-black,
	soapbox


### Training the parent network (optional)
1. All the training sequences of DAVIS 2016 are required to train the parent model, thus download it from [here](https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip) if you don't have it. 
2. Place the dataset in this repository or create a soft link to it (`ln -s /path/to/DAVIS/ DAVIS`) if you have it somewhere else.
3. Download the VGG 16 model trained on Imagenet from the TF model zoo from [here](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz).
4. Place the vgg_16.ckpt file inside `models/`.
5. Edit the 'User defined parameters' (eg. gpu_id) in file `osvos_parent_demo.py`.
6. Run `python osvos_parent_demo.py`. This step takes 20 hours to train (Titan-X Pascal), and ~15GB for loading data and online data augmentation. Change dataset.py accordingly, to adjust to a less memory-intensive setup.


video demonstration: https://youtu.be/e7tclBV6ktU

If needed, please contact at rahmanje[at]ualberta[dot]ca, subho[at]ualberta[dot]ca.
