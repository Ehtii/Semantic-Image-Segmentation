import sys, os
import tensorflow as tf
import subprocess

sys.path.append("models")

from models.MobileUNet import build_mobile_unet
from models.PSPNet import build_pspnet
from models.DeepLabV3 import build_deeplabv3

SUPPORTED_MODELS = ["MobileUNet",  "PSPNet",  "DeepLabV3"]

SUPPORTED_FRONTENDS = ["ResNet50", "ResNet101"]

def download_checkpoints(model_name):
    subprocess.check_output(["python", "utils/get_pretrained_checkpoints.py", "--model=" + model_name])



def build_model(model_name, net_input, num_classes, crop_width, crop_height, frontend="ResNet101", is_training=True):
	# Get the selected model. 
	# Some of them require pre-trained ResNet

	print("Preparing the model ...")

	if model_name not in SUPPORTED_MODELS:
		raise ValueError("The model you selected is not supported. The following models are currently supported: {0}".format(SUPPORTED_MODELS))

	if frontend not in SUPPORTED_FRONTENDS:
		raise ValueError("The frontend you selected is not supported. The following models are currently supported: {0}".format(SUPPORTED_FRONTENDS))

	if "ResNet50" == frontend and not os.path.isfile("models/resnet_v2_50.ckpt"):
	    download_checkpoints("ResNet50")
	if "ResNet101" == frontend and not os.path.isfile("models/resnet_v2_101.ckpt"):
	    download_checkpoints("ResNet101")
	
	network = None
	init_fn = None
	if model_name == "MobileUNet" or model_name == "MobileUNet-Skip":
	    network = build_mobile_unet(net_input, preset_model = model_name, num_classes=num_classes)
	elif model_name == "PSPNet":
	    # Image size is required for PSPNet
	    # PSPNet requires pre-trained ResNet weights
	    network, init_fn = build_pspnet(net_input, label_size=[crop_height, crop_width], preset_model = model_name, frontend=frontend, num_classes=num_classes, is_training=is_training)
	elif model_name == "DeepLabV3":
	    # DeepLabV requires pre-trained ResNet weights
	    network, init_fn = build_deeplabv3(net_input, preset_model = model_name, frontend=frontend, num_classes=num_classes, is_training=is_training)
	else:
	    raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")

	return network, init_fn
