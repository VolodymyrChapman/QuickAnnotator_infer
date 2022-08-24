import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from unet import UNet
import ttach as tta
from skimage import io
from sklearn import feature_extraction
from config import config, basedir

#################### Quick Annotator UNet for ROI

# retrieve CUDA device requested in config.ini
def get_torch_device(gpuid = None):
    # Output a torch device with a preferred gpuid (use -2 to force cpu)
    if not gpuid:
        gpuid = config.getint("cuda","gpuid", fallback=0)

    device = torch.device(gpuid if gpuid != -2 and torch.cuda.is_available() else 'cpu')
    return device

# -----helper function to split data into batches
def divide_batch(l, n):
    for i in range(0, l.shape[0], n):
        yield l[i:i + n, ::]

def QA_infer(basedir = basedir):
    # extract args
    # resize = int(args['resize'])
    patch_size = int(config['common']['patchsize'])
    batch_size = int(config['get_prediction']['batchsize'])
    stride_size = patch_size // 2

    # load model:
        # ----- load network
    device = get_torch_device(config['cuda']['gpuid'])

    # check for model files in basedir provided by user
    model_list = [model for model in os.listdir(basedir) if '.pth' in model]
    
    # If multiple models provided or no models provided, raise exception
    if len(model_list) > 1:
        raise ValueError(f'The QA model directory basedir} has more than 1 QA model\nPlease ensure only 1 model is in this directory')
    elif len(model_list) == 0:
        raise ValueError(f'The QA model directory {basedir} is empty\nPlease place a QA model checkpoint in this directory')
    
    # retrieve model, if one pth file in basedir
    model_file = model_list[0]

    checkpoint = torch.load(os.path.join(basedir, model_file), map_location=lambda storage,
                                                            loc: storage)  # load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666
    model = UNet(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"],
                 padding=checkpoint["padding"], depth=checkpoint["depth"], wf=checkpoint["wf"],
                 up_mode=checkpoint["up_mode"], batch_norm=checkpoint["batch_norm"]).to(device)
    model.load_state_dict(checkpoint["model_dict"])
    model.eval()
    tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
    
    # create output image directory
    output_dir = os.path.join(basedir, 'output')
    
    # ask user if to continue if 
    if os.path.exists(output_dir):
        print('Warning - output directory exists for this project.\nWould you like to proceed with inference & overwrite? (y/n)')
        proceed = input()

        if proceed == 'n':
            raise Exception('Inference terminated by user - please empty output directory and repeat')
        elif proceed == 'y':
            pass
        else:
            raise Exception('Input not y or n - please retry and provide one of these options')

    input_dir = os.path.join(basedir, 'input')
    input_image_files = os.listdir(input_dir)

    # predict on all images
    for image in input_image_files:
        # parse image
        img = io.imread(image)
        
        # initiate output image
        io_shape_orig = np.array(img.shape)

        # add half the stride as padding around the image, so that we can crop it away later
        io = np.pad(io, [(stride_size // 2, stride_size // 2), (stride_size // 2, stride_size // 2), (0, 0)],
                    mode="reflect")

        io_shape_wpad = np.array(io.shape)

        # pad to match an exact multiple of unet patch size, otherwise last row/column are lost
        npad0 = int(np.ceil(io_shape_wpad[0] / patch_size) * patch_size - io_shape_wpad[0])
        npad1 = int(np.ceil(io_shape_wpad[1] / patch_size) * patch_size - io_shape_wpad[1])

        io = np.pad(io, [(0, npad0), (0, npad1), (0, 0)], mode="constant")

        arr_out = feature_extraction.image._extract_patches(io, (patch_size, patch_size, 3), stride_size)
        arr_out_shape = arr_out.shape
        arr_out = arr_out.reshape(-1, patch_size, patch_size, 3)

        # in case we have a large network, lets cut the list of tiles into batches
        output = np.zeros((0, checkpoint["n_classes"], patch_size, patch_size))
        batch_index = 0
        for batch_arr in divide_batch(arr_out, batch_size):
            batch_index += 1
            print(f'PROGRESS: Generating Prediction, Batch {batch_size * batch_index}/{arr_out.shape[0]}', flush=True)
            arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)

            # ---- get results
            output_batch = tta_model(arr_out_gpu)

            # --- pull from GPU and append to rest of output
            output_batch = output_batch.detach().cpu().numpy()

            output = np.append(output, output_batch, axis=0)

        output = output.transpose((0, 2, 3, 1))

        # turn from a single list into a matrix of tiles
        output = output.reshape(arr_out_shape[0], arr_out_shape[1], patch_size, patch_size, output.shape[3])

        # remove the padding from each tile, we only keep the center
        output = output[:, :, stride_size // 2:-stride_size // 2, stride_size // 2:-stride_size // 2, :]

        # turn all the tiles into an image
        output = np.concatenate(np.concatenate(output, 1), 1)

        # incase there was extra padding to get a multiple of patch size, remove that as well
        output = output[0:io_shape_orig[0], 0:io_shape_orig[1], :]  # remove padding, crop back
        output = output.argmax(axis=2) * (256 / (output.shape[-1] - 1) - 1)
        
        # create and output mask over original image
        fig, axes = plt.subplots(1, 2, figsize = (18,12))
        axes[0].imshow(img)
        axes[0].set_title('Original image', fontsize = 14)
        
        axes[1].imshow(img)
        axes[1].imshow(output, cmap = 'brg', alpha =0.7*(output>0) )
        axes[1].set_title('QA infer', fontsize = 14)
        # for ax in axes:
        #     ax.axis('off')
        fig.tight_layout()
        
        # save
        plt.savefig(os.path.join(output_dir, image))

        # clear all memory leaks
        # clear the current axes.
        plt.cla() 
        # clear the current figure.
        plt.clf() 
        # closes all the figure windows.
        plt.close('all')