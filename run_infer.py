import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from unet import UNet
import ttach as tta
import skimage.io as skio
from sklearn import feature_extraction
# TODO: import config functions into this file so no input needed when importing functions from this file
# i.e. consequence of importing from config here
from config import config, basedir
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

# Note!!! please ensure the dash '-' is not used within filenames of the images you would like to predict on

# The underscore is used as a separator for saving predictions from individual models where multiple models are used. 
# If this character is used within image file names, this script will error out with a corresponding naming error. 
# If your images do have an underscore within them, the easiest way to avoid this error is by changing the below 
# sep string to a character that is not contained within your image names, such as a dash (-), 
# comma (,) or full stop (.)

sep = '_'

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

def check_sep_use(filename, sep):
    if len(filename.split(sep)) > 2:
        raise ValueError(f'''The following filename contains the separator character {sep} within 
            the filename: {filename}\nTo resolve, please either change the value of the sep variable at the top of the run_infer.py file
            or remove the separator character from the filename''')

def run_model_infer(model_path, device, stride_size, batch_size, patch_size, sep, output_dir):
    checkpoint = torch.load(model_path, map_location=lambda storage,
                                                            loc: storage)  # load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666
    model = UNet(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"],
                 padding=checkpoint["padding"], depth=checkpoint["depth"], wf=checkpoint["wf"],
                 up_mode=checkpoint["up_mode"], batch_norm=checkpoint["batch_norm"]).to(device)
    model.load_state_dict(checkpoint["model_dict"])
    model.eval()
    tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

    input_dir = os.path.join(basedir, 'input')
    input_image_files = os.listdir(input_dir)

    # predict on all images
    for image in input_image_files:
        
        # check that separator character not used within image name
        check_sep_use(image, sep)

        # if image name is fine, parse image
        img = skio.imread(os.path.join(input_dir, image) )

        io = cv2.resize(src = img, dsize = (0, 0), fx=1, fy=1)
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

        # output prediction as numpy array - named after model used and image
        outfile_name = os.path.split(model_path)[-1] # get model name from path
        outfile_name = os.path.splitext(outfile_name)[0] #remove extension
        outfile_name = outfile_name + sep + os.path.splitext(image)[0] + '.npy' # add image name, removing extension 
        np.save(os.path.join(output_dir, outfile_name), output)
        
def arr_list_mean_round(arr_list):
    # stack
    arr_stack = np.stack(arr_list, axis = 0)
    # mean
    arr_stack = np.mean(arr_stack, axis = 0) / 255 # as max value at 255
    # round
    arr_stack = np.round(arr_stack, 0)

    return arr_stack

def mean_ensemble(basedir, sep, plot = True):
    img_dir = os.path.join(basedir, 'input')
    # make dir for combined (ensemble) predictions if doesn't exist
    output_dir = os.path.join(basedir, 'output')
    
    # combined prediction not put in separate directory

    # combined_pred_dir = os.path.join(output_dir, 'ensemble_predictions')
    # if os.path.exists(combined_pred_dir) == False:
    #     os.mkdir(combined_pred_dir)

    # retrieve list of unique image values
    output_file_list = [filename for filename in os.listdir(output_dir) if '.npy' in filename] #remove non-npy files
    uniq_image_files = set([image_name.split(sep)[-1] for image_name in output_file_list]) # remove model ID from name
    uniq_image_files = [os.path.splitext(image_name)[0] for image_name in uniq_image_files] # remove extension from unique image names
    
    # for each unique image, retrieve combined ensemble mask for predictions
    for uniq_image in uniq_image_files:
        # retrieve all predictions for that unique image
        preds_filenames = [pred for pred in output_file_list if uniq_image in pred]
        # parse prediction numpy arrays
        preds = [np.load(os.path.join(output_dir, pred)) for pred in preds_filenames]
        # retrieve mean and rounded (combined) prediction
        out_pred = arr_list_mean_round(preds)
        # output combined prediction
        np.save(os.path.join(output_dir, 'combined' + sep + uniq_image + '.npy'), out_pred)
        
        # if plots desired, plot predictions of each model
                    
        #TODO: Plot predictions against original image + final consensus prediction
        if plot:
            # create and output mask over original image
            fig, axes = plt.subplots(2, len(preds_filenames), figsize = (int(len(preds_filenames) * 6),12))
            # retrieve model titles
            titles = [filename.split(sep)[1] for filename in preds_filenames]
            
            # plot original image on bottom row
            #TODO: What about other image formats? Need to use search-like rather than hard-coded extension
            img_filename = uniq_image + '.png' 

            img = skio.imread(os.path.join(img_dir, img_filename))
            axes[1][0].imshow(img)
            axes[1][0].set_title('Original image', fontsize = 14)

            # Plot combined / ensemble prediction on bottom row
            axes[1][1].imshow(img)
            axes[1][1].imshow(out_pred, cmap = 'brg', alpha =0.7*(out_pred>0) )
            axes[1][1].set_title('Mean consensus prediction', fontsize = 14)
            
            for i in range(len(preds)):
                axes[0][i].imshow(img)
                axes[0][i].imshow(preds[i], cmap = 'brg', alpha =0.7*(preds[i]>0) )
                axes[0][i].set_title(f'Model: {titles[i]}', fontsize = 14)
        
        # turn axes ticks off
            for ax in axes.ravel():
                ax.set_axis_off()

            fig.tight_layout()
            
            # save
            plt.savefig(os.path.join(output_dir, 'Comparison_plot_' + img_filename))

            # clear all memory leaks
            # clear the current axes.
            plt.cla() 
            # clear the current figure.
            plt.clf() 
            # closes all the figure windows.
            plt.close('all')

def QA_mask_eval(gt_mask, pred_mask):       
        # ravel and index regions that have annotations (gt == QA_gt_mask[:,:,2])
        annot_regions = gt_mask[:,:,2].ravel()
        
        # in gt_mask[:,:,1], 0 val corresponds to negative and unlabelled regions.
        # indexing only annot regions means 0 corresponds to negative regions,
        # afterwards
        gt_annot_regions = gt_mask[:,:,1].ravel()[annot_regions]
        pred_annot_regions = pred_mask.ravel()[annot_regions]
        
        # calculate and output desired metrics  - F1-score and AUC for overall fit eval;
        # precision and recall to eval classification characteristics
        outdict = {'annotated_area(px)': sum(annot_regions),
                        'precision':precision_score(gt_annot_regions, pred_annot_regions),
                        'recall': recall_score(gt_annot_regions, pred_annot_regions),
                        'f1_score':f1_score(gt_annot_regions, pred_annot_regions),
                        'roc_auc':roc_auc_score(gt_annot_regions, pred_annot_regions)
                        }

        return outdict

def QA_eval(basedir):
    #TODO: What if multiple predictions exist for an image? i.e. multiple models input and ensemble generated, too?

    # if gt dir exists but infer dir doesn't exist, make
    gt_dir = os.path.join(basedir, 'gt')
    pred_dir = os.path.join(basedir, 'output')
    
    # retrieve just QA masks from pred dir
    pred_files = [pred_file for pred_file in os.listdir(pred_dir) if '.npy' in pred_file]
    # QA gt masks output as png... how to future-proof in case of other formats?
    gt_files = [gt_file for gt_file in os.listdir(gt_dir) if '.png' in gt_file]
    
    if os.path.exists(gt_dir) == False:
        raise ValueError('''gt directory does not exist in project directory.
                        \nPlease create a directory named gt in the project directory 
                        and either fill with \ngt masks to retrieve metrics or
                        create separate dirs for data\n partitions - 
                        train, cv, holdout - to retrieve metrics separately''')
    
    # TODO: option for multiple data partitions i.e. using os.walk allows for subdirectories?


    outlist = []

    for gt_file in gt_files:
        # get core image name from QA mask - remove ext and trailing '_mask'
        # TODO: What to do in case of no trailing '_mask' ? 
        img_name = sep.join(gt_file.split('_')[:-1])

        gt_mask = skio.imread(os.path.join(gt_dir, gt_file)) > 0

        # search for mask using core image name - 

        pred_file_list = [pred for pred in pred_files if img_name in pred]

        # in edge case of no pred available - skip this loop
        if len(pred_file_list) == 0:
            print(f'No predictions for image: {img_name} - Skipping!')
            continue
        
        # TODO: Parallelise? Currently slow and only uses one core. Can fairly easily use processpoolexecutor for this
        # For-loop implementation
        # for used as could be multiple appropriate pred files
        for pred in pred_file_list:
            # parse pred
            pred_mask = np.load(os.path.join(pred_dir, pred)) > 0

            # retrieve metric dict for image and add image name to keys / vals
            metric_dict = QA_mask_eval(gt_mask, pred_mask)
            metric_dict['imagename'] = img_name

            # Assumes model_id + sep+ '.npy' naming convention for models 
            model_id = sep.join(pred.split(sep)[:-1])
            metric_dict['model_id'] = model_id

            outlist.append(metric_dict)

    
    out_df = pd.DataFrame(outlist)
    # output as tsv
    out_df.to_csv(os.path.join(gt_dir, 'eval.tsv'), sep = '\t', index = False)
        
def QA_infer(basedir, sep):
    # extract args
    # resize = int(args['resize'])
    patch_size = int(config.getint('common','patchsize'))
    batch_size = int(config.getint('get_prediction','batchsize'))
    stride_size = patch_size // 2

    # prediction output directory
    output_dir = os.path.join(basedir, 'output')
    # if output dir already exists (has been run before), ask user to continue
    if os.path.exists(output_dir):
        print('Warning - output directory exists for this project.\nWould you like to proceed with inference & overwrite? (y/n)')
        proceed = input()
        
        # raise exception if user doesn't want to continue
        if proceed == 'n':
            raise Exception('Inference terminated by user - please empty output directory and repeat')
        elif proceed == 'y':
            pass
        else:
            raise Exception('Input not y or n - please retry and provide one of these options')
    # create outdir if doesn't exist 
    else:
        os.mkdir(output_dir)

    # load model:
        # ----- load network
    device = get_torch_device()

    # check for model files in basedir provided by user
    model_dir = os.path.join(basedir, 'models')
    model_list = [model for model in os.listdir(model_dir) if '.pth' in model]
    
    # If multiple models provided or no models provided, raise exception
    if len(model_list) > 1:
        print(f'{len(model_list)} models detected - retrieving predictions for all models and generating mean prediction')
    if len(model_list) == 0:
        raise ValueError(f'The QA model directory {basedir} is empty\nPlease place a QA model checkpoint in this directory')
    
    # for each model present, run inference on data
    for model_file in model_list:
        model_path = os.path.join(model_dir, model_file)

        run_model_infer(model_path, device, stride_size, batch_size, patch_size, sep, output_dir)
        print(f'Inference complete for model ID: {model_file}')
    
    # if more than one model was used for inference, output a combined / ensembled result (mean of all + rounding)
    if len(model_list) > 1:
        print('Creating ensemble prediction')
        mean_ensemble(basedir, sep)
        print('Ensemble prediction complete!')
    
    print('All inference complete!')

    # if gt dir exists, eval available 
    if os.path.exists(os.path.join(basedir, 'gt')):
        print('Evaluating prediction against ground truth')
        QA_eval(basedir)

# run inference
if __name__ == '__main__': 
    QA_infer(basedir, sep)