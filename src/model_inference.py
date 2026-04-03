import numpy as np
import matplotlib.pyplot as plt
import torch

@torch.no_grad()
def make_prediction(model, gradient, device=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    prediction, *_ = model( torch.FloatTensor(gradient).unsqueeze(0).unsqueeze(0).to(device) )
    prediction = np.squeeze( prediction ).detach().cpu().numpy()
    
    return prediction

def plot_prediction(input_image, ground_truth, prediction, minVal=-1, maxVal=1, inputMin = -8, inputMax = 8):
    
    plt.rcParams["figure.figsize"] = (20,10)
    diff = prediction - ground_truth
    minVal = diff.min() if minVal is None else minVal
    maxVal = diff.max() if maxVal is None else maxVal
    
    fig, axs = plt.subplots(1, 4)
    
    axs[0].set_title('ground truth')
    axs[0].imshow(ground_truth, cmap='gray', vmin=0, vmax=1)
    axs[0].axis('off')
    
    axs[1].set_title('input image')
    axs[1].imshow(input_image, cmap='gray', vmin=inputMin, vmax=inputMax)
    axs[1].axis('off')
    
    axs[2].set_title('reconstruction')
    axs[2].imshow(prediction, cmap='gray', vmin=0, vmax=1)
    axs[2].axis('off')
    
    axs[3].set_title(f'difference scaled ({minVal},{maxVal})')
    im = axs[3].imshow(diff, cmap='gray', vmin=minVal, vmax=maxVal)
    axs[3].axis('off')
    
    # divider = make_axes_locatable(axs[3])
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()

    print( f'MAE is {np.sum(np.abs(diff))/(512*512)}' )
    print( f'MSE is {np.sum(diff**2)/(512 * 512)}')
    
def run_test(model, transform, image_file, multiplier = 1.0, minVal=-1, maxVal=1, inputMin = -8, inputMax = 8, rotate90 = None, flip = None, norm_recon = False, plot = False):
    model.eval()
    input_image, ground_truth = transform(image_file)
    input_image = input_image * multiplier
    #print(f"Multiplier is {multiplier}")
    if flip is not None:
        input_image = np.fliplr(input_image).copy()
        ground_truth = np.fliplr(ground_truth).copy()
        
    if rotate90 is not None:
        input_image = np.rot90(input_image, rotate90, (1,0)).copy()
        ground_truth = np.rot90(ground_truth, rotate90, (1,0)).copy()

    prediction = make_prediction(model, input_image)
    
    if norm_recon:
        pred_scaled = (prediction - prediction.mean())/ (prediction.std() + 1e-8)
        pred_scaled = pred_scaled * ground_truth.std() + ground_truth.mean()
        prediction = pred_scaled
        
    mae = (np.abs(prediction - ground_truth)).mean()
    if (plot):
        plot_prediction(input_image, ground_truth, prediction, minVal=minVal, 
                    maxVal=maxVal, inputMin = inputMin, inputMax = inputMax)
    return ground_truth, input_image, prediction, mae


    