from .unets import VanillaEncoder512, DynamicMCNN, UNetModelParallel

from ._weights_api import Weights, WeightsEnum

import torchvision.transforms as tv_transforms
from .transforms import *

__all__ = ['munet_v10', 'MUNet_v10_Weights', 'transforms']

transforms = dict(
    individual_transform=tv_transforms.Compose([
        OpenImage(),
        tv_transforms.Resize(512),
        tv_transforms.CenterCrop(512),
        ToNumpyArray(),
        IndividualNorm(),
        EdgeTransform()
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    default_transform=tv_transforms.Compose([
        OpenImage(),
        tv_transforms.Resize(512),
        tv_transforms.Grayscale(),    
        tv_transforms.CenterCrop(512),
        ToNumpyArray(),
        DefaultNorm(),
    ]),
    default_transform_only=tv_transforms.Compose([
        OpenImage(),
        tv_transforms.Resize(512),
        tv_transforms.Grayscale(),    
        tv_transforms.CenterCrop(512),
        ToNumpyArray(),
        DefaultNormOnly(),
    ]),
)

class MUNet_v10_Weights(WeightsEnum):        
    GAA_ORIG = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-projects/edgenet/weights/mcnn_pytorch_v10_attn_take2-fdb7bc87.pth.tar",
        transforms=transforms,
        meta={
            "repo": "https://github.com/harvard-visionlab/edgenet",
            "log_url": None,
            "params_url": None,
            "train_script": "MCNN_PyTorch-v10.ipynb",
            "task": "edges-to-grayscale-recon",
            "dataset": "training_dataset1",
            "arch": "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_metrics": {

            },
            "_docs": """
                This is the original `pilot` model, trained in a notebook ages ago.
            """,
        },
    )
    GAA_REP1 = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-projects/edgenet/logs/debug/munet_v10/8a6ac210-75a8-435a-bfc2-f3703764d9b1/munet_v10_final_weights-45a87dd19a.pth",
        transforms=transforms,
        meta={
            "repo": "https://github.com/harvard-visionlab/edgenet",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-projects/edgenet/logs/debug/munet_v10/8a6ac210-75a8-435a-bfc2-f3703764d9b1/munet_v10_log-45a87dd19a.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-projects/edgenet/logs/debug/munet_v10/8a6ac210-75a8-435a-bfc2-f3703764d9b1/munet_v10_params-45a87dd19a.json",
            "train_script": "train.py",
            "task": "edges-to-grayscale-recon",
            "dataset": "training_dataset1",
            "arch": "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_metrics": {

            },
            "_docs": """
                This is the first replication of the model using the original training_dataset1, and
                the train.py script in `https://github.com/harvard-visionlab/edgenet`.
            """,
        },
    )
    
    SS_EDGE_DS1_REP1 = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/920413ea-bfa9-4013-bc75-54b1d5a48b2b/munet_v10_final_weights-461ea7dca5.pth",
        transforms=transforms,
        meta={
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/920413ea-bfa9-4013-bc75-54b1d5a48b2b/munet_v10_log-461ea7dca5.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/920413ea-bfa9-4013-bc75-54b1d5a48b2b/munet_v10_params-461ea7dca5.json",
            "train_script": "train.py",
            "task": "edges-to-grayscale-recon",
            "dataset": "training_dataset1",
            "arch": "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_metrics": {

            },
            "_docs": """
                This is first replication of the edgenet on the training_dataset1 by Srijani Saha
            """,
        },
    )
    
    SS_EDGE_DS1_REP2 = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/9907681d-0cc2-49aa-ba86-52c084985081/munet_v10_final_weights-6930b20c3d.pth",
        transforms=transforms,
        meta={
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/9907681d-0cc2-49aa-ba86-52c084985081/munet_v10_log-6930b20c3d.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/9907681d-0cc2-49aa-ba86-52c084985081/munet_v10_params-6930b20c3d.json",
            "train_script": "train.py",
            "task": "edges-to-grayscale-recon",
            "dataset": "training_dataset1",
            "arch": "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_metrics": {

            },
            "_docs": """
                This is second replication of the edgenet on the training_dataset1 by Srijani Saha
            """,
        },
    )
    
    SS_EDGE_DS1_REP3 = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/5ab9ccfa-c938-426a-8c09-de7f9e91f185/munet_v10_final_weights-eefaab0f72.pth",
        transforms=transforms,
        meta={
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/5ab9ccfa-c938-426a-8c09-de7f9e91f185/munet_v10_log-eefaab0f72.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/5ab9ccfa-c938-426a-8c09-de7f9e91f185/munet_v10_params-eefaab0f72.json",
            "train_script": "train.py",
            "task": "edges-to-grayscale-recon",
            "dataset": "training_dataset1",
            "arch": "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_metrics": {

            },
            "_docs": """
                This is third replication of the edgenet on the training_dataset1 by Srijani Saha
            """,
        },
    )
    
    SS_EDGE_DS1_REP4 = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/d1c2da8d-fd6a-4256-8a79-a04c40a7b967/munet_v10_final_weights-76223d3cfe.pth",
        transforms=transforms,
        meta={
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/d1c2da8d-fd6a-4256-8a79-a04c40a7b967/munet_v10_log-76223d3cfe.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/d1c2da8d-fd6a-4256-8a79-a04c40a7b967/munet_v10_params-76223d3cfe.json",
            "train_script": "train.py",
            "task": "edges-to-grayscale-recon",
            "dataset": "training_dataset1",
            "arch": "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_metrics": {

            },
            "_docs": """
                This is fourth replication of the edgenet on the training_dataset1 by Srijani Saha
            """,
        },
    )

    SS_EDGE_DS1_REP5 = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/dd3a3bb1-9978-4fd9-b3b3-9c670f5924bf/munet_v10_final_weights-93e9c13f32.pth",
        transforms=transforms,
        meta={
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/dd3a3bb1-9978-4fd9-b3b3-9c670f5924bf/munet_v10_log-93e9c13f32.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/dd3a3bb1-9978-4fd9-b3b3-9c670f5924bf/munet_v10_params-93e9c13f32.json",
            "train_script": "train.py",
            "task": "edges-to-grayscale-recon",
            "dataset": "training_dataset1",
            "arch": "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_metrics": {

            },
            "_docs": """
                This is fifth replication of the edgenet on the training_dataset1 by Srijani Saha
            """,
        },
    )
    

    SS_EDGE_DS1_REP6 = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/1600880a-578a-415f-8be3-da13eaf73bef/munet_v10_final_weights-c02bacfc67.pth",
        transforms=transforms,
        meta={
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/1600880a-578a-415f-8be3-da13eaf73bef/munet_v10_log-c02bacfc67.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/1600880a-578a-415f-8be3-da13eaf73bef/munet_v10_params-c02bacfc67.json",
            "train_script": "train.py",
            "task": "edges-to-grayscale-recon",
            "dataset": "training_dataset1",
            "arch": "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_metrics": {

            },
            "_docs": """
                This is sixth replication of the edgenet on the training_dataset1 by Srijani Saha
            """,
        },
    )
    
    SS_EDGE_DS2_REP1 = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/6f4e2b84-7a0d-4b97-bbea-5f67e2349ae9/munet_v10_final_weights-c10424db31.pth",
        transforms=transforms,
        meta={
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/6f4e2b84-7a0d-4b97-bbea-5f67e2349ae9/munet_v10_log-c10424db31.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/6f4e2b84-7a0d-4b97-bbea-5f67e2349ae9/munet_v10_params-c10424db31.json",
            "train_script": "train.py",
            "task": "edges-to-grayscale-recon",
            "dataset": "training_dataset2",
            "arch": "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_metrics": {

            },
            "_docs": """
                This is first training of the edgenet on the training_dataset2
            """,
        },
    )
    
    SS_EDGE_DS2_REP2 = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/29e00b43-bf9f-4bb8-a270-4e4734bdc5e2/munet_v10_final_weights-c8ee3601c2.pth",
        transforms=transforms,
        meta={
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/29e00b43-bf9f-4bb8-a270-4e4734bdc5e2/munet_v10_log-c8ee3601c2.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/29e00b43-bf9f-4bb8-a270-4e4734bdc5e2/munet_v10_params-c8ee3601c2.json",
            "train_script": "train.py",
            "task": "edges-to-grayscale-recon",
            "dataset": "training_dataset2",
            "arch": "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_metrics": {

            },
            "_docs": """
                This is second training of the edgenet on the training_dataset2
            """,
        },
    )
    
    SS_EDGE_DS2_REP3 = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/559df79d-5c3c-4571-b40e-eb5a65e24e8f/munet_v10_final_weights-b3816f5b28.pth",
        transforms=transforms,
        meta={
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/559df79d-5c3c-4571-b40e-eb5a65e24e8f/munet_v10_log-b3816f5b28.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/559df79d-5c3c-4571-b40e-eb5a65e24e8f/munet_v10_params-b3816f5b28.json",
            "train_script": "train.py",
            "task": "edges-to-grayscale-recon",
            "dataset": "training_dataset2",
            "arch": "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_metrics": {

            },
            "_docs": """
                This is third training of the edgenet on the training_dataset2
            """,
        },
    )
    
    SS_EDGE_DS2_REP4 = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/743ed571-7b33-4062-8fc6-58e2fe8aaf15/munet_v10_final_weights-e80e91ed99.pth",
        transforms=transforms,
        meta={
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/743ed571-7b33-4062-8fc6-58e2fe8aaf15/munet_v10_log-e80e91ed99.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/743ed571-7b33-4062-8fc6-58e2fe8aaf15/munet_v10_params-e80e91ed99.json",
            "train_script": "train.py",
            "task": "edges-to-grayscale-recon",
            "dataset": "training_dataset2",
            "arch": "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_metrics": {

            },
            "_docs": """
                This is fourth training of the edgenet on the training_dataset2
            """,
        },
    )
    
    SS_EDGE_DS2_REP5 = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/3f252e34-22be-401b-89e3-12b939808833/munet_v10_final_weights-95f4bc3393.pth",
        transforms=transforms,
        meta={
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/3f252e34-22be-401b-89e3-12b939808833/munet_v10_log-95f4bc3393.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/3f252e34-22be-401b-89e3-12b939808833/munet_v10_params-95f4bc3393.json",
            "train_script": "train.py",
            "task": "edges-to-grayscale-recon",
            "dataset": "training_dataset2",
            "arch": "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_metrics": {

            },
            "_docs": """
                This is fifth training of the edgenet on the training_dataset2
            """,
        },
    )
    
    SS_EDGE_DS2_REP6 = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/a9d19301-44b7-4c9d-9366-70dd224a1d8d/munet_v10_final_weights-b61c6d3b24.pth",
        transforms=transforms,
        meta={
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/a9d19301-44b7-4c9d-9366-70dd224a1d8d/munet_v10_log-b61c6d3b24.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/a9d19301-44b7-4c9d-9366-70dd224a1d8d/munet_v10_params-b61c6d3b24.json",
            "train_script": "train.py",
            "task": "edges-to-grayscale-recon",
            "dataset": "training_dataset2",
            "arch": "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_metrics": {

            },
            "_docs": """
                This is sixth training of the edgenet on the training_dataset2
            """,
        },
    )
    
    SS_EDGE_DS2_REP7 = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/51417231-4484-4976-b0a5-14a5e098a554/munet_v10_final_weights-2e7453f902.pth",
        transforms=transforms,
        meta={
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/51417231-4484-4976-b0a5-14a5e098a554/munet_v10_log-2e7453f902.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/51417231-4484-4976-b0a5-14a5e098a554/munet_v10_params-2e7453f902.json",
            "train_script": "train.py",
            "task": "edges-to-grayscale-recon",
            "dataset": "training_dataset2",
            "arch": "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_metrics": {

            },
            "_docs": """
                This is seventh training of the edgenet on the training_dataset2
            """,
        },
    )

    SS_EDGE_DS2_REP8 = Weights(
        url="https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/4657bb73-9397-4a35-b1b1-8971978ffc0c/munet_v10_final_weights-6fa2249803.pth",
        transforms=transforms,
        meta={
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/4657bb73-9397-4a35-b1b1-8971978ffc0c/munet_v10_log-6fa2249803.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge/4657bb73-9397-4a35-b1b1-8971978ffc0c/munet_v10_params-6fa2249803.json",
            "train_script": "train.py",
            "task": "edges-to-grayscale-recon",
            "dataset": "training_dataset2",
            "arch": "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_metrics": {

            },
            "_docs": """
                This is eigth training of the edgenet on the training_dataset2
            """,
        },
    )


    
    SS_EDGE_8k_b_REP1 = Weights(
         url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/0f0e7504-c288-4f3a-93ca-bab42fda6f95/munet_v10_final_weights-06d61c1f3f.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/0f0e7504-c288-4f3a-93ca-bab42fda6f95/munet_v10_log-06d61c1f3f.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/0f0e7504-c288-4f3a-93ca-bab42fda6f95/munet_v10_params-06d61c1f3f.json",
            "train_script" : "train_edge_ffcv.py",
            "task": "edge_based_recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 1st instance of training the original architecture on edge-based reconstruction on 10k ImageNet Images (8k training, 2048 validation split) after 
                ensuring that there is matched input preprocessing transforms with the 1k model training (added random horizontal flip).
            """,
        },
    )
    
         
    SS_EDGE_8k_b_REP2 = Weights(
         url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/41a5cba4-f2b4-445a-9ad8-91d3d4e9aed3/munet_v10_final_weights-cc74559890.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/41a5cba4-f2b4-445a-9ad8-91d3d4e9aed3/munet_v10_log-cc74559890.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/41a5cba4-f2b4-445a-9ad8-91d3d4e9aed3/munet_v10_params-cc74559890.json",
            "train_script" : "train_edge_ffcv.py",
            "task": "edge_based_recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 2nd instance of training the original architecture on edge-based reconstruction on 10k ImageNet Images (8k training, 2048 validation split) after 
                ensuring that there is matched input preprocessing transforms with the 1k model training (added random horizontal flip).
            """,
        },
    )
    
    SS_EDGE_8k_b_REP3 = Weights(
         url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/41efeea8-e31e-4961-9e26-02e6babca6ab/munet_v10_final_weights-27da478484.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/41efeea8-e31e-4961-9e26-02e6babca6ab/munet_v10_log-27da478484.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/41efeea8-e31e-4961-9e26-02e6babca6ab/munet_v10_params-27da478484.json",
            "train_script" : "train_edge_ffcv.py",
            "task": "edge_based_recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                 This is the 3rd instance of training the original architecture on edge-based reconstruction on 10k ImageNet Images (8k training, 2048 validation split) after 
                ensuring that there is matched input preprocessing transforms with the 1k model training (added random horizontal flip).
            """,
        },
    )
    
    SS_EDGE_8k_b_REP4 = Weights(
         url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/475c8526-c209-489e-9350-497dd3fb3638/munet_v10_final_weights-b3cdb08d68.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/475c8526-c209-489e-9350-497dd3fb3638/munet_v10_log-b3cdb08d68.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/475c8526-c209-489e-9350-497dd3fb3638/munet_v10_params-b3cdb08d68.json",
            "train_script" : "train_edge_ffcv.py",
            "task": "edge_based_recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 4th instance of training the original architecture on edge-based reconstruction on 10k ImageNet Images (8k training, 2048 validation split) after 
                ensuring that there is matched input preprocessing transforms with the 1k model training (added random horizontal flip).
            """,
        },
    )
    
    SS_EDGE_8k_b_REP5 = Weights(
         url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/68da028a-7bdd-4a6b-94d2-4dc19abb69cd/munet_v10_final_weights-63f8997f81.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/68da028a-7bdd-4a6b-94d2-4dc19abb69cd/munet_v10_log-63f8997f81.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/68da028a-7bdd-4a6b-94d2-4dc19abb69cd/munet_v10_params-63f8997f81.json",
            "train_script" : "train_edge_ffcv.py",
            "task": "edge_based_recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 5th instance of training the original architecture on edge-based reconstruction on 10k ImageNet Images (8k training, 2048 validation split) after 
                ensuring that there is matched input preprocessing transforms with the 1k model training (added random horizontal flip).
            """,
        },
    )
    
    SS_EDGE_8k_b_REP6 = Weights(
         url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/8cedcf08-d49c-490e-bd8c-ac14d633b7de/munet_v10_final_weights-353b10d343.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/8cedcf08-d49c-490e-bd8c-ac14d633b7de/munet_v10_log-353b10d343.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/8cedcf08-d49c-490e-bd8c-ac14d633b7de/munet_v10_params-353b10d343.json",
            "train_script" : "train_edge_ffcv.py",
            "task": "edge_based_recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 6th instance of training the original architecture on edge-based reconstruction on 10k ImageNet Images (8k training, 2048 validation split) after 
                ensuring that there is matched input preprocessing transforms with the 1k model training (added random horizontal flip).
            """,
        },
    )
    

    SS_EDGE_8k_b_REP7 = Weights(
         url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/bda040f8-535b-4591-811b-3c5dd569be4b/munet_v10_final_weights-d8c0d81f9d.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/bda040f8-535b-4591-811b-3c5dd569be4b/munet_v10_log-d8c0d81f9d.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/bda040f8-535b-4591-811b-3c5dd569be4b/munet_v10_params-d8c0d81f9d.json",
            "train_script" : "train_edge_ffcv.py",
            "task": "edge_based_recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
             This is the 7th instance of training the original architecture on edge-based reconstruction on 10k ImageNet Images (8k training, 2048 validation split) after 
             ensuring that there is matched input preprocessing transforms with the 1k model training (added random horizontal flip).
            """,
        },
    )
    

    SS_EDGE_8k_b_REP8 = Weights(
         url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/ecb4cf35-df08-4b61-aaa5-0bbc26572569/munet_v10_final_weights-bd162fbd18.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/ecb4cf35-df08-4b61-aaa5-0bbc26572569/munet_v10_log-bd162fbd18.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_b/ecb4cf35-df08-4b61-aaa5-0bbc26572569/munet_v10_params-bd162fbd18.json",
            "train_script" : "train_edge_ffcv.py",
            "task": "edge_based_recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 8th instance of training the original architecture on edge-based reconstruction on 10k ImageNet Images (8k training, 2048 validation split) after 
                ensuring that there is matched input preprocessing transforms with the 1k model training (added random horizontal flip).
            """,
        },
    )
    
    SS_EDGE_8k_c_REP1 = Weights(
         url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/0d2149b6-dba4-43aa-b35b-f89718d348e0/munet_v10_final_weights-0e70ad7077.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/0d2149b6-dba4-43aa-b35b-f89718d348e0/munet_v10_log-0e70ad7077.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/0d2149b6-dba4-43aa-b35b-f89718d348e0/munet_v10_params-0e70ad7077.json",
            "train_script" : "train_edge_ffcv.py",
            "task": "edge_based_recon",
            "dataset": "training_dataset5_8k_c_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 1st instance of training the original architecture on edge-based reconstruction on 10k ImageNet Images (8k_c training, 2048 validation split) after 
                ensuring that there is matched input preprocessing transforms with the 1k model training (added random horizontal flip).
            """,
        },
    )
    
    
    SS_EDGE_8k_c_REP2 = Weights(
         url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/0e5225bb-7c8c-4e1f-b11a-c4fc1e624002/munet_v10_final_weights-9a75d376ee.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/0e5225bb-7c8c-4e1f-b11a-c4fc1e624002/munet_v10_log-9a75d376ee.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/0e5225bb-7c8c-4e1f-b11a-c4fc1e624002/munet_v10_params-9a75d376ee.json",
            "train_script" : "train_edge_ffcv.py",
            "task": "edge_based_recon",
            "dataset": "training_dataset5_8k_c_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
               This is the 2nd instance of training the original architecture on edge-based reconstruction on 10k ImageNet Images (8k_c training, 2048 validation split) after 
               ensuring that there is matched input preprocessing transforms with the 1k model training (added random horizontal flip).
            """,
        },
    )
    
    
    SS_EDGE_8k_c_REP3 = Weights(
         url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/64d781c2-eecb-46e4-8d08-51df6cd62db4/munet_v10_final_weights-1d2c3a6cbf.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/64d781c2-eecb-46e4-8d08-51df6cd62db4/munet_v10_log-1d2c3a6cbf.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/64d781c2-eecb-46e4-8d08-51df6cd62db4/munet_v10_params-1d2c3a6cbf.json",
            "train_script" : "train_edge_ffcv.py",
            "task": "edge_based_recon",
            "dataset": "training_dataset5_8k_c_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 3rd instance of training the original architecture on edge-based reconstruction on 10k ImageNet Images (8k_c training, 2048 validation split) after 
                ensuring that there is matched input preprocessing transforms with the 1k model training (added random horizontal flip).
            """,
        },
    )
    
    SS_EDGE_8k_c_REP4 = Weights(
         url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/724dc817-83cc-41ba-8ae7-aa44bfe6b9e1/munet_v10_final_weights-3c0cbdf466.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/724dc817-83cc-41ba-8ae7-aa44bfe6b9e1/munet_v10_log-3c0cbdf466.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/724dc817-83cc-41ba-8ae7-aa44bfe6b9e1/munet_v10_params-3c0cbdf466.json",
            "train_script" : "train_edge_ffcv.py",
            "task": "edge_based_recon",
            "dataset": "training_dataset5_8k_c_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 4th instance of training the original architecture on edge-based reconstruction on 10k ImageNet Images (8k_c training, 2048 validation split) after 
                ensuring that there is matched input preprocessing transforms with the 1k model training (added random horizontal flip).
            """,
        },
    )
    
    
    SS_EDGE_8k_c_REP5 = Weights(
         url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/775b2d10-372d-4088-9f1f-de4d48c3c959/munet_v10_final_weights-ea6a6deaca.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/775b2d10-372d-4088-9f1f-de4d48c3c959/munet_v10_log-ea6a6deaca.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/775b2d10-372d-4088-9f1f-de4d48c3c959/munet_v10_params-ea6a6deaca.json",
            "train_script" : "train_edge_ffcv.py",
            "task": "edge_based_recon",
            "dataset": "training_dataset5_8k_c_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 5th instance of training the original architecture on edge-based reconstruction on 10k ImageNet Images (8k_c training, 2048 validation split) after 
                ensuring that there is matched input preprocessing transforms with the 1k model training (added random horizontal flip).
            """,
        },
    )
    
    SS_EDGE_8k_c_REP6 = Weights(
         url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/c02cb2cf-2427-4793-8c85-30efa1592110/munet_v10_final_weights-e9d8595f88.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/c02cb2cf-2427-4793-8c85-30efa1592110/munet_v10_log-e9d8595f88.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/c02cb2cf-2427-4793-8c85-30efa1592110/munet_v10_params-e9d8595f88.json",
            "train_script" : "train_edge_ffcv.py",
            "task": "edge_based_recon",
            "dataset": "training_dataset5_8k_c_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 6th instance of training the original architecture on edge-based reconstruction on 10k ImageNet Images (8k_c training, 2048 validation split) after 
                ensuring that there is matched input preprocessing transforms with the 1k model training (added random horizontal flip).
            """,
        },
    )
    

    SS_EDGE_8k_c_REP7 = Weights(
         url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/cc3f2bb1-9378-4cfc-ac47-d08439c8fb78/munet_v10_final_weights-d29ace4bef.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/cc3f2bb1-9378-4cfc-ac47-d08439c8fb78/munet_v10_log-d29ace4bef.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/cc3f2bb1-9378-4cfc-ac47-d08439c8fb78/munet_v10_params-d29ace4bef.json",
            "train_script" : "train_edge_ffcv.py",
            "task": "edge_based_recon",
            "dataset": "training_dataset5_8k_c_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 7th instance of training the original architecture on edge-based reconstruction on 10k ImageNet Images (8k_c training, 2048 validation split) after 
                ensuring that there is matched input preprocessing transforms with the 1k model training (added random horizontal flip).
            """,
        },
    )

    SS_EDGE_8k_c_REP8 = Weights(
         url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/dac62a8a-d60f-4872-a4d4-c7520ee910dc/munet_v10_final_weights-c8fb3f2115.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/dac62a8a-d60f-4872-a4d4-c7520ee910dc/munet_v10_log-c8fb3f2115.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_edge_ffcv_hor_flip/8k_c/dac62a8a-d60f-4872-a4d4-c7520ee910dc/munet_v10_params-c8fb3f2115.json",
            "train_script" : "train_edge_ffcv.py",
            "task": "edge_based_recon",
            "dataset": "training_dataset5_8k_c_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
            This is the 8th instance of training the original architecture on edge-based reconstruction on 10k ImageNet Images (8k_c training, 2048 validation split) after 
            ensuring that there is matched input preprocessing transforms with the 1k model training (added random horizontal flip).
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_10_DS1_REP1 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/01db484c-7a3a-454b-a148-b9e4ee876381/munet_v10_final_weights-5e450ce160.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/01db484c-7a3a-454b-a148-b9e4ee876381/munet_v10_log-5e450ce160.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/01db484c-7a3a-454b-a148-b9e4ee876381/munet_v10_params-5e450ce160.json",
            "train_script" : "train_denoise_v4.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_1k_dataset1",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 1st instance of an array of 8 models trained on the dataset of 1k_a with noise added with standard deviation of 0.10, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_10_DS1_REP2 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/694d737d-cfd2-44f4-823e-ccaf741e0b6f/munet_v10_final_weights-c35145bd9e.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/694d737d-cfd2-44f4-823e-ccaf741e0b6f/munet_v10_log-c35145bd9e.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/694d737d-cfd2-44f4-823e-ccaf741e0b6f/munet_v10_params-c35145bd9e.json",
            "train_script" : "train_denoise_v4.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_1k_dataset1",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 2nd instance of an array of 8 models trained on the dataset of 1k_a with noise added with standard deviation of 0.10, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    
    SS_GAUS_DENOISE_v4_10_DS1_REP3 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/703973c3-82b1-4d69-bf11-d02e21539909/munet_v10_final_weights-56e6b347af.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/703973c3-82b1-4d69-bf11-d02e21539909/munet_v10_log-56e6b347af.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/703973c3-82b1-4d69-bf11-d02e21539909/munet_v10_params-56e6b347af.json",
            "train_script" : "train_denoise_v4.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_1k_dataset1",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 3nd instance of an array of 8 models trained on the dataset of 1k_a with noise added with standard deviation of 0.10, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_10_DS1_REP4 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/71d24215-acf6-44a1-b42d-f1df7e0d9907/munet_v10_final_weights-8b25d70c09.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/71d24215-acf6-44a1-b42d-f1df7e0d9907/munet_v10_log-8b25d70c09.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/71d24215-acf6-44a1-b42d-f1df7e0d9907/munet_v10_params-8b25d70c09.json",
            "train_script" : "train_denoise_v4.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_1k_dataset1",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 4th instance of an array of 8 models trained on the dataset of 1k_a with noise added with standard deviation of 0.10, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    
    SS_GAUS_DENOISE_v4_10_DS1_REP5 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/775bdc7c-70d1-48e0-889c-eaa508e19cfd/munet_v10_final_weights-a7de33c86d.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/775bdc7c-70d1-48e0-889c-eaa508e19cfd/munet_v10_log-a7de33c86d.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/775bdc7c-70d1-48e0-889c-eaa508e19cfd/munet_v10_params-a7de33c86d.json",
            "train_script" : "train_denoise_v4.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_1k_dataset1",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 5th instance of an array of 8 models trained on the dataset of 1k_a with noise added with standard deviation of 0.10, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_10_DS1_REP6 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/b013a3bf-fee3-433b-9baf-ed6da7982065/munet_v10_final_weights-99684c7471.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/b013a3bf-fee3-433b-9baf-ed6da7982065/munet_v10_log-99684c7471.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/b013a3bf-fee3-433b-9baf-ed6da7982065/munet_v10_params-99684c7471.json",
            "train_script" : "train_denoise_v4.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_1k_dataset1",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 6th instance of an array of 8 models trained on the dataset of 1k_a with noise added with standard deviation of 0.10, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
   
    
    SS_GAUS_DENOISE_v4_10_DS1_REP7 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/b683032a-ae39-4aaa-890b-a525b08246ea/munet_v10_final_weights-38ac2345c7.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/b683032a-ae39-4aaa-890b-a525b08246ea/munet_v10_log-38ac2345c7.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/b683032a-ae39-4aaa-890b-a525b08246ea/munet_v10_params-38ac2345c7.json",
            "train_script" : "train_denoise_v4.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_1k_dataset1",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 7th instance of an array of 8 models trained on the dataset of 1k_a with noise added with standard deviation of 0.10, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_10_DS1_REP8 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/d60ffde4-3678-4f94-b243-1c413003e7d5/munet_v10_final_weights-e6f81e23cd.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/d60ffde4-3678-4f94-b243-1c413003e7d5/munet_v10_log-e6f81e23cd.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_10/d60ffde4-3678-4f94-b243-1c413003e7d5/munet_v10_params-e6f81e23cd.json",
            "train_script" : "train_denoise_v4.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_1k_dataset1",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 8th instance of an array of 8 models trained on the dataset of 1k_a with noise added with standard deviation of 0.10, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    

    SS_GAUS_DENOISE_v4_30_DS1_REP1 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/10b68660-0fd6-4107-94c7-b14ff9bf4db2/munet_v10_final_weights-477e3dd4c8.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/10b68660-0fd6-4107-94c7-b14ff9bf4db2/munet_v10_log-477e3dd4c8.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/10b68660-0fd6-4107-94c7-b14ff9bf4db2/munet_v10_params-477e3dd4c8.json",
            "train_script" : "train_denoise_v4.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_1k_dataset1",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 1st instance of an array of 8 models trained on the dataset of 1k_a with noise added with standard deviation of 0.30, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_30_DS1_REP2 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/64519651-1a01-43dd-90c7-ae963bd6f222/munet_v10_final_weights-8878f6af6c.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/64519651-1a01-43dd-90c7-ae963bd6f222/munet_v10_log-8878f6af6c.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/64519651-1a01-43dd-90c7-ae963bd6f222/munet_v10_params-8878f6af6c.json",
            "train_script" : "train_denoise_v4.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_1k_dataset1",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 2nd instance of an array of 8 models trained on the dataset of 1k_a with noise added with standard deviation of 0.30, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    

    SS_GAUS_DENOISE_v4_30_DS1_REP3 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/64ca9053-c1e1-40cb-ba92-27e47d7d3356/munet_v10_final_weights-23bcb52c0d.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/64ca9053-c1e1-40cb-ba92-27e47d7d3356/munet_v10_log-23bcb52c0d.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/64ca9053-c1e1-40cb-ba92-27e47d7d3356/munet_v10_params-23bcb52c0d.json",
            "train_script" : "train_denoise_v4.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_1k_dataset1",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 3rd instance of an array of 8 models trained on the dataset of 1k_a with noise added with standard deviation of 0.30, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_30_DS1_REP4 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/66909a8b-af47-4093-b4ef-cc684d8ea695/munet_v10_final_weights-cf0114b66c.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/66909a8b-af47-4093-b4ef-cc684d8ea695/munet_v10_log-cf0114b66c.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/66909a8b-af47-4093-b4ef-cc684d8ea695/munet_v10_params-cf0114b66c.json",
            "train_script" : "train_denoise_v4.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_1k_dataset1",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 4th instance of an array of 8 models trained on the dataset of 1k_a with noise added with standard deviation of 0.30, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_30_DS1_REP5 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/846f57d8-679f-45f2-9abd-f26477427ce4/munet_v10_final_weights-e24475e985.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/846f57d8-679f-45f2-9abd-f26477427ce4/munet_v10_log-e24475e985.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/846f57d8-679f-45f2-9abd-f26477427ce4/munet_v10_params-e24475e985.json",
            "train_script" : "train_denoise_v4.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_1k_dataset1",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 5th instance of an array of 8 models trained on the dataset of 1k_a with noise added with standard deviation of 0.30, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    

    SS_GAUS_DENOISE_v4_30_DS1_REP6 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/98610a0d-ddd8-4f6f-8c9e-4c5509445920/munet_v10_final_weights-912cbe08eb.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/98610a0d-ddd8-4f6f-8c9e-4c5509445920/munet_v10_log-912cbe08eb.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/98610a0d-ddd8-4f6f-8c9e-4c5509445920/munet_v10_params-912cbe08eb.json",
            "train_script" : "train_denoise_v4.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_1k_dataset1",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 6th instance of an array of 8 models trained on the dataset of 1k_a with noise added with standard deviation of 0.30, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    

    SS_GAUS_DENOISE_v4_30_DS1_REP7 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/9a705384-5088-4f4b-ba72-f531f5f3e124/munet_v10_final_weights-9a8a58b7a6.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/9a705384-5088-4f4b-ba72-f531f5f3e124/munet_v10_log-9a8a58b7a6.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/9a705384-5088-4f4b-ba72-f531f5f3e124/munet_v10_params-9a8a58b7a6.json",
            "train_script" : "train_denoise_v4.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_1k_dataset1",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 7th instance of an array of 8 models trained on the dataset of 1k_a with noise added with standard deviation of 0.30, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_30_DS1_REP8 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/e23d0738-a2e1-4f1d-8bf1-42fb7e5e0d38/munet_v10_final_weights-422e9165be.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/e23d0738-a2e1-4f1d-8bf1-42fb7e5e0d38/munet_v10_log-422e9165be.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/1k_a/std_30/e23d0738-a2e1-4f1d-8bf1-42fb7e5e0d38/munet_v10_params-422e9165be.json",
            "train_script" : "train_denoise_v4.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_1k_dataset1",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 8th instance of an array of 8 models trained on the dataset of 1k_a with noise added with standard deviation of 0.30, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
   

    SS_GAUS_DENOISE_v4_10_8k_b_REP1 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/3bef83fd-bae8-4df4-aaee-a0b87c1fb3bf/munet_v10_final_weights-df0e6b2cfa.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/3bef83fd-bae8-4df4-aaee-a0b87c1fb3bf/munet_v10_log-df0e6b2cfa.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/3bef83fd-bae8-4df4-aaee-a0b87c1fb3bf/munet_v10_params-df0e6b2cfa.json",
            "train_script" : "train_denoise_ffcv_v1.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 1st instance of an array of 8 models trained on the dataset of 8k_b with noise added with standard deviation of 0.10, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )

    SS_GAUS_DENOISE_v4_10_8k_b_REP2 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/b95b2600-cfba-414a-9aaf-f40e18406ec9/munet_v10_final_weights-b2fae716d3.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/b95b2600-cfba-414a-9aaf-f40e18406ec9/munet_v10_log-b2fae716d3.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/b95b2600-cfba-414a-9aaf-f40e18406ec9/munet_v10_params-b2fae716d3.json",
            "train_script" : "train_denoise_ffcv_v1.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 2nd instance of an array of 8 models trained on the dataset of 8k_b with noise added with standard deviation of 0.10, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_10_8k_b_REP3 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/0309199e-b718-431c-ac77-86befc885a29/munet_v10_final_weights-84fa1f0fa0.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/0309199e-b718-431c-ac77-86befc885a29/munet_v10_log-84fa1f0fa0.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/0309199e-b718-431c-ac77-86befc885a29/munet_v10_params-84fa1f0fa0.json",
            "train_script" : "train_denoise_ffcv_v1.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 3rd instance of an array of 8 models trained on the dataset of 8k_b with noise added with standard deviation of 0.10, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_10_8k_b_REP4 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/07380ca2-77d1-49ca-a147-8a76320bd8e7/munet_v10_final_weights-0fc1488c74.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/07380ca2-77d1-49ca-a147-8a76320bd8e7/munet_v10_log-0fc1488c74.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/07380ca2-77d1-49ca-a147-8a76320bd8e7/munet_v10_params-0fc1488c74.json",
            "train_script" : "train_denoise_ffcv_v1.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 4th instance of an array of 8 models trained on the dataset of 8k_b with noise added with standard deviation of 0.10, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_10_8k_b_REP5 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/14cbfb19-b4e5-4827-a683-1632ef86c34f/munet_v10_final_weights-df96206eae.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/14cbfb19-b4e5-4827-a683-1632ef86c34f/munet_v10_log-df96206eae.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/14cbfb19-b4e5-4827-a683-1632ef86c34f/munet_v10_params-df96206eae.json",
            "train_script" : "train_denoise_ffcv_v1.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 5th instance of an array of 8 models trained on the dataset of 8k_b with noise added with standard deviation of 0.10, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_10_8k_b_REP6 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/51afd6df-97e5-4d1e-9434-8dc410f0b3e9/munet_v10_final_weights-f4ef5053de.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/51afd6df-97e5-4d1e-9434-8dc410f0b3e9/munet_v10_log-f4ef5053de.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/51afd6df-97e5-4d1e-9434-8dc410f0b3e9/munet_v10_params-f4ef5053de.json",
            "train_script" : "train_denoise_ffcv_v1.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 6th instance of an array of 8 models trained on the dataset of 8k_b with noise added with standard deviation of 0.10, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_10_8k_b_REP7 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/9f0e3d21-1d5d-443f-a38d-d25e0d03cb6f/munet_v10_final_weights-a969b988c2.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/9f0e3d21-1d5d-443f-a38d-d25e0d03cb6f/munet_v10_log-a969b988c2.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/9f0e3d21-1d5d-443f-a38d-d25e0d03cb6f/munet_v10_params-a969b988c2.json",
            "train_script" : "train_denoise_ffcv_v1.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 7th instance of an array of 8 models trained on the dataset of 8k_b with noise added with standard deviation of 0.10, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_10_8k_b_REP8 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/a8bd1596-9ff5-4858-8e58-3a70468bb485/munet_v10_final_weights-a7989a5199.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/a8bd1596-9ff5-4858-8e58-3a70468bb485/munet_v10_log-a7989a5199.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_10/a8bd1596-9ff5-4858-8e58-3a70468bb485/munet_v10_params-a7989a5199.json",
            "train_script" : "train_denoise_ffcv_v1.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 8th instance of an array of 8 models trained on the dataset of 8k_b with noise added with standard deviation of 0.10, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    

    SS_GAUS_DENOISE_v4_30_8k_b_REP1 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/5d421f86-f270-458e-8fa3-5a0689b2f08c/munet_v10_final_weights-f9803d3d2e.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/5d421f86-f270-458e-8fa3-5a0689b2f08c/munet_v10_log-f9803d3d2e.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/5d421f86-f270-458e-8fa3-5a0689b2f08c/munet_v10_params-f9803d3d2e.json",
            "train_script" : "train_denoise_ffcv_v1.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 1st instance of an array of 8 models trained on the dataset of 8k_b with noise added with standard deviation of 0.30, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_30_8k_b_REP2 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/db2d0a1b-8c9a-423a-8d86-8df0bc9e2776/munet_v10_final_weights-c53cbc2022.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/db2d0a1b-8c9a-423a-8d86-8df0bc9e2776/munet_v10_log-c53cbc2022.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/db2d0a1b-8c9a-423a-8d86-8df0bc9e2776/munet_v10_params-c53cbc2022.json",
            "train_script" : "train_denoise_ffcv_v1.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 2nd instance of an array of 8 models trained on the dataset of 8k_b with noise added with standard deviation of 0.30, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_30_8k_b_REP3 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/1008ef68-c49d-46dc-8604-6b6ace39f271/munet_v10_final_weights-3d8e7c7a1b.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/1008ef68-c49d-46dc-8604-6b6ace39f271/munet_v10_log-3d8e7c7a1b.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/1008ef68-c49d-46dc-8604-6b6ace39f271/munet_v10_params-3d8e7c7a1b.json",
            "train_script" : "train_denoise_ffcv_v1.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 3rd instance of an array of 8 models trained on the dataset of 8k_b with noise added with standard deviation of 0.30, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_30_8k_b_REP4 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/4ae0b442-6787-4c35-b6db-c2330d750de3/munet_v10_final_weights-a2ef1f8e9e.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/4ae0b442-6787-4c35-b6db-c2330d750de3/munet_v10_log-a2ef1f8e9e.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/4ae0b442-6787-4c35-b6db-c2330d750de3/munet_v10_params-a2ef1f8e9e.json",
            "train_script" : "train_denoise_ffcv_v1.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 4th instance of an array of 8 models trained on the dataset of 8k_b with noise added with standard deviation of 0.30, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_30_8k_b_REP5 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/7c06487d-0892-415b-89f6-2c643f7b7c8c/munet_v10_final_weights-0759c0ff72.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/7c06487d-0892-415b-89f6-2c643f7b7c8c/munet_v10_log-0759c0ff72.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/7c06487d-0892-415b-89f6-2c643f7b7c8c/munet_v10_params-0759c0ff72.json",
            "train_script" : "train_denoise_ffcv_v1.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 5th instance of an array of 8 models trained on the dataset of 8k_b with noise added with standard deviation of 0.30, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_30_8k_b_REP6 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/b43ef056-cee1-44a5-84e1-e8a0f449a398/munet_v10_final_weights-6924a3f84b.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/b43ef056-cee1-44a5-84e1-e8a0f449a398/munet_v10_log-6924a3f84b.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/b43ef056-cee1-44a5-84e1-e8a0f449a398/munet_v10_params-6924a3f84b.json",
            "train_script" : "train_denoise_ffcv_v1.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 6th instance of an array of 8 models trained on the dataset of 8k_b with noise added with standard deviation of 0.30, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_30_8k_b_REP7 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/bd7e8b4b-1586-4042-8383-babeaeb292c5/munet_v10_final_weights-43aefee2f1.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/bd7e8b4b-1586-4042-8383-babeaeb292c5/munet_v10_log-43aefee2f1.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/bd7e8b4b-1586-4042-8383-babeaeb292c5/munet_v10_params-43aefee2f1.json",
            "train_script" : "train_denoise_ffcv_v1.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 7th instance of an array of 8 models trained on the dataset of 8k_b with noise added with standard deviation of 0.30, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
    
    SS_GAUS_DENOISE_v4_30_8k_b_REP8 = Weights(
        url='https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/e23011ed-4f5f-4abf-a45f-ab9686264ee1/munet_v10_final_weights-bb9bcee8c7.pth',
        transforms = transforms,
        meta = {
            "repo": "https://github.com/srij-saha/lightnesslayers",
            "log_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/e23011ed-4f5f-4abf-a45f-ab9686264ee1/munet_v10_log-bb9bcee8c7.txt",
            "params_url": "https://s3.us-east-1.wasabisys.com/visionlab-members/srijanisaha/Projects/lightnesslayers/logs/debug/munet_v10_denoise/gaussian_v4/8k_b/std_30/e23011ed-4f5f-4abf-a45f-ab9686264ee1/munet_v10_params-bb9bcee8c7.json",
            "train_script" : "train_denoise_ffcv_v1.py",
            "task": "denoise-gaussian-recon",
            "dataset": "training_dataset5_8k_b_ffcv",
            "arch" : "dynamic_mcnn_vanilla_encoder_512",
            "num_params": None,
            "_docs": """
                This is the 8th instance of an array of 8 models trained on the dataset of 8k_b with noise added with standard deviation of 0.30, 
                making sure noise is added during the validation phase. Original image was scaled between -1 and 1 and the input to the 
                model was clipped between -1 to 1.
            """,
        },
    )
        
    DEFAULT = GAA_ORIG
    
def munet_v10(weights=None, dev1="cuda:0", dev2="cuda:0", progress=True):
    
    encoder = VanillaEncoder512(projection=False, inplace=True)
    model = UNetModelParallel(DynamicMCNN(encoder, n_out=1, img_size=(512,512), self_attention=True),
                              split_size=1, dev1=dev1, dev2=dev2)
    
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))
        
    return model