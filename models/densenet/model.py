import torch
from fastai.data.all import *
from fastai.vision.all import *

# Define the same custom head used during training
class CustomHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hd = nn.Sequential(
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(p = 0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace = True),
            nn.BatchNorm1d(512),
            nn.Dropout(p = 0.2),
            nn.Linear(512, out_features),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.hd(x)


def load_model(model_name, bitsize, data_path='./data/'):
    """
    Loads the saved model weights with name from saved_weights directory and 
    returns the initialized model

    Args:
        name: The name of the saved path file (.pth)
        bitsize: The bitsize of the generated hash for the model

    Returns:
        The DenseNet model instance loaded with the specified weights 
    """

    # Load the model architecture
    model = models.densenet161(weights=None)  # weights=None for no pretrained weights

    # Replace the head with your custom head
    model.classifier = CustomHead(4416, bitsize)

    # Load the saved state dictionary
    state_dict = torch.load(
        './models/densenet/saved_weights/' + model_name, 
        map_location = torch.device('cuda'))
    
    # Modify the state_dict keys
    modified_state_dict = {}
    for k, v in state_dict.items(): 
        if k.startswith('0.0.'):
            k = 'features.' + k[4:]  # Replace '0.0.' with 'features.'
        elif k.startswith('1.'):
            k = 'classifier.' + k[2:]  # Replace '1.' with 'classifier.'
        modified_state_dict[k] = v

    # Load the state dictionary into the model
    model.load_state_dict(modified_state_dict)

    # Move the model to the GPU
    model = model.to(torch.device('cuda'))

    # Set the model to evaluation mode
    model.eval()

    # --- Create a FastAI Learner ---

    data_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(),
        item_tfms=Resize(224),
        get_y=parent_label
    )

    # Create DataLoaders
    dls = data_block.dataloaders(data_path)

    # Create a Learner object
    learner = Learner(dls, model, metrics=error_rate)

    return learner
