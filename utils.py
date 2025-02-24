import torch

from torchvision.transforms.v2 import Compose,Resize,RandomHorizontalFlip,RandomAffine,ToDtype,PILToTensor,ToImage,Normalize,Lambda
from torchvision import tv_tensors
from torchvision.transforms.autoaugment import RandAugment
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode
from torch import Tensor

from torch.utils.data.dataset import Subset

from datasets import labelled_TensorDataset


import torch


from typing import Dict, List, Optional, Tuple

from numpy import inf

import pickle

from itertools import cycle


RED    = "\33[31m"
BLUE   = "\33[34m"
GREEN  = "\33[32m"
NORMAL = "\33[0m"

def create_weak_aug(size:Tuple[int,int]) -> Compose:
    """
    Function for weak augmentations.
    size : tuple of int for deciding the new size of the image

    returns a tensor at the end
    """
    transform = Compose(
        [
            ToImage(),
            Resize(size = size, antialias=True),
            RandomHorizontalFlip(),
            RandomAffine(degrees=0,translate=(0.125,0.125)),
            #PILToTensor(),
            Lambda(lambda x: x.float()/255.0),
            ToDtype(torch.float32,
                    scale=True,
                ), 
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # ToDtype(dtype={tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64})
        ]
    )
    return transform


def create_strong_aug(size:Tuple[int,int]) -> Compose:
    """
    Function for strong augmentations, as it is defined below.
    size : tuple of int for deciding the new size of the image

    returns a tensor at the end
    """
    augmenter = OurRandAug(magnitude=torch.randint(5,10,size=(1,)))
    transform = Compose(
        [   
            ToImage(),
            Resize(size=size, antialias=True),
            augmenter,
            Lambda(lambda x: x.float()/255.0),
            ToDtype(torch.float32,
                    scale=True,
                ), 
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        ]
    )
    return transform

def create_valid_transform(size:Tuple[int,int]) -> Compose:
    transform = Compose(
        [   
            ToImage(),
            Resize(size=size, antialias=True),
            Lambda(lambda x: x.float()/255.0),
            ToDtype(torch.float32,
                    scale=True,
                ), 
            
        ]
    )
    return transform


class OurRandAug(RandAugment):
    """
    Just to adapt all the transformation parameters to the one in the FixMatch Paper:
    https://arxiv.org/abs/2001.07685
    
    """
    def __init__(
        self,
        num_ops: int = 4,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(-0.3, 0.3, num_bins), True),
            "ShearY": (torch.linspace(-0.3, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 0.3 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 0.3 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(-30.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.05, 0.95, num_bins), True),
            "Color": (torch.linspace(0.05, 0.95, num_bins), True),
            "Contrast": (torch.linspace(0.05, 0.95, num_bins), True),
            "Sharpness": (torch.linspace(0.05, 0.95, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(1.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

class info_nce(torch.nn.Module):
    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, query, key):
        query_normalized = torch.nn.functional.normalize(query,dim=1)
        key_normalized = torch.nn.functional.normalize(key,dim=1)

        predicted = query_normalized @ key_normalized.T
        predicted.div_(self.temperature)

        labels = torch.arange(len(query),device=query.device) #gives int labels that would be used in the cross_entropy

        return torch.nn.functional.cross_entropy(predicted,labels)

def fixmatch_loss(model,label_batch,unlabel_batch,threshold,weight,criterion):
    """
    Assumes label_batch is [Tensor(B,C,H,W),labels] and unlabel_batch is [weak_batch,strong_batch]
    as done with datasets defined in other files.
    """
    bad_index = criterion.ignore_index

    pred_label = model(label_batch[0])#predictions for labelled set, unormalized logits always
    _,labels_model = pred_label.max(dim=1)

    loss_s = criterion(pred_label,label_batch[1]).mean()

    acc = (labels_model == label_batch[1]).sum().item()

    pred_unlabel = model(unlabel_batch[0]).softmax(dim=1)#predictions on weak augmented unlabel set
    logits, labels = pred_unlabel.max(dim=1)

    predstrong_unlabel = model(unlabel_batch[1])

    _,labels_strong = pred_unlabel.max(dim=1) 

    acc += (labels_strong == labels).sum().item()

    labels.masked_fill_(logits < threshold,bad_index)
    loss_u = criterion(predstrong_unlabel,labels)
    loss_u[torch.isnan(loss_u)]=0

    if (loss_u.isnan()).sum():
        print('nan')
        return loss_s,acc

    return loss_s + weight*loss_u.mean(),acc
    
def labelset_split(label_set, n_per_label : int = 100):
    """
    Given a labelled_TensorDataset and a number of instances for each label, return 2 labelled_TensorDatasets, the first one with said label distributions and a 2nd one generally used as a validation set.
    """
    mask0 = label_set.tensors[1] == 0
    mask1 = label_set.tensors[1] == 1

    ind_0 = torch.nonzero(mask0)
    ind_1 = torch.nonzero(mask1)

    ind_0r = torch.randperm(ind_0.size(0)).tolist()
    ind_1r = torch.randperm(ind_1.size(0)).tolist()

    return [Subset(label_set,ind_0[ind_0r[:n_per_label]].tolist() + ind_1[ind_1r[:n_per_label]].tolist()),
            Subset(label_set,ind_0[ind_0r[n_per_label:]].tolist() + ind_1[ind_1r[n_per_label:]].tolist())]



    
        

def train(
        model,
        label_loader,
        unlabel_loader,
        optim,
        criterion,
        device,
        threshold,
        loss_weight,
        verbose= False
    ) -> Tuple[float, float]:
    
    model.train()
    model.to(device)
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    total_length_data = len(unlabel_loader) 
    for batch_idx, (label_batch, unlabel_batch) in enumerate(zip(cycle(label_loader),unlabel_loader)):
        optim.zero_grad()

        # Get images and labels
        label_batch = label_batch[0].to(device),label_batch[1].to(device)
        unlabel_batch = unlabel_batch[0].to(device),unlabel_batch[1].to(device)

        # Forward propagation
        loss,acc = fixmatch_loss(model,label_batch,unlabel_batch,threshold,loss_weight,criterion)
        
        # Backward pass
        loss.backward()

        optim.step()

        epoch_loss += loss.item()
        epoch_accuracy += acc
        
        if verbose:
            header = f"{GREEN}[Step: {batch_idx+1}/{total_length_data}]{NORMAL}"
            print(f"\r{header} Batch avg loss: {loss.item() / len(label_loader):.4f}", end="")

    if verbose:
        print("\r", end="")

    return epoch_loss / total_length_data, epoch_accuracy / total_length_data

def train_nofix(
        model,
        label_loader,
        optim,
        criterion,
        device,
        verbose= False
    ) -> Tuple[float, float]:
    
    model.train()
    model.to(device)
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    total = 0.0
    
    total_length_data = len(label_loader) 
    for batch_idx, label_batch in enumerate(label_loader):
        optim.zero_grad()

        # Get images and labels
        label_batch = label_batch[0].to(device),label_batch[1].to(device)

        # Forward propagation
        pred = model(label_batch[0])
        _,pred_labels = pred.max(dim=1)

        loss = criterion(pred,label_batch[1]).mean()
        acc = (pred_labels == label_batch[1]).sum().item()
        # Backward pass
        loss.backward()

        optim.step()

        epoch_loss += loss.item()
        epoch_accuracy += acc
        total += len(label_batch)

        if verbose:
            header = f"{GREEN}[Step: {batch_idx+1}/{len(label_loader)}]{NORMAL}"
            print(f"\r{header} Batch avg loss: {loss.item() / len(label_batch):.4f}", end="")

    if verbose:
        print("\r", end="")

    return epoch_loss / total_length_data, epoch_accuracy / total

def validate(
        model,
        loader,
        criterion,
        device,
        verbose
    ) -> Tuple[float, float]:

    model.eval()

    epoch_loss = 0.0
    epoch_accuracy = 0.0
    total = 0.0
    for step, batch_data in enumerate(loader):
        # Get images and labels
        images = batch_data[0].type(torch.float32).to(device)
        labels = batch_data[1].to(device)

        with torch.no_grad():
            # Forward propagation
            #outputs = model(images).squeeze()
            pred= model(images)#predictions on weak augmented unlabel set
            pred_soft = pred.softmax(dim=1)
            logits, labels_model = pred_soft.max(dim=1)
            # Loss computation
            loss = criterion(pred, labels).mean()

        epoch_loss += loss.item()
        epoch_accuracy += (labels_model == labels).sum().item()
        total += len(batch_data)
        if verbose:
            header = f"{GREEN}[Step: {step+1}/{len(loader)}]{NORMAL}"
            print(f"\r{header} Batch avg loss: {loss.item() / len(batch_data):.4f}", end="")

    if verbose:
        print("\r", end="")

    return epoch_loss / len(loader), epoch_accuracy / total

def epoch_loop(model, 
        label_loader,
        unlabel_loader, 
        valid_loader, 
        optimizer,
        criterion, 
        device,
        num_epochs,
        threshold,
        loss_weight,
        save_path,
        start_epoch = 0,
        best_val_loss = None,
        verbose = True):
    t_history = []
    v_history = []
    print('Fixmatch with threshold: ',threshold)
    if best_val_loss==None:
        best_loss = float("inf")

    for epoch in range(start_epoch,num_epochs):
        header = f"{BLUE}[Epoch: {epoch+1}/{num_epochs}]{NORMAL}"

        
        t_loss, t_accuracy = train(model,label_loader,unlabel_loader,optimizer,criterion,device,threshold,loss_weight,verbose=verbose)
        

        t_history.append((t_loss, t_accuracy))
        with open("t_history.pickle", "wb") as handle:
            pickle.dump(t_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if (epoch+1) % 1 == 0:
            
            v_loss, v_accuracy = validate(model,valid_loader,criterion,device,verbose)
            

            v_history.append((v_loss, v_accuracy))
            with open("v_history.pickle", "wb") as handle:
                pickle.dump(v_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if v_loss < best_loss:
                print(f"{header} Training")
                print(f"{header} Avg loss: {t_loss:.4f} | Accuracy: {t_accuracy:.4f}")
                print(f"{header} Validation")
                print(f"{header} Avg loss: {v_loss:.4f} | Accuracy: {v_accuracy:.4f}")
                best_loss = v_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": v_loss
                }, save_path)

                print(f"\n{GREEN}Best model so far. Saving model as model.pth{NORMAL}\n")

def epoch_loop_nofix(model, 
        label_loader,
        valid_loader, 
        optimizer,
        criterion, 
        device, 
        num_epochs,
        save_path,
        best_val_loss = None,
        verbose = True):
    t_history = []
    v_history = []
    if best_val_loss==None:
        best_loss = float("inf")

    for epoch in range(num_epochs):
        header = f"{BLUE}[Epoch: {epoch+1}/{num_epochs}]{NORMAL}"

        
        t_loss, t_accuracy = train_nofix(model,label_loader,optimizer,criterion,device,verbose=False)
        if verbose:
            print(f"{header} Training")
            print(f"{header} Avg loss: {t_loss:.4f} | Accuracy: {t_accuracy:.4f}")

        t_history.append((t_loss, t_accuracy))
        with open("t_history.pickle", "wb") as handle:
            pickle.dump(t_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if (epoch+1) % 1 == 0:
            
            v_loss, v_accuracy = validate(model,valid_loader,criterion,device,verbose=False)
            if verbose:
                print(f"{header} Validation")
                print(f"{header} Avg loss: {v_loss:.4f} | Accuracy: {v_accuracy:.4f}")

            v_history.append((v_loss, v_accuracy))
            with open("v_history.pickle", "wb") as handle:
                pickle.dump(v_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if v_loss < best_loss:
                best_loss = v_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": v_loss,
                    "training_history": t_history,
                    "validation_history": v_history,
                }, save_path)
                if verbose:
                    print(f"\n{GREEN}Best model so far. Saving model as model.pth{NORMAL}\n")


def train_enc(
        model,
        unlabel_loader,
        optim,
        criterion,
        device,
        verbose= False
    ) -> Tuple[float, float]:
    
    model.train()
    model.to(device)
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    
    total_length_data = len(unlabel_loader)
    for batch_idx, (normal_batch, augmented_batch) in enumerate(unlabel_loader):
        optim.zero_grad()

        # Get images and labels
        normal_batch = normal_batch.to(device)
        augmented_batch = augmented_batch.to(device)

        queries = model(normal_batch)
        keys = model(augmented_batch)
        # Forward propagation
        loss = criterion(queries,keys)
        
        # Backward pass
        loss.backward()

        optim.step()

        epoch_loss += loss.item()
        #epoch_accuracy += acc

        if verbose:
            header = f"{GREEN}[Step: {batch_idx+1}/{total_length_data}]{NORMAL}"
            print(f"\r{header} Batch avg loss: {loss.item() / len(queries):.4f}", end="")

    if verbose:
        print("\r", end="")

    return epoch_loss / total_length_data, epoch_accuracy / total_length_data


def epoch_loop_encode(model, 
        unlabel_loader, 
        optimizer,
        criterion, 
        device, 
        num_epochs,
        save_path,
        best_val_loss = None,
        verbose = True):
    t_history = []
    v_history = []
    if best_val_loss==None:
        best_loss = float("inf")

    for epoch in range(num_epochs):
        header = f"{BLUE}[Epoch: {epoch+1}/{num_epochs}]{NORMAL}"

        
        t_loss, t_accuracy = train_enc(model,unlabel_loader,optimizer,criterion,device,verbose=False)
        

        t_history.append((t_loss, t_accuracy))
        with open("t_history.pickle", "wb") as handle:
            pickle.dump(t_history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if t_loss < best_loss:
                print(f"{header} Training")
                print(f"{header} Avg loss: {t_loss:.4f}")
                best_loss = t_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": t_loss,
                    "loss_hist":t_history,
                }, save_path)

                print(f"\n{GREEN}Best model so far. Saving model as model.pth{NORMAL}\n")