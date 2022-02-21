# head dims:512,512,512,512,512,512,512,512,128
# code is basicly:https://github.com/google-research/deep_representation_one_class
from pathlib import Path
from tqdm import tqdm
import datetime
import argparse
import random 
import numpy as np
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



from dataset import MVTecAT, Repeat, chexpert_dataset,ZhanglabWithLabel
from cutpaste import AnatMix, CutPasteNormal,CutPasteScar, CutPaste3Way, CutPasteUnion,AnatMix, cut_paste_collate_fn
from model import ProjectionNet
from eval import eval_model
from utils import str2bool

def run_training(data_type="zhanglab",
                 model_dir="models",
                 epochs=256,
                 pretrained=True,
                 test_epochs=10,
                 freeze_resnet=20,
                 learninig_rate=0.03,
                 optim_name="SGD",
                 batch_size=64,
                 head_layer=8,
                 cutpate_type=CutPasteNormal,
                 device = "cuda",
                 workers=8,
                 size = 256,
                 seed = 0):
    torch.multiprocessing.freeze_support()
    # TODO: use script params for hyperparameter
    # Temperature Hyperparameter currently not used
    temperature = 0.2

    weight_decay = 0.00003
    momentum = 0.9
    #TODO: use f strings also for the date LOL
    model_name = f"model-{data_type}-{cutpate_type.__name__}-seed_{seed}" + '-{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now() )


    #augmentation:
    min_scale = 1
    g = torch.Generator()
    g.manual_seed(seed)

    # create Training Dataset and Dataloader
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.Resize((size,size)))
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(transforms.Normalize(mean=[0.5],
                                                        std=[0.5]))
    if data_type == 'zhanglab':
        data_path = '/mnt/hdd/zhanglab-chest-xrays/chest_xray'
        train_data = ZhanglabWithLabel(data_path, data_type, transform = train_transform, size=int(size * (1/min_scale)))
    elif data_type == 'chexpert':
        data_path = '/home/jsato/pytorch-cutpaste/chexpert_dataset'
        train_data = chexpert_dataset(data_path, data_type, transform = train_transform, size=int(size * (1/min_scale)))

    
    dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True,
                            shuffle=True, num_workers=workers, #collate_fn=cut_paste_collate_fn,
                            pin_memory=True, generator=g,)

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(Path("logdirs") / model_name)

    # create Model:
    head_layers = [512]*head_layer+[128]
    num_classes = 2 if cutpate_type is not CutPaste3Way else 3
    model = ProjectionNet(pretrained=pretrained, head_layers=head_layers, num_classes=num_classes)
    model.to(device)

    if freeze_resnet > 0 and pretrained:
        model.freeze_resnet()

    loss_fn = torch.nn.CrossEntropyLoss()
    if optim_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learninig_rate, momentum=momentum,  weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, epochs)
        #scheduler = None
    elif optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learninig_rate, weight_decay=weight_decay)
        scheduler = None
    else:
        print(f"ERROR unkown optimizer: {optim_name}")

    step = 0
    num_batches = len(dataloader)
    def get_data_inf():
        while True:
            for out in dataloader:
                yield out
    dataloader_batch = iter(dataloader)
    # From paper: "Note that, unlike conventional definition for an epoch,
    #              we define 256 parameter update steps as one epoch.
    best_roc = 0
    for step in tqdm(range(epochs)):
        epoch = int(step / 1)
        #if epoch == freeze_resnet:
        #    model.unfreeze()
        for i in range(256):
        
            batch_embeds = []
            try:
                data,labels = next(dataloader_batch)
            except StopIteration:
                dataloader_batch = iter(dataloader)
                data,labels = next(dataloader_batch)
            xs = data.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            embeds, logits = model(xs)
            
    #         embeds = F.normalize(embeds, p=2, dim=1)
    #         embeds1, embeds2 = torch.split(embeds,x1.size(0),dim=0)
    #         ip = torch.matmul(embeds1, embeds2.T)
    #         ip = ip / temperature

    #         y = torch.arange(0,x1.size(0), device=device)
    #         loss = loss_fn(ip, torch.arange(0,x1.size(0), device=device))

            # calculate label
            y = labels
            #y = y.repeat_interleave(xs[0].size(0))
            loss = loss_fn(logits, y)

            # regulize weights:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step(epoch)
        writer.add_scalar('loss', loss.item(), step)
        
#         predicted = torch.argmax(ip,axis=0)
        predicted = torch.argmax(logits,axis=1)
#         print(logits)
#         print(predicted)
#         print(y)
        accuracy = torch.true_divide(torch.sum(predicted==y), predicted.size(0))
        print(accuracy)
        writer.add_scalar('acc', accuracy, step)
        if scheduler is not None:
            writer.add_scalar('lr', scheduler.get_last_lr()[0], step)
        
        # save embed for validation:
        #if test_epochs > 0 and epoch % test_epochs == 0:
        #    batch_embeds.append(embeds.cpu().detach())
        writer.add_scalar('epoch', epoch, step)

        # run tests
        if test_epochs > 0 and epoch % test_epochs == 0:
            # run auc calculation
            #TODO: create dataset only once.
            #TODO: train predictor here or in the model class itself. Should not be in the eval part
            #TODO: we might not want to use the training datat because of droupout etc. but it should give a indecation of the model performance???
            # batch_embeds = torch.cat(batch_embeds)
            # print(batch_embeds.shape)
            model.eval()
            roc_auc= eval_model(model_name, data_type, device=device,
                                save_plots=False,
                                size=size,
                                show_training_data=False,
                                model=model,
                                mode = 'valid')
                                #train_embed=batch_embeds)

            model.train()
            writer.add_scalar('eval_auc', roc_auc, step)
            print('score is ',roc_auc,'best score is ',best_roc)
            if roc_auc > best_roc:
                best_roc = roc_auc
                print('best score! save model.')
                torch.save(model.state_dict(), model_dir / f"{model_name}.tch")
    model.eval()
    model.load_state_dict(torch.load( model_dir / f"{model_name}.tch"))
    test_roc_auc = eval_model(model_name,data_type,device=device,
                            save_plots = False,
                            size = size,
                            show_training_data=False,
                            model = model,
                            mode = 'test')
    writer.add_scalar('final auc', test_roc_auc)
    print('test_roc_auc is ',test_roc_auc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
    parser.add_argument('--type', default="zhanglab",
                        help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')

    parser.add_argument('--epochs', default=256, type=int,
                        help='number of epochs to train the model , (default: 256)')
    
    parser.add_argument('--model_dir', default="models",
                        help='output folder of the models , (default: models)')
    
    parser.add_argument('--no-pretrained', dest='pretrained', default=True, action='store_false',
                        help='use pretrained values to initalize ResNet18 , (default: True)')
    
    parser.add_argument('--test_epochs', default=10, type=int,
                        help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')                  

    parser.add_argument('--freeze_resnet', default=20, type=int,
                        help='number of epochs to freeze resnet (default: 20)')
    
    parser.add_argument('--lr', default=0.03, type=float,
                        help='learning rate (default: 0.03)')

    parser.add_argument('--optim', default="sgd",
                        help='optimizing algorithm values:[sgd, adam] (dafault: "sgd")')

    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize (dafault: "64")')   

    parser.add_argument('--head_layer', default=1, type=int,
                    help='number of layers in the projection head (default: 1)')
    
    parser.add_argument('--variant', default="3way", choices=['normal', 'scar', '3way', 'union','anatmix'], help='cutpaste variant to use (dafault: "3way")')
    
    parser.add_argument('--cuda', default=0, type=str,
                    help='num of cuda to use')
    parser.add_argument('--seed', default=0, type=int,
                    help='set random_seed')
    
    parser.add_argument('--workers', default=8, type=int, help="number of workers to use for data loading (default:8)")


    args = parser.parse_args()
    print(args)
    all_types = ['zhanglab','chexpert']

    if args.type == "all":
        types = all_types
    else:
        types = args.type.split(",")

    os.environ["CUDA_VISIBLE_DEVICES"] =str(args.cuda)
    
    variant_map = {'normal':CutPasteNormal, 'scar':CutPasteScar, '3way':CutPaste3Way, 'union':CutPasteUnion,'anatmix': AnatMix}
    variant = variant_map[args.variant]
    
    device = "cuda" 
    
    # create modle dir
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    # save config.
    with open(Path(args.model_dir) / "run_config.txt", "w") as f:
        f.write(str(args))

    seed_everything(args.seed)

    for data_type in types:
        print(f"training {data_type}")
        run_training(data_type,
                     model_dir=Path(args.model_dir),
                     epochs=args.epochs,
                     pretrained=args.pretrained,
                     test_epochs=args.test_epochs,
                     freeze_resnet=args.freeze_resnet,
                     learninig_rate=args.lr,
                     optim_name=args.optim,
                     batch_size=args.batch_size,
                     head_layer=args.head_layer,
                     device=device,
                     cutpate_type=variant,
                     workers=args.workers,
                     seed = args.seed)