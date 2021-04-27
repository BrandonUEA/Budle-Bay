import torch
import torchvision
import os
import re
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from datasets import bb_all, budle_bay
from architectures import resunet
from sys import float_info


EPS = float_info.epsilon

def update_ema_variables(model, ema_model, alpha=0.99):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def test_bb():

    preds_dir = r'D:\models\tconv-cat-res-101-unet\SGD_RGB\preds'

    """
    transforms_ms1 = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    transforms_ms2 = torchvision.transforms.Compose([
        torchvision.transforms.Normalize([0.485, 0.456, 0.406, 0.5662, 0.5893, 0.4333, 0.4321, 0.5683, 0.4527,
                                          0.5312, 0.4734, 0.5128, 0.5256, 0.2853],
                                         [0.229, 0.224, 0.225, 0.0689, 0.0687, 0.0289, 0.0505, 0.0487, 0.0605,
                                          0.0710, 0.0577, 0.0723, 0.0556, 0.0406])
    ])
    """

    transforms_ms1 = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    transforms_ms2 = torchvision.transforms.Compose([
        torchvision.transforms.Normalize([0.485, 0.456, 0.406, 0.7295, 0.2830, 0.5038, 0.3637],
                                         [0.229, 0.224, 0.225, 0.0133, 0.0233, 0.0143, 0.0149])
    ])

    print('building loaders...')

    test_set = bb_all.BudleBay(root=r'D:\meta_shape\SPLIT_RGB_OP\tiled_mat', transform=transforms_ms1,
                               target_transform=transforms_ms2)

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1, shuffle=False, num_workers=1
    )

    model = resunet.resnet101unet(num_classes=7, pretrained=True).cuda()

    print('building model... testing...')
    model.cuda()
    model_dir = r'D:\models\tconv-cat-res-101-unet\SGD_RGB\saved_models\bb_sup_EPS_132_RESUNET101_TRAIN.pth'
    model.load_state_dict(torch.load(model_dir)['state_dict'])
    model.eval()

    for step, (img, img_path) in enumerate(test_loader):
        print('----------------------------------------------------------------')
        with torch.no_grad():

            [l_coords, split] = split_image(img)
            coord_idx = 0
            count = 0

            for b_img in split:
                b_img = b_img.cuda()
                out = model(b_img)
                _, predicted = torch.max(out.data, 1)
                np_predicted = predicted.cpu().detach().numpy()
                coords = l_coords[coord_idx]

                s = img_path[0].split("\\")
                f = s[-1].replace('.mat', '')

                im_coords = list(map(int, re.findall(r'\d+', f)))

                out_dir = str(im_coords[0]) + '_' + str(im_coords[1]) + '_' + str(im_coords[2]) + '_' + str(im_coords[3])
                out_dir = os.path.join(preds_dir, out_dir)

                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                out_file = str(coords[0]) + '_' + str(coords[1]) + '.mat'

                count += 1
                coord_idx += 1

                out_file = os.path.join(out_dir, out_file)
                print(out_file)

                sio.savemat(out_file, {'preds': np_predicted})


def test_fold(test_loader, model, cmt_root):

    model.eval()

    with torch.no_grad():
        for i, (image, label) in tqdm.tqdm((enumerate(test_loader))):
            x = image.cuda()
            y = label.cuda()
            m = y != -1

            out = model(x)
            _, p = torch.max(out.data, 1)

            p = p.view(-1)
            y = y.view(-1)
            m = m.view(-1)
            p = p[m].detach().cpu().numpy()
            y = y[m].detach().cpu().numpy()

            p_f = 'p_' + str(i) + '.mat'
            y_f = 'y_' + str(i) + '.mat'

            p_out = os.path.join(cmt_root, p_f)
            y_out = os.path.join(cmt_root, y_f)

            sio.savemat(p_out, {'p': p})
            sio.savemat(y_out, {'y': y})

    model.train()

    return None


def train_fold():

    root = r'D:\models\tconv-cat-res-101-unet'

    batch_size = 12
    num_epochs = 150

    weights_ms = torch.Tensor([0.780817335389991,
                            0.190213823607973,
                            24.7876106194690,
                            371.814159292035,
                            217.694300518135,
                            9.49062570589564,
                            3.23565652676165]).cuda()

    weights_rgb = torch.Tensor([0.409123210327029,
                                0.310603589789510,
                                36.2574635856786,
                                250.029909926428,
                                342.363011549084,
                                21.2530964313720,
                                0.797030042416119]).cuda()

    transforms_ms1 = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    transforms_ms2 = torchvision.transforms.Compose([
        torchvision.transforms.Normalize([0.485, 0.456, 0.406, 0.7295, 0.2830, 0.5038, 0.3637],
                                         [0.229, 0.224, 0.225, 0.0133, 0.0233, 0.0143, 0.0149])
    ])

    print('building loaders...')

    val_set = budle_bay.BudleBay(root=r'F:\bb_data_256x256_resnet_rgb', split='val_rgb', transform=transforms_ms1,
                                 target_transform=transforms_ms2)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1, shuffle=False, num_workers=1
    )

    test_set = budle_bay.BudleBay(root=r'F:\bb_data_256x256_resnet_rgb', split='test_rgb', transform=transforms_ms1,
                                 target_transform=transforms_ms2)

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1, shuffle=False, num_workers=1
    )

    train_set = budle_bay.BudleBay(root=r'F:\bb_data_256x256_resnet_rgb', split='train_rgb', transform=transforms_ms1,
                                  target_transform=transforms_ms2)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size, shuffle=True, num_workers=1
    )


    # MODEL
    model = resunet.resnet101unet(num_classes=7, pretrained=True).cuda()
    model_T = resunet.resnet101unet(num_classes=7, pretrained=True).cuda()
    model_T.load_state_dict(model.state_dict())
    for param in model_T.parameters():
        param.requires_grad = False
        param.detach_()


    # OPTIM, LOSS and SCHEDULER
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    sche = torch.optim.lr_scheduler.StepLR(opt, step_size=70, gamma=0.1)
    criterion = nn.CrossEntropyLoss(weights_rgb, ignore_index=-1, reduction='mean')
    torch.cuda.empty_cache()

    val_losses_epoch = []
    train_losses_epoch = []

    if os.path.exists(r'losses/train-loss_resnet101-unet.txt') and os.path.exists(
            r'losses/val-loss_resnet101-unet.txt'):
        os.remove(r'losses/train-loss_resnet101-unet.txt')
        os.remove(r'losses/val-loss_resnet101-unet.txt')

    for epoch in range(0, num_epochs):

        train_losses = []
        val_losses = []
        sup_losses = []
        unsup_losses = []
        str_ep = str(epoch)

        for i, (image, label) in tqdm.tqdm(enumerate(train_loader)):

            x = image.cuda()
            y = label.cuda()

            # sup loss
            pred_logits_S = model(x)
            loss1 = criterion(pred_logits_S, y)

            # unsup loss
            with torch.no_grad():
                pred_logits_T = model_T(x).detach()

            # Logits to probs and consistency
            pred_prob_T = F.softmax(pred_logits_T, dim=1)

            # Conf threshold and hard labels
            pred_conf_t, pred_hard_T = torch.max(pred_prob_T, dim=1)

            # Conf threshold mask
            pred_conf_mask = (pred_conf_t >= 0.97).float().mean()

            loss2 = F.cross_entropy(pred_logits_S, pred_hard_T, reduction='none')
            loss2 = (loss2 * (y == -1).float()) * pred_conf_mask * 0.10

            loss = loss1 + loss2.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema_variables(model, model_T)

            if torch.isnan(loss) == 0:
                train_losses.append(loss.item())
                sup_losses.append(loss1.item())
                unsup_losses.append(loss2.mean().item())

        # EPOCH TRAINING LOSS
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_sup_loss = sum(sup_losses) / len(sup_losses)
        avg_unsup_loss = sum(unsup_losses) / len(unsup_losses)
        out_str = str(avg_train_loss)
        out_str = out_str + "\n"
        with open("losses/train-loss_SGD_resnet101-unet.txt", "a") as f:
            f.write(out_str)
        f.close()

        model.eval()

        with torch.no_grad():
            for i, (image, label) in tqdm.tqdm(enumerate(val_loader)):
                x = image.cuda()
                y = label.cuda()  # add dim for transformaction
                val_outputs = model(x)
                val_loss = criterion(val_outputs, y)
                if torch.isnan(val_loss) == 0:
                    val_losses.append(val_loss.item())
            # EPOCH VALIDATION LOSS
            avg_val_loss = sum(val_losses) / len(val_losses)
        sche.step()

        out_str = str(avg_val_loss)
        out_str = out_str + "\n"
        with open("losses/val-loss_SGD_resnet101-unet.txt", "a") as f:
            f.write(out_str)
        f.close()

        if (epoch > 0) and (avg_val_loss < min(val_losses_epoch)):
            print('---------------------------------------------------------------------')
            print('Validation loss at epoch {}: {}'.format(epoch, avg_val_loss))
            print('Previous best validation loss : {}'.format(min(val_losses_epoch)))
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer': opt.state_dict(), 'val_loss': avg_val_loss}#, 'scheduler': sche}
            out_file = 'bb_sup_EPS_' + str_ep + '_RESUNET101_VAL.pth'
            out_dir = os.path.join(root, 'models')
            for f in os.listdir(out_dir):
                if f.endswith("_VAL.pth"):
                    os.remove(os.path.join(out_dir, f))
            out = os.path.join(out_dir, out_file)
            torch.save(state, out)

        model_check = os.path.join(root, 'models', 'bb_sup_EPS_' + str_ep + '_RESUNET101_VAL.pth')

        if (epoch > 0) and (avg_train_loss < min(train_losses_epoch)) and (not (os.path.exists(model_check))):
            print('---------------------------------------------------------------------')
            print('Train loss at epoch {}: {}'.format(epoch, avg_train_loss))
            print('Previous best train loss : {}'.format(min(train_losses_epoch)))

            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer': opt.state_dict(), 'val_loss': avg_val_loss}#, 'scheduler': sche}
            out_file = 'bb_sup_EPS_' + str_ep + '_RESUNET101_TRAIN.pth'
            out_dir = os.path.join(root, 'models')
            for f in os.listdir(out_dir):
                if f.endswith("_TRAIN.pth"):
                    os.remove(os.path.join(out_dir, f))
            out = os.path.join(out_dir, out_file)
            torch.save(state, out)

        if epoch % 10 == 0:
            print('=====================================================================')
            print('EPOCH: {}'.format(epoch + 1))
            print('CURRENT TRAIN LOSS: {}'.format(avg_train_loss))
            print('CURRENT SUP LOSS: {}'.format(avg_sup_loss))
            print('CURRENT UNSUP LOSS: {}'.format(avg_unsup_loss))
            print('CURRENT VAL LOSS: {}'.format(avg_val_loss))
            print('CURRENT LEARNING RATE: {}'.format(opt.param_groups[0]['lr']))

        val_losses_epoch.append(avg_val_loss)
        train_losses_epoch.append(avg_train_loss)
        model.train()

    print('------------    TEST FOLD    ----------------')

    val_check = '_RESUNET101_VAL.pth'
    train_check = '_RESUNET101_TRAIN.pth'
    model_dir = os.path.join(root, 'models')
    print('student preds...')
    for file in os.listdir(model_dir):
        if file.endswith(val_check):
            cmt_root = os.path.join(root, 'cmts-single', 'val')
            model_dir = os.path.join(root, 'models', file)
            model.load_state_dict(torch.load(model_dir)['state_dict'])
            test_fold(test_loader, model, cmt_root)
        if file.endswith(train_check):
            cmt_root = os.path.join(root, 'cmts-single', 'train')
            model_dir = os.path.join(root, 'models', file)
            model.load_state_dict(torch.load(model_dir)['state_dict'])
            test_fold(test_loader, model, cmt_root)


if __name__ == '__main__':
    train_fold()
    #test_bb()

