import torch
import torchvision
import os
import re
import numpy as np
import random as rand
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from datasets import bb_all, budle_bay
from architectures import resunet, models
from sys import float_info


EPS = float_info.epsilon


def generate_masks(bs, inds):
    masks = torch.zeros((bs, 1, 100, 100))
    for i in range(0, len(inds)):
        b_id, r, c = inds[i]
        masks[b_id, 0, r, c] = 1
    return masks


def inverse_transforms(batch_size, num_batches, theta):
    inv_theta = torch.zeros(num_batches, batch_size, 2, 3)
    for i in range(0, num_batches):
        for j in range(0, batch_size):
            temp_inv = theta[i, :, 0:2, 0:2]
            temp_inv = temp_inv.inverse()
            inv_theta[i, :, 0:2, 0:2] = temp_inv
    return inv_theta


def train_bb_reg():

    batch_size = 12
    num_epochs = 300

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print('building loaders...')
    train_set = budle_bay.BudleBay(root=r'E:\bb_data_100x100', split='train_reg', transform=transforms,
                                   download=True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size, shuffle=True, num_workers=1
    )

    test_set = budle_bay.BudleBay(root=r'E:\bb_data_100x100', split='test_reg', transform=transforms,
                                  download=True)

    test_loader = torch.utils.data.DataLoader(

        test_set,
        batch_size=1, shuffle=False, num_workers=1
    )

    print('building model... training...')
    #MODEL
    model = models.UNet(n_channels=3, n_classes=5)
    model.cuda()

    #OPTIM and LOSS
    opt = torch.optim.Adam(model.parameters(), lr=0.00001)
    # Loss choices:
    # when the target is one or more binary outputs, where a binary output is a proability
    # in range (0,1), you want BCE, or specifically BCE with logits
    # This in effect:
    # - applies a sigmoid non-linearity
    # - then computes BCE
    # However, our problem is not like this.
    # The target vector is a vector of probabilities that add up to 1
    # So in fact we want a softmax non-linearity
    # Then we want to apply regular BCE.
    # Add a small epsilon (something like 1e-6 or 1e-8) to the *target* probability vector
    # to prevent the BCE from resulting in NaN
    criterion = nn.BCELoss(reduction='none')
    #reduction = 'none'. apply mask after loss then take mean.
    torch.cuda.empty_cache()

    #NEED TO ADD TRANSFORMS TO MAKE NET SEE DIFFERENT EXAMPLES <---- THIS (DONE)
    n_b = len(train_loader)
    running_losses = []

    for epoch in range(0, num_epochs):

        theta = generate_transforms(batch_size, n_b)

        for i, (image, label) in enumerate(train_loader):

            x = image.cuda()
            y = label.cuda()

            if i < n_b - 1:
                grid = F.affine_grid(theta[i], image.size()).cuda()
                x_aug = F.grid_sample(x, grid).cuda()
                y = F.grid_sample(y, grid).cuda()
                y = y.float()
                outputs = model(x_aug)
            else:
                outputs = model(x)
                y = y.float()

            loss_mask = y >= 0
            outputs = F.softmax(outputs, dim=1)
            outputs = outputs + EPS
            y = y + EPS
            loss = criterion(outputs, y)
            loss = torch.mul(loss, loss_mask.float())
            loss = torch.sum(loss) / torch.sum(loss_mask)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if torch.isnan(loss) == 0:
                running_losses.append(loss.item())

        if epoch % 30 == 0:

            #TEST AND SAVE MODEL

            test_acc = []
            mIOUs = []
            for j, (t_img, t_lab) in enumerate(test_loader):
                t_x = t_img.cuda()
                t_y = t_lab.cuda()
                t_y = t_y.long()
                total = t_y.size(0) * 15 * 15
                t_outputs = model(t_x)
                _, preds = torch.max(t_outputs.data, 1)
                intersection = np.logical_and(preds.detach().cpu().numpy(), t_y.detach().cpu().numpy())
                union = np.logical_or(preds.detach().cpu().numpy(), t_y.detach().cpu().numpy())
                iou_score = np.sum(intersection) / np.sum(union)
                correct = (preds == t_y).sum().item()
                test_acc.append(correct/total)
                mIOUs.append(iou_score)
            avg_test_acc = sum(test_acc) / len(test_acc)
            avg_test_miou = sum(mIOUs) / len(mIOUs)

            print('Average loss over last 10 epochs: {}'.format(sum(running_losses) / len(running_losses)))
            print('Test accuracy for epoch {}: {}'.format(epoch + 1, avg_test_acc))
            print('Test mIOU for epoch {}: {}'.format(epoch + 1, avg_test_miou))

            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer': opt.state_dict(), 'losslogger': (sum(running_losses) / len(running_losses))}
            str_ep = str(epoch)
            out_file = 'bb_sup' + str_ep + '.pth'
            out_dir = r'D:\models\lr_00001_reg'
            out = os.path.join(out_dir, out_file)
            torch.save(state, out)
            running_losses = []


def inverse_transforms(batch_size, num_batches, theta):
    inv_theta = torch.zeros(num_batches, batch_size, 2, 3)
    for i in range(0, num_batches):
        for j in range(0, batch_size):
            temp_inv = theta[i, :, 0:2, 0:2]
            temp_inv = temp_inv.inverse()
            inv_theta[i, :, 0:2, 0:2] = temp_inv
    return inv_theta


def generate_transforms(batch_size, num_batches):
    theta = torch.zeros(num_batches, batch_size, 2, 3)
    for i in range(0, num_batches):
        for j in range(0, batch_size):
            r = rand.randrange(10, 20, 1)
            flip_horizontal = rand.randint(0, 100)
            flip_vertical = rand.randint(0, 100)
            hf = 1
            vf = 1
            if flip_horizontal < 20:
                hf = -1
            if flip_vertical < 20:
                vf = -1

            angle = np.pi / r

            theta[i, j, :, :2] = torch.tensor([[hf * np.cos(angle), -1.0 * np.sin(angle)],
                                            [np.sin(angle), vf * np.cos(angle)]])

    theta[:, :, :, 2] = 0

    return theta


def generate_masks(bs, inds):
    masks = torch.zeros((bs, 1, 200, 200))
    for i in range(0, len(inds)):
        b_id, r, c = inds[i]
        masks[b_id, 0 , r, c]


def split_image(img):

    split_images = []
    coords = []
    for x in range(0, 4501, 750):
        x_end = x + 1500
        for y in range(0, 4501, 750):
            y_end = y + 1500
            #print('x: {}'.format(x))
            #print('y: {}'.format(y))
            ##print('x_end: {}'.format(x_end))
            #print('y_end: {}'.format(y_end))
            split_images.append(img[:, :, x:x_end, y:y_end])
            coords.append((x, y))

    return split_images, coords


def test_bb():

    preds_dir = r'D:\models\tconv-cat-res-50-unet\SGD_RGB\preds'

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.224, 0.225])
    ])

    print('building loaders...')

    test_set = bb_all.BudleBay(root=r'D:\meta_shape\SPLIT_RGB_OP\tiled_mat', transform=transforms,
                               download=True)


    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1, shuffle=False, num_workers=1
    )

    #model = models.UNet(n_channels=5, n_classes=8)
    model = resunet.resnet50unet(num_classes=8, pretrained=True).cuda()
    #model = resunet.resnet101unet(num_classes=8, pretrained=True).cuda()

    print('building model... testing...')
    model.cuda()
    model_dir = r'D:\models\tconv-cat-res-50-unet\SGD_RGB\models\bb_sup279.pth'
    model.load_state_dict(torch.load(model_dir)['state_dict'])
    model.eval()

    for step, (img, img_path) in enumerate(test_loader):
        print('----------------------------------------------------------------')
        with torch.no_grad():
            split, l_coords = split_image(img)
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


def test_obj():

    me = torch.tensor([0.5776, 0.6278, 0.5831, 0.6128, 0.6283])
    sd = torch.tensor([0.0751, 0.0629, 0.0750, 0.0675, 0.0660])

    me_inet = torch.tensor([0.485, 0.485, 0.485])
    sd_inet = torch.tensor([[0.229, 0.224, 0.225]])

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.485, 0.485, 0.605228727147337,
                                          0.622354405561884],
                                         [0.229, 0.224, 0.225, 0.0676, 0.0663])
    ])

    print('building loaders...')
    test_set = budle_bay.BudleBay(root=r'F:\bb_data_256x256_resnet', split='test_ms', transform=transforms,
                                  download=True)

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1, shuffle=True, num_workers=1
    )

    print('building model... training...')
    #model = models.UNet(n_channels=5, n_classes=8).cuda()
    model = resunet.resnet50unet(num_classes=8, pretrained=True).cuda()
    #model = resunet.resnet101unet(num_classes=8, pretrained=True).cuda()


    root = r'D:\models\cat-res-50-unet\lr_1e-1_SGD\models'
    model_list = os.listdir(root)
    model_accs = []
    model_ious = []
    f = open('accs_1e-1_sgd_cat-res-50-unet.txt', 'w')
    mapping = {(0, 0, 0): 0,
               (255, 178, 102): 1,
               (102, 51, 0): 2,
               (128, 128, 128): 3,
               (0, 102, 0): 4,
               (0, 76, 153): 5,
               (204, 0, 0): 6,
               (102, 255, 102): 7,
               (25, 51, 0): 8}
    cmt = torch.zeros([8, 8], dtype=torch.uint8)

    for m in model_list:
        path = os.path.join(root, m)
        model.load_state_dict(torch.load(path)['state_dict'])
        model.eval()
        running_accs = []
        running_ious = []

        with torch.no_grad():
            for i, (image, label) in tqdm.tqdm((enumerate(test_loader))):
                x = image.cuda()
                y = label.cuda()  # add dim for transformaction

                """
                y = y + 1
                r_y = torch.zeros([1, y.shape[1], y.shape[2]], dtype=torch.float32)
                g_y = torch.zeros([1, y.shape[1], y.shape[2]], dtype=torch.float32)
                b_y = torch.zeros([1, y.shape[1], y.shape[2]], dtype=torch.float32)
                for k in mapping:
                    m = (y == mapping[k])
                    k = torch.tensor(list(k), dtype=torch.float32)
                    r_y[m] = k[0] / 255.
                    g_y[m] = k[1] / 255.
                    b_y[m] = k[2] / 255.

                rgb_y = torch.cat((r_y, g_y, b_y), dim=0)
                torchvision.utils.save_image(torch.unsqueeze(rgb_y, dim=0), 'kek.png', normalize=False)
                y = y - 1
                """

                m = y != -1

                """
                f_m = torch.zeros([1, y.shape[1], y.shape[2]], dtype=torch.float32)
                f_m[m] = 1.
                torchvision.utils.save_image(torch.unsqueeze(f_m, dim=0), 'kek2.png', normalize=False)
                """

                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)

                """
                predicted = predicted + 1
                r_p = torch.zeros([1, y.shape[1], y.shape[2]], dtype=torch.float32)
                g_p = torch.zeros([1, y.shape[1], y.shape[2]], dtype=torch.float32)
                b_p = torch.zeros([1, y.shape[1], y.shape[2]], dtype=torch.float32)

                for k in mapping:
                    m = (predicted == mapping[k])
                    k = torch.tensor(list(k), dtype=torch.float32)
                    r_p[m] = k[0] / 255.
                    g_p[m] = k[1] / 255.
                    b_p[m] = k[2] / 255.

                rgb_p = torch.cat((r_p, g_p, b_p), dim=0)
                torchvision.utils.save_image(torch.unsqueeze(rgb_p, dim=0), 'kek3.png', normalize=False)
                predicted = predicted - 1
                """

                c = (predicted == y)
                inter = (c & m).long().sum().detach().cpu().item()
                uni = (c | m).long().sum().detach().cpu().item()

                """
                f_c = torch.zeros([1, y.shape[1], y.shape[2]], dtype=torch.float32)
                f_c[c] = 1.
                torchvision.utils.save_image(torch.unsqueeze(f_c, dim=0), 'kek4.png', normalize=False)
                """

                running_accs.append(inter / m.sum().item())
                running_ious.append(inter / uni)
                """
                for t, p in zip(y.view(-1), predicted.view(-1)):
                    if t != -1:
                        cmt[t.long(), p.long()] += 1
                """


        print('===============================================================================')
        #print(cmt)
        avg_iou = sum(running_ious) / len(running_ious)
        avg_acc = sum(running_accs) / len(running_accs)
        model_ious.append(avg_iou)
        model_accs.append(avg_acc)
        print('iou for model {}: {}'.format(path, avg_iou))
        print('acc for model {}: {}'.format(path, avg_acc))
        f.write(str(avg_acc) + "\n")
        print('===============================================================================')

    ids = sorted(range(len(model_accs)), key=lambda k: model_accs[k])
    ids.reverse()

    for j in range(0, 5):
        print(model_list[ids[j]])

    f.close()


def train_bb():

    batch_size = 16
    num_epochs = 500

    weights_ms = torch.Tensor([0.282850822537822, 0.2537590766793680, 27.8166939543522,
                            404.374689165187, 7850.44655172414, 4627.29573170732,
                            93.6345109813276, 2.10803321919521]).cuda()

    weights_rgbs = torch.Tensor([0.811273407196349, 0.163799991223827, 17.3363930433927,
                                 207.868997758377, 4592.26140312771, 10635.1969315895,
                                 59.2314134673570, 1.71621986131774]).cuda()


    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.485, 0.485], [0.229, 0.224, 0.225])
    ])

    print('building loaders...')
    train_set = budle_bay.BudleBay(root=r'F:\bb_data_256x256_resnet_rgb', split='train_rgb', transform=transforms,
                                   download=True)

    test_set = budle_bay.BudleBay(root=r'F:\bb_data_256x256_resnet_rgb', split='test_rgb', transform=transforms,
                                  download=True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size, shuffle=True, num_workers=1
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1, shuffle=False, num_workers=1
    )

    print('building model...')

    #MODEL
    #model = models.UNet(n_channels=5, n_classes=8).cuda()
    #model = resunet.resnet101unet(num_classes=8, pretrained=True).cuda()

    model = resunet.resnet50unet(num_classes=8, pretrained=True).cuda()

    #OPTIM, LOSS and SCHEDULER
    opt = torch.optim.AdamW(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    #opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.99, weight_decay=0.01)

    sche = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, factor=0.5, patience=10, min_lr=1e-6, verbose=True)

    criterion = nn.CrossEntropyLoss(weights_rgbs, ignore_index=-1, reduction='mean')
    torch.cuda.empty_cache()

    val_losses_epoch = []
    for epoch in range(0, num_epochs):

        #theta = generate_transforms(batch_size, n_b)
        running_losses = []
        val_losses = []

        for i, (image, label) in tqdm.tqdm(enumerate(train_loader)):

            x = image.cuda()
            y = label.cuda() #add dim for transformaction

            """ 
            if i < n_b - 1:
                grid = F.affine_grid(theta[i], image.size()).cuda()
                x_aug = F.grid_sample(x, grid).cuda()
                y = F.grid_sample(y, grid).cuda()
                y = y.long()
                outputs = model(x_aug)
            else:
                outputs = model(x)
                y = label.cuda()
                y = y.long()
            """

            outputs = model(x)
            loss = criterion(outputs, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if torch.isnan(loss) == 0:
                running_losses.append(loss.item())

        # EPOCH TRAINING LOSS
        out_str = str(sum(running_losses) / len(running_losses))
        out_str = out_str + "\n"
        with open("train-loss_ADAM_tconv-cat-res-50-unet_256x256_rgb.txt", "a") as f:
            f.write(out_str)
        f.close()

        model.eval()

        with torch.no_grad():
            for i, (image, label) in tqdm.tqdm(enumerate(test_loader)):
                x = image.cuda()
                y = label.cuda()  # add dim for transformaction
                val_outputs = model(x)
                val_loss = criterion(val_outputs, y)
                if torch.isnan(val_loss) == 0:
                    val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
        sche.step(avg_val_loss)

        out_str = str(avg_val_loss)
        out_str = out_str + "\n"
        with open("val-loss_ADAM_tconv-cat-res-50-unet_256x256_rgb.txt", "a") as f:
            f.write(out_str)
        f.close()

        if (epoch > 0) and (avg_val_loss < min(val_losses_epoch)):
            print('---------------------------------------------------------------------')
            print('Validation loss at epoch {}: {}'.format(epoch, avg_val_loss))
            print('Previous best validation loss : {}'.format(min(val_losses_epoch)))

            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                     'optimizer': opt.state_dict(), 'val_loss': avg_val_loss, 'scheduler': sche}
            str_ep = str(epoch)
            out_file = 'bb_sup' + str_ep + '.pth'
            out_dir = r'D:\models\tconv-cat-res-50-unet\ADAMW_RGB\models'
            out = os.path.join(out_dir, out_file)
            torch.save(state, out)

        if epoch % 10 == 0:
            print('=====================================================================')
            print('EPOCH: {}'.format(epoch + 1))
            print('CURRENT TRAIN LOSS: {}'.format(sum(running_losses) / len(running_losses)))
            print('CURRENT VAL LOSS: {}'.format(avg_val_loss))
            print('CURRENT LEARNING RATE: {}'.format(opt.param_groups[0]['lr']))

        val_losses_epoch.append(avg_val_loss)
        model.train()


if __name__ == '__main__':
    test_bb()

