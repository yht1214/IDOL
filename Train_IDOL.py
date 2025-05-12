import math
import os
import torch.optim
from torch import nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import Set_seed
from Dataset import *
import Config as config
# from IDPil_CNF import *
# from IDPil_closs import *
from IDPil_dkgi import *
import ssl
import smtplib
from email.header import Header
from email.utils import formataddr
from email.mime.text import MIMEText

def save_model(model, model_output_dir, epoch):
    save_model_file = os.path.join(model_output_dir, "epoch_{}.pth".format(epoch))
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    torch.save(model, save_model_file)

def train_model(model, loss_func, dataset, optimizer, w_params, epoch):

    model.train()

    batch_msw_loss = 0
    batch_rmw_loss = 0
    batch_r34_loss = 0
    batch_mslp_loss = 0

    batch_MIsh_f_loss = 0
    batch_MIsh_y_loss = 0
    batch_MIsp_y_loss = 0

    batch_loss = 0
    item = 0
    batch_num = 0

    msw_error = 0
    rmw_error = 0
    r34_error = 0
    mslp_error = 0

    for batch, data in enumerate(tqdm(dataset)):
        '''5, 156, 156'''
        k8_btemp = data["k8_btemp"]
        k8_btemp = k8_btemp.to(config.device)

        pxh_k8_btemp = data["pxh_k8_btemp"]
        pxh_k8_btemp = pxh_k8_btemp.to(config.device)
        '''cat pxh and now'''
        btemp = torch.cat([pxh_k8_btemp, k8_btemp], dim=1)

        pre_level = data["pre_level"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
        norm_t = data['norm_t'].to(torch.float32).to(config.device).unsqueeze(dim=-1)
        plt = torch.cat([pre_level, norm_t], dim=-1)

        pre_tcf = data["pre_tcf"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
        pre_isr = data["pre_isr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
        pre_pwr = data["pre_pwr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
        pre_prr = data["pre_prr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
        pre_pr = torch.cat([pre_tcf, pre_isr, pre_pwr, pre_prr], dim=-1)

        msw_label = data["msw"]
        msw_label = msw_label.to(config.device)
        rmw_label = data["rmw"]
        rmw_label = rmw_label.to(config.device)
        mslp_label = data["mslp"]
        mslp_label = mslp_label.to(config.device)
        r34_label = data["r34"]
        r34_label = r34_label.to(config.device)
        TCA_Lable = [i.unsqueeze(1) for i in [msw_label, mslp_label, rmw_label, r34_label]]
        TCA_Lable = torch.cat(TCA_Lable, dim=1)

        optimizer.zero_grad()

        # msw, mslp, rmw, r34 = model(btemp, plt, TCA_Lable)
        # msw, mslp, rmw, r34, loss_sp = model(btemp, plt, TCA_Lable)
        # msw, mslp, rmw, r34 = model(btemp, plt, pre_pr, TCA_Lable)
        msw, mslp, rmw, r34, loss_sp = model(btemp, plt, pre_pr, TCA_Lable)

        print("msw_label={}".format(msw_label))
        print("msw={}".format(msw))
        print("mslp_label={}".format(mslp_label))
        print("mslp={}".format(mslp))
        print("rmw_label={}".format(rmw_label))
        print("rmw={}".format(rmw))
        print("r34_label={}".format(r34_label))
        print("r34={}".format(r34))

        msw_loss = loss_func(msw.float(), msw_label.float())
        mslp_loss = loss_func(mslp.float(), mslp_label.float())
        rmw_loss = loss_func(rmw.float(), rmw_label.float())
        r34_loss = loss_func(r34.float(), r34_label.float())

        # cnf_loss = (cnf_shloss + cnf_sploss) * 0.1
        total_loss = w_params[0] * msw_loss + w_params[1] * mslp_loss + w_params[2] * rmw_loss + w_params[3] * r34_loss\
                     + 0.5 * loss_sp

        total_loss.backward()
        optimizer.step()

        # 监控梯度
        if batch % 5 == 0:
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = round(param.grad.data.norm(2).item(), 5)
                    print(
                        f'Epoch [{epoch}], Step [{batch + 1}/{len(dataset)}], Gradient norm: {grad_norm}')
        # print("Train Epoch = {} loss_sh = {}".format(epoch, loss_sh.data.item()))
        print("Train Epoch = {} loss_sp = {}".format(epoch, loss_sp.data.item()))
        print("Train Epoch = {} msw Loss = {}".format(epoch, msw_loss.data.item()))
        print("Train Epoch = {} mslp Loss = {}".format(epoch, mslp_loss.data.item()))
        print("Train Epoch = {} rmw Loss = {}".format(epoch, rmw_loss.data.item()))
        print("Train Epoch = {} r34 Loss = {}".format(epoch, r34_loss.data.item()))
        print("Train Epoch = {} Loss = {}".format(epoch, total_loss.data.item()))
        batch_loss += total_loss.data.item()

        # batch_MIsh_f_loss += MIsh_f.item()
        # batch_MIsh_y_loss += loss_sh.item()
        batch_MIsp_y_loss += loss_sp.item()

        batch_msw_loss += msw_loss.data.item()
        batch_mslp_loss += mslp_loss.data.item()
        batch_rmw_loss += rmw_loss.data.item()
        batch_r34_loss += r34_loss.data.item()

        # 反归一化看train error
        msw_label_re = msw_label.cpu().detach().numpy() * (170 - 35) + 35
        msw_re = msw.cpu().detach().numpy() * (170 - 35) + 35
        msw_error = msw_error + np.sum(np.abs(msw_re - msw_label_re))

        mslp_label_re = mslp_label.cpu().detach().numpy() * (1010 - 882) + 882
        mslp_re = mslp.cpu().detach().numpy() * (1010 - 882) + 882
        mslp_error = mslp_error + np.sum(np.abs(mslp_re - mslp_label_re))

        rmw_label_re = rmw_label.cpu().detach().numpy() * (130 - 5) + 5
        rmw_re = rmw.cpu().detach().numpy() * (130 - 5) + 5
        rmw_error = rmw_error + np.sum(np.abs(rmw_re - rmw_label_re))

        r34_label_re = r34_label.cpu().detach().numpy() * (406.25 - 10) + 10
        r34_re = r34.cpu().detach().numpy() * (406.25 - 10) + 10
        r34_error = r34_error + np.sum(np.abs(r34_re - r34_label_re))

        batch_num += 1
        item += len(msw_re)

    print("Train Epoch = {} mean loss = {} ".format(epoch, batch_loss / batch_num))
    print("Train Epoch = {} mean loss_sh = {} ".format(epoch, batch_MIsp_y_loss / batch_num))
    print("Train Epoch = {} mean msw error = {} ".format(epoch, msw_error / item))
    print("Train Epoch = {} mean mslp error = {} ".format(epoch, mslp_error / item))
    print("Train Epoch = {} mean rmw error = {} ".format(epoch, rmw_error / item))
    print("Train Epoch = {} mean r34 error = {} ".format(epoch, r34_error / item))

    return batch_loss / batch_num, msw_error / item, mslp_error / item, rmw_error / item, r34_error / item, \
           batch_MIsh_f_loss / batch_num, batch_MIsh_y_loss / batch_num, batch_MIsp_y_loss / batch_num

def valid_model(model, loss_func, dataset, epoch):

    model.eval()

    batch_loss = 0
    batch_cnf_loss = 0

    batch_num = 0
    item = 0

    msw_error = 0
    mslp_error = 0
    rmw_error = 0
    r34_error = 0

    batch_MIsh_f_loss = 0
    batch_MIsh_y_loss = 0
    batch_MIsp_y_loss = 0

    with torch.no_grad():
        for batch, data in enumerate(tqdm(dataset)):
            '''5, 156, 156'''
            k8_btemp = data["k8_btemp"]
            k8_btemp = k8_btemp.to(config.device)

            pxh_k8_btemp = data["pxh_k8_btemp"]
            pxh_k8_btemp = pxh_k8_btemp.to(config.device)
            '''cat pxh and now'''
            btemp = torch.cat([pxh_k8_btemp, k8_btemp], dim=1)

            pre_level = data["pre_level"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            norm_t = data['norm_t'].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            plt = torch.cat([pre_level, norm_t], dim=-1)

            pre_tcf = data["pre_tcf"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            pre_isr = data["pre_isr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            pre_pwr = data["pre_pwr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            pre_prr = data["pre_prr"].to(torch.float32).to(config.device).unsqueeze(dim=-1)
            pre_pr = torch.cat([pre_tcf, pre_isr, pre_pwr, pre_prr], dim=-1)

            msw_label = data["msw"]
            msw_label = msw_label.to(config.device)
            rmw_label = data["rmw"]
            rmw_label = rmw_label.to(config.device)
            mslp_label = data["mslp"]
            mslp_label = mslp_label.to(config.device)
            r34_label = data["r34"]
            r34_label = r34_label.to(config.device)
            TCA_Lable = [i.unsqueeze(1) for i in [msw_label, mslp_label, rmw_label, r34_label]]
            TCA_Lable = torch.cat(TCA_Lable, dim=1)

            # msw, mslp, rmw, r34 = model(btemp, plt, TCA_Lable)
            # msw, mslp, rmw, r34, loss_sp = model(btemp, plt, TCA_Lable)
            # msw, mslp, rmw, r34 = model(btemp, plt, pre_pr, TCA_Lable)
            msw, mslp, rmw, r34, loss_sp = model(btemp, plt, pre_pr, TCA_Lable)

            print("msw_label={}".format(msw_label))
            print("msw={}".format(msw))
            print("mslp_label={}".format(mslp_label))
            print("mslp={}".format(mslp))
            print("rmw_label={}".format(rmw_label))
            print("rmw={}".format(rmw))
            print("r34_label={}".format(r34_label))
            print("r34={}".format(r34))

            msw_loss = loss_func(msw.float(), msw_label.float())
            mslp_loss = loss_func(mslp.float(), mslp_label.float())
            rmw_loss = loss_func(rmw.float(), rmw_label.float())
            r34_loss = loss_func(r34.float(), r34_label.float())

            # cnf_loss = (cnf_shloss + cnf_sploss) * 0.1
            total_loss = w_params[0] * msw_loss + w_params[1] * mslp_loss + \
                         w_params[2] * rmw_loss + w_params[3] * r34_loss\
                         + 0.5 * loss_sp
            # print("Valid Epoch = {} loss_sh = {}".format(epoch, loss_sh.data.item()))
            print("Valid Epoch = {} loss_sp = {}".format(epoch, loss_sp.data.item()))
            print("Valid Epoch = {} msw Loss = {}".format(epoch, msw_loss.data.item()))
            print("Valid Epoch = {} mslp Loss = {}".format(epoch, mslp_loss.data.item()))
            print("Valid Epoch = {} rmw Loss = {}".format(epoch, rmw_loss.data.item()))
            print("Valid Epoch = {} r34 Loss = {}".format(epoch, r34_loss.data.item()))

            batch_loss += total_loss.data.item()
            # batch_MIsh_f_loss += MIsh_f.item()
            # batch_MIsh_y_loss += loss_sh.item()
            batch_MIsp_y_loss += loss_sp.item()

            msw_label_re = msw_label.cpu().detach().numpy() * (170 - 35) + 35
            msw_re = msw.cpu().detach().numpy() * (170 - 35) + 35
            msw_error = msw_error + np.sum(np.abs(msw_re - msw_label_re))

            mslp_label_re = mslp_label.cpu().detach().numpy() * (1010 - 882) + 882
            mslp_re = mslp.cpu().detach().numpy() * (1010 - 882) + 882
            mslp_error = mslp_error + np.sum(np.abs(mslp_re - mslp_label_re))

            rmw_label_re = rmw_label.cpu().detach().numpy() * (130 - 5) + 5
            rmw_re = rmw.cpu().detach().numpy() * (130 - 5) + 5
            rmw_error = rmw_error + np.sum(np.abs(rmw_re - rmw_label_re))

            r34_label_re = r34_label.cpu().detach().numpy() * (406.25 - 10) + 10
            r34_re = r34.cpu().detach().numpy() * (406.25 - 10) + 10
            r34_error = r34_error + np.sum(np.abs(r34_re - r34_label_re))

            batch_num += 1
            item += len(msw_re)

    print("Valid Epoch = {} mean loss = {} ".format(epoch, batch_loss / batch_num))
    print("Valid Epoch = {} mean loss_sh = {} ".format(epoch, batch_MIsp_y_loss / batch_num))
    print("Valid Epoch = {} mean msw error = {} ".format(epoch, msw_error / item))
    print("Valid Epoch = {} mean mslp error = {} ".format(epoch, mslp_error / item))
    print("Valid Epoch = {} mean rmw error = {} ".format(epoch, rmw_error / item))
    print("Valid Epoch = {} mean r34 error = {} ".format(epoch, r34_error / item))

    return batch_loss / batch_num, msw_error / item, mslp_error / item, rmw_error / item, r34_error / item, \
           batch_MIsh_f_loss / batch_num, batch_MIsh_y_loss / batch_num, batch_MIsp_y_loss / batch_num

def train(model, Estim_Loss, optimizer, w_params, epochs):
    train_transform = None
    valid_transform = None
    '''k_era_var'''
    # train_dataset = Findpxh_Dataset_kv(config.train_k8_path, config.train_zqtuv_path, 3, None, config.data_format)
    # valid_dataset = Findpxh_Dataset_kv(config.valid_k8_path, config.valid_zqtuv_path, 3, None, config.data_format)
    '''k'''
    train_dataset = Findpxh_Dataset_k_pr(config.train_k8_path, 3, None, config.data_format)
    valid_dataset = Findpxh_Dataset_k_pr(config.valid_k8_path, 3, None, config.data_format)
    '''k_era_var_aux'''
    # train_dataset = Findpxh_Dataset_kva(config.train_k8_path, config.train_zqtuv_path,
    #                                     config.train_aux_path, 3, None, config.data_format)
    # valid_dataset = Findpxh_Dataset_kva(config.valid_k8_path, config.valid_zqtuv_path,
    #                                     config.valid_aux_path, 3, None, config.data_format)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=True
    )

    '''
    训练模型
    '''
    start_epoch = 0
    train_loss_array = np.zeros(epochs + 1)
    valid_loss_array = np.zeros(epochs + 1)

    train_mishf_array = np.zeros(epochs + 1)
    valid_mishf_array = np.zeros(epochs + 1)
    train_mishy_array = np.zeros(epochs + 1)
    valid_mishy_array = np.zeros(epochs + 1)
    train_mispy_array = np.zeros(epochs + 1)
    valid_mispy_array = np.zeros(epochs + 1)

    train_msw_error_array = np.zeros(epochs + 1)
    valid_msw_error_array = np.zeros(epochs + 1)
    train_mslp_error_array = np.zeros(epochs + 1)
    valid_mslp_error_array = np.zeros(epochs + 1)
    train_rmw_error_array = np.zeros(epochs + 1)
    valid_rmw_error_array = np.zeros(epochs + 1)
    train_r34_error_array = np.zeros(epochs + 1)
    valid_r34_error_array = np.zeros(epochs + 1)

    for epoch in range(start_epoch + 1, epochs + 1):
        train_epoch_loss, train_msw_error, train_mslp_error, train_rmw_error, train_r34_error, \
        train_MIsh_f_loss, train_MIsh_y_loss, train_MIsp_y_loss = \
            train_model(model, Estim_Loss, train_dataloader, optimizer, w_params, epoch)
        valid_epoch_loss, valid_msw_error, valid_mslp_error, valid_rmw_error, valid_r34_error, \
        valid_MIsh_f_loss, valid_MIsh_y_loss, valid_MIsp_y_loss = \
            valid_model(model, Estim_Loss, valid_dataloader, epoch)

        train_loss_array[epoch] = train_epoch_loss
        valid_loss_array[epoch] = valid_epoch_loss
        train_mishf_array[epoch] = train_MIsh_f_loss
        valid_mishf_array[epoch] = valid_MIsh_f_loss
        train_mishy_array[epoch] = train_MIsh_y_loss
        valid_mishy_array[epoch] = valid_MIsh_y_loss
        train_mispy_array[epoch] = train_MIsp_y_loss
        valid_mispy_array[epoch] = valid_MIsp_y_loss

        train_msw_error_array[epoch] = train_msw_error
        valid_msw_error_array[epoch] = valid_msw_error
        train_mslp_error_array[epoch] = train_mslp_error
        valid_mslp_error_array[epoch] = valid_mslp_error
        train_rmw_error_array[epoch] = train_rmw_error
        valid_rmw_error_array[epoch] = valid_rmw_error
        train_r34_error_array[epoch] = train_r34_error
        valid_r34_error_array[epoch] = valid_r34_error

        # 模型保存
        if epoch % config.save_model_iter == 0:
            save_model(model, config.model_output_dir_dkgi, epoch)

    # 绘制loss
    fig_loss, ax_loss = plt.subplots(figsize=(12, 8))
    train_loss, = plt.plot(np.arange(1, epochs + 1), train_loss_array[1:], 'r')
    val_loss, = plt.plot(np.arange(1, epochs + 1), valid_loss_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('Estimation loss')
    plt.title("train/valid loss vs epoch")
    # 添加图例
    ax_loss.legend(handles=[train_loss, val_loss], labels=['train_epoch_loss', 'val_epoch_loss'],
                         loc='best')
    fig_loss.savefig(config.save_fig_dir_dkgi + 'loss.png')
    plt.close(fig_loss)

    fig_mishf, ax_mishf = plt.subplots(figsize=(12, 8))
    train_mishf, = plt.plot(np.arange(1, epochs + 1), train_mishf_array[1:], 'r')
    val_mishf, = plt.plot(np.arange(1, epochs + 1), valid_mishf_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('mishf loss')
    plt.title("train/valid mishf loss vs epoch")
    # 添加图例
    ax_mishf.legend(handles=[train_mishf, val_mishf], labels=['train_epoch_loss', 'val_epoch_loss'],
                   loc='best')
    fig_mishf.savefig(config.save_fig_dir_dkgi + 'mishf_loss.png')
    plt.close(fig_mishf)

    fig_mishy, ax_mishy = plt.subplots(figsize=(12, 8))
    train_mishy, = plt.plot(np.arange(1, epochs + 1), train_mishy_array[1:], 'r')
    val_mishy, = plt.plot(np.arange(1, epochs + 1), valid_mishy_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('mishy loss')
    plt.title("train/valid mishy loss vs epoch")
    # 添加图例
    ax_mishy.legend(handles=[train_mishy, val_mishy], labels=['train_epoch_loss', 'val_epoch_loss'],
                    loc='best')
    fig_mishy.savefig(config.save_fig_dir_dkgi + 'mishy_loss.png')
    plt.close(fig_mishy)

    fig_mispy, ax_mispy = plt.subplots(figsize=(12, 8))
    train_mispy, = plt.plot(np.arange(1, epochs + 1), train_mispy_array[1:], 'r')
    val_mispy, = plt.plot(np.arange(1, epochs + 1), valid_mispy_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('mispy loss')
    plt.title("train/valid mispy loss vs epoch")
    # 添加图例
    ax_mispy.legend(handles=[train_mispy, val_mispy], labels=['train_epoch_loss', 'val_epoch_loss'],
                    loc='best')
    fig_mispy.savefig(config.save_fig_dir_dkgi + 'mispy_loss.png')
    plt.close(fig_mispy)

    fig_msw_error, ax_msw_error = plt.subplots(figsize=(12, 8))
    train_msw_error, = plt.plot(np.arange(1, epochs + 1),
                                                      train_msw_error_array[1:], 'r')
    valid_msw_error, = plt.plot(np.arange(1, epochs + 1),
                                       valid_msw_error_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('msw error')
    plt.title("train/valid error vs epoch")
    # 添加图例
    ax_msw_error.legend(handles=[train_msw_error,
                              valid_msw_error],
                     labels=['train_error', 'valid_error'],
                     loc='best')
    fig_msw_error.savefig(config.save_fig_dir_dkgi + 'msw_error.png')
    plt.close(fig_msw_error)

    fig_mslp_error, ax_mslp_error = plt.subplots(figsize=(12, 8))
    train_mslp_error, = plt.plot(np.arange(1, epochs + 1),
                                 train_mslp_error_array[1:], 'r')
    valid_mslp_error, = plt.plot(np.arange(1, epochs + 1),
                                 valid_mslp_error_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('mslp error')
    plt.title("train/valid error vs epoch")
    # 添加图例
    ax_mslp_error.legend(handles=[train_mslp_error,
                                 valid_mslp_error],
                        labels=['train_error', 'valid_error'],
                        loc='best')
    fig_mslp_error.savefig(config.save_fig_dir_dkgi + 'mslp_error.png')
    plt.close(fig_mslp_error)

    fig_rmw_error, ax_rmw_error = plt.subplots(figsize=(12, 8))
    train_rmw_error, = plt.plot(np.arange(1, epochs + 1),
                                 train_rmw_error_array[1:], 'r')
    valid_rmw_error, = plt.plot(np.arange(1, epochs + 1),
                                 valid_rmw_error_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('RMW Error')
    plt.title("train/valid error vs epoch")
    # 添加图例
    ax_rmw_error.legend(handles=[train_rmw_error,
                                 valid_rmw_error],
                        labels=['train_error', 'valid_error'],
                        loc='best')
    fig_rmw_error.savefig(config.save_fig_dir_dkgi + 'rmw_error.png')
    plt.close(fig_rmw_error)

    fig_r34_error, ax_r34_error = plt.subplots(figsize=(12, 8))
    train_r34_error, = plt.plot(np.arange(1, epochs + 1),
                                 train_r34_error_array[1:], 'r')
    valid_r34_error, = plt.plot(np.arange(1, epochs + 1),
                                 valid_r34_error_array[1:], 'g')
    plt.xlabel('epochs')
    plt.ylabel('R34 Error')
    plt.title("train/valid error vs epoch")
    # 添加图例
    ax_r34_error.legend(handles=[train_r34_error,
                                 valid_r34_error],
                        labels=['train_error', 'valid_error'],
                        loc='best')
    fig_r34_error.savefig(config.save_fig_dir_dkgi + 'r34_error.png')
    plt.close(fig_r34_error)

def send_email(content):
    # qq email code :sirayzxaphldcaaj
    # QQ邮件服务器的安全端口465，注意是数字格式
    smtp_port = 465
    # QQ邮件服务器的地址
    smtp_host = 'smtp.qq.com'
    # 登录QQ邮件服务器的用户名
    user_name = '329769800@qq.com'
    # 此处填写自己申请到的登录QQ邮件服务器的授权码
    user_pass = 'sirayzxaphldcaaj'
    # 发件人邮箱地址
    sender = '329769800@qq.com'
    # 收件人邮箱地址，列表中可以包含多个收件人地址
    receivers = ['329769800@qq.com']
    # 构造邮件，邮件体内容是一段文本，采用'utf-8'
    email_msg = MIMEText('your project:' + content +' already over', 'plain', 'utf-8')
    # 构造邮件头中主题，突出邮件内容重点
    email_msg['Subject'] = Header('程序运行通知', 'utf-8').encode()
    # 构造邮件头中的发件人，包括昵称和邮箱账号
    email_msg['From'] = formataddr((Header('project notification', 'utf-8').encode(),
                                    '329769800@qq.com'))
    # 构造邮件头中的收件人，包括昵称和邮箱账号
    email_msg['To'] = formataddr((Header('project notification', 'utf-8').encode(),
                                  '329769800@qq.com'))
    context = ssl.create_default_context()
    try:
        # 采用with结构登录邮箱并发送邮件，执行结束后可自动断开与邮件服务器的连接
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as email_svr:
            # 输入QQ邮箱的账号和授权码后登录
            email_svr.login(user_name, user_pass)
            # 邮箱登录成功后即可发送邮件
            # 将收件人、抄送人、密送人以加号连接
            # email_msg.as_string()是将MIMEText对象或MIMEMultipart对象变为str
            email_svr.sendmail(sender, receivers, email_msg.as_string())
    # 如果发生可预知的smtp类错误，则执行下面代码
    except smtplib.SMTPException as e:
        print('smtp发生错误，邮件发送失败，错误信息为：', e)
    # 如果发生不可知的异常则执行下面语句结构中的代码
    except Exception as e:
        print('发生不可知的错误，错误信息为：', e)
    # 如果没发生异常则执行else语句结构中的代码
    else:
        print('邮件发送未发生任何异常，一切正常！')
    # 无论是否发生异常，均执行finally语句结构中的代码
    finally:
        print('邮件发送程序已执行完毕！')


if __name__ == '__main__':
    Set_seed.setup_seed(0)
    '''IDPil_closs     IDPil_dkgi   IDPil_gmm_dkgi'''
    model = IDPil_dkgi().to(config.device)

    Estim_Loss = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.4)

    w_params = [1, 1, 1, 1]
    train(model, Estim_Loss, optimizer, w_params, config.epochs)

    send_email("PIDspgmm")