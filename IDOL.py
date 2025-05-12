import math
import numpy as np
import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import models
import Config as config
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from scipy.spatial.distance import pdist, squareform
import torch.distributions as dist
import scipy.stats as stats
from MI_Calculate import *
import ssl
import smtplib
from email.header import Header
from email.utils import formataddr
from email.mime.text import MIMEText
from scipy.stats import gaussian_kde
from scipy.integrate import quad

#########################
### based on IDPil-ICML
### 暗知识图随机k种变化-自学习softmax(w): (b, 3)
### k=3: 随机高斯噪声/图size减小-关联矩阵1值减小/结点度增大-关联矩阵1值增加）
### add loss: max(MI(y, PIDsh))
#########################

class vgg_enc(nn.Module):
    def __init__(self, Pre_Train):
        super().__init__()
        self.backbone = models.vgg13_bn(Pre_Train)
        # print('模型各部分名称', self.backbone._modules.keys())
        self.backbone.features._modules['0'] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.share_net = self.backbone.features
        self.flatten = nn.Flatten()

    def forward(self, x_ir):
        x = self.share_net(x_ir)  # 512, 4, 4
        return x

class Plt_Dist(nn.Module):
    def __init__(self, in_dim, id_dim):
        super(Plt_Dist, self).__init__()
        self.fc_m = nn.Linear(in_dim, 1)
        self.fc_s = nn.Linear(in_dim, 1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.id_dim = id_dim
    def forward(self, plt, dist):
        b, d = plt.size()
        m = self.relu(self.fc_m(plt))
        s = self.relu(self.fc_s(plt))
        '''由plt决定强度身份模式'''
        dist_samp = m + s * dist
        # dist_samp = dist_samp.to(config.device)
        return dist_samp

class Plt_GMMDist(nn.Module):
    def __init__(self, in_dim, num_components, feature_dim):
        super().__init__()
        self.num_components = num_components
        self.feature_dim = feature_dim
        # 定义可学习参数
        self.alpha_v = nn.Linear(in_dim, num_components)
        self.alpha_p = nn.Linear(in_dim, num_components)
        self.relu = nn.ReLU()
        # self.alpha_logits = nn.Parameter(torch.randn(num_components))  # 混合权重logits
        self.mu = nn.Parameter(torch.randn(num_components, feature_dim))  # 均值向量
        self.log_var = nn.Parameter(torch.randn(num_components, feature_dim))  # 对数方差
    def forward(self, plt, dist):
        """
        输入：dist 形状为 (batch_size, feature_dim)
        输出：对数概率密度，形状为 (batch_size)
        """
        batch_size = dist.size(0)

        # 计算混合权重（Softmax归一化）
        alpha_v = F.softmax(self.relu(self.alpha_v(plt)), dim=1)  # (b, num_components)
        alpha_p = F.softmax(self.relu(self.alpha_p(plt)), dim=1)  # (b, num_components)
        # alpha_v = F.softmax(self.alpha_v(plt), dim=1)  # (b, num_components)
        # alpha_p = F.softmax(self.alpha_p(plt), dim=1)  # (b, num_components)

        # 计算方差（确保正值）
        var = torch.exp(self.log_var)  # (num_components, feature_dim)

        # 扩展维度以支持广播计算
        dist_expanded = dist.unsqueeze(1)  # (batch_size, 1, feature_dim)
        mu_expanded = self.mu.unsqueeze(0)  # (1, num_components, feature_dim)
        var_expanded = var.unsqueeze(0)  # (1, num_components, feature_dim)
        gmm_dist = mu_expanded + var_expanded * dist_expanded # (batch_size, num_components, feature_dim)
        weighted_gmm_vdist = torch.sum(alpha_v.unsqueeze(dim=-1) * gmm_dist, dim=1)    # (batch_size, feature_dim)
        weighted_gmm_pdist = torch.sum(alpha_p.unsqueeze(dim=-1) * gmm_dist, dim=1)    # (batch_size, feature_dim)
        return weighted_gmm_vdist, weighted_gmm_pdist

class PltGMM_Dist(nn.Module):
    def __init__(self, in_dim, num_components, feature_dim):
        super().__init__()
        self.num_components = num_components
        self.feature_dim = feature_dim
        # 定义可学习参数
        self.alpha_v = nn.Parameter(torch.randn(num_components))  # 混合权重logits
        self.alpha_p = nn.Parameter(torch.randn(num_components))  # 混合权重logits
        self.mu = nn.Linear(in_dim, num_components)
        self.log_var = nn.Linear(in_dim, num_components)
        self.relu = nn.ReLU()
    def forward(self, plt, dist):
        """
        输入：dist 形状为 (batch_size, feature_dim)
        输出：对数概率密度，形状为 (batch_size)
        """
        batch_size = dist.size(0)

        # 计算混合权重（Softmax归一化）
        alpha_v = F.softmax(self.alpha_v, dim=0)  # (num_components,)
        alpha_p = F.softmax(self.alpha_p, dim=0)  # (num_components,)

        # 计算方差（确保正值）
        var = self.relu(self.log_var(plt))  # (b, num_components)
        mu = self.relu(self.mu(plt))

        # 扩展维度以支持广播计算
        dist_expanded = dist.unsqueeze(1)  # (batch_size, 1, feature_dim)
        mu_expanded = mu.unsqueeze(-1)  # (b, num_components, 1)
        var_expanded = var.unsqueeze(-1)  # (b, num_components, 1)

        gmm_dist = mu_expanded + var_expanded * dist_expanded # (batch_size, num_components, feature_dim)
        weighted_gmm_vdist = torch.sum(alpha_v.unsqueeze(dim=-1) * gmm_dist, dim=1)    # (batch_size, feature_dim)
        weighted_gmm_pdist = torch.sum(alpha_p.unsqueeze(dim=-1) * gmm_dist, dim=1)    # (batch_size, feature_dim)
        return weighted_gmm_vdist, weighted_gmm_pdist

'''b, 16'''
def error_notice():
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
    email_msg = MIMEText(' your project IDPil-constraint loss : nan error over', 'plain', 'utf-8')
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
def Dist(shf, id_dim=16):
    b, c, n = shf.size()
    flatten = nn.Flatten()
    '''初始分布意味着PR引导后属性之间的潜在联系'''
    # fm = torch.mean(shf).unsqueeze(dim=-1).repeat(1, id_dim)
    # fs = torch.std(shf)
    # fs = torch.maximum(fs, torch.tensor(1e-6))
    # fs = fs.unsqueeze(dim=-1).repeat(1, id_dim)
    # fm = torch.mean(flatten(shf), dim=1).unsqueeze(dim=-1).repeat(1, id_dim)
    # fs = torch.std(flatten(shf), dim=1).unsqueeze(dim=-1).repeat(1, id_dim)
    '''b, 512, 16--->b, 16'''
    if torch.isnan(shf).any():
        error_notice()
        raise ValueError("shf contains NaN")
    fm = torch.mean(shf, dim=1)
    fs = torch.std(shf, dim=1)
    fm = torch.clamp(fm, min=-10, max=10)
    fs = torch.clamp(fs, min=1e-6, max=10)
    # fs = torch.maximum(fs, torch.tensor(1e-6))
    dist_samp = torch.normal(fm, fs)
    # dist_samp = dist_samp.to(config.device)
    return dist_samp

'''进入循环前，未知偏置项：自学习参数'''
class EquFM(nn.Module):
    def __init__(self, in_dim, uk_node_num, tolerance=1e-3):
        super(EquFM, self).__init__()
        self.tolerance = tolerance
        self.ukn_num = uk_node_num
        self.adj = self.create_adj_matrix(uk_node_num)
        self.gcn1 = GCNConv(in_dim, 2 * in_dim)
        self.gcn2 = GCNConv(2 * in_dim, in_dim)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.fc11 = nn.Linear(in_dim * (uk_node_num + 2), in_dim)
        self.fc12 = nn.Linear(in_dim * 2, in_dim)
        self.fc13 = nn.Linear(in_dim * 2, in_dim)
        self.fc14 = nn.Linear(in_dim * (uk_node_num + 2), in_dim)

    def create_adj_matrix(self, num_nodes):
        # 初始化为对角线全为1的矩阵
        adj = torch.eye(num_nodes)

        # 设置相邻节点连接
        for i in range(num_nodes):
            if i > 0:
                adj[i, i - 1] = 1
                adj[i - 1, i] = 1
            if i < num_nodes - 1:
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1

        return adj
    def FM(self, x, num_iter): ##[B, 16]
        # memory = torch.zeros_like(x)  # [B, 16]
        H = torch.zeros_like(x)
        C = torch.randn_like(x) * 1e-6
        uk_nodef = nn.Parameter(torch.randn_like(x).unsqueeze(dim=1).repeat(1, self.ukn_num, 1))
        '''随机采样生成0/1矩阵'''
        # adj = torch.bernoulli(torch.sigmoid(self.adjp))
        adj = self.adj
        adj = adj.to(config.device)
        '''b, uk_node_num, id_num'''
        uk_gdata = GraphD_Construt(uk_nodef, adj)
        ukn, uk_eidx = uk_gdata.x.to(config.device), uk_gdata.edge_index.to(config.device)
        # ukn, uk_eidx = uk_gdata.x, uk_gdata.edge_index
        # wadj = adj * self.sigmoid(self.adjw)
        # wuk_gdata = GraphD_Construt(uk_nodef, wadj)
        # wukn, wuk_eidx = wuk_gdata.x.to(config.device), wuk_gdata.edge_index.to(config.device)
        '''b, uk_node_num * id_num'''
        ukf = self.gcn2(self.gcn1(ukn, uk_eidx), uk_eidx)
        # ukf = self.gcn2(self.gcn1(wukn, wuk_eidx), wuk_eidx)
        ukf = self.flatten(ukf)
        for i in range(num_iter):  # T次循环
            '''遗忘门'''
            h_old = H
            a_xh = torch.sigmoid(self.fc11(torch.cat([H, x, ukf], dim=1)))
            ca_xh = C*a_xh
            '''输入门'''
            ga = torch.sigmoid(self.fc12(torch.cat((H, x), dim=1)))
            gv = torch.tanh(self.fc13(torch.cat((H, x), dim=1)))
            C = ca_xh + ga*gv
            '''输出门'''
            a_xh1 = torch.sigmoid(self.fc14(torch.cat([H, x, ukf], dim=1)))
            H = a_xh1 * torch.tanh(C)
            # memory, H = self.self_attention_memory(memory.unsqueeze(dim=1), H.unsqueeze(dim=1))  # H:torch.Size([B, 1, 16, 16])
            # if jsd_converg(H, h_old) < self.tolerance:
            #     print("jsd of H and h_old = ", jsd_converg(H, h_old))
            #     print("iterations converged really by tolerance, iter = ", i+1)
            #     break
            if torch.norm(H - h_old) < self.tolerance:
                print("iterations converged really by tolerance, iter = ", i+1)
                break
        if i == num_iter - 1:
            print("iterations converged by num_iter=", i+1)
        return H, i + 1

    def forward(self, x, num_iter): #torch.Size([B, 16])
        B,_ = x.size()
        H, iter = self.FM(x, num_iter)
        return H, iter

'''P(r)+V(r)+nop2v'''
class TCAID_pv_nop2v(nn.Module):
    def __init__(self, in_dim, id_dim):
        super(TCAID_pv_nop2v, self).__init__()
        # self.distp = Plt_Dist(in_dim, id_dim)
        # self.distv = Plt_Dist(in_dim, id_dim)
        '''PltGMM_Dist  Plt_GMMDist'''
        self.gmm = Plt_GMMDist(in_dim, 3, id_dim)
        self.uk_node_num = 3
        self.id_dim = id_dim
        # self.uknf_rm = nn.Linear(id_dim * 2, id_dim * self.uk_node_num)
        # self.A = nn.Parameter(torch.randn(id_dim))
        self.A = nn.Parameter(torch.randn(1))
        self.F2_rm = nn.Parameter(torch.randn(2))
        self.F3_rm = nn.Parameter(torch.randn(2))
        self.iter_pow1 = EquFM(id_dim, self.uk_node_num)

        # self.uknf_r34 = nn.Linear(id_dim * 2, id_dim * self.uk_node_num)
        self.F4_r34 = nn.Parameter(torch.randn(2))
        self.F5_r34 = nn.Parameter(torch.randn(2))
        # self.iter_pow2 = SA_FC_LSTM(id_dim)
        self.iter_pow2 = EquFM(id_dim, self.uk_node_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, plt, dist, num_iter=20):
        '''
        Args:
            dist: (batch_size, id_dim=16)
            plt: (batch_size, in_dim=2)
            goal: (batch_size, 4)
        Returns:
            4 task 代表表征pp: (batch_size, 1, id_dim=16)
        '''
        # idp = self.distp(plt, dist)
        # idv = self.distv(plt, dist)
        idv, idp = self.gmm(plt, dist)

        b, d = idp.size()
        '''
        Holland部分
        rm = [A / (lnF2(v) - lnF3(v))]^pow = C^pow
        r34 = [C / (lnF4(p) - lnF5(p))]^pow = D^pow
        '''
        F2P = self.F2_rm[0] * idv + self.F2_rm[1]
        F3P = self.F3_rm[0] * idv + self.F3_rm[1]
        lnF2p = torch.log(self.sigmoid(F2P))
        lnF3p = torch.log(self.sigmoid(F3P))
        # b, n, dim = idp.size()
        # A = self.A.reshape(1, 1, dim)
        # C = A / (lnF2p - lnF3p)
        C = self.A[0] / (lnF2p - lnF3p)
        '''idr1 = torch.pow(C, self.pow[0])'''
        idr1, iter1 = self.iter_pow1(C, num_iter)
        # uk_nodef_r1 = self.uknf_rm(torch.cat([idv, idp], dim=-1)).reshape(b, self.uk_node_num, self.id_dim)
        # idr1, iter1 = self.iter_pow1(C, num_iter, uk_nodef_r1)

        F4P = self.F4_r34[0] * idp + self.F4_r34[1]
        F5P = self.F5_r34[0] * idp + self.F5_r34[1]
        lnF4p = torch.log(self.sigmoid(F4P))
        lnF5p = torch.log(self.sigmoid(F5P))
        D = C / (lnF4p - lnF5p)
        idr2, iter2 = self.iter_pow2(D, num_iter)
        # uk_nodef_r2 = self.uknf_r34(torch.cat([idv, idp], dim=-1)).reshape(b, self.uk_node_num, self.id_dim)
        # idr2, iter2 = self.iter_pow2(D, num_iter, uk_nodef_r2)
        # idr2 = torch.pow(D, iter1)
        return idv, idp, idr1, idr2

class TCAID_pv_nop2v_ori(nn.Module):
    def __init__(self, in_dim, id_dim):
        super(TCAID_pv_nop2v_ori, self).__init__()
        self.distp = Plt_Dist(in_dim, id_dim)
        self.distv = Plt_Dist(in_dim, id_dim)
        self.uk_node_num = 3
        self.id_dim = id_dim
        # self.uknf_rm = nn.Linear(id_dim * 2, id_dim * self.uk_node_num)
        # self.A = nn.Parameter(torch.randn(id_dim))
        self.A = nn.Parameter(torch.randn(1))
        self.F2_rm = nn.Parameter(torch.randn(2))
        self.F3_rm = nn.Parameter(torch.randn(2))
        self.iter_pow1 = EquFM(id_dim, self.uk_node_num)

        # self.uknf_r34 = nn.Linear(id_dim * 2, id_dim * self.uk_node_num)
        self.F4_r34 = nn.Parameter(torch.randn(2))
        self.F5_r34 = nn.Parameter(torch.randn(2))
        # self.iter_pow2 = SA_FC_LSTM(id_dim)
        self.iter_pow2 = EquFM(id_dim, self.uk_node_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, plt, dist, num_iter=20):
        '''
        Args:
            dist: (batch_size, id_dim=16)
            plt: (batch_size, in_dim=2)
            goal: (batch_size, 4)
        Returns:
            4 task 代表表征pp: (batch_size, 1, id_dim=16)
        '''
        idp = self.distp(plt, dist)
        idv = self.distv(plt, dist)

        b, d = idp.size()
        '''
        Holland部分
        rm = [A / (lnF2(v) - lnF3(v))]^pow = C^pow
        r34 = [C / (lnF4(p) - lnF5(p))]^pow = D^pow
        '''
        F2P = self.F2_rm[0] * idv + self.F2_rm[1]
        F3P = self.F3_rm[0] * idv + self.F3_rm[1]
        lnF2p = torch.log(self.sigmoid(F2P))
        lnF3p = torch.log(self.sigmoid(F3P))
        # b, n, dim = idp.size()
        # A = self.A.reshape(1, 1, dim)
        # C = A / (lnF2p - lnF3p)
        C = self.A[0] / (lnF2p - lnF3p)
        '''idr1 = torch.pow(C, self.pow[0])'''
        idr1, iter1 = self.iter_pow1(C, num_iter)
        # uk_nodef_r1 = self.uknf_rm(torch.cat([idv, idp], dim=-1)).reshape(b, self.uk_node_num, self.id_dim)
        # idr1, iter1 = self.iter_pow1(C, num_iter, uk_nodef_r1)

        F4P = self.F4_r34[0] * idp + self.F4_r34[1]
        F5P = self.F5_r34[0] * idp + self.F5_r34[1]
        lnF4p = torch.log(self.sigmoid(F4P))
        lnF5p = torch.log(self.sigmoid(F5P))
        D = C / (lnF4p - lnF5p)
        idr2, iter2 = self.iter_pow2(D, num_iter)
        # uk_nodef_r2 = self.uknf_r34(torch.cat([idv, idp], dim=-1)).reshape(b, self.uk_node_num, self.id_dim)
        # idr2, iter2 = self.iter_pow2(D, num_iter, uk_nodef_r2)
        # idr2 = torch.pow(D, iter1)
        return idv, idp, idr1, idr2

def initialize_weights(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # if m.bias is not None:
            #     nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

class GEncOpt(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(GEncOpt, self).__init__()

        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

    def reparametrize(self, mu, logstd, init_dist):
        return mu + init_dist * torch.exp(logstd)

    def forward(self, data, init_dist):
        x, edge_index = data.x.to(config.device), data.edge_index.to(config.device)
        # x, edge_index = data.x, data.edge_index
        mu, logvar = self.encode(x, edge_index)
        z = self.reparametrize(mu, logvar, init_dist)
        # return z, mu, logvar
        return z

class GInvar_EncOpt(nn.Module):
    def __init__(self, in_channels, out_channels, k=3):
        super(GInvar_EncOpt, self).__init__()
        self.k = k
        self.conv = GCNConv(in_channels, 2 * out_channels)  # 新增的基础GCN层
        self.flatten = nn.Flatten()
        self.vary_gate = nn.Linear(32, 3)
        self.conv_vary1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_vary2 = GCNConv(in_channels, 2 * out_channels)
        self.conv_vary3 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def GraphD_Construt(self, nodef, adj):
        # 构造边索引（Edge Index）和边权重（Edge Weight）
        edge_index = []
        edge_weight = []
        b, n, d = nodef.size()
        # 构造输入和估计值之间的边
        for i in range(n):
            for j in range(n):
                if adj[i, j] != 0:
                    edge_index.append([i, j])  # 从输入节点到输出节点
                    edge_index.append([j, i])  # 从输出节点到输入节点
                    edge_weight.append(adj[i, j])
                    edge_weight.append(adj[i, j])

        edge_index = torch.tensor(edge_index).t().contiguous()  # 转置并转换为tensor
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        data = Data(x=nodef, edge_index=edge_index, edge_attr=edge_weight)
        return data

    def add_noise(self, nodef):
        return nodef + torch.randn_like(nodef) * 0.1

    def graph_cut(self, adj):
        adj = adj.clone()
        n = adj.size(0)
        rows, cols = torch.where(torch.triu(adj) == 1)
        indices = torch.randperm(len(rows))[:max(1, len(rows) // 2)]
        adj[rows[indices], cols[indices]] = 0
        adj[cols[indices], rows[indices]] = 0
        return adj

    def edge_add(self, adj):
        adj = adj.clone()
        n = adj.size(0)
        triu = torch.triu(torch.ones_like(adj, dtype=torch.bool), 1)
        zeros = (adj == 0) & triu
        rows, cols = torch.where(zeros)
        if len(rows) > 0:
            indices = torch.randperm(len(rows))[:min(15 - (adj == 1).sum(), len(rows))]
            adj[rows[indices], cols[indices]] = 1
            adj[cols[indices], rows[indices]] = 1
        return adj

    def reparametrize(self, mu, logvar, init_dist):
        return mu + init_dist * torch.exp(logvar)

    def invarloss(self, w, w_hat):
        l1 = nn.L1Loss()
        invarl = l1(w, w_hat)
        # ce = nn.CrossEntropyLoss()
        # invarl = ce(w, w_hat)
        return invarl
    def forward(self, nodef, adj, init_dist):
        b, n, _ = nodef.shape

        # 基础特征提取
        gdata = self.GraphD_Construt(nodef, adj).to(config.device)
        x = self.conv(gdata.x, gdata.edge_index).view(b, n, -1)     # b, 4, 8
        vary_weight = F.softmax(self.vary_gate(self.flatten(x)), dim=-1)    # b, 3

        # 生成三种变换特征
        adj_cut = self.graph_cut(adj)
        adj_add = self.edge_add(adj)
        vary1_data = self.GraphD_Construt(self.add_noise(nodef), adj).to(config.device)
        vary2_data = self.GraphD_Construt(nodef, adj_cut).to(config.device)
        vary3_data = self.GraphD_Construt(nodef, adj_add).to(config.device)

        vary1_x = self.conv_vary1(vary1_data.x, vary1_data.edge_index).view(b, n, -1)
        vary2_x = self.conv_vary2(vary2_data.x, vary2_data.edge_index).view(b, n, -1)
        vary3_x = self.conv_vary3(vary3_data.x, vary3_data.edge_index).view(b, n, -1)

        # 加权融合
        invar_x = (vary_weight[:, 0].view(b, 1, 1) * vary1_x +
                   vary_weight[:, 1].view(b, 1, 1) * vary2_x +
                   vary_weight[:, 2].view(b, 1, 1) * vary3_x)

        # 最终编码
        invar_data = self.GraphD_Construt(invar_x, adj).to(config.device)
        mu = self.conv_mu(invar_data.x, invar_data.edge_index)
        logvar = self.conv_logstd(invar_data.x, invar_data.edge_index)
        z = self.reparametrize(mu, logvar, init_dist)

        # w = self.vary_gate(self.flatten(invar_x)
        w = F.softmax(self.vary_gate(self.flatten(invar_x)), dim=-1)
        # op_idx = torch.argmax(w, dim=1)
        # invarl = self.invarloss(vary_weight, op_idx)
        # soft_w = F.softmax(w, dim=-1)
        invarl = self.invarloss(w, vary_weight)

        # return z, mu, logvar
        return z, mu, logvar, invarl

class GInvar_Att(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 共享的特征编码器
        self.base_conv = GCNConv(in_channels, 8)
        self.vary_convs = nn.ModuleList([
            GCNConv(in_channels, 8) for _ in range(3)])

        # 交叉注意力机制
        self.attn = nn.MultiheadAttention(8, 4)
        self.norm = nn.LayerNorm(8)

        # 不变特征生成
        self.mu_net = GCNConv(8, out_channels)

        self.logvar_net = GCNConv(8, out_channels)

    def GraphD_Construt(self, nodef, adj):
        # 构造边索引（Edge Index）和边权重（Edge Weight）
        edge_index = []
        edge_weight = []
        b, n, d = nodef.size()
        # 构造输入和估计值之间的边
        for i in range(n):
            for j in range(n):
                if adj[i, j] != 0:
                    edge_index.append([i, j])  # 从输入节点到输出节点
                    edge_index.append([j, i])  # 从输出节点到输入节点
                    edge_weight.append(adj[i, j])
                    edge_weight.append(adj[i, j])

        edge_index = torch.tensor(edge_index).t().contiguous()  # 转置并转换为tensor
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        data = Data(x=nodef, edge_index=edge_index, edge_attr=edge_weight)
        return data
    def graph_cut(self, adj):
        adj = adj.clone()
        n = adj.size(0)
        rows, cols = torch.where(torch.triu(adj) == 1)
        indices = torch.randperm(len(rows))[:max(1, len(rows) // 2)]
        adj[rows[indices], cols[indices]] = 0
        adj[cols[indices], rows[indices]] = 0
        return adj
    def edge_add(self, adj):
        adj = adj.clone()
        n = adj.size(0)
        triu = torch.triu(torch.ones_like(adj, dtype=torch.bool), 1)
        zeros = (adj == 0) & triu
        rows, cols = torch.where(zeros)
        if len(rows) > 0:
            indices = torch.randperm(len(rows))[:min(15 - (adj == 1).sum(), len(rows))]
            adj[rows[indices], cols[indices]] = 1
            adj[cols[indices], rows[indices]] = 1
        return adj
    def forward(self, nodef, adj, init_dist):
        b, n, _ = nodef.shape
        gdata = self.GraphD_Construt(nodef, adj).to(config.device)

        # 生成多视图特征
        views = []
        for i in range(3):
            if i == 0:  # 噪声视图
                perturbed = self.GraphD_Construt(nodef + torch.randn_like(nodef) * 0.01, adj).to(config.device)
            elif i == 1:  # 边删除视图
                perturbed = self.GraphD_Construt(nodef, self.graph_cut(adj)).to(config.device)
            else:  # 边添加视图
                perturbed = self.GraphD_Construt(nodef, self.edge_add(adj)).to(config.device)

            views.append(self.vary_convs[i](perturbed.x, perturbed.edge_index))

        # 注意力融合
        fused, _ = self.attn(
            self.base_conv(gdata.x, gdata.edge_index).view(1, -1, 8),  # Query
            torch.stack(views).view(3, -1, 8),  # Key
            torch.stack(views).view(3, -1, 8)  # Value
        )
        fused = fused.view(b, n, 8)
        invar_gdata = self.GraphD_Construt(fused, adj).to(config.device)
        # 生成分布参数
        mu = self.mu_net(invar_gdata.x, invar_gdata.edge_index)
        logvar = self.logvar_net(invar_gdata.x, invar_gdata.edge_index)
        z = mu + torch.exp(logvar) * init_dist
        return z, mu, logvar

class GEnc_GMMDist(nn.Module):
    def __init__(self, in_channels, out_channels, num_components, feature_dim):
        super().__init__()
        self.num_components = num_components
        self.feature_dim = feature_dim
        self.relu = nn.ReLU()
        self.conv_z = GCNConv(in_channels, 2 * out_channels)
        self.alpha_sh = GCNConv(2 * out_channels, 1)
        self.mu = nn.Parameter(torch.randn(num_components, feature_dim))  # 均值向量
        self.log_var = nn.Parameter(torch.randn(num_components, feature_dim))  # 对数方差
    def forward(self, gdata, dist):
        """
        输入：dist 形状为 (batch_size, feature_dim)
        输出：对数概率密度，形状为 (batch_size)
        """
        batch_size = dist.size(0)
        z = self.conv_z(gdata.x, gdata.edge_index)
        # 计算混合权重（Softmax归一化）
        alpha_sh = F.softmax(self.alpha_sh(z, gdata.edge_index).squeeze(dim=-1), dim=-1)      # (b, num_components)
        # 计算方差（确保正值）
        var = torch.exp(self.log_var)  # (num_components, feature_dim)

        # 扩展维度以支持广播计算
        dist_expanded = dist.unsqueeze(1)  # (batch_size, 1, feature_dim)
        mu_expanded = self.mu.unsqueeze(0)  # (1, num_components, feature_dim)
        var_expanded = var.unsqueeze(0)  # (1, num_components, feature_dim)
        gmm_dist = mu_expanded + var_expanded * dist_expanded # (batch_size, num_components, feature_dim)
        weighted_gmm_dist = torch.sum(alpha_sh.unsqueeze(dim=-1) * gmm_dist, dim=1)    # (batch_size, feature_dim)
        return weighted_gmm_dist

class GMM_GEncDist(nn.Module):
    def __init__(self, in_channels, out_channels, num_components, feature_dim):
        super().__init__()
        self.num_components = num_components
        self.feature_dim = feature_dim
        self.vary_GVAE = nn.ModuleList([
            GEncOpt(in_channels, out_channels) for _ in range(num_components)])
        self.alpha_sh = nn.Parameter(torch.randn(num_components))  # 混合权重logits
    def forward(self, gdata, dist):
        """
        输入：dist 形状为 (batch_size, feature_dim)
        输出：对数概率密度，形状为 (batch_size)
        """
        batch_size = dist.size(0)
        vary_dists = []
        for i in range(self.num_components):
            vary_dists.append(self.vary_GVAE[i](gdata, dist))
        # b, num_components, 4, 4
        vary_dists = torch.cat([i.unsqueeze(dim=1) for i in vary_dists], dim=1)
        alpha_sh = F.softmax(self.alpha_sh, dim=0).unsqueeze(dim=0).unsqueeze(dim=-1)  # (1, num_components, 1)
        weighted_gmm_dist = torch.sum(alpha_sh * vary_dists.view(batch_size, self.num_components, -1), dim=1)    # (batch_size, feature_dim)
        return weighted_gmm_dist

class PRG_SALSTM8(nn.Module):
    def __init__(self, in_channels):
        super(PRG_SALSTM8, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

        self.conv7 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.conv8 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.conv9 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

        self.conv10 = nn.Conv2d(in_channels*2+1, in_channels, kernel_size=1)
        self.conv11 = nn.Conv2d(in_channels*2+1, in_channels, kernel_size=1)
        self.conv12 = nn.Conv2d(in_channels*2+1, in_channels, kernel_size=1)
        self.conv13 = nn.Conv2d(in_channels*2+1, in_channels, kernel_size=1)

    def sa_conv_lstm(self, x, en_1d): ##[T, B, 512, 4, 4] [B,1,4,4]
        # #看sa-conv-lstm的
        # M,H：每一个小的 都是（B，256）
        memory = torch.zeros_like(x[0])  # [B, 512, 4, 4]
        H = torch.zeros_like(x[0])
        C = torch.randn_like(x[0]) * 1e-6
        for i in range(x.size(0)):  # T次循环
            a_xh = torch.sigmoid(self.conv10(torch.cat((H, x[i], en_1d), dim=1)))  #到单通道吗
            ca_xh = C*a_xh
            ga = torch.sigmoid(self.conv11(torch.cat((H, x[i], en_1d), dim=1)))
            gv = torch.tanh(self.conv12(torch.cat((H, x[i], en_1d), dim=1)))
            C = ca_xh + ga*gv
            a_xh1 = torch.sigmoid(self.conv13(torch.cat((H, x[i], en_1d), dim=1)))
            H = a_xh1*torch.tanh(C)
            memory, H = self.self_attention_memory(memory, H)  # H:torch.Size([B, 1, 16, 16])
        return H

    def self_attention_memory(self, m, h): #[B, 512, 4, 4]
        vh = self.conv1(h)
        kh = self.conv2(h)
        qh = self.conv3(h)

        qh = torch.transpose(qh, 2, 3)
        ah = F.softmax(kh*qh,dim=-1) #基本全是0.0625 0.0624
        zh = vh*ah

        km = self.conv4(m)
        vm = self.conv5(m)
        am = F.softmax(qh*km,dim=-1)
        zm = vm*am
        z0 = torch.cat((zh, zm), dim=1)
        z = self.conv6(z0)
        hz = torch.cat((h, z), dim=1)

        ot = torch.sigmoid(self.conv7(hz))  #到单通道吗
        gt = torch.tanh(self.conv8(hz))
        it = torch.sigmoid(self.conv9(hz))

        gi = gt*it
        mf = (1-it)*m
        mt = gi+mf
        ht = ot*mt

        return mt,ht

    def forward(self, x, en_1d): #torch.Size([B, T, 1, 16, 16]) #[B,1,16,16]
        B,_,_,_,_ = x.size()  #最好还是B,T,C,H,W
        x = x.permute(1, 0, 2, 3, 4) #torch.Size([T, B, 1, 16, 16])
        H = self.sa_conv_lstm(x, en_1d)#[T, B, 1, 16, 16] [B,1,16,16]
        # flattened_tensor = H.view(B, -1)
        return H #(B,256) 特别趋同

def GraphD_Construt(nodef, adj):
    # 构造边索引（Edge Index）和边权重（Edge Weight）
    edge_index = []
    edge_weight = []
    b, n, d = nodef.size()
    # 构造输入和估计值之间的边
    for i in range(n):
        for j in range(n):
            if adj[i, j] != 0:
                edge_index.append([i, j])  # 从输入节点到输出节点
                edge_index.append([j, i])  # 从输出节点到输入节点
                edge_weight.append(adj[i, j])
                edge_weight.append(adj[i, j])

    edge_index = torch.tensor(edge_index).t().contiguous()  # 转置并转换为tensor
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    data = Data(x=nodef, edge_index=edge_index, edge_attr=edge_weight)
    return data

class IDPil_dkgi(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = vgg_enc(False)
        self.enc2 = vgg_enc(False)

        '''先时序融合得到(b, 512, 4, 4)分布采样得到初始的共享ID（分布）；
                用PRF作为节点特征(b, 4, 4)与Source矩阵送入GraphVAE生成调节的均值和方差'''
        self.pr_fc = nn.Linear(4, 16)
        self.prg_fus = PRG_SALSTM8(in_channels=512)
        '''GInvar_EncOpt    GInvar_Att  GEnc_GMMDist    GMM_GEncDist'''
        self.opt = GEnc_GMMDist(in_channels=4, out_channels=4, num_components=4, feature_dim=16)

        self.decoup = TCAID_pv_nop2v_ori(in_dim=2, id_dim=16)

        '''Transformation with SA'''
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.id_dim, nhead=4, dim_feedforward=config.id_dim * 2,
                                                   activation="gelu", dropout=0.1, batch_first=True)
        self.encoder1 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder3 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder4 = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.flatten = nn.Flatten()

        self.output_msw = nn.Sequential(
            nn.Linear(514 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.output_mslp = nn.Sequential(
            nn.Linear(514 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.output_rmw = nn.Sequential(
            nn.Linear(514 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.output_r34 = nn.Sequential(
            nn.Linear(514 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x, plt, pr, shgt):
        f1 = self.enc1(x[:, :4]).unsqueeze(dim=1)      # b, 1, 512, 4, 4
        f2 = self.enc1(x[:, 4:]).unsqueeze(dim=1)      # b, 1, 512, 4, 4
        tf = torch.cat([f1, f2], dim=1)          # b, 2, 512, 4, 4
        b, t, c, h, w = tf.size()
        prf = self.pr_fc(pr).reshape(b, 1, h, w)            # b, 1, 4, 4
        prgf = self.prg_fus(tf, prf).reshape(b, c, h * w)   # b, 512, 16

        dist = Dist(prgf, id_dim=16)
        nodef = prf.squeeze(dim=1)  # b, 4, 4
        adj = torch.tensor([[0, 0, 1, 1],
                            [1, 0, 1, 0],
                            [1, 1, 0, 0],
                            [0, 1, 0, 1]], dtype=torch.float)

        '''GInvar_Att'''
        # shid, mu, logvar = self.opt(nodef, adj, dist.reshape(b, h, w))
        '''GEnc_GMMDist'''
        gdata = GraphD_Construt(nodef, adj).to(config.device)
        shid = self.opt(gdata, dist)

        shid = shid.reshape(b, h * w)

        idv, idp, idr1, idr2 = self.decoup(plt, dist)

        '''b, 16--->b, 1, 16'''
        idv, idp, idr1, idr2 = idv.unsqueeze(dim=1), idp.unsqueeze(dim=1), idr1.unsqueeze(dim=1), idr2.unsqueeze(dim=1)
        shid = shid.unsqueeze(dim=1)
        '''shid, pid, F'''
        f1 = self.encoder1(torch.cat([shid, idv, prgf], dim=1))
        f2 = self.encoder2(torch.cat([shid, idp, prgf], dim=1))
        f3 = self.encoder3(torch.cat([shid, idr1, prgf], dim=1))
        f4 = self.encoder4(torch.cat([shid, idr2, prgf], dim=1))
        '''pid, shid, F'''
        # f1 = self.encoder1(torch.cat([idv, shid, prgf], dim=1))
        # f2 = self.encoder2(torch.cat([idp, shid, prgf], dim=1))
        # f3 = self.encoder3(torch.cat([idr1, shid, prgf], dim=1))
        # f4 = self.encoder4(torch.cat([idr2, shid, prgf], dim=1))
        '''shid, F, pid'''
        # f1 = self.encoder1(torch.cat([shid, prgf, idv], dim=1))
        # f2 = self.encoder2(torch.cat([shid, prgf, idp], dim=1))
        # f3 = self.encoder3(torch.cat([shid, prgf, idr1], dim=1))
        # f4 = self.encoder4(torch.cat([shid, prgf, idr2], dim=1))
        f1 = self.flatten(f1)
        f2 = self.flatten(f2)
        f3 = self.flatten(f3)
        f4 = self.flatten(f4)

        msw = self.output_msw(f1)
        mslp = self.output_mslp(f2)
        rmw = self.output_rmw(f3)
        r34 = self.output_r34(f4)
        msw, mslp, rmw, r34 = msw[:, 0], mslp[:, 0], rmw[:, 0], r34[:, 0]

        '''PIDsh constraint: MI of (y, PIDsh)'''
        # loss_sh = self.MIloss(shid.squeeze(dim=1), shgt)
        # print("NMI_shid_shgt_loss: ", loss_sh.item())
        '''共性表征约束损失-SimR'''
        # loss_sh = self.SimRloss(self.flatten(prgf), shid)
        # print("SimRsh_f loss: ", loss_sh.item())
        SimRsh_y = self.SimRloss(shid.squeeze(dim=1), shgt)
        print("SimRsh_y loss: ", SimRsh_y.item())
        SimRsp_y = self.SimRloss(idv.squeeze(dim=1), shgt[:, 0].unsqueeze(dim=1)) +\
                   self.SimRloss(idp.squeeze(dim=1), shgt[:, 1].unsqueeze(dim=1)) +\
                   self.SimRloss(idr1.squeeze(dim=1), shgt[:, 2].unsqueeze(dim=1)) +\
                   self.SimRloss(idr2.squeeze(dim=1), shgt[:, 3].unsqueeze(dim=1))
        print("SimRsp_y loss: ", SimRsp_y.item())
        SimR_loss = SimRsh_y + SimRsp_y
        # return msw, mslp, rmw, r34, invarl
        return msw, mslp, rmw, r34, SimR_loss
        # return msw, mslp, rmw, r34
    def SimRloss(self, m, n):
        sim_m = m @ m.transpose(0,1)
        sim_n = n @ n.transpose(0,1)
        DiffL = nn.L1Loss()
        simrloss = DiffL(sim_m, sim_n)
        return simrloss
    def initialize(self):
        initialize_weights(self)

class SelfAttentionLSTM8(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionLSTM8, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv6 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

        self.conv7 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.conv8 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.conv9 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

        self.conv10 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.conv11 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.conv12 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        self.conv13 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

    def sa_conv_lstm(self, x): ##[T, B, 512, 4, 4]
        # #看sa-conv-lstm的
        # M,H：每一个小的 都是（B，256）
        memory = torch.zeros_like(x[0])  # [B, 512, 4, 4]
        H = torch.zeros_like(x[0])
        C = torch.randn_like(x[0]) * 1e-6
        for i in range(x.size(0)):  # T次循环
            a_xh = torch.sigmoid(self.conv10(torch.cat((H, x[i]), dim=1)))  #到单通道吗
            ca_xh = C*a_xh
            ga = torch.sigmoid(self.conv11(torch.cat((H, x[i]), dim=1)))
            gv = torch.tanh(self.conv12(torch.cat((H, x[i]), dim=1)))
            C = ca_xh + ga*gv
            a_xh1 = torch.sigmoid(self.conv13(torch.cat((H, x[i]), dim=1)))
            H = a_xh1*torch.tanh(C)
            memory, H = self.self_attention_memory(memory, H)  # H:torch.Size([B, 512, 4, 4])
        return H

    def self_attention_memory(self, m, h): #[B, 512, 4, 4]
        vh = self.conv1(h)
        kh = self.conv2(h)
        qh = self.conv3(h)

        qh = torch.transpose(qh, 2, 3)
        ah = F.softmax(kh*qh,dim=-1) #基本全是0.0625 0.0624
        zh = vh*ah

        km = self.conv4(m)
        vm = self.conv5(m)
        am = F.softmax(qh*km,dim=-1)
        zm = vm*am
        z0 = torch.cat((zh, zm), dim=1)
        z = self.conv6(z0)
        hz = torch.cat((h, z), dim=1)

        ot = torch.sigmoid(self.conv7(hz))  #到单通道吗
        gt = torch.tanh(self.conv8(hz))
        it = torch.sigmoid(self.conv9(hz))

        gi = gt*it
        mf = (1-it)*m
        mt = gi+mf
        ht = ot*mt

        return mt,ht

    def forward(self, x):
        B,_,_,_,_ = x.size()  #最好还是B,T,C,H,W
        x = x.permute(1, 0, 2, 3, 4)
        H = self.sa_conv_lstm(x)        # [B, 512, 4, 4]
        return H

class PIDspgmm(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = vgg_enc(False)
        self.enc2 = vgg_enc(False)

        self.fus = SelfAttentionLSTM8(in_channels=512)

        self.decoup = TCAID_pv_nop2v(in_dim=2, id_dim=16)

        '''Transformation with SA'''
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.id_dim, nhead=4, dim_feedforward=config.id_dim * 2,
                                                   activation="gelu", dropout=0.1, batch_first=True)
        self.encoder1 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder3 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder4 = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.flatten = nn.Flatten()

        self.output_msw = nn.Sequential(
            nn.Linear(513 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.output_mslp = nn.Sequential(
            nn.Linear(513 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.output_rmw = nn.Sequential(
            nn.Linear(513 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.output_r34 = nn.Sequential(
            nn.Linear(513 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x, plt, shgt):
        f1 = self.enc1(x[:, :4]).unsqueeze(dim=1)      # b, 1, 512, 4, 4
        f2 = self.enc1(x[:, 4:]).unsqueeze(dim=1)      # b, 1, 512, 4, 4
        tf = torch.cat([f1, f2], dim=1)          # b, 2, 512, 4, 4
        b, t, c, h, w = tf.size()
        fusedf = self.fus(tf)  # b, 512, 4, 4
        fusedf = fusedf.reshape(b, c, h * w)  # b, 512, 16

        dist = Dist(fusedf, id_dim=16)
        idv, idp, idr1, idr2 = self.decoup(plt, dist)

        '''b, 16--->b, 1, 16'''
        idv, idp, idr1, idr2 = idv.unsqueeze(dim=1), idp.unsqueeze(dim=1), idr1.unsqueeze(dim=1), idr2.unsqueeze(dim=1)
        '''pid, F'''
        f1 = self.encoder1(torch.cat([idv, fusedf], dim=1))
        f2 = self.encoder2(torch.cat([idp, fusedf], dim=1))
        f3 = self.encoder3(torch.cat([idr1, fusedf], dim=1))
        f4 = self.encoder4(torch.cat([idr2, fusedf], dim=1))

        f1 = self.flatten(f1)
        f2 = self.flatten(f2)
        f3 = self.flatten(f3)
        f4 = self.flatten(f4)

        msw = self.output_msw(f1)
        mslp = self.output_mslp(f2)
        rmw = self.output_rmw(f3)
        r34 = self.output_r34(f4)
        msw, mslp, rmw, r34 = msw[:, 0], mslp[:, 0], rmw[:, 0], r34[:, 0]

        '''特性表征约束损失-SimR'''
        SimRsp_y = self.SimRloss(idv.squeeze(dim=1), shgt[:, 0].unsqueeze(dim=1)) +\
                   self.SimRloss(idp.squeeze(dim=1), shgt[:, 1].unsqueeze(dim=1))
        print("SimRsp_y loss: ", SimRsp_y.item())
        return msw, mslp, rmw, r34, SimRsp_y
    def SimRloss(self, m, n):
        sim_m = m @ m.transpose(0,1)
        sim_n = n @ n.transpose(0,1)
        DiffL = nn.L1Loss()
        simrloss = DiffL(sim_m, sim_n)
        return simrloss
    def initialize(self):
        initialize_weights(self)

'''PIDspgmm + PIDsh原'''
class IDPil(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = vgg_enc(False)
        self.enc2 = vgg_enc(False)

        '''先时序融合得到(b, 512, 4, 4)分布采样得到初始的共享ID（分布）；
                用PRF作为节点特征(b, 4, 4)与Source矩阵送入GraphVAE生成调节的均值和方差'''
        self.pr_fc = nn.Linear(4, 16)
        self.prg_fus = PRG_SALSTM8(in_channels=512)
        '''GEncOpt  GInvar_EncOpt    GInvar_Att'''
        self.opt = GEncOpt(in_channels=4, out_channels=4)

        self.decoup = TCAID_pv_nop2v(in_dim=2, id_dim=16)

        '''Transformation with SA'''
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.id_dim, nhead=4, dim_feedforward=config.id_dim * 2,
                                                   activation="gelu", dropout=0.1, batch_first=True)
        self.encoder1 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder3 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder4 = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.flatten = nn.Flatten()

        self.output_msw = nn.Sequential(
            nn.Linear(514 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.output_mslp = nn.Sequential(
            nn.Linear(514 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.output_rmw = nn.Sequential(
            nn.Linear(514 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.output_r34 = nn.Sequential(
            nn.Linear(514 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x, plt, pr, shgt):
        f1 = self.enc1(x[:, :4]).unsqueeze(dim=1)      # b, 1, 512, 4, 4
        f2 = self.enc1(x[:, 4:]).unsqueeze(dim=1)      # b, 1, 512, 4, 4
        tf = torch.cat([f1, f2], dim=1)          # b, 2, 512, 4, 4
        b, t, c, h, w = tf.size()
        prf = self.pr_fc(pr).reshape(b, 1, h, w)            # b, 1, 4, 4
        prgf = self.prg_fus(tf, prf).reshape(b, c, h * w)   # b, 512, 16

        dist = Dist(prgf, id_dim=16)
        nodef = prf.squeeze(dim=1)  # b, 4, 4
        adj = torch.tensor([[0, 0, 1, 1],
                            [1, 0, 1, 0],
                            [1, 1, 0, 0],
                            [0, 1, 0, 1]], dtype=torch.float)

        gdata = GraphD_Construt(nodef, adj)
        shid, mu, logvar = self.opt(gdata, dist.reshape(b, h, w))
        # shid, mu, logvar = self.opt(nodef, adj, dist.reshape(b, h, w))
        # shid, mu, logvar, invarl = self.opt(nodef, adj, dist.reshape(b, h, w))
        # print("dkgi invar loss: ", invarl.item())

        shid = shid.reshape(b, h * w)

        idv, idp, idr1, idr2 = self.decoup(plt, dist)

        '''b, 16--->b, 1, 16'''
        idv, idp, idr1, idr2 = idv.unsqueeze(dim=1), idp.unsqueeze(dim=1), idr1.unsqueeze(dim=1), idr2.unsqueeze(dim=1)
        shid = shid.unsqueeze(dim=1)
        '''shid, pid, F'''
        f1 = self.encoder1(torch.cat([shid, idv, prgf], dim=1))
        f2 = self.encoder2(torch.cat([shid, idp, prgf], dim=1))
        f3 = self.encoder3(torch.cat([shid, idr1, prgf], dim=1))
        f4 = self.encoder4(torch.cat([shid, idr2, prgf], dim=1))
        '''pid, shid, F'''
        # f1 = self.encoder1(torch.cat([idv, shid, prgf], dim=1))
        # f2 = self.encoder2(torch.cat([idp, shid, prgf], dim=1))
        # f3 = self.encoder3(torch.cat([idr1, shid, prgf], dim=1))
        # f4 = self.encoder4(torch.cat([idr2, shid, prgf], dim=1))
        '''shid, F, pid'''
        # f1 = self.encoder1(torch.cat([shid, prgf, idv], dim=1))
        # f2 = self.encoder2(torch.cat([shid, prgf, idp], dim=1))
        # f3 = self.encoder3(torch.cat([shid, prgf, idr1], dim=1))
        # f4 = self.encoder4(torch.cat([shid, prgf, idr2], dim=1))
        f1 = self.flatten(f1)
        f2 = self.flatten(f2)
        f3 = self.flatten(f3)
        f4 = self.flatten(f4)

        msw = self.output_msw(f1)
        mslp = self.output_mslp(f2)
        rmw = self.output_rmw(f3)
        r34 = self.output_r34(f4)
        msw, mslp, rmw, r34 = msw[:, 0], mslp[:, 0], rmw[:, 0], r34[:, 0]

        '''PIDsh constraint: MI of (y, PIDsh)'''
        # loss_sh = self.MIloss(shid.squeeze(dim=1), shgt)
        # print("NMI_shid_shgt_loss: ", loss_sh.item())
        '''共性表征约束损失-SimR'''
        # loss_sh = self.SimRloss(self.flatten(prgf), shid)
        # print("SimRsh_f loss: ", loss_sh.item())
        # SimRsh_y = self.SimRloss(shid.squeeze(dim=1), shgt)
        # print("SimRsh_y loss: ", SimRsh_y.item())
        # SimRsp_y = self.SimRloss(idv.squeeze(dim=1), shgt[:, 0].unsqueeze(dim=1)) + \
        #            self.SimRloss(idp.squeeze(dim=1), shgt[:, 1].unsqueeze(dim=1))
        # print("SimRsp_y loss: ", SimRsp_y.item())
        # auxloss = SimRsh_y + SimRsp_y
        # auxloss = loss_sh + SimRsp_y
        # return msw, mslp, rmw, r34, invarl
        # return msw, mslp, rmw, r34, loss_sh
        return msw, mslp, rmw, r34
    def SimRloss(self, m, n):
        sim_m = m @ m.transpose(0,1)
        sim_n = n @ n.transpose(0,1)
        DiffL = nn.L1Loss()
        simrloss = DiffL(sim_m, sim_n)
        return simrloss
    def IDsp_DSimloss(self, IDsp, Ysp):
        std_idsp = torch.std(IDsp)
        std_ysp = torch.std(Ysp, dim=-1)
        DiffL = nn.L1Loss()
        DSimloss = DiffL(std_idsp, std_ysp)
        return DSimloss
    def MIloss(self, m, n):
        MI = calculate_MI(m, n)
        MIloss = -torch.log(MI)
        return MIloss
    def initialize(self):
        initialize_weights(self)

'''PIDspgmm + PIDsh_dkgi'''
class IDPil_gmm_dkgi(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = vgg_enc(False)
        self.enc2 = vgg_enc(False)

        '''先时序融合得到(b, 512, 4, 4)分布采样得到初始的共享ID（分布）；
                用PRF作为节点特征(b, 4, 4)与Source矩阵送入GraphVAE生成调节的均值和方差'''
        self.pr_fc = nn.Linear(4, 16)
        self.prg_fus = PRG_SALSTM8(in_channels=512)
        '''GInvar_Att     GEnc_GMMDist'''
        self.opt = GEnc_GMMDist(in_channels=4, out_channels=4, num_components=4, feature_dim=16)

        self.decoup = TCAID_pv_nop2v(in_dim=2, id_dim=16)

        '''Transformation with SA'''
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.id_dim, nhead=4, dim_feedforward=config.id_dim * 2,
                                                   activation="gelu", dropout=0.1, batch_first=True)
        self.encoder1 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder2 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder3 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.encoder4 = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.flatten = nn.Flatten()

        self.output_msw = nn.Sequential(
            nn.Linear(514 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.output_mslp = nn.Sequential(
            nn.Linear(514 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.output_rmw = nn.Sequential(
            nn.Linear(514 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.output_r34 = nn.Sequential(
            nn.Linear(514 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x, plt, pr, shgt):
        f1 = self.enc1(x[:, :4]).unsqueeze(dim=1)      # b, 1, 512, 4, 4
        f2 = self.enc1(x[:, 4:]).unsqueeze(dim=1)      # b, 1, 512, 4, 4
        tf = torch.cat([f1, f2], dim=1)          # b, 2, 512, 4, 4
        b, t, c, h, w = tf.size()
        prf = self.pr_fc(pr).reshape(b, 1, h, w)            # b, 1, 4, 4
        prgf = self.prg_fus(tf, prf).reshape(b, c, h * w)   # b, 512, 16

        dist = Dist(prgf, id_dim=16)
        nodef = prf.squeeze(dim=1)  # b, 4, 4
        adj = torch.tensor([[0, 0, 1, 1],
                            [1, 0, 1, 0],
                            [1, 1, 0, 0],
                            [0, 1, 0, 1]], dtype=torch.float)

        '''GInvar_Att'''
        # shid, mu, logvar = self.opt(nodef, adj, dist.reshape(b, h, w))
        '''GEnc_GMMDist'''
        gdata = GraphD_Construt(nodef, adj).to(config.device)
        shid = self.opt(gdata, dist)

        shid = shid.reshape(b, h * w)

        idv, idp, idr1, idr2 = self.decoup(plt, dist)

        '''b, 16--->b, 1, 16'''
        idv, idp, idr1, idr2 = idv.unsqueeze(dim=1), idp.unsqueeze(dim=1), idr1.unsqueeze(dim=1), idr2.unsqueeze(dim=1)
        shid = shid.unsqueeze(dim=1)
        '''shid, pid, F'''
        f1 = self.encoder1(torch.cat([shid, idv, prgf], dim=1))
        f2 = self.encoder2(torch.cat([shid, idp, prgf], dim=1))
        f3 = self.encoder3(torch.cat([shid, idr1, prgf], dim=1))
        f4 = self.encoder4(torch.cat([shid, idr2, prgf], dim=1))

        f1 = self.flatten(f1)
        f2 = self.flatten(f2)
        f3 = self.flatten(f3)
        f4 = self.flatten(f4)

        msw = self.output_msw(f1)
        mslp = self.output_mslp(f2)
        rmw = self.output_rmw(f3)
        r34 = self.output_r34(f4)
        msw, mslp, rmw, r34 = msw[:, 0], mslp[:, 0], rmw[:, 0], r34[:, 0]

        SimRsh_y = self.SimRloss(shid.squeeze(dim=1), shgt)
        print("SimRsh_y loss: ", SimRsh_y.item())
        return msw, mslp, rmw, r34, SimRsh_y
    def SimRloss(self, m, n):
        sim_m = m @ m.transpose(0,1)
        sim_n = n @ n.transpose(0,1)
        DiffL = nn.L1Loss()
        simrloss = DiffL(sim_m, sim_n)
        return simrloss
    def initialize(self):
        initialize_weights(self)

if __name__ == '__main__':
    '''example'''
    xtest = torch.randn(8, 8, 156, 156)
    x1 = torch.randn(8, 4, 156, 156)
    x2 = torch.randn(8, 4, 156, 156)
    pr = torch.randn(8, 4)
    plt = torch.randn(8, 2)
    id = torch.randn(8, 16)
    goal = torch.randn(8, 4)
    pl = torch.randn(8, 1)
    # samp_res = truncated_normal_sampling(id)
    # dl = kl_divergence_loss(id, goal)
    # print(dl)
    net = IDPil_dkgi()
    msw, mslp, rmw, r34, SimRsh_y = net(xtest, plt, pr, goal)
    end = 1
