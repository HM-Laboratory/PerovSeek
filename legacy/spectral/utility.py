import pandas as pd
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader,random_split
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import r2_score 
import math
import matplotlib.pyplot as plt
from scipy.constants import h, c, eV
import os
import model
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from scipy.interpolate import interp1d

def dataload(data,index,batch):
    Abs, PL_t, PL_b, pce_label = data
    train_index, val_index, test_indices = index
    # merge
    spectrum = torch.FloatTensor(np.concatenate([Abs, PL_t, PL_b], axis=-1))
    
    spectrum_train = spectrum[train_index]
    pce_train = torch.FloatTensor(pce_label[train_index,:].reshape(-1,1))
    
    spectrum_val = spectrum[val_index]
    pce_val = torch.FloatTensor(pce_label[val_index,:].reshape(-1,1))
    
    spectrum_test = spectrum[test_indices]
    pce_test = torch.FloatTensor(pce_label[test_indices,:].reshape(-1,1))
    
    train_dataset = TensorDataset(spectrum_train,pce_train)   
    val_dataset = TensorDataset(spectrum_val,pce_val)   
    test_dataset = TensorDataset(spectrum_test,pce_test)    

    # 创建 DataLoader
    data_train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )
    data_val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )


    return [train_dataset, val_dataset, test_dataset],[data_train_dataloader,data_val_dataloader]


def model_define(in_dim,dropout,n,model_path,params):

    # pre-trained model definition
    spectrum_model = torch.load(model_path,weights_only=False)#.load_state_dict(state_dict)
    # fine-tune model definition
    model = model2.FineTuneModel2(spectrum_model,n=n, in_dim = in_dim, dropout_rate = dropout).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = params['lr'], weight_decay = params['weight_decay'])

    return model, optimizer, params

def datadivision(total_num,train_num,test_num):
    indices = np.arange(total_num)
    np.random.seed(42)
    np.random.shuffle(indices)  # 随机打乱索引
    test_indices = indices[-test_num:]
    train_index = indices[:train_num]
    val_index = indices[train_num:-test_num]

    return train_index, val_index, test_indices


def load_ptmodel(path,device):
    m = model.MAE(
        in_channel=1, embed_dim=36, decoder_dim=18, patch_size=[15,11], stride=[15,11], num_patches=[20,12],   
        mask_ratio=[0.6,0.4], encoder_depth=24,decoder_depth=1,mlp_ratio = 4,qkv_bias = True,
        num_encoder_heads=6, num_decoder_heads=6,device=device
    ).to(device)
    # model = model3.MAE(
    #     in_channel=1, embed_dim=36, decoder_dim=36, patch_size=[15,11], stride=[15,11], num_patches=[20,12],   
    #     mask_ratio=0.5, encoder_depth=24,decoder_depth=1,mlp_ratio = 4,qkv_bias = True,
    #     num_encoder_heads=6, num_decoder_heads=6,device=device
    # ).to(device)
        
    # 加载保存的 state_dict
    state_dict = torch.load(path,weights_only=True, map_location=device)
    
    # 将 state_dict 加载到模型
    m.load_state_dict(state_dict)

    return m


def plot_distribution_and_calculate_probability(data, region, plot=True):
    """
    绘制数据分布图并计算 [-1, +1] 范围内的概率。
    
    参数:
        data: 输入数据 (PyTorch 张量或 NumPy 数组)
        scale_factor: 数据缩放因子 (默认 25.0，基于上下文)
        plot: 是否绘制分布直方图 (默认 True)
    
    返回:
        float: [-1, +1] 范围内的概率
    """

    low_region, high_region = region

    data = data.flatten()  # 确保一维
        # 转换为 NumPy 数组并去缩放
    valid_mask = ~np.isnan(data)
    
    # 提取有效数据
    data = data[valid_mask]
    
    # 使用 KDE 估计概率密度
    kde = gaussian_kde(data)
    
    # 计算 [-2, +2] 范围内的概率（积分）
    prob, _ = quad(kde, low_region, high_region)
    prob = min(max(prob, 0.0), 1.0)  # 限制在 [0, 1]
    
    # 绘制分布图
    if plot:
        # 数据范围（均值 ± 3 标准差）
        mean = np.mean(data)
        std = np.std(data)
        x = np.linspace(mean - 3*std, mean + 3*std, 1000)
        kde_values = kde(x)
        
        # 绘制直方图和 KDE 曲线
        plt.figure(figsize=(5.5, 5.5))
        plt.hist(data, bins=50, density=True, alpha=0.5, color='skyblue', edgecolor='black', label='Histogram')
        plt.plot(x, kde_values, 'r-', label='KDE', linewidth=2)
        
        # 标记 [-1, +1] 范围
        plt.fill_between(x, 0, kde_values, where=(x >= low_region) & (x <= high_region), color='coral', alpha=0.3, label=f'[{low_region}, {high_region}] Range')
        
        plt.xlabel('Prediction error (%)',fontsize=16)
        plt.ylabel('Density',fontsize=16)
        plt.xlim(-20,20)
        plt.title(f'Distribution (Probability in [{low_region}, {high_region}]: {prob:.4f})')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return prob

def parameter_status_table(model):
    data = []
    for name, param in model.named_parameters():
        data.append([
            name,
            param.shape,
            f"{param.numel()/1e3:.1f}K",
            '✓' if param.requires_grad else '✗'
        ])
    
    df = pd.DataFrame(data, columns=['Parameter', 'Shape', 'Size', 'Trainable'])
    print(df.to_markdown(index=False))

def set_seed(seed):
    # 设置 Python 的随机种子
    random.seed(seed)

    # 设置 NumPy 的随机种子
    np.random.seed(seed)

    # 设置 PyTorch CPU 的随机种子
    torch.manual_seed(seed)

    # 设置 PyTorch CUDA 的随机种子（如果使用 GPU）
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU


#     # 设置 PyTorch 的全局确定性行为
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

def adjust_learning_rate(optimizer, current_epoch, params):
    # lr cos 0->pai decay
    lr = params['min_lr'] + (params['lr'] - params['min_lr']) * 0.5 * \
         (1. + math.cos(
             math.pi * (current_epoch - params['warmup_epochs']) / (params['epochs'] - params['warmup_epochs'])))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def dataset_division(data,batch):  
    
    seed = 42
    set_seed(seed)
    
    data_tensor = torch.FloatTensor(data)
    # 创建完整数据集
    data_full_dataset = TensorDataset(data_tensor)

    # 按比例划分训练集、验证集和测试集
    train_size = int(0.8 * len(data_full_dataset))  # 60% 训练集
    val_size = int(0.1 * len(data_full_dataset))    # 20% 验证集
    test_size = len(data_full_dataset) - train_size - val_size  # 剩余 20% 测试集

    data_train_dataset, data_val_dataset, data_test_dataset = random_split(
        data_full_dataset, 
        [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(seed)  # 设置随机种子
    )

    # 创建 DataLoader
    data_train_dataloader = DataLoader(
        dataset=data_train_dataset,
        batch_size=batch,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )

    data_val_dataloader = DataLoader(
        dataset=data_val_dataset,
        batch_size=batch,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )

    data_test_dataloader = DataLoader(
        dataset=data_test_dataset,
        batch_size=batch,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    
    return data_train_dataloader, data_val_dataloader, data_test_dataloader 


def train(model, optimizer, train_dataloader, val_dataloader, params,norm_spec_loss=False):
    device = params['device']
    num_epochs = params['epochs']
    loss_list = []  # 训练集损失
    val_loss_list = []  # 验证集损失
    lr_list = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()  # 确保处于训练模式
        epoch_loss = 0

        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            # print(data[0].shape)
            # 计算训练损失

            loss, _, _, _ = model.forward_loss([data[0][:,:,:300].to(device), data[0][:,:,300:].to(device)], norm_spec_loss=norm_spec_loss)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            epoch_loss += loss.item()
            adjust_learning_rate(optimizer, i / len(train_dataloader) + epoch + 1, params)

        current_lr = optimizer.param_groups[0]['lr']
        avg_train_loss = epoch_loss / (i + 1)
        loss_list.append(avg_train_loss)
        lr_list.append(current_lr)

        # 验证阶段
        model.eval()  # 设置为评估模式
        val_loss = 0
        with torch.no_grad():  # 关闭梯度计算
            for i, data in enumerate(val_dataloader):
                # 计算验证损失
                loss, _, _, _ = model.forward_loss([data[0][:,:,:300].to(device), data[0][:,:,300:].to(device)], norm_spec_loss=norm_spec_loss)
                val_loss += loss.item()

        avg_val_loss = val_loss / (i + 1)
        val_loss_list.append(avg_val_loss)

        # 打印训练和验证结果
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.10f}, '
              f'Val Loss: {avg_val_loss:.10f}, '
              f'lr: {current_lr:.6f}')

    return loss_list, val_loss_list, lr_list
    
def fine_train(model, optimizer, train_dataloader, val_dataloader, params, save):
    LOSS = torch.nn.HuberLoss(delta=params['delta'])
    device = params['device']
    num_epochs = params['epochs']
    loss_list = []    # 训练集损失
    val_loss_list = [] # 验证集损失
    lr_list = []
    
    # 初始化最小验证损失和模型保存路径
    min_val_loss = float('inf')
    best_model_path = None

    if save is not None:
        if not os.path.exists(save):
            # 如果不存在，就创建文件夹
            os.makedirs(save)

    for epoch in range(num_epochs):   
        # 训练阶段
        model.train()  # 确保处于训练模式
        epoch_loss = 0
        
        for i, data in enumerate(train_dataloader):  
            optimizer.zero_grad()  

            # 计算训练损失  
            pred,_ = model(data[0].to(device))
            loss = torch.sum((pred-data[-1].to(device))**2)#LOSS(pred, data[-1].to(device))

            # 反向传播  
            loss.backward()  

            # 更新参数  
            optimizer.step()  

            epoch_loss += loss.item()
            adjust_learning_rate(optimizer, i / len(train_dataloader) + epoch + 1, params)
            
        current_lr = optimizer.param_groups[0]['lr']
        avg_train_loss = epoch_loss / (i + 1)
        loss_list.append(avg_train_loss)
        lr_list.append(current_lr)
        
        # 验证阶段
        model.eval()  # 设置为评估模式
        val_loss = 0
        with torch.no_grad():  # 关闭梯度计算
            for i, data in enumerate(val_dataloader):
                # 计算验证损失  
                pred,_ = model(data[0].to(device))
                loss = torch.sum((pred-data[-1].to(device))**2)#LOSS(pred, data[-1].to(device))
                val_loss += loss.item()
        avg_val_loss = val_loss / (i + 1)
        val_loss_list.append(avg_val_loss)

        # 打印训练和验证结果
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.10f}, '
              f'Val Loss: {avg_val_loss:.10f}, '
              f'lr: {current_lr:.6f}')
        
        # 保存验证损失最小的模型
        if (epoch+1) % 10 == 0:
            torch.save(model, f'{save}/Fusion_epoch_{epoch + 1}.pth')
        if save and avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            # 删除之前的模型（如果存在）
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            # 保存新模型
            best_model_path = f'{save}/Fusion_best_epoch_{epoch + 1}.pth'
            torch.save(model, best_model_path)
            print(f'New best model saved at {best_model_path} with Val Loss: {min_val_loss:.10f}')

    plt.plot(loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.show()

    return loss_list, val_loss_list, lr_list,best_model_path


def data_loading_wave(path, pl_t_norm, pl_b_norm, abs_start_index, pl_start_index, with_wave):
    
    df = pd.read_excel(path,sheet_name=None)
    
    def normalization(data):
        mean = data.mean(axis=-1).reshape(-1, 1)
        std = data.std(axis=-1).reshape(-1, 1)

        data_norm = (data - mean) / (std + 1e-8)

        return data_norm, mean, std

    #abs_end_index = 240
    pl_end_index = pl_start_index+66

    Abs_wavelength = df['Abs'].iloc[:, 0].values[abs_start_index:]
    PL_wavelength = df['PLs_top'].iloc[:, 0].values[pl_start_index:pl_end_index]
    
    Abs_wavelength_relative = Abs_wavelength - 400
    PL_wavelength_relative = PL_wavelength - 400

    Abs = df['Abs'].values[abs_start_index:, 1:].T.reshape(-1, 1, Abs_wavelength.shape[0])
    pl_t = df['PLs_top'].values[pl_start_index:pl_end_index, 1:].T.reshape(-1, 1, PL_wavelength.shape[0])
    pl_b = df['PLs_bottom'].values[pl_start_index:pl_end_index, 1:].T.reshape(-1, 1, PL_wavelength.shape[0])

    if (Abs_wavelength[1]-Abs_wavelength[0]) != 2:
        wavelength_old = Abs_wavelength  # (201,)
        wavelength_new = np.arange(402, 1000 + 2, 2)  # (301,)
        
        # 提取需要插值的列 (从第 2 列开始)
        data_to_interpolate = Abs.reshape(Abs.shape[0],-1).T  # 形状 (201, N-1)

        # 创建插值函数，应用于所有列
        f = interp1d(wavelength_old, data_to_interpolate, axis=0, kind='linear', bounds_error=False, fill_value="extrapolate")
        result_data = f(wavelength_new).T.reshape(Abs.shape[0],1,300)  # 形状 (301, N-1)

        Abs = result_data
        Abs_wavelength = wavelength_new
        Abs_wavelength_relative = Abs_wavelength - 400

    Abs = Abs/4
    pl_t = pl_t/pl_t_norm
    pl_b = pl_b/pl_b_norm

    if with_wave:
    
        Abs = np.concatenate([Abs, np.tile(Abs_wavelength_relative, (Abs.shape[0], 1)).reshape(Abs.shape[0],1,Abs.shape[-1])],axis=1)
        pl_t = np.concatenate([pl_t, np.tile(PL_wavelength_relative, (pl_t.shape[0], 1)).reshape(pl_t.shape[0],1,pl_t.shape[-1])],axis=1)
        pl_b = np.concatenate([pl_b, np.tile(PL_wavelength_relative, (pl_b.shape[0], 1)).reshape(pl_b.shape[0],1,pl_b.shape[-1])],axis=1)


    try:
        pce_r1 = df['Param']['PCE#rs1'].values.reshape(-1, 1)
        voc_r1 = df['Param']['Voc#rs1'].values.reshape(-1, 1)
        ff_r1 = df['Param']['FF#rs1'].values.reshape(-1, 1)
        jsc_r1 = df['Param']['Jsc#rs1'].values.reshape(-1, 1)
    except Exception as e:
        pce_r1 = None

    
    try:

        photo_df = df['Param'][
            ['Scattering_H', 'Scattering_S', 'Scattering_L', 'Transmission_H', 'Transmission_S', 'Transmission_L']]

        # 将 HSL 转换为 RGB
        def hsl_to_rgb(h, s, l):
            # 将色相从 [0, 360] 归一化到 [0, 1]
            h = h / 360.0
            # 使用 colorsys 转换
            r, g, b = colorsys.hls_to_rgb(h, l, s)
            return r, g, b

        # 应用到 DataFrame
        photo_df[['sR', 'sG', 'sB']] = photo_df.apply(
            lambda row: hsl_to_rgb(row['Scattering_H'], row['Scattering_S'] / 100, row['Scattering_L'] / 100),
            axis=1,
            result_type='expand'
        )
        photo_df[['tR', 'tG', 'tB']] = photo_df.apply(
            lambda row: hsl_to_rgb(row['Transmission_H'], row['Transmission_S'] / 100, row['Transmission_L'] / 100),
            axis=1, result_type='expand'
        )

        image_feat = photo_df[['sR', 'sG', 'sB', 'tR', 'tG', 'tB']].values
    except:
        image_feat = None


    return Abs, pl_t, pl_b, image_feat, [pce_r1,voc_r1,ff_r1,jsc_r1], Abs_wavelength,PL_wavelength,df

def myplots(axes, y_true_list, y_pred_list, region):
    fs = 14
    color_list = ["blue", "orange", "red"]


    for i in range(len(y_true_list)):
        # 找到 y_true 和 y_pred 中均非 NaN 的索引
        valid_mask = ~np.isnan(y_true_list[i]) & ~np.isnan(y_pred_list[i])
        
        # 提取有效数据
        y_true_valid = y_true_list[i][valid_mask]
        y_pred_valid = y_pred_list[i][valid_mask]

        
        rmse = RMSE(y_true_valid, y_pred_valid)
        mae = np.abs(y_true_valid - y_pred_valid).mean()
        title = " (RMSE = %.2f, r2: %.2f, MAE = %.2f, num: %d)" % (
            rmse, r2_score(y_true_valid, y_pred_valid), mae, len(y_true_valid)
        )
        
        axes.scatter(y_true_valid, y_pred_valid, alpha=0.6, c=color_list[i], label=title)
        axes.plot(region, region, 'k--', alpha=0.75, zorder=0)
        # axes.errorbar(y_true_list[i], y_pred_list[i], yerr=var_list[i], ms=0, 
        #               ls='', capsize=2, alpha=0.6,
        #               color='gray', zorder=0)
        
        # 设置坐标轴标签，指定字体大小（例如 16）
        axes.set_xlabel('Ground Truth', fontsize=16)
        axes.set_ylabel('Prediction', fontsize=16)
        
        axes.set_xlim(region)
        axes.set_ylim(region)
        axes.legend()
        axes.grid(True, linestyle='-.')