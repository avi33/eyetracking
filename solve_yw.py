# import torch
import matplotlib.pyplot as plt
import numpy as np

def unfold(x, kernel_size, stride):
    input_size = x.shape[0]

    unfolded_size = (input_size - kernel_size) // stride + 1

    # Create a view with a sliding window using stride_tricks
    shape = (unfolded_size, kernel_size)
    strides = (stride * x.strides[0], x.strides[0])

    unfolded_array = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    
    return unfolded_array



class YullieWalker:
    def __init__(self):
        pass

    @staticmethod
    def build_system_of_equantions(x, win_len, step=1):
        y = x[win_len::step]
        # X = x[:-step].unfold(dimension=0, size=win_len, step=step).flip(dims=[-1, ])    
        X = np.flip(unfold(x[:-step], kernel_size=win_len, stride=step), axis=-1)
        return X, y    
    
    @staticmethod
    def solve_yw_torch(X, y):
        A = torch.matmul(X.T, X)
        b = torch.matmul(X.T, y)
        theta = torch.linalg.lstsq(A, b).solution
        return theta

    @staticmethod
    def solve_yw(X, y):
        A = X.T @ X
        b = X.T @ y
        theta = np.linalg.lstsq(A, b)[0]
        return theta   

    @staticmethod
    def est_ar_params(x, win_len, step):
        # if isinstance(x, np.ndarray):
        #     x = torch.from_numpy(x)
        R, r = YullieWalker.build_system_of_equantions(x=x, win_len=win_len, step=step)
        theta = YullieWalker.solve_yw(R, r)
        y_pred = R @ theta #torch.matmul(R, theta)
        y_pred = np.concatenate((x[:win_len], y_pred)) #torch.cat((x[:win_len], y_pred))
        # y_pred = y_pred.numpy()
        # theta = theta.numpy()
        return y_pred, theta
        

if __name__ == "__main__":
    fs = 200
    f = 12
    t = torch.arange(0, 1, 1/fs).float()
    x = torch.sin(2*np.pi*f*t)
    std_noise = 0.1
    y = x + torch.randn_like(x)*std_noise    
    y_pred, theta = YullieWalker.est_ar_params(y.numpy(), win_len=3, step=1)
    err = torch.from_numpy(y_pred)-y
    err_rel = 20*torch.log10(err.abs().mean()/y.abs().mean())
    err_abs = 20*torch.log10(err.abs().mean())
    print(20*np.log10(std_noise))
    plt.plot(y, '.-')
    plt.plot(y_pred, 'ro-')        
    plt.title("rel err:{:.2f} abs err:{:.2f}".format(err_rel, err_abs))
    plt.grid(True)
    plt.show()   