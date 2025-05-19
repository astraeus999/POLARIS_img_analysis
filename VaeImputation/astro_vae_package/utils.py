import numpy as np

def crop_central_squared_image(img, edge_size=256):
    h,w = img.shape
    c_x, c_y = h//2, w//2
    half_edge = edge_size//2
    return img[c_x-half_edge:c_x+half_edge, c_y-half_edge:c_y+half_edge]
    
def get_log_img(img):
    data = img.copy()
    data[data <= 1] = 1
    log_data = np.log(data)
    return log_data

def normalize_img(img):
    data = img.copy()
    data = data.reshape(-1,)
    m = np.mean(data[data>0])
    s = np.std(data[data>0])
    if s == 0: 
       s = 1
    norm = (data[data>0] - m)/s
    # norm = NormalizeVec(data[data>0])
    data[data > 0] = norm
    return data.reshape(img.shape)
    
def convert_to_tensor(data, device='cpu'):
    return torch.tensor(data, dtype=torch.float32).to(device)

def convert_to_tensors(data):
    return [torch.tensor(x, dtype=torch.float32) for x in data]