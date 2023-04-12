import os
import numpy as np
import pandas as pd

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('../data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD3':
        data_path = os.path.join('../data/PeMSD3/pems03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7':
        data_path = os.path.join('../data/PeMSD7/pems07.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PeMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'COVID_CA':
        data_path = os.path.join('../data/Covid_19_CA/CA_COVID.npz')
        data = np.load(data_path)['arr_0']
    elif dataset == 'COVID_TX':
        data_path = os.path.join('../data/Covid_19_TX/TX_COVID.npz')
        data = np.load(data_path)['arr_0']
    elif dataset =='ETH_BY':
        data_path = os.path.join('../data/Eth_Bytom/Bytom_node_features.npz')
        data = np.load(data_path)['arr_0'][12 :, :, 0]
    elif dataset =='ETH_DE':
        data_path = os.path.join('../data/Eth_Decen/Decentraland_node_features.npz')
        data = np.load(data_path)['arr_0'][:, :, 0]
    elif dataset == 'ETH_GO':
        data_path = os.path.join('../data/Eth_Golem/Golem_node_features.npz')
        data = np.load(data_path)['arr_0']
    elif dataset == 'METR-LA':
        print('loading metra-la data!')       
        data={}
        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join('../data/METR-LA/', category + '.npz'))
            data['x_' + category] = cat_data['x']
            data['y_' + category] = cat_data['y']
	
    else:
        raise ValueError
    return data
'''
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    
'''
def load_token_price_csv(dataset):
    if dataset == 'ETH_DE':
        token_close_price = (pd.read_csv('../data/Prices_Closed/Decentraland_price.csv').values[:, 4]).reshape(-1, 1)
    elif dataset == 'ETH_BY':
        token_close_price = (pd.read_csv('../data/Prices_Closed/Bytom_price.csv').values[:, 4]).reshape(-1, 1)
    elif dataset == 'ETH_GO':
        token_close_price = (pd.read_csv('../data/Prices_Closed/Golem_price.csv').values[:, 4]).reshape(-1, 1)
    token_close_price = token_close_price.astype(np.float32)
    print(dataset+'price data loaded')
    return token_close_price
