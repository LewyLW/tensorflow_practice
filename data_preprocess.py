import pandas as pd
import numpy as np

def get_datasets(batch_size, time_step, input_size):

    data_frame = pd.read_csv('./dataset.csv')
    datas = data_frame.iloc[:,2:].values
    # norm_datas = (datas - np.mean(datas, axis=0))/np.std(datas, axis=0)
    train_size = int(len(datas)*0.8)//batch_size * batch_size + time_step
    train_datas = datas[:train_size]
    train_datas = (train_datas - np.mean(train_datas, axis=0))/np.std(train_datas, axis=0)
    test_datas = datas[train_size:]
    test_datas = (test_datas - np.mean(test_datas, axis=0))/np.std(test_datas, axis=0)

    # get the train sets
    train_x, train_y = [], []
    batch_x, batch_y = [], []
    for i in range(train_size -time_step):
        batch_x.append(train_datas[i:i+time_step, :-1])
        batch_y.append(train_datas[i:i+time_step, -1, np.newaxis])

        if (i+1)%batch_size == 0:
            train_x.append(batch_x)
            train_y.append(batch_y)
            batch_x, batch_y = [], []
        # train_x.append(np.reshape(x, [-1, time_step, input_size]))
        # train_y.append(np.reshape(y, [-1, time_step, 1]))

    # get the test sets
    test_x, test_y = [], []
    for i in range(len(test_datas)//time_step):
        x = test_datas[i*time_step : (i+1)*time_step, :-1]
        y = test_datas[i*time_step : (i+1)*time_step, -1]
        test_x.append(np.reshape(x, [-1, time_step, input_size]))
        test_y.append(np.reshape(y, [-1, time_step, 1]))
    
    return train_x, train_y, test_x, test_y


