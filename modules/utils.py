import pandas as pd
import numpy as np

def get_device_data(full_df, device_id, cols):
    device_df = full_df[full_df.device_number==device_id]
    return device_df[cols]

def preprocessing(device_df):
    device_df.dropna(inplace=True)
    device_df = device_df.rename({'timestamp':'time', 'pm2_5_calibrated_value':'pm2_5'}, axis=1)
    device_df['time'] = [time.timestamp()/3600 for time in device_df['time']]
    return device_df

def cross_validation(final_df, idx, kernel_variance, lengthscales, likelihood_variance, trainable_kernel, 
                     trainable_variance, trainable_lengthscales):
    device_indices = final_df[final_df.latitude==latitudes[idx]].index
    # device_df = jinja_df[jinja_df.device_number == device_ids[idx]]
    # assert(len(device_indices) == len(device_df)-device_df.pm2_5_calibrated_value.isna().sum())
    
    test_df = final_df.loc[device_indices]
    assert(len(test_df.longitude.unique()) == 1)
    
    train_df = pd.concat([final_df, test_df]).drop_duplicates(keep=False)
    # assert(len(train_df.longitude.unique()) == len(longitudes)-1)
    assert len(final_df) == len(test_df) + len(train_df)
    
    X_train = train_df.iloc[:, 0:-1]
    y_train = train_df.iloc[:, -1]
    X_train, y_train = np.array(X_train), np.array(y_train).reshape(-1, 1)
    if X_train.shape[0] > 9999:
        X_train = X_train[::2, :]
        y_train = y_train[::2, :]
    
    X_test = test_df.iloc[:, 0:-1]
    y_test = test_df.iloc[:, -1]
    X_test, y_test = np.array(X_test), np.array(y_test).reshape(-1, 1)
    #to delete
    #X_train, y_train, X_test, y_test = X_train[:100, :], y_train[:100, :], X_test[:100, :], y_test[:100, :]
    
    if lengthscales == 'train_shape':
        lengthscales = np.ones(X_train.shape[1])
    
    if (lengthscales is None) & (kernel_variance is None):
        k = gpflow.kernels.RBF() + gpflow.kernels.Bias()
    elif lengthscales is None:
        k = gpflow.kernels.RBF(variance=kernel_variance) + gpflow.kernels.Bias()
    elif kernel_variance is None:
        k = gpflow.kernels.RBF(lengthscales=lengthscales) + gpflow.kernels.Bias()
    else:
        k = gpflow.kernels.RBF(lengthscales=lengthscales, variance=kernel_variance) + gpflow.kernels.Bias()
        
    m = gpflow.models.GPR(data=(X_train, y_train), kernel=k, mean_function=None)
    if likelihood_variance is None:
        pass
    else:
        m.likelihood.variance.assign(likelihood_variance)
    set_trainable(m.kernel.kernels[0].variance, trainable_kernel)
    set_trainable(m.likelihood.variance, trainable_variance)
    set_trainable(m.kernel.kernels[0].lengthscales, trainable_lengthscales)
    
    #optimization
    opt = gpflow.optimizers.Scipy()
    def objective_closure():
        return - m.log_marginal_likelihood()
    
    opt_logs = opt.minimize(objective_closure,
                            m.trainable_variables,
                            options=dict(maxiter=100))

    #prediction
    mean, var = m.predict_f(X_test)
    
    rmse = sqrt(mean_squared_error(y_test, mean.numpy()))
    return rmse
    
#     return mean.numpy(), var.numpy(), Xtest, Ytest, round(rmse, 2)