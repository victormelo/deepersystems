import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split

def read_csv(filename):
    f = open(filename)
    f.readline()  # skip the header
    data =  np.loadtxt(f, delimiter = ',')[:, 1:]

    return data

def save_results(pred, filename):
    ids = np.arange(pred.shape[0])
    tosave = np.c_[ids,pred]
    np.savetxt(filename, tosave, fmt=['%d', '%f', '%f'],
                                   delimiter=',',
                                   header="id,slope,intercept",
                                   comments="")

def build_model():
    model = Sequential()
    model.add(Dense(100, input_dim=20, init='normal', activation='relu'))
    model.add(Dense(50, init='normal', activation='relu'))
    model.add(Dense(20, init='normal', activation='relu'))
    model.add(Dense(2, init='normal'))
    model.compile(loss='mean_squared_error', optimizer = 'adam')
    return model

def test_model(X, Y, regressor):
    X_train, X_test, y_train, y_test = train_test_split(Xtrain, ytrain, test_size=0.34)
    regressor.fit(X_train, y_train)
    pred_test = regressor.predict(X_test)
    print("[TEST] Slope mse: %s" % ((pred_test[:, 0] - y_test[:, 0])**2).mean())
    print("[TEST] Intercept mae: %s" % (abs(pred_test[:, 1] - y_test[:, 1])).mean())

def evaluate_model(Xtrain, ytrain, Xevaluation, regressor):
    regressor.fit(Xtrain, ytrain)
    pred_train = regressor.predict(Xtrain)
    pred_evaluation = regressor.predict(Xevaluation)
    save_results(pred_train, 'train_pred.csv')
    save_results(pred_evaluation, 'test_evaluation.csv')

if __name__ == '__main__':
    test = False
    Xtrain = read_csv('data/train_100k.csv')
    ytrain = read_csv('data/train_100k.truth.csv')
    Xevaluation = read_csv('data/test_100k.csv')
    regressor = KerasRegressor(build_fn=build_model, nb_epoch=250, batch_size=200, verbose=1)

    if test:
        test_model(Xtrain, ytrain, regressor)
    else:
        evaluate_model(Xtrain, ytrain, Xevaluation, regressor)
