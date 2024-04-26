import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.optimize import fmin_l_bfgs_b





# =========================== staro ==================================================
class MultinomialLogReg:
    def __init__(self):
        pass
    
    def softmax(u):
        u = np.hstack((u, np.zeros((u.shape[0], 1))))
        u = u - np.max(u, axis=1, keepdims=True)
        exps = np.exp(u)
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def build(self, X, y, intercept=True):
        classes_count = int(np.max(y) + 1)
        feats_count = X.shape[1]

        if intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
            feats_count += 1
        
        samples_range = np.arange(X.shape[0])
        def loss(betas):
            betas = betas.reshape((feats_count, classes_count-1))
            probs = MultinomialLogReg.softmax(X @ betas)
            return -np.sum(np.log(probs[samples_range, y]))

        betas_0 = np.zeros((feats_count, classes_count-1))
        betas_opt, _, _ = fmin_l_bfgs_b(loss, betas_0.flatten(), approx_grad=True, maxiter=10000)
        betas_opt = betas_opt.reshape((feats_count, classes_count-1))

        return MultinomialModel(classes_count, feats_count, betas_opt, intercept)

class MultinomialModel:
    def __init__(self, classes_count, feats_count, betas, intercept):
        self.classes_count = classes_count
        self.feats_count = feats_count
        self.betas = betas
        self.intercept = intercept
    
    def predict(self, X, epsilon=0):
        if self.intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        probs = MultinomialLogReg.softmax(X @ self.betas)
        probs = epsilon + (1 - 2 * epsilon) * probs
        return probs

# =========================== / staro ==================================================








class ANNRegression():
    # interno so posamezni primeri stolpci v matriki X, zunanje dosegljive funkcije pa obratno (kot zahtevano)
    def __init__(self, units=[], lambda_=0):
        self.units = units
        self.lambda_ = lambda_
        self.w = []


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, a):
        return a * (1 - a)
    

    # ti funkciji bosta drugacni za klasifikacijo
    def last_layer(self, x):
        return x
    def preprocess_y(self, y):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        return y
    

    def params_to_weights(self, params):
        return [params[self.params_upto_l[l]:self.params_upto_l[l+1]].reshape(self.w[l].shape) for l in range(len(self.w))]
    
    def weights_to_params(self, weights):
        return np.concatenate([w.flatten() for w in weights])
    

    def initialize(self, X, y, scale_weights=1):
        y = self.preprocess_y(y)
        n = X.shape[0]

        # velikosti plasti
        sizes = [X.shape[1]] + self.units + [y.shape[1]]
        self.sizes = sizes

        # incializira utezi in ustvari matrike gradientov
        self.w = [np.random.normal(0, scale_weights / np.sqrt(sizes[i]), (sizes[i+1], sizes[i]+1)) for i in range(len(sizes)-1)]
        self.dw = [np.zeros_like(w) for w in self.w]

        # za pretvorbo vektorja parametrov v matrike
        self.params_upto_l = [0] + [np.sum([np.prod(w.shape) for w in self.w[:l]]) for l in range(1, len(self.w)+1)]

        # ustvari matrike aktivacij
        self.a = [np.ones((sizes[l]+1, n)) for l in range(len(sizes))]
        self.a[-1] = np.ones((sizes[-1], n))
        self.delta = [np.zeros_like(a) for a in self.a]

        self.n_weights = np.sum([np.prod(w[:, :-1].shape) for w in self.w])

        return X, y
    
    
    def fit(self, X, y):
        X, y = self.initialize(X, y)

        params = self.weights_to_params(self.w)
        params_opt, _, _ = fmin_l_bfgs_b(lambda ps, X, y: self.cost_and_grad(X, y, self.params_to_weights(ps)), params, args=(X.T, y.T))
        # params_opt = params
        self.w = self.params_to_weights(params_opt)

        return self
    

    def forward_pass(self, X, ws):
        a = self.a
        a[0][:-1, :] = X
        for l in range(1, len(self.w)):
            a[l][:-1, :] = self.sigmoid(ws[l-1] @ a[l-1])

        l = len(self.w)
        a[l] = self.last_layer(ws[l-1] @ a[l-1])
        return a
    

    def cost_function(self, preds, y):
        return 0.5 * np.sum((preds - y)**2) / y.shape[1]

    def cost(self, y, aa, ws):
        # izvzamemo zadnji stolpec utezi, ki predstavlja bias
        return self.cost_function(aa[-1], y) + self.lambda_ / (2 * self.n_weights) * np.sum([np.sum(w[:, :-1]**2) for w in ws])
    

    
    def last_layer_derivative(self, a, y):
        return a - y


    def cost_and_grad(self, X, y, ws=None):
        if ws is None:
            ws = self.w
        dw = self.dw

        n = X.shape[1]

        # forward pass
        aa = self.forward_pass(X, ws)
        cost = self.cost(y, aa, ws)
        # zadnji layer
        delta = self.last_layer_derivative(aa[-1], y)
        # backward pass
        for l in range(len(ws)-1, -1, -1):
            dw[l] = delta @ aa[l].T / n
            dw[l][:, :-1] += self.lambda_ * ws[l][:, :-1] / self.n_weights
            delta = (ws[l][:, :-1].T @ delta) * self.sigmoid_prime(aa[l][:-1, :])

        return cost, self.weights_to_params(dw)


    def predict(self, X):
        # ustvari matrike aktivacij
        n = X.shape[0]
        self.a = [np.ones((self.sizes[l]+1, n)) for l in range(len(self.sizes))]
        self.a[-1] = np.ones((self.sizes[-1], n))

        preds = self.forward_pass(X.T, self.w)[-1]
        if preds.shape[0] == 1:
            return preds.flatten()
        return preds.T

    def weights(self):
        return [w.T for w in self.w]
    
    def cost_at_params(self, params, X, y):
        X = X.T
        y = self.preprocess_y(y).T
        ws = self.params_to_weights(params)
        aa = self.forward_pass(X, ws)
        return self.cost(y, aa, ws)


class ANNClassification(ANNRegression):
    def last_layer(self, x):
        # softmax
        x = x - np.mean(x, axis=0, keepdims=True)
        u = np.exp(x)
        return u / np.sum(u, axis=0, keepdims=True)

    def preprocess_y(self, y):
        y = super().preprocess_y(y)
        # one hot encoding for final layer
        y = (y == np.unique(y)).astype(int)
        return y

    def cost_function(self, preds, y):
        # cross entropy
        return -np.sum(y * np.log(preds)) / y.shape[1]
    








def numerical_gradient():
    X = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])
    y = np.array([0, 1, 2, 3])
    hard_y = np.array([0, 1, 1, 0])

    np.random.seed(42)

    print("Comapring gradients")
    print("Classification before fitting")
    fitter = ANNClassification(units=[3], lambda_=0.001)
    fitter.initialize(X, hard_y)
    compare_numerical_gradient(fitter, X, hard_y)
    m = fitter.fit(X, hard_y)
    print("Classification after fitting")
    compare_numerical_gradient(m, X, hard_y)
    print()

    print("Regression before fitting")
    fitter = ANNRegression(units=[3], lambda_=0.001)
    fitter.initialize(X, y)
    compare_numerical_gradient(fitter, X, y)
    m = fitter.fit(X, y)
    print("Regression after fitting")
    compare_numerical_gradient(m, X, y)

def compare_numerical_gradient(model, X, y, eps=1e-6, precision=10):
    cost, grad = model.cost_and_grad(X.T, model.preprocess_y(y).T)
    params = model.weights_to_params(model.w)
    errors = np.zeros_like(params)
    estimates = np.zeros_like(params)
    for i in range(len(params)):
        params[i] += eps
        c1 = model.cost_at_params(params, X, y)
        params[i] -= 2*eps
        c2 = model.cost_at_params(params, X, y)
        params[i] += eps
        estimate = (c1 - c2) / (2*eps)
        estimates[i] = estimate
        errors[i] = 2 * np.abs(estimate - grad[i]) / np.abs(estimate + grad[i])

    print("Cost:", cost.round(10))
    print("Mean relative difference between numerical and analytical gradient: ", np.mean(errors).round(precision))
    print("Max relative difference between numerical and analytical gradient: ", np.max(errors).round(precision))
    print("Stanard deviation of relative difference between numerical and analytical gradient: ", np.std(errors).round(precision))





def housing2r():
    print("Housing2r")
    housing2r = pd.read_csv('housing2r.csv')
    housing2r_y = housing2r.y.to_numpy()
    housing2r_X = housing2r.drop(columns=['y']).to_numpy()

    from sklearn.linear_model import LinearRegression
    def mse(y, preds):
        return np.mean((y - preds)**2)

    y = housing2r_y
    x = housing2r_X
    k = 4
    n_poskusov = 3

    np.random.seed(42)
    idx = np.random.permutation(len(y))
    x, y = x[idx], y[idx]
    id_splits = np.array_split(idx, k)
    x_folds, y_folds = np.array_split(x, k), np.array_split(y, k)
    losses = [np.zeros(len(y)) for i in range(n_poskusov)]
    preds = [np.zeros(len(y)) for i in range(n_poskusov)]
    for i in range(k):
        x_tr = np.concatenate([x_folds[j] for j in range(k) if j != i])
        y_tr = np.concatenate([y_folds[j] for j in range(k) if j != i])
        x_test = x_folds[i]
        y_test = y_folds[i]


        lr = LinearRegression()
        lr.fit(x_tr, y_tr)
        p = lr.predict(x_test)

        losses[0][id_splits[i]] = mse(y_test, p)
        preds[0][id_splits[i]] = p

        fitter = ANNRegression(units=[], lambda_=0.0)
        m = fitter.fit(x_tr, y_tr)
        p = m.predict(x_test)
        losses[1][id_splits[i]] = mse(y_test, p)
        preds[1][id_splits[i]] = p

        fitter = ANNRegression(units=[5], lambda_=0.1)
        m = fitter.fit(x_tr, y_tr)
        p = m.predict(x_test)
        losses[2][id_splits[i]] = mse(y_test, p)
        preds[2][id_splits[i]] = p
    
    print()
    imena = ["LogisticRegression", "ANN no hidden layers", "ANN hidden layer"]
    for i in range(n_poskusov):
        print("")
        print(imena[i])
        print(f"Loss: {losses[i].mean()} +/- {losses[i].std() / np.sqrt(len(losses[i]))}")



def log_loss(y, p):
    return -np.log(p[np.arange(len(y)), y])

def housing3():
    print("Housing3")
    housing3 = pd.read_csv('housing3.csv')
    housing3_y = housing3.Class
    # encode categories as integers
    housing3.Class = pd.Categorical(housing3.Class)
    housing3.Class = housing3.Class.cat.codes
    housing3_y = housing3.Class.to_numpy()
    housing3_X = housing3.drop(columns=['Class']).to_numpy()

    logreg = MultinomialLogReg()

    y = housing3_y
    x = housing3_X
    k = 4
    n_poskusov = 2

    np.random.seed(42)
    idx = np.random.permutation(len(y))
    x, y = x[idx], y[idx]
    id_splits = np.array_split(idx, k)
    x_folds, y_folds = np.array_split(x, k), np.array_split(y, k)
    losses = [np.zeros(len(y)) for i in range(n_poskusov)]
    accuracies = [np.zeros(len(y)) for i in range(n_poskusov)]
    preds = [np.zeros((len(y), 2)) for i in range(n_poskusov)]
    for i in range(k):
        x_tr = np.concatenate([x_folds[j] for j in range(k) if j != i])
        y_tr = np.concatenate([y_folds[j] for j in range(k) if j != i])
        x_test = x_folds[i]
        y_test = y_folds[i]

        model = logreg.build(x_tr, y_tr)
        p = model.predict(x_test)
        losses[0][id_splits[i]] = log_loss(y_test, p)
        accuracies[0][id_splits[i]] = (y_test == np.argmax(p, axis=1))
        preds[0][id_splits[i], :] = p

        model = ANNClassification(units=[3], lambda_=0.5)
        model.fit(x_tr, y_tr)
        p = model.predict(x_test)
        losses[1][id_splits[i]] = log_loss(y_test, p)
        accuracies[1][id_splits[i]] = (y_test == np.argmax(p, axis=1))
        preds[1][id_splits[i], :] = p
    
    imena = ["MultinomialRegression", "ANN 1 hidden layer"]
    for i in range(n_poskusov):
        print("")
        print(imena[i])
        print(f"Loss: {losses[i].mean()} +/- {losses[i].std() / np.sqrt(len(losses[i]))}")




def create_final_predictions():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    train_X = df_train.drop(columns=['id', 'target']).to_numpy(dtype=np.float32)

    df_train.target = df_train.target.map(lambda s: int(s[-1]) - 1)
    train_y = df_train.target.to_numpy()

    scaler = StandardScaler()
    scaler = scaler.fit(train_X)
    train_X = scaler.transform(train_X)

    test_X = df_test.drop(columns=['id']).to_numpy(dtype=np.float32)
    test_X = scaler.transform(test_X)

    np.random.seed(42)
    model = ANNClassification(units=[20], lambda_=0.3)
    model.fit(train_X, train_y)
    preds = model.predict(test_X)

    ids = np.arange(1, preds.shape[0] + 1)
    df = pd.DataFrame(preds, columns=['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9'])
    df['id'] = ids
    df = df[['id', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']]
    df.to_csv('final.txt', index=False)




def final_CV():
    print("Evaluating LogReg and ANN on final data using 4fold CV")
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    df_train = pd.read_csv('train.csv')

    train_X = df_train.drop(columns=['id', 'target']).to_numpy(dtype=np.float32)

    df_train.target = df_train.target.map(lambda s: int(s[-1]) - 1)
    train_y = df_train.target.to_numpy()

    def log_loss(y, p):
        return -np.log(p[np.arange(len(y)), y])

    y = train_y
    x = train_X
    k = 4
    n_poskusov = 2

    np.random.seed(42)

    idx = np.random.permutation(len(y))
    x, y = x[idx], y[idx]


    id_splits = np.array_split(idx, k)
    x_folds, y_folds = np.array_split(x, k), np.array_split(y, k)

    losses = [np.zeros(len(y)) for i in range(n_poskusov)]
    accuracies = [np.zeros(len(y)) for i in range(n_poskusov)]
    preds = [np.zeros((len(y), 9)) for i in range(n_poskusov)]

    models = []

    for i in range(k):
        x_tr = np.concatenate([x_folds[j] for j in range(k) if j != i])
        y_tr = np.concatenate([y_folds[j] for j in range(k) if j != i])
        x_test = x_folds[i]
        y_test = y_folds[i]

        scaler = StandardScaler()
        scaler = scaler.fit(x_tr)
        x_tr = scaler.transform(x_tr)
        x_test = scaler.transform(x_test)

        print("sklearn")
        logreg = LogisticRegression(multi_class='multinomial', max_iter=1000)
        logreg.fit(x_tr, y_tr)
        p = logreg.predict_proba(x_test)
        loss = log_loss(y_test, p)
        print(loss.mean())
        losses[0][id_splits[i]] = log_loss(y_test, p)
        accuracies[0][id_splits[i]] = (y_test == np.argmax(p, axis=1))
        preds[0][id_splits[i], :] = p

        print("my")
        model = ANNClassification(units=[20], lambda_=0.3)
        model.fit(x_tr, y_tr)
        p = model.predict(x_test)
        loss = log_loss(y_test, p)
        print(loss.mean())
        losses[1][id_splits[i]] = log_loss(y_test, p)
        accuracies[1][id_splits[i]] = (y_test == np.argmax(p, axis=1))
        preds[1][id_splits[i], :] = p

        models.append(model)

    imena = ["MultinomialRegression", "ANN"]
    for i in range(n_poskusov):
        print("")
        print(imena[i])
        print(f"Loss {i}: {losses[i].mean()} +/- {losses[i].std() / np.sqrt(len(losses[i]))}")
        print(f"Accuracy {i}: {accuracies[i].mean()} +/- {accuracies[i].std() / np.sqrt(len(accuracies[i]))}")



if __name__ == "__main__":
    print("\n")
    numerical_gradient()
    print("\n")
    housing2r()
    print("\n")
    housing3()
    print("\n")
    final_CV()
    print("\n")
    print("Creating final predictions")
    create_final_predictions()

    




