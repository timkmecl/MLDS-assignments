import numpy as np
from scipy.optimize import fmin_l_bfgs_b




def log_score(y, p):
    samples_range = np.arange(len(y))
    return np.mean(np.log(p[samples_range, y]))



MBOG_TRAIN = 200
def multinomial_bad_ordinal_good(n, rand):
    X = np.zeros((n, 2))
    s = np.zeros(n)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        X[i, 0] = rand.gauss(0,3)
        X[i, 1] = rand.gauss(0,1.5)
        s[i] = X[i, 0] + 0.3*X[i, 1] + rand.gauss(0, 0.05)
        
    for j in range(-5, 5):
        select = (s > j) & (s <= j+1)
        y[select] = j+6
    
    y[s>j+1] = j+7
    return X, y






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
    




class OrdinalLogReg:
    def __init__(self):
        pass
    
    def cdf(x):
        return 1 / (1 + np.exp(-x))
    
    def build(self, X, y, intercept=True, min_delta=1e-8):
        classes_count = int(np.max(y) + 1)
        feats_count = X.shape[1]

        if intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
            feats_count += 1
        
        samples_range = np.arange(X.shape[0])
        def loss(theta):
            betas = theta[:feats_count]
            deltas = np.hstack((0, theta[feats_count:]))
            u = X @ betas

            breaks = np.cumsum(deltas)
            left_breaks = np.hstack((-np.inf, breaks))
            right_breaks = np.hstack((breaks, np.inf))

            u = u[:, np.newaxis]
            probs = OrdinalLogReg.cdf(right_breaks - u) - OrdinalLogReg.cdf(left_breaks - u)

            return -np.sum(np.log(probs[samples_range, y]))
        
        theta_0 = np.zeros((feats_count + classes_count - 2))
        theta_0[feats_count:] = min_delta

        bounds = [(None, None)] * feats_count + [(min_delta, None)] * (classes_count - 2)
        theta_opt, _, _ = fmin_l_bfgs_b(loss, theta_0, bounds=bounds, approx_grad=True, maxiter=10000)
        
        betas = theta_opt[:feats_count]
        deltas = np.hstack((0, theta_opt[feats_count:]))

        return OrdinalModel(classes_count, feats_count, betas, deltas, intercept)


class OrdinalModel:
    def __init__(self, classes_count, feats_count, betas, deltas, intercept):
        self.classes_count = classes_count
        self.feats_count = feats_count
        self.betas = betas
        self.deltas = deltas
        self.intercept = intercept
    
    def predict(self, X, epsilon=0):
        if self.intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        u = X @ self.betas
        breaks = np.cumsum(self.deltas)
        left_breaks = np.hstack((-np.inf, breaks))
        right_breaks = np.hstack((breaks, np.inf))
    
        u = u[:, np.newaxis]
        probs = OrdinalLogReg.cdf(right_breaks - u) - OrdinalLogReg.cdf(left_breaks - u)
        
        probs = epsilon + probs
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs
    




def compare_implementations():
    # generate data by genearting random points inside unit square, classes are based on the quadrants
		np.random.seed(0)
		X = np.random.rand(1000, 2)*2 - 1
		X1 = X + np.random.normal(0, 0.1, (1000, 2))
		y = np.zeros(1000, dtype=int)
		y[np.logical_and(X1[:, 0] > 0, X1[:, 1] > 0)] = 3
		y[np.logical_and(X1[:, 0] > 0, X1[:, 1] < 0)] = 0
		y[np.logical_and(X1[:, 0] < 0, X1[:, 1] > 0)] = 1
		y[np.logical_and(X1[:, 0] < 0, X1[:, 1] < 0)] = 2


		print("My implementation")
		model = MultinomialLogReg()
		c = model.build(X, y, intercept=False)
		prob = c.predict(X)
		print("Predictions:")
		print(prob.round(4))
		print("Betas:")
		print(c.betas.round(3))


		print()
		print("Sklearn implementation")
		from sklearn.linear_model import LogisticRegression
		model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='none', fit_intercept=False)
		model.fit(X, y)
		print("Predictions:")
		print(model.predict_proba(X).round(4))
		print("Transformed betas:")
		print((model.coef_.T - model.coef_.T[:, -1][:, np.newaxis]).round(3))
        


def difference_ordinal_multinomial():
		import random
		# test the loss difference between ordinal and multinomial 100 times
		X, y = multinomial_bad_ordinal_good(MBOG_TRAIN, random.Random(1013))
		X1, y1 = multinomial_bad_ordinal_good(1000, random.Random(1014))

		diffs = []
		for i in range(100):
				print(i, "/ 100")
				X, y = multinomial_bad_ordinal_good(MBOG_TRAIN, random.Random(1011+2*i))
				X1, y1 = multinomial_bad_ordinal_good(1000, random.Random(1011+2*i+1))

				model = MultinomialLogReg()
				c = model.build(X, y, intercept=True)
				prob = c.predict(X1, epsilon=1e-8)
				score_m = log_score(y1, prob)

				model = OrdinalLogReg()
				c = model.build(X, y, intercept=True)
				prob = c.predict(X1, epsilon=1e-8)
				score_o = log_score(y1, prob)

				diffs.append(score_o - score_m)


		# plot the differences
		import matplotlib.pyplot as plt
		import seaborn as sns
		sns.set()
		plt.figure(figsize=(8, 6))
		sns.histplot(diffs, kde=True, bins=10)
		plt.xlabel("Difference in log-likelihood")
		plt.ylabel("Count")
		# plt.show()
		plt.savefig('differences.svg')

		print("Mean difference:", np.mean(diffs).round(2))
		print("Stdev of differences:", np.std(diffs).round(2))
		print("Percentage of positive differences:", 100* np.mean(np.array(diffs) > 0))





if __name__ == "__main__":
		print("Comparison of my implementation with sklearn's one:")
		compare_implementations()
            
		print("\nDifference between ordinal and multinomial:")
		difference_ordinal_multinomial()

		# ==== Data analysis ======
		print("Data analysis:")
		import pandas as pd
		import matplotlib.pyplot as plt
		
		data = pd.read_csv("dataset.csv", sep=";")
		y = data["ShotType"]
		X = data.drop("ShotType", axis=1)
		X = pd.get_dummies(X, columns=["Competition", "PlayerType", "Transition", "TwoLegged", "Movement"])

		categories = y.unique()
		categories = np.hstack((categories[categories != "other"], categories[categories == "other"]))
		y = y.map({categories[i]: i for i in range(len(categories))})

		from sklearn.preprocessing import StandardScaler
		# function to bootstrap the data
		def bootstrap_data(X, y):
				n = len(y)
				indices = np.random.choice(np.arange(n), n, replace=True)
				# oob
				X1, y1 = X.iloc[indices], y.iloc[indices]
				X2, y2 = X.iloc[~np.isin(np.arange(n), indices)], y.iloc[~np.isin(np.arange(n), indices)]
				# standardize
				scaler = StandardScaler()
				scaler = scaler.fit(X1)
				X1 = scaler.transform(X1)
				X2 = scaler.transform(X2)
				return X1, y1, X2, y2


		accuracies = []
		scores = []
		betas = []

		# bootstrap the data 100 times
		from tqdm import tqdm
		np.random.seed(1011)
		for i in tqdm(range(100)):
				X1, y1, X2, y2 = bootstrap_data(X, y)
				model = MultinomialLogReg()
				c = model.build(X1, y1, intercept=False)
				prob = c.predict(X2, epsilon=1e-8)
				score = log_score(y2, prob)
				scores.append(score)
				accuracies.append(np.mean(np.argmax(prob, axis=1) == y2))
				betas.append(c.betas)
				

		print("Mean accuracy:", np.mean(accuracies).round(2), "+-", (np.std(accuracies)).round(2))
		print("Mean log-score:", np.mean(scores).round(2), "+-", (np.std(scores)).round(2))

		betas2 = []
		for b in betas:
				b2 = np.hstack((b, np.zeros((b.shape[0], 1))))
				b2 = b2 - b2.mean(axis=1, keepdims=True)
				betas2.append(b2)

		betas2 = np.array(betas2)
		B2 = betas2.mean(axis=0)


		# confidence intervals
		alpha = 0.05
		lower = np.percentile(betas2, 100*alpha/2, axis=0)
		upper = np.percentile(betas2, 100*(1-alpha/2), axis=0)

		B1 = np.empty_like(B2, dtype=object)
		print()
		for i in range(B2.shape[0]):
				print()
				for j in range(B2.shape[1]):
						B1[i, j] = f"{B2[i, j].round(1)} ({lower[i, j].round(2)} | {upper[i, j].round(2)})"
						print(B1[i, j], end="\t")


		print()
		print("Most prominent features:")
		# for each category (column) get the rows with highest absolute values
		for j in range(B2.shape[1]):
				print()
				print("\t", categories[j])
				for i in np.argsort(np.abs(B2[:, j]))[-5:]:
						print(B1[i, j], "\t", X.columns[i])

		# display the same in a plot as subplots
		fig, ax = plt.subplots(1, B2.shape[1], figsize=(15, 5))

		for j in range(B2.shape[1]):
				# plt.figure(figsize=(3, 5))
				plt.sca(ax[j])
				most_prominent = np.argsort(np.abs(B2[:, j]))[-8:]
				plt.errorbar(np.arange(len(most_prominent)), B2[most_prominent, j], yerr=[B2[most_prominent, j] - lower[most_prominent, j], upper[most_prominent, j] - B2[most_prominent, j]], fmt='.')
				plt.xticks(np.arange(len(most_prominent)), [X.columns[i] for i in most_prominent], rotation=90)
				if j > 0:
						plt.ylim(-2.5, 3.6)
				
				plt.title(categories[j])
		plt.show()


		# heatmap of all
		betas_df = pd.DataFrame(B2.T, columns=X.columns)
		import seaborn as sns
		sns.set()
		plt.figure(figsize=(7.5, 4.5))
		sns.heatmap(betas_df, annot=True, cbar=False, fmt=".1f", cmap="coolwarm", center=0, vmin=-5, vmax=5)
		plt.xticks(rotation=90)
		plt.yticks(ticks=np.arange(len(categories))+0.5, labels=categories, rotation=0)
		plt.title("Coefficients for the multinomial logistic regression model")
		plt.tight_layout()
		plt.show()


		# plot correlations between features
		plt.figure(figsize=(7.5, 7.5))

		sns.heatmap(X.corr().round(1), annot=True, cmap='coolwarm', cbar=False)
		plt.title("Correlation matrix of the features")
		plt.tight_layout()
		plt.show()