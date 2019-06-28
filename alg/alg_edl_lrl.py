import scipy.io as sio
from sklearn.cluster import KMeans
from scipy.optimize import fmin_l_bfgs_b
from alg.evaluation_metrics import *
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

lambda1 = 1e-7
lambda2 = 1e-7
m = 5


# read data set .mat
def read_mat(url):
    data = sio.loadmat(url)
    return data


# output model
def predict_func(x, m_theta, f_dim, l_dim):
    m_theta = m_theta.reshape(f_dim, l_dim)
    numerator = np.exp(np.dot(x, m_theta))
    denominator = np.sum(np.exp(np.dot(x, m_theta)), 1).reshape(-1, 1) + 0.00001
    return numerator / denominator


# w: q*L, x: n*q, d_: n*L, z: a list which includes m arrays
def obj_func1(w, x, d_, z, f_dim, l_dim):
    w = w.reshape(f_dim, l_dim)
    term1 = 0.5 * np.sum((predict_func(x, w, f_dim, l_dim) - d_) ** 2)
    term2 = np.linalg.norm(w, ord=2) ** 2
    term3 = 0.
    for i in range(m):
        term3 += np.linalg.norm(z[i], ord='nuc')
    loss1 = term1 + lambda1 * term2 + lambda2 * term3
    return loss1


# the objective function of sub-question
# x: a list which includes the result of clustering
# d_: a list which includes the result of clustering
# z: a list which includes m arrays
# Lambda: a list of Lagrange Multiplier
# rho: a list which includes m members
def obj_func2(w, x, d_, z, Lambda, rho, f_dim, l_dim):
    w = w.reshape(f_dim, l_dim)
    term1 = 0.
    term2 = 0.
    term3 = 0.
    for i in range(m):
        term1 += 0.5 * np.sum((predict_func(x[i], w, f_dim, l_dim) - d_[i]) ** 2)
        term2 += np.sum(Lambda[i] * (predict_func(x[i], w, f_dim, l_dim)-z[i]))
        term3 += (rho[i] / 2) * np.sum((predict_func(x[i], w, f_dim, l_dim)-z[i]) ** 2)
    # print(term1 + term2 + term3)
    loss2 = term1 + term2 + term3 + lambda1 * np.linalg.norm(w, ord=2) ** 2
    # print("sub_loss:", str(loss2))
    return loss2


# update w, the prime of parameter w
def fprime(w, x, d_, z, Lambda, rho, f_dim, l_dim):
    w = w.reshape(f_dim, l_dim)
    gradient = np.zeros_like(w)
    for i in range(m):
        modProb = np.exp(np.dot(x[i], w))
        sumProb = np.sum(modProb, 1)
        disProb = modProb / (sumProb.reshape(-1, 1) + 0.000001)
        gradient += np.dot(np.transpose(x[i]), (disProb - d_[i]) * (disProb - disProb*disProb))
        gradient += np.dot(np.transpose(x[i]), Lambda[i] * (disProb - disProb*disProb))
        gradient += rho[i] * np.dot(np.transpose(x[i]), (disProb - z[i]) * (disProb - disProb*disProb))
    gradient += 2 * lambda1 * w
    return gradient.ravel()


# update z
def update_z(w, x, z, Lambda, rho, f_dim, l_dim):
    w = w.reshape(f_dim, l_dim)
    z_new = []
    for i in range(m):
        u, sigma, vt = np.linalg.svd(predict_func(x[i], w, f_dim, l_dim)+Lambda[i]/rho[i])
        sigma_new = [s if s-(lambda2/rho[i]) > 0 else 0 for s in sigma]
        temp = np.diag(sigma_new)
        height, width = z[i].shape
        if len(sigma) < width:
            temp = np.c_[temp, np.zeros([len(sigma), width-len(sigma)])]
        if len(sigma) < height:
            temp = np.r_[temp, np.zeros([height-len(sigma), width])]
        z_new.append(np.dot(np.dot(u, temp), vt))
    return z_new


# update Lambda
def update_Lambda(w, x, z, Lambda, rho, f_dim, l_dim):
    w = w.reshape(f_dim, l_dim)
    Lambda_new = []
    for i in range(m):
        Lambda_new.append(Lambda[i] + rho[i]*(predict_func(x[i], w, f_dim, l_dim)-z[i]))
    return Lambda_new


if __name__ == "__main__":
    data1 = read_mat(r"../datasets/SJAFFE.mat")
    # data1 = read_mat(r"./datasets/Movie.mat")
    features = data1["features"]

    # normalization is needed
    # scaler = preprocessing.StandardScaler().fit(features)
    # features = scaler.transform(features)

    label_real = data1["labels"]
    features_dim = len(features[0])
    labels_dim = len(label_real[0])

    result1 = []
    result2 = []
    result3 = []
    result4 = []
    result5 = []
    result6 = []

    for t in range(10):
        x_train, x_test, y_train, y_test = train_test_split(features, label_real, test_size=0.2, random_state=2)
        # initialize
        w = np.random.rand(features_dim, labels_dim)    # update
        kmeans = KMeans(n_clusters=m).fit(y_train)
        kmeans_result = kmeans.predict(y_train)
        x_result = []
        d_result = []
        for i in range(m):
            x_result.append([])
            d_result.append([])
        for i in range(len(x_train)):   # 后期len(features)改为训练集大小
            x_result[kmeans_result[i]].append(list(features[i]))
            d_result[kmeans_result[i]].append(list(label_real[i]))
        z = []  # update
        for i in range(m):
            # z.append(np.ones_like(d_result[i]))
            z.append(np.zeros_like(d_result[i]))
            # z.append(np.ones_like(d_result[i]) / labels_dim)
        Lambda = []     # update
        for i in range(m):
            Lambda.append(np.zeros_like(d_result[i]))
        rho = np.ones(m) * (10 ** -6)     # parameter
        rho_max = 10 ** 6
        beta = 1.1  # increase factor
        # update step
        loss = obj_func1(w, features, label_real, z, features_dim, labels_dim)
        print(loss)
        for i in range(50):
            print("-" * 20)
            # print(obj_func2(w, x_result, d_result, z, Lambda, rho, features_dim, labels_dim))
            result = fmin_l_bfgs_b(obj_func2, w, fprime, args=(x_result, d_result, z, Lambda, rho, features_dim, labels_dim),
                                   pgtol=0.001, maxiter=10)
            w = result[0]
            z = update_z(w, x_result, z, Lambda, rho, features_dim, labels_dim)
            Lambda = update_Lambda(w, x_result, z, Lambda, rho, features_dim, labels_dim)
            loss_new = obj_func1(w, x_train, y_train, z, features_dim, labels_dim)
            if abs(loss - loss_new) < 10 ** -8 or loss_new > loss:
                break
            rho = np.min([rho[0]*beta, rho_max]) * np.ones(m)
            loss = loss_new
            print(loss)

        # predict the label distributions of test set
        pre_test = predict_func(x_test, w, features_dim, labels_dim)

        # add each result to a list
        result1.append(chebyshev(y_test, pre_test))
        print("No." + str(t) + ": " + str(chebyshev(y_test, pre_test)))
        result2.append(clark(y_test, pre_test))
        print("No." + str(t) + ": " + str(clark(y_test, pre_test)))
        result3.append(canberra(y_test, pre_test))
        print("No." + str(t) + ": " + str(canberra(y_test, pre_test)))
        result4.append(kl(y_test, pre_test))
        print("No." + str(t) + ": " + str(kl(y_test, pre_test)))
        result5.append(cosine(y_test, pre_test))
        print("No." + str(t) + ": " + str(cosine(y_test, pre_test)))
        result6.append(intersection(y_test, pre_test))
        print("No." + str(t) + ": " + str(intersection(y_test, pre_test)))

    print(result1)
    print(result2)
    print(result3)
    print(result4)
    print(result5)
    print(result6)

    print("chebyshev:", np.mean(result1), "+", np.std(result1))
    print("clark:", np.mean(result2), "+", np.std(result2))
    print("canberra:", np.mean(result3), "+", np.std(result3))
    print("kl:", np.mean(result4), "+", np.std(result4))
    print("cosine:", np.mean(result5), "+", np.std(result5))
    print("intersection:", np.mean(result6), "+", np.std(result6))








