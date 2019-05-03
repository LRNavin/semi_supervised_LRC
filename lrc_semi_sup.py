import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_moons, make_circles, make_checkerboard, make_blobs, make_biclusters, make_classification, make_regression

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


np.random.seed()


def shuffle_ordered_arrays(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def prepare_labeled_unlabeled(X, y, num_labeled, num_unlabeled):
    perclass_labelled   = num_labeled/2;
    perclass_unlabelled = num_unlabeled/2;

    # 11 - Dimensional data
    class_positive = X[:4000, :]
    class_negative = X[4000:-1, :]

    # class label
    label_positive = y[:4000]
    label_negative = y[4000:-1]

    class_positive, label_positive = shuffle_ordered_arrays(class_positive, label_positive)
    class_negative, label_negative = shuffle_ordered_arrays(class_negative, label_negative)

    X_label = np.concatenate( (class_positive[:perclass_labelled] ,
                             class_negative[:perclass_labelled]), axis=0)
    X_un_label = np.concatenate( (class_positive[num_labeled:num_labeled + perclass_unlabelled] ,
                                class_negative[num_labeled:num_labeled + perclass_unlabelled]), axis=0)

    y_label = np.concatenate( (label_positive[:perclass_labelled] ,
                             label_negative[:perclass_labelled]), axis=0)
    y_un_label = np.concatenate( (label_positive[num_labeled:num_labeled + perclass_unlabelled] ,
                                label_negative[num_labeled:num_labeled + perclass_unlabelled]), axis=0)

    return (
        X_label,
        X_un_label,
        y_label,
        y_un_label
    )

def get_top_fit_selftrain(clf, X_u, top_per):

    y_prob = clf.predict_proba(X_u)
    y_prob_max = np.max(y_prob, axis=1)

    prob_indices = np.argpartition(y_prob_max, -top_per)[-top_per:]

    y_most_prob = np.argmax(y_prob[prob_indices], axis=1)
    X_most_prob = X_u[prob_indices]

    return X_most_prob, y_most_prob


def self_training(clf, X_l, y_l, X_u):

    # Initial fit on labeled data
    clf.fit(X_l, y_l)

    # Predict Unlabeled data and create new Decision Boundary on new data (labelled + unlabelled)
    if X_u.shape[0] > 1:
        y_u_pred =  get_label_predict(clf, X_u) #[1 if  i >= 0.5 else 0 for i in clf.predict(X_u)]
        #
        # X_most_prob, y_most_prob = get_top_fit_selftrain(clf, X_u, 5)

        # Select only the best 10 % fits
        X_l = np.concatenate((X_l, X_u), axis=0)
        y_l = np.concatenate((y_l, y_u_pred), axis=0)


        clf.fit(X_l, y_l)

    return clf


def self_training_v2(clf, X_l, y_l, X_u, top_fit=5):
        """Self-training of the classifier clf, by predicting labels
        for X_u, and retraining with the top_k most probable predictions
        """
        # Initial fit on labeled data
        clf.fit(X_l, y_l)

        while X_u.shape[0] >= top_fit:
            # Predict probabilities for classes
            # and select N most probable predictions

            pred = np.dot(X_u, clf.coef_) + clf.intercept_  # clf.predict(X)

            y_prob = np.absolute(pred)

            prob_indices = np.argpartition(y_prob, -top_fit)[:top_fit]

            y_prob = np.array([1 if  i >= 0.5 else 0 for i in y_prob])

            y_best = y_prob[prob_indices]
            X_best = X_u[prob_indices]

            # Add these to labeled data
            y_l = np.concatenate((y_l, y_best), axis=0)
            X_l = np.concatenate((X_l, X_best), axis=0)

            # Remove from unlabeled data
            X_u = np.delete(X_u, prob_indices, axis=0)

            # Train classifier with predicted labels
            clf.fit(X_l, y_l)

        # In case num_unlabeled does not divide nicely with top_k
        last_data_points = num_unlabeled % top_fit

        if last_data_points and X_u.shape[0] > 1:
            y_u_pred =  get_label_predict(clf, X_u) #[1 if i >= 0.5 else 0 for i in clf.predict(X_u)]

            y_l = np.concatenate((y_l, y_u_pred), axis=0)
            X_l = np.concatenate((X_l, X_u), axis=0)

            clf.fit(X_l, y_l)

        return clf


def propagate_labels(X_u, y_u, X_l, y_l, num_unlabeled):
    # unlabeled samples are represented by -1 in labelprop
    y_u_placeholder = np.zeros(num_unlabeled) - 1

    X_train_prop = np.concatenate((X_l, X_u), axis=0)
    y_train_prop = np.concatenate((y_l, y_u_placeholder), axis=0)

    prop = LabelPropagation()
    prop.fit(X_train_prop, y_train_prop)

    y_train_lrc = prop.transduction_

    X_train_lrc = np.concatenate((X_l, X_u), axis=0)

    return X_train_lrc, y_train_lrc


def spread_labels(X_u, y_u, X_l, y_l, num_unlabeled):
    # unlabeled samples are represented by -1 in labelprop
    y_u_placeholder = np.zeros(num_unlabeled) - 1

    X_train_prop = np.concatenate((X_l, X_u), axis=0)
    y_train_prop = np.concatenate((y_l, y_u_placeholder), axis=0)

    prop = LabelSpreading()
    prop.fit(X_train_prop, y_train_prop)

    y_train_lrc = prop.transduction_

    X_train_lrc = np.concatenate((X_l, X_u), axis=0)

    return X_train_lrc, y_train_lrc

def split_classwise_train_test(X, y, percent):

    # 11 - Dimensional data
    class_positive = X[:5000, :]
    class_negative = X[5000:-1, :]

    # class label
    label_positive = y[:5000]
    label_negative = y[5000:-1]

    X_train1, X_test1, y_train1, y_test1 = train_test_split(class_positive, label_positive, test_size=percent)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(class_negative, label_negative, test_size=percent)

    return np.concatenate((X_train1, X_train2), axis=0) ,\
           np.concatenate((X_test1, X_test2), axis=0),\
           np.concatenate((y_train1, y_train2), axis=0),\
           np.concatenate((y_test1, y_test2), axis=0)


def get_label_predict(clf, X):

    model_pred = np.dot(X, clf.coef_) + clf.intercept_ #clf.predict(X)
    label=[1 if  i >= 0.5 else 0 for i in model_pred]

    return label


#Data selection ----------- 1 - Two Gauss , 2 - Cresent Moon, 3 - Classification Data
dataset = ["Two Gauss", "Cresent Moon Data", "Classification Data", "Regression Data", "Gauss Artif", "Blobs", "Same Covar"]
data_choice = 1
if data_choice == 1:
    gauss_data = pd.read_csv('twoGaussians.csv', header=None)

    # Get data in a format that fits sklearn
    gauss_data[11] = pd.Categorical(gauss_data[11])
    gauss_data[11] = gauss_data[11].cat.codes
    X_raw = gauss_data.values

    y = X_raw[:, -1]
    X = X_raw[:, :-1]

    # X = X[:, :2]
    #
    # # Check Gen Data
    # plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
    # plt.show()

elif data_choice == 2:
    X, y = make_moons(10000, shuffle=False, noise=0.1)
    X = np.flip(X, axis=0)
    y = np.flip(y, axis=0)

    #Check Gen Data
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
    plt.show()
elif data_choice == 3:
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X *= 2 + rng.uniform(size=X.shape)

    # Check Gen Data
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
    plt.show()
elif data_choice == 4:
    X, y = make_regression(10000, 2)
    y = [1 if i >= 0 else 0 for i in y]
    X = np.flip(X, axis=0)
    y = np.flip(y, axis=0)

    # Check Gen Data
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
    plt.show()
elif data_choice == 5:
    mean   = 1
    std_p  = 2
    std_n  = 3
    p = np.random.normal(mean, std_p, [5000,2])
    n = np.random.normal(mean+6.0, std_n, [5000,2])

    l_p = np.zeros(5000) + 1
    l_n = np.zeros(5000)

    X = np.concatenate((p, n), axis=0)
    y = np.concatenate((l_p, l_n), axis=0)

    # Check Gen Data
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.1)
    plt.show()
elif data_choice == 6:
    X, y = make_circles(n_samples=10000, noise=0.5)
    y = [1 if i >= 0 else 0 for i in y]
    X = np.flip(X, axis=0)
    y = np.flip(y, axis=0)

    # Check Gen Data
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
    plt.show()
elif data_choice == 7:
    mean1 = np.array([0, 0])
    mean2 = np.array([10, 0])

    cov = np.array([[2, 0], [0, 2]])

    class1 = np.random.multivariate_normal(mean1, cov, 5000)
    class2 = np.random.multivariate_normal(mean2, cov, 5000)

    l_p = np.zeros(5000) + 1
    l_n = np.zeros(5000)

    X = np.concatenate((class1, class2), axis=0)
    y = np.concatenate((l_p, l_n), axis=0)

    # Check Gen Data
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
    plt.show()
else:
    X, y = np.random.rand(10000, 2), None
    X = np.flip(X, axis=0)
    y = np.flip(y, axis=0)

    # Check Gen Data
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)
    plt.show()

#Find Mean and STD of dataset --
class_p = X[:5000, :]
class_n = X[5000:-1, :]

mean_p = np.mean(class_p)
mean_n = np.mean(class_n)
print ("------- Mean of +ve class - {} and Mean of -ve class - {}".format(mean_p, mean_n))

std_p = np.std(class_p)
std_n = np.std(class_n)
print ("------- STD of +ve class - {} and STD of -ve class - {}".format(std_p, std_n))


# Reserve 20% of data for testing classifiers
X_train, X_test, y_train, y_test = split_classwise_train_test(X, y, percent=.2)

# Select data for supervised and unsupervised training
num_labeled = 16
nums_unlabeled = [0, 8, 16, 32, 64, 128, 256, 512]

N_iterations = 500  #average error rates over (Curve Smoothening)

# Allocate array to hold error_rates from all classifiers
error_rates = np.ndarray(shape=(N_iterations, len(nums_unlabeled), 3))
sq_loss = np.ndarray(shape=(N_iterations, len(nums_unlabeled), 3))

# ######################## --Supervised Learning-- ########################
print("===\tNow Supervised training\t===\n")
accuracy_sum_iter = 0
for i, num_unlabeled in enumerate(nums_unlabeled):

    for t in range(N_iterations):
        # Set up data for Supervised Learning
        X_l, X_u, y_l, y_u = prepare_labeled_unlabeled(
            X_train, y_train, num_labeled+num_unlabeled, num_unlabeled
        )
        # Train a model supervised on all data
        clf = LinearRegression()
        clf.fit(X_l, y_l)
        y_pred = get_label_predict(clf, X_test) #[1 if i >= 0.5 else 0 for i in clf.predict(X_test)]
        accuracy_sum_iter = accuracy_sum_iter + (1 - accuracy_score(y_pred=y_pred, y_true=y_test))

print("Average Accuracy from training LRC on 8 per class labelled data (Supervised Leanring): {:.4f}\n".format(
    accuracy_sum_iter/(N_iterations*8)
    ))

for i, num_unlabeled in enumerate(nums_unlabeled):
    print("****\t Training on {} labelled data, {} unlabelled data\t****\n".format(num_labeled, num_unlabeled))

    for j in range(N_iterations):
        if (j + 1) % 10 == 0:
            print('Iteration: {}'.format(j + 1))

        # ######################## --Semi-supervised 1-- ########################
        # 1 - Train on labeled data first,
        # 2 - predict labels for unlabeled data,
        # 3 - and train classifier further with these predicted labels

        # Set up data for Same Decision Boundary Train
        X_l, X_u, y_l, y_u = prepare_labeled_unlabeled(
            X_train, y_train, num_labeled, num_unlabeled
        )

        clf = LinearRegression()
        clf = self_training(clf, X_l, y_l, X_u)

        # Do predictions for test set and evaluate
        y_pred =  get_label_predict(clf, X_test) #[1 if buff >= 0.5 else 0 for buff in clf.predict(X_test)]
        y_error = mean_squared_error(y_test, clf.predict(X_test))
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        error_rates[j, i, 0] = 1 - accuracy
        sq_loss[j, i, 0] = y_error


        # ######################## --Semi-supervised 2-- ########################
        # Find labels for unlabeled data with label propagation

        # Set up data for LabelPropagation
        X_l, X_u, y_l, y_u = prepare_labeled_unlabeled(
            X_train, y_train, num_labeled, num_unlabeled
        )

        if num_unlabeled != 0:
            # Semi-Supervised learning using Label Propogation
            X_train_lrc, y_train_lrc = propagate_labels(X_u, y_u, X_l, y_l, num_unlabeled)

            clf = LinearRegression()
            clf.fit(X_train_lrc, y_train_lrc)

            # Do predictions for test set and evaluate
            y_pred =  get_label_predict(clf, X_test) #[1 if  buff >= 0.5 else 0 for buff in clf.predict(X_test)]
            y_error = mean_squared_error(y_test, clf.predict(X_test))
            accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

        else:
            #0 - unlabelled data
            clf = LinearRegression()
            clf.fit(X_l, y_l)

            # Do predictions for test set and evaluate
            y_pred =  get_label_predict(clf, X_test) #[1 if buff >= 0.5 else 0 for buff in clf.predict(X_test)]
            y_error = mean_squared_error(y_test,
                                         clf.predict(X_test)) 
            accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

        error_rates[j, i, 1] = 1 - accuracy
        sq_loss[j, i, 1] = y_error


        # ######################## --Semi-supervised 3-- ########################
        # Find labels for unlabeled data with label propagation

        # Set up data for LabelSpread
        # X_l, X_u, y_l, y_u = prepare_labeled_unlabeled(
        #     X_train, y_train, num_labeled, num_unlabeled
        # )
        #
        # if num_unlabeled != 0:
        #     # Semi-Supervised learning using Label Spreading
        #     X_train_lrc, y_train_lrc = spread_labels(X_u, y_u, X_l, y_l, num_unlabeled)
        #
        #     clf = LinearRegression()
        #     clf.fit(X_train_lrc, y_train_lrc)
        #
        #     # Do predictions for test set and evaluate
        #     y_pred =  get_label_predict(clf, X_test) # [1 if buff >= 0.5 else 0 for buff in clf.predict(X_test)]
        #     y_error = mean_squared_error(y_test, clf.predict(X_test))
        #     accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        #
        # else:
        #     # 0 - unlabelled data
        #     clf = LinearRegression()
        #     clf.fit(X_l, y_l)
        #
        #     # Do predictions for test set and evaluate
        #     y_pred =  get_label_predict(clf, X_test) #[1 if buff >= 0.5 else 0 for buff in clf.predict(X_test)]
        #     y_error = mean_squared_error(y_test,
        #                                  clf.predict(X_test))
        #     accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        #
        # error_rates[j, i, 2] = 1 - accuracy
        # sq_loss[j, i, 2] = y_error

    print('\n')

avg_error_rates = np.mean(error_rates, axis=0)
std_error_rates = np.std(error_rates, axis=0)

avg_sq_loss = np.mean(sq_loss, axis=0)
std_sq_loss = np.std(sq_loss, axis=0)

print(avg_error_rates)
colors = ['blue', 'red', 'green']

fig, ax1 = plt.subplots(nrows=1, sharex=True)

ax1.errorbar(x=nums_unlabeled, y=avg_error_rates[:, 0], yerr=std_error_rates[:, 0])
ax1.errorbar(x=nums_unlabeled, y=avg_error_rates[:, 1], yerr=std_error_rates[:, 1])
#ax1.errorbar(x=nums_unlabeled, y=avg_error_rates[:, 2], yerr=std_error_rates[:, 2])

#ax1.plot(nums_unlabeled, avg_error_rates[:, 0], color=colors[1])
#ax1.plot(nums_unlabeled, avg_error_rates[:, 1], color=colors[2])
#ax1.plot(nums_unlabeled, avg_error_rates[:, 2], color=colors[2])
ax1.legend(['Self Training', 'Label Propagation'])#, 'LabelSpreading'])
ax1.set_title(dataset[data_choice-1])
ax1.set_xlabel('\# of unlabeled samples')
ax1.set_ylabel('True Error')
ax1.grid()
plt.show()

fig, ax2 = plt.subplots(nrows=1, sharex=True)

ax2.errorbar(x=nums_unlabeled, y=avg_sq_loss[:, 0], yerr=std_sq_loss[:, 0])
ax2.errorbar(x=nums_unlabeled, y=avg_sq_loss[:, 1], yerr=std_sq_loss[:, 1])
#ax2.errorbar(x=nums_unlabeled, y=avg_sq_loss[:, 2], yerr=std_sq_loss[:, 2])

#ax2.plot(nums_unlabeled, avg_sq_loss[:, 0], color=colors[1])
#ax2.plot(nums_unlabeled, avg_sq_loss[:, 1], color=colors[2])
#ax2.plot(nums_unlabeled, avg_sq_loss[:, 2], color=colors[2])
ax2.legend(['Self Training', 'Label Propagation'])#, 'LabelSpreading'])
ax2.set_title(dataset[data_choice-1])
ax2.set_xlabel('\# of unlabeled samples')
ax2.set_ylabel('Mean Squared Loss')
ax2.grid()

#plt.tight_layout()

#lt.savefig('figures/learning_curves.eps', dpi=300)
plt.show()


