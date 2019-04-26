import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Set up matplotlib to get pretty text
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


np.random.seed()


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def prepare_labeled_unlabeled(X, y, num_labeled, num_unlabeled):
    return (
        X[:num_labeled],
        X[num_labeled:num_labeled + num_unlabeled],
        y[:num_labeled],
        y[num_labeled:num_labeled + num_unlabeled]
    )


def self_training(clf, X_l, y_l, X_u, top_k=5):
    """Self-training of the classifier clf, by predicting labels
    for X_u, and retraining with the top_k most probable predictions
    """
    # Initial fit on labeled data
    clf.fit(X_l, y_l)

    while X_u.shape[0] >= top_k:
        # Predict probabilities for classes
        # and select N most probable predictions
        y_prob = clf.predict_proba(X_u)
        y_prob_max = np.max(y_prob, axis=1)

        prob_indices = np.argpartition(y_prob_max, -top_k)[-top_k:]

        y_most_prob = np.argmax(y_prob[prob_indices], axis=1)
        X_most_prob = X_u[prob_indices]

        # Add these to labeled data
        y_l = np.concatenate((y_l, y_most_prob), axis=0)
        X_l = np.concatenate((X_l, X_most_prob), axis=0)

        # Remove from unlabeled data
        X_u = np.delete(X_u, prob_indices, axis=0)

        # Train classifier with predicted labels
        clf.fit(X_l,  y_l)

    # In case num_unlabeled does not divide nicely with top_k
    last_data_points = num_unlabeled % top_k

    if last_data_points and X_u.shape[0] > 1:
        y_u_pred = clf.predict(X_u)
        clf.fit(X_u, y_u_pred)

    return clf


def propagate_labels(X_u, y_u, X_l, num_unlabeled):
    # unlabeled samples are represented by -1 in labelprop
    y_u_placeholder = np.zeros(num_unlabeled) - 1

    X_train_prop = np.concatenate((X_l, X_u), axis=0)
    y_train_prop = np.concatenate((y_l, y_u_placeholder), axis=0)

    prop = LabelPropagation(gamma=15)
    prop.fit(X_train_prop, y_train_prop)

    y_train_lrc = prop.transduction_

    X_train_lrc = np.concatenate((X_l, X_u), axis=0)

    return X_train_lrc, y_train_lrc


magic = pd.read_csv('twoGaussians.csv', header=None)

# Get data in a format that fits sklearn
magic[11] = pd.Categorical(magic[11])
magic[11] = magic[11].cat.codes

# Get data as arrays, shuffle, and separate features from labels
X_raw = magic.values

np.random.shuffle(X_raw)

y = X_raw[:, -1]
X = X_raw[:, :-1]

# Normalize X to get unit standard deviation
col_std = np.std(X, axis=1)
for j in range(X.shape[1]):
    X[:, j] = X[:, j] / col_std[j]


# Train a model supervised on all data
print("===\tNow training on all data with labels\t===\n")
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)

y_pred = clf.predict(X)

accuracy_all_data = accuracy_score(y_pred=y_pred, y_true=y)
print("Accuracy from training lda on all data: {:.4f}\n".format(
        accuracy_all_data
    ))

# Reserve 20% of data for testing classifiers
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# Select data for supervised and unsupervised training
num_labeled = 16
nums_unlabeled = [0, 10, 20, 40, 80, 160, 320, 640]

N_iterations = 10  # Number of iterations to average error rates over

# Allocate array to hold error_rates from all classifiers
error_rates = np.ndarray(shape=(N_iterations, len(nums_unlabeled), 2))
log_probs = np.ndarray(shape=(N_iterations, len(nums_unlabeled), 2))


for i, num_unlabeled in enumerate(nums_unlabeled):
    print("===\tNow training on {} datapoints with labels, {} without\t===\n".format(num_labeled, num_unlabeled))

    for j in range(N_iterations):
        if (j + 1) % 10 == 0:
            print('Iteration: {}'.format(j + 1))

        # Shuffle training data for each iteration
        X_train, y_train = unison_shuffled_copies(X_train, y_train)

        X_l, X_u, y_l, y_u = prepare_labeled_unlabeled(
            X_train, y_train, num_labeled, num_unlabeled
        )

        # Semi-supervised 1
        # Train on labeled data first, predict labels for unlabeled data,
        # and train classifier further with these predicted labels
        clf = LinearDiscriminantAnalysis()
        clf = self_training(clf, X_l, y_l, X_u)

        # Do predictions for test set and evaluate
        y_pred = clf.predict(X_test)
        y_probs = np.sum(np.max(clf.predict_log_proba(X_test), axis=1))
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        error_rates[j, i, 0] = 1 - accuracy
        log_probs[j, i, 0] = y_probs

        # ## Semi-supervised 2
        # Find labels for unlabeled data with label propagation

        # Set up data for LabelPropagation
        X_l, X_u, y_l, y_u = prepare_labeled_unlabeled(
            X_train, y_train, num_labeled, num_unlabeled
        )

        if num_unlabeled == 0:
            # First iteration
            error_rates[j, i, 1] = 1 - accuracy
            log_probs[j, i, 1] = y_probs
            continue

        X_train_lrc, y_train_lrc = propagate_labels(X_u, y_u, X_l, num_unlabeled)

        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train_lrc, y_train_lrc)

        # Do predictions for test set and evaluate
        y_pred = clf.predict(X_test)
        y_probs = np.sum(np.max(clf.predict_log_proba(X_test), axis=1))
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        error_rates[j, i, 1] = 1 - accuracy
        log_probs[j, i, 1] = y_probs
    print('\n')

avg_error_rates = np.mean(error_rates, axis=0)
std_error_rates = np.std(error_rates, axis=0)

avg_log_probs = np.mean(log_probs, axis=0)
std_log_probs = np.std(log_probs, axis=0)

print(avg_error_rates)
print(std_error_rates)
colors = ['blue', 'red']

fig, [ax1, ax2] = plt.subplots(nrows=2, sharex=True)
ax1.plot(nums_unlabeled, avg_error_rates[:, 0], color=colors[0])
ax1.plot(nums_unlabeled, avg_error_rates[:, 1], color=colors[1])
ax1.legend(['Self-training', 'Labelpropagation'])

# ax1.set_xlabel('\# unlabeled data points')
ax1.set_ylabel('Error rate')
ax1.grid()

ax2.plot(nums_unlabeled, avg_log_probs[:, 0], color=colors[0])
ax2.plot(nums_unlabeled, avg_log_probs[:, 1], color=colors[1])
ax2.legend(['Self-training', 'Labelpropagation'])

ax2.set_xlabel('\# unlabeled data points')
ax2.set_ylabel('Log-Likelihood')
ax2.grid()

plt.tight_layout()

plt.savefig('figures/learning_curves.eps', dpi=300)
plt.show()