import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE, Isomap
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import kurtosis
from ucimlrepo import fetch_ucirepo


def load_datasets():
    wholesale_customers_data = fetch_ucirepo(id=292)
    wholesale_customers = wholesale_customers_data.data.features

    car_evaluation_data = fetch_ucirepo(id=19)
    car_evaluation = pd.concat(
        [car_evaluation_data.data.features, car_evaluation_data.data.targets], axis=1
    )

    return wholesale_customers, car_evaluation


def preprocess_wholesale(wholesale_customers):
    wholesale_customers["Total Spend"] = wholesale_customers[
        ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
    ].sum(axis=1)
    wholesale_customers["Category"] = pd.qcut(
        wholesale_customers["Total Spend"], q=3, labels=["Low", "Medium", "High"]
    )

    label_encode_wholesale = LabelEncoder()
    wholesale_customers["Category"] = label_encode_wholesale.fit_transform(
        wholesale_customers["Category"]
    )

    X_wholesale = wholesale_customers.drop(columns=["Total Spend", "Category"])
    y_wholesale = wholesale_customers["Category"]

    scaler_wholesale = StandardScaler()
    X_wholesale_scaled = scaler_wholesale.fit_transform(X_wholesale)
    return X_wholesale_scaled, y_wholesale, wholesale_customers


def preprocess_car(car_evaluation):
    label_encode_car = LabelEncoder()
    for column in ["buying", "maint", "doors", "persons", "lug_boot", "safety"]:
        car_evaluation[column] = label_encode_car.fit_transform(car_evaluation[column])

    car_evaluation["class"] = label_encode_car.fit_transform(car_evaluation["class"])
    X_car = car_evaluation.drop(columns=["class"])
    y_car = car_evaluation["class"]

    scaler_car = StandardScaler()
    X_car_scaled = scaler_car.fit_transform(X_car)
    return X_car_scaled, y_car, car_evaluation, label_encode_car


def apply_clustering(X, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)

    em = GaussianMixture(n_components=n_clusters, random_state=42)
    em_labels = em.fit_predict(X)

    return kmeans_labels, em_labels


def apply_dimensionality_reduction(X, n_components=5):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    ica = FastICA(n_components=n_components, random_state=42)
    X_ica = ica.fit_transform(X)
    ica_kurtosis = kurtosis(X_ica, axis=0)

    rp = GaussianRandomProjection(n_components=n_components, random_state=42)
    X_rp = rp.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    isomap = Isomap(n_components=n_components)
    X_isomap = isomap.fit_transform(X)

    return X_pca, X_ica, X_rp, X_tsne, X_isomap, ica_kurtosis


def train_evaluate_nn_a1(X_train, X_test, y_train, y_test):
    param_grid_nn = {
        "hidden_layer_sizes": [(50,), (100,), (200,)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
        "max_iter": [10000],
        "solver": ["adam", "sgd"],
    }

    nn_grid_search = GridSearchCV(
        MLPClassifier(), param_grid_nn, cv=5, scoring="accuracy", n_jobs=-1
    )

    nn_grid_search.fit(X_train, y_train)
    best_nn = nn_grid_search.best_estimator_
    y_pred = best_nn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, zero_division=1, output_dict=True)

    return accuracy, cm, cr


def plot_clusters(X, labels, title, label_encoder=None):
    plt.figure(figsize=(10, 6))
    if label_encoder:
        labels = label_encoder.inverse_transform(labels)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="viridis")
    plt.title(title)
    plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_kurtosis(kurtosis_values, title):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(kurtosis_values) + 1), kurtosis_values, marker="o")
    plt.title(title)
    plt.xlabel("Independent Components")
    plt.ylabel("Kurtosis")
    plt.grid(True)
    plt.show()


wholesale_data, car_data = load_datasets()

X_wholesale, y_wholesale, wholesale_customers = preprocess_wholesale(wholesale_data)
X_car, y_car, car_evaluation, car_label_encoder = preprocess_car(car_data)

wholesale_kmeans_labels, wholesale_em_labels = apply_clustering(X_wholesale)
car_kmeans_labels, car_em_labels = apply_clustering(X_car)

wholesale_customers["kmeans_labels"] = wholesale_kmeans_labels
wholesale_customers["em_labels"] = wholesale_em_labels

car_evaluation["kmeans_labels"] = car_kmeans_labels
car_evaluation["em_labels"] = car_em_labels

X_wholesale_train, X_wholesale_test, y_wholesale_train, y_wholesale_test = (
    train_test_split(X_wholesale, y_wholesale, test_size=0.3, random_state=42)
)
X_car_train, X_car_test, y_car_train, y_car_test = train_test_split(
    X_car, y_car, test_size=0.3, random_state=42
)

(
    X_wholesale_pca,
    X_wholesale_ica,
    X_wholesale_rp,
    X_wholesale_tsne,
    X_wholesale_isomap,
    wholesale_ica_kurtosis,
) = apply_dimensionality_reduction(X_wholesale)
X_car_pca, X_car_ica, X_car_rp, X_car_tsne, X_car_isomap, car_ica_kurtosis = (
    apply_dimensionality_reduction(X_car)
)

X_car_train_pca, X_car_test_pca, y_car_train_pca, y_car_test_pca = train_test_split(
    X_car_pca, y_car, test_size=0.3, random_state=42
)
X_car_train_ica, X_car_test_ica, y_car_train_ica, y_car_test_ica = train_test_split(
    X_car_ica, y_car, test_size=0.3, random_state=42
)
X_car_train_rp, X_car_test_rp, y_car_train_rp, y_car_test_rp = train_test_split(
    X_car_rp, y_car, test_size=0.3, random_state=42
)
X_car_train_tsne, X_car_test_tsne, y_car_train_tsne, y_car_test_tsne = train_test_split(
    X_car_tsne, y_car, test_size=0.3, random_state=42
)
X_car_train_isomap, X_car_test_isomap, y_car_train_isomap, y_car_test_isomap = (
    train_test_split(X_car_isomap, y_car, test_size=0.3, random_state=42)
)

(
    X_wholesale_train_pca,
    X_wholesale_test_pca,
    y_wholesale_train_pca,
    y_wholesale_test_pca,
) = train_test_split(X_wholesale_pca, y_wholesale, test_size=0.3, random_state=42)
(
    X_wholesale_train_ica,
    X_wholesale_test_ica,
    y_wholesale_train_ica,
    y_wholesale_test_ica,
) = train_test_split(X_wholesale_ica, y_wholesale, test_size=0.3, random_state=42)
X_wholesale_train_rp, X_wholesale_test_rp, y_wholesale_train_rp, y_wholesale_test_rp = (
    train_test_split(X_wholesale_rp, y_wholesale, test_size=0.3, random_state=42)
)
(
    X_wholesale_train_tsne,
    X_wholesale_test_tsne,
    y_wholesale_train_tsne,
    y_wholesale_test_tsne,
) = train_test_split(X_wholesale_tsne, y_wholesale, test_size=0.3, random_state=42)
(
    X_wholesale_train_isomap,
    X_wholesale_test_isomap,
    y_wholesale_train_isomap,
    y_wholesale_test_isomap,
) = train_test_split(X_wholesale_isomap, y_wholesale, test_size=0.3, random_state=42)


def visualize_reduced_data(X_reduced, y, title, label_encoder=None):
    plt.figure(figsize=(10, 6))
    if label_encoder:
        y = label_encoder.inverse_transform(y)
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette="viridis")
    plt.title(title)
    plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


visualize_reduced_data(X_car_pca[:, :2], y_car, "PCA Reduced Data", car_label_encoder)
visualize_reduced_data(X_car_ica[:, :2], y_car, "ICA Reduced Data", car_label_encoder)
visualize_reduced_data(X_car_rp[:, :2], y_car, "RP Reduced Data", car_label_encoder)
visualize_reduced_data(X_car_tsne, y_car, "t-SNE Reduced Data", car_label_encoder)
visualize_reduced_data(
    X_car_isomap[:, :2], y_car, "Isomap Reduced Data", car_label_encoder
)

wholesale_kmeans_labels_pca, wholesale_em_labels_pca = apply_clustering(X_wholesale_pca)
wholesale_kmeans_labels_ica, wholesale_em_labels_ica = apply_clustering(X_wholesale_ica)
wholesale_kmeans_labels_rp, wholesale_em_labels_rp = apply_clustering(X_wholesale_rp)
wholesale_kmeans_labels_tsne, wholesale_em_labels_tsne = apply_clustering(
    X_wholesale_tsne
)
wholesale_kmeans_labels_isomap, wholesale_em_labels_isomap = apply_clustering(
    X_wholesale_isomap
)

car_kmeans_labels_pca, car_em_labels_pca = apply_clustering(X_car_pca)
car_kmeans_labels_ica, car_em_labels_ica = apply_clustering(X_car_ica)
car_kmeans_labels_rp, car_em_labels_rp = apply_clustering(X_car_rp)
car_kmeans_labels_tsne, car_em_labels_tsne = apply_clustering(X_car_tsne)
car_kmeans_labels_isomap, car_em_labels_isomap = apply_clustering(X_car_isomap)

plot_clusters(
    X_wholesale_pca[:, :2], wholesale_kmeans_labels_pca, "K-Means on PCA Wholesale Data"
)
plot_clusters(
    X_wholesale_tsne, wholesale_kmeans_labels_tsne, "K-Means on t-SNE Wholesale Data"
)
plot_clusters(
    X_wholesale_isomap[:, :2],
    wholesale_kmeans_labels_isomap,
    "K-Means on Isomap Wholesale Data",
)

plot_clusters(
    X_car_pca[:, :2],
    car_kmeans_labels_pca,
    "K-Means on PCA Car Data",
    car_label_encoder,
)
plot_clusters(
    X_car_tsne, car_kmeans_labels_tsne, "K-Means on t-SNE Car Data", car_label_encoder
)
plot_clusters(
    X_car_isomap[:, :2],
    car_kmeans_labels_isomap,
    "K-Means on Isomap Car Data",
    car_label_encoder,
)

nn_accuracy_car_pca, cm_car_pca, cr_car_pca = train_evaluate_nn_a1(
    X_car_train_pca, X_car_test_pca, y_car_train, y_car_test
)
nn_accuracy_car_ica, cm_car_ica, cr_car_ica = train_evaluate_nn_a1(
    X_car_train_ica, X_car_test_ica, y_car_train, y_car_test
)
nn_accuracy_car_rp, cm_car_rp, cr_car_rp = train_evaluate_nn_a1(
    X_car_train_rp, X_car_test_rp, y_car_train, y_car_test
)
nn_accuracy_car_tsne, cm_car_tsne, cr_car_tsne = train_evaluate_nn_a1(
    X_car_train_tsne, X_car_test_tsne, y_car_train, y_car_test
)
nn_accuracy_car_isomap, cm_car_isomap, cr_car_isomap = train_evaluate_nn_a1(
    X_car_train_isomap, X_car_test_isomap, y_car_train, y_car_test
)

print("Neural Network Accuracy with PCA (Car):", nn_accuracy_car_pca)
print(pd.DataFrame(cr_car_pca).transpose())
print("Neural Network Accuracy with ICA (Car):", nn_accuracy_car_ica)
print(pd.DataFrame(cr_car_ica).transpose())
print("Neural Network Accuracy with RP (Car):", nn_accuracy_car_rp)
print(pd.DataFrame(cr_car_rp).transpose())
print("Neural Network Accuracy with t-SNE (Car):", nn_accuracy_car_tsne)
print(pd.DataFrame(cr_car_tsne).transpose())
print("Neural Network Accuracy with Isomap (Car):", nn_accuracy_car_isomap)
print(pd.DataFrame(cr_car_isomap).transpose())

X_car_with_clusters = car_evaluation.drop(columns=["class"])
y_car_with_clusters = car_evaluation["class"]
(
    X_car_with_clusters_train,
    X_car_with_clusters_test,
    y_car_with_clusters_train,
    y_car_with_clusters_test,
) = train_test_split(
    X_car_with_clusters, y_car_with_clusters, test_size=0.3, random_state=42
)

nn_accuracy_clusters, cm_clusters, cr_clusters = train_evaluate_nn_a1(
    X_car_with_clusters_train,
    X_car_with_clusters_test,
    y_car_with_clusters_train,
    y_car_with_clusters_test,
)

print("Neural Network Accuracy with Clustering Features (Car):", nn_accuracy_clusters)
print(pd.DataFrame(cr_clusters).transpose())

nn_accuracy_wholesale_pca, cm_wholesale_pca, cr_wholesale_pca = train_evaluate_nn_a1(
    X_wholesale_train_pca,
    X_wholesale_test_pca,
    y_wholesale_train_pca,
    y_wholesale_test_pca,
)
nn_accuracy_wholesale_ica, cm_wholesale_ica, cr_wholesale_ica = train_evaluate_nn_a1(
    X_wholesale_train_ica,
    X_wholesale_test_ica,
    y_wholesale_train_ica,
    y_wholesale_test_ica,
)
nn_accuracy_wholesale_rp, cm_wholesale_rp, cr_wholesale_rp = train_evaluate_nn_a1(
    X_wholesale_train_rp, X_wholesale_test_rp, y_wholesale_train_rp, y_wholesale_test_rp
)
nn_accuracy_wholesale_tsne, cm_wholesale_tsne, cr_wholesale_tsne = train_evaluate_nn_a1(
    X_wholesale_train_tsne,
    X_wholesale_test_tsne,
    y_wholesale_train_tsne,
    y_wholesale_test_tsne,
)
nn_accuracy_wholesale_isomap, cm_wholesale_isomap, cr_wholesale_isomap = (
    train_evaluate_nn_a1(
        X_wholesale_train_isomap,
        X_wholesale_test_isomap,
        y_wholesale_train_isomap,
        y_wholesale_test_isomap,
    )
)

print("Neural Network Accuracy with PCA (Wholesale):", nn_accuracy_wholesale_pca)
print(pd.DataFrame(cr_wholesale_pca).transpose())
print("Neural Network Accuracy with ICA (Wholesale):", nn_accuracy_wholesale_ica)
print(pd.DataFrame(cr_wholesale_ica).transpose())
print("Neural Network Accuracy with RP (Wholesale):", nn_accuracy_wholesale_rp)
print(pd.DataFrame(cr_wholesale_rp).transpose())
print("Neural Network Accuracy with t-SNE (Wholesale):", nn_accuracy_wholesale_tsne)
print(pd.DataFrame(cr_wholesale_tsne).transpose())
print("Neural Network Accuracy with Isomap (Wholesale):", nn_accuracy_wholesale_isomap)
print(pd.DataFrame(cr_wholesale_isomap).transpose())

X_wholesale_with_clusters = wholesale_customers.drop(
    columns=["Total Spend", "Category"]
)
y_wholesale_with_clusters = wholesale_customers["Category"]
(
    X_wholesale_with_clusters_train,
    X_wholesale_with_clusters_test,
    y_wholesale_with_clusters_train,
    y_wholesale_with_clusters_test,
) = train_test_split(
    X_wholesale_with_clusters, y_wholesale_with_clusters, test_size=0.3, random_state=42
)

nn_accuracy_clusters_wholesale, cm_clusters_wholesale, cr_clusters_wholesale = (
    train_evaluate_nn_a1(
        X_wholesale_with_clusters_train,
        X_wholesale_with_clusters_test,
        y_wholesale_with_clusters_train,
        y_wholesale_with_clusters_test,
    )
)

print(
    "Neural Network Accuracy with Clustering Features (Wholesale):",
    nn_accuracy_clusters_wholesale,
)
print(pd.DataFrame(cr_clusters_wholesale).transpose())

results_car = {
    "Feature Type": ["PCA", "ICA", "RP", "t-SNE", "Isomap", "Clusters"],
    "Accuracy": [
        nn_accuracy_car_pca,
        nn_accuracy_car_ica,
        nn_accuracy_car_rp,
        nn_accuracy_car_tsne,
        nn_accuracy_car_isomap,
        nn_accuracy_clusters,
    ],
}
results_car_df = pd.DataFrame(results_car)

plt.figure(figsize=(10, 6))
sns.barplot(data=results_car_df, x="Feature Type", y="Accuracy")
plt.title("Neural Network Accuracy with Different Features (Car Data)")
plt.xlabel("Feature Type")
plt.ylabel("Accuracy")
plt.show()

results_wholesale = {
    "Feature Type": ["PCA", "ICA", "RP", "t-SNE", "Isomap", "Clusters"],
    "Accuracy": [
        nn_accuracy_wholesale_pca,
        nn_accuracy_wholesale_ica,
        nn_accuracy_wholesale_rp,
        nn_accuracy_wholesale_tsne,
        nn_accuracy_wholesale_isomap,
        nn_accuracy_clusters_wholesale,
    ],
}
results_wholesale_df = pd.DataFrame(results_wholesale)

plt.figure(figsize=(10, 6))
sns.barplot(data=results_wholesale_df, x="Feature Type", y="Accuracy")
plt.title("Neural Network Accuracy with Different Features (Wholesale Data)")
plt.xlabel("Feature Type")
plt.ylabel("Accuracy")
plt.show()

cm_df_car_pca = pd.DataFrame(cm_car_pca)
cm_df_car_ica = pd.DataFrame(cm_car_ica)
cm_df_car_rp = pd.DataFrame(cm_car_rp)
cm_df_car_tsne = pd.DataFrame(cm_car_tsne)
cm_df_car_isomap = pd.DataFrame(cm_car_isomap)
cm_df_clusters_car = pd.DataFrame(cm_clusters)
cm_df_clusters_wholesale = pd.DataFrame(cm_clusters_wholesale)

print("Confusion Matrix for PCA (Car):\n", cm_df_car_pca)
print("Confusion Matrix for ICA (Car):\n", cm_df_car_ica)
print("Confusion Matrix for RP (Car):\n", cm_df_car_rp)
print("Confusion Matrix for t-SNE (Car):\n", cm_df_car_tsne)
print("Confusion Matrix for Isomap (Car):\n", cm_df_car_isomap)
print("Confusion Matrix for Clusters (Car):\n", cm_df_clusters_car)
print("Confusion Matrix for Clusters (Wholesale):\n", cm_df_clusters_wholesale)

plot_kurtosis(
    car_ica_kurtosis, "Kurtosis of Independent Components from ICA (Car Data)"
)
plot_kurtosis(
    wholesale_ica_kurtosis,
    "Kurtosis of Independent Components from ICA (Wholesale Data)",
)
