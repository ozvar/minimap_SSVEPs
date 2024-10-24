from mne.decoding import LinearModel, Vectorizer, get_coef
import yaml
import pickle
from pathlib import Path
from src import preprocessing, classification, vizualisation, utils, spectral_analysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from mne import EvokedArray

# Declare file locations
root_dir = Path.home() / "Git" / "minimap_SSVEPs"
data_dir = root_dir / "data"
logs_dir= root_dir / "results" / "kokkinakis"
fig_dir = root_dir / "results" / "kokkinakis" / "figures"
params_dir = root_dir / "code" / "conf"

# Load params
with open(params_dir / "parameters.yaml", 'r') as file:
    parameters = yaml.safe_load(file)
# Initialize vizualisation settings
vizualisation.sns_styleset()

# Get paths for data files
left_data_folder = preprocessing.get_all_sample_filenames(
    data_dir / "kokkinakis" / "left_minimap_eeg_data",
    "edf"
)
right_data_folder = preprocessing.get_all_sample_filenames(
    data_dir / "kokkinakis" / "right_minimap_eeg_data",
    "edf"
)

# Load data if already preprocessed
try:
    left_minimap_epochs, right_minimap_epochs = utils.load_epochs(data_dir / "kokkinakis" / "epochs.pickle") 
# Otherwise preprocess it
except TypeError:
    left_minimap_epochs = preprocessing.ingest_samples(
            left_data_folder,
            parameters["preprocessing"]
    )
    right_minimap_epochs = preprocessing.ingest_samples(
            right_data_folder,
            parameters["preprocessing"]
    )
    # Equalize the number of epochs for each event between the two sets
    preprocessing.equalize_epoch_counts(
            left_minimap_epochs,
            right_minimap_epochs
    )
    # Pickle the data for later
    with open(data_dir / "kokkinakis" / "epochs.pickle", 'wb') as f:
        pickle.dump([left_minimap_epochs, right_minimap_epochs], f
        )

# Set labels for plots
group_labels = {
    left_minimap_epochs: 'left_group',
    right_minimap_epochs: 'right_group'
}

X, y = classification.prepare_clf_data(
    left_minimap_epochs,
    right_minimap_epochs,
    method='concat_epochs'
)

# Normally this would be a quick pytest, but I'm doing it quickly here 
# because we have a specific hardcoded way of reading files above.
X_2, y_2 = classification.prepare_clf_data(
    left_minimap_epochs,
    right_minimap_epochs,
)
X_psd = classification.compute_psd_and_features(X)
validation_comp =  X_2 == X_psd
assert validation_comp.all()

lr_clf = make_pipeline(
    Vectorizer(),
    StandardScaler(),
    LinearModel(
        LogisticRegression(max_iter=200)
    )
)
lr_clf.fit(X, y)

for name in ("patterns_", "filters_"):
    coef = get_coef(lr_clf, name, inverse_transform=True)
    evoked = EvokedArray(coef, X.info, tmin=X.tmin)
    fig = evoked.plot_topomap()
    fig.suptitle(f"Combined Minimap Epochs {name[:-1]}")