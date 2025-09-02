import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import sklearn

# Versions
print(f"scikit-learn: {sklearn.__version__}")
print(f"xgboost: {xgb.__version__}")

# ------------------------------
# Load Dataset
# ------------------------------
columns = [
    "Communication Duration", "Communication Protocol", "Service Type", "flag",
    "Source Data Volume", "Destination Data Volume", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "Communication Count", "Service Communication Count",
    "Session Error Rate", "Service Error Rate", "Reception Error Rate", "Service Reception Error Rate",
    "Same Service Communication Rate", "Different Service Communication Rate", "srv_diff_host_rate",
    "Destination Host Count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

train_data = pd.read_csv("nsl-kdd_dataset.csv", header=None, names=columns)
train_data.drop_duplicates(inplace=True)

# ------------------------------
# Map attacks to categories
# ------------------------------
attack_mapping = {
    # DoS
    "back": "Traffic Flooding Attack", "land": "Traffic Flooding Attack", "neptune": "Traffic Flooding Attack",
    "pod": "Traffic Flooding Attack", "smurf": "Traffic Flooding Attack", "apache2": "Traffic Flooding Attack",
    "udpstorm": "Traffic Flooding Attack", "worm": "Traffic Flooding Attack",
    # Probe
    "ipsweep": "Topology Discovery Attack", "nmap": "Topology Discovery Attack", "portsweep": "Topology Discovery Attack",
    "satan": "Topology Discovery Attack", "mscan": "Topology Discovery Attack", "saint": "Topology Discovery Attack",
    # R2L
    "ftp_write": "Unauthorized Access Attack", "guess_passwd": "Unauthorized Access Attack", "imap": "Unauthorized Access Attack",
    "multihop": "Unauthorized Access Attack", "phf": "Unauthorized Access Attack", "spy": "Unauthorized Access Attack",
    "warezclient": "Unauthorized Access Attack", "warezmaster": "Unauthorized Access Attack",
    "sendmail": "Unauthorized Access Attack", "named": "Unauthorized Access Attack",
    "xlock": "Unauthorized Access Attack", "xsnoop": "Unauthorized Access Attack",
    "snmpgetattack": "Unauthorized Access Attack", "snmpguess": "Unauthorized Access Attack",
    # U2R
    "buffer_overflow": "Privilege Escalation Attack", "loadmodule": "Privilege Escalation Attack",
    "perl": "Privilege Escalation Attack", "rootkit": "Privilege Escalation Attack", "httptunnel": "Privilege Escalation Attack",
    "ps": "Privilege Escalation Attack", "sqlattack": "Privilege Escalation Attack",
    # Normal
    "normal": "Normal"
}

train_data['attack_category'] = train_data['label'].map(attack_mapping)
train_data.dropna(inplace=True)

# ------------------------------
# Encode categorical features
# ------------------------------
categorical_columns = ["Communication Protocol", "Service Type", "flag", "label", "attack_category"]
imputer = SimpleImputer(strategy='most_frequent')
train_data[categorical_columns] = imputer.fit_transform(train_data[categorical_columns])

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    label_encoders[col] = le

# ------------------------------
# Features & Target
# ------------------------------
X = train_data[
    ["Communication Duration", "Communication Protocol", "Service Type",
     "Source Data Volume", "Destination Data Volume", "Communication Count",
     "Service Communication Count", "Session Error Rate", "Service Error Rate",
     "Reception Error Rate", "Service Reception Error Rate",
     "Same Service Communication Rate", "Different Service Communication Rate",
     "Destination Host Count"]
]
y = train_data['attack_category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------------
# Oversampling using SMOTE
# ------------------------------
smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# ------------------------------
# Define Base Models
# ------------------------------
rf_model = RandomForestClassifier(
    n_estimators=50, max_depth=9, min_samples_split=15, min_samples_leaf=10,
    max_features=0.5, max_samples=0.8, random_state=42, bootstrap=True
)

gb_model = GradientBoostingClassifier(
    n_estimators=10, learning_rate=0.05, max_depth=5, random_state=42
)

xgb_model = xgb.XGBClassifier(
    objective='multi:softmax', max_depth=4, eta=0.05, num_class=4, nthread=4, random_state=42
)

# ------------------------------
# Train Models
# ------------------------------
rf_model.fit(X_resampled, y_resampled)
gb_model.fit(X_resampled, y_resampled)
xgb_model.fit(X_resampled, y_resampled)

# Save models and encoders
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(gb_model, "gb_model.pkl")
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(label_encoders["attack_category"], "label_encoder.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# ------------------------------
# Predictions & Blending
# ------------------------------
rf_preds = rf_model.predict_proba(X_test)
gb_preds = gb_model.predict_proba(X_test)
xgb_preds = xgb_model.predict_proba(X_test)

weights = [0.4, 0.3, 0.3]
blended_preds = (weights[0] * rf_preds + weights[1] * gb_preds + weights[2] * xgb_preds)

final_preds = np.argmax(blended_preds, axis=1)

# ------------------------------
# Evaluation
# ------------------------------
accuracy = accuracy_score(y_test, final_preds)
print(f"Blending Model Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, final_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, final_preds))
