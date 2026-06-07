import numpy as np
import pandas as pd 
import pickle
import json
import os
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,f1_score,accuracy_score,silhouette_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings 
warnings.filterwarnings("ignore")

print("Pipeline Started")
def generate_loan_dataset(n_samples: int = 5000 , random_state: int =42) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    data = {
        "age": rng.randint(22,65,n_samples),
        "income": rng.normal(55000,20000,n_samples).clip(15000,200000),
        "loan_amount":rng.normal(15000,8000,n_samples).clip(1000,50000),
        "loan_term_months":rng.choice([12,24,36,48,60], n_samples),
        "credit_score":rng.normal(650,80,n_samples).clip(300,8500).astype(int),
        "num_existing_loans":rng.randint(0,5,n_samples),
        "employment_years":rng.exponential(5,n_samples).clip(300,850).astype(int),
        "debt_to_income":rng.beta(2,5,n_samples),
        "has_collateral":rng.choice([0,1],n_samples,p=[0.6,0.4]),
        "education": rng.choice(["High School", "Bachelor", "Master", "PhD"], n_samples, p=[0.3, 0.4, 0.2, 0.1]), "loan_purpose": rng.choice(["Personal", "Business", "Education", "Home", "Auto"], n_samples),
         "region": rng.choice(["North", "South", "East", "West"], n_samples),

    }
    df = pd.DataFrame(data)

    #Enigneer synthetic default probability (realistic business logic)
    default_prob = (
        0.3 * (1 - (df["credit_score"] - 300) /550) +
        0.2 * (df["debt_to_income"]) +
        0.15 * (df["num_existing_loans"] / 5 ) +
        0.1 * (1 - df["has_collateral"]) +
        0.1 * (df["loan_amount"] / df["income"]) +
        rng.normal(0 , 0.5, n_samples)
    ).clip(0 ,1 )

    df["default"] = (default_prob > 0.45).astype(int)
    df["loan_id"] = [f"LN{str(i).zfill(6)}" for i in range(n_samples)]

    return df

#Feature Engineering 
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    #ratio feature(key for finance ML)
    df["loan_to_income_ratio"]    = df["loan_amount"] / df["income"]
    df["monthly_payment_estimate"] = df["loan_amount"] / df["loan_term_months"]
    df["payment_to_income_ratio"] = df["monthly_payment_estimate"] / (df["income"] / 12)
    df["credit_score_band"]       = pd.cut(df["credit_score"],
                                        bins=[299, 579, 669, 739, 799, 851],
                                           labels=[1, 2, 3, 4, 5],                                           
                                           include_lowest=True
                                            )
    print("Credit Score Band NaN Count:",
    df["credit_score_band"].isna().sum())
    df["age_group"]               = pd.cut(df["age"],bins=[21, 30, 40, 50, 66],
                                           labels=[0, 1, 2, 3],    
                                           include_lowest=True
                                                )
 
    print("Age Group NaN Count:",
      df["age_group"].isna().sum())

    df["age_group"] = df["age_group"].fillna(0).astype(int)

    df["credit_score_band"] = df["credit_score_band"].fillna(1).astype(int)
    df["is_high_risk_purpose"]    = df["loan_purpose"].isin(["Personal"]).astype(int)
     # Encode categoricals

    for col in ["education", "loan_purpose", "region"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])

    return df

def get_feature_columns():
    return["age", "income", "loan_amount", "loan_term_months", "credit_score",
        "num_existing_loans", "employment_years", "debt_to_income", "has_collateral",
        "loan_to_income_ratio", "monthly_payment_estimate", "payment_to_income_ratio",
        "credit_score_band", "age_group", "is_high_risk_purpose",
        "education_enc", "loan_purpose_enc", "region_enc"]


#supervised learning - classification

def train_supervised_models(X_train, X_test, y_train, y_test):
    """
    Train & evaluate multiple classifiers.
    Returns best model + metrics dict.
    """
    model = {
        "Logistic Regewssion" : Pipeline([("imputer", SimpleImputer(strategy="median")),
                                          ("scaler", StandardScaler()),
                                          ("clf" , LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1))
                                          ]),
        "Random Forest" :Pipeline([("imputer", SimpleImputer(strategy="median")),
                                   ("clf", RandomForestClassifier( n_estimators=200, max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=-1 ))
                                   ]),
      "Gradient Boosting": Pipeline([ ("imputer", SimpleImputer(strategy="median")), 
                                     ("scaler", StandardScaler()),
                                     ("clf", GradientBoostingClassifier( n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42 )) 
                                     ]),
                }
    results = {}
    best_model = None
    best_auc = 0.0
    cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    for name , pipeline in model.items():
        print(f"\n── Training: {name} ──") 
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_pred_prob = pipeline.predict_proba(X_test)[:,1]

        cv_scores = cross_val_score(pipeline,X_train,y_train,cv = cv, scoring="roc_auc", n_jobs=1)

        metrics = { 
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_pred_prob), 4), 
            "cv_auc_mean": round(cv_scores.mean(), 4), 
            "cv_auc_std": round(cv_scores.std(), 4), 
            "report": classification_report(y_test, y_pred, output_dict=True), 
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist() 
            }
        results[name] = metrics
        print(f" AUC: {metrics['roc_auc']} | F1: {metrics['f1_score']} | CV-AUC: {metrics['cv_auc_mean']} ± {metrics['cv_auc_std']}")

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model = (name , pipeline)

    print(f"\n✅ Best Model: {best_model[0]} (AUC={best_auc})") 
    return best_model, results

# unsupervised learning - customer segementation 
def train_clustering(df: pd.DataFrame):
    """ K-Means clustering for customer segmentation. Groups customers by financial behavior. """
    cluster_features = ["income", "credit_score", "debt_to_income", 
                        "loan_to_income_ratio","employment_years"]
    X_cluster = df[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Find optimal K using Elbow + Silhouette
    silhouette_scores = {}
    for k in range(2,8):
        km = KMeans(n_clusters=k,random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        silhouette_scores[k] = round(silhouette_score(X_scaled,labels), 4)
    
    optimal_k = max(silhouette_scores, key = silhouette_scores.get)
    print(f"\n📊 Optimal clusters (Silhouette): K={optimal_k}")

    # Final model
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df["segment"] = kmeans.fit_predict(X_scaled)

    # Segment summary
    segment_summary = df.groupby("segment").agg(count=("segment", "count"),         
                                                avg_income=("income", "mean"),
                                                avg_credit_score=("credit_score", "mean"),
                                                avg_dti=("debt_to_income", "mean"), 
                                                default_rate=("default", "mean") ).round(2).to_dict()
    
    return kmeans,scaler,cluster_features,segment_summary,silhouette_scores

#feature importance
def extract_feature_importance(model_pipeline, feature_names):
    """Extract feature importance from tree-based models."""
    try:
        clf = model_pipeline.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            return{}
        importance_dict = dict(zip(feature_names,importances.tolist()))
        return dict(sorted(importance_dict.items(), key= lambda x:x[1],reverse=True))
    except Exception:
        return{}

#main -orchestrate full pipeline
def run_pipeline():
    print("=" * 55)
    print(" LOAN RISK PREDICTION — ML PIPELINE") 
    print("=" * 55)  

    os.makedirs("models",exist_ok=True)

    print("\n[1/5] Generating dataset...") 
    df = generate_loan_dataset(n_samples=5000) 
    df = feature_engineering(df) 
    df.to_csv("data/loans.csv", index=False) 
    print(f" Dataset: {df.shape} | Default rate: {df['default'].mean():.2%}")   

    print("\n[2/5] Supervised learning...") 
    features = get_feature_columns() 
    X = df[features] 
    y = df["default"] 
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, stratify=y, random_state=42 ) 
    (best_name, best_pipeline), all_results = train_supervised_models( X_train, X_test, y_train, y_test )

    print("\n[3/5] Unsupervised clustering...") 
    kmeans, cluster_scaler, cluster_features, segment_summary, silhouette_scores = train_clustering(df)

    print("\n[4/5] Feature importance...") 
    feature_importance = extract_feature_importance(best_pipeline, features)

    print("\n[5/5] Saving artifacts...")
    with open("models/best_model.pkl", "wb") as f: 
        pickle.dump(best_pipeline, f) 
    with open("models/kmeans_model.pkl", "wb") as f: 
        pickle.dump(kmeans, f) 
    with open("models/cluster_scaler.pkl", "wb") as f: 
        pickle.dump(cluster_scaler, f) 
    with open("models/feature_columns.pkl", "wb") as f: 
        pickle.dump(features, f) 
    with open("models/cluster_features.pkl","wb") as f: 
        pickle.dump(cluster_features, f)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)
    
    metadata = {
        "best_model_name": best_name,
        "best_model_metrics": all_results[best_name],
        "all_model_results": all_results,
        "feature_importance": feature_importance, 
        "segment_summary":segment_summary, 
        "silhouette_scores": {str(k): v for k, v in silhouette_scores.items()}, 
        "features": features, 
        "cluster_features": cluster_features,
        "dataset_stats": { "total_samples": int(len(df)), 
                          "default_rate": round(float(df["default"].mean()), 4), 
                          "train_size": int(len(X_train)), 
                          "test_size": int(len(X_test)) }
    }

    with open("models/metadata.json", "w") as f: 
        json.dump(metadata, f, indent=2)

    print("\n✅ Pipeline complete. Artifacts saved to models/") 
    print(f" Best Model : {best_name}") 
    print(f" ROC-AUC : {all_results[best_name]['roc_auc']}") 
    print(f" F1-Score : {all_results[best_name]['f1_score']}") 
    return metadata

if __name__ == "__main__":
    print("Before Run Pipeline")
    run_pipeline()
    print("After Run Pipeline")





    






