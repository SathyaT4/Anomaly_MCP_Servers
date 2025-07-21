from typing import Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture # Changed from KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import io
import warnings
import csv
import numpy as np
import hashlib
import time
import joblib # For saving/loading models

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans") # Still for general clustering warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class GMMAnomalyDetector:
    def __init__(self, output_plot_directory='anomaly_reports', trained_models_directory='trained_models'):
        self.output_plot_directory = output_plot_directory
        self.trained_models_directory = trained_models_directory
        self._ensure_directories()
        self.scaler = None
        self.gmm_model = None
        self.pca_model = None
        self.numerical_features = None
        self.anomaly_threshold = None
        self.model_id = None # To store the ID of the loaded/trained model

    def _ensure_directories(self):
        os.makedirs(self.output_plot_directory, exist_ok=True)
        os.makedirs(self.trained_models_directory, exist_ok=True)

    def preprocess_data(self, df_original: pd.DataFrame):
        df = df_original.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        numerical_features = df.select_dtypes(include=['number']).columns.tolist()
        exclude_cols = ['timestamp']
        uname_info_cols = [col for col in numerical_features if col.endswith('_uname_info')]
        exclude_cols.extend(uname_info_cols)
        self.numerical_features = [col for col in numerical_features if col not in exclude_cols]

        # for col in self.numerical_features:
        #     df[col] = df[col].ffill()
        #     df[col] = df[col].bfill()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(df[self.numerical_features])
        df_scaled = pd.DataFrame(X_scaled, columns=self.numerical_features, index=df.index)
        return df_scaled


    def find_optimal_n_components(self, df_scaled: pd.DataFrame, k_range: range = range(1, 11), criterion: str = 'bic') -> int:
        aic_values = []
        bic_values = []

        for k in k_range:
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(df_scaled)
            aic_values.append(gmm.aic(df_scaled))
            bic_values.append(gmm.bic(df_scaled))

        # Choose optimal k based on selected criterion
        if criterion == 'bic':
            optimal_k = k_range[bic_values.index(min(bic_values))]
        elif criterion == 'aic':
            optimal_k = k_range[aic_values.index(min(aic_values))]
        else:
            raise ValueError("Criterion must be either 'bic' or 'aic'")

        # Plotting AIC and BIC
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, aic_values, marker='o', linestyle='--', label='AIC')
        plt.plot(k_range, bic_values, marker='o', linestyle='-', label='BIC')
        plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal {criterion.upper()} k={optimal_k}')
        plt.title('AIC/BIC for Optimal Number of GMM Components')
        plt.xlabel('Number of Components (K)')
        plt.ylabel('Information Criterion Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(k_range)
        plt.tight_layout()
        plt.close()

        return optimal_k


    def train_model(self, df_original: pd.DataFrame, n_components: int, anomaly_threshold_percentile: float = 0.95, model_id: Optional[str] = None):
        print("Starting model training with GMM...")

        df_scaled = self.preprocess_data(df_original)

        # Generate a model_id if not provided
        if model_id is None:
            # Hash of current timestamp and n_components for a unique ID
            self.model_id = hashlib.sha256(f"{time.time()}-{n_components}".encode()).hexdigest()[:10]
        else:
            self.model_id = model_id
        
        # Create a model-specific output directory
        model_output_dir = os.path.join(self.output_plot_directory, self.model_id)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Also create a model-specific trained_models directory
        model_save_dir = os.path.join(self.trained_models_directory, self.model_id)
        os.makedirs(model_save_dir, exist_ok=True)
        
        print(f"Model ID for this training run: {self.model_id}")
        print(f"Output directory for plots/reports: {model_output_dir}")
        print(f"Save directory for trained models: {model_save_dir}")


        self.gmm_model = GaussianMixture(n_components=n_components, random_state=42)
        self.gmm_model.fit(df_scaled)
        cluster_labels = self.gmm_model.predict(df_scaled)

        df_scaled['Cluster'] = cluster_labels
        df_original['Cluster'] = cluster_labels # Add to original for analysis

        # Store component means (equivalent to centroids for GMM)
        cluster_centroids = self.gmm_model.means_

        # Save trained model components
        joblib.dump(self.scaler, os.path.join(model_save_dir, 'scaler.joblib'))
        joblib.dump(self.gmm_model, os.path.join(model_save_dir, 'gmm_model.joblib'))

        # If PCA was used for training and needs to be persisted for prediction, save it
        self.pca_model = PCA(n_components=2, random_state=42) # Re-initialize for this purpose
        X_pca = self.pca_model.fit_transform(df_scaled[self.numerical_features])
        df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = cluster_labels
        joblib.dump(self.pca_model, os.path.join(model_save_dir, 'pca_model.joblib')) # Save PCA model

        print("GMM model, scaler, and PCA model saved.")

        # --- Anomaly Detection and Metric Attribution ---
        # Anomaly scores based on negative log-likelihood
        anomaly_scores = -self.gmm_model.score_samples(df_scaled[self.numerical_features])
        df_original['Anomaly_Score'] = anomaly_scores
        df_original['Is_Anomaly'] = False # Initialize

        self.anomaly_threshold = df_original['Anomaly_Score'].quantile(anomaly_threshold_percentile)
        df_original['Is_Anomaly'] = df_original['Anomaly_Score'] > self.anomaly_threshold

        anomalies_df = df_original[df_original['Is_Anomaly']].sort_values(by='Anomaly_Score', ascending=False)
        
        all_anomaly_details = []
        if not anomalies_df.empty:
            print(f"Generating anomaly report for {len(anomalies_df)} anomalies...")
            for i, anomaly_row_original in anomalies_df.iterrows():
                anomaly_timestamp = anomaly_row_original['timestamp']
                anomaly_score = anomaly_row_original['Anomaly_Score']
                cluster_id = anomaly_row_original['Cluster'] # Most probable component

                if i in df_scaled.index:
                    anomaly_row_scaled = df_scaled.loc[i][self.numerical_features].astype(float)
                else:
                    print(f"Warning: Scaled data not found for index {i}, skipping anomaly metric attribution.")
                    continue

                assigned_component_mean = cluster_centroids[cluster_id]
                deviations = np.abs(anomaly_row_scaled.values - assigned_component_mean)
                deviation_series = pd.Series(deviations, index=self.numerical_features, dtype='float64')
                top_contributing_metrics = deviation_series.nlargest(5)

                for metric, dev_score in top_contributing_metrics.items():
                    original_value = anomaly_row_original.get(metric, None)
                    all_anomaly_details.append({
                        'Timestamp': anomaly_timestamp,
                        'Anomaly_Score': anomaly_score,
                        'Cluster': cluster_id,
                        'Metric': metric,
                        'Scaled_Deviation': round(dev_score, 4),
                        'Original_Value': round(original_value, 2) if pd.notnull(original_value) else 'NA'
                    })

        output_csv_path = os.path.join(model_output_dir, 'all_anomalies_with_top5_metrics.csv')
        keys = ['Timestamp', 'Anomaly_Score', 'Cluster', 'Metric', 'Scaled_Deviation', 'Original_Value']

        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_anomaly_details)
        print(f"Anomaly report saved to: {output_csv_path}")

        # --- Generate Plots ---
        plots_generated = []
        # Plot 1: AIC/BIC plot (already generated by find_optimal_n_components)
        # We need to call it if it wasn't called before or ensure it's saved to the model_output_dir
        aic_bic_plot_path, _, _ = self.find_optimal_n_components(df_scaled, k_range=range(1, 11))
        # Ensure it's moved/copied to the model-specific directory
        final_aic_bic_path = os.path.join(model_output_dir, os.path.basename(aic_bic_plot_path))
        if aic_bic_plot_path != final_aic_bic_path:
             os.rename(aic_bic_plot_path, final_aic_bic_path) # Move it to the specific model folder
        plots_generated.append(final_aic_bic_path)


        # Plot 2: Cluster Visualization (PCA)
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Cluster'], cmap='viridis', s=50, alpha=0.6, label='Data Points')
        plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], marker='X', s=200, c='red', edgecolor='black', label='Component Means') # Use component means
        plt.title(f'GMM Clusters (K={n_components}) on PC1 vs PC2 for Model ID: {self.model_id}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(scatter, label='Component ID')
        plt.legend()
        plt.grid(True)
        cluster_plot_path = os.path.join(model_output_dir, f'gmm_clusters_pc1_pc2_k{n_components}_model_{self.model_id}.png')
        plt.savefig(cluster_plot_path)
        plt.close()
        plots_generated.append(cluster_plot_path)
        print(f"Cluster plot saved to: {cluster_plot_path}")

        # Plot 3: Anomaly Timeline Plot
        plt.figure(figsize=(15, 7))
        plt.plot(df_original['timestamp'], df_original['Anomaly_Score'], label='Anomaly Score', color='blue', alpha=0.7)
        anomalous_points_df = df_original[df_original['Is_Anomaly']]
        plt.scatter(anomalous_points_df['timestamp'], anomalous_points_df['Anomaly_Score'], color='red', s=50, zorder=5, label='Anomaly Detected')
        plt.axhline(y=self.anomaly_threshold, color='green', linestyle='--', label=f'Anomaly Threshold ({self.anomaly_threshold:.2f})')
        plt.title(f'Anomaly Scores Over Time for Model ID: {self.model_id}')
        plt.xlabel('Timestamp')
        plt.ylabel('Anomaly Score (Negative Log-Likelihood)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        anomaly_timeline_plot_path = os.path.join(model_output_dir, f'anomaly_timeline_model_{self.model_id}.png')
        plt.savefig(anomaly_timeline_plot_path)
        plt.close()
        plots_generated.append(anomaly_timeline_plot_path)
        print(f"Anomaly timeline plot saved to: {anomaly_timeline_plot_path}")

        return {
            "model_id": self.model_id,
            "trained_model_path": model_save_dir,
            "anomaly_report_path": output_csv_path,
            "plots_path": plots_generated
        }
