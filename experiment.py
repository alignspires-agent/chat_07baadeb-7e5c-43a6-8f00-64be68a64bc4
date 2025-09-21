import sys
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import warnings
import os

# Disable warnings and set single-threaded mode for serverless environment
warnings.filterwarnings('ignore')
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Set up logging with reduced verbosity
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MoSESFramework:
    """
    Simplified implementation of the MoSEs framework for AI-generated text detection
    Based on the paper: "MoSEs: Uncertainty-Aware AI-Generated Text Detection via Mixture of Stylistics Experts"
    """
    
    def __init__(self, n_prototypes=5, n_components=32, random_state=42):
        self.n_prototypes = n_prototypes
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.scaler = StandardScaler()
        self.prototypes = {}
        self.cte_model = None
        self.srr_features = None
        self.srr_labels = None
        
    def extract_linguistic_features(self, texts):
        """
        Extract linguistic features from text (simulated for demonstration)
        In a real implementation, this would use actual text processing
        """
        logger.info("Extracting linguistic features from texts")
        
        # Simulated feature extraction - replace with actual text processing
        n_samples = len(texts)
        np.random.seed(self.random_state)
        features = np.zeros((n_samples, 7))  # 7 linguistic features
        
        # Simulate feature values (replace with actual feature extraction)
        features[:, 0] = np.random.randint(50, 500, n_samples)  # text length
        features[:, 1] = np.random.uniform(-5, -1, n_samples)   # log-prob mean
        features[:, 2] = np.random.uniform(0.1, 2.0, n_samples) # log-prob variance
        features[:, 3] = np.random.uniform(0.01, 0.3, n_samples) # 2-gram repetition
        features[:, 4] = np.random.uniform(0.01, 0.2, n_samples) # 3-gram repetition
        features[:, 5] = np.random.uniform(0.3, 0.8, n_samples)  # type-token ratio
        features[:, 6] = np.random.uniform(-1, 1, n_samples)     # semantic feature
        
        return features
    
    def build_srr(self, texts, labels):
        """
        Build Stylistics Reference Repository (SRR)
        """
        logger.info("Building Stylistics Reference Repository (SRR)")
        
        try:
            # Extract linguistic features
            features = self.extract_linguistic_features(texts)
            
            # Store features and labels
            self.srr_features = features
            self.srr_labels = labels
            
            # Create prototypes using K-means clustering
            logger.info(f"Creating {self.n_prototypes} prototypes using K-means")
            kmeans = KMeans(n_clusters=self.n_prototypes, random_state=self.random_state, 
                          n_init=10, max_iter=100)
            cluster_labels = kmeans.fit_predict(features)
            
            # Calculate cluster centroids as prototypes
            for i in range(self.n_prototypes):
                cluster_mask = (cluster_labels == i)
                if np.sum(cluster_mask) > 0:
                    self.prototypes[i] = np.mean(features[cluster_mask], axis=0)
            
            logger.info(f"SRR built with {len(texts)} samples and {len(self.prototypes)} prototypes")
            return features
            
        except Exception as e:
            logger.error(f"Error building SRR: {e}")
            sys.exit(1)
    
    def sar_router(self, input_features, m=3):
        """
        Stylistics-Aware Router (SAR) - finds m nearest prototypes
        """
        if not self.prototypes:
            logger.error("Prototypes not initialized. Build SRR first.")
            sys.exit(1)
        
        try:
            # Convert prototypes to array
            prototype_array = np.array(list(self.prototypes.values()))
            
            # Calculate distances to all prototypes
            distances = cdist(input_features.reshape(1, -1), prototype_array, metric='euclidean')
            
            # Find m nearest prototypes
            nearest_indices = np.argsort(distances[0])[:m]
            nearest_prototypes = [list(self.prototypes.keys())[i] for i in nearest_indices]
            
            return nearest_prototypes
            
        except Exception as e:
            logger.error(f"Error in SAR router: {e}")
            sys.exit(1)
    
    def retrieve_reference_samples(self, prototype_indices):
        """
        Retrieve reference samples based on prototype indices
        """
        if self.srr_features is None or self.srr_labels is None:
            logger.error("SRR not initialized. Build SRR first.")
            sys.exit(1)
        
        # For simplicity, return all samples (in real implementation, filter by prototype)
        # In a real implementation, you would filter samples based on their cluster assignments
        return self.srr_features, self.srr_labels
    
    def train_cte(self, X_train, y_train):
        """
        Train Conditional Threshold Estimator (CTE) using logistic regression
        """
        logger.info("Training Conditional Threshold Estimator (CTE)")
        
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X_train)
            
            # Train logistic regression model
            self.cte_model = LogisticRegression(random_state=self.random_state, max_iter=500)
            self.cte_model.fit(X_scaled, y_train)
            
            logger.info("CTE model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training CTE model: {e}")
            sys.exit(1)
    
    def predict(self, text_features):
        """
        Make prediction using the MoSEs framework
        """
        if self.cte_model is None:
            logger.error("CTE model not trained. Train model first.")
            sys.exit(1)
        
        try:
            # Route through SAR
            prototype_indices = self.sar_router(text_features)
            
            # Retrieve reference samples (simplified)
            ref_features, ref_labels = self.retrieve_reference_samples(prototype_indices)
            
            # Scale input features
            text_features_scaled = self.scaler.transform(text_features.reshape(1, -1))
            
            # Predict using CTE
            prediction = self.cte_model.predict(text_features_scaled)
            confidence = self.cte_model.predict_proba(text_features_scaled)
            
            return prediction[0], np.max(confidence)
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            sys.exit(1)

def generate_synthetic_data(n_samples=500):
    """
    Generate synthetic data for demonstration purposes
    In a real implementation, use actual text data
    """
    logger.info(f"Generating {n_samples} synthetic samples")
    
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate synthetic features (replace with actual text data)
        features = np.random.randn(n_samples, 7)
        
        # Generate synthetic labels (0 = human, 1 = AI-generated)
        labels = np.random.randint(0, 2, n_samples)
        
        # Generate synthetic text samples (placeholder)
        texts = [f"Sample text {i}" for i in range(n_samples)]
        
        return texts, features, labels
        
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        sys.exit(1)

def main():
    """
    Main function to demonstrate the MoSEs framework
    """
    logger.info("Starting MoSEs framework demonstration")
    
    try:
        # Generate synthetic data with reduced size for faster execution
        texts, features, labels = generate_synthetic_data(n_samples=500)
        
        # Split data into reference and test sets
        split_idx = int(0.8 * len(texts))
        train_texts, test_texts = texts[:split_idx], texts[split_idx:]
        train_features, test_features = features[:split_idx], features[split_idx:]
        train_labels, test_labels = labels[:split_idx], labels[split_idx:]
        
        logger.info(f"Data split: {len(train_texts)} training, {len(test_texts)} test samples")
        
        # Initialize MoSEs framework
        moses = MoSESFramework(n_prototypes=5, n_components=32)
        
        # Build SRR
        moses.build_srr(train_texts, train_labels)
        
        # Train CTE
        moses.train_cte(train_features, train_labels)
        
        # Test the framework with reduced logging
        logger.info("Testing MoSEs framework on test samples")
        predictions = []
        confidences = []
        
        # Process test samples with reduced logging
        for i, (feature, true_label) in enumerate(zip(test_features, test_labels)):
            pred, confidence = moses.predict(feature)
            predictions.append(pred)
            confidences.append(confidence)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions, average='weighted')
        
        logger.info(f"Final Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Average Confidence: {np.mean(confidences):.4f}")
        
        # Summary of findings
        logger.info("Summary: The simplified MoSEs framework has been implemented successfully.")
        logger.info("Note: This implementation uses synthetic data for demonstration.")
        logger.info("For real applications, replace with actual text features and embeddings.")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()