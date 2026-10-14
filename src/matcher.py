import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

class SnowLeopardMatcher:
    def __init__(self):
        """
        Initialize the matcher for snow leopard identification
        """
        self.reference_features = {}
        self.reference_names = {}
        
    def add_reference(self, name, features, image_path):
        """
        Add a reference individual to the database
        
        Args:
            name: Name/ID of the individual
            features: Feature embeddings
            image_path: Path to reference image
        """
        if name not in self.reference_features:
            self.reference_features[name] = []
            self.reference_names[name] = []
        
        self.reference_features[name].append(features)
        self.reference_names[name].append(image_path)
        
    def find_match(self, query_features, threshold=0.7):
        """
        Find the best matching individual for query features
        
        Args:
            query_features: Feature embeddings of query image
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            Tuple of (best_match_name, similarity_score, all_scores)
        """
        if not self.reference_features:
            return None, 0.0, {}
        
        best_match = None
        best_score = -1
        all_scores = {}
        
        # Compare with each reference individual
        for name, feature_list in self.reference_features.items():
            # Calculate similarity with all images of this individual
            similarities = []
            for ref_features in feature_list:
                # Reshape for cosine similarity
                query = query_features.reshape(1, -1)
                ref = ref_features.reshape(1, -1)
                
                # Calculate cosine similarity
                sim = cosine_similarity(query, ref)[0][0]
                similarities.append(sim)
            
            # Use maximum similarity for this individual
            max_sim = max(similarities)
            all_scores[name] = max_sim
            
            if max_sim > best_score:
                best_score = max_sim
                best_match = name
        
        # Check if best match meets threshold
        if best_score < threshold:
            return "Unknown", best_score, all_scores
        
        return best_match, best_score, all_scores
    
    def get_top_matches(self, query_features, top_k=3):
        """
        Get top K matching individuals
        
        Args:
            query_features: Feature embeddings of query image
            top_k: Number of top matches to return
            
        Returns:
            List of tuples (name, similarity_score)
        """
        if not self.reference_features:
            return []
        
        all_scores = {}
        
        # Calculate scores for all individuals
        for name, feature_list in self.reference_features.items():
            similarities = []
            for ref_features in feature_list:
                query = query_features.reshape(1, -1)
                ref = ref_features.reshape(1, -1)
                sim = cosine_similarity(query, ref)[0][0]
                similarities.append(sim)
            
            all_scores[name] = max(similarities)
        
        # Sort by similarity score
        sorted_matches = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_matches[:top_k]
    
    def get_reference_images(self, name):
        """
        Get all reference image paths for a specific individual
        
        Args:
            name: Name/ID of the individual
            
        Returns:
            List of image paths for that individual
        """
        if name in self.reference_names:
            return self.reference_names[name]
        return []
    
    def get_statistics(self):
        """
        Get statistics about the reference database
        """
        total_individuals = len(self.reference_features)
        total_images = sum(len(imgs) for imgs in self.reference_features.values())
        
        return {
            'total_individuals': total_individuals,
            'total_images': total_images,
            'individuals': list(self.reference_features.keys())
        }