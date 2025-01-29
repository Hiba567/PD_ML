import json
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier 
import opensmile

class EmoIdBrain:
    def __init__(self, hparams):
        self.hparams = hparams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #OpenSMILE
        #self.smile = opensmile.Smile(
            #feature_set=opensmile.FeatureSet.eGeMAPSv02,
            #feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        #)

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors
        )
        
        #Random Forest Classifier
        self.model = RandomForestClassifier(
            n_estimators=self.hparams.get('n_estimators', 100),  #Default 100
            max_depth=self.hparams.get('max_depth', None),
            min_samples_split=self.hparams.get('min_samples_split', 2),
            min_samples_leaf=self.hparams.get('min_samples_leaf', 1),
            random_state=42  
        )
        
        self.features_cache = {}
    
    def extract_features(self, wav_path):
        if wav_path in self.features_cache:
            return self.features_cache[wav_path]
        
        try:
            features = self.smile.process_file(wav_path).mean()
            self.features_cache[wav_path] = features
            return features
        except Exception as e:
            print(f"Error extracting features from {wav_path}: {str(e)}")
            return None

    def prepare_data(self, data):
        features_list = []
        labels_list = []
        genders_list = []
        ids_list = []
        
        for id_, item in tqdm(data.items(), desc="Extracting features"):
            wav_path = item['wav'].replace('__DATA_PATH__', self.hparams['data_folder'])
            features = self.extract_features(wav_path)
            
            if features is not None:
                features_list.append(features)
                label = 1 if item['label'] == 'True' else 0
                labels_list.append(label)
                genders_list.append(item['gender'])
                ids_list.append(id_)
        
        X = np.vstack(features_list)
        y = np.array(labels_list)
        return X, y, genders_list, ids_list

    def train(self, train_data, valid_data):
        X_train, y_train, _, _ = self.prepare_data(train_data)
        X_valid, y_valid, _, _ = self.prepare_data(valid_data)
        
        print("Training")
        self.model.train(X_train, y_train)
        
        #Validation metrics
        valid_pred = self.model.predict(X_valid)
        valid_acc = accuracy_score(y_valid, valid_pred)
        valid_f1 = f1_score(y_valid, valid_pred, average='weighted')
        print(f"Validation Accuracy: {valid_acc:.4f}")
        print(f"Validation F1 Score: {valid_f1:.4f}")

    def evaluate(self, test_data):
        X_test, y_test, genders, ids = self.prepare_data(test_data)
        
        print("Testing")
        predictions = self.model.predict(X_test)
        
        #Test metrics
        test_acc = accuracy_score(y_test, predictions)
        test_f1 = f1_score(y_test, predictions, average='weighted')
        
        #Metrics for male/female
        male_mask = np.array(genders) == 'Male'
        female_mask = np.array(genders) == 'Female'
        
        male_f1 = f1_score(y_test[male_mask], predictions[male_mask], average='weighted')
        female_f1 = f1_score(y_test[female_mask], predictions[female_mask], average='weighted')
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Male F1 Score: {male_f1:.4f}")
        print(f"Female F1 Score: {female_f1:.4f}")
        
        # Save predictions
        results_df = pd.DataFrame({
            'ID': ids,
            'True_Label': ['True' if label == 1 else 'False' for label in y_test],
            'Predicted_Label': ['True' if pred == 1 else 'False' for pred in predictions],
            'Gender': genders
        })
        results_df.to_csv(os.path.join(self.hparams['output_folder'], 'predictions.csv'), index=False)

def dataio_prep(hparams):
    """Prepare the datasets"""
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    
    for dataset in data_info:
        with open(data_info[dataset], 'r') as f:
            datasets[dataset] = json.load(f)
    
    return datasets

if __name__ == "__main__":
    # Parse arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    
    # Load hyperparameters
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    
    # Prepare datasets
    datasets = dataio_prep(hparams)
    
    # Initialize and train model
    model = EmoIdBrain(hparams)
    model.train(datasets["train"], datasets["valid"])
    model.evaluate(datasets["test"])