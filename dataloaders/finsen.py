# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# # from torch.utils.data import DataLoader, TensorDataset

# from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
# import torch

# # Other imports remain the same

# def create_weighted_sampler(labels):
#     class_counts = np.bincount(labels)
#     class_weights = 1. / class_counts
#     sample_weights = class_weights[labels]
#     return sample_weights  # Return sample weights, not the sampler


# # Function to read Finsen data
# def read_finsen_data(file_path):
#     df = pd.read_csv(file_path)
#     texts = df['Content'].tolist()
#     labels = LabelEncoder().fit_transform(df['Category'])
#     return texts, labels

# # Function to create train and validation DataLoaders
# def get_train_valid_loader(batch_size=128, valid_size=0.1, get_val_temp=0, random_seed=42, shuffle=True, num_workers=4, pin_memory=False):
#     # texts, labels = read_finsen_data('./data/FinSen/finsen.csv')
#     texts, labels = read_finsen_data('./adafocal-main/data/FinSen/finsen.csv')

#     # Vectorize and split the dataset
#     vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
#     features = vectorizer.fit_transform(texts).toarray()
#     X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=valid_size, random_state=random_seed)

#     # Create weighted sampler for the training set to handle class imbalance
#     train_weights = create_weighted_sampler(y_train)
#     # train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
#     train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)


#     # Convert to tensors and create DataLoaders
#     train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
#     val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

#     # Use sampler for train_loader
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    
#     # Validation loader doesn't need weighted sampling
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

#     return train_loader, val_loader
#     # # Convert to tensors and create DataLoaders
#     # train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
#     # val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
#     # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
#     # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

#     # return train_loader, val_loader

# # Function to create a test DataLoader
# def get_test_loader(batch_size=128, shuffle=True, num_workers=4, pin_memory=False):
#     # texts, labels = read_finsen_data('./data/FinSen/finsen.csv')  # Ensure this path is correct
#     texts, labels = read_finsen_data('./adafocal-main/data/FinSen/finsen.csv')

#     # Vectorize and split the dataset for test data
#     vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
#     features = vectorizer.fit_transform(texts).toarray()
#     _, X_test, _, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=True)

#     # Convert to tensors and create DataLoader
#     test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

#     return test_loader

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch

def create_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]
    weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return weighted_sampler

def read_finsen_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['Content'].tolist()
    labels = df['Category'].apply(lambda x: x.strip()).tolist()  # Ensure labels are stripped of whitespace
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    return texts, labels, label_encoder.classes_

def prepare_data_loaders(texts, labels, batch_size, valid_size, random_seed, shuffle, num_workers, pin_memory):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    features = vectorizer.fit_transform(texts).toarray()
    X_train, X_val_test, y_train, y_val_test = train_test_split(features, labels, test_size=valid_size + 0.1, random_state=random_seed)  # Adjust valid_size + test_size
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=random_seed)  # Split remaining data equally for val and test

    train_sampler = create_weighted_sampler(y_train)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader

def get_data_loaders(finsen_data_path, batch_size=128, valid_size=0.2, random_seed=42, shuffle=True, num_workers=4, pin_memory=False):
    texts, labels, _ = read_finsen_data(finsen_data_path)
    return prepare_data_loaders(texts, labels, batch_size, valid_size, random_seed, shuffle, num_workers, pin_memory)

    # Now, train_loader, val_loader, and test_loader can be used in training and evaluation.
