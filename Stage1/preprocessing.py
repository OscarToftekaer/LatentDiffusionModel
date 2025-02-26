import json
import random
from collections import defaultdict
import torchio as tio
import torch
import nibabel as nib
import numpy as np
import os

############## FOR SPLITTING THE DATASET BASED ON PATIENTS ############################
def split_patient_sessions(json_path):
    # Load the data from the JSON file
    with open(json_path, 'r') as f:
        data_dict = json.load(f)

    # Group sessions by patient
    patient_sessions = defaultdict(list)
    for key, value in data_dict.items():
        patient_id = key.split('/')[0]
        #patient_sessions[patient_id].append({'seg': value['seg'], 'pet': value['pet']})
        patient_sessions[patient_id].append({'seg': value['seg'], 'pet': value['pet']})
    # Shuffle patients
    patients = list(patient_sessions.keys())
    random.shuffle(patients)

    # Split patients into train (80%), validation (10%), and test (10%)
    total_patients = len(patients)
    train_split = int(0.8 * total_patients)
    val_split = int(0.9 * total_patients)

    train_patients = patients[:train_split]
    val_patients = patients[train_split:val_split]
    test_patients = patients[val_split:]

    # Collect the session CT and PET paths for each group
    train_list = [session for patient in train_patients for session in patient_sessions[patient]]
    val_list = [session for patient in val_patients for session in patient_sessions[patient]]
    test_list = [session for patient in test_patients for session in patient_sessions[patient]]

    return train_list, val_list, test_list



def create_subjects_dataset(paths_list, transforms=None):
    subjects = []
    for paths in paths_list:
        subject = tio.Subject(
            pet=tio.ScalarImage(paths['pet']),
            label=tio.LabelMap(paths['seg']),
        )
        
        subjects.append(subject)
    dataset = tio.SubjectsDataset(subjects, transform=transforms)
    return dataset



def create_queue(dataset, patch_size, probabilities_dict, max_length=4, samples_per_volume=2, shuffle_subjects=True, shuffle_patches=True):
    label_sampler = tio.LabelSampler(
        patch_size=patch_size,
        label_name='label',
        label_probabilities=probabilities_dict
    )
    queue = tio.Queue(
        subjects_dataset=dataset,
        max_length=max_length,
        samples_per_volume=samples_per_volume,
        sampler=label_sampler,
        num_workers=6,
        shuffle_subjects=shuffle_subjects,
        shuffle_patches=shuffle_patches
    )
    return queue


def create_transforms(patch_size3d_train=(16, 16, 16), patch_size3d_val=(16, 16, 16)):
    # Calculate padding sizes based on the patch sizes
    padding_size_train = tuple(dim // 2 for dim in patch_size3d_train)
    padding_size_val = tuple(dim // 2 for dim in patch_size3d_val)

    # Define training transforms
    training_transforms = tio.Compose([
        tio.Pad(padding_size_train, padding_mode=-1, include=['pet']),  # Pad 'pet' with -1
        tio.Pad(padding_size_train, padding_mode=0, include=['label']),  # Pad 'label' with 0
    ])

    # Define validation transforms
    validation_transforms = tio.Compose([
        tio.Pad(padding_size_val, padding_mode=-1, include=['pet']),  # Pad 'pet' with -1
        tio.Pad(padding_size_val, padding_mode=0, include=['label']),  # Pad 'label' with 0
    ])

    return training_transforms, validation_transforms




def create_queue_h2o(dataset, patch_size, probabilities_dict, max_length=4, samples_per_volume=2, shuffle_subjects=True, shuffle_patches=True):
    label_sampler = tio.LabelSampler(
        patch_size=patch_size,
        label_name='label',
        label_probabilities=probabilities_dict
    )
    queue = tio.Queue(
        subjects_dataset=dataset,
        max_length=max_length,
        samples_per_volume=samples_per_volume,
        sampler=label_sampler,
        num_workers=6,
        shuffle_subjects=shuffle_subjects,
        shuffle_patches=shuffle_patches
    )
    return queue


def create_transforms_h2o(patch_size3d_train=(16, 16, 16), patch_size3d_val=(16, 16, 16)):
    # Calculate padding sizes based on the patch sizes
    padding_size_train = tuple(dim // 2 for dim in patch_size3d_train)
    padding_size_val = tuple(dim // 2 for dim in patch_size3d_val)

    # Define training transforms
    training_transforms = tio.Compose([
        tio.Pad(padding_size_train, padding_mode=-1, include=['pet']),  # Pad 'pet' with -1
        tio.Pad(padding_size_train, padding_mode=0, include=['label']),  # Pad 'label' with 0
        tio.RandomAffine(
            scales=(0.9, 1.1),  # Random scaling between -10% and +10%
            degrees=(10, 10, 10),  # Random rotation between -10 and +10 degrees
            include=['pet', 'label']  # Apply to both 'pet' and 'label'
        ),
    ])

    # Define validation transforms (no augmentation)
    validation_transforms = tio.Compose([
        tio.Pad(padding_size_val, padding_mode=-1, include=['pet']),  # Pad 'pet' with -1
        tio.Pad(padding_size_val, padding_mode=0, include=['label']),  # Pad 'label' with 0
    ])

    return training_transforms, validation_transforms
