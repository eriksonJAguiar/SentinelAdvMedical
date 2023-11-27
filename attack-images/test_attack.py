import pytest
import load_dataset


def test_load_attacked_dataset():
    dataset_name = "melanoma"
    root_path = "./"
    csv_path = "/home/eriksonaguiar/codes/OOD-Detection/dataset/MelanomaDB/ham1000_dataset.csv"
    
    val_attack_dataset, num_class = load_dataset.load_attacked_database_df(root_path=root_path, csv_path=csv_path, batch_size=32, image_size=(224,224), percentage_attacked=0.2, test_size=0.3)

    load_dataset.show_images(val_attack_dataset, dataset_name)

    assert val_attack_dataset is not None