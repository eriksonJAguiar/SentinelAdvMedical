import pytest
import load_dataset
import generate_attacks

    
def test_generate_attack_resnet50_FGSM():
    
    dataset_name = "melanoma"
    root_path = "./"
    csv_path = "/home/eriksonaguiar/codes/OOD-Detection/dataset/MelanomaDB/ham1000_dataset.csv"
    
    val_attack_dataset, num_class = load_dataset.load_attacked_database_df(root_path=root_path, csv_path=csv_path, batch_size=32, image_size=(224,224), percentage_attacked=0.2, test_size=0.3)
    
    model_path = "/home/eriksonaguiar/codes/OOD-Detection/clean-train/trained-weights/resnet50-melanoma-exp0.ckpt"
    
    adv_images, true_labels = generate_attacks.generate_attack(
                        model_path=model_path,
                        model_name="resnet50",
                        input_shape=(224,224),
                        lr=0.0001,
                        nb_class=num_class,
                        attack_name="FGSM",
                        data_loader=val_attack_dataset,
                        eps=0.1
                    )
    
    load_dataset.show_random_adv_image(adv_images, dataset_name, "FGSM_resnet50")
    
    assert adv_images is not None

def test_generate_attack_densenet_FGSM():
    
    dataset_name = "melanoma"
    root_path = "./"
    csv_path = "/home/eriksonaguiar/codes/OOD-Detection/dataset/MelanomaDB/ham1000_dataset.csv"
    
    val_attack_dataset, num_class = load_dataset.load_attacked_database_df(root_path=root_path, csv_path=csv_path, batch_size=32, image_size=(224,224), percentage_attacked=0.2, test_size=0.3)
    
    model_path = "/home/eriksonaguiar/codes/OOD-Detection/clean-train/trained-weights/densenet-melanoma-exp0.ckpt"
    
    adv_images, true_labels = generate_attacks.generate_attack(
                        model_path=model_path,
                        model_name="densenet",
                        input_shape=(224,224),
                        lr=0.0001,
                        nb_class=num_class,
                        attack_name="FGSM",
                        data_loader=val_attack_dataset,
                        eps=0.1
                    )
    
    load_dataset.show_random_adv_image(adv_images, dataset_name, "FGSM_densenet")
    
    assert adv_images is not None
    
def test_generate_attack_efficientnet_FGSM():
    
    dataset_name = "melanoma"
    root_path = "./"
    csv_path = "/home/eriksonaguiar/codes/OOD-Detection/dataset/MelanomaDB/ham1000_dataset.csv"
    
    val_attack_dataset, num_class = load_dataset.load_attacked_database_df(root_path=root_path, csv_path=csv_path, batch_size=32, image_size=(224,224), percentage_attacked=0.2, test_size=0.3)
    
    model_path = "/home/eriksonaguiar/codes/OOD-Detection/clean-train/trained-weights/efficientnet-melanoma-exp0.ckpt"
    
    adv_images, true_labels = generate_attacks.generate_attack(
                        model_path=model_path,
                        model_name="efficientnet",
                        input_shape=(224,224),
                        lr=0.0001,
                        nb_class=num_class,
                        attack_name="FGSM",
                        data_loader=val_attack_dataset,
                        eps=0.1
                    )
    
    load_dataset.show_random_adv_image(adv_images, dataset_name, "FGSM_efficientnet")
    
    assert adv_images is not None

def test_generate_attack_inceptionv3_FGSM():
    
    dataset_name = "melanoma"
    root_path = "./"
    csv_path = "/home/eriksonaguiar/codes/OOD-Detection/dataset/MelanomaDB/ham1000_dataset.csv"
    
    val_attack_dataset, num_class = load_dataset.load_attacked_database_df(root_path=root_path, csv_path=csv_path, batch_size=32, image_size=(224,224), percentage_attacked=0.2, test_size=0.3)
    
    model_path = "/home/eriksonaguiar/codes/OOD-Detection/clean-train/trained-weights/inceptionv3-melanoma-exp0.ckpt"
    
    adv_images, true_labels = generate_attacks.generate_attack(
                        model_path=model_path,
                        model_name="inceptionv3",
                        input_shape=(224,224),
                        lr=0.0001,
                        nb_class=num_class,
                        attack_name="FGSM",
                        data_loader=val_attack_dataset,
                        eps=0.1
                    )
    
    load_dataset.show_random_adv_image(adv_images, dataset_name, "FGSM_inceptionv3")
    
    assert adv_images is not None

def test_generate_attack_vgg16_FGSM():
    
    dataset_name = "melanoma"
    root_path = "./"
    csv_path = "/home/eriksonaguiar/codes/OOD-Detection/dataset/MelanomaDB/ham1000_dataset.csv"
    
    val_attack_dataset, num_class = load_dataset.load_attacked_database_df(root_path=root_path, csv_path=csv_path, batch_size=32, image_size=(224,224), percentage_attacked=0.2, test_size=0.3)
    
    model_path = "/home/eriksonaguiar/codes/OOD-Detection/clean-train/trained-weights/vgg16-melanoma-exp0.ckpt"
    
    adv_images, true_labels = generate_attacks.generate_attack(
                        model_path=model_path,
                        model_name="vgg16",
                        input_shape=(224,224),
                        lr=0.0001,
                        nb_class=num_class,
                        attack_name="FGSM",
                        data_loader=val_attack_dataset,
                        eps=0.1
                    )
    
    load_dataset.show_random_adv_image(adv_images, dataset_name, "FGSM_vgg16")
    
    assert adv_images is not None

def test_generate_attack_vgg19_FGSM():
    
    dataset_name = "melanoma"
    root_path = "./"
    csv_path = "/home/eriksonaguiar/codes/OOD-Detection/dataset/MelanomaDB/ham1000_dataset.csv"
    
    val_attack_dataset, num_class = load_dataset.load_attacked_database_df(root_path=root_path, csv_path=csv_path, batch_size=32, image_size=(224,224), percentage_attacked=0.2, test_size=0.3)
    
    model_path = "/home/eriksonaguiar/codes/OOD-Detection/clean-train/trained-weights/vgg19-melanoma-exp0.ckpt"
    
    adv_images, true_labels = generate_attacks.generate_attack(
                        model_path=model_path,
                        model_name="vgg19",
                        input_shape=(224,224),
                        lr=0.0001,
                        nb_class=num_class,
                        attack_name="FGSM",
                        data_loader=val_attack_dataset,
                        eps=0.1
                    )
    
    load_dataset.show_random_adv_image(adv_images, dataset_name, "FGSM_vgg19")
    
    assert adv_images is not None