# SentinelAdvMedical

**Title:** SentinelAdvMedical: toward adversarial attacks detection on medical image classification via Out-Of-Distribution strategies.
**Author(s):** Erikson Júlio de Aguiar, Univ. de São Paulo (Brazil), Univ. of Florida (United States); Agma Juci Juci Machado Traina, Univ. de São Paulo (Brazil); Abdelsalam Helal, Univ. of Florida (United States), Univ. degli Studi di Bologna (Italy)

## Abstract

Deep Learning (DL) comprehends methods to enhance medical image classification and help physicians speed up diagnosis. However, these methods present security issues and are vulnerable to adversarial attacks that result in the model’s misclassification, presenting severe consequences in the medical field. We propose SentinelAdvMedical, a novel pipeline to detect adversarial attacks by employing controlled Out-of-Distributions (OOD) strategies to enhance the “immunity” of DL models. Towards that end, we studied the classification of Optical Coherence Tomography (OCT) images of Skin lesions with ResNet50. Our findings show that the best OOD detectors for OCT and Skin Lesion datasets are MaxLogits and Entropy, which outperform baselines Maximum Softmax Probabilities (MSP) and Mahalanobis feature-based score. To conduct this study, we developed a novel pipeline and studied the application of OOD strategies against adversarial examples, aiming to detect them and provide security specialists with a path to check possible attacked spots in medical datasets employing the best OOD detectors in these settings.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To use SentinelAdvMedical, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/erjulioaguiar/SentinelAdvMedical.git
    ```
2. Navigate to the project directory:
    ```bash
    cd SentinelAdvMedical
    ```
3. Run the main script:
    ```bash
    python run_experiments.py --dataset_name 'oct/melanoma' --dataset_csv 'path_to_dataset' --weights_path 'weights_path' --nb_class '4/7'
    ```

<!-- ## Citation

If you use this code or our method in your research, please cite our paper:

```
@article{SentinelAdvMedical2025,
  title={SentinelAdvMedical: toward adversarial attacks detection on medical image classification via Out-Of-Distribution strategies},
  author={Erjulio Aguiar and others},
  journal={SPIE Medical Imaging: Computer-Aided Diagnosis},
  year={2025},
} -->
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We would like to thank our collaborators and the funding agencies for their support.

For more information, please email to [Erjulio Aguiar](mailto:erjulioaguiar@usp.br) or text me on [LinkedIn]().