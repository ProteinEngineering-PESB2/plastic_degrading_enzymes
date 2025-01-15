# Discovering potential plastic degrading enzymes using machine learning strategies

This repository contains the source files and supplementary information for the implementations and use cases presented in the work:


David Medina-Ortiz<sup>1,2</sup>, Diego Alvares-Saravia<sup>2,3</sup>, Nicole Soto-García<sup>2</sup>, Diego Sandoval-Vargas<sup>1,4</sup>, Jacqueline Aldridge<sup>2</sup>, Sebastián Rodríguez<sup>1,4</sup>, Bárbara Andrews<sup>1,4</sup>, Juan A. Asenjo<sup>1,4</sup> Anamaría Daza<sup>1,4∗</sup><br>
Discovering potential plastic degrading enzymes using machine learning strategies. <br>
https://doi.org/XXXX<br>

<sup>*1*</sup><sub>Centre for Biotechnology and Bioengineering, CeBiB, Universidad de Chile, Beauchef 851, Santiago, Chile</sub> <br>
<sup>*2*</sup><sub>Departamento de Ingeniería En Computación, Universidad de Magallanes, Avenida Bulnes 01855, Punta Arenas, Chile.</sub> <br>
<sup>*2*</sup><sub>Centro Asistencial de Docencia e Investigación, CADI, Universidad de Magallanes. Av. Los Flamencos 01364, Punta Arenas, Chile.</sub> <br>
<sup>*4*</sup><sub>Departamento de Ingeniería Química, Biotecnología y Materiales, Universidad de Chile, Beauchef 851, Santiago, Chile</sub> <br>
<sup>*\**</sup><sub>Corresponding author</sub> <br>

---
## Table of Contents
- [A summary of the proposed work](#summary)
- [Requirements and instalation](#requirements)
- [Implemented pipeline](#pipeline)
- [Raw data and preprocessing](#data)
- [Numerical representation strategies](#numerical)
- [Training, selecting, and generating models](#training)
- [Using models](#using)
- [References](#references)
---

<a name="summary"></a>

## Discovering potential plastic degrading enzymes using machine learning strategies

Plastic pollution presents a critical environmental challenge, necessitating innovative and sustainable solutions. In this context, biodegradation using microorganisms and enzymes offers an environmentally friendly alternative. This work introduces an AI-driven framework that integrates machine learning (ML) and generative models to accelerate the discovery and design of plastic-degrading enzymes. By leveraging pre-trained protein language models and curated datasets, we developed seven ML-based binary classification models to identify enzymes targeting specific plastic substrates, achieving an average accuracy of 89\%. The framework was applied to over 6,000 enzyme sequences from the RemeDB to classify enzymes targeting diverse plastics, including PET, PLA, and Nylon. Besides, generative learning strategies combined with trained classification models in this work were applied for \textit{de novo} generation of PET-degrading enzymes. Structural bioinformatics validated potential candidates through \textit{in-silico} analysis, highlighting differences in physicochemical properties between generated and experimentally validated enzymes. Moreover, generated sequences exhibited lower molecular weights and higher aliphatic indices, features that may enhance interactions with hydrophobic plastic substrates. These findings highlight the utility of AI-based approaches in enzyme discovery, providing a scalable and efficient tool for addressing plastic pollution. Future work will focus on experimental validation of promising candidates and further refinement of generative strategies to optimize enzymatic performance. 

<a name="requirements"></a>

## Requirements

<a name="pipeline"></a>

## Implemented pipeline to train classification models

<a name="data"></a>

## Raw data and processing strategies

The folder [raw_data](raw_data) contains:
- **enzymes_plastics**: pivoted csv file with each plastic-substrate target detected 
- **data_sequences**: csv file with unique sequences collected from the pivoted dataset
- **reme_db_sequences**: csv file with plastic-degrading enzymes extracted from RemeDB [2]

<a name="numerical"></a>

## Numerical representation strategies explored in this work

This work explores different numerical representation strategies to process the input enzyme sequences. See the notebooks in folder [encoding_approaches](src/encoding_approaches) for details about the execution.

The encoding strategies including:

1. Feature engineering
2. Frequency encoders
3. k-mers encoders
4. One-hot encoders
5. Ordinal encoders
6. Physicochemical-based encoders
7. FFT-based encoders
8. Embeddings throug pre-trained protein language models.

- In the case of physicochemical-based and FFT-based the physicochemical properties encoders used were extracted from [3]. 

- In the case of embedding, the bioembedding library was employed [4].

The numerical representation strategies take the input data [input_data](raw_data/data_sequences.csv), apply the encoder strategy, and generate the outputs with the encoder sequences. The folder [processed_dataset](processed_dataset) contains the results of all encoder strategies explored in this work.

<a name="training"></a>

## Training strategies applied to develop classification models

<a name="using"></a>

## Using the models trained with new enzyme sequences

<a name="references"> </a>

## References

- [1] Dallago, C., Schütze, K., Heinzinger, M., Olenyi, T., Littmann, M., Lu, A. X., ... & Rost, B. (2021). Learned embeddings from deep learning to visualize and predict protein sets. Current Protocols, 1(5), e113.
- [2] Sankara Subramanian, S. H., Balachandran, K. R. S., Rangamaran, V. R., & Gopal, D. (2020). RemeDB: tool for rapid prediction of enzymes involved in bioremediation from high-throughput metagenome data sets. Journal of Computational Biology, 27(7), 1020-1029.
- [3] 
- [4]