# Dataset of parallel Mixtec-Spanish texts for the development of language technologies
This repository contains a database of Spanish texts translated into Mixtec. The dataset is a very valuable resource for developing language technologies in the Mixtec language, automatic translation, text tagging, or summaries. The texts identified in the corpus are Laws, Religion, Health, and Educational. The parallel corpus contains 14,587 pairs of Spanish-Mixtec sentences.

This repository contains fine-tuning notebooks of the mBART-50, M2M100, and gpt-4o-mini-2024-07-18 models. These models were fine-tuned with the Mixtec data to test their performance and use in the development of language technologies. After fine-tuning, the new models were evaluated with BLEU and TER metrics.

## Intallation
1. Clone the repository
```bash
https://github.com/hermilocap/ParalellCorpusMixtec-Spanish.git
```
2. Navigate to the project directory
```bash
cd ParalellCorpusMixtec-Spanish
cd notebooks
```  
2. Generate the environment using <br />
```bash
python -m venv env
```
Activate env. If you work on Windows
```bash
env\Scripts\activate.ps1
```
or
```bash
env\Scripts\activate
```
