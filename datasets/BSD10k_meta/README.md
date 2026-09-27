# BSD10k (Broad Sound Dataset 10k) – v1.2

The **BSD10k dataset** (Broad Sound Dataset 10k) is an open collection of human-labeled sounds containing over 10k [Freesound](https://freesound.org/) audio clips, annotated according to the 23 second-level classes defined in the **Broad Sound Taxonomy** (BST). BST is currently being [used in Freesound](https://freesound.org/help/faq/#the-broad-sound-taxonomy) for organization, filtering, and post-processing tasks. The dataset was created at the [Music Technology Group](https://www.upf.edu/web/mtg) of Universitat Pompeu Fabra. The current version of the dataset is v1.2.


### Dataset characteristics

This document describes the third version of the dataset, **BSD10k v1.2**, which includes updates and additional sounds over the original BSD10k (v1.0) dataset. The dataset consists of 10,956 sounds from [Freesound](https://freesound.org/), totaling 35.25 hours of single-labeled audio. The sounds are cropped to a maximum length of 30 seconds, resulting in variable durations ranging from ~0.01 to 30s. Audio lengths vary due to the heterogeneity of the sound classes and the range of contributions from Freesound users. The original files downloaded from Freesound are converted to a standardized format of uncompressed WAV files with 44.1 kHz sampling rate, 16-bit depth, and mono channel. The dataset’s audio files occupy approximately 11.2 GB when unzipped and can be found in the `audio` folder.

All sounds have been manually labeled by human annotators and categorized into 23 classes, which are the second-level categories of the Broad Sound Taxonomy (see details below). The annotated data has a non-uniform distribution among the top-level and their second-level classes. The detailed class distribution of this dataset version is presented below. In this listing, classes are ordered by number of instances.

* Sound effects (fx): 3989 (12.47 hours)
    * Objects / House appliances (fx-o): 1211 (2.75 hours)
    * Human sounds and actions (fx-h): 975 (1.96 hours)
    * Natural elements and explosions (fx-n): 652 (3.31 hours)
    * Experimental (fx-ex): 297 (0.74 hours)
    * Other mechanisms, engines, machines (fx-m): 279 (1.19 hours)
    * Electronic / Design (fx-el): 208 (0.23 hours)
    * Vehicles (fx-v): 201 (1.31 hours)
    * Animals (fx-a): 166 (0.99 hours)
* Instrument samples (is): 2359 (3.16 hours)
    * Percussion (is-p): 606 (0.54 hours)
    * Wind (is-w): 566 (0.74 hours)
    * String (is-s): 528 (0.70 hours)
    * Piano / Keyboard instruments (is-k): 362 (0.83 hours)
    * Synths / Electronic (is-e): 297 (0.34 hours)
* Music (m): 1727 (5.36 hours)
    * Solo instrument (m-si): 721 (2.15 hours)
    * Solo percussion (m-sp): 683 (1.73 hours)
    * Multiple instruments (m-m): 323 (1.48 hours)
* Soundscapes (ss): 1534 (11.20 hours)
    * Urban (ss-u): 719 (5.19 hours)
    * Nature (ss-n): 391 (2.96 hours)
    * Synthetic / Artificial (ss-s): 217 (1.58 hours)
    * Indoors (ss-i): 207 (1.48 hours)
* Speech (sp): 1347 (3.07 hours)
    * Solo speech (sp-s): 806 (0.93 hours)
    * Processed / Synthetic (sp-p): 364 (1.19 hours)
    * Conversation / Crowd (sp-c): 177 (0.95 hours)
    
For each audio file, the **current version** of the dataset (BSD10k v1.2) includes the following: the `category label` assigned during annotation and the annotator's `confidence score`, descriptive metadata (`title`, `tags`, `description`), and provenance information (`ID`, `uploader`, `license`), all provided in `BSD10k_metadata.csv`. We also provide precomputed audio and text embeddings to facilitate further analysis and reproducibility, located in the `features` folder. For more details on the dataset creation and its contents, please refer to our paper "*Heterogeneous Sound Classification with the Broad Sound Taxonomy and Dataset*", specifically Section 3.1. This version of the dataset (BSD10k v1.2) is first used in the [DCASE 2026 Challenge - Task 1](https://dcase.community/challenge2026/task-heterogeneous-audio-classification). An overview of the BSD10k dataset is also available on the [support site](https://github.com/allholy/BSD10k/).


### Taxonomy

The **Broad Sound Taxonomy (BST)** organizes sounds into a two-level hierarchical structure with 5 top-level and 23 second-level categories. The top-level categories cover distinct types of sounds: *Music*, *Instrument samples*, *Speech*, *Sound effects*, and *Soundscapes*. The taxonomy is designed to classify *any type* of sound while remaining broad, comprehensive, and easy to use. It can be used for *organizing and filtering sounds* in heterogeneous sound collections, such as Freesound, as well as in personal sound libraries. More details about the categories can be found in `BST_description.csv`, and additional information about the taxonomy is provided in the journal paper "*A General-Purpose Sound Taxonomy for the Classification of Heterogeneous Sound Collections*". The taxonomy release versions can be found in the [BST support site](https://github.com/MTG/BroadSoundTaxonomy).


### Citation

When using *all or part* of the BSD10k dataset, please cite our papers:

**Dataset creation (original version)** (available from [[UPF e-repositori](http://hdl.handle.net/10230/68432)] [[arXiv](https://arxiv.org/abs/2410.00980)] [[DCASE2024 proceedings](https://zenodo.org/records/13871309)]):

> ```bibtex
> @inproceedings{anastasopoulou2024heterogeneous,
>   title = {Heterogeneous Sound Classification with the {{Broad Sound Taxonomy}} and {{Dataset}}},
>   author = {Anastasopoulou, Panagiota and Torrey, Jessica and Serra, Xavier and Font, Frederic},
>   booktitle = {Proc. {{Workshop}} on {{Detection}} and {{Classification}} of {{Acoustic Scenes}} and {{Events}} ({{DCASE}})},
>   year = {2024}
> }
> ```

**Updated dataset version (v1.1)** (available from [[UPF e-repositori](http://hdl.handle.net/10230/71472)] [[DCASE2025 proceedings](https://zenodo.org/records/17251589)]):

> ```bibtex
> @inproceedings{anastasopoulou2025hierarchical,
>   title = {Hierarchical and Multimodal Learning for Heterogeneous Sound Classification},
>   author = {Anastasopoulou, Panagiota and Dal R{\'i}, Francesco Ardan and Serra, Xavier and Font, Frederic},
>   booktitle = {Proc. {{Workshop}} on {{Detection}} and {{Classification}} of {{Acoustic Scenes}} and {{Events}} ({{DCASE}})},
>   year = {2025}
> }
> ```

Note: Since BSD10k-v1.2 is first used as part of the [DCASE 2026 Challenge - Task 1](https://dcase.community/challenge2026/task-heterogeneous-audio-classification) and is not yet tied to an academic paper, we recommend citing the original dataset paper and specifying the version (v1.2) in your methods section.


### License

BSD10k is released in its entirety under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). We note, though, that each audio file is released under its own Creative Commons (CC) license, as defined by the respective uploader in Freesound. Some sounds require attribution to their original authors, while others forbid commercial reuse. If the dataset is used in a commercial setting, the sounds with CC BY-NC licenses should be excluded.

This is the distribution of sounds per license:

- CC0: 3,334
- CC BY: 5,970
- CC BY-NC: 1,240
- CC Sampling+: 412

Links to the *license deeds* for each sound can be further accessed through `BSD10k_metadata.csv`.


### Data structure

BSD10k can be accessed as follows:
<div class="highlight"><pre><span></span>root/
├── audio/                     Audio files
├── metadata/                  Metadata files
│   ├── BSD10k_metadata.csv        Dataset's metadata
│   ├── BST_description.csv        Taxonomy information
│   └── BST_diagram.png            Taxonomy diagram
├── features/                  Precomputed embeddings
│   ├── clap_audio_embeddings      Audio embeddings
│   └── clap_text_embeddings       Text embeddings
└── README.md                  Documentation (that you are now reading)
</pre></div>


`BSD10k_metadata.csv` is the main metadata file, containing annotations and additional information for each sound. Each row corresponds to one sound and includes the following fields:

- `sound_id`: Freesound ID used as the unique identifier of the sound. The audio files found in the `audio` folder are named using this ID, with a .wav extension for the audio format.
- `class`: Second-level class code of the sound.
- `class_idx`: Class index of 3 digits, where the first digit corresponds to the index of the top-level class, and the last 2 digits to the index of the second-level class ('00' denotes a top-level class), both ordered according to the taxonomy.
- `class_top`: Corresponding top-level class code. It is derived from the full (second-level) class code by taking the part before the hyphen (-).
- `confidence`: Annotator's confidence score assigned to each sound during the annotation process. It ranges from 1 (very unconfident) to 5 (very confident).
- `uploader`: User who uploaded the sound in Freesound.
- `license`: Link to the license of the sound.
- `title`: Sound title provided by the uploader.
- `tags`: Tags associated with the sound provided by the uploader.
- `description`: Description of the sound provided by the uploader.

The mapping of class codes to their corresponding *full class names* can be found in `BST_description.csv`, which also includes a description and examples for each class (minor ancillary updates from the last version). A diagram of the taxonomy (`BST_diagram.png`) is also included for a quick overview of the categories.

The `features` folder contains two subfolders with audio and text embeddings, both extracted using the `630k-audioset-fusion-best.pt` checkpoint of the [LAION-CLAP](https://github.com/LAION-AI/CLAP/) model. The text embeddings use all available textual descriptive metadata, including title, tags, and description.
 
 
### Versioning details

**v1.2 – 2026-04-08**  (current)

- Sound count: 10,956
- Total hours: 35.25
- Metadata fields: sound_id, class (code, index, top level), confidence, title, tags, description, uploader, license
- Features (precomputed): CLAP audio and text embeddings
- Notes: Corrected detected human labeling errors for improved consistency; updated category indices (added the class_index column to the taxonomy file and updated the class_index column in the metadata).

**v1.1 – 2025-10-14**

- Sound count: 10,956
- Total hours: 35.25
- Metadata fields: sound_id, class (code, index, top level), confidence, title, tags, description, uploader, license
- Features (precomputed): CLAP audio and text embeddings
- Notes: Added new sounds and corrected human labeling errors; included sound descriptions to the textual metadata, annotation confidence scores, and precomputed embeddings; minor updates in taxonomy file.

**v1.0 – 2024-07-11**

- Sound count: 10,309
- Total hours: 32.5
- Metadata fields: sound_id, class (code, index, top level), title, tags, uploader, license
- Notes: Initial version

### Acknowledgments

This research is partially funded by the Generalitat de Catalunya (2023FI-100252, Joan Oró program), the IA y Música Cátedra (TSI-100929-2023-1, Cátedras ENIA 2022, SE Digitalización e IA, EU NGEU), and the IMPA project (PID2023-152250OB-I00, MCIU, AEI, co-funded by EU).

 
### Contact

You are welcome to contact Panagiota Anastasopoulou if you have any questions, at panagiota.anastasopoulou@upf.edu.
