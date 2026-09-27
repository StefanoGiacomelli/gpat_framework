# BSD35k-CS (Broad Sound Dataset 35k - Crowd Sourced)

The **BSD35k-CS dataset** (Broad Sound Dataset 35k - Crowd Sourced) is an open collection of human-labeled sounds containing about 35k [Freesound](https://freesound.org/) audio clips, annotated according to the second-level classes defined in the **Broad Sound Taxonomy** (BST). BST is currently being [used in Freesound](https://freesound.org/help/faq/#the-broad-sound-taxonomy) for organization, filtering, and post-processing tasks. The dataset was created at the [Music Technology Group](https://www.upf.edu/web/mtg) of Universitat Pompeu Fabra. This dataset is a complement to the [already existing BSD10k dataset](https://zenodo.org/records/17250001), and triples the size of it. However, unlike BSD10k in which sounds were annotated by experts, the **annotations of BSD35k-CS come from the authors of the sounds themselves,** who uploaded the sounds to Freesound and selected a BST category for them. That is why we call it _Crowd Sourced_. It should therefore be expected that the annotations in BSD35k-CS are **more noisy and less coherent than in BSD10k**. All the sounds included in BSD35k-CS were uploaded in Freesound between April 1st 2025 and Janurary 27th 2026 (both included).

This dataset has been prepared by Panagiota Anastasopoulou and Frederic Font at the [Music Technology Group](https://upf.edu/web/mtg) of [Universitat Pompeu Fabra](https://upf.edu). Feel free to contact us at panagiota.anastasopoulou@upf.edu and frederic.font@upf.edu if you have any questions.

### **Dataset characteristics**

The dataset consists of 33,829 sounds from [Freesound](https://freesound.org/), totaling 152 hours of single-labeled audio. The sounds are cropped to a maximum length of 30 seconds, resulting in variable durations ranging from 7ms to 30s. Audio lengths vary due to the heterogeneity of the sound classes and the range of contributions from Freesound users. The original files downloaded from Freesound are converted to a standardized format of uncompressed WAV files with 44.1 kHz sampling rate, 16-bit depth, and mono channel. The dataset’s audio files occupy approximately 45 GB when unzipped.

All the sounds included in BSD35k-CS were uploaded in Freesound between April 1st 2025 and Janurary 27th 2026 (both included), and were categorized by their creators into one of the classes available at the second-level of the Broad Sound Taxonomy (see details below). Each audio file includes the following metadata: the _category label_ assigned at upload time by the Freesound users who created the sound, additional descriptive metadata also provided by the author (_title_, _tags_, _description_), and provenance information (_Freesound ID_, _uploader_, _license_). Further details about metadata are provided below. BSD35k-CS features the class distribution portraied below. Note how the distribution is highly unbalanced. In this listing, classes are ordered by number of instances.

*   Sound effects (fx): 14,722 (46.16 hours)
    *   Objects / House appliances (fx-o): 4,534
    *   Electronic / Design (fx-el): 2,000 (5.08 hours)
    *   Vehicles (fx-v): 1,852 (4.47 hours)
    *   Other mechanisms, engines, machines (fx-m): 1,542 (4.52 hours)
    *   Human sounds and actions (fx-h): 1,423 (4.76 hours)
    *   Other (fx-other): 1,143 (3.29 hours)
    *   Experimental (fx-ex): 839 (2.60 hours)
    *   Animals (fx-a): 730 (3.29 hours)
    *   Natural elements and explosions (fx-n): 659 (3.63 hours)
*   Soundscapes (ss): 8,844 (65.57 hours)
    *   Nature (ss-n): 4,831 (39.32 hours)
    *   Urban (ss-u): 2,573 (16.42 hours)
    *   Synthetic / Artificial (ss-s): 596 (3.84 hours)
    *   Other (ss-other): 461 (3.16 hours)
    *   Indoors (ss-i): 383 (2.83 hours)
*   Music (m): 4,885 (27.55 hours)
    *   Multiple instruments (m-m): 2,044 (15.25 hours)
    *   Solo instrument (m-si): 1,344 (6.80 hours)
    *   Solo percussion (m-sp): 964 (2.89 hours)
    *   Other (m-other): 533 (2.60 hours)
*   Instrument samples (is): 4,171 (10.15 hours)
    *   Percussion (is-p): 2,321 (7.42 hours)
    *   Synths / Electronic (is-e): 1,360 (1.56 hours)
    *   String (is-s): 293 (0.45 hours)
    *   Other (is-other): 120 (0.42 hours)
    *   Piano / Keyboard instruments (is-k): 43 (0.16 hours)
    *   Wind (is-w): 34 (0.14 hours)
*   Speech (sp): 1,207 (2.95 hours)
    *   Solo speech (sp-s): 980 (1.85 hours)
    *   Other (sp-other): 108 (0.50 hours)
    *   Processed / Synthetic (sp-p): 69 (0.27 hours)
    *   Conversation / Crowd (sp-c): 50 (0.33 hours)

In addition to textual metadata, we also provide precomputed audio-based and text-based [LAION-CLAP](https://github.com/LAION-AI/CLAP/) embeddings to facilitate further experiments, analysis and reproducibility. The text-based embeddings were extracting using a combination of existing textual descriptive metadata (including title, tags, and description) as input. Both types of embeddings were computed using the `630k-audioset-fusion-best.pt` checkpoint.

### **Taxonomy**

The **Broad Sound Taxonomy (BST)** organizes sounds into a two-level hierarchical structure with 5 top-level and 23 second-level categories. The top-level categories cover distinct types of sounds: _Music_, _Instrument samples_, _Speech_, _Sound effects_, and _Soundscapes_. The taxonomy is designed to classify _any type_ of sound while remaining broad, comprehensive, and easy to use. It can be used for _organizing and filtering sounds_ in heterogeneous sound collections, such as Freesound, as well as in personal sound libraries. More details about the categories can be found in `BST_description.csv`, and additional information about the taxonomy is provided in the journal paper "_A General-Purpose Sound Taxonomy for the Classification of Heterogeneous Sound Collections_". 

### **Citation**

When using _all or part_ of the BSD35k-CS dataset, please cite the corresponding [Zenodo page](https://zenodo.org/uploads/19187100): 

> @dataset{anastasopoulou\_2026\_19187100,  
>   author = {Anastasopoulou, Panagiota and Font Corbera, Frederic},  
>   title  = {BSD35k-CS (Broad Sound Dataset 35k - Crowd Sourced)},  
>   month = mar,  
>   year = 2026,  
>   publisher = {Zenodo},  
>   doi = {10.5281/zenodo.19187100},  
>   url = {https://doi.org/10.5281/zenodo.19187100},  
> }

### **License**

BSD35k-CS is released in its entirety under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). We note, though, that each audio file is released under its own Creative Commons (CC) license, as defined by the respective uploader in Freesound. Some sounds require attribution to their original authors, while others forbid commercial reuse. **If the dataset is used in a commercial setting, the sounds with CC BY-NC licenses should be excluded**.

This is the distribution of sounds per license:

*   CC0: 25,035
*   CC BY: 6,811
*   CC BY-NC: 1,983

Links to the _license deeds_ for each sound can be further accessed through `BSD350k-CS_metadata.csv`.

### **Data structure**

BSD35k-CS can be accessed as follows:

```
𝚛𝚘𝚘𝚝/  
├── 𝚊𝚞𝚍𝚒𝚘/                       𝙰𝚞𝚍𝚒𝚘 𝚏𝚒𝚕𝚎𝚜  
├── 𝚖𝚎𝚝𝚊𝚍𝚊𝚝𝚊/                     𝙼𝚎𝚝𝚊𝚍𝚊𝚝𝚊 𝚏𝚒𝚕𝚎𝚜  
│  ├── 𝙱𝚂𝙳35k-CS_𝚖𝚎𝚝𝚊𝚍𝚊𝚝𝚊.𝚌𝚜𝚟     𝙳𝚊𝚝𝚊𝚜𝚎𝚝'𝚜 𝚖𝚎𝚝𝚊𝚍𝚊𝚝𝚊  
│  ├── 𝙱𝚂𝚃_𝚍𝚎𝚜𝚌𝚛𝚒𝚙𝚝𝚒𝚘𝚗.𝚌𝚜𝚟        𝚃𝚊𝚡𝚘𝚗𝚘𝚖𝚢 𝚒𝚗𝚏𝚘𝚛𝚖𝚊𝚝𝚒𝚘𝚗  
│  └── 𝙱𝚂𝚃_𝚍𝚒𝚊𝚐𝚛𝚊𝚖.𝚙𝚗𝚐            𝚃𝚊𝚡𝚘𝚗𝚘𝚖𝚢 𝚍𝚒𝚊𝚐𝚛𝚊𝚖  
├── 𝚏𝚎𝚊𝚝𝚞𝚛𝚎𝚜/                     𝙿𝚛𝚎𝚌𝚘𝚖𝚙𝚞𝚝𝚎𝚍 𝚎𝚖𝚋𝚎𝚍𝚍𝚒𝚗𝚐𝚜  
│  ├── 𝚌𝚕𝚊𝚙_𝚊𝚞𝚍𝚒𝚘_𝚎𝚖𝚋𝚎𝚍𝚍𝚒𝚗𝚐𝚜       𝙰𝚞𝚍𝚒𝚘 𝚎𝚖𝚋𝚎𝚍𝚍𝚒𝚗𝚐𝚜  
│  └── 𝚌𝚕𝚊𝚙_𝚝𝚎𝚡𝚝_𝚎𝚖𝚋𝚎𝚍𝚍𝚒𝚗𝚐𝚜        𝚃𝚎𝚡𝚝 𝚎𝚖𝚋𝚎𝚍𝚍𝚒𝚗𝚐𝚜  
└── 𝚁𝙴𝙰𝙳𝙼𝙴.𝚖𝚍                    𝙳𝚘𝚌𝚞𝚖𝚎𝚗𝚝𝚊𝚝𝚒𝚘𝚗 (𝚝𝚑𝚊𝚝 𝚢𝚘𝚞 𝚊𝚛𝚎 𝚗𝚘𝚠 𝚛𝚎𝚊𝚍𝚒𝚗𝚐)
```

`BSD35k-CS_metadata.csv` is the main metadata file, containing annotations and additional information for each sound. Each row corresponds to one sound and includes the following fields:

*   `sound_id`: Freesound ID used as the unique identifier of the sound. The audio files found in the `audio`folder are named using this ID, with a .wav extension for the audio format.
*   `class`: Second-level class code of the sound.
*   `class_idx`: Class index of 3 digits, with the first digit corresponding to the index of top level class, and the 2 last digits to the indix of second level class, both ordered according to the taxonomy.
*   `class_top`: Corresponding top-level class code. It is derived from the full (second-level) class code by taking the part before the hyphen (-).
*   `confidence`: This column is empty in this dataset. It is kept as part of the metadata file for data consistency with BSD10k.
*   `uploader`: User who uploaded the sound in Freesound.
*   `license`: Link to the license of the sound.
*   `title`: Sound title provided by the uploader.
*   `tags`: Tags associated with the sound provided by the uploader.
*   `description`: Description of the sound provided by the uploader.

The mapping of class codes and class idx to their corresponding _full class names_ can be found in `BST_description.csv`, which also includes a description and examples for each class. A diagram of the taxonomy (`BST_diagram.png`) is also included for a quick overview of the categories.

The `features` folder contains two subfolders with the aforementioned audio and text embeddings respectively.


### Acknowledgments

This research is partially funded by the Generalitat de Catalunya (2023FI-100252, Joan Oró program), the IA y Música Cátedra (TSI-100929-2023-1, Cátedras ENIA 2022, SE Digitalización e IA, EU NGEU), and the IMPA project (PID2023-152250OB-I00, MCIU, AEI, co-funded by EU).

