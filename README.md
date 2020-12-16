# Sentiment Analysis of News Articles Towards Different German Parties

## Team Members
Kim-Celine Kahl  
ei260@stud.uni-heidelberg.de  

Miguel Heidegger  
tf268@stud.uni-heidelberg.de

Sophia Matthis  
cq270@stud.uni-heidelberg.de

## Setup

1. [Install](https://pipenv.pypa.io/en/latest/#install-pipenv-today) ```pipenv```. You might want to set ```export PIPENV_VENV_IN_PROJECT=1``` in your ```.bashrc/.zshrc``` for local virtual environments. Thereby you are making sure that all dependencies for your application are stored in the same directory under the `.venv` folder.  

2. Clone repository into preferred directory (or simply download the source code and rename the folder as you like): `git clone https://github.com/yakuhzi/news-analysis`  

3. Install packages: `cd news-analysis && pipenv install --dev`  

4. Init ```.env``` file. Use this file to store all your environment variables, such as credentials or encryption phrases. This file should never be added to your public repository but should always stay local.

5. Activate virtual environment: `pipenv shell`  

6. Test setup: `pipenv run main`  

7. Install Git hooks: `pre-commit install`


## Testing

A script to automatically execute tests is already defined in the project's `Pipfile`. Therefore you can simply run: `pipenv run test`
To generate a report on code coverage alongside run: `pipenv run test && pipenv run report`

## Results for the milestone

### Feedback from the first meeting with mentor
The key points from the first meeting with our mentor were that overall the project idea is good. However,
in our initial report we did not consider the fact that also different topics of the articles might affect the sentiment,
e.g. if a news paper writes a lot about renewable energies, car lobby, etc.

This is why we also want to consider the topics for the sentiment analysis.

Also, through the discussion we gained some information about which data to use. There is already an existing data set
from the university which is provided for this project. Details about the data are described in the section 
[Data Sources](#data-sources).

### Data Sources 

### NER Tagging 
As one of the first parts of the project, NER tagging was performed in order to find political parties and members of
the parties to identify relevant articles in the data set. For NER tagging, spacy is used with the 
[de_core_news_lg language model](https://spacy.io/models/de#de_core_news_lg), which has the best score for NER tagging
compared to the other German language models provided by spacy.

The results of the NER tagging are stored in JSON files and Pandas dataframes, for each news agencies one 
(naming JSON e.g. "src/data/bild_ner.json", naming Pandas datframe variable e.g. df_bild_ner). 

Structure of JSON NER files:

```json
[
  {
    "persons": [
      "Person 1 in text",
      "Person 2 in text"
    ],
    "organizations": [
      "Organization 1 in text",
      "Organization 2 in text"
    ]
  }  
]
```

Structure of data frames:

| index of article | persons       | organizations  |
| ---------------- |:-------------:| :-----:         |
| 0                | ["Person 1 in text", "Person 2 in text"] | ["Organization 1 in text",  "Organization 2 in text"] |
| 1                | ...  |   ...  |

Results of the NER tagging:

To get a first insight how good the NER tagging performs, the first 100 articles of each agency were tagged.
Overall, the NER tagging performs quite okay, especially it is good in recognizing different parties.

Here is one example of the Tagesschau data set (article at index 4):

![Screenshot](figures/NER_tagging.PNG)

* Organizations
    * Many correctly recognized → especially things like Union, Liberale very good (not “actual” name of party/ 
    different naming as usual)
    * Only one missed
    * Some differences in classification with/ without article for the party, not directly false
    * False classified: “Partei” occurs often, might be related to the German party 
    “Die Partei” → difficult case, 
    “Corona” might occur because it was most likely not in the training set of the German language model
* Persons
    * Almost everything correct → in “von Lucke” only “Lucke was recognized 
    (hard case, because “von” is a regular German word, in most cases not associated with names)
    * Again “Corona-Krise” was most likely not in training set of language model



Other general findings after a first insight into the tagged data were:
* In general, (most likely) not known words are often recognized as organizations → problems with longer 
expressions and expressions with hyphens
    * Examples Bild
        * 90-Liter-Kompressorkühlschrank
        * Drei Leasing-Schnäppchen
        * Hybrid-PKW
        * CO2-Grenzen
        * 310-PS-Antrieb
    * Examples Tagsschau
        * Antarktis-Durchquerung
        * Ex-Präsident
        * Corona-Pandemie
        * CSU-euphorische Bierzeltstimmung
    * Examples TAZ
        * Corona
        * Schlusslicht
        * Bund und Ländern
        * Sechs-Monats-Frist
        * +++ Corona News
        * Sars-CoV-2
* Seems like persons are more often recognized correctly, but also some outliers where not clear pattern is 
recognizable why they are tagged as persons (assumption: most likely also words that are not known from 
training the model)
    * Examples Bild
        * Luxus-Limo
        * Preisschock
        * Dranhalten
        * Blitzmarathons
    * Examples Tagesschau
        * Verheerende Buschfeuer
        * Gestrichene
    * Examples TAZ
        * Zopfstränge
        * Backwettbewerbe
        * Coronatests
        * Selbstisolation
* For persons, sometimes “role” is also tagged, sometimes not
    * “FDP-Chef Lindner” → tagged “Lindner”
    * „Bundeswirtschaftsminister Altmaier“ → tagged „Bundeswirtschaftsminister Altmaier“
    * „Verkehrsminister Scheuer“ → „Verkehrsminister Scheuer“
    * „FDP-Ausschussmitglied Luksic“ → tagged „FDP-Ausschussmitglied Luksic“
* In very few cases, persons are tagged as organizations (when their name is associated with an organization, 
e.g. as role: WikiLeaks-Gründer Assange) 
* In general, tagesschau and tagging seems to work better than BILD tagging → might be topic related 
(the first 100 articles of taz& tagesschau were mainly about politics, for BILD mainly cars)

**SUMMARY:** NER tagging with spacy and german language model does not work perfectly, but quite good and good enough 
for a first filtering of the text to recognize in which text political parties/ actors are mentioned at all









