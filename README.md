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

4. Activate virtual environment: `pipenv shell`  

5. Download spacy corpus `python -m spacy download de_core_news_lg`

6. Test setup: `pipenv run main`  (more about typical ways to run the program are shown in the section [below](#run-the-program-in-different-modes))

7. Install Git hooks: `pre-commit install` (developer setup)


## Testing

A script to automatically execute tests is already defined in the project's `Pipfile`. Therefore you can simply run: `pipenv run test`
To generate a report on code coverage alongside run: `pipenv run test && pipenv run report`

## Run the program in different modes
Here are typical use cases how you might want to run the program. Make sure you are in the project directory when executing the code.
1. I want to see the results for sentiment and topics in the GUI
    ```
   pipenv run main -g
    ```
   __WARNING:__ If you have not started the program before, the processing of the whole dataframe is performed, which takes ~2-3 hours. If you want to see the results of a smaller subset of the data you can limit the amount by
   ```
   pipenv run main -g -n 1000
   ```
   In this example, only 1000 articles are read in and processed
2. I want to force the processing again (because I want to use a different number of articles/ a different dataset)
    ```
   pipenv run main -f
    ```
   
3. I want to label data
    ```
   pipenv run main -l 0-100
    ```
   In this example, the first 100 articles are outputted to the console for labeling. You can change the range of articles by specifying different \<start\>-\<end\>.\
   The result of the labeling is stored in a file named "labeled_paragraphs.json".\
   For more information about how to label paragraphs, see [Labeling Data](#labeling-data)
  
4. I want to train the sentiment threshold
    ```
   pipenv run main -t
    ```
   Note that you need a labeled dataset to train the threshold. After training, the "labeled_paragraphs.json" and "paragraphs.json" are overwritten with the new sentiment labels determined with the optimized threshold.
   
4. I want to evaluate the results
    ```
   pipenv run main -c
    ```
   Note that you need a labeled dataset to evaluate the results.


Of course the command line arguments can also be used in different combinations than the use cases explained above. For an overview of possible command line arguments with description enter
```
pipenv run main -h
```

## Using the GUI
This section describes how you can use the GUI (command line argument -g).\
For explanation, some parts in the GUI are overlayed with a colored box in the screenshot below.
![GUIdescription](figures/gui_description.png)
In the colored boxes you can see the following:
* Red: these are the different types of graphs you can show. 
  By pressing one of the buttons, the respective graph is shown in the center of the screen. 
  In this example, the graph for "Show Topics of Parties and Media" is shown. 
  To display the "Show Time Course for Word", you have to specify the word of which you want to see the frequency in the text box on the left hand side (here: "corona-krise")
* Green: These are different filter criteria to select certain data of interest for you. 
  For specifying a time range, you need to enable the "Filter dates" checkbox and type in a date in the format "YYYY-MM-DD". 
  Note that the option "How many topics per party& media to show (Max 10)" is only available for "Show Topics per Party", "Show Topics per Media" and "Show Topics per Party and Media"
* Orange: If you changed one of the filter criteria and want to update the graph in the middle, you need to click "Update filter criteria".
* Purple: For some types of graphs, there are multiple graphs which can be viewed one after the other in the GUI ("Sentiment by Party", "Sentiment by Outlet", "Show Time Course").
  You can navigate through them with "Show previous" or "Show next".
* Blue: if you click on this button, additional information what can be seen on this graph is shown as a text in a pop up window.

## Labeling Data
This section describes how you can label data if you run the program in labeling mode (command line argument -l \<start\>-\<end\>).

For each article you can see the following output in the console.\
================================================\
\<Title of the article\>\
++++++++++++++++++++++++++++++++++++++++++++++++\
\<One paragraph of the article\>

Note that for all articles the tagged persons and parties are anonymized by "\<Person\>" or "\<Partei\>" to not influence your rating based on what party/ person the article is about.

Just type in the console which Polarity (-1: Negative, 0: Neutral, 1: Positive) you think is stated towards the party. 
After pressing enter type in if you think the paragraph is subjective or objective (0: Objective, 1: Subjective).
After pressing enter again, the next paragraph appears.\
The result of the labeling is stored in a file named "labeled_paragraphs.json".

## Planning State

| Description                                      | Milestone          | Deadline   | Started  | Achieved | 
|--------------------------------------------------|--------------------|------------|----------|----------|
| Project Idea                                     | Proposal           | 27.11.2020 | X        | X        |
| Initial Research                                 | Proposal           | 27.11.2020 | X        | X        |
| Proposal Report                                  | Proposal           | 27.11.2020 | X        | X        |
|                                                  |                    |            |          |          |
| Repository Setup                                 | December Milestone	| 18.02.2020 | X        | X        |
| Alignment Meeting with Mentor                    | December Milestone	| 18.02.2020 | X        | X        |
| Dataset Analysis                                 | December Milestone | 18.02.2020 | X        | X        |
| Dataset Statistics                               | December Milestone | 18.02.2020 | X        | X        |
| Preprocessing Pipeline                           | December Milestone | 18.02.2020 | X        | X        |
| Topic Detection                                  | December Milestone | 18.02.2020 | X        | X        |
| Milestone README                                 | December Milestone | 18.02.2020 | X        | X        |
|                                                  |                    |            |          |          |
| Negation Handling                                | Final Report       | February   | X        | X        |
| Incorporate SentiWS (Sentiment Lexicon)          | Final Report       | February   | X        | X        |
| Sentiment Calculation                            | Final Report       | February   | X        | X        |
| Sentiment Visualization                          | Final Report       | February   | X        | X        |
| Sentiment Threshold                              | Final Report       | February   | X        | X        |
| Filter Options                                   | Final Report       | February   | X        | X        |
| Dimension Analysis                               | Final Report       | February   | X        | X        |
| Final Codebase                                   | Final Report       | February   | X        | X        |
| Final README                                     | Final Report       | February   | X        | X        |
| Final Report                                     | Final Report       | 15.03.2021 | X        | -        |
|                                                  |                    |            |          |          |
| Project Video                                    | Presentation       | 25.02.2021 | X        | X        |

### Data Sources 
As data sources, around 100.000 german news articles from three different news agencies are used. those are namely:
* Bild: 75826 articles
* Tagesschau: 12284 articles 
* TAZ: 13401 articles

With [NER tagging](#ner-tagging), the data was filtered such that only articles which deal with a German political party remain. 
After this filtering, the ratio between all articles and articles of interest looks the following:

![OverallStat](figures/articles_overall_stat.png)

To visualize the ratio between the individual agencies a bit better, the following charts show the ration for each agency

![BildStat](figures/articles_bild_stat.png) Ratio between all articles and articles of interest: 8514/75826 = 11.23%

![TagesschauStat](figures/articles_tagesschau_stat.png)Ratio between all articles and articles of interest: 2373/12284 = 19.32%

![TazStat](figures/articles_taz_stat.png)Ratio between all articles and articles of interest: 3468/13401 = 25.88%

Overall the ratio of articles of interest (those which deal with parties) looks the following

![RelevantArticles](figures/articles_relevant_articles_stat.png)

Overall, we found 14355 articles of interest.

Although "Bild" hs the lowest ratio of articles of interest / articles, the Bild articles make up nearly 60% when only looking only at the 
articles of interest. This is of cause caused by the significantly higher amount of "Bild" articles overall. 

## Pipeline
![Pipeline](figures/pipeline.png)
### Preprocessing
Before the data can be processed any further, it needs to be preprocessed. The challenge hereby was to choose preprocessing methods that keep enough information to determine the objectivity (sentiment analysis) later on.
For this reason the pipeline, which is described in the following, was created.
* Remove quotations: paragraphs with direct speech should be removed as the sentiment of a speaker is not representative for the sentiment of the paragraph.
* Removing special characters:
    Secondly some characters that are unimportant for further processing need to be removed. 
    Additionally with the help of regex, letters from "a-z", numbers from "0-9", the hyphen as well as the - in German commonly used - umlauts ("ä", "ö", "ü") are kept. Everything else is removed from the text.
* Tokenization:
    This segments the specified text into tokens 
* NER Tagging (tag persons and tag organizations): This is explained in more detail later on.
* Extract parties: The parties which are present in an article are extracted and stored as list.
* Determine sentiment polarity: With the sentiment lexicon SentiWS, a polarity score is stored for each token.
* POS tagging: 
    Here, part-of-speech tags are assigned to the tokens.
* Lemmatization:
    The tokens of the text are saved in their respective base form. This was preferred before stemming, since the sentiment analysis is done by using a sentiment lexicon that weights words by their positive or negative indication. 
* Negation Handling: The sentiment polarity is inverted for the surrounding words around a negation word (window size of 4 words before and 4 words after negation word).

#### NER Tagging 
As one of the first parts of the project, NER tagging was performed in order to find political parties and members of
the parties to identify relevant articles in the data set. For NER tagging, spacy is used with the 
[de_core_news_lg language model](https://spacy.io/models/de#de_core_news_lg), which has the best score for NER tagging
compared to the other German language models provided by spacy.

Results of the NER tagging:

To get a first insight how good the NER tagging performs, the first 100 articles of each agency were tagged.
Overall, the NER tagging performs quite okay, especially it is good in recognizing different parties.

Here is one example of the Tagesschau data set (article at index 4):

![Screenshot](figures/ner_tagging.png)

**SUMMARY:** NER tagging with spacy and german language model does not work perfectly, but quite good and good enough 
for a first filtering of the text to recognize in which text political parties/ actors are mentioned at all.


### Sentiment Analysis
For the sentiment analysis, each word of a paragraph was weighted by a sentiment score from SentiWS. Also the TF-IDF score of each word was calculated. By performing the dot product of the sentiment score, and the TF-IDF scores, the overall sentiment of a sentence was calculated. 
This was done by using a threshold. If the dot product is above this threshold for greater 0 the sentiment is mapped as positive, 
if it smaller for below 0 it is mapped negative and otherwise neutral.

#### Training the thresholds for mapping the sentiment
As the threshold of mapping the sentiment score to a conrete label "Positive", "Negative" and "Neutral" requires some finetuning, we calculated the best threshold maximizing the sum of the f1-score of the positive, negative and neutral f1-scores. The visualization of the chosen threshold is shown in the image below.

![Score Threshold Training](figures/score_threshold.png)

#### Context Sentiment
As we labeled our data as positive or negative only when the paragraph was against or for a party, and not if only the topic itself was negative, we introduce the context sentiment, that only keeps the polarity values of words, that are close to a party. Polarities of words that are far from a party, are set to 0. We hoped that using the context sentiment, we could capture only the context around a party. This introduces a new threshold for the window around a party to consider for the sentiment.

#### Training the window threshold for mapping the sentiment
The following image shows the same optimization as above, but this time for the window size. For each window size, the already best score threshold is considered.
As the f1 sum does not increase much after a window of 8, this threshold is chosen as 31 would be almost the whole paragraph.

![Score Threshold Training](figures/window_threshold.png)


#### Sentiment Clustering
Instead of using SentiWS to calculate the sentiment scores of each word, we tried also to perform a clustering of the sentiment to get more accurate results. But as you can see in the following image, the clusters are not towards a sentiment and thus cannot be used for our task.

![Coherence Score](figures/sentiment_clustering.png)  


## Storing the Results
The paragraphs of the processed dataframe is stored in a JSON-file called "paragraphs.json".
The JSON object is then structured as follows:

```json
[
  {
    "article_index": "index of article (int)",
    "text": ["text", "as", "list", "of", "tokens"],
    "media": "Bild|Tagesschau|TAZ",
    "date": "date in format yyy-mm-dd",
    "original_text": "original text as string",
    "persons": [
      "Person 1 in text",
      "Person 2 in text"
    ],
    "organizations": [
      "Organization 1 in text",
      "Organization 2 in text"
    ],
    "pos_tags":["pos_token1", "pos_token2"],
    "nouns":["noun1", "noun2"],
    "authors": null,
    "references": [],
    "polarity": ["polarity_sentiws_token1", "polarity_sentiws_token2"],
    "sentiment_score": "sentiment score (dot product as float)",
    "sentiment": "Positive|Negative|Neutral (with SentiWS)",
    "polarity_textblob": "float of polarity",
    "sentiment_textblob": "Positive|Negative|Neutral",
    "polarity_context": ["0 (not in context of party)", "polarity_sentiws (in context of party)"],
    "sentiment_score_context": "sentiment score (dot product as float)",
    "sentiment_context": "Positive|Negative|Neutral (with SentiWS, only considering words in the context of a party -> fixed window size)"
  }  
]
```

### Topic Detection
As discussed in the first alignment meeting with our advisor, we want to apply topic detection on our preprocessed articles to be able to compare the sentiment not only across parties, but also across different topics the journalists are talking about. This should give us the oppurtunity to filter out some bias if e.g. a news publisher is focused on specific topics like e.g. the natural environment and therefore is generally more critically against parties that have a different point of view about this topic.

For the implementation of the topic detection we used the library `gensim`. We created a class `TopicDetection` that does handle all the tasks of topic detection. It can calculate the TF-IDF scores and the document similarity of a given corpus. Also we implemented three different methods that return `gensim` models of the `Latent Semantic Analysis (LSA)`, `Latent Dirichlet Allocation (LDA)` and `Hierarchical Dirichlet Process (HDP)`, that can be applied on our data to gather the most important topics. We used different models to calculate the topics, as we wanted to compare the results of each method to choose the best performing one.

As a measure of quality for the topics, we calculated the coherence score. As `LSA` and `LDA` are depending on the number of topics as a parameter, we implemented a method called `plot_coherence_scores()` that test empirically which number of topics results in the best coherence score.

Another method of analysing the quality of the topics is using a visual representation of the topics with `pyLDAvis`. Therefore the method `visualize_topics()` of the TopicDetection class can be used.

#### Results
![Coherence Score](figures/coherence_score.png)  
The figure above shows that the coherence score of LDA is increasing almost linearly with the number of topics. This is not ideal, as we want to cluster the topics of multiple articles and not create a topic per article. To improve the performance, we may need to improve the preprocessing and also make further analysis on how to adjust the parameters of the models.  

![Topic Visualization](figures/topic_visualization.png)  
Also the visual representation of the LDA topics shows that the topics are not chosen perfectly. There is no real clustering as multiple unrelated topics are merged together (e.g. sonneborn and mordkommision).

Although the figures above only show results from LDA, the results of the other methods are fairly similar. We have chosen `HDP` as another option for topic modeling, because it determines the number of topics automatically and no parameter for this has to be provided. But using this model, far too many topics are generated and almost no generalization of similar topics are made.

We also tried to cluster the topics using Hierarchical Agglomerative Clustering, but without much success. The following figure shows how the Dendrogram, that should help to find the optimal numbers of clusters. Normally the best choice of number of clusters should be where the largest vertical distance doesn't intersect any of the clusters. Here this is the case before any clusters are merged. Therefore, the number of optimal clusters should be equal to the number of articles. Obviously this is not what we want.

![HAC](figures/hac.png)  

To overcome this issues, we need to further investigate why our documents can't be grouped that easy by a topic. As already mentioned, we need to further finetune the preprocessed data and the parameters of some models or maybe look at a completely different approach.

### Keyword Extraction
As the results of the topic detections are not as good as expected, a basic keyword extraction was done using TF-IDF scores. For each party the top three words are taken, and the count of each word occurrence in the paragraphs is counted. The results are shown in the following bipartite graph:  

![Keyword extraction](figures/keyword_extraction.png)   



