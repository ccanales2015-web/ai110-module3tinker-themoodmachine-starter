# Model Card: Mood Machine

## 1. Model Overview

**Model type:**
This project compares two classifiers: a rule-based model in `mood_analyzer.py` and a machine-learning model in `ml_experiments.py`.

**Intended purpose:**
The Mood Machine is an educational prototype that classifies short, informal text posts as **positive**, **negative**, **neutral**, or **mixed**. It is intended to demonstrate how data, labels, explicit rules, and learned patterns affect model behavior. It is not intended to assess a person's actual emotional or mental-health state.

**How it works (brief):**
The rule-based model converts text to lowercase tokens, looks for words in positive and negative vocabulary lists, and adds or subtracts one point for each sentiment match. It also reverses the polarity of the next sentiment-bearing word after `not`, `never`, `no`, or `ain't`. If both positive and negative evidence are found, it returns `mixed`; otherwise, the sign of the score determines positive or negative, and a score of zero returns neutral.

The ML model uses scikit-learn's `CountVectorizer` to represent posts as bag-of-words features and trains a `LogisticRegression` classifier on the labeled examples in `SAMPLE_POSTS` and `TRUE_LABELS`.

## 2. Data

**Dataset description:**
The final dataset contains **18 short posts**, each paired with one label in `TRUE_LABELS`. The starter contained six posts. Eight additional posts were first added to cover realistic informal language, emojis, sarcasm, and mixed feelings. Four more labeled posts were added during the evaluation experiment. The final label counts are:

| Label | Number of posts |
|---|---:|
| Positive | 6 |
| Negative | 5 |
| Mixed | 4 |
| Neutral | 3 |
| **Total** | **18** |

The dataset and labels were checked to ensure `len(SAMPLE_POSTS) == len(TRUE_LABELS)`.

**Labeling process:**
The added examples and proposed labels were reviewed before being added. Labels were selected according to the overall meaning intended by each short post, rather than simply counting positive and negative words. Some examples are inherently subjective. For example, “The weather is nice, but I have too much work” was labeled mixed because it expresses both a positive observation and a negative situation. Different readers could reasonably disagree about the strength or overall label of some ambiguous posts.

**Important characteristics of the dataset:**

- It includes informal expressions and slang, such as “lowkey” and “fire.”
- It includes emoji use.
- It includes sarcasm, including “I studied all night and still failed. Great.”
- It includes mixed feelings and contrasting clauses.
- It includes negation and short neutral statements.
- It consists of short, English-language posts rather than long documents or conversation threads.

**Possible issues with the dataset:**
Eighteen examples are far too few to represent the variety of real-world language. The labels are human judgments and may be subjective. The four classes are not perfectly balanced, and some language styles and expressions appear only once or a few times. The dataset does not provide demographic information, so it cannot support claims about performance for particular demographic groups.

## 3. How the Rule Based Model Works

**Your scoring rules:**
The model uses the positive and negative word lists in `dataset.py`. Each recognized positive word adds one point, and each recognized negative word subtracts one point. The model lowercases text and uses a regular expression to tokenize words while retaining apostrophes in contractions and removing surrounding punctuation.

The intentional improvement is negation handling. The words `not`, `never`, `no`, and `ain't` reverse the polarity of the next sentiment-bearing word. For example, “not happy” is treated as negative evidence, while “ain't sad” is treated as positive evidence. The model does not implement general language understanding or a complete grammatical analysis of negation.

The label rule also checks whether both positive and negative evidence are present. If both are present, the prediction is `mixed`; otherwise, a positive score gives `positive`, a negative score gives `negative`, and a zero score gives `neutral`.

**Strengths of this approach:**

- Its decisions are relatively transparent: the matched words and score can be inspected.
- It behaves predictably on simple examples containing words in its vocabulary, such as “I love this class so much” and “Today was a terrible day.”
- It can handle the specific negation patterns that were explicitly added and tested.
- It does not require a training process.

**Weaknesses of this approach:**
The classifier depends heavily on its small, fixed vocabulary. It may ignore meaningful words it does not recognize, and a single recognized word can dominate a sentence. For example, “I studied all night and still failed. Great.” was classified positive because `great` was recognized as positive, while the sarcastic context was not understood. “That concert was absolutely fire!” was classified neutral because `fire` was not in the positive word list. The model also does not interpret emoji meaning; emojis are not included in its sentiment scoring.

## 4. How the ML Model Works

**Features used:**
The model uses `CountVectorizer` from scikit-learn. This is a bag-of-words representation: it converts the words in each post into numeric word-count features. A `LogisticRegression` classifier learns associations between those features and the labels.

**Training data:**
The model is trained on all 18 posts in `SAMPLE_POSTS` and their corresponding labels in `TRUE_LABELS`.

**Training behavior:**
With the original 14 examples, the ML model classified all 14 training examples correctly. After four more examples were added, it classified all 18 training examples correctly. The predictions for the original 14 examples did not change after the four additions, and all four new examples were also classified correctly.

These results are **training accuracy**, not an estimate of performance on unseen posts. The model was evaluated on the same examples used to train it. Its 100% result therefore does not demonstrate that it will generalize to new language or real-world messages.

**Label-sensitivity experiment:**
To examine how the learned model responds to its training labels, one label was deliberately changed in a controlled experiment: “The weekend should be fun” was changed from `positive` to `negative`, and the model was retrained on the modified dataset. The prediction for that same post changed from `positive` to `negative`, while the other 17 predictions remained unchanged. This demonstrates that the model learned the supplied label for that example; it does not establish how the model would respond to label changes in a larger or different dataset.

**Strengths and weaknesses:**
Compared with the rule-based model, the ML model learned associations from complete labeled examples and correctly reproduced all labels in this small training set. However, it is difficult to interpret exactly why it made an individual prediction, and a small dataset can encourage overfitting. Its behavior also depends on the examples and labels supplied. A perfect score on this training set should not be treated as evidence that the model is reliable.

## 5. Evaluation

**How you evaluated the model:**
Both models were evaluated against the 18 labeled posts in `dataset.py`. The rule-based model made 11 correct predictions out of 18 (**61.1% accuracy**). The ML model made 18 correct predictions out of 18 (**100% training accuracy**).

The following class-level results are also calculated on those same 18 examples. Precision, recall, and F1 are useful because accuracy alone does not show which labels are being confused. These values are descriptive training-set results; they are not independent test-set metrics.

| Model / Label | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Rule-based — Negative | 0.75 | 0.60 | 0.67 | 5 |
| Rule-based — Neutral | 0.38 | 1.00 | 0.55 | 3 |
| Rule-based — Positive | 0.80 | 0.67 | 0.73 | 6 |
| Rule-based — Mixed | 1.00 | 0.25 | 0.40 | 4 |
| ML — Negative | 1.00 | 1.00 | 1.00 | 5 |
| ML — Neutral | 1.00 | 1.00 | 1.00 | 3 |
| ML — Positive | 1.00 | 1.00 | 1.00 | 6 |
| ML — Mixed | 1.00 | 1.00 | 1.00 | 4 |

**Examples of correct predictions:**

1. “I love this class so much” was classified positive by both models. The rule-based model recognized `love` as a positive word.
2. “Today was a terrible day” was classified negative by both models. The rule-based model recognized `terrible` as a negative word.
3. “The movie was awful, but the ending was great” was classified mixed by both models. The rule-based model found both negative and positive evidence.

**Examples of incorrect predictions:**

1. “I studied all night and still failed. Great.” was labeled negative by a human but classified positive by the rule-based model. It recognized `great` but did not understand that the sentence was sarcastic.
2. “That concert was absolutely fire!” was labeled positive but classified neutral by the rule-based model. The slang term `fire` was not in its vocabulary.
3. “I'm exhausted but genuinely proud of myself” was labeled mixed but classified neutral by the rule-based model. The model did not recognize `exhausted` or `proud` as sentiment words.

The rule-based model's class-level recall for mixed posts was only 0.25: it correctly identified one of the four mixed examples. This illustrates a weakness that the overall accuracy figure alone would not reveal.

**Stress-test observations:**
A separate exploratory set of 16 deliberately difficult examples produced 7 correct predictions (43.75%) for the rule-based model before the targeted `ain't` change. The examples covered sarcasm, slang, emojis, mixed feelings, and language variation. The model passed 0/2 sarcasm examples and 0/3 slang examples; it received 3/3 emoji examples correct, although the model does not actually interpret emoji meaning; it received 1/2 mixed examples and 3/6 language-style examples correct. These small, manually labeled stress tests are diagnostic examples, not a representative benchmark. After identifying a failure with “I ain't happy today,” `ain't` was added to the negation vocabulary. Four targeted before-and-after checks then passed, including “I ain't happy today” and “I ain't sad today.” The change was intentionally narrow and did not attempt to solve sarcasm or slang.

**Comparison of the models:**
On the 18 labeled examples, the ML model reproduced the labels more accurately than the rule-based model. The rule-based model's errors were inspectable and often traceable to missing vocabulary or a simple rule. The ML model achieved perfect training accuracy, but this comparison does not establish that it performs better on unseen text. A separate held-out test set would be needed for a fairer generalization comparison.

## 6. Limitations

The most important limitations are:

- **Small dataset:** Eighteen examples cannot represent the range of real posts, dialects, contexts, or emotional expression.
- **Training-set evaluation:** The ML model's 100% accuracy is measured on the same data used for training. It may perform substantially worse on unseen examples.
- **Sarcasm and context:** “I studied all night and still failed. Great.” shows how a positive keyword can conceal a negative or sarcastic meaning.
- **Slang and vocabulary coverage:** “That concert was absolutely fire!” shows that an unfamiliar positive expression can be missed. Other tested slang terms such as “sick” and “wicked” were also not recognized.
- **Mixed emotions:** The rule-based model often misses one side of a mixed statement when the relevant word is absent from its vocabulary.
- **Emoji interpretation:** The current tokenizer removes emoji characters rather than assigning them sentiment. A correct prediction on a sentence containing an emoji may therefore be due to its words, not the emoji.
- **Limited negation scope:** The rule reverses the next recognized sentiment word; it does not understand every grammatical form or the full scope of negation.
- **Subjective labels:** Different people may assign different labels to ambiguous or sarcastic posts.

The targeted `ain't` change improved the specific colloquial-negation examples tested, but it did not make the classifier broadly reliable across informal language.

## 7. Ethical Considerations

A mood classifier can misinterpret a person's message, especially when the message uses sarcasm, slang, dialect, emojis, or mixed emotions. If used in a real application, an incorrect label could cause a person's words to be misunderstood or could lead someone to make an unwarranted assumption about that person's feelings.

The stress tests suggest that the rule-based model may underserve people who use informal, colloquial, regional, or non-standard English expressions that are missing from its vocabulary. The examples do not identify the writers' demographic backgrounds, so they do not establish that a particular demographic group was measured or harmed. They do show a language-style coverage problem that could affect communities whose everyday expressions differ from the small vocabulary used here.

This system should not be used to diagnose mental-health conditions, assess risk, make employment or educational decisions, or make consequential judgments about individuals. Real applications would also need a clear purpose, appropriate consent, data minimization, and protection for private messages.

**Overt bias:**
In the exploratory stress tests, failures clustered around sarcasm, slang, and some informal language styles. For example, “That concert was absolutely fire!” was classified neutral, and “I ain't happy today” was initially classified positive before the targeted negation improvement. This suggests that the system may underserve users who communicate with colloquial or regionally varied English. The small test set is not sufficient to make claims about demographic groups, but it reveals that the model's language coverage is uneven.

**Covert compliance and AI assistance:**
AI assistance was used to brainstorm example posts, explain the starter code, suggest a negation improvement, help design stress tests, and interpret evaluation results. The proposed dataset examples and labels were reviewed before they were accepted. However, the initial suggestions and implementation work did not proactively identify all the coverage concerns that later appeared in stress testing. The gaps became clearer when the system was deliberately tested with sarcasm, slang, emojis, and language variation. This demonstrates a covert-compliance risk: an AI assistant can help carry out a narrow design without sufficiently challenging the assumptions behind its vocabulary or data. Human review and deliberate testing were necessary to expose those limitations.

## 8. Ideas for Improvement

1. **Create a larger, more balanced dataset** with multiple examples for each label and a wider range of writing styles.
2. **Use a held-out test set** or cross-validation so that ML performance is measured on examples that were not used for training.
3. **Review labels with multiple people** to identify ambiguous examples and document reasonable disagreements.
4. **Expand language coverage carefully**, including slang and regional expressions, while checking that additions do not create new false predictions.
5. **Add explicit emoji handling** and test positive, negative, and context-dependent emoji use.
6. **Improve negation handling** so that it considers phrase structure and the scope of negation rather than only reversing the next sentiment-bearing word.
7. **Inspect precision, recall, F1, and confusion matrices** on independent data rather than relying on accuracy alone.
8. **Evaluate language-style coverage** with a broader and more carefully designed set of examples before considering any real-world use.
