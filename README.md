#  *My Data Journey: Portfolio Overview*

*Hey there!🖖 Welcome to my portfolio.*  
> *“Learning to code is learning to think. And thinking with data is learning to see the world in a new way.”*

*This is a technical and personal showcase of the tools, languages, and technologies I’ve learned and applied on my journey through the world of **Big Data**, **Data Science**, **Machine Learning**, and **Artificial Intelligence**.*

*Each section includes a brief description and a direct link to the corresponding repository. Feel free to explore, learn, and hopefully find some inspiration!*


---

## 🚀 *Case study: Machine Learning + MLOps*  ➡️ *[View Repository](https://github.com/Oridi24/Case-study-Traffic-Accident-prediction-ML-MLOPs.git)*   
> *"Imbalanced data? Welcome to real life… and rush hour traffic."*

📌 *Traffic Accident Severity Prediction: Case Study Description:*
*Road traffic accidents have a significant social and economic impact. This project develops a Machine Learning model to predict whether an accident will be minor (0) or severe (1), based on contextual factors such as road conditions, weather, time, and number of vehicles involved.*

- ***Challenge**: Highly imbalanced dataset (~80% minor, 20% severe).*
- ***Approach**: Preprocessing, feature engineering, cross-validation, and testing multiple algorithms.* 

  ***Development Methodology:***

1. *Data Preprocessing*
2. *Model Training & Evaluation:*
    - ***Algorithms tested**: Decision Trees, Random Forest, Gradient Boosting, XGBoost, Logistic Regression (with class weights and SMOTE), Balanced Random Forest, and Easy Ensemble.*
3. *Validation techniques: StratifiedKFold for ensemble methods and GridSearchCV for hyperparameter tuning.*
4. *Metrics for Performance Assessment: Precision, Recall, F1-score, ROC-AUC, Average Precision.*
5. *Deployment: The final model was deployed with FastAPI as a REST endpoint (/predict), allowing real-time accident severity predictions.Output includes both the predicted class (0 = minor, 1 = severe) and the probability of a severe accident.*
6. *Recorded Video Walkthrough: a 8-minute presentation explaining the case study, the modeling approach, and the practical implications of deploying the predictive system.*

***Results:***
 - ✅ *Balanced Random Forest delivered the best trade-off, with strong recall for severe accidents.*
---


## 🤖 *Machine Learning*:  ➡️ *[View Repository](https://github.com/Oridi24/Machine-Learning.git)*   
> *"Teaching machines to see, think, and decide —> Where math meets magic: welcome to the world of Machine Learning."*

*This project addresses a real-world regression challenge and a complete supervised learning pipeline: predicting Airbnb listing prices from a complex, noisy dataset obtained via web scraping with over 700.000 records.*

***Development Methodology:***

- *Exploratory Data Analysis (EDA) and outlier detection*
- *Robust preprocessing (missing value imputation, encoding, scaling)*
- *Feature transformation (logarithmic, polynomial, etc.)*
- *Model benchmarking and tuning using cross-validation*
- *Documentation & conclusions*
  
***Technologies and libraries:***
- *`pandas`, `numpy` `matplotlib`, `seaborn`,`scikit-learn`.*  
- ***Models used**: `Ridge`, `Lasso`, `DecisionTreeRegressor`, `HistGradientBoostingRegressor`, `RandomForestRegressor`, `SVR`,`LightGBM`*

💡 *Special focus on avoiding data leakage and optimizing model generalization through stratified validation and careful pipeline design.*

---

## 📖 *Natural Language Processing (NLP)* :  ➡️ *[View Repository](https://github.com/Oridi24/Natural-Language-Processing-NLP-.git)*  
> *"AI that understands your words: that’s NLP."*

*This project presents a complete NLP pipeline for a binary sentiment classifier applied to over 150,000 Amazon gourmet product reviews, covering the most important stages in a traditional workflow, and demonstrates a hands-on approach to solving a real-world text classification problem using classic NLP techniques and machine learning tools:*

***Development Methodology:***

1. ***Corpus Download & Exploratory Data Analysis (EDA)**: Visualizing the distribution of reviews, extracting frequent n-grams, plotting word clouds, and analyzing word embeddings with Word2Vec.*
2. ***Text Preprocessing:** Implementation of a robust text cleaning function (lowercasing, punctuation and stopword removal, etc.) to prepare raw reviews for modeling.*
3. ***Model Training & Evaluation:** Training and comparing two different machine learning models using a Bag-of-Words representation, evaluating their performance with precision, recall, and F1-score.*
4. ***Final Report & Conclusions:** Interpretation of results, final model selection, and insights about potential improvements.*
   
***Technologies and libraries:***
- *`pandas`, `numpy` `matplotlib`, `seaborn`,`scikit-learn`.*  ⚠️
- ***Models used**: `Ridge`, `Lasso`, `DecisionTreeRegressor`, `HistGradientBoostingRegressor`, `RandomForestRegressor`, `SVR`,`LightGBM`* ⚠️

---

 ## 🚀 *Algorithms deployment MLOps* ➡️ *[View Repository](https://github.com/Oridi24/Algorithms-deployment-MLOps/blob/main/README.md)*
 > *From theory to deployment: transforming models into solutions.*

*This project aims to strengthen and apply key skills in data science, machine learning engineering, and production deployment, covering the entire lifecycle of a classification model, from data exploration and training to deployment as a functional API.
The project also integrates NLP components using Hugging Face, enabling sentiment analysis and text summarization, all exposed through RESTful endpoints powered by FastAPI*.


- ***Understand the complete ML pipeline**, from data preparation to model evaluation and persistence.*
- ***Refactor Jupyter notebooks into clean, modular Python scripts**, ensuring reusability and maintainability of the codebase.*
-  ***Deploy models as RESTful APIs** using FastAPI, enabling easy external interaction with the trained model.*
-  ***Integrate Natural Language Processing (NLP) tasks** with Hugging Face pipelines for real-world applications as sentiment analysis and summarization.*

---

## 🧠 *Deep Learning*:  ➡️ *[View Repository](https://github.com/Oridi24/Deep-Learning.git)*          
> *"Not just learning —> Deep learning: Discovering patterns humans can’t see."*

*Developed in **PyTorch**, this project applies multimodal Deep Learning to predict user engagement levels for tourist Points of Interest by combining image data with structured metadata. It features a custom CNN for visual analysis, a feedforward network for tabular data, and fusion for binary classification. Includes preprocessing, geospatial clustering, normalization, training loop with early stopping, and model evaluation.*

***Technologies and libraries:***
- `pandas`, `numpy`, `mamatplotlib`, `seaborn`
- *`scikit-learn` for geospatial clustering (`KMeans`), normalization (`StandardScaler, MinMaxScaler`), `train/test split`.*
- *PyTorch for neural network architecture (CNN), custom dataset class, dataloaders, model training, and evaluation.*
- *`torchvision.transforms` for image preprocessing (resizing, normalization, tensor conversion).*
- *`tqdm`:  real-time progress bar during training.*

⚠️ ampliar
---


## ☁️ *Big Data Architecture (GCP + Hadoop*):  ➡️ *[View Repository](https://github.com/Oridi24/BD-Architecture.git)*
> *"From local commands to distributed flows in the cloud."*
 
*Design and deployment of Big Data systems using **Google Cloud Platform** and the **Hadoop** ecosystem.* 

***Technologies explored:***
- *Distributed Hadoop architecture: `Hive`, `ElasticSearch(Apache)`, `HBase`, `HDFS`, `YARN`**  
- *Google Cloud Dataproc for large-scale processing, virtual machines via Compute Engine, private networks (VPC), Cloud Storage buckets, and more*

---
## 🌐*SQL*: ➡️ *[View Repository](https://github.com/Oridi24/SQL-Activities.git)*
> *"Select. Join. Analyze. Repeat: Query smarter, not harder."*

*Hands-on exercises using **SQL**, and **BigQuery** focused on querying relational databases.*

***Topics include:***
- *Joins, subqueries, aggregation functions, filtering, and conditional logic, real-world data analysis practice.*

---

## 📊 *Power BI:*  ➡️ *[View Repository](https://github.com/Oridi24/Power-BI-KC.git)*
> *"Data that speaks. Insights that lead. -See your data come alive: empower decisions with every click."*

*Creating data storytelling with visual design*

***Key skills:***   
- *Data connections from Excel, SQL, and more.*
- *Interactive dashboard creation, Custom metrics with DAX*

---

## 📈 *Statistics & Linear Algebra*:  ➡️ *[View Repository](https://github.com/Oridi24/Statistics-Linear-Algebra-Data-Minning.git)*

*Projects developed in **R**, focused on statistical analysis from foundational concepts to applied techniques.*

***Techniques applied:***
- *Exploratory and confirmatory analysis, outlier detection, and handling.*
- *Statistical tests (t-tests, ANOVA, correlations), metrics like MSE and R².*   
- *Linear regression and model diagnostics.*

---

## 💻 *Python for Data Analysis*:  ➡️ *[View Repository](https://github.com/Oridi24/Python-Activities.git))*

*Practical Python applications for working with structured datasets.*

---

## ⚙️ *Scala & Apache Spark:*  ➡️ *[View Repository](https://github.com/Oridi24/Scala-Spark.git)*

*Strong foundation in **Scala** and its integration with **Apache Spark** for distributed data processing.*

***Focus areas:***
- *Functional programming principles with Scala.* 
- *Spark DataFrames, RDDs, transformations and Test-Driven Development (TDD) methodology for worlflows.*

---

## *Final Thoughts*:

*Every project here represents much more than code — it's a testament to continuous learning, perseverance, and problem-solving.*  
*If you've made it this far, thank you for your time. I hope this portfolio inspires you or helps in your own journey.*

📫 ***[Let’s connect on LinkedIn](www.linkedin.com/in/orionis-di-ciaccio-168592185)*** 
 

