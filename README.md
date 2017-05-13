# Book Recommendation System

## Dependencies
- graphlab-create
- SFrames
- numpy
- matplotlib

## Running the code
Final+Notebook.py contains the executable code that can be run directly through terminal. The script will call suggest() function which maps all the outputs from different models together. You can adjust that function and refer to the comments to know more about the parameters of function. The output of this function call will be information about five recommended books and their cover image.
[Look at Final_Notebook.ipynb notebook to know more about predicted output format]

## What are Recommender Systems?
In layman terms, a recommender systems can predict something about a user based on user's past activities. Here the activity can be - a purchase history of the user, a book rated by some user, a movie rated by user on netflix. Example of Recommender System is the recommended products that amazon website shows to its users based on their past purchases.

## What's goes into a recommender system and what comes out on the other side?
It works in two phases 
### While building the reommender system: 
In this phase we need to provide the data about every single user(means a lots of data). Then a mathematical model is formulated using some state of the art prediction algorithm which has a lots of statistics behind it. And we get a Black Box of trained model that can be used to make predictions.
### While predicting outputs: 
In this phase only the user id is required(in most cases) and our model can output a list of items that a user is likely to purchase or in our case the books the user might love to read.

## What kind of Techniques are used to build these kinds of Recommender Systems?
Recommender Systems are hot research topics and they lie in the realm of Machine Learning. There are many approches that can be used to build a recommender system like Matrix Factorization, Collaborative filtering, Item-based similarity, User-based similarity etc.

## So, what's so special about this recommender system?
Generally, most simplest of recommender systems are constructed based on one of the techniques mentioned above. But these recommender systems do not provide the right accuracy which is the most crucial aspect of recommender systems that are to be used in commercial applications. Inaccurate results can incurr significant loss and wastage of resources.
Now there are some ways to increase accuracy like:
### -Get more data about user.
### -Use a better algorithm or switch to another technique.
### -Train models using different techniques and ensemble them together to improve results

So, my dataset was static and I already had the cutting edge techniques for predictions, so I chose to ensemble models trained differently and output cumulative predictions.
The five models that I made use of were:
### -Popularity based model
### -Classification based model
### -User based Collaborative Filtering model
### -Cooccurence matrix model 
### -Matrix Factorization model

## Which libraries or tools are used in building this RS?
Programming Language: Python 2.7
IDE: jupyter-notebook
libraries: graphlab-create/SFrames | (Just for the Matrix factrization model, all other models are implemented from SCRATCH)

## What are all those files and folders there for?
### FOLDERS:: 
### book_data_clean|explicit_rating_data|implicit_rating_data|predicted_implicit_data|user_data_clean: 
All of these folders contain the clean data derived from the noisy data that I was working with. Various preprocessing techniques were used to make this data usable. The information about what paritcular folder is about can be found in comments, also the different preprocessing techniques used to filter this data is mentioned in comments
### regression_model_data:
This folder contains the first step for construction of classification based model which is to train explicit data on regression model. Data is stored in some format used by graphlab-create so it might not make sense but again comments can be referred to know what's that folder is doing.
          
### NOTEBOOKS:: 
All the notebooks are in .ipynb format which can be directly viewed on github only without downloading or running them.

### Co-occurence Matrix Recommender.ipynb :
As the name suggest this notebook contains the implementation of Co-occurence matrix based model.
### Data Preprocessing.ipynb : 
All the preprocessing steps taken to clean data for usability are employed in this notebook.
### Final Notebook.ipynb : 
This notebook contains the code assembled from all other distinct notebooks i.e. this is where ENSEMbLING of all 5 models is performed. This notbook alone with the required data is enough for using this model.
### Popularity.ipynb : 
Contains code for popularity based model.
### Predicting Implicit Values.ipynb : 
Some play around code for increasing accuracy of the model. Comments can be referred for the use of this code in improving accuracy of the model.
### Project Report: 
A brief info of the challenges I faced and tricks i have used to make best predictions.
### Regression Based Recommender.ipynb : 
Code for classification based model.
### User-Similarity model.ipynb : 
Code for User-similarity based model 
### cooccurrence dict.npy: 
A python dictionary storing numpy arrays, which are used in cooccurence model
### modelPerformance(1)(2)(3): 
A comparison between matrix factorization based models trained fon different data (implicit and explicit, with or without rankings)
### rating_dictionary: 
Another dictionary storing numpy arrays used in user-simlilarity model

## Is this project of any use? Its not a Website or a mobile application?
This python code is well modularized and can be integrated with any webapp or mobile application with some glue code and hence it can work as a Core Engine for your Book Recommending Website. There is even no need to include the CSV files as the cleaned data is already stored in folders and hence it won't consume much space. There is only one function call which takes as parameter a UserId and another parameter denoting wheater the user is a new user or an old user and ouput is well defined description about book with book's image(amazon url's are used to display images hence internet connection is required for displaying images)  
-graphlab-create and python 2.7 must be installed to execute this project.

CAUTION:
Matrix Factorization model cannot be used directly as it was trained solely on graphlab-create and was quite large in size to be uploaded on github and hence the code using matrix factorization based model has been commented so that it do not hinder the execution of rest of the code.
But, the output shown in the final notebook include predictions from Matrix Factorization model, so you can get a fair idea of what's this model is doing.

## Is there any other cool aspect of this project?
Yes, one feature of this Recommender System is that it can also be used for prediction for completely new users with no past rating history. Popularity based model and Classification based model will be used for making predictions for such users based on the characteristics of users like thier location etc.
This feature is not a rare one, today almost every recommender system is capable of making predictions for new users but its definitely one of the strongest feature that a recommender system can provide.
