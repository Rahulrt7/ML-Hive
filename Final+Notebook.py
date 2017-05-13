import graphlab as gl
import numpy as np
import math
from operator import itemgetter
from IPython.core.display import Image, display

# Loading user data and modifying it to work with Regression based model
user_data = gl.load_sframe("./user_data_clean/")
# remove rows where country is not mentioned in location
fil = []
for item in user_data["location"]:
    temp = item.split(",")
    if len(temp) <= 2 or temp[2] == "":
        fil.append(False)
    else:
        fil.append(True)
fil = gl.SArray(data=fil)

user_data = user_data[fil]

# locations where city is not mentioned replace states with name of their country rather than excluding them
# and convert a complete string of location to a list of strings containg city name and country name as elements
def modify(st):
    st = st.split(",")
    if st[1] == " " or st[1] == " n/a":
        st[1] = st[2]
    del(st[0])
    st_0 = st[0].strip() 
    st_1 = st[1].strip()
    return st_0 + ", " + st_1

user_data["location"] = user_data["location"].apply(modify)

# Loading required data
book_data = gl.SFrame("./csv_files/BX-Books.csv")
book_data = book_data.rename({"ISBN":"book_id", "Book-Title":"title", "Book-Author":"author", "Year-Of-Publication":"year",
                      "Publisher":"publisher"})

book_ratings_data = gl.SFrame.read_csv("./csv_files/BX-Book-Ratings.csv", delimiter=";")
book_ratings_data.rename({"User-ID":"user_id", "ISBN":"book_id", "Book-Rating":"ratings"})

#following four lines of code extract users at random who has rated books greater than 8(high rating) 
high_rated_data = book_ratings_data[book_ratings_data["ratings"] >= 8]
low_rated_data = book_ratings_data[book_ratings_data["ratings"] < 8]
train_data_1, test_data = gl.recommender.util.random_split_by_user(high_rated_data, 
                                                                         user_id="user_id", item_id="book_id")
train_data = train_data_1.append(low_rated_data)

class PopularityModel:
    
    most_popular_books_ids = []
    most_popular_books = []
    
    """
    Function that will take ratings data as argument. It first select books rated 10 from range 0-10 and 
    then count the number of books rated maximum times. After sorting books according to their counts, it checks
    if the book id is present in book data, if it do then it appends that book and its id to respective lists.
    """
    def predict(self, train_data=None, n=2, user_id="user_id", item_id="item_id", 
                              user_data=None, item_data=None, rating="rating"):
        
        # Count how many times a book is rated 10 and sort in descending order
        rating_10 = train_data[train_data[rating] == 10]
        popular_books = rating_10.groupby(key_columns=item_id, 
                                              operations={"count": gl.aggregate.COUNT()})
        popular_books = popular_books.sort("count", ascending=False)
    
        pos_list = [] 
        for i in range(len(item_data[item_id])):
            if len(pos_list) == 5: break
            if popular_books[item_id][i] in item_data[item_id]:
                pos_list.append(i)
        
        for pos in pos_list:
            self.most_popular_books_ids.append(popular_books[item_id][pos])
        for ids in self.most_popular_books_ids:
            self.most_popular_books.append(item_data[item_data[item_id] == ids][["title", 
                                                            "author", "year", "publisher"]][0])
        return self.most_popular_books[0:n], self.most_popular_books_ids[0:n]
    
class RegressionModel:
    
    """
    This funciton takes as argument user's age and location(consisting state) and outputs two lists one containing ids
    of recommended books and other list contains title of recommended books. Currently, it only choose a movie among 
    3000 movies randomly chosen from IMPLICIT test dataset having total count of 45000 movies .(Note that model was 
    trained on explicit dataset which is different from implicit dataset).
    Count of movies can be increased(by modifiying max variable) if required to search among more movies, but it will take 
    considerable time depending on the machine this function is evaluated upon.
    """
    def predict(self, location, age, search_over, n=3):
        # Load required models and data
        regression_model = gl.load_model("./regression_model_file/")
        book_data = gl.load_sframe("./book_data_clean/")
        implicit_data = gl.load_sframe("./implicit_rating_data/")
        book_data.filter_by(implicit_data["book_id"], "book_id")
        
        # Select approx (search_over) books by splitting data RANDOMLY
        split = search_over/45000.0
        book_data, other_data = book_data.random_split(split)
        
        predicted_ratings = []
        count = 0
        for book in book_data:
            if count == search_over:
                break
            count += 1
            book["location"] = location
            book["age"] = age
            rating = regression_model.predict(book)[0]
            if rating >= 8.0:
                predicted_ratings.append((book["book_id"], rating))
    
        predicted_ratings = sorted(predicted_ratings, key=itemgetter(1), reverse=True)

        # Recommeded books in decresing values of ratings
        recommended_books_id = []
        for i in range(5):
            recommended_books_id.append(predicted_ratings[i][0])

        recommended_books = []
        for book in recommended_books_id:
            for item in book_data:
                if book in item["book_id"]:
                    del(item["book_id"])
                    recommended_books.append(item)
                    break
        return recommended_books[0:n], recommended_books_id[0:n]
    
class SimilarityModel:
    
    # Returns a distance based similarity score based for user1 and user2
    # Score between (0-1) score 1 means distance zero, higher the score more similar the users are
    def euclid(self, ratings, user1, user2):
        flag = 0
        for item in ratings[user1]:
            if item in ratings[user2]:
                flag = 1; break
            
        # if no ratings in common, return 0
        if flag == 0: return 0
    
        # Add up the squares of all differences
        sum_squares = sum([pow(ratings[user1][item]-ratings[user2][item],2) 
                       for item in ratings[user1] if item in ratings[user2]])
    
        return 1/(1+sum_squares) 
    
    # Returns pearson corelation coefficient for user1 and user2
    # Score between -1 and 1 more score means more similarity b/w users 
    def pearson(self, rats, user1, user2):
        # List of rated items
        shared_items = {}
        for item in rats[user1]:
            if item in rats[user2]:
                shared_items[item] = 1
            
        n = len(shared_items)
        # if no common item, return 0
        if n == 0: return 0
    
        # Add up all the ratings
        sum1 = sum([rats[user1][item] for item in shared_items])
        sum2 = sum([rats[user2][item] for item in shared_items])
    
        # Sum up all the squares of ratings
        sum1Sq = sum([pow(rats[user1][item],2) for item in shared_items])
        sum2Sq = sum([pow(rats[user2][item],2) for item in shared_items])
    
        # Sum up all the products
        prodSum = sum([rats[user1][item]*rats[user2][item] for item in shared_items]) 
    
        # Calculate pearson score
        num = prodSum - (sum1*sum2/n)
        temp = math.sqrt((sum1Sq - pow(sum1,2)/n) * (sum2Sq - pow(sum2,2)/n))
        if temp == 0: return 0
    
        score = num/temp
        return score
    
    """
    Computing similarity of one user to every other user in dataset.
    This function will return a list of tuples with tuples containing similarity and id of the user

    This function returns (n) most similar users where n is the number of movies we want our recommender to recommend,
    (n) here can be increased to get even better results
    """
    def getSimilarUsers(self, ratings, user, n=50):
        sim = [(other, self.pearson(ratings, user, other)) for other in ratings if other!=user]
    
        # Sort list so that more similar users appear at top
        sim = sorted(sim, key=itemgetter(1), reverse=True)
    
        # If first similarity is 0 means no similar user found, use euclid in such case
        if sim[0][1] == 0:
            sim = [(other, self.euclid(ratings, user, other)) for other in ratings if other!=user]
    
        # n denotes number of results to be returned
        return sim[0:n]
    
    def getRecommendations(self, ratings, user, n=5):
        totals = {}
        simSums = {}
        # Get a list of n most similar users
        similar_users = self.getSimilarUsers(ratings, user, n*10)
    
        # For every similar user in similar_users rate the movie that user has'nt rated yet
        for similar in similar_users:
            other = similar[0]
            sim = similar[1]
            # if similarity less than 0, ignore
            if(sim <= 0): continue
            
            for item in ratings[other]:
                # only score movies user hasn't seen yet
                if item not in ratings[user] or ratings[user][item] == 0:
                    # similarity * other user rating
                    totals.setdefault(item, 0)
                    totals[item] += ratings[other][item]*sim
                    # sum of similarities
                    simSums.setdefault(item, 0)
                    simSums[item] += sim
    
        # Normalize predicted ratings and store then as tuples in a list
        rankings = [(item, total/simSums[item]) for item,total in totals.items()]
        rankings = sorted(rankings, key=itemgetter(1), reverse=True)
        return rankings[0:n]
    
    def predict(self, ratings, user, n=5):
        book_data = gl.load_sframe("./book_data_clean/")
        ids_ratings = self.getRecommendations(ratings, user, n+50)
        #list storing details of recommended books
        list_of_books = []
        list_of_ids = []
    
        # Serach a book via its id in book_data and append all its details along with rating to list_of_books
        count = 0
        for item in ids_ratings:
            if count == n: break
            # if book details not present in book_data, skip over to next until (n) books are appended to list
            if item[0] not in book_data["book_id"]: continue
            
            count += 1
            book = book_data[book_data["book_id"] == item[0]][0]
            if item[1] > 10:
                book["rating"] = 10
            else:
                book["rating"] = item[1]
            # append id to another list and delete book id from dictionary
            list_of_ids.append(book["book_id"])
            del(book["book_id"])
            del(book["rating"])
            list_of_books.append(book)
        
        return list_of_books[0:n], list_of_ids[0:n]
    
class CooccurModel:
    """
    Using co_dict rather than matrix SFrame (constructed using co_dict), this will make computation much more efficient, 
    The score list store keys (in the corpus) and scores, on the basis of user's reading history

    This cooccurrence dictionary is really sparse (5% of original data) hence I was able to find recommendation only 
    for 15 users out of 100 users(for which I tried to compute recommendation).
    To increase the number of users which get recommendations, cooccur dictionary must be computed for other 95% data

    This function will loops over all the users present in rating dictionary and will SKIP those user for which no 
    similar movies are found.
    
    n-> denotes the maximum number of books to be recommended to a user
    """
    def predict(self, rating_dict, co_dict, userId=None, n=5):
        recom_books = {}
        
        # Rating dictionary stores user as keys and another dictionary as values
        # containing (book/corresponding ratings give by user) as key/value pair
        if userId in rating_dict.keys():
            user_rating = rating_dict[userId]
            score = []
            flag = 0
    
            # co_dict contains book_ids as keys and another dict as values containing
            # book_ids and normalized similarity between those books(as key/value pair)
            # Loop over all the books in the inventory
            for bookId,book_sim in co_dict.items():
                temp = 0
            
                # Loop over all the previouly rated book by a user and add the similarity b/w 
                # current book and EACH of the previously rated book.
                # Compute final score by dividing total number of books user has already rated
                for prev_rated in user_rating.keys(): 
                    if prev_rated in book_sim.keys():
                        temp += book_sim[prev_rated]
                    
                if temp != 0:
                    # To NORMALIZE score, divide score by total number of previouly rated books 
                    temp /= len(user_rating)
                    flag = 1
                    score.append((bookId, temp))
            score = sorted(score, key=itemgetter(1), reverse=True)[0:n]
    
            if flag == 1:
                recom_books.setdefault(userId, 0)
                recom_books[userId] = score
        return recom_books
    
""" 
Function to get recommendations based on five different models.
For a new usr with no previous history of interaction with books, set new_user to True, and pass age and location as 
function arguments.
If the user has alredy interacted with books(i.e. previous history of user is available in data for user) then just pass 
the user_id which is stored in the data.

(reg_max_search) variable denotes the number of books to be searched for recommendation it can be increased to search over 
upto 45000 books.
Decrease reg_max_search value to lower computation time.
sim_method can be chaged to euclid if similarity is to be calculated on the basis of euclidean distance.
"""
def suggest(new_user=False, loc=None, age=0, reg_max_search=3000, user_id=None, image_size="M"):
    
    total_list_books = []
    total_list_ids = []
    if new_user == True:
        # If new user recommend books only on the basis of popularity model and Regression model
        # Recommend 3 books via Regression model and 2 books based on popularity model
        reg_model = RegressionModel()
        reg_books, reg_books_ids = reg_model.predict(loc, age, reg_max_search)
        
        pop_model = PopularityModel()
        pop_books, pop_books_ids = pop_model.predict(train_data, item_data=book_data, user_id="user_id", 
                                       item_id="book_id", rating="ratings")
        # Append the books recommended by popularity and regression model to total list
        for book in pop_books:
            total_list_books.append(book)
        for book in reg_books:
            total_list_books.append(book)
        for i in pop_books_ids:
            total_list_ids.append(i)
        for i in reg_books_ids:
            total_list_ids.append(i)
        
    else:
        # Changing the column names in book_data table for compatibility with all models 
        mod_book_data = book_data[["book_id", "title", "year", "author", "publisher"]]
        
        """
        # If old user then predict on the basis of similarity, cooccurrence and Factorization model
        
        # Using ranking factorization model
        # Selecting specific columns from book data
        rank_fact_model = gl.load_model("./my_models/rank_imp_model/")
        fact_book_ids = list(rank_fact_model.recommend(users=[user_id])["book_id"])[0:5]
        for bookId in fact_book_ids:
            if bookId in mod_book_data["book_id"]:
                info = mod_book_data[mod_book_data["book_id"] == bookId][0]
                total_list_ids.append(info["book_id"])
                del(info["book_id"])
                total_list_books.append(info)
        """
        
        
        # Using Similarity model
        critics = np.load("rating_dictionary.npy").item()
        sim_model = SimilarityModel()
        sim_books, sim_ids =  sim_model.predict(critics, user_id)
        for book in sim_books:
            total_list_books.append(book)
        for i in sim_ids:
            total_list_ids.append(i)
        
        # Using cooccurence matrix based model
        # Loading required data
        rating_dict = np.load("rating_dictionary.npy").item()
        co_dict = np.load("cooccurrence dict.npy").item()
        # To check if the returned dictionary is empty
        flag = 0 
        co_model = CooccurModel()
        co_books = co_model.predict(rating_dict, co_dict, user_id)
        if co_books:
            co_books = co_books[user_id]
            flag = 1
        
        if flag == 1:
            for item in co_books:
                bookId = item[0]
                if bookId in mod_book_data["book_id"]:
                    book_info = mod_book_data[mod_book_data["book_id"] == bookId][0]
                    total_list_ids.append(book_info["book_id"])
                    del(book_info["book_id"])
                    total_list_books.append(book_info)
                    
        
        
        # Code to ensure that exactly five books are recommended to user
        count = len(total_list_ids)
        # If recommended books greater than 5 just strip
        if  count > 5:
            total_list_books = total_list_books[0:5]
            total_list_ids = total_list_ids[0:5]
        
        # If recommended books less than 5 use regression model or popularity model to fill the gap
        elif count < 5:
            # total book to recommend is 5, counting the missing values
            miss = 5 - count
        
            # Using regression model to fill missing values    
            if user_id in user_data["user_id"]:
                user = user_data["user_id"]
                reg_model = RegressionModel()
                reg_books, reg_books_ids = reg_model.predict(user["location"], user["age"], reg_max_search)
                # appending reg_books and ids to total lists
                total_list_books, total_list_ids = append(total_list_books, total_list_ids, 
                                                          reg_books, reg_books_ids, miss)
                
            # If regression model fails then use popularity model
            else:
                pop_model = PopularityModel()
                pop_books, pop_books_ids = pop_model.predict(train_data, item_data=book_data, user_id="user_id", 
                                       item_id="book_id", rating="ratings")
                # appending pop_books and ids to total lists
                total_list_books, total_list_ids = append(total_list_books, total_list_ids, 
                                                          pop_books, pop_books_ids, miss)
                
    show(total_list_books, total_list_ids, image_size)
                        
def append(total, totalids, books, bookids, miss):
    temp = 0
    for book in books:
        if temp == miss: break
        total.append(book)
        temp += 1
    temp = 0
    for i in bookids:
        if temp == miss: break
        totalids.append(i)
        temp += 1 
    return total, totalids

def show(books, bookids, size):
    if size == "M":
        dis = "Image-URL-M"
    else:
        dis = "Image-URL-L"
    temp = -1
    for i in bookids:
        book = book_data[book_data["book_id"] == i][0]
        if i in book_data["book_id"] and book[dis].startswith("http"):
            display(Image(url=book[dis]))
        else:
            print "IMAGE FOR THIS BOOK IS NOT AVAILABLE"
        temp += 1   
        print "Title of Book :: ", books[temp]["title"]
        print "Author of Book :: ", books[temp]["author"]
        print "Year of Publication :: ", books[temp]["year"]
        print "Publisher :: ", books[temp]["publisher"]
        
# Set image size to "L" to display large image or to "S" to display small image
suggest(user_id="114078", image_size="M")

suggest(new_user=True, loc="delhi, india", age=21)


