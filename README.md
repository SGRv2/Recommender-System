# Movie Recommendation System
Recommendation systems are at the heart of several eCommerce and content consumption businesses of the likes of Amazon and Netflix. With time, these systems have evolved to deliver
highly personalized recommendations to users and have become an effective way to increase user retention on said services.

This project explores ways to implement such recommendation systems by leveraging existing data about the users’ past choices. We try to solve this problem through a user-based collaborative-filtering approach. We then briefly explore some shortcomings of the approach and attempt to come up with improvements to it.

## Description of Files
* `RS_main.py` - The main file to run recommender system including **user-item matrix generation**, **neighborhood generation**, **prediction** and and **performance evaluation**. This should take command line arguments: `input` as `rating.csv` and save the output of MAE performance evaluation in `eval.csv`.

* `test.py` - This file will take input as a list of test user (already saved by students in test `user.txt`) and save output in `output.csv`

* `rating.csv` - Contains movie ratings given by users.

* `eval.csv` - Contains the result of MAE performance evaluation performed when running `RS_main.py`.

* `output.csv` - Contains top-5 movie recommendations for every user. There is a single movie-recommendation per row (i.e. there will be five rows per user id)

## How to run
### Using `run.sh`
Ideally, executing `run.sh` in the project directory should run the project and display results. However if that does not work, look at the section below to see how the code can be run manually

### Running manually
1. Navigate to the project directory
```console
dushyant@ubuntu:~$ cd Recommender-System
```

2. Run `RS_main.py` using `ratings.csv` as the user-ratings dataset and storing the output in `eval.csv`.
```console
dushyant@ubuntu:~/Recommender-System$ python3 RS_main.py --input ratings.csv --output eval.csv
```

3. Run `test.py` with the file `test_users.txt` as input, storing the output in the file `output.csv`.
```console
dushyant@ubuntu:~/Recommender-System$ python3 test.py --input test_users.txt --output output.csv
```

### Running content boosted version
1. Navigate to the project directory
```console
dushyant@ubuntu:~$ cd Recommender-System
```

2. Run `RS_main.py` using `ratings.csv` as the user-ratings dataset and storing the output in `eval.csv`.
```console
dushyant@ubuntu:~/Recommender-System$ python3 RS_main.py --input ratings.csv --output eval.csv --content-boost
```

3. Run `test.py` with the file `test_users.txt` as input, storing the output in the file `output.csv`.
```console
dushyant@ubuntu:~/Recommender-System$ python3 test.py --input test_users.txt --output output.csv --content_boost
```

