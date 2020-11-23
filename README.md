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

# How to run
You will need to install dependencies before running the project, so maybe make sure you're in a python virtual environment or a conda environment before you proceed. 
The first thing to do is to navigate to the project directory - 
```console
dushyant@ubuntu:~$ cd Recommender-System
```

## Using `run.sh`
Ideally, executing `run.sh` in the project directory should run the project and display results. This script will install the dependencies if not already installed and then run the project.

You'll probably need to give it executable permissions first.
```console
dushyant@ubuntu:~/Recommender-System$ sudo chmod +x ./run.sh
```

To run the project as implemented in **Part-A** (i.e. without the improvements), just run the script as is.
```console
dushyant@ubuntu:~/Recommender-System$ ./run.sh
```

To run the project as implemented in **Part-B** (i.e. with the improvements), run the script with the `--improved` flag.
```console
dushyant@ubuntu:~/Recommender-System$ ./run.sh --improved
```

**This script is tested for Linux (Ubuntu). If it doesn't run on Windows (or doesn't run on your Linux system for some reason), you can still run the project manually as illustrated below**.

## Running manually
### Running Part-A
To run Part-A of the assignment (i.e. original, bare-bones recommendation system without improvements), run the following commands - 

1. Run `RS_main.py` using `ratings.csv` as the user-ratings dataset and storing the output in `eval.csv`.
```console
dushyant@ubuntu:~/Recommender-System$ python3 RS_main.py --input ratings.csv --output eval.csv
```

2. Run `test.py` with the file `test_user.txt` as input, storing the output in the file `output.csv`.
```console
dushyant@ubuntu:~/Recommender-System$ python3 test.py --input test_user.txt --output output.csv
```

### Running Part-B
To run Part-B of the assignment (i.e. recommendation system with improvements), just add the `content_boost` flag to the commands you used in Part-A -

1. Running `RS_main.py`
```console
dushyant@ubuntu:~/Recommender-System$ python3 RS_main.py --input ratings.csv --output eval.csv --content_boost
```

2. Running `test.py`
```console
dushyant@ubuntu:~/Recommender-System$ python3 test.py --input test_user.txt --output output.csv --content_boost
```
