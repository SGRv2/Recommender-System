# Movie Recommendation System
Recommendation systems are at the heart of several eCommerce and content consumption businesses of the likes of Amazon and Netflix. With time, these systems have evolved to deliver
highly personalized recommendations to users and have become an effective way to increase user retention on said services.

This project explores ways to implement such recommendation systems by leveraging existing data about the users’ past choices. We try to solve this problem through a user-based collaborative-filtering approach. We then briefly explore some shortcomings of the approach and attempt to come up with improvements to it.

## Description of Files
* `RS_main.py` - 
* `test.py` - 

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
