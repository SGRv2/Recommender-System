echo "Installing dependencies. I hope you're in a virtual environment"
pip3 install -r requirements.txt
echo ""
echo ""

echo "*************** Running RS_main.py to evaluate the recommendation-system model ***************"
echo ""
echo "### Input dataset: ratings.csv"
echo "### Output:        eval.csv"
echo ""
if [[ $* == *--improved* ]]
then
	echo "Running improved recommendation system (Part-B)"
	python3 RS_main.py --input ratings.csv --output eval.csv --content_boost
else
	echo "Running original recommendation system (Part-A)"
	python3 RS_main.py --input ratings.csv --output eval.csv
fi

echo "Done. Run 'cat eval.csv' to see output"
echo ""
echo ""
echo ""
echo ""

echo "*************** Running test.py to generate top recommendations for users ***************"
echo ""
echo "### Input User List: test_user.txt"
echo "### Output:          output.csv"
echo ""
if [[ $* == *--improved* ]]
then
	echo "Generating predictions using improved recommendation system (Part-B)"
	python3 test.py --input test_user.txt --output output.csv --content_boost
else
	echo "Generating predictions using original recommendation system (Part-A)"
	python3 test.py --input test_user.txt --output output.csv
fi

echo "Done. Run 'cat output.csv' to see output"
