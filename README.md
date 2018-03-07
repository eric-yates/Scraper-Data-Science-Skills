# job-scraper

Respectfully scrapes job postings from Indeed and creates visualizations

## Getting Started

These instructions will get a copy of the project up and running on your local machine.

### Prerequisites

What things you need to install the software and how to install them

```
Python 2.7

pandas
nltk
numpy
seaborn
matplotlib
beautifulsoup4
wordcloud
```

### Installing

If your computer does not already have Python 2.7 installed, download Python 2.7 [here](https://www.python.org/downloads/)

By default, Python should come with pip (a package manager). Use it to install the following dependencies by opening the Terminal and entering the commands as follows for each of the prerequisites:

```
pip install pandas
```

And then

```
pip install nltk
```

And so forth until all the prerequisites have been installed. Spell each prerequisite exactly as listed in the prerequisites section with the same capitalization.

### Usage

After setting up, run the Python script from the IDLE. It will ask for user input for:

```
Query - The search for job position, skills, etc. (Example: data scientist healthcare)
City - Optional: The city to search in. If left blank (ie. just hit enter), it will search nationwide.
State - Optional: Use a 2 letter abbreviation (ie. NY for New York), although it should work even with full name.
```

Then it will ask for a second query to compare results with the first query. This is optional. If left blank (ie. just hit enter), only the first query/city/state will be searched and no comparisons will be made. This will only display the wordcloud of skills and the percentage of job postings that contain each skill.

If filled in, the program will scrape job postings, seperately but automatically, for both query/city/state combinations. It will then display the wordcloud and percentage of job postings that contain each skill, for both combinations, as well as the percent difference graph between skills (ie. more common/less common skills with respect to the second query) and the adjusted score (for the second query).

## Built With

* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) - Web scraping tool that pulls job summaries from Indeed
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/) - Data manipulation
* [Matplotlib](https://matplotlib.org/) - Graphing data
* [Seaborn](https://seaborn.pydata.org/) - Extra layer on top of Matplotlib for better looking graphs

## Authors

* **Eric Yates** - [Github Profile](https://github.com/eric-yates)

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/eric-yates/job-scraper/blob/master/LICENSE.md) file for details

## Acknowledgments

* Jesse Steinweg-Woods: Part of the code comes from [Web Scraping Indeed for Key Data Science Job Skills](https://jessesw.com/Data-Science-Skills/)
