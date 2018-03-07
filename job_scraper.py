import urllib2
import re

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker

from bs4 import BeautifulSoup
from time import sleep
from nltk.corpus import stopwords
from collections import Counter
from datetime import datetime
from matplotlib import pyplot as plt
from wordcloud import WordCloud

"""
Web Scraping Indeed for Key Data Science Job Skills
https://jessesw.com/Data-Science-Skills/

"""

cd = '~/Documents/Projects/job-scraper/'

    
def make_soup(url):

    html = urllib2.urlopen(url).read()

    return BeautifulSoup(html, 'lxml')


def text_cleaner(soup):
    """
    Inputs: A URL
    Outputs: Cleaned text
    """

    job_summary = soup.find(id='job_summary')

    text = job_summary.get_text()

    lines = (line.strip() for line in text.splitlines())

    chunks = (phrase.strip() for line in lines for phrase in line.split('  '))

    text = ''.join(chunk + ' ' for chunk in chunks if chunk).encode('utf-8')

    try:
        text = text.decode('unicode_escape').encode('ascii', 'ignore')

    except:
        print 'Error decoding text'
        return
    
    text = re.sub("[^a-zA-Z.+3]"," ", text)
    
    text = text.lower().split()

    stop_words = set(stopwords.words('english'))

    text = [w for w in text if w not in stop_words]
    
    return list(set(text))


def split_text(text, operand='+', plusified=''):
    """
    Replaces whitespaces with the operand (default: +)
    Depending on the use case, may want to set operand='%2B'
    """

    term_list = text.split()

    for term in term_list:
        plusified += term + operand

    return plusified[:-len(operand)] # Removes the last operand


def spaces_to_pluses(q, city, state):
    """
    Calls the split_text() function on the query, city, and state
    to replace spaces with + symbols (or any other operand specified).
    
    Used to create the query URL (a URL cannot contain spaces).
    """

    if city and state:
        return split_text(q), split_text(city), split_text(state)

    else:
        return split_text(q), 'Nationwide', ' '


def calc_seed(q, city, state):

    if city != 'Nationwide':
        return ''.join(['https://www.indeed.com/jobs?q=', q, '&l=', city, '%2C+', state]) # Join all of our strings together so that indeed will search correctly

    else:
        return ''.join(['https://www.indeed.com/jobs?q=', q])


def calc_num_jobs(soup):

    jobs_str = soup.find(id = 'searchCount').string.encode('utf-8')
    job_numbers = re.findall('\d+', jobs_str) # Finds all numbers in jobs_str
    
    if len(job_numbers) >= 3: # num_jobs >= 1000
        return int(job_numbers[1]) * 1000 + int(job_numbers[2])

    else:   # 1 < num_jobs < 1000
        return int(job_numbers[1])


def calc_num_pages(num_jobs, max_pages):

    num_pages = num_jobs/10

    return min(num_pages+1, max_pages)


def calc_current_page(seed, i):

    start_num = str(i*10)

    return ''.join([seed, '&start=', start_num])


def calc_all_urls(soup):

    all_postings = soup.find(id='resultsCol')
        
    all_urls = ['https://www.indeed.com/' + link.get('href')
                    for link in all_postings.find_all('a') if link.get('href')]
        
    return filter(lambda x: 'clk?' in x, all_urls)
    

def get_job_descriptions(q='data+scientist', city='', state='', max_pages=10):
    '''
    This function will take a desired city/state and look for all new job
    postings on Indeed.com. It will crawl all of the job postings and keep
    track of how manyuse a preset list of typical data science skills. The
    final percentage for each skill is then displayed at the end of the
    collation. 
        
    Inputs: The location's city and state. These are optional. If no
    city/state is input,the function will assume a national search (this can
    take a while!). Input the city/state as strings, such as:
    
    skills_info('Chicago', 'IL').
    
    Use a two letter abbreviation for the state.
    
    Output: A bar chart showing the most commonly desired skills in the job
    market for the query.
    '''

    base_url = 'https://www.indeed.com'

    q, city, state = spaces_to_pluses(q, city, state) # Turn spaces into '+'
    print 'Query:"' + q + '" in ' + city + ', ' + state + '\n'

    seed = calc_seed(q, city, state) # Creates URL for seed page

    soup = make_soup(seed) # Creates BeautifulSoup object from HTML
    
    num_jobs = calc_num_jobs(soup) # Finds total number of jobs for query
    print 'There were', num_jobs, 'jobs found in', city, '\n'

    num_pages = calc_num_pages(num_jobs, max_pages) # Number of pages to search
    print 'Getting pages 1-' + str(num_pages) + ':'

    job_descriptions = []

    for i in xrange(1, num_pages+1):
        
        print 'Getting page', i
        
        current_page = calc_current_page(seed, i)

        soup = make_soup(current_page)

        all_urls = calc_all_urls(soup)
        
        for j in xrange(len(all_urls)):
            
            try:
                soup = make_soup(all_urls[j])
                current_description = text_cleaner(soup)
                job_descriptions.append(current_description)
                
            except:
                continue
            
            sleep(1) # Respect the server by not overloading it

    num_posts = len(job_descriptions)
       
    print '\nThere were', num_posts, 'jobs successfully found. \n\n'
    
    return job_descriptions, num_posts


def get_freqs(job_descriptions):
    
    doc_frequency = Counter()
    [doc_frequency.update(item) for item in job_descriptions]

    words = dict((k, v) for k, v in doc_frequency.items() if v >= 10)

    languages = Counter({'R':doc_frequency['r'],
                         'Python':doc_frequency['python'],
                         'Java':doc_frequency['java'],
                         'C++':doc_frequency['c++'],
                         'Ruby':doc_frequency['ruby'],
                         'Perl':doc_frequency['perl'],
                         'Matlab':doc_frequency['matlab'],
                         'HTML': doc_frequency['html'],
                         'CSS': doc_frequency['css'],
                         'JavaScript':doc_frequency['javascript'],
                         'Scala': doc_frequency['scala']
                         })
                      
    analysis_tools = Counter({'Excel':doc_frequency['excel'],
                              'Tableau':doc_frequency['tableau'],
                              'D3.js':doc_frequency['d3.js'],
                              'SAS':doc_frequency['sas'],
                              'SPSS':doc_frequency['spss'],
                              'D3':doc_frequency['d3']
                              })  

    hadoop_tools = Counter({'Hadoop':doc_frequency['hadoop'],
                            'MapReduce':doc_frequency['mapreduce'],
                            'Spark':doc_frequency['spark'],
                            'Pig':doc_frequency['pig'],
                            'Hive':doc_frequency['hive'],
                            'Shark':doc_frequency['shark'],
                            'Oozie':doc_frequency['oozie'],
                            'ZooKeeper':doc_frequency['zookeeper'],
                            'Flume':doc_frequency['flume'],
                            'Mahout':doc_frequency['mahout']
                            })
                
    databases = Counter({'SQL':doc_frequency['sql'],
                         'NoSQL':doc_frequency['nosql'],
                         'HBase':doc_frequency['hbase'],
                         'Cassandra':doc_frequency['cassandra'],
                         'MongoDB':doc_frequency['mongodb'],
                         'PostgreSQL':doc_frequency['postgresql'],
                         'MySQL':doc_frequency['mysql']
                         })

    libraries = Counter({'Numpy':doc_frequency['numpy'],
                         'Pandas':doc_frequency['pandas'],
                         'Scikit-learn':doc_frequency['scikit'] + doc_frequency['scikit-learn'],
                         'Scipy':doc_frequency['scipy'],
                         'Matplotlib':doc_frequency['matplotlib'],
                         'Seaborn':doc_frequency['seaborn'],
                         'Tensorflow':doc_frequency['tensorflow'],
                         'Keras':doc_frequency['keras'],
                         'Plotly':doc_frequency['plotly'],
                         'Theano':doc_frequency['theano'],
                         'NLTK':doc_frequency['nltk'],
                         'Scrapy':doc_frequency['scrapy']
                         })
    skills = languages + analysis_tools + hadoop_tools + databases + libraries
                                  
    return skills, words

    
def get_df(skill_frequencies, num_posts, query, city):
    
    df = pd.DataFrame(skill_frequencies.items(),
                      columns = ['Term', 'NumPostings'])                  

    df['Percentage'] = df['NumPostings'] * 100 / num_posts

    save_df(df, num_posts, query, city)

    return df


def save_df(df, num_posts, query, city):

    path = ''.join([cd, str(num_posts), '-', split_text(query), '-',
                    split_text(city), '-', timestamp, '.csv'])
    
    df.to_csv(path)
    

def trim_df(df, q=None):

    return df.query(q)


def create_title(query, city, state):

    if state and city:
        return (query + ' in ' + city + ', ' + state + '"?')

    else:
        return (query + ' in Nationwide"?')


def plot_skills(df, query, city, state, q='Percentage > 5', y='Percentage'):

    df = trim_df(df, q)

    df.sort_values(y, ascending=False, inplace=True)

    sns.set_style("whitegrid")

    sns.barplot(x='Term', y=y, data=df, color='b', dodge=False)

    title = create_title(query, city, state)

    plt.title('What skills are needed for "' + title)
    plt.xlabel('')
    plt.ylabel('% Job Postings')
    plt.xticks(rotation=60)
    plt.tight_layout()

    plt.savefig('plot-' + split_text(query) + timestamp + '.jpg')

    plt.show()

    return


def calc_diffs(df1, df2, term='Percentage'):

    df1_avg = df1[term].mean()
    df2_avg = df2[term].mean()

    df2['Weighted'] = df1_avg/df2_avg * df2[term]
    df2['df1'] = df2['Term'].map(df1.set_index('Term')[term])
    df2['Difference'] = 100 * (df2['Weighted'] - df2['df1']) / df2['df1']

    df2 = trim_df(df2, q='Weighted > 5 or df1 > 5')
    
    mini = min(df2['Difference'])
    maxi = max(df2['Difference'])

    diff_range = maxi - mini

    df2['Score'] = df2['Percentage'] * (1 + np.tanh((df2['Difference']) / diff_range))

    return df2


def plot_diffs(df, y='Difference', query2, city2, state2):

    df.sort_values(y, ascending=False, inplace=True)

    df['color'] = df.Difference.apply(lambda x: 'More Common' if x>0
                                      else 'Less Common')

    sns.set_style("whitegrid")
    
    ax = sns.barplot(x='Term', y=y, data=df, hue='color',
                     palette={'More Common': 'g', 'Less Common': 'r'},
                     dodge=False)

    ax.legend().set_visible(False)

    #ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

    title = create_title(query2, city2, state2)

    plt.title('What skills are more common in "' + title)
    plt.xlabel('')
    plt.ylabel('% Difference')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(0.735, 0.965))

    plt.savefig('diff-' + timestamp + '.jpg')
    plt.show()

    return

def plot_score(df, y='Score', query2, city2, state2):

    df.sort_values(y, ascending=False, inplace=True)

    df['color'] = df.Difference.apply(lambda x: 'More Common' if x>0
                                      else 'Less Common')

    sns.set_style("whitegrid")
    
    ax = sns.barplot(x='Term', y=y, data=df, hue='color',
                     palette={'More Common': 'g', 'Less Common': 'r'},
                     dodge=False)

    plt.legend().set_visible(False)

    title = create_title(query2, city2, state2)

    plt.title('What skills should be mastered by "' + title)
    plt.xlabel('')
    plt.ylabel('Adjusted Score')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(0.735, 0.915))
    plt.savefig('score-' + timestamp + '.jpg')
    plt.show()

    return


def create_counts(df, numlistings, term, counts=[]):

    terms = df['Term'].tolist()
    numpostings = df[term].tolist()

    for a, b in zip(terms, numpostings):
        counts.append((a, b))

    return counts


def create_text(counts, text=''):

    for pair in counts:
        text += (pair[0] + ' ') * pair[1]

    return text


def create_wordcloud(df, query, city, state, numlistings, term, c='white',
                     width=1200, height=1000, collocations=False,
                     prefer_horizontal=1.0):

    counts = create_counts(df, numlistings, term)

    text = create_text(counts)

    wordcloud = WordCloud(background_color=c,
                          width=width,
                          height=height,
                          collocations=collocations,
                          prefer_horizontal=prefer_horizontal,
                          ).generate(text)

    title = create_title(query, city, state)

    plt.title(title)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('wc-' + split_text(query) + '-' + split_text(city) + '-'
                + timestamp + '.jpg')
    plt.show()

    return


def get_all(query, city, state):

    job_descriptions, num_posts = get_job_descriptions(q=query,
                                                       city=city,
                                                       state=state,
                                                       max_pages=max_pages)

    skills, words = get_freqs(job_descriptions)

    df = get_df(skills, num_posts, query, city)

    return num_posts, skills, words, df


def group_average(df, skills, total=0):

    df = df.loc[df['Term'].isin(skills)]

    for i, difference in enumerate(df['Difference']):
        total += difference

    return total / (i + 1)
        
                  
if __name__ == '__main__':

    query = raw_input('First search? ')
    city = raw_input('In what city? Leave blank: Search nationwide. ') # If left '', will search nationwide
    state = raw_input('In what state? Leave blank: Search nationwide. ') # Use two-letter abbreviation (ie. 'CO', 'IL',...)

    print('')

    # Comparison of queries will be with respect to this set
    query2 = raw_input('Second search? Leave blank: No second search. ')

    if query2:
        city2 = raw_input('In what city? Leave blank: Search nationwide. ')
        state2 = raw_input('In what state? Leave blank: Search nationwide. ')

    print('')

    max_pages = input('How many pages to search? Returns (~14 x pages) jobs. ')

    print('')

    display_wc = raw_input('Display wordclouds? yes or no. ' )

    print('')

    timestamp = '{:%Y-%m-%d-%H.%M.%S}'.format(datetime.now())

    num_posts, skills, words, df1 = get_all(query, city, state)

    if query2:
        num_posts2, skills2, words2, df2 = get_all(query2, city2, state2)

    if display_wc in ('y', 'Y', 'yes', 'Yes'):
        create_wordcloud(df1, query, city, state, num_posts, 'NumPostings')

    if query2 and display_wc in ('y', 'Y', 'yes', 'Yes'):
        create_wordcloud(df2, query2, city2, state2, num_posts2, 'NumPostings')

    plot_skills(df1, query, city, state)

    if query2:
        
        plot_skills(df2, query2, city2, state2)
        
        df3 = calc_diffs(df1, df2)

        averages = {

        'vendor':[['Excel', 'SAS', 'Tableau', 'SPSS', 'Matlab'], 0],
        'open_source': [['Hadoop', 'Spark', 'Hive', 'Tensorflow', 'Pig', 'Pandas', 'Scikit-learn'], 0],
        'language': [['R', 'SQL', 'Python', 'Java', 'C++', 'Perl', 'Scala'], 0],
        'statistics': [['Excel', 'SPSS', 'SAS', 'R'], 0],
        'big_data': [['Tensorflow', 'Pig', 'Hadoop', 'Spark', 'Hive', 'Scikit-learn'], 0]

        }

        for group in averages:
            averages[group][1] = group_average(df3, averages[group][0])
            print group + ': ' + str(np.round(averages[group][1], 1)) + '%'
        
        plot_diffs(df3, query2, city2, state2)
        plot_score(df3, query2, city2, state2)
