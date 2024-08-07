import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import requests
from bs4 import BeautifulSoup
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from webdriver_manager.firefox import GeckoDriverManager
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import time
import re



##################################################
AI71_API_KEY = os.getenv('AI71_API_KEY')
model_name = "tiiuae/falcon-180B-chat"
AI71_BASE_URL = "https://api.ai71.ai/v1/"


#Medium Article Search and Scrapping using Selenium and BeautifulSoup

def get_urls_and_titles_Med(url):

    #options = Options()
    options = FirefoxOptions()
    options.headless = True
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--allow-running-insecure-content')
    #driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)
    
    urls_and_titles = []
    
    driver.get(url)
    time.sleep(5)  # Wait for the page to load

    try:
        article_elements = driver.find_elements(By.TAG_NAME, 'a')
        for a_tag in article_elements:
            href = a_tag.get_attribute('href')
            title = a_tag.text.strip()

            if title and href and href.startswith('http'):
                urls_and_titles.append((href, title))
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.quit()
    
    return urls_and_titles

def search_articles_Med(topic):
    search_urls = {
        "Medium": f"https://medium.com/search?q={topic.replace(' ', '+')}"
    }

    results = {}

    for source, url in search_urls.items():
        results[source] = get_urls_and_titles_Med(url)

    return results

# AnalyticsVidhya and KDNuggets Article Search and Scrapping using BeautifulSoup

def get_urls_and_titles_KDN_AV(url, topic_words):

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    urls_and_titles = []
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    
    for a_tag in soup.find_all('a', href=True):
        title = a_tag.get_text(strip=True)
        href = a_tag['href']
        if title and href.startswith('http') :
            urls_and_titles.append((href, title))
    
    return urls_and_titles

def search_articles_KDN_AV(topic):
    topic_words = topic.split()
    search_urls = {
        "Analytics Vidhya": f"https://www.analyticsvidhya.com/?s={topic.replace(' ', '+')}",
        "KDnuggets": f"https://www.kdnuggets.com/?s={topic.replace(' ', '+')}"
    }

    results = {}

    for source, url in search_urls.items():
        results[source] = get_urls_and_titles_KDN_AV(url, topic_words)

    return results


# rule-based filtering to get actual articles on the search results page

def get_filtered_title_urls(search_topic):
    
    search_topic_list = search_topic.lower().split()

    filtered_title_urls = []

    for k in list(article_results.keys()):
        url_ttl_list = article_results[k]
        url_ttl_list_new = []

        for url_ttl in url_ttl_list: 
            url = url_ttl[0]
            title = url_ttl[1]

            if any(word in url.lower() for word in search_topic_list) or any(word in title.lower() for word in search_topic_list):

                if k=='Medium':
                    if 'towards' in url:
                        filtered_title_urls.append( (title,url) )

                if k=='Analytics Vidhya':
                    if 'blog/20' in url:
                        filtered_title_urls.append( (title,url) )

                if k=='KDnuggets':
                    filtered_title_urls.append( (title,url) )
                    
                    
    return filtered_title_urls

# remove dupicate detections of same url with different tags

def filter_urls_by_title_length(filtered_title_urls):
    url_dict = {}
    
    for title, url in filtered_title_urls:
        if url in url_dict:
            if len(title) > len(url_dict[url][0]):
                url_dict[url] = (title, url)
        else:
            url_dict[url] = (title, url)
    
    return list(url_dict.values())


# get the blog text after removing irrelvant text

def get_blog_text(url):

    response = requests.get(url,verify=False)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the URL: {url}")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    for script in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
        script.extract()  # Remove these tags from the soup

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


# Title Relevance - LangChain Prompt and JSON Output Parser with Pydantic Schema

class TitlesInfo(BaseModel):
    title_ranks: List[int] = Field(description="List of int Numbers representing the Title Numbers after ranking")
    
parser = JsonOutputParser(pydantic_object=TitlesInfo)
format_instructions = parser.get_format_instructions()


template = """
From the Numbered List of Article Titles, find the Top 5 articles most relevant to the topic: {topic} \
. Then rank and sort them based on how relevant they are to the topic: {topic}, and create a sorted List of ints representing Title Numbers,\
which sorted by relevance to the topic, from highest to lowest. Your output should \
only be a JSON with title_ranks key having that List of ints as value. Don't write anything else.
Following is the Numbered List of Article Titles.
{text}
{format_instructions}
"""

prompt_template = ChatPromptTemplate.from_template(template)


# Article Scores and keyword extraction - LangChain Prompt and JSON Output Parser with Pydantic Schema

class BlogInfo(BaseModel):
    relevance_score: int = Field(description="int from 0-10 representing topic relevance score of article")
    detail_score: int = Field(description="int from 0-10 representing detailed coverage score of article")
    organization_score: int = Field(description="int from 0-10 representing content organization score of article")
    content_score: int = Field(description="int from 0-10 representing content quality score of article")
    code_score: int = Field(description="int from 0-10 representing code presence of article")
    keywords: str = Field(description="a String with keywords in articles separated by commas")
    
parserB = JsonOutputParser(pydantic_object=BlogInfo)
format_instructionsB = parserB.get_format_instructions()


templateB = """
You have to process an article on the topic of {topic} and give keywords and 0-10 range rating scores of different aspects of article \n
0 means bad score and and 10 means very good score.\n
Find relevance_score, which will be 0 to 10 score of article in terms of relevance to topic {topic} \n
Find detail_score, which will be 0 to 10 score of article in terms of details and topic coverage \n
Find organization_score, which will be 0 to 10 score of article in terms of content structure and organization \n
Find content_score, which will be 0 to 10 score of article in terms of content quality \n
Find code_score, which will be 0 to 10 score of article in terms of how much practical examples or programming code are present \n
Find keywords, which will be a String representing 5 important key-words of article separated by commas \n
Your output should only be a JSON with all the 6 fields that you have to find.\
Don't write anything other than that JSON.\n
Following is the Article Text to process:
{text} \n

{format_instructions}
"""


def fetch_article_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    article_text = soup.get_text(separator=' ')
    return article_text


# finding Date text pattern using regex

def find_first_date(article_text):
    # Date patterns to search for
    date_patterns = [
        r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec),? \d{4}\b',  # e.g., 05 Aug, 2024
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, \d{4}\b',     # e.g., Mar 5, 2024
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b'  # e.g., June 19, 2024
    ]

    for pattern in date_patterns:
        match = re.search(pattern, article_text)
        if match:
            return match.group(0)
    return None

def parse_date(date_str):
    # Define the date formats corresponding to the patterns
    date_formats = [
        "%d %b, %Y",  # e.g., 05 Aug, 2024
        "%b %d, %Y",  # e.g., Mar 5, 2024
        "%B %d, %Y"   # e.g., June 19, 2024
    ]

    for date_format in date_formats:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue
    return None

def calculate_months_difference(past_date):
    current_date = datetime.now()
    difference = relativedelta(current_date, past_date)
    return difference.years * 12 + difference.months


# Finding Publish Date from Article page text

def get_first_date_from_article(url):
    article_text = fetch_article_text(url)
    if article_text:
        date_str = find_first_date(article_text)
        if date_str:
            date_obj = parse_date(date_str)
            if date_obj:
                months_difference = calculate_months_difference(date_obj)
                return (date_str, date_obj, months_difference)
    return None

# rule-based article recency score calculation using months passed

def get_recency_score(months_passed):
    if months_passed<=12:
        if months_passed<3:
            return 0.9
        if months_passed<6:
            return 0.75
        if months_passed<9:
            return 0.65
        return 0.6
    else:
        if months_passed<24:
            if months_passed<18:
                return 0.5
            else:
                return 0.35
        else:
            return 0.2
    
    return 0.2


##################################################

# Displaying Results

st.title ("AI BlogCurate")
st.header ("Best AI and Data Science Articles Search and Ranking")

                        
def streamlit_display(result_list):

    st.write("---"*12)
    for i in range(len(result_list)):
        st.subheader("Rank: "+str(i+1))
        st.markdown("**{}**".format(result_list[i]['title']))
        st.markdown("**{}**".format(result_list[i]['date']))
        st.markdown("**{}**".format(result_list[i]['url']))
        st.write("**Overall Score:**",result_list[i]['overall_score'])
        st.write("**Topic Relevance Score:**",result_list[i]['relevance_score'])
        st.write("**Content Organization Score:**",result_list[i]['organization_score'])
        st.write("**Content Quality Score:**",result_list[i]['content_score'])
        st.write("**Practical Demonstration Score:**",result_list[i]['code_score'])
        st.write("**Date Recency Score:**",result_list[i]['recency_score'])
        st.write("**Keywords:**",result_list[i]['keywords'])
        st.write("---"*12)

search_topic = st.text_area("Enter Topic:")
button_pressed = st.button("Search and Rank")

if button_pressed and search_topic:

    tim = time.time()

    st.write(f"Searching and Processing Articles on: {search_topic}...")

    ##################

    # Pipeline to search articles, get information, shortlist articles and estimate scores 

    result_list = None
    try:
        print('KDN AV Search')
        article_results_KDN_AVN = search_articles_KDN_AV(search_topic)
        print('Med Search')
        article_results_M = search_articles_Med(search_topic)
        article_results = article_results_M.copy()
        article_results.update(article_results_KDN_AVN) 
        filtered_title_urls = get_filtered_title_urls(search_topic)
        filtered_title_urls_r = filter_urls_by_title_length(filtered_title_urls)

        chat180 = ChatOpenAI(
            model=model_name,
            api_key=AI71_API_KEY,
            base_url=AI71_BASE_URL,
            streaming=False,)

        titles_text = ""
        for i,t in enumerate(filtered_title_urls_r):
            titles_text += 'Title Number '+str(i+1)+' : \n'
            titles_text += t[0] + '\n'

        messages = prompt_template.format_messages(topic=search_topic,text=titles_text, format_instructions=format_instructions)
        response = chat180(messages)
        output_dict=  parser.parse(response.content)

        top_5_articles = []
        N=0
        for i in output_dict['title_ranks'][:5]:
            t = filtered_title_urls_r[i]
            top_5_articles.append( {'title':t[0],'url':t[1],'title_rank':N+1}  )
            N+=1

        top_5_articles_N = []
        for i in range(len(top_5_articles)):
            print(i)
            try:
                blog = top_5_articles[i]
                url = blog['url']
                blog_text = get_blog_text(url)
                blog['text'] = blog_text[:5200]
                top_5_articles_N.append(blog)
            except Exception as e:
                print(e)

        final_results = []

        for i in range( len(top_5_articles_N) ):
            text = top_5_articles_N[i]['text']

            prompt_templateB = ChatPromptTemplate.from_template(templateB)
            messagesB = prompt_templateB.format_messages(topic=search_topic,text=text, format_instructions=format_instructionsB)

            responseB = chat180(messagesB)
            output_dictB = parserB.parse(responseB.content)

            output_dictB['title'] = top_5_articles_N[i]['title']
            output_dictB['url'] = top_5_articles_N[i]['url']

            dateres = get_first_date_from_article( output_dictB['url']  )
            if dateres is not None:
                date_str, date_obj, months_passed = dateres
                output_dictB['recency_score'] = get_recency_score(months_passed)
                output_dictB['date'] = date_obj.strftime('%d %b, %Y') 
            else:
                output_dictB['recency_score'] = 0.4
                output_dictB['date'] = ''


            final_results.append(output_dictB)

        for i in range(len(final_results)):
            final_results[i]['overall_score'] = round(0.25*float(final_results[i]['relevance_score'])+0.15*float(final_results[i]['detail_score'])+0.2*float(final_results[i]['organization_score'])+0.2*float(final_results[i]['content_score'])+0.1*float(final_results[i]['code_score']),2)+round(0.1*float(final_results[i]['recency_score']),2)
            final_results[i]['overall_score'] = round(final_results[i]['overall_score'],2)

        final_results = sorted(final_results, key=lambda x: x['overall_score'], reverse=True)
        result_list = final_results


    except Exception as e:
        print('E1',e)
        st.write('Could not find and process articles on the topic')


    print(time.time()-tim,'sec')

    streamlit_display(result_list)


  
        


