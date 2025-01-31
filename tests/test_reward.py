from init_test import init_test
init_test()

import asyncio
import numpy as np
from typing import List, Optional, Tuple
import time
import random
import nltk
from nltk.corpus import brown
from collections import Counter

from duckduckgo_search import DDGS
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup, Tag, NavigableString
from urllib.parse import urljoin
import bittensor as bt

from webgenie.tasks import Task, Solution, ImageTask
from webgenie.rewards import (
    LighthouseReward,
    QualityReward,
    VisualReward,
)

from webgenie.datasets.dataset import Dataset, DatasetEntry

from webgenie.protocol import WebgenieImageSynapse
from webgenie.tasks.metric_types import (
    ACCURACY_METRIC_NAME,
    SEO_METRIC_NAME,
    QUALITY_METRIC_NAME,
)
from webgenie.constants import IMAGE_TASK_TIMEOUT, GROUND_TRUTH_HTML_LOAD_TIME

from webgenie.helpers.htmls import html_to_screenshot, preprocess_html, is_empty_html
from neurons.miners.openai_miner import OpenaiMiner
# from neurons.miners.hf_miner import HfMiner


metrics = {
   ACCURACY_METRIC_NAME: VisualReward(),
#    SEO_METRIC_NAME: LighthouseReward(),
    # QUALITY_METRIC_NAME: QualityReward(),
}

CHROME_HTML_LOAD_TIME = 60000 # miner html load time
JAVASCRIPT_RUNNING_TIME = 1000 # javascript running time

def get_english_words():
    nltk.download("brown", quiet=True)
    words = brown.words()
    word_freq = Counter(word.lower() for word in words)
    most_common = word_freq.most_common(25000)
    common_words = [word for word, _ in most_common]
    return common_words

async def get_random_website_url(retries: int = 3) -> Optional[str]:
    english_words = get_english_words()
    try:
        ddg = DDGS()
        for _ in range(retries):
            random_words = " ".join(random.sample(english_words, 5))
            results = list(ddg.text(random_words))
            if results:
                website_url = random.choice(results)["href"]
                print(f"👟👟👟👟  {website_url}")
                return website_url
                
    except Exception as ex:
        print(f"Failed to get search results from DuckDuckGo: {ex}")
    return None

async def get_rendered_html(url):
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, timeout=CHROME_HTML_LOAD_TIME)
            
            # Wait for 10 seconds to ensure content loads
            # await page.wait_for_timeout(GROUND_TRUTH_HTML_LOAD_TIME)
            
            await page.wait_for_load_state('networkidle')
            await page.wait_for_timeout(JAVASCRIPT_RUNNING_TIME)
            
            rendered_html = await page.content()  # Get the rendered HTML
            
            await page.close()
            await browser.close()

            # Parse the HTML with BeautifulSoup
            soup = BeautifulSoup(rendered_html, 'html.parser')

            # Attributes that need to be absolute
            attributes = ['href', 'src', 'srcset']

            # Find all elements with 'href', 'src', or 'srcset' attributes
            for attr in attributes:
                for element in soup.find_all(attrs={attr: True}):
                    original_attr = element[attr]
                    # Handle 'srcset' differently because it can contain multiple URLs
                    if attr == 'srcset':
                        new_urls = []
                        parts = original_attr.split(',')
                        for part in parts:
                            # Split on whitespace and check if there is a descriptor
                            pieces = part.strip().split(maxsplit=1)
                            if len(pieces) == 2:
                                url_part, descriptor = pieces
                            elif len(pieces) == 1:
                                url_part = pieces[0]
                                descriptor = ''
                            else:
                                continue

                            new_url = urljoin(url, url_part.strip())
                            if descriptor:
                                new_urls.append(f"{new_url} {descriptor}")
                            else:
                                new_urls.append(new_url)

                        element[attr] = ', '.join(new_urls)
                    else:
                        element[attr] = urljoin(url, original_attr)
            # Remove all script tags
            for script in soup.find_all('script'):
                script.decompose()
            # Return the modified HTML as a string
            return str(soup)
    except Exception as e:
        raise Exception(f"Error in get_rendered_html: {e}")
    
    
async def shorten_html(html_content, max_p_count = 10, max_text_length = 200):
    """
    Removes excess <p> tags and trims text inside <p> tags if the text length exceeds the max limit.

    :param html_content: The HTML content as a string.
    :param max_p_count: The maximum number of <p> tags allowed in any parent tag.
    :param max_text_length: The maximum length of text allowed inside <p> tags.
    :return: Modified HTML content with excess <p> tags removed and text inside <p> tags shortened.
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all tags that contain <p> as direct children
        for tag in soup.find_all(True):  # True will find all tags
            # Find only <p> tags as direct children (not nested <p> tags)
            p_tags = [child for child in tag.find_all('p', recursive=False)]
            
            if len(p_tags) > max_p_count:
                # Remove excess <p> tags
                excess_p_tags = p_tags[max_p_count:]
                for p_tag in excess_p_tags:
                    p_tag.decompose()  # Remove the excess <p> tag
        
        # Traverse through all <p> tags and handle text nodes inside them
        for p_tag in soup.find_all('p'):  # Find all <p> tags
            for child in p_tag.contents:
                if isinstance(child, NavigableString):
                    text_str = str(child)
                    if len(text_str) > max_text_length:
                        shortened = text_str[:max_text_length] + "..."  # Shorten the text
                        child.replace_with(shortened)  # Replace the original text with the shortened version

        return str(soup)
    except Exception as e:
        bt.logging.error(f"Error in shorten_html: {e}")
        raise e

async def generate_context() -> Tuple[DatasetEntry, str]:
    try:
        bt.logging.info("Generating Random Website context")
        website_url = await get_random_website_url()
        if website_url is None:
            raise Exception("Failed to get a valid website URL")
        bt.logging.debug(f"Generated website URL: {website_url}")
        html = await get_rendered_html(website_url)
        html = await shorten_html(html)
        return DatasetEntry(
            src="random_website",
            topic="random_website",
            ground_truth_html=html,
            prompt="",
            base64_image="",
        ), website_url
    except Exception as e:
        bt.logging.error(f"Error in generate_context: {e}")
        raise e
    
async def calculate_scores(task: Task, solutions: List[Solution]) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}

    for metric_name, reward_model in metrics.items():
        print(metric_name)
        start_time = time.time()
        reward_scores = await reward_model.reward(task, solutions)
        execution_time = time.time() - start_time
        print(f"Execution time: {execution_time:.2f} seconds")
        scores[metric_name] = reward_scores
        print(scores[metric_name])
    return scores

async def generate_task() -> Tuple[Task, bt.Synapse, str]:
    bt.logging.info("Generating Image task")
    
    
    dataset_entry, website_url = await generate_context()
    bt.logging.debug(f"Generated dataset entry: {dataset_entry.src}")

    ground_truth_html = preprocess_html(dataset_entry.ground_truth_html)
    bt.logging.info(f"Preprocessed ground truth html")
    if not ground_truth_html :
        raise ValueError("Invalid ground truth html")

    if is_empty_html(ground_truth_html):
        raise ValueError("Empty ground truth html")
    
    base64_image = await html_to_screenshot(ground_truth_html, page_load_time=GROUND_TRUTH_HTML_LOAD_TIME)    
    bt.logging.debug(f"Screenshot generated for {dataset_entry.src}")
    image_task = ImageTask(
        base64_image=base64_image, 
        ground_truth_html=ground_truth_html,
        timeout=IMAGE_TASK_TIMEOUT,
        src=dataset_entry.src,
    )
    return (
        image_task,  
        WebgenieImageSynapse(base64_image=base64_image, task_id=image_task.task_id),
        website_url,
    )

async def main():
    is_base = False
    if is_base:
        ground_truth_html_path = "tests/work/original.html"
        with open(ground_truth_html_path, "r") as f:
            ground_truth_html = f.read()
        
        print("HTML to screenshot")
        start_time = time.time()
        base64_image = await html_to_screenshot(ground_truth_html)
        execution_time = time.time() - start_time
        print(f"Execution time: {execution_time:.2f} seconds")
        
        task = ImageTask(
            ground_truth_html=ground_truth_html,
        )
            
        synapse = WebgenieImageSynapse(
            base64_image = base64_image,
        )
    else:
        # miner = OpenaiMiner(neuron=None)
        # miner = HfMiner(neuron=None)
        task, synapse, website_url = await generate_task()
        print(f"🚩{website_url}")

    print("💢Miner forward image💢")
    start_time = time.time()
    # synapse = await miner.forward_image(synapse)
    # print(synapse.html)
    synapse.html = task.ground_truth_html
    execution_time = time.time() - start_time
    # print(f"Execution time: {execution_time:.2f} seconds")
    solutions = [Solution(html=synapse.html) for _ in range(1)]

    print("🎉Calculate scores🎉")
    start_time = time.time()
    scores = await calculate_scores(task, solutions)
    execution_time = time.time() - start_time
    # print(f"Execution time: {execution_time:.2f} seconds")
    try:
        print(f"{website_url}: {scores['Accuracy'][0]}")
        with open("/workspace/experiment.txt", "a") as file:
            file.write(f"{website_url}: {scores['Accuracy'][0]}\n")
    except:
        print("💣Error arised")

if __name__ == "__main__":
    asyncio.run(main())
